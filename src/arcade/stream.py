"""TCP broadcast server for the arcade → dashboard link.

The race replay hosts this server (when strategy mode is on) and publishes
a merged arcade+strategy state as newline-delimited JSON on each arcade
frame. The PySide6 dashboard subprocess subscribes via
`src.arcade.dashboard.stream_client.TelemetryStreamClient` and reacts to
updates on its Qt event loop.

Pattern ported from Tom Shaw's `f1_replay/f1-race-replay/src/services/stream.py`
and trimmed to stdlib-only: the arcade process must not import PySide6 so
we can launch the dashboard as a subprocess without pulling Qt into the
replay window. The client class lives in the dashboard package (Qt-aware),
kept separate so this module never needs to import PySide6.

The payload contract
--------------------
Every message is one JSON object on its own line, built by
`F1ArcadeView._broadcast_if_due`, carrying:

- `schema_version` (`STREAM_SCHEMA_VERSION`): bumped when a key is renamed,
  removed, or changes meaning. Adding a key an old consumer can ignore does
  not bump it.
- `seq`: strictly increasing by 1 per message sent. It exists because a
  consumer that polls a latest-payload slot on its own timer beats against
  this producer, and both failure modes are otherwise invisible: a `seq`
  repeated is a duplicate read, a `seq` skipped is a dropped frame.
- `arcade`, `strategy`, `playback`: the state itself. The frozen shape lives
  in `tests/surfaces/test_arcade_wire_contract.py`, which is the thing that
  actually fails when a producer-side change breaks a consumer.

Nothing non-finite may reach the wire: `json.dumps` writes a bare `NaN`,
which Python's parser accepts and `JSON.parse` rejects, so one missing
telemetry value would otherwise drop the whole message for a web consumer.
"""

from __future__ import annotations

import json
import logging
import socket
import threading
import time

logger = logging.getLogger(__name__)


class TelemetryStreamServer:
    """Non-blocking TCP server that broadcasts JSON dicts to all clients.

    Runs in a daemon thread, accepts up to many simultaneous connections,
    writes `json.dumps(data).encode() + b"\\n"` to every live socket on
    `broadcast()`. Dead sockets are pruned on the next broadcast; no
    heartbeat needed because the replay pushes at ≥5 Hz. Designed to be
    started inside `F1ArcadeView._init_strategy_layer` and torn down in
    `on_hide_view`."""

    def __init__(self, host: str = "127.0.0.1", port: int = 9998) -> None:
        self.host = host
        self.port = port
        self._server_socket: socket.socket | None = None
        self._clients: list[socket.socket] = []
        self._clients_lock = threading.Lock()
        self._running = False

    def start(self) -> None:
        """Bind the listening socket and spawn the accept thread."""
        if self._running:
            return
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((self.host, self.port))
        sock.listen(5)
        self._server_socket = sock
        self._running = True
        threading.Thread(
            target=self._accept_loop, daemon=True, name="TelemetryStreamAccept"
        ).start()
        logger.info("TelemetryStreamServer listening on %s:%d", self.host, self.port)

    def stop(self) -> None:
        """Close all sockets and signal the accept thread to exit."""
        self._running = False
        if self._server_socket is not None:
            try:
                self._server_socket.close()
            except OSError:
                # Already closed/broken (e.g. peer reset first) — nothing to undo.
                pass
            self._server_socket = None
        with self._clients_lock:
            for client in list(self._clients):
                try:
                    client.close()
                except OSError:
                    # Same idempotent-close rationale as the server socket above.
                    pass
            self._clients.clear()
        logger.info("TelemetryStreamServer stopped")

    def broadcast(self, data: dict) -> None:
        """Send one JSON-encoded payload to every connected client.

        Failed sockets are removed from the pool, so a dead dashboard does
        not block the arcade window. Called from arcade's main thread via
        `on_update`; the JSON encode happens inline (sub-millisecond for
        our payload size ≤ 10 KB) to keep the wire order deterministic."""
        if not self._running:
            return
        with self._clients_lock:
            if not self._clients:
                return
        try:
            message = json.dumps(data, separators=(",", ":")).encode("utf-8") + b"\n"
        except (TypeError, ValueError) as exc:
            logger.warning("Broadcast JSON encode failed: %s", exc)
            return

        dead: list[socket.socket] = []
        with self._clients_lock:
            clients_snapshot = list(self._clients)
        for client in clients_snapshot:
            try:
                client.sendall(message)
            except OSError:
                dead.append(client)
        if dead:
            self._prune_clients(dead)

    def client_count(self) -> int:
        with self._clients_lock:
            return len(self._clients)

    # --- internals --------------------------------------------------------

    def _accept_loop(self) -> None:
        while self._running and self._server_socket is not None:
            try:
                client_socket, addr = self._server_socket.accept()
            except OSError:
                if self._running:
                    logger.debug("Accept interrupted")
                return
            logger.info("Stream client connected from %s", addr)
            with self._clients_lock:
                self._clients.append(client_socket)
            threading.Thread(
                target=self._keepalive_loop,
                args=(client_socket,),
                daemon=True,
                name="TelemetryStreamClient",
            ).start()

    def _keepalive_loop(self, client_socket: socket.socket) -> None:
        """Hold the socket open until it dies. We do not expect reads from
        the dashboard; this thread just keeps the FD alive and prunes it
        once the remote end closes."""
        try:
            while self._running:
                time.sleep(1.0)
        finally:
            try:
                client_socket.close()
            except OSError:
                # Remote end may have already dropped the connection.
                pass
            self._prune_clients([client_socket])

    def _prune_clients(self, dead: list[socket.socket]) -> None:
        with self._clients_lock:
            for client in dead:
                try:
                    self._clients.remove(client)
                except ValueError:
                    # Another thread already pruned this socket first — fine.
                    pass
