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
  **It counts messages this server sent, not messages this client
  received**, so a window that attaches mid-race sees its first `seq`
  somewhere in the hundreds. A consumer treats its own first observed
  value as the origin and only the deltas as meaningful.
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
import math
import queue
import socket
import threading
import time

logger = logging.getLogger(__name__)

# How long a client may take to accept a broadcast before it is treated as
# dead. `sendall` is called from the pyglet main thread, so a subscriber that
# stops reading blocks the replay window itself, not just its own view. The
# span change made that worse rather than better: a bigger message fills the
# socket buffer in fewer broadcasts, cutting the measured time-to-freeze from
# about 130 s to 0.7 s. A real dashboard on localhost accepts a 30 KB message
# in microseconds, so anything past 50 ms is not slow, it is gone.
CLIENT_SEND_TIMEOUT_S = 0.05


def _json_safe(value):
    """Replace every non-finite float with None, recursively.

    The `arcade` block guards its own floats, but `strategy` is a bare
    `asdict()` of DTOs carrying raw XGBoost, TCN and LightGBM output, and a
    model that cannot compute a value hands back NaN. Three of the guards
    on the way in are `or`/truthiness tests, which NaN passes: `nan or 0.0`
    is `nan`.

    Sanitising rather than dropping is deliberate. One unusable model
    output should cost that field, not the whole tick: the panels that do
    not depend on it keep updating, and None is a value every consumer
    already has to handle. `allow_nan=False` below is then the assertion
    that this worked, not the policy itself.
    """
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _blame(value, path: str = "") -> str:
    """Point at the first leaf `json.dumps` cannot encode, for the drop log.

    Without it the encoder's message names the type and not the field, and
    the whole reason a broadcast was dropped stays invisible.

    Two kinds of leaf get named, because the sanitiser only handles the
    first: a non-finite float, which `_json_safe` would have turned into
    None, and anything the encoder simply cannot take. `np.float32(nan)`,
    `np.int64` and `np.bool_` are not `float`/`int`/`bool` subclasses, so
    they slip past the sanitiser, kill the whole tick, and used to leave
    this function returning the empty string exactly when it was needed.
    """
    if isinstance(value, dict):
        for key, item in value.items():
            found = _blame(item, f"{path}.{key}" if path else str(key))
            if found:
                return found
    elif isinstance(value, list):
        for index, item in enumerate(value):
            found = _blame(item, f"{path}[{index}]")
            if found:
                return found
    elif isinstance(value, float) and not math.isfinite(value):
        return f"{path or '<root>'}={value}"
    elif not isinstance(value, (str, int, float, bool, type(None))):
        return f"{path or '<root>'}=<{type(value).__name__}>"
    return ""


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
        # Depth 1 with drop-oldest. Each payload is a COMPLETE snapshot, not a
        # delta, so a consumer that falls behind wants the newest one and
        # nothing else; `seq` is what makes the discard visible to it.
        self._outbox: queue.Queue[bytes] = queue.Queue(maxsize=1)

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
        threading.Thread(target=self._send_loop, daemon=True, name="TelemetryStreamSend").start()
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
        """Queue one JSON-encoded payload for every connected client.

        **Returns without touching a socket.** `sendall` blocks, and this is
        called from the pyglet main thread on `on_update`, so a subscriber
        that stops reading used to freeze the replay window itself — not its
        own view, the whole race. The span change made that far likelier by
        enlarging the message: the measured time-to-freeze against a stalled
        client fell from about 130 s to 0.7 s, and the product always opens
        two subscribers.

        The encode stays here, on the caller's thread: it is sub-millisecond,
        it keeps wire order deterministic, and it keeps the "which field was
        NaN" log next to the tick that produced it."""
        if not self._running:
            return
        with self._clients_lock:
            if not self._clients:
                return
        payload = _json_safe(data)
        try:
            # allow_nan=False is what makes the "nothing non-finite reaches
            # the wire" promise above a mechanism rather than a claim. The
            # default is True: json.dumps writes a bare NaN or Infinity,
            # Python's own parser reads it back, and JSON.parse rejects the
            # whole message. A web consumer would have lost every tick that
            # carried one model output it could not compute.
            message = json.dumps(payload, separators=(",", ":"), allow_nan=False).encode() + b"\n"
        except (TypeError, ValueError) as exc:
            # Dropping the message is the point: an unparseable one is worse
            # than a missing one, and `seq` makes the hole visible. Blame the
            # ORIGINAL, not the sanitised copy: the copy cannot contain a
            # non-finite value by construction, so pointing the log at it
            # would print nothing every time.
            logger.warning("Broadcast dropped, payload not JSON-safe: %s | %s", exc, _blame(data))
            return

        self._enqueue(message)

    def _enqueue(self, message: bytes) -> None:
        """Hand the message to the sender thread, discarding a stale one."""
        try:
            self._outbox.put_nowait(message)
            return
        except queue.Full:
            pass
        try:
            self._outbox.get_nowait()
        except queue.Empty:
            # The sender drained it between the two calls; nothing to discard.
            pass
        try:
            self._outbox.put_nowait(message)
        except queue.Full:
            # It refilled again, which means the sender is keeping up with a
            # newer payload than this one. Dropping this is the right outcome.
            logger.debug("Broadcast outbox full, dropping a superseded payload")

    def _send_loop(self) -> None:
        """Drain the outbox onto the sockets, off the pyglet thread."""
        while self._running:
            try:
                message = self._outbox.get(timeout=0.5)
            except queue.Empty:
                continue  # the timeout is what lets `stop()` end this thread
            dead: list[socket.socket] = []
            with self._clients_lock:
                clients_snapshot = list(self._clients)
            for client in clients_snapshot:
                try:
                    client.sendall(message)
                except OSError:
                    # socket.timeout is an OSError subclass, so a subscriber
                    # that stopped reading is pruned by the same path as one
                    # that closed. The Qt client already reconnects.
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
            # Bounded so one stalled subscriber cannot freeze the replay's
            # own frame loop; a timeout is then treated as death, below.
            client_socket.settimeout(CLIENT_SEND_TIMEOUT_S)
            with self._clients_lock:
                self._clients.append(client_socket)
            threading.Thread(
                target=self._keepalive_loop,
                args=(client_socket,),
                daemon=True,
                name="TelemetryStreamClient",
            ).start()

    def _keepalive_loop(self, client_socket: socket.socket) -> None:
        """Hold a reference to the socket until the server stops.

        It never reads, so it CANNOT see the remote end close - measured: a
        client that disconnects is still counted three seconds later, and
        only disappears once a broadcast fails to send. A dead client is
        detected by `_send_loop`, never here, so `client_count()` may
        include ghosts until the next broadcast. The docstring used to
        promise the prune this loop does not do.
        """
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
