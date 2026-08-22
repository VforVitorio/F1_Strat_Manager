"""TCP broadcast server for the arcade → PITWALL link.

The race replay hosts this server (when strategy mode is on) and publishes
a merged arcade+strategy state as newline-delimited JSON on each arcade
frame. The PITWALL subprocess subscribes via
`src.pitwall.stream_client.ArcadeStreamClient` and pushes what it reads to
two webview windows.

Pattern ported from Tom Shaw's `f1_replay/f1-race-replay/src/services/stream.py`
and trimmed to stdlib-only. The reason survives the toolkit it was written
for: the replay process must not import the consumer's UI stack, so the
consumer is a subprocess and the client class lives with it. That was Qt
until sprint 7 retired it; it is pywebview now, and the constraint is the
same one.

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
import select
import socket
import threading
import time
from collections.abc import Callable

logger = logging.getLogger(__name__)

# How long a client may take to accept a broadcast before it is treated as
# dead. It bounds what ONE subscriber can cost the sender thread, which is
# also the thread that builds and encodes the next payload, so a stalled
# client delays the next tick rather than blocking the replay window. (This
# comment used to say `sendall` ran on the pyglet main thread and froze the
# window itself. That stopped being true in `d24a59e1`, which moved the write
# to `_send_loop`; the rewrite reached `broadcast`'s docstring and not this
# constant.)
#
# The threshold that matters is bytes buffered rather than seconds: a bigger
# message fills a stalled subscriber's socket buffer in fewer broadcasts, so
# it is pruned sooner. Measured against a peer that never reads, the time to
# prune fell from 1.26 s with two telemetry spans to 0.50 s with twenty.
# A real dashboard on localhost accepts a 30 KB message in microseconds, so
# anything past 50 ms is not slow, it is gone.
CLIENT_SEND_TIMEOUT_S = 0.05
# A transient `accept()` error is retried rather than treated as shutdown, but
# a listening socket that keeps failing is not going to recover, and a hot
# retry loop is worse than an honest surrender.
ACCEPT_RETRY_DELAY_S = 0.1
ACCEPT_ERROR_LIMIT = 20


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

    **Tuples walk with the lists.** `_blame` was taught them and this
    function was not, so for one commit the pair disagreed about what is
    encodable: a NaN inside a tuple survived sanitising, killed the whole
    tick, and the paragraph above became false for exactly the container
    the sibling had just learned. `json.dumps` writes a tuple as an array,
    so returning a list changes no wire byte, and `dataclasses.asdict`
    preserves tuples — the first tuple-typed model field arms it.
    """
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
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

    **The oracle is the encoder, not an allowlist.** An allowlist of
    `str/int/float/bool/None` blames a tuple, which `json.dumps` encodes
    perfectly well as an array — and because the first blame wins, a
    payload carrying both a tuple and a real culprit named the tuple and
    never reached the culprit. The drop log then pointed at an innocent
    field while the tick that died stayed unexplained.
    """
    if isinstance(value, dict):
        for key, item in value.items():
            # The encoder rejects a non-str/int/float/bool/None KEY too, and
            # walking only the values left this blind to it - the same empty
            # string the paragraph above says was the original bug, on a
            # different input class. `{np.int64(lap): prob}` is one
            # comprehension away from a `per_agent` block.
            if not isinstance(key, (str, int, float, bool, type(None))):
                return f"{path or '<root>'}.<key {key!r}>=<{type(key).__name__}>"
            found = _blame(item, f"{path}.{key}" if path else str(key))
            if found:
                return found
        return ""
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            found = _blame(item, f"{path}[{index}]")
            if found:
                return found
        return ""
    if isinstance(value, float) and not math.isfinite(value):
        return f"{path or '<root>'}={value}"
    try:
        json.dumps(value)
    except (TypeError, ValueError):
        return f"{path or '<root>'}=<{type(value).__name__}>"
    return ""


class TelemetryStreamServer:
    """Non-blocking TCP server that broadcasts JSON payloads to all clients.

    Two daemon threads: one accepts connections, one drains the outbox.
    `broadcast()` takes a FACTORY and queues it; the sender thread runs it,
    encodes the result as `json.dumps(...).encode() + b"\\n"` and writes that
    to every live socket, so the caller never builds, encodes or sends. Dead
    sockets are pruned on the next broadcast; no heartbeat needed because the
    replay pushes at >=5 Hz. Designed to be started inside
    `F1ArcadeView._init_strategy_layer` and torn down in `on_hide_view`."""

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
        self._outbox: queue.Queue[Callable[[], dict]] = queue.Queue(maxsize=1)

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

    def broadcast(self, build: Callable[[], dict]) -> None:
        """Queue one payload for every connected client, BUILT off this thread.

        Takes a factory rather than a dict, and that is the whole point. The
        caller is the pyglet frame loop, and assembling a tick is not cheap:
        measured on the real Melbourne 2025 replay, `_broadcast_if_due` cost
        its caller 3.19 ms on a steady 8x tick and 5.41 ms on a seek, of which
        the recursive `asdict` over the decision history is 2.48 ms and the
        JSON encode 1.81 ms. Schema v2's twenty telemetry spans take the seek
        tick to 32 ms, about two dropped frames at 60 FPS. Handing over a
        closure moves all of it to the sender thread that already existed for
        `sendall`, and leaves the frame loop paying one queue put.

        **Returns without touching a socket, and without calling `build`.**

        What supersede-drop discards is now the JOB rather than the bytes, one
        stage earlier and with the same policy: each payload is a complete
        snapshot, so a consumer that falls behind wants the newest and nothing
        else. A tick that misses its slot is never built at all."""
        if not self._running:
            return
        with self._clients_lock:
            if not self._clients:
                return
        self._enqueue(build)

    def _encode(self, build: Callable[[], dict]) -> bytes | None:
        """Run one payload factory and encode the result, or say why not.

        Both halves are guarded because both run on the sender daemon now, and
        a daemon thread that raises dies silently: the replay window keeps
        running, subscribers keep their sockets open, and nothing is ever sent
        again. `_accept_loop` below carries the same lesson for the same
        reason. That is why the factory's guard is deliberately broad, and why
        it logs rather than passing: the payload is assembled from twenty
        drivers' telemetry and five agents' dataclasses, so enumerating what it
        can raise means enumerating all of that.
        """
        try:
            data = build()
        except Exception:
            logger.exception("Broadcast dropped, the payload factory raised")
            return None
        try:
            # allow_nan=False is what makes the "nothing non-finite reaches
            # the wire" promise above a mechanism rather than a claim. The
            # default is True: json.dumps writes a bare NaN or Infinity,
            # Python's own parser reads it back, and JSON.parse rejects the
            # whole message. A web consumer would have lost every tick that
            # carried one model output it could not compute.
            payload = _json_safe(data)
            return json.dumps(payload, separators=(",", ":"), allow_nan=False).encode() + b"\n"
        except (TypeError, ValueError) as exc:
            # Dropping the message is the point: an unparseable one is worse
            # than a missing one, and `seq` makes the hole visible. Blame the
            # ORIGINAL, not the sanitised copy: the copy cannot contain a
            # non-finite value by construction, so pointing the log at it
            # would print nothing every time. The seq goes in the line because
            # the log no longer sits next to the tick that produced it.
            logger.warning(
                "Broadcast dropped, payload not JSON-safe: seq=%s %s | %s",
                data.get("seq") if isinstance(data, dict) else None,
                exc,
                _blame(data),
            )
            return None

    def _enqueue(self, job: Callable[[], dict]) -> None:
        """Hand the job to the sender thread, discarding a stale one."""
        try:
            self._outbox.put_nowait(job)
            return
        except queue.Full:
            pass
        try:
            self._outbox.get_nowait()
        except queue.Empty:
            # The sender drained it between the two calls; nothing to discard.
            pass
        try:
            self._outbox.put_nowait(job)
        except queue.Full:
            # It refilled again, which means the sender is keeping up with a
            # newer payload than this one. Dropping this is the right outcome.
            logger.debug("Broadcast outbox full, dropping a superseded payload")

    def _send_loop(self) -> None:
        """Build, encode and write, off the pyglet thread.

        The build and the encode moved here from the caller in #1049; the
        write has been here since `d24a59e1`. One thread does all three, in
        wire order, so a payload cannot overtake an older one.
        """
        while self._running:
            try:
                job = self._outbox.get(timeout=0.5)
            except queue.Empty:
                continue  # the timeout is what lets `stop()` end this thread
            message = self._encode(job)
            if message is None:
                continue  # `_encode` logged why; `seq` makes the hole visible
            dead: list[socket.socket] = []
            with self._clients_lock:
                clients_snapshot = list(self._clients)
            for client in clients_snapshot:
                try:
                    client.sendall(message)
                except OSError:
                    # socket.timeout is an OSError subclass, so a subscriber
                    # that stopped reading is pruned by the same path as one
                    # that closed. `_prune_clients` CLOSES it, which is what
                    # lets the peer reconnect: a stalled subscriber's socket
                    # is still perfectly healthy at the TCP level, so merely
                    # forgetting it left the consumer reading a connection
                    # that would never carry another byte and never end.
                    dead.append(client)
            if dead:
                self._prune_clients(dead)

    def client_count(self) -> int:
        with self._clients_lock:
            return len(self._clients)

    # --- internals --------------------------------------------------------

    def _accept_loop(self) -> None:
        # A transient accept error is not the same event as `stop()` closing
        # the listening socket, and the two used to share one unconditional
        # `return`. Any winsock hiccup while running therefore ended the only
        # thread that admits subscribers, for the rest of the session, leaving
        # one DEBUG line the arcade's INFO level never shows: no PITWALL or Qt
        # window could ever attach again and nothing said why.
        consecutive_errors = 0
        while self._running and self._server_socket is not None:
            try:
                client_socket, addr = self._server_socket.accept()
            except OSError as err:
                if not self._running:
                    return  # `stop()` closed the socket; this is the exit path
                consecutive_errors += 1
                if consecutive_errors >= ACCEPT_ERROR_LIMIT:
                    logger.error(
                        "Stream server giving up on accept after %d consecutive errors; "
                        "no further subscribers can attach: %s",
                        consecutive_errors,
                        err,
                    )
                    return
                logger.warning("Stream server accept failed (%s); retrying", err)
                time.sleep(ACCEPT_RETRY_DELAY_S)
                continue
            consecutive_errors = 0
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
        """Watch one subscriber for its disconnect, and end when it does.

        It used to `time.sleep(1.0)` in a loop, which meant it could not see
        the remote end close: a client that disconnected was still counted
        until a LATER broadcast failed (the first send after a close lands in
        the kernel buffer and succeeds), and the thread itself slept on until
        the server stopped - measured at exactly one leaked thread per
        connection, five cycles taking the count from 2 to 7.

        `select` on the same cadence costs the same and answers the question.
        Subscribers never send, so readable means the peer sent FIN or a
        reset; either way this connection is over.
        """
        try:
            while self._running:
                readable, _, errored = select.select([client_socket], [], [client_socket], 1.0)
                if errored:
                    return
                if not readable:
                    continue
                try:
                    if client_socket.recv(1) == b"":
                        return  # clean FIN from the peer
                except OSError:
                    return
        finally:
            self._prune_clients([client_socket])

    def _prune_clients(self, dead: list[socket.socket]) -> None:
        """Forget these subscribers AND close their sockets.

        The close is the load-bearing half. A subscriber pruned for a send
        timeout has a perfectly healthy TCP connection - it simply stopped
        reading - so dropping it from the list left the consumer holding a
        socket that would never carry another byte and never reach EOF.
        Measured: a 1.24-second stall was enough to be pruned, after which
        that window sat on a dead feed showing a green "Connected" chip until
        the arcade process itself exited. Closing here makes the peer's next
        `recv` fail, which is the signal its reconnect loop already waits for.
        """
        with self._clients_lock:
            for client in dead:
                try:
                    self._clients.remove(client)
                except ValueError:
                    # Another thread already pruned this socket first — fine.
                    pass
        # Outside the lock: `close()` can block briefly and nothing else needs
        # the list to stay held while it does.
        for client in dead:
            try:
                client.close()
            except OSError:
                # Remote end may have already dropped the connection.
                pass
