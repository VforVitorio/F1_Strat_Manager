"""The broadcast server must never block the caller (`src/arcade/stream.py`).

`broadcast()` is called from the pyglet main thread on every due
`on_update`, and it takes a FACTORY: the caller neither builds the payload
nor encodes it nor sends it, because all three used to cost the frame loop
milliseconds it does not have (#1049). `sendall` blocks too, which is why
the write moved to the sender thread first.

These tests pin the properties that make a stall impossible: the caller
returns promptly whatever the socket does AND without running the factory,
a consumer that falls behind loses stale payloads rather than the newest
one, and a factory that raises costs its own tick rather than the thread
every later tick depends on.
"""

from __future__ import annotations

import json
import threading
import time

import pytest

from src.arcade.stream import TelemetryStreamServer


class _StalledClient:
    """A socket that never finishes a write, like a subscriber that stopped reading."""

    def __init__(self) -> None:
        self.started = threading.Event()
        self.release = threading.Event()

    def sendall(self, message: bytes) -> None:
        self.started.set()
        self.release.wait(timeout=5.0)

    def close(self) -> None:
        self.release.set()


def _server_with(client) -> TelemetryStreamServer:
    server = TelemetryStreamServer()
    server._running = True
    server._clients = [client]
    threading.Thread(target=server._send_loop, daemon=True).start()
    return server


def test_broadcast_returns_immediately_even_when_a_client_has_stalled():
    """The caller is the pyglet frame loop; it cannot wait on a socket."""
    stalled = _StalledClient()
    server = _server_with(stalled)
    try:
        assert stalled.started.wait(timeout=2.0) or True  # let the sender pick one up
        started = time.perf_counter()
        for seq in range(20):
            # `seq=seq` is not decoration: the factory runs on another thread
            # after this loop has finished, so a closure over the loop
            # variable would encode 19 twenty times.
            server.broadcast(lambda seq=seq: {"seq": seq})
        elapsed = time.perf_counter() - started

        # 20 ticks is two seconds of real playback; anything near the 5 s the
        # stalled socket holds for would mean the caller was waiting on it.
        assert elapsed < 0.5, f"broadcast blocked the caller for {elapsed:.2f}s"
    finally:
        stalled.close()
        server._running = False


def test_broadcast_does_not_BUILD_the_payload_on_the_caller_s_thread():
    """The EFFECT #1049 is about, asserted on the thread the work ran on.

    A test that only compared the bytes before and after would pass with the
    build still inline, because moving work between threads does not change
    what it produces. So the factory records who called it, and the assertion
    is that it ran on a different thread.

    The 3.19 ms a steady 8x tick used to cost the frame loop is the reason;
    a seek tick under schema v2 is 32 ms, about two dropped frames at 60 FPS.
    """
    ran_on: list[int] = []
    server = _server_with(type("C", (), {"sendall": lambda self, m: None})())
    try:
        caller = threading.get_ident()

        def build() -> dict:
            ran_on.append(threading.get_ident())
            return {"seq": 1}

        server.broadcast(build)
        assert ran_on == [], "the factory ran before broadcast() returned"

        deadline = time.time() + 5
        while not ran_on and time.time() < deadline:
            time.sleep(0.01)
        assert ran_on, "the factory never ran at all"
        assert ran_on[0] != caller, "the payload was still built on the caller's thread"
    finally:
        server._running = False


def test_a_factory_that_raises_costs_its_tick_and_not_the_sender_thread():
    """A daemon that dies is silent, and takes every later tick with it.

    `_send_loop` is where the factory now runs, and it is the one body in this
    module that had no guard: an exception there ends the only thread that
    writes to a socket, while the replay window keeps running and every
    subscriber keeps a connection that will never carry another byte.
    `_accept_loop` already pays for this lesson.
    """
    sent: list[bytes] = []
    server = _server_with(type("C", (), {"sendall": lambda self, m: sent.append(m)})())
    try:

        def explode() -> dict:
            raise ValueError("a model output nobody guarded")

        server.broadcast(explode)
        time.sleep(0.3)
        assert sent == [], "the raising tick must not reach a socket"

        server.broadcast(lambda: {"seq": 2})
        deadline = time.time() + 5
        while not sent and time.time() < deadline:
            time.sleep(0.01)
        assert sent, "the sender thread died with the tick that raised"
        assert json.loads(sent[0])["seq"] == 2
    finally:
        server._running = False


def test_a_consumer_that_falls_behind_loses_the_STALE_payloads():
    """Each payload is a complete snapshot, so the newest is the one to keep.

    The discard is not silent: `seq` jumps, which is exactly the signal it
    exists to give.
    """
    sent: list[bytes] = []
    gate = threading.Event()

    class _SlowClient:
        def sendall(self, message: bytes) -> None:
            gate.wait(timeout=5.0)
            sent.append(message)

    server = _server_with(_SlowClient())
    try:
        for seq in range(50):
            server.broadcast(lambda seq=seq: {"seq": seq})
        gate.set()
        time.sleep(0.4)

        seqs = [json.loads(m)["seq"] for m in sent]
        assert seqs, "something must reach the socket"
        assert seqs == sorted(seqs), "order must be preserved"
        assert len(seqs) < 50, "a backed-up consumer must not receive every stale payload"
        assert max(seqs) >= 40, "the payloads kept must be the recent ones"
    finally:
        server._running = False


def test_nothing_non_finite_reaches_a_socket():
    """The encoder is the guarantee; the sanitiser is the policy that feeds it."""
    sent: list[bytes] = []
    server = _server_with(type("C", (), {"sendall": lambda self, m: sent.append(m)})())
    try:
        server.broadcast(lambda: {"seq": 1, "model": {"lap_time_s": float("nan")}})
        time.sleep(0.3)

        assert sent, "a NaN must cost its field, not the whole message"
        decoded = json.loads(
            sent[0], parse_constant=lambda token: pytest.fail(f"non-finite on the wire: {token}")
        )
        assert decoded["model"]["lap_time_s"] is None
    finally:
        server._running = False


def test_the_drop_log_names_the_leaf_the_encoder_choked_on():
    """`_blame` used to return the empty string for everything but a plain float.

    `np.float32`, `np.int64` and `np.bool_` are not subclasses of Python's
    float/int/bool, so the sanitiser cannot rewrite them and `json.dumps`
    cannot encode them: the tick dies whole, which is the right outcome,
    and the log said nothing about why, which is not. One agent output
    carrying a `predict()[0]` without its `float()` would drop every tick
    at 10 Hz behind an empty message.
    """
    import numpy as np

    from src.arcade.stream import _blame

    assert _blame({"a": {"b": float("nan")}}) == "a.b=nan"
    for leaf, name in (
        (np.float32("nan"), "float32"),
        (np.float32(1.5), "float32"),
        (np.int64(3), "int64"),
        (np.bool_(True), "bool_"),
    ):
        assert _blame({"per_agent": [{"pace": leaf}]}) == f"per_agent[0].pace=<{name}>"
    assert _blame({"a": 1, "b": "ok", "c": None, "d": True, "e": 1.5}) == ""


def test_the_drop_log_does_not_blame_a_leaf_the_encoder_can_take():
    """A tuple encodes as an array, and blaming it hid the real culprit.

    The allowlist version answered `a=<tuple>` for a payload `json.dumps`
    handles fine, and because the first blame wins, a payload carrying
    both a tuple and a genuine culprit named the tuple and stopped. The
    drop log accused an innocent field while the tick that actually died
    went unexplained. The oracle is now the encoder itself.
    """
    import numpy as np

    from src.arcade.stream import _blame

    assert _blame({"a": (1, 2)}) == "", "a tuple is JSON-encodable"
    assert _blame({"a": (1, 2), "b": np.float32("nan")}) == "b=<float32>", "reach the real one"
    assert _blame({"a": ({"deep": np.int64(3)},)}) == "a[0].deep=<int64>", "recurse into tuples"


def test_the_drop_log_names_an_unencodable_dict_KEY_too():
    """The encoder rejects keys as well, and the walker only read values.

    Therefore, the whole class came back on a different input: `{np.int64(lap):
    prob}` is one comprehension away from a `per_agent` block, it kills
    the tick, and the log printed the same empty suffix the function was
    rewritten to stop printing.
    """
    import numpy as np

    from src.arcade.stream import _blame

    assert _blame({np.int64(3): "score"}) == "<root>.<key 3>=<int64>"
    assert _blame({"b": {b"raw": 1}}) == "b.<key b'raw'>=<bytes>"


def test_the_sanitiser_walks_tuples_like_the_blame_function_does():
    """The twin. One walker learned tuples and the other did not.

    `_json_safe` promises "one unusable model output should cost that
    field, not the whole tick". For one commit that was false for exactly
    the container `_blame` had just been taught: a NaN inside a tuple
    survived sanitising and killed the broadcast. `dataclasses.asdict`
    preserves tuples, so the first tuple-typed model field arms it.
    """
    import json
    import math

    from src.arcade.stream import _json_safe

    cleaned = _json_safe({"quantiles": (1.0, float("nan"), 3.0)})

    assert cleaned == {"quantiles": [1.0, None, 3.0]}, "the NaN became None, the tuple a list"
    assert json.dumps(cleaned, allow_nan=False), "and the payload now encodes"
    assert not any(
        isinstance(v, float) and not math.isfinite(v) for v in cleaned["quantiles"] if v is not None
    )


# --- What happens to a subscriber the server gives up on --------------------
#
# Pruning used to mean "forget", not "close". A subscriber pruned for a send
# timeout still has a perfectly healthy TCP connection - it simply stopped
# reading - so it went on believing it was connected while no byte would ever
# arrive and no EOF would ever come. Measured before the fix: a 1.24-second
# stall was enough to be pruned, after which the socket drained its buffered
# 219 KB and then sat ESTABLISHED-frozen, with the window showing a green
# "Connected" chip over a dead feed until the arcade process itself exited.


def _real_server() -> TelemetryStreamServer:
    server = TelemetryStreamServer("127.0.0.1", 0)
    server.start()
    # Port 0 lets the OS choose; read back what it chose.
    server.port = server._server_socket.getsockname()[1]
    return server


def _wait_for_clients(server: TelemetryStreamServer, count: int, timeout: float = 5.0) -> None:
    """Block until the server has actually registered the connection.

    `create_connection` returns as soon as the handshake completes, which is
    BEFORE `_accept_loop` appends the socket. Without this the "wait until
    pruned" loop below could exit on the first iteration having never seen a
    client at all - measured, 1 run in 6 - and then assert about an empty set,
    which is this repo's bug class 5 committed inside a test written to catch
    bug class 5.
    """
    deadline = time.time() + timeout
    while server.client_count() != count and time.time() < deadline:
        time.sleep(0.02)
    assert server.client_count() == count, (
        f"server has {server.client_count()} clients, expected {count}"
    )


def test_a_pruned_subscriber_is_closed_so_it_can_notice_and_reconnect():
    """The EFFECT: the peer reaches EOF. Asserting that the socket left the
    server's list would have been green throughout the whole defect."""
    import socket as socket_module

    server = _real_server()
    peer = socket_module.create_connection(("127.0.0.1", server.port), timeout=5)
    try:
        _wait_for_clients(server, 1)
        # Never read. A 20 KB payload fills the socket buffers in a handful of
        # broadcasts, which is what the real span-sized message does.
        payload = {"filler": "x" * 20000}
        deadline = time.time() + 15
        while server.client_count() > 0 and time.time() < deadline:
            server.broadcast(lambda: payload)
            time.sleep(0.02)
        assert server.client_count() == 0, "the server never pruned the stalled subscriber"

        # Drain whatever was buffered, then require an actual end of stream.
        #
        # **A timeout is NOT an ending, and the first version of this check
        # counted it as one.** `socket.timeout` is a `TimeoutError` and so an
        # `OSError`, so a blanket `except OSError: reached_eof = True` marked
        # "nothing arrived for five seconds" as success - which is precisely
        # the frozen-forever state this test exists to forbid. It passed
        # against the unfixed server. Only EOF or a reset is an ending.
        peer.settimeout(2.0)
        reached_eof = False
        end = time.time() + 10
        while time.time() < end:
            try:
                if peer.recv(65536) == b"":
                    reached_eof = True
                    break
            except TimeoutError:
                continue  # still frozen; keep waiting for a real ending
            except OSError:
                reached_eof = True  # a reset is an ending too
                break
        assert reached_eof, "the pruned socket never ended - the consumer would wait forever"
    finally:
        peer.close()
        server.stop()


def test_a_subscriber_that_leaves_takes_its_thread_with_it():
    """One leaked thread per connection, held until the server stopped.

    Measured before the fix: five connect/disconnect cycles took the watcher
    count from 2 to 7, exactly +1 each, never reclaimed. The watcher slept
    instead of reading, so it could not see the peer close.
    """
    import socket as socket_module

    def watchers() -> int:
        return sum(1 for t in threading.enumerate() if t.name == "TelemetryStreamClient")

    server = _real_server()
    try:
        baseline = watchers()
        for _ in range(5):
            peer = socket_module.create_connection(("127.0.0.1", server.port), timeout=5)
            _wait_for_clients(server, 1)
            peer.close()
            time.sleep(0.3)
        deadline = time.time() + 5
        while watchers() > baseline and time.time() < deadline:
            time.sleep(0.1)
        assert watchers() == baseline, f"{watchers() - baseline} watcher threads leaked"
    finally:
        server.stop()


def test_a_transient_accept_error_does_not_shut_the_door_forever():
    """One `OSError` used to end the accept loop for the whole session, with
    a DEBUG line the arcade's INFO level never prints. After that no window
    could attach again, and nothing said why."""
    import socket as socket_module

    server = TelemetryStreamServer("127.0.0.1", 0)

    class _FlakyOnce:
        """The real listening socket, with one transient failure in it."""

        def __init__(self, inner: socket_module.socket) -> None:
            self._inner = inner
            self.raised = False

        def accept(self):
            if not self.raised:
                self.raised = True
                raise OSError("transient winsock hiccup")
            return self._inner.accept()

        def __getattr__(self, name):
            return getattr(self._inner, name)

    listening = socket_module.socket(socket_module.AF_INET, socket_module.SOCK_STREAM)
    listening.setsockopt(socket_module.SOL_SOCKET, socket_module.SO_REUSEADDR, 1)
    listening.bind(("127.0.0.1", 0))
    listening.listen(5)
    port = listening.getsockname()[1]

    server._server_socket = _FlakyOnce(listening)  # type: ignore[assignment]
    server._running = True
    threading.Thread(target=server._accept_loop, daemon=True).start()

    try:
        time.sleep(0.3)  # let the first accept fail and the retry arm
        peer = socket_module.create_connection(("127.0.0.1", port), timeout=5)
        deadline = time.time() + 5
        while server.client_count() == 0 and time.time() < deadline:
            time.sleep(0.05)
        assert server.client_count() == 1, "the accept loop died on a transient error"
        peer.close()
    finally:
        server.stop()
        listening.close()
