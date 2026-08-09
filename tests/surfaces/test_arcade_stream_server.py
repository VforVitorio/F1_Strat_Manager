"""The broadcast server must never block the caller (`src/arcade/stream.py`).

`broadcast()` is called from the pyglet main thread on every due
`on_update`. `sendall` blocks. A subscriber that stops reading therefore
used to freeze the replay window itself, and the sprint's span change made
that far likelier by enlarging the message: the measured time-to-freeze
fell from about 130 s to 0.7 s, with the product always opening two
subscribers.

These tests pin the two properties that make that impossible: the caller
returns promptly whatever the socket does, and a consumer that falls
behind loses stale payloads rather than the newest one.
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
            server.broadcast({"seq": seq})
        elapsed = time.perf_counter() - started

        # 20 ticks is two seconds of real playback; anything near the 5 s the
        # stalled socket holds for would mean the caller was waiting on it.
        assert elapsed < 0.5, f"broadcast blocked the caller for {elapsed:.2f}s"
    finally:
        stalled.close()
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
            server.broadcast({"seq": seq})
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
        server.broadcast({"seq": 1, "model": {"lap_time_s": float("nan")}})
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
    handles fine — and because the first blame wins, a payload carrying
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

    So the whole class came back on a different input: `{np.int64(lap):
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
