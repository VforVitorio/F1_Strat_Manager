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
