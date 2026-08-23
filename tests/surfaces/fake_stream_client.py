"""One stand-in for `ArcadeStreamClient`, shared by every surface test that needs it.

**One, not two.** `test_pitwall_host.py` and `test_pitwall_agents_view.py` each grew
their own `_FakeClient` with the same docstring, and when the client gained
`snapshot()` for #1060 only one of them learned about it: fourteen tests in the
other file failed on a method their fake did not have. That is this repo's dominant
defect wearing a test helper, so the copy is deleted rather than patched.

What it has to be faithful about, because a stub that is not makes the guards pass
against the fix and against the defect alike:

- `latest` and the signal log move TOGETHER. The real client appends to the log
  inside the same lock that assigns the slot, which is what makes `snapshot()` a
  consistent pair; here `latest` is a property whose setter does both, so a test
  that assigns `client.latest = payload` cannot leave the two disagreeing.
- The log is BOUNDED by the real `SIGNAL_LOG_DEPTH`. Written as a plain list, this
  fake remembered 69 ticks the real deque had already evicted and the
  unplaceable-cursor guard failed against correct code.
- The entries are built by the real `_signals_of`, so a change to what a payload's
  continuity flags mean reaches the fake without anyone editing it.
"""

from __future__ import annotations

from collections import deque

from src.pitwall.stream_client import SIGNAL_LOG_DEPTH, TickSignals, _signals_of


class FakeStreamClient:
    """Stands in for the socket, so the host's own logic is what is tested."""

    def __init__(self, payload: dict | None = None, connected: bool = True) -> None:
        self._latest: dict | None = None
        self._arrival = 0
        self._signals: deque[TickSignals] = deque(maxlen=SIGNAL_LOG_DEPTH)
        self.connected = connected
        self.started = False
        self.stopped = False
        if payload is not None:
            self.latest = payload

    @property
    def latest(self) -> dict | None:
        return self._latest

    @latest.setter
    def latest(self, payload: dict | None) -> None:
        """Publish one tick, slot and log together, exactly as `_consume` does."""
        self._latest = payload
        if payload is None:
            return
        self._arrival += 1
        self._signals.append(_signals_of(payload, self._arrival))

    def receive(self, payload: dict) -> None:
        """Read as intent at a call site that is publishing rather than assigning."""
        self.latest = payload

    def snapshot(self) -> tuple[dict | None, tuple[TickSignals, ...]]:
        return self._latest, tuple(self._signals)

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.stopped = True
