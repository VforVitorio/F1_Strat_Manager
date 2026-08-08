"""The object the two windows see as `pywebview.api`, and nothing else.

It holds no rendering logic and no formatting. Every method maps one to one
onto something the UI needs, and each is short enough to read at a glance.

Two properties this file exists to guarantee, both of which were named as
traps before a line was written:

1. **`get_tick` is sequenced, never a blind slot.** Two windows polling one
   latest-payload slot on independent 10 Hz timers were measured reading a
   different frame on 58 % of polls - 15 duplicate reads and 15 skips out of
   54. Passing the last sequence a window saw removes both, and the sequence
   is not invented here: the producer already stamps `seq` on every message.
2. **Closing one window must not stop the shared client.** The client is
   owned by this host, not by a window, and the count below is what makes
   that explicit rather than accidental. It is the single place the property
   can regress, and it regresses silently.
"""

from __future__ import annotations

import logging

from src.pitwall.agents_view import AgentsViewBuilder
from src.pitwall.stream_client import ArcadeStreamClient

logger = logging.getLogger(__name__)


class PitwallHost:
    """The js_api surface: one sequenced tick reader, shared by every window.

    Invariants:

    - The client is started once and stopped once, by this object. A window
      never touches it.
    - `get_tick` returns None when the caller is already up to date. That is
      "nothing new", not "nothing there": the UI keeps what it has.
    """

    def __init__(self, client: ArcadeStreamClient, window_count: int) -> None:
        self._client = client
        self._windows_open = window_count
        self._agents = AgentsViewBuilder()
        self._agents_connection: str | None = None

    def start(self) -> None:
        self._client.start()

    def get_tick(self, since_seq: int = -1) -> dict | None:
        """Return the latest payload if the caller has not seen it yet.

        `since_seq` is the `seq` of the last payload this window rendered.
        A window that has never rendered one passes -1, which is below every
        sequence the producer emits.

        **The comparison is inequality, not "greater than".** The slot holds
        only the newest payload, so anything the caller has not already
        rendered is news - including a LOWER sequence, which can only mean
        the producer restarted. `seq > since_seq` withheld exactly that
        case: relaunch the arcade with the windows open and both froze on
        the dead race, `live` and silent, until the new run's sequence
        passed the old one's. A ten-minute race meant ten minutes of frozen
        screen. The consumer side already copes, because the new run's
        `frame_index` jumps backwards and `FrameClock` reports `rewound`.

        A payload with no `seq` is returned unconditionally. That only
        happens against a producer older than the one in this repo, where
        there is nothing to compare and the honest answer is the data.
        """
        payload = self._client.latest
        if payload is None:
            return None
        seq = payload.get("seq")
        if seq is None or seq != since_seq:
            return payload
        return None

    def _connection_label(self) -> str:
        """The three states `HeaderBar.set_connection` paints.

        "Disconnected" needs a memory: before the first tick the socket is
        retrying and the honest word is "Connecting...", while after one
        the same state means the arcade went away.
        """
        if self._client.connected:
            return "Connected"
        return "Disconnected" if self._agents_connection else "Connecting..."

    def get_agents_view(self, since_seq: int = -1) -> dict | None:
        """The whole AGENTS window, already formatted, or None when nothing changed.

        The window is a renderer. Every headline, body line, colour and
        status glyph in the returned dict is produced by the code that
        paints the Qt window, so the two cannot describe the same lap
        differently - which is what "1:1" has to mean if it is going to
        survive a sprint.

        Returned on a tick the caller has not seen, **and** on a change of
        connection state with no new tick, which is the only way a window
        learns the arcade died: once the producer stops, `seq` stops
        advancing and a purely sequence-driven view would keep rendering
        the last frame of a dead race with a green "Connected" chip.
        """
        connection = self._connection_label()
        payload = self.get_tick(since_seq)
        if payload is None:
            if connection == self._agents_connection:
                return None
            payload = self._client.latest
            if payload is None:
                return None
        self._agents_connection = connection
        return self._agents.build(payload, connection)

    def release_window(self) -> int:
        """Record that one window has closed; stop the client at the last one.

        Returns how many are still open, which is what makes the property
        testable without a display: closing one of two must leave the client
        running, and only the second close tears it down.
        """
        self._windows_open = max(0, self._windows_open - 1)
        if self._windows_open == 0:
            logger.info("Last Pitwall window closed - stopping the stream client")
            self._client.stop()
        return self._windows_open

    def shutdown(self) -> None:
        """Unconditional teardown, for the process exiting rather than a window closing."""
        self._windows_open = 0
        self._client.stop()
