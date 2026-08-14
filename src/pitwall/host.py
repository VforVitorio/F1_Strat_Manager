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

from src.f1_strat_manager.data_cache import get_data_root
from src.pitwall.agents_view import AgentsViewBuilder
from src.pitwall.session_data import SessionLaps, unavailable
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
        # The last label AGENTS was served, so that view can return on a
        # connection change with no new tick. NOT the memory the label itself
        # needs - see `_ever_connected`.
        self._agents_connection: str | None = None
        # Has the socket EVER been up? This is what separates "Connecting..."
        # from "Disconnected", and it belongs to the host because both windows
        # ask. It used to be inferred from `_agents_connection`, which made the
        # answer depend on whether the AGENTS window had polled: with only the
        # DATA window open, a producer that died read "Connecting..." forever.
        self._ever_connected = False
        # The BULK channel's state. `_bulk_reveal` is the last map served, so
        # the revision can advance on a rewind as readily as on a completed
        # lap; `_session_key` is what makes pointing the arcade at another
        # race replace the loaded laps instead of serving the previous one's.
        self._bulk_rev = 0
        self._bulk_reveal: dict[str, int] = {}
        self._session: SessionLaps | None = None
        self._session_key: tuple[int, str] | None = None
        # The LIVE channel's state, masked by the clock rather than by
        # completed laps. `_live_view` is the last block SERVED, so the
        # revision moves exactly when the screen would change and not ten
        # times a second - and it cannot miss a change, which a hash of the
        # block's shape could and did (#934).
        self._live_rev = 0
        self._live_view: dict[str, dict] = {}

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

    def get_connection(self) -> str:
        """The three states both windows paint: Connected / Connecting... / Disconnected.

        "Disconnected" needs a memory, because the socket looks identical
        before the arcade has ever spoken and after it has gone away. The
        honest word for the first is "Connecting...".

        The memory is `_ever_connected` and it is read from the CLIENT, not
        from what some window was last served: DATA's band-1 strip is the
        second caller, and a per-window memory would have told it
        "Connecting..." about a producer that had been up for an hour and
        died, whenever the AGENTS window was closed.

        Public because the strip polls it directly. It cannot be derived on
        the client side from tick freshness: a PAUSED replay stops sending
        ticks while the socket stays perfectly up, and the strip renders the
        pause right next to this - so the freshness heuristic would put
        "Disconnected" beside "PAUSED" and be wrong.
        """
        if self._client.connected:
            self._ever_connected = True
            return "Connected"
        return "Disconnected" if self._ever_connected else "Connecting..."

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
        connection = self.get_connection()
        payload = self.get_tick(since_seq)
        if payload is None:
            if connection == self._agents_connection:
                return None
            payload = self._client.latest
            if payload is None:
                return None
        self._agents_connection = connection
        return self._agents.build(payload, connection)

    def get_bulk(self, since_rev: int = -1) -> dict | None:
        """The race's lap table, masked to what the clock has revealed.

        The second data channel. The tick carries the instant; this carries
        everything the timing table and the bests panel show, which is static
        parquet known before lap 1 and therefore a progressive reveal rather
        than a stream to accumulate.

        **The comparison is inequality, exactly as `get_tick`'s is, and for a
        sharper reason.** A rewind LOWERS the revealed set, so `rev >
        since_rev` would withhold precisely the un-reveal - the client would
        keep rows the clock has taken back, which is the leak host-side
        masking exists to prevent, reintroduced one level up. A test that
        never rewinds stays green through it.

        The revision advances when the reveal map changes in EITHER
        direction, so a caller holding `rev` is holding "the view I have is
        current", not "I have seen this many laps".

        Returns None when the caller is up to date, matching `get_tick`: the
        UI keeps what it has. The race being absent from disk is NOT that
        case - it returns an explicit unavailable payload, because a tower
        rendering zero rows silently is the same pixel as a tower whose
        reveal is broken.
        """
        payload = self._client.latest
        if payload is None:
            return None
        arcade = payload.get("arcade") or {}
        reveal = self._reveal_map(arcade)
        if reveal != self._bulk_reveal:
            self._bulk_reveal = reveal
            self._bulk_rev += 1
        if self._bulk_rev == since_rev:
            return None

        view = self._masked_view(arcade, reveal)
        view["rev"] = self._bulk_rev
        return view

    def get_live_lap(self, since_rev: int = -1) -> dict | None:
        """The lap each driver is ON, with only the sectors he has crossed.

        The tower's sector columns show the lap in progress, blank at the line
        and filling as each sector goes by. That needs a mask driven by the
        CLOCK rather than by completed laps, and the two cannot share one
        payload: a sector opens somewhere in the field every 2.22 s, and
        re-sending the whole revealed race at that cadence is about 154 KB/s
        against the tick's own 58. This block is 2 KB for twenty drivers.

        The host still applies the mask, which is the window's load-bearing
        invariant. What changes is which clock it reads.

        The revision compares on inequality for the same reason `get_bulk`'s
        does: a rewind CLOSES sectors, and `rev > since_rev` would withhold
        exactly that - leaving a sector on screen that the car has not yet
        reached this time round.
        """
        payload = self._client.latest
        if payload is None:
            return None
        arcade = payload.get("arcade") or {}
        session = self._session_for(arcade)
        if session is None:
            return None

        reveal = self._reveal_map(arcade)
        clock = float(arcade.get("t") or 0.0)
        global_t_min = float(arcade.get("global_t_min") or 0.0)
        drivers = session.live_lap(reveal, clock, global_t_min)

        # **The signature IS the payload, not a hash of its shape.** It used
        # to be `tuple(v is not None for v in row.values())` - which cells are
        # filled - and that is lossy in exactly the direction that leaks: a
        # (driver, lap) pair determines the values, but the lap entered the
        # signature only as a constant True. Measured on the real race, 3,667
        # sampled pairs at least ten seconds apart share a presence pattern
        # with completely different numbers, the worst of them a 28-minute
        # rewind across the wet start - and across it `get_live_lap` answered
        # None, so the client kept a dry-lap sector time on a screen whose
        # clock said lap 2 (#934).
        #
        # `get_bulk` was immune because its signature is the reveal MAP, which
        # is its full input state. A signature that does not determine the
        # payload is not a signature.
        #
        # Comparing the dict itself still leaves the revision still through
        # the hundreds of ticks between two crossings, which is the whole
        # point of having one.
        if drivers != self._live_view:
            self._live_view = drivers
            self._live_rev += 1
        if self._live_rev == since_rev:
            return None
        return {"rev": self._live_rev, "drivers": drivers}

    @staticmethod
    def _reveal_map(arcade: dict) -> dict[str, int]:
        """Laps completed per driver, which is what both readers mask on."""
        return {
            code: int(state.get("laps_completed") or 0)
            for code, state in (arcade.get("drivers") or {}).items()
        }

    def _session_for(self, arcade: dict) -> SessionLaps | None:
        """The loaded race for this tick, or None when it is not on disk.

        Cached on (year, location) so pointing the arcade at another race
        replaces the laps instead of serving the previous one's - the
        stale-state class #904 already paid for once on the AGENTS history.
        """
        year, location = arcade.get("year"), arcade.get("location")
        if not isinstance(year, int) or not isinstance(location, str):
            return None
        if self._session_key != (year, location):
            self._session_key = (year, location)
            self._session = SessionLaps.load(get_data_root(), year, location)
        return self._session

    def _masked_view(self, arcade: dict, reveal: dict[str, int]) -> dict:
        """Load the race once, then slice it; or say it is not on disk.

        The load is cached on (year, location) rather than repeated, and the
        cache is keyed so that pointing the arcade at another race replaces
        it instead of serving the previous one's laps - the stale-state class
        #904 already paid for once on the AGENTS history.
        """
        year, location = arcade.get("year"), arcade.get("location")
        session = self._session_for(arcade)
        if session is None:
            return unavailable(
                year if isinstance(year, int) else None,
                location if isinstance(location, str) else None,
            )
        return session.masked_view(reveal, float(arcade.get("global_t_min") or 0.0))

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
