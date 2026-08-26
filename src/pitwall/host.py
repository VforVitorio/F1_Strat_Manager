"""The object the two windows see as `pywebview.api`, and nothing else.

It holds no rendering logic and no formatting. Every method maps one to one
onto something the UI needs, and each is short enough to read at a glance.

Two properties this file exists to guarantee, both of which were named as
traps before a line was written:

1. **`get_tick` is sequenced, never a blind slot.** Two windows polling one
   latest-payload slot on independent 10 Hz timers were measured reading a
   different frame on 58% of polls - 15 duplicate reads and 15 skips out of
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
from src.pitwall.agents_view.panels import CONNECTION_COLOURS
from src.pitwall.radio_feed import RadioCorpus
from src.pitwall.radio_feed import unavailable as radio_unavailable
from src.pitwall.session_data import SessionLaps, unavailable
from src.pitwall.stream_client import ArcadeStreamClient, TickSignals

logger = logging.getLogger(__name__)


def with_missed_signals(payload: dict, signals: tuple[TickSignals, ...], since_seq: int) -> dict:
    """Return `payload` carrying the eviction signals of the ticks the caller missed.

    `rewound` and `dropped` describe the gap between two ticks, not the state of
    one, so a tick the latest-payload slot overwrote before a window polled took
    its signal with it. Folding the missed range forward is what makes the signal
    survive the discard (#1060).

    **The fold lands on a COPY, three levels deep.** `get_tick` hands every caller
    the same dict object, so folding in place lets one window rewrite the block
    another is still holding: window A folds `dropped=5`, window B polls with a
    different cursor and rewrites the block to its own range, and A's payload now
    reads 0 before it is ever serialised. Copying the three containers the fold
    touches costs three shallow dicts a poll and makes the answer the caller's own.

    **The range is expressed in ARRIVAL order, never in `seq`.** `seq` restarts at
    1 when the arcade relaunches - the case this module's own comparison exists for
    - so a range keyed on it either excludes every entry of the new run or, once
    two runs' numbers collide, matches twice.

    A cursor the log cannot place is served UNMERGED: a window polling for the
    first time, or one asleep past the log's 6.4 s, has no knowable range, and
    inventing a `dropped` for it would be a fabricated number in the one field
    whose whole job is to say something real happened.

    --- WHERE TO CHANGE IF THE WIRE'S CONTINUITY FIELDS CHANGE ---
    `src/arcade/app.py:_telemetry_span_bounds` produces them, `_signals_of` in
    `stream_client.py` reads them off the payload, and `lib/frameClock.ts` plus
    `features/data/useTraceFrame.ts` consume them. A third continuity field has to
    land in all four.
    """
    missed = _missed_after(signals, since_seq)
    if not missed:
        return payload
    rewound = any(entry.rewound for entry in missed)
    dropped = sum(entry.dropped for entry in missed)
    arcade = payload.get("arcade")
    telemetry = (arcade or {}).get("telemetry")
    if not isinstance(arcade, dict) or not isinstance(telemetry, dict):
        return payload  # a producer that sends no telemetry block has none to fold
    # `missed` always ends with the served payload's own entry, so these two are
    # already the merged answer; when they equal what the payload says, nothing
    # was discarded and the caller keeps the original object.
    if rewound == bool(telemetry.get("rewound")) and dropped == telemetry.get("dropped"):
        return payload
    merged_telemetry = {**telemetry, "rewound": rewound, "dropped": dropped}
    return {**payload, "arcade": {**arcade, "telemetry": merged_telemetry}}


def _missed_after(signals: tuple[TickSignals, ...], since_seq: int) -> list[TickSignals]:
    """The log entries after the caller's cursor, up to and including the newest.

    The newest entry always describes the payload being served, because
    `snapshot()` reads the slot and the log under one lock.

    When `since_seq` appears more than once - only reachable if a relaunched
    producer's numbering reaches an old entry that the reconnect did not clear -
    the NEWEST match wins. It yields the smaller range, and over-reporting
    `dropped` costs a spurious eviction of the very buffer this protects.
    """
    if not signals:
        return []
    for index in range(len(signals) - 1, -1, -1):
        if signals[index].seq == since_seq:
            return list(signals[index + 1 :])
    return []


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
        # Has the socket EVER been up? This is what separates "Connecting..."
        # from "Disconnected", and it belongs to the host because both windows
        # ask. Inferring it from `_agents_connection` made the answer depend on
        # whether the AGENTS window had polled: with only the DATA window open,
        # a producer that died read "Connecting..." forever.
        self._ever_connected = False
        # The BULK channel's state. `_bulk_signature` is (year, location,
        # reveal map) - everything the payload is a function of - so the
        # revision advances on a rewind as readily as on a completed lap, and
        # cannot miss a race switch. `_session_key` is what makes pointing the
        # arcade at another race replace the loaded laps rather than serve the
        # previous one's.
        self._bulk_rev = 0
        self._bulk_signature: tuple | None = None
        self._session: SessionLaps | None = None
        self._session_key: tuple[int, str] | None = None
        # The radio/RCM feed of the SAME race, loaded under the SAME key in
        # `_session_for`. It has no revision of its own: it rides in the bulk
        # payload because it is a function of the bulk's signature exactly.
        self._radio: RadioCorpus | None = None
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

        **The payload carries the eviction signals of the ticks this caller
        never saw**, folded in by `with_missed_signals`. `rewound` and
        `dropped` describe what happened BETWEEN two ticks rather than the
        state of one, so a tick the slot overwrote before this window polled
        used to take its signal with it - and `FrameClock` then reported
        `continuous` across the hole while the trace buffer kept appending
        samples from unrelated parts of the race (#1060).
        """
        payload, signals = self._client.snapshot()
        if payload is None:
            return None
        seq = payload.get("seq")
        if seq is None or seq != since_seq:
            return with_missed_signals(payload, signals, since_seq)
        return None

    def _connection_label(self) -> str:
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

    def get_connection(self) -> dict[str, str]:
        """The socket state as a word AND its colour, from one map.

        **The colour rides with the word because it used to ride separately.**
        The AGENTS window took it from `CONNECTION_COLOURS` through the view,
        the DATA strip mapped the same three words to CSS classes of its own,
        and the two disagreed about "Connecting..." - amber on one window, dim
        grey on the other, for one socket, on two windows a reader has open
        side by side. A word plus a colour from the same lookup cannot do that.

        Also, the AGENTS window could not paint the word at all before its first
        tick: its boot literal hardcoded "Connecting..." in amber, so a socket
        that came up and had not yet delivered a lap read as still connecting,
        in the wrong colour, for the whole startup.
        """
        label = self._connection_label()
        return {"label": label, "colour": CONNECTION_COLOURS[label]}

    def get_agents_view(
        self, since_seq: int = -1, since_connection: str | None = None
    ) -> dict | None:
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

        **`since_connection` is what the CALLER last rendered, not what the host
        last saw (#950).** A single host field left two consumers racing for
        it: the first to notice the producer had died consumed the transition
        and the second never learned about it. Measured over 50
        polls, a browser on `/agents.html` kept a green chip on a dead race
        forever while the window beside it had already gone red. The loopback
        server is not hypothetical - `__main__` starts it unconditionally.

        A host field cannot answer "has this changed since YOU looked"; only
        the caller knows. So it joins `since_seq` and `since_rev`, which solved
        the identical problem for the tick and the bulk by asking rather than
        remembering. Adding a second host field instead would have been the
        third copy of one mistake - and the fix for its sibling is three lines
        above, where `_ever_connected` was lifted OUT of exactly this slot.

        `None` means "I have rendered nothing", so a first poll always gets a
        view.
        """
        connection = self._connection_label()
        payload = self.get_tick(since_seq)
        if payload is None:
            if connection == since_connection:
                return None
            payload = self._client.latest
            if payload is None:
                return None
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
        # The RACE is part of the signature, not only the reveal map. The map
        # alone does not determine this payload: a race switch whose first
        # observed tick carried the previous race's per-driver counters would
        # serve the old table as current. Unreachable in practice - a new
        # replay's first tick is an all-zero map - but it is the structural
        # rule #934 just cost, one channel over: a signature that does not
        # determine the payload is not a signature.
        signature = (arcade.get("year"), arcade.get("location"), reveal)
        if signature != self._bulk_signature:
            self._bulk_signature = signature
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
            # The twin `get_bulk` has always had this branch and this one did
            # not. Plain None means "keep what you have", so pointing the
            # arcade at a race with no parquet left the PREVIOUS race's sector
            # times, flags and colours on the new race's rows indefinitely,
            # beside a table that had correctly gone to `available=False`.
            # An empty block once is the honest render: the cells fall to
            # dashes and stay there. `since_rev` goes through so it is served
            # ONCE and not re-sent on every poll of a race that has no laps.
            return self._serve_live({}, since_rev)

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
        return self._serve_live(drivers, since_rev)

    def _serve_live(self, drivers: dict, since_rev: int = -1) -> dict | None:
        """Advance the revision when the block changed, then answer the caller."""
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

        **The radio corpus loads HERE, under the same key, not beside it.** Two
        caches on the same race with two invalidation points is how one of them
        comes to serve the previous race, which is the twin this repo pays for
        most often, and which has already been caught between these very two
        channels.

        **The malformed-tick return clears them too, which it did not.** One
        invalidation point is not enough if there is an early return above it:
        a tick naming no race sent the TABLE to its unavailable payload while
        `_masked_view` went on serving the previous race's radio out of a
        corpus this method had skipped over - 46 messages of a race the panel
        beside it had already given up on. Not reachable from today's producer,
        which always publishes an int year and a str location, and fixed anyway
        because the branch exists to be defensive and was not.
        """
        year, location = arcade.get("year"), arcade.get("location")
        if not isinstance(year, int) or not isinstance(location, str):
            self._session_key = None
            self._session = None
            self._radio = None
            return None
        if self._session_key != (year, location):
            self._session_key = (year, location)
            self._session = SessionLaps.load(get_data_root(), year, location)
            self._radio = RadioCorpus.load(get_data_root(), year, location)
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
            view = unavailable(
                year if isinstance(year, int) else None,
                location if isinstance(location, str) else None,
            )
        else:
            view = session.masked_view(reveal, float(arcade.get("global_t_min") or 0.0))
        # The radio feed rides IN this payload rather than on a channel of its
        # own, because it is a pure function of exactly what already signs this
        # one: (year, location, reveal map). A second channel would need a
        # second signature, and a signature that does not determine its payload
        # is the defect #934 cost a sprint. Measured cost of carrying it: the
        # bulk is 66,991 / 152,657 / 337,289 bytes at reveal L10 / L24 / L57 on
        # the real Melbourne payload, and the largest feed in the whole corpus
        # (Monaco, 210 events) is about 31 KB - 9%.
        view["radio"] = (
            radio_unavailable() if self._radio is None else self._radio.masked_view(reveal)
        )
        return view

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
