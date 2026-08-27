"""The Arcade race replay view orchestrating playback, cars, and panels.

Refactored from a root `arcade.Window` into an `arcade.View` so the menu
view can spawn a fresh replay whenever the user confirms a configuration.
Construction contract: the caller creates the `arcade.Window` first, loads
`SessionData` + `Track` in the main thread, and hands them plus the window
reference to this view so every `arcade.Text` allocated here (and in child
panels) has an active GL context from the start.
"""

from __future__ import annotations

import logging
import math
import os
import subprocess
import sys

import arcade
from src.arcade.config import (
    ACCENT,
    BG_COLOR,
    CAR_BG_ALPHA,
    CAR_BG_RADIUS,
    CAR_BORDER_COLOR,
    CAR_BORDER_WIDTH,
    CAR_LABEL_FONT_SIZE,
    CAR_RADIUS,
    DEFAULT_SPEED_IDX,
    DRIVER_BOX_GAP,
    DRIVER_BOX_HEIGHT,
    DRIVER_BOX_WIDTH,
    DRS_OPEN_CODES,
    FONT_BODY,
    FONT_TITLE,
    FPS,
    LEADERBOARD_RIGHT_MARGIN,
    LEADERBOARD_WIDTH,
    MARGIN_BOTTOM,
    MARGIN_LEFT,
    MARGIN_RIGHT,
    MARGIN_TOP,
    PLAYBACK_SPEEDS,
    SEEK_RATE_MULTIPLIER,
    STREAM_BROADCAST_EVERY_N_FRAMES,
    STREAM_HISTORY_TAIL,
    STREAM_HOST,
    STREAM_MAX_SPAN_FRAMES,
    STREAM_PORT,
    STREAM_SCHEMA_VERSION,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
    TEXT_TERTIARY,
)
from src.arcade.data import FrameData, SessionData
from src.arcade.gaps import RaceGapCalculator
from src.arcade.overlays import (
    ControlsLegend,
    DriverInfoPanel,
    LeaderboardPanel,
    ProgressBar,
    RaceEventsPanel,
    WeatherPanel,
    track_status_label,
)
from src.arcade.track import Track

logger = logging.getLogger(__name__)


def _telemetry_span_bounds(
    last_sent_idx: int, frame_idx: int, max_span: int, moved_back: bool = False
) -> tuple[int, bool, int]:
    """Return `(span_start, rewound, dropped)` for the frames this tick should send.

    The span is `frames[span_start : frame_idx + 1]`, so `span_start >
    frame_idx` is how "no new samples" is expressed. Pause and rewind get
    explicit branches here rather than falling out of a negative slice,
    which would silently send the wrong window:

    - **forward** (the normal case): everything after the last frame sent,
      up to and including the current one. Every frame the clock crossed
      goes out exactly once, at any playback speed.
    - **paused**: `frame_idx == last_sent_idx`, so the span is empty. Zero
      new samples, not a repeat of the last one. Repeats are what made a
      paused arcade look alive on the wire while nothing moved.
    - **rewound**: the user seeked backwards. The span is empty and the
      caller is told, because a consumer keyed on distance-within-lap now
      holds samples for track the car has yet to re-drive, and only a
      clear fixes that.

    `max_span` caps the span. Smooth playback never reaches it: the widest
    the clock produces is about 60 frames per tick, 0.1 s at 8x with the
    seek multiplier. What does reach it is **a click on the progress bar**,
    which can jump the index by tens of thousands of frames, and a process
    stall. Either way the frames in between are not sent, so the count is
    returned and published: a consumer that only saw `rewound` would read a
    contiguous `seq` and a forward-only clock and conclude nothing was
    lost, which is precisely the thing `seq` exists to make impossible.

    `moved_back` carries what the integer comparison cannot see: the clock
    is a float, so a backwards seek that does not cross a frame boundary
    looks exactly like a pause and the consumer never clears. It is
    reachable on any `on_key_release(LEFT)` landing mid-tick, and bounded
    at one frame of stale samples, which is why it is small and not zero.
    """
    if frame_idx < last_sent_idx or moved_back:
        return min(frame_idx, last_sent_idx) + 1, True, 0
    span_start = last_sent_idx + 1
    capped = max(span_start, frame_idx - max_span + 1)
    return capped, False, capped - span_start


def _frames_to_telemetry_span(
    frames: list | None,
    span_start: int,
    frame_idx: int,
    circuit_length_m: float,
    has_position: bool = True,
) -> list[dict]:
    """Pack `frames[span_start : frame_idx + 1]` for the wire, oldest first.

    Bounds are clamped rather than trusted, but not for the reason this
    docstring used to give. `SessionLoader` resamples every driver onto the
    one global timeline, so all twenty arrays are exactly `total_frames`
    long: measured on Melbourne 2025, 154,173 for all of them, no exceptions.
    Nothing here is ever "shorter than the timeline".

    What the clamp actually guards is the CALLER's arithmetic. `span_start`
    is `last_sent_idx + 1` and can sit past the end on the last tick of a
    race, and `frame_idx` is a float clock truncated to an int; the empty
    slice both produce is the honest answer, and a raise would not be.
    """
    if not frames:
        return []
    lo = max(0, span_start)
    hi = min(len(frames) - 1, frame_idx)
    samples = (
        _frame_to_telemetry(frames[i], circuit_length_m, has_position) for i in range(lo, hi + 1)
    )
    return [s for s in samples if s is not None]


def _lap_fraction(rel_dist: float) -> float | None:
    """Clamp a fraction-of-lap into [0, 1], or return None when it is unknown.

    The loader now derives `rel_dist` from the driver's own distance
    (`data.py:_lap_fraction_from_distance`), so it is finite and inside
    [0, 1] for every frame of every driver, so this guard fires on nothing
    in the sessions measured. It stays because the two failure modes it
    exists for are both silent, and one of them was live:

    - ``json.dumps`` writes a bare ``NaN``, which Python's own parser
      accepts and ``JSON.parse`` rejects, so a single unknown value took
      the whole payload down for a web consumer. FastF1 used to leave
      ``RelativeDistance`` NaN for 100% of one driver's frames.
    - clamping is worse than dropping: ``min(1.0, nan)`` is ``1.0``, and
      1.0 means "at the line", so the car with no position data would be
      drawn exactly on the lap boundary rather than nowhere.

    Unknown stays None. `has_position` on the wire is what tells a
    consumer to say so instead of rendering an empty chart.
    """
    value = float(rel_dist)
    if not math.isfinite(value):
        return None
    return min(1.0, max(0.0, value))


def _frame_to_telemetry(frame, circuit_length_m: float, has_position: bool = True) -> dict | None:
    """Pack a ``FrameData`` into the dict the telemetry window consumes.

    Uses ``frame.rel_dist * circuit_length`` as the broadcast ``dist``
    because ``frame.dist`` is the race-cumulative accumulator and would
    push the X axis to tens of kilometres as the race progresses. The
    telemetry chart wants per-lap distance (resets to 0 each lap) so
    the traces always occupy the full circuit-length range.

    Throttle and brake arrive already on 0-100 and already clamped: the
    scale is decided ONCE per session in `data.py` (`_pedal_multiplier`),
    not guessed per frame here. The guess was `if value <= 1.0:
    value *= 100`, which cannot tell "0-1 scale, full throttle" from
    "0-100 scale, barely lifting" and published 72,104 sub-1% openings as
    80-odd per cent on Melbourne 2025 alone. ``t`` is included so the
    delta-time chart can interpolate rival vs main."""
    if frame is None:
        return None
    throttle = float(frame.throttle)
    brake = float(frame.brake)
    rel_dist = _lap_fraction(frame.rel_dist) if has_position else None
    lap_dist = None if rel_dist is None else round(rel_dist * float(circuit_length_m or 0.0), 1)
    return {
        "lap": int(frame.lap),
        "t": round(float(frame.t), 3),
        "dist": lap_dist,
        "speed": round(float(frame.speed), 1),
        "throttle": round(throttle, 1),
        "brake": round(brake, 1),
        "gear": int(frame.gear),
        "drs": int(frame.drs),
        # **Decoded here, not forked into TypeScript.** The open set lives in
        # `config.DRS_OPEN_CODES`; `OwnCarTraces` refuses to fork it into
        # TypeScript, which is why its DRS lane could not exist until the producer
        # published the answer instead of the code - the same treatment
        # `track_status_label` already gets. EVERY driver's span flows through this
        # one function, so no car's lane can be fed by a different rule. (It said
        # "both spans, main and rival" until schema v2 put twenty on the wire.)
        # No schema bump: `stream.py`'s own contract says adding a key an old
        # consumer can ignore does not bump it.
        "drs_open": int(frame.drs) in DRS_OPEN_CODES,
    }


class F1ArcadeView(arcade.View):
    """Renders the race replay and owns the playback state machine.

    Lives inside a `arcade.Window` provided by `main.py`. The window is
    passed in so self.window is populated immediately: every arcade.Text
    created in this __init__ or its child panels sees the active GL context
    right away. Call via `window.show_view(F1ArcadeView(window, ...))`."""

    def __init__(
        self,
        window: arcade.Window,
        session_data: SessionData,
        track: Track,
        driver_main: str,
        driver_rival: str | None = None,
        year: int = 2024,
        strategy_enabled: bool = False,
        team: str | None = None,
    ) -> None:
        super().__init__(window=window)
        arcade.set_background_color(BG_COLOR)

        self._session = session_data
        self._track = track
        # Built once: finding the lap-line crossings walks every driver's
        # frames, and each query afterwards is two dict lookups. Building it
        # lazily would put that walk inside a frame rather than inside the
        # load the user is already waiting through.
        self._gaps = RaceGapCalculator(session_data)
        self._driver_main = driver_main
        self._driver_rival = driver_rival
        self._year = year
        self._strategy_enabled = strategy_enabled
        self._team = team
        self._strategy_connector = None  # set by __init__ if strategy_enabled
        self._strategy_state = None
        self._stream_server = None
        self._pitwall_proc: subprocess.Popen | None = None
        # Pushed onto the WINDOW (not the view) when the strategy layer starts,
        # popped in `on_hide_view`. See `_on_window_close`.
        self._close_handler: dict[str, object] | None = None
        self._broadcast_tick: int = 0
        # Highest frame index already put on the wire. -1 means "nothing sent
        # yet", so the first broadcast emits a single sample rather than the
        # whole race. Advanced on every due tick even when no client is
        # attached, so a consumer that connects on lap 40 gets the current
        # tick's span and not 40 laps of backlog.
        self._last_broadcast_idx: int = -1
        # Counts payloads actually put on the wire, so a consumer can tell a
        # duplicate read from a skipped one. Both are real: two independent
        # 10 Hz pollers reading one latest-payload slot were measured
        # duplicating 15 of 54 reads and skipping 15 of 54. Without a
        # sequence neither is visible from the consumer side.
        self._broadcast_seq: int = 0
        # The float clock as of the last due tick. `_last_broadcast_idx` is
        # truncated, so on its own it cannot see a sub-frame rewind.
        self._last_broadcast_clock: float = -1.0

        self._frame_index: float = 0.0
        self._speed_idx: int = DEFAULT_SPEED_IDX
        self._is_paused: bool = False
        self._is_rewinding: bool = False
        self._is_forwarding: bool = False
        self._was_paused_before_hold: bool = False
        self._show_progress_bar: bool = True
        self._show_drs_zones: bool = True
        self._show_all_cars: bool = True
        self._selected_drivers: set[str] = {driver_main}
        if driver_rival:
            self._selected_drivers.add(driver_rival)

        w, h = window.width, window.height
        self._track.update_scaling(
            w,
            h,
            margin_left=MARGIN_LEFT,
            margin_right=MARGIN_RIGHT,
            margin_bottom=MARGIN_BOTTOM,
            margin_top=MARGIN_TOP,
        )

        self._lap_label = arcade.Text(
            "LAP",
            20,
            h - 20,
            ACCENT,
            11,
            bold=True,
            font_name=FONT_TITLE,
            anchor_x="left",
            anchor_y="top",
        )
        self._lap_text = arcade.Text(
            "1/58",
            20,
            h - 36,
            TEXT_PRIMARY,
            22,
            bold=True,
            font_name=FONT_TITLE,
            anchor_x="left",
            anchor_y="top",
        )
        self._time_text = arcade.Text(
            "00:00:00  x1.0",
            20,
            h - 66,
            TEXT_TERTIARY,
            12,
            font_name=FONT_BODY,
            anchor_x="left",
            anchor_y="top",
        )

        self._weather = WeatherPanel()
        self._leaderboard = LeaderboardPanel(
            x=w - LEADERBOARD_RIGHT_MARGIN,
            top_y=h - 20,
            width=LEADERBOARD_WIDTH,
        )
        # Race-events HUD card (Safety Car / VSC / Yellow / Red flag).  Sits
        # right under the leaderboard; the panel hides itself unless the
        # current lap's TrackStatus is non-clear.
        self._race_events = RaceEventsPanel(
            x=w - LEADERBOARD_RIGHT_MARGIN,
            top_y=h - 220,
            width=LEADERBOARD_WIDTH,
        )
        # One table with a column per driver rather than a card each. Two cards
        # repeated the same six labels under two headers and took 354 px of the
        # left column for twelve values (#1102).
        followed = [(driver_main, self._color_for(driver_main))]
        if driver_rival:
            followed.append((driver_rival, self._color_for(driver_rival)))
        self._driver_info = DriverInfoPanel(
            x=20,
            top_y=h - 200,
            width=DRIVER_BOX_WIDTH,
            height=DRIVER_BOX_HEIGHT,
            drivers=followed,
        )

        self._progress_bar = ProgressBar(
            total_frames=session_data.total_frames,
            total_laps=session_data.max_lap_number,
            events=session_data.events,
            left_margin=MARGIN_LEFT,
            right_margin=MARGIN_RIGHT,
        )
        self._progress_bar.on_resize(w)
        self._controls_legend = ControlsLegend()

        self._car_label_main = arcade.Text(
            driver_main,
            0,
            0,
            self._color_for(driver_main),
            CAR_LABEL_FONT_SIZE,
            bold=True,
            font_name=FONT_BODY,
            anchor_x="center",
            anchor_y="bottom",
        )
        self._car_label_rival = arcade.Text(
            driver_rival or "",
            0,
            0,
            self._color_for(driver_rival) if driver_rival else TEXT_SECONDARY,
            CAR_LABEL_FONT_SIZE,
            bold=True,
            font_name=FONT_BODY,
            anchor_x="center",
            anchor_y="top",
        )

        if self._strategy_enabled:
            self._init_strategy_layer()

        logger.info(
            "F1ArcadeView ready: %s vs %s, %d drivers, %d frames, strategy=%s",
            driver_main,
            driver_rival,
            len(session_data.frames_by_driver),
            session_data.total_frames,
            self._strategy_enabled,
        )

    def _init_strategy_layer(self) -> None:
        """Start the local strategy driver, the TCP broadcast server and
        the PITWALL subprocess.

        The strategy UI lives entirely in that subprocess: the arcade replay
        keeps the track, leaderboard and car animations (the replay-first
        concerns) and broadcasts merged arcade+strategy state over TCP so
        PITWALL can render the orchestrator card, the six sub-agent cards,
        the charts and the DATA window. It is spawned last so a slow UI boot
        never delays the replay window."""
        from src.arcade.strategy import SimConnector, SimulateRequestDTO, StrategyState
        from src.arcade.stream import TelemetryStreamServer

        gp_name = self._resolve_gp_name()
        # Provider defaults to OpenAI (what the agents load with
        # ``F1_LLM_PROVIDER=openai``, ChatOpenAI model=gpt-4.1-mini for
        # N25-N30 and the orchestrator model for N31). ``F1_LLM_PROVIDER``
        # env wins so a user running LM Studio locally (set it to
        # "lmstudio") keeps working without a code edit.
        provider = os.environ.get("F1_LLM_PROVIDER") or "openai"
        request = SimulateRequestDTO(
            year=self._year,
            gp=gp_name,
            driver=self._driver_main,
            team=self._team or "",
            driver2=self._driver_rival,
            risk_tolerance=0.5,
            no_llm=False,
            provider=provider,
            interval_s=0.0,
        )
        self._strategy_state = StrategyState()
        # Pass the lap provider so SimConnector blocks at each lap until the
        # arcade replay catches up, so pausing the visor with SPACE in V2 now
        # also pauses the agentic flow instead of letting it storm ahead
        # through V3, V4, V5 …
        self._strategy_connector = SimConnector(
            request=request,
            state=self._strategy_state,
            current_lap_provider=self._current_arcade_lap,
        )
        self._strategy_connector.start()

        try:
            self._stream_server = TelemetryStreamServer(host=STREAM_HOST, port=STREAM_PORT)
            self._stream_server.start()
        except OSError as exc:
            logger.warning("Stream server failed to bind %s:%d (%s)", STREAM_HOST, STREAM_PORT, exc)
            self._stream_server = None

        self._spawn_pitwall()
        # **A window CLOSE never reaches `on_hide_view`, and that is a fact about
        # the toolkit rather than about this class.** In the installed arcade,
        # `on_hide_view` is invoked from exactly two places: `Window.show_view`,
        # which hides the previous view before showing the next, and the explicit
        # `Window.hide_view`. `Window.close` calls neither - it sets `closed`,
        # delegates to pyglet and unschedules the clock. So the teardown ran when
        # the user navigated back to the MENU and never when they closed the
        # window, which is the ordinary way to end a session, and PITWALL was
        # left running against a broadcast that had stopped (#947).
        #
        # Registered here rather than in `__init__` because there is nothing to
        # tear down without `--strategy`, and kept as an attribute so it can be
        # POPPED again: a handler pushed onto the window outlives the view, and
        # a stale one would fire this view's teardown after the user had moved on.
        self._close_handler = {"on_close": self._on_window_close}
        self.window.push_handlers(**self._close_handler)

    def _on_window_close(self) -> None:
        """Route a window close through the one teardown, then let pyglet close.

        Returning None (rather than `True`) leaves the default handler in place,
        so the window still closes. `on_hide_view` is idempotent by construction -
        `_terminate` takes `None` and every field it clears is set to `None` - so
        the menu -> viewer -> menu -> close path running it twice is harmless.
        """
        self.on_hide_view()

    def _spawn_pitwall(self) -> None:
        """Launch the PITWALL windows as a child process.

        The only companion window there is. It ran alongside the Qt dashboard
        so every PITWALL panel could be compared against the window it
        replaces while that window still existed. The Qt one has since been
        retired, and the captures that baseline stands on are
        committed under `documents/dev_docs/migration/pitwall/` rather than
        living in a session scratchpad, precisely so retiring it destroys
        nothing.

        A failed spawn is logged and swallowed: the replay keeps playing
        without its companion. The commonest failure is that the UI
        bundle has not been built, which `src.pitwall.__main__` reports with
        the exact command."""
        try:
            creationflags = subprocess.CREATE_NEW_CONSOLE if os.name == "nt" else 0
            self._pitwall_proc = subprocess.Popen(
                [sys.executable, "-m", "src.pitwall"],
                creationflags=creationflags,
            )
            logger.info("Pitwall subprocess spawned (pid=%s)", self._pitwall_proc.pid)
        except (OSError, ValueError) as exc:
            logger.warning("Pitwall spawn failed (%s) — arcade continues without it", exc)
            self._pitwall_proc = None

    def _resolve_gp_name(self) -> str:
        """Return the GP label fed to the strategy pipeline.

        Prefers the FastF1 Location (``Suzuka``, ``Melbourne``, …) because
        that is what the ``data/raw/<year>/`` folders use. Falls back to
        ``get_gp_names(year)`` (sourced from the canonical per-year
        calendar JSON) and finally to ``GP_TO_LOCATION`` for menu inputs
        that still carry a country-style label from the legacy table."""
        from src.arcade.config import GP_TO_LOCATION, get_gp_names

        if self._session.location:
            return self._session.location
        gp_name = self._session.gp_name or get_gp_names(self._year).get(1, "Sakhir")
        return GP_TO_LOCATION.get(gp_name, gp_name)

    def on_hide_view(self) -> None:
        """Tear down the strategy driver, the stream server and both companion windows."""
        if self._strategy_connector is not None:
            self._strategy_connector.stop()
        if self._stream_server is not None:
            self._stream_server.stop()
            self._stream_server = None
        # Through the helper rather than inline. `_terminate`'s own docstring
        # says a second copy of its block would be the twin that stops getting
        # fixed, and an inline copy WAS that twin until the helper existed.
        # One companion window survives; the helper stays because the
        # reason it exists is about the block, not about the count.
        self._pitwall_proc = self._terminate(self._pitwall_proc, "Pitwall")
        # Pop the close handler with the rest. Leaving it pushed is how the NEXT
        # view's close would run THIS view's teardown - the same class of defect
        # as the one this method exists to fix, one level up.
        if self._close_handler is not None:
            self.window.remove_handlers(**self._close_handler)
            self._close_handler = None

    @staticmethod
    def _terminate(proc: subprocess.Popen | None, name: str) -> None:
        """Stop a companion window process, whatever state it is in.

        Three outcomes: it exits, it does not and gets killed, or it was
        already gone. Every companion window goes through here, which is
        the point: a second copy of this block would be the twin that stops
        getting fixed, and for one release the Qt dashboard's teardown was
        exactly that copy. One window is left and the helper stays anyway -
        the reason it exists is about the block, not about the count.
        """
        if proc is None:
            return None
        try:
            proc.terminate()
            proc.wait(timeout=3.0)
        except subprocess.TimeoutExpired:
            logger.warning("%s did not exit in 3s — killing", name)
            proc.kill()
        except OSError as exc:
            # terminate()/wait() on an already-dead or inaccessible process
            # raise OSError subclasses; nothing else is documented for these.
            logger.warning("%s teardown error: %s", name, exc)
        return None

    # --- Arcade event loop -----------------------------------------------

    def on_update(self, delta_time: float) -> None:
        seek_rate = SEEK_RATE_MULTIPLIER * max(1.0, self.playback_speed)
        max_f = float(self._session.total_frames - 1)

        if self._is_rewinding:
            self._frame_index = max(0.0, self._frame_index - delta_time * FPS * seek_rate)
        elif self._is_forwarding:
            self._frame_index = min(max_f, self._frame_index + delta_time * FPS * seek_rate)

        if not self._is_paused:
            self._frame_index += delta_time * FPS * self.playback_speed
            self._frame_index = max(0.0, min(max_f, self._frame_index))

        # Drive the race-events HUD fade animation off the same delta the rest
        # of the panels see.  Lap is read from the main driver's current frame
        # so the status follows whichever driver the user is following.
        self._race_events.update(delta_time, self._current_track_status())

        self._broadcast_if_due()

    def _current_arcade_lap(self) -> int:
        """Return the lap number the user is currently watching (main driver).

        Used by the race-events HUD card and as the playback gate that
        keeps the strategy SimConnector in sync with what is on screen.
        Falls back to 0 when the main driver has no frames yet (very
        first frame after view construction).
        """
        frames = self._session.frames_by_driver.get(self._driver_main)
        if not frames:
            return 0
        idx = max(0, min(int(self._frame_index), len(frames) - 1))
        return int(getattr(frames[idx], "lap", 1) or 1)

    def _current_track_status(self) -> str:
        """Return the FastF1 ``TrackStatus`` digit string for the active lap."""
        return self._session.track_status_by_lap.get(self._current_arcade_lap(), "")

    def _broadcast_if_due(self) -> None:
        """Throttle the TCP broadcast to ~10 Hz regardless of arcade FPS."""
        if self._stream_server is None or self._strategy_state is None:
            return
        self._broadcast_tick = (self._broadcast_tick + 1) % STREAM_BROADCAST_EVERY_N_FRAMES
        if self._broadcast_tick != 0:
            return
        frame_idx = int(self._frame_index)
        span_start, rewound, dropped = _telemetry_span_bounds(
            self._last_broadcast_idx,
            frame_idx,
            STREAM_MAX_SPAN_FRAMES,
            moved_back=self._frame_index < self._last_broadcast_clock,
        )
        self._last_broadcast_clock = self._frame_index
        # Advance the marker before the no-subscriber return, so the span
        # always covers one tick of playback and never the backlog since
        # whenever a client last happened to be attached.
        self._last_broadcast_idx = frame_idx
        if self._stream_server.client_count() == 0:
            return  # no subscriber, skip the serialisation cost
        self._broadcast_seq += 1
        # Everything the payload needs from the render thread is read HERE and
        # passed by value, because the closure below runs later and elsewhere.
        # `self.playback_speed` read inside it would be whatever the user had
        # selected by the time the sender got round to the job, not the speed
        # this tick was produced at, and `self._frame_index` would be a float
        # that has moved on.
        seq = self._broadcast_seq
        speed = self.playback_speed
        paused = self._is_paused
        total_frames = self._session.total_frames
        strategy_state = self._strategy_state

        def build_payload() -> dict:
            """Assemble one tick. Runs on the stream server's sender thread.

            Safe off the pyglet thread because of what it reads: `SessionData`
            and `RaceGapCalculator` are immutable once constructed, the palette
            is a dict lookup, `_rank_drivers` is a static function of its
            arguments, and `snapshot_dict` takes `StrategyState`'s own lock.
            Everything else is a local captured above.

            It is a closure rather than a payload because building it is the
            expensive part: 3.19 ms on a steady 8x tick and 5.41 ms on a seek,
            measured on the real replay, against a 16.7 ms frame budget."""
            return {
                "schema_version": STREAM_SCHEMA_VERSION,
                "seq": seq,
                "arcade": self._build_arcade_snapshot(frame_idx, span_start, rewound, dropped),
                "strategy": strategy_state.snapshot_dict(STREAM_HISTORY_TAIL),
                "playback": {
                    "speed": speed,
                    "paused": paused,
                    "frame_index": frame_idx,
                    "total_frames": total_frames,
                },
            }

        self._stream_server.broadcast(build_payload)

    def _build_arcade_snapshot(
        self, frame_idx: int, span_start: int, rewound: bool, dropped: int = 0
    ) -> dict:
        """Compact version of the per-frame dict a stream consumer needs.

        Lighter than the internal `_build_frame_dict` consumed by the
        panels: `throttle` and `brake` stay out of the 20-car block
        because only the two featured cars chart them. They ride in
        `telemetry` instead.

        `active` and `rel_dist` are here on purpose and are NOT
        cosmetic. Without `active` a retired car is indistinguishable
        from a running one: `np.interp` clamps past a driver's last
        sample, so a lap-1 DNF keeps broadcasting its crash-site `dist`,
        `speed` and `lap` for the rest of the race. Without `rel_dist`
        there is no way to place a car around the current lap.
        `dist % circuit_length_m` is not a substitute, because `dist`
        accumulates each lap as actually driven (an in-lap and an
        out-lap are neither the same length as each other nor as the
        fastest lap `circuit_length_m` is measured from), so the
        residual drifts across a race and drifts most for the cars that
        pitted."""
        drivers: dict[str, dict] = {}
        for code, frames in self._session.frames_by_driver.items():
            if not frames or frame_idx >= len(frames):
                continue
            f = frames[frame_idx]
            # A car whose telemetry never places it has NO fraction of the lap,
            # and the loader cannot say so in the value: it derives `rel_dist`
            # from a distance that never advances, which comes out as a finite
            # 0.0 - "at the line", a position a real car can hold. On Melbourne
            # 2025 that is HAD on all 154,173 frames, 2,935 of them `active`
            # (#856). Saying it here rather than in the loader keeps the pickle
            # format untouched: a CACHE_VERSION bump would cost every user a
            # full reload of every GP to express something the wire expresses
            # for free.
            has_position = bool(self._session.has_position.get(code, True))
            lap_fraction = _lap_fraction(f.rel_dist) if has_position else None
            drivers[code] = {
                "lap": f.lap,
                "dist": round(f.dist, 1),
                "rel_dist": None if lap_fraction is None else round(lap_fraction, 4),
                "speed": round(f.speed, 1),
                "compound": f.tyre,
                "tyre_life": round(f.tyre_life, 1),
                "active": bool(f.active),
                # False when this driver's telemetry never places the car, so a
                # consumer says "no position data" instead of drawing an empty
                # chart under a populated header.
                "has_position": has_position,
            }
        # --- Race order and the reveal coordinates (#857) -------------------
        # Published by the producer so no consumer re-derives the order from
        # `dist` (race-cumulative: the wrong leader on 37% of sampled frames)
        # or from `lap` (a rounded interpolation that flickers +-1 at the
        # line). `_rank_drivers` is the SAME code the arcade leaderboard ranks
        # with, so the wire and the panel cannot drift apart.
        #
        # Per driver:
        # - `laps_completed` carries the DATA window's strict per-driver
        #   reveal (reveal lap L iff L <= laps_completed). It reads the
        #   crossing map, so it is monotone while playing forward - swept
        #   over 20 drivers x 154,173 frames, no counter-example. It is NOT
        #   exact against the parquet: 76 of 921 crossings (8.3%) open
        #   before the parquet's `Time` by more than HALF A FRAME (0.02 s),
        #   worst case 0.463 s, because the `lap` field these crossings are
        #   detected from is an interpolation rather than a line detector.
        #   The half-frame threshold is the convention, and stating it is the
        #   point: strictly greater than zero the count is 110 (11.9%) and
        #   beyond a full 40 ms frame it is 49 (5.3%). Sub-half-frame is
        #   rounding noise, but a reader cannot know that from the number.
        #   Monotone and per driver, not frame-accurate.
        # - `progress` is the ordering coordinate, laps plus fraction of the
        #   current lap; a consumer derives laps-down positionally from it.
        #   None when the telemetry never places the car (#886).
        # - `has_finished` separates a chequered flag from a retirement:
        #   `active` alone reads the winner as OUT (#855). Its value is only
        #   as good as the flag anchor, which since #879 is the official
        #   classification rather than an inference.
        ranked = LeaderboardPanel._rank_drivers({"drivers": drivers}, self._gaps, frame_idx)
        race_order: list[str] = []
        for code, data, progress in ranked:
            race_order.append(code)
            known = progress is not None and data["has_position"]
            data["progress"] = round(progress, 4) if known else None
            data["laps_completed"] = self._gaps.laps_completed(code, frame_idx)
            data["has_finished"] = self._gaps.has_finished(code)
        main_frame = None
        main_frames = self._session.frames_by_driver.get(self._driver_main)
        if main_frames and frame_idx < len(main_frames):
            main_frame = main_frames[frame_idx]
        # A telemetry SPAN per driver, oldest first, not the single current
        # point: `{drivers: {CODE: [...]}}`.
        #
        # The clock advances `delta_time * FPS * speed` indices per second
        # over 25 Hz data while the broadcast fires at ~10 Hz, so sending one
        # point per tick discarded 60% of the trace at 1x and 95% at 8x: a
        # speed trace went from a point every 8 metres to one every 170. The
        # producer already holds the whole array, so the span costs a slice
        # and no disk read.
        #
        # **Keyed off `drivers`, not off `frames_by_driver`** (#1048). It used
        # to carry two spans under the ROLE keys `main` and `rival`, chosen
        # once at launch, so a consumer could only ever chart those two cars.
        # Publishing all twenty is what lets PITWALL pin any row without a
        # control channel back to the producer, which is a decision this
        # project does not want to take. Iterating the block built above
        # rather than the session makes these keys identical to `drivers`,
        # `driver_colors` and `race_order` by construction: four per-driver
        # maps on one payload is already three chances to drift, and it does
        # not need a fourth rule.
        #
        # **A retired car still carries a span**, because its frame array runs
        # the full length of the race and the values stop changing. On
        # Melbourne 2025 that is three lap-1 retirements, 14.5% of the arcade
        # block on a seek tick and about 13.8% of the whole message once the
        # strategy block rides along. What each of them costs per tick moves with
        # playback: a span is `FPS x speed / 10 Hz` samples, so 2-3 at 1x and 20
        # at 8x, not a fixed handful. They stay: `active` and `has_position` are published beside
        # them for exactly this, and a key set that came and went as cars
        # retired would be a second rule again. **A consumer gates a span on
        # `drivers[code].active` and `.has_position`** rather than on whether
        # the span is empty.
        circuit_length = float(self._session.circuit_length_m or 0.0)
        positions = self._session.has_position
        telemetry = {
            # `has_position` rides into each span so it says the same thing the
            # drivers block does. Without it the same car reads "no position"
            # in one half of the payload and "at the line" in the other, and
            # the telemetry window keys every sample into distance bucket 0.
            "drivers": {
                code: _frames_to_telemetry_span(
                    self._session.frames_by_driver[code],
                    span_start,
                    frame_idx,
                    circuit_length,
                    bool(positions.get(code, True)),
                )
                for code in drivers
            },
            # True when the user seeked backwards. The span is empty and the
            # consumer must drop what it has: a buffer keyed on
            # distance-within-lap holds samples for track the car has not
            # re-driven yet, and nothing else would ever evict them.
            "rewound": rewound,
            # Frames the clock crossed that this tick could NOT carry,
            # because a forward jump (a progress-bar click) outran the span
            # cap. Zero on every normal tick. Without it a forward seek is
            # invisible: the sequence stays contiguous and the clock still
            # runs forwards, so a consumer appending samples would splice
            # two unrelated parts of the race into one trace.
            "dropped": dropped,
        }
        track_status = (
            self._session.track_status_by_lap.get(int(main_frame.lap), "") if main_frame else ""
        )
        status_label = track_status_label(track_status)
        return {
            "gp_name": self._session.gp_name,
            # FastF1's authoritative Location. `gp_name` is whatever the
            # caller resolved through `get_gp_names(year)`, which normally
            # reads the same canonical calendar and agrees with this to the
            # letter. The two diverge on ONE path: when
            # `data/tire_compounds_by_race.json` is missing or lacks the year,
            # `get_gp_names` falls back to a hardcoded 2024 table, and 2025
            # round 3 comes back "Australia" when it is Suzuka. That fallback
            # is also what names the session pickle, so a wrong `gp_name`
            # mislabels the cache as well as the header - which is exactly why
            # a consumer resolving `data/raw/<year>/<gp>/` must read this
            # field and not the label on screen.
            "location": self._session.location,
            "year": self._year,
            "lap": main_frame.lap if main_frame else 1,
            "t": main_frame.t if main_frame else 0.0,
            # Session-time origin of `t`: `t + global_t_min` is FastF1
            # SessionTime seconds, the clock laps/weather/intervals parquet
            # are keyed on. `t` alone is only `frame_index * DT`.
            "global_t_min": round(float(self._session.global_t_min), 3),
            "total_laps": self._session.max_lap_number,
            "race_order": race_order,
            # The tower colours its rows by driver, and a hardcoded palette in
            # TypeScript is exactly the drift `palette.py` exists to prevent -
            # five copies of this palette have already been found. The producer
            # publishes what the arcade itself draws with.
            "driver_colors": {code: list(self._color_for(code)) for code in drivers},
            # FastF1 TrackStatus digits for the lap on screen, the same source
            # the arcade's own pill reads. "" when the loader has no entry for
            # that lap, which a consumer renders as clear - the same collapse
            # the arcade already makes.
            "track_status": track_status,
            # And the same digits DECODED, for the same reason `driver_colors`
            # is here: the priority order (red > SC > VSC > yellow) and the
            # four labels are a project rule, and a consumer decoding the
            # digits itself would be a second copy of it in another language.
            # `None` when the loader has no entry, which is NOT the same as a
            # green track and must not render as one.
            "track_status_label": status_label[0] if status_label else None,
            "track_status_color": list(status_label[1]) if status_label else None,
            # Circuit length lets the telemetry window anchor the X axis
            # once and forget, because without it the charts would autorange to
            # the current sample's max and shift every broadcast.
            "circuit_length_m": round(self._session.circuit_length_m or 0.0, 1),
            "driver_main": self._driver_main,
            "driver_rival": self._driver_rival,
            "drivers": drivers,
            "telemetry": telemetry,
        }

    def on_draw(self) -> None:
        self.clear()
        self._track.draw(show_drs=self._show_drs_zones)
        frame_idx = int(self._frame_index)
        frame = self._build_frame_dict(frame_idx)

        # Draw the 18 non-featured cars first as small dimmed dots so the
        # featured main/rival dots paint on top and always read clearly.
        # Toggled by the ``A`` key (``self._show_all_cars``).
        if self._show_all_cars:
            self._draw_background_cars(frame_idx)

        self._draw_car(self._driver_main, self._car_label_main, above=True)
        if self._driver_rival:
            self._draw_car(self._driver_rival, self._car_label_rival, above=False)

        self._leaderboard.draw(
            frame, self._session.driver_colors, self._gaps, frame_idx, self._selected_drivers
        )
        # Anchor the race-events pill right under the leaderboard's bottom
        # edge. The leaderboard's row count varies per session, so
        # ``bottom_y`` (set during draw above) is read instead of hard-coding an offset.
        self._race_events.set_top(self._leaderboard.bottom_y - RaceEventsPanel.GAP_FROM_LEADERBOARD)
        self._race_events.draw()
        self._weather.draw(frame, self.window.height)
        sorted_progress = self._leaderboard.sorted_progress(frame, self._gaps, frame_idx)
        # DRIVER_BOX_GAP separates the weather card from the driver table. The
        # two are 14 px apart on screen rather than 32, because
        # `WeatherPanel.bottom_y` is computed from the last row's baseline and
        # sits 18 px above the card's own edge.
        self._driver_info.set_top(self._weather.bottom_y - DRIVER_BOX_GAP)
        self._driver_info.draw(frame, sorted_progress, self._gaps, frame_idx)

        if self._show_progress_bar:
            self._progress_bar.draw(self.window.width, frame_idx)
        # The legend decides between the full list and a one-line hint from the
        # room the column above it actually left. It used to draw over the rival
        # card at the default 720, where two stacked cards left 146 px against a
        # list that needs 158 (#1096). One table leaves 263 there, so the full
        # list fits, and the collapse now fires below about 615 px of window
        # instead of below 788. Measured from the panel rather than assumed,
        # because the room still depends on the window's height.
        self._controls_legend.draw(
            space_below=self._driver_info.bottom_y - self._controls_legend.bottom
        )
        self._update_hud(frame)

    def on_key_press(self, symbol: int, modifiers: int) -> None:
        if symbol == arcade.key.C:
            # Force-open overrides the room check, so a user who asks for the
            # list gets it even where it overlaps: they summoned it and the same
            # key dismisses it.
            self._controls_legend.toggle()
        elif symbol == arcade.key.ESCAPE:
            self.window.close()
        elif symbol == arcade.key.SPACE:
            self._is_paused = not self._is_paused
        elif symbol == arcade.key.LEFT:
            self._was_paused_before_hold = self._is_paused
            self._is_rewinding = True
            self._is_paused = True
        elif symbol == arcade.key.RIGHT:
            self._was_paused_before_hold = self._is_paused
            self._is_forwarding = True
            self._is_paused = True
        elif symbol == arcade.key.UP:
            self._speed_idx = min(len(PLAYBACK_SPEEDS) - 1, self._speed_idx + 1)
        elif symbol == arcade.key.DOWN:
            self._speed_idx = max(0, self._speed_idx - 1)
        elif symbol == arcade.key.KEY_1:
            self._speed_idx = PLAYBACK_SPEEDS.index(0.5)
        elif symbol == arcade.key.KEY_2:
            self._speed_idx = PLAYBACK_SPEEDS.index(1.0)
        elif symbol == arcade.key.KEY_3:
            self._speed_idx = PLAYBACK_SPEEDS.index(2.0)
        elif symbol == arcade.key.KEY_4:
            self._speed_idx = PLAYBACK_SPEEDS.index(4.0)
        elif symbol == arcade.key.R:
            self._frame_index = 0.0
            self._speed_idx = DEFAULT_SPEED_IDX
            self._is_paused = False
        elif symbol == arcade.key.D:
            self._show_drs_zones = not self._show_drs_zones
        elif symbol == arcade.key.B:
            self._show_progress_bar = not self._show_progress_bar
        elif symbol == arcade.key.A:
            self._show_all_cars = not self._show_all_cars

    def on_key_release(self, symbol: int, modifiers: int) -> None:
        if symbol == arcade.key.LEFT:
            self._is_rewinding = False
            self._is_paused = self._was_paused_before_hold
        elif symbol == arcade.key.RIGHT:
            self._is_forwarding = False
            self._is_paused = self._was_paused_before_hold

    def on_mouse_press(self, x: int, y: int, button: int, modifiers: int) -> None:
        seek = self._progress_bar.on_mouse_press(x, y)
        if seek is not None:
            self._frame_index = float(seek)
            return
        code = self._leaderboard.hit_test(x, y)
        if code is None:
            return
        if modifiers & arcade.key.MOD_SHIFT:
            self._selected_drivers ^= {code}
        else:
            self._selected_drivers = {code}

    def on_resize(self, width: float, height: float) -> None:
        self._track.update_scaling(
            int(width),
            int(height),
            margin_left=MARGIN_LEFT,
            margin_right=MARGIN_RIGHT,
            margin_bottom=MARGIN_BOTTOM,
            margin_top=MARGIN_TOP,
        )
        self._leaderboard.x = int(width) - LEADERBOARD_RIGHT_MARGIN
        self._leaderboard.set_top(int(height) - 20)
        # The race-events pill rides the leaderboard's right edge; on_draw
        # re-anchors top_y from leaderboard.bottom_y but the x must follow
        # window resizes the same way the leaderboard does.
        self._race_events.x = self._leaderboard.x
        self._lap_label.y = int(height) - 20
        self._lap_text.y = int(height) - 36
        self._time_text.y = int(height) - 66
        self._progress_bar.on_resize(int(width))

    # --- Helpers ---------------------------------------------------------

    @property
    def playback_speed(self) -> float:
        return PLAYBACK_SPEEDS[self._speed_idx]

    def _color_for(self, code: str | None) -> tuple[int, int, int]:
        if not code:
            return TEXT_SECONDARY
        return self._session.driver_colors.get(code, TEXT_PRIMARY)

    def _build_frame_dict(self, frame_idx: int) -> dict:
        drivers_dict: dict[str, dict] = {}
        main_frame: FrameData | None = None
        for code, frames in self._session.frames_by_driver.items():
            if not frames or frame_idx >= len(frames):
                continue
            f = frames[frame_idx]
            drivers_dict[code] = {
                "x": f.x,
                "y": f.y,
                "speed": f.speed,
                "gear": f.gear,
                "drs": f.drs,
                "throttle": f.throttle,
                "brake": f.brake,
                "lap": f.lap,
                "dist": f.dist,
                "rel_dist": f.rel_dist,
                "tyre": f.tyre,
                "tyre_life": f.tyre_life,
                "active": f.active,
            }
            if code == self._driver_main:
                main_frame = f

        lap = main_frame.lap if main_frame else 1
        return {
            "lap": lap,
            "t": main_frame.t if main_frame else 0.0,
            "drivers": drivers_dict,
            # Real per-lap FastF1 weather (#616), built once at session load
            # by SessionLoader._extract_weather_by_lap and cached on
            # SessionData.weather_by_lap. An older cache or a session with no
            # weather data resolves to {}, and WeatherPanel then renders "N/A"
            # per field. A single NaN sample degrades the same way, because the
            # loader stores it as None under the key and overlays._reading
            # coalesces it there (#1087) rather than formatting it and raising.
            "weather": self._session.weather_by_lap.get(lap, {}),
        }

    def _draw_background_cars(self, frame_idx: int) -> None:
        """Render every non-featured driver as a small dimmed dot.

        Skips the main and rival codes (they draw later with the full
        radius + label + outline, so they always sit on top of the
        field). Small cars are unlabeled: 20 labels at once would turn
        the track into a tag cloud. Alpha is applied so the featured
        dots still dominate visually."""
        featured = {self._driver_main}
        if self._driver_rival:
            featured.add(self._driver_rival)
        for code, frames in self._session.frames_by_driver.items():
            if code in featured or not frames or frame_idx >= len(frames):
                continue
            f = frames[frame_idx]
            # A car FastF1 gave no position for has x = y = 0 and would be
            # drawn at one fixed point on the circuit for the whole race
            # (HAD, Melbourne 2025: 2,935 active frames there). #886 fixed
            # the ORDERING to say "unknown"; the drawing kept the sentinel.
            if not f.active or not self._session.has_position.get(code, True):
                continue
            sx, sy = self._track.project(f.x, f.y)
            r, g, b = self._color_for(code)
            arcade.draw_circle_filled(sx, sy, CAR_BG_RADIUS, (r, g, b, CAR_BG_ALPHA))

    def _draw_car(self, code: str, label: arcade.Text, above: bool) -> None:
        frames = self._session.frames_by_driver.get(code)
        if not frames:
            return
        idx = int(self._frame_index)
        if idx >= len(frames):
            return
        f = frames[idx]
        # Same guard as the background field: no position data means there is
        # nowhere honest to draw this car, and (0, 0) projects to a real point
        # on the circuit rather than to nothing (#886).
        if not f.active or not self._session.has_position.get(code, True):
            return
        sx, sy = self._track.project(f.x, f.y)
        color = self._color_for(code)
        arcade.draw_circle_filled(sx, sy, CAR_RADIUS, color)
        arcade.draw_circle_outline(sx, sy, CAR_RADIUS, CAR_BORDER_COLOR, CAR_BORDER_WIDTH)
        # Main driver label sits above the dot, rival below, so they never
        # overlap when the two cars are side by side.
        label.x = sx
        label.y = sy + CAR_RADIUS + 4 if above else sy - CAR_RADIUS - 4
        label.draw()

    def _update_hud(self, frame: dict) -> None:
        lap = frame.get("lap", 1)
        total = self._session.max_lap_number
        self._lap_text.text = f"{lap}/{total}"
        t = frame.get("t", 0.0)
        hh = int(t // 3600)
        mm = int((t % 3600) // 60)
        ss = int(t % 60)
        paused = "  PAUSED" if self._is_paused else ""
        self._time_text.text = f"{hh:02d}:{mm:02d}:{ss:02d}  x{self.playback_speed}{paused}"
        self._lap_label.draw()
        self._lap_text.draw()
        self._time_text.draw()
