"""The telemetry span the arcade broadcast puts on the wire (#841).

The producer sends every sample the replay clock crossed since the previous
tick, not the single current point. What makes that correct is arithmetic
over two integers, so it is tested as arithmetic: the tests below simulate
the playback clock at each speed and assert the union of the spans is the
frames the clock actually crossed, once each.

The three cases that used to be accidents rather than branches -- pause,
backwards seek, and a stalled process -- get a test apiece, because each
one produced a specific wrong behaviour: repeated samples, a negative
slice, and an unbounded payload.
"""

from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("arcade", reason="the arcade replay is an optional surface")

from src.arcade.app import (  # noqa: E402
    F1ArcadeView,
    _frames_to_telemetry_span,
    _telemetry_span_bounds,
)
from src.arcade.config import (  # noqa: E402
    DRS_OPEN_CODES,
    DT,
    FPS,
    PLAYBACK_SPEEDS,
    STREAM_BROADCAST_EVERY_N_FRAMES,
    STREAM_MAX_SPAN_FRAMES,
)
from src.arcade.data import FrameData, SessionData  # noqa: E402
from src.arcade.gaps import RaceGapCalculator  # noqa: E402

# The real cadence: `on_update` runs at the arcade library default 60 Hz and
# every 6th call broadcasts, so a tick is one sixth of a second of wall
# clock. `_broadcast_if_due` counts on_update calls, not replay frames,
# which is why the loss used to grow with playback speed.
ON_UPDATE_HZ = 60.0
TICK_SECONDS = STREAM_BROADCAST_EVERY_N_FRAMES / ON_UPDATE_HZ
CIRCUIT_LENGTH_M = 5278.0


def _collect_spans(speeds_and_ticks, start_frame: float = 0.0):
    """Run the real clock and bounds arithmetic, returning the frames sent.

    Yields `(span_start, frame_idx)` per tick alongside the running list of
    every frame index put on the wire, so a test can assert on either.
    """
    frame_index = start_frame
    last_sent = -1
    sent: list[int] = []
    spans: list[tuple[int, int]] = []
    for speed, n_ticks in speeds_and_ticks:
        for _ in range(n_ticks):
            frame_index += TICK_SECONDS * FPS * speed
            frame_idx = int(frame_index)
            span_start, rewound, _ = _telemetry_span_bounds(
                last_sent, frame_idx, STREAM_MAX_SPAN_FRAMES
            )
            last_sent = frame_idx
            spans.append((span_start, frame_idx))
            if not rewound:
                sent.extend(range(span_start, frame_idx + 1))
    return sent, spans


# --- The acceptance criterion: every frame once, at every speed -------------


@pytest.mark.parametrize("speed", PLAYBACK_SPEEDS)
def test_every_frame_the_clock_crosses_is_sent_exactly_once(speed):
    """No gaps and no duplicates, at 0.25x through 8x.

    Before the span this failed in both directions: 60 % of frames never
    left the process at 1x and 95 % at 8x, while at 0.25x the same frame
    was re-sent because the broadcast outran the clock.
    """
    sent, _ = _collect_spans([(speed, 60)])

    assert sent == sorted(sent), "samples must arrive oldest first"
    assert len(sent) == len(set(sent)), "a sample was sent twice"
    assert sent == list(range(sent[0], sent[-1] + 1)), "a frame was skipped"


def test_the_sample_count_matches_the_frames_the_clock_advanced():
    """At 1x, one second of playback must deliver 25 samples, not 10."""
    sent, _ = _collect_spans([(1.0, round(1.0 / TICK_SECONDS))])

    assert len(sent) == pytest.approx(FPS, abs=1)


def test_continuity_survives_a_speed_change():
    """Changing speed mid-race must not tear a hole in the trace."""
    sent, _ = _collect_spans([(1.0, 20), (8.0, 20), (0.5, 20)])

    assert sent == list(range(sent[0], sent[-1] + 1))
    assert len(sent) == len(set(sent))


# --- The three branches that used to be accidents ---------------------------


def test_pause_sends_no_new_samples_rather_than_repeating_the_last_one():
    """A paused clock does not advance, so there is nothing new to send.

    `_broadcast_if_due` keeps firing while paused (the throttle sits
    outside the pause gate), so without an explicit empty span the same
    sample went out ten times a second and any consumer that appends
    accumulated duplicates.
    """
    span_start, rewound, dropped = _telemetry_span_bounds(400, 400, STREAM_MAX_SPAN_FRAMES)

    assert span_start > 400, "a paused tick must produce an empty span"
    assert rewound is False, "pause is not a discontinuity"
    assert list(range(span_start, 401)) == []


def test_a_backwards_seek_is_empty_and_flagged():
    """Rewind must be a branch, not a negative slice."""
    span_start, rewound, dropped = _telemetry_span_bounds(400, 250, STREAM_MAX_SPAN_FRAMES)

    assert rewound is True
    assert span_start > 250, "nothing may be appended on a rewind"


def test_a_forward_jump_is_capped_and_the_lost_frames_are_counted():
    """A jump smooth playback could not produce is capped and reported.

    Smooth playback tops out near 60 frames per tick. What reaches the cap
    is a click on the progress bar, which sets the index directly; a
    process stall is the rarer second case.
    """
    span_start, rewound, dropped = _telemetry_span_bounds(10, 90_000, STREAM_MAX_SPAN_FRAMES)

    assert rewound is False
    assert 90_000 - span_start + 1 == STREAM_MAX_SPAN_FRAMES
    # The frames the cap could not carry are counted and published, because
    # a forward jump is otherwise invisible: the sequence stays contiguous
    # and the clock still runs forwards.
    assert dropped == 90_000 - 10 - STREAM_MAX_SPAN_FRAMES
    assert _telemetry_span_bounds(10, 40, STREAM_MAX_SPAN_FRAMES)[2] == 0

    # The cap must never resurrect a span that pause emptied.
    paused_start, _, _ = _telemetry_span_bounds(400, 400, STREAM_MAX_SPAN_FRAMES)
    assert paused_start == 401


def test_the_first_tick_sends_one_sample_not_the_whole_race():
    """`last_sent_idx` starts at -1, which must mean "nothing yet", not "from zero"."""
    span_start, _, _ = _telemetry_span_bounds(-1, 0, STREAM_MAX_SPAN_FRAMES)

    assert span_start == 0
    assert list(range(span_start, 1)) == [0]


# --- Packing the span -------------------------------------------------------


def _frames(n: int) -> list[FrameData]:
    return [
        FrameData(
            t=i * DT,
            x=0.0,
            y=0.0,
            speed=200.0 + i,
            gear=6,
            drs=0,
            throttle=80.0,
            brake=0.0,
            lap=1 + i // 10,
            dist=float(i) * 10.0,
            rel_dist=(i % 10) / 10.0,
            tyre=1,
            tyre_life=5.0,
        )
        for i in range(n)
    ]


def test_the_span_is_packed_oldest_first_and_keeps_the_sample_shape():
    packed = _frames_to_telemetry_span(_frames(50), 10, 14, CIRCUIT_LENGTH_M)

    assert len(packed) == 5
    assert [s["t"] for s in packed] == sorted(s["t"] for s in packed)
    assert set(packed[0]) == {
        "lap",
        "t",
        "dist",
        "speed",
        "throttle",
        "brake",
        "gear",
        # The raw FastF1 code AND the decoded answer. PITWALL's DRS lane reads the
        # boolean, because the open set lives in `config.DRS_OPEN_CODES` and the
        # window refuses to fork it into TypeScript; the raw code stays for the
        # track overlay and for anything that needs to tell 8 ("eligible") from 0.
        "drs",
        "drs_open",
    }


def test_drs_open_decodes_the_code_rather_than_publishing_it():
    """The one thing the boolean is for: the consumer never sees a code.

    8 is the case that makes this worth asserting - "eligible, not open", and the
    third-commonest code on the real session. A naive `drs > 0` would publish it as
    open and paint a DRS lane that is wrong for a fifth of the lap.
    """
    from src.arcade.app import _frame_to_telemetry
    from src.arcade.config import DRS_OPEN_CODES

    def packed(code: int) -> dict:
        frame = _frames(1)[0]
        frame.drs = code
        return _frame_to_telemetry(frame, CIRCUIT_LENGTH_M)

    for code in sorted(DRS_OPEN_CODES):
        assert packed(code)["drs_open"] is True, code
    for code in (0, 1, 8):
        assert packed(code)["drs_open"] is False, code
    # And the raw code survives alongside it, so nothing that reads it breaks.
    assert packed(8)["drs"] == 8


def test_the_span_is_clamped_to_the_frames_a_driver_actually_has():
    """A driver whose telemetry is shorter than the global timeline must not raise."""
    assert _frames_to_telemetry_span(_frames(20), 15, 40, CIRCUIT_LENGTH_M) == pytest.approx(
        _frames_to_telemetry_span(_frames(20), 15, 19, CIRCUIT_LENGTH_M)
    )
    assert _frames_to_telemetry_span([], 0, 5, CIRCUIT_LENGTH_M) == []
    assert _frames_to_telemetry_span(None, 0, 5, CIRCUIT_LENGTH_M) == []
    # An empty span is empty, not the whole array.
    assert _frames_to_telemetry_span(_frames(20), 6, 5, CIRCUIT_LENGTH_M) == []


def test_a_span_crossing_a_lap_boundary_carries_both_laps():
    """The consumer clears per sample, so the producer must not hide the crossing."""
    packed = _frames_to_telemetry_span(_frames(50), 7, 13, CIRCUIT_LENGTH_M)

    assert {s["lap"] for s in packed} == {1, 2}


# --- End to end through the snapshot builder --------------------------------


def test_the_snapshot_publishes_spans_and_the_rewind_flag():
    session = SessionData(
        gp_name="Australia",
        location="Melbourne",
        year=2025,
        frames_by_driver={"NOR": _frames(50), "PIA": _frames(50)},
        circuit_length_m=CIRCUIT_LENGTH_M,
        total_frames=50,
    )
    view = SimpleNamespace(
        _session=session,
        _driver_main="NOR",
        _driver_rival="PIA",
        _year=2025,
        _gaps=RaceGapCalculator(session),
        _color_for=lambda code: (255, 255, 255),
    )

    moving = F1ArcadeView._build_arcade_snapshot(view, 20, 16, False)["telemetry"]
    assert len(moving["main"]) == 5
    assert len(moving["rival"]) == 5
    assert moving["rewound"] is False

    paused = F1ArcadeView._build_arcade_snapshot(view, 20, 21, False)["telemetry"]
    assert paused["main"] == [] and paused["rival"] == []

    rewound = F1ArcadeView._build_arcade_snapshot(view, 20, 21, True)["telemetry"]
    assert rewound["main"] == []
    assert rewound["rewound"] is True


def test_single_driver_mode_publishes_an_empty_rival_span():
    session = SessionData(
        gp_name="Australia",
        location="Melbourne",
        year=2025,
        frames_by_driver={"NOR": _frames(50)},
        circuit_length_m=CIRCUIT_LENGTH_M,
        total_frames=50,
    )
    view = SimpleNamespace(
        _session=session,
        _driver_main="NOR",
        _driver_rival=None,
        _year=2025,
        _gaps=RaceGapCalculator(session),
        _color_for=lambda code: (255, 255, 255),
    )

    telemetry = F1ArcadeView._build_arcade_snapshot(view, 20, 16, False)["telemetry"]

    assert telemetry["rival"] == []
    assert len(telemetry["main"]) == 5


def test_the_payload_growth_stays_small_at_the_fastest_speed():
    """Two drivers carry traces, so 8x costs about 20 samples a tick, not 200."""
    frames_per_tick = TICK_SECONDS * FPS * max(PLAYBACK_SPEEDS)

    assert math.ceil(frames_per_tick) <= 25


def test_the_drs_open_set_has_exactly_one_home_in_the_source():
    """No module may carry its own copy of {10, 12, 14}, `config` excepted.

    **This is the guard the single-homing commit did not write, and the commit
    needed it.** `feat(arcade): the wire publishes a decoded drs_open` moved the set
    into `config.DRS_OPEN_CODES`, claimed in its own message that "two subsystems
    decode it now", and left `overlays.py`'s `_drs_label` and `_drs_color` holding a
    literal `(10, 12, 14)` each. Three copies of a set whose entire purpose was to
    have one - the twin that never got the fix, which is this repo's most frequent
    defect and the one an adversarial gate found here again.

    Structural, not textual: the tree is parsed and every set / tuple / list of
    integer constants is compared as a SET, so a reordered `(14, 10, 12)`, a
    `{10, 12, 14}` and a `[10, 12, 14]` all count. A grep for one spelling is what
    lets the next copy through.
    """
    import ast

    # **Anchored to THIS FILE, not to the working directory.** The first version used
    # relative roots, and a second gate pass executed it from `tests/`: `Path("src/arcade")`
    # globbed nothing, the loop iterated over an empty set, and the guard PASSED with a
    # planted copy sitting in the tree. A guard written to catch this repo's dominant defect
    # was itself the empty-set failure - so the count below is asserted first.
    repo = Path(__file__).resolve().parents[2]
    roots = ("src/arcade", "src/pitwall", "src/simulation", "scripts")
    allowed = (repo / "src/arcade/config.py").resolve()
    offenders: list[str] = []
    scanned = 0

    lowest_open = min(DRS_OPEN_CODES)

    for root in roots:
        for path in (repo / root).rglob("*.py"):
            if path.resolve() == allowed:
                continue
            scanned += 1
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                # Form one: the set written out, in any container and any order.
                if isinstance(node, (ast.Set, ast.Tuple, ast.List)):
                    values = [
                        element.value
                        for element in node.elts
                        if isinstance(element, ast.Constant) and isinstance(element.value, int)
                    ]
                    if len(values) == len(node.elts) and set(values) == set(DRS_OPEN_CODES):
                        offenders.append(f"{path.relative_to(repo)}:{node.lineno} (a literal set)")
                    continue
                # **Form two: the THRESHOLD, which the first version walked straight past.**
                # `drs >= 10` is not a copy of the set, it is a DIVERGENT copy: it calls 11
                # and 13 open, and those exist - 401 and 515 frames on Melbourne (#1002) -
                # while `DRS_OPEN_CODES` calls them closed. `scripts/verify_drs_zones.py`
                # carried exactly that, and the census reported the tree clean.
                if not isinstance(node, ast.Compare) or len(node.ops) != 1:
                    continue
                if not isinstance(node.ops[0], (ast.GtE, ast.Gt)):
                    continue
                right = node.comparators[0]
                if not (isinstance(right, ast.Constant) and isinstance(right.value, int)):
                    continue
                if right.value not in (lowest_open, lowest_open - 1):
                    continue
                subject = ast.unparse(node.left).lower()
                if "drs" in subject:
                    offenders.append(f"{path.relative_to(repo)}:{node.lineno} (a threshold)")

    assert scanned > 20, (
        f"the census only visited {scanned} files, so it is asserting about nearly nothing. "
        "The roots are resolved from this file's location; check they still exist."
    )
    assert offenders == [], (
        "a second copy of the DRS open set entered the source: "
        f"{offenders}. Import `DRS_OPEN_CODES` from src.arcade.config instead."
    )


def test_eligible_is_not_open_at_every_site_that_decodes_it():
    """The 8 the open set is defined AGAINST, asserted as the rendered EFFECT.

    Three consumers decode these codes and each could drift alone: the wire's
    `drs_open`, and the driver box's label and colour. Asserting the constant only
    would pass while a consumer compared against the wrong one - so this asserts
    what each one PRODUCES for 8 and for 10.

    FastF1's own channel docs (`fastf1/_api.py`, `car_data`) are the source:
    8 is "Detected, Eligible once in Activation Zone", 10 / 12 / 14 are all On. The
    comment that shipped in `track.py` and moved into `config.py` said value 10 was
    the eligible one, which is wrong in both halves and is an invitation to widen the
    set to include 8 - the exact change that would draw an open wing on a closed one.
    """
    from src.arcade.config import DRS_ELIGIBLE_CODE
    from src.arcade.overlays import DriverInfoPanel

    assert DRS_ELIGIBLE_CODE not in DRS_OPEN_CODES, "eligible is not open"

    assert DriverInfoPanel._drs_label(DRS_ELIGIBLE_CODE) == "AVAIL"
    assert DriverInfoPanel._drs_label(10) == "ON"
    open_colour = DriverInfoPanel._drs_color(10)
    assert DriverInfoPanel._drs_color(DRS_ELIGIBLE_CODE) != open_colour, (
        "an eligible car must not be painted with the open colour"
    )
    for code in DRS_OPEN_CODES:
        assert DriverInfoPanel._drs_label(code) == "ON", f"code {code} is documented as On"
        assert DriverInfoPanel._drs_color(code) == open_colour


# --- #1002: the discrete channels stop being interpolated ----------------------


def _melbourne_or_skip():
    """The cached arcade session, or a skip. The pickle is not in git."""
    from src.arcade.config import ARCADE_CACHE_DIR, CACHE_VERSION

    cached = ARCADE_CACHE_DIR / "Melbourne_2025_race.pkl"
    if not cached.exists():
        pytest.skip("the Melbourne 2025 arcade pickle is not on this install")
    import pickle

    with cached.open("rb") as handle:
        session = pickle.load(handle)
    if session.version != CACHE_VERSION:
        pytest.skip(f"the cached pickle is {session.version}, not {CACHE_VERSION}")
    return session


def _active_frames(session) -> list:
    frames = [
        frame for driver in session.frames_by_driver.values() for frame in driver if frame.active
    ]
    # The set is DISCOVERED, so its size is asserted before anything reads it: an
    # empty list would make every assertion below vacuously true, which is the
    # exact shape of the green-on-nothing guard this repo has already shipped once.
    assert len(frames) > 1_000_000, f"only {len(frames)} active frames"
    return frames


def test_no_served_frame_carries_a_gear_the_car_cannot_select():
    """The EFFECT of #1002, on the frames the arcade actually broadcasts.

    Not "the resampler is nearest-neighbour" - that is the mechanism, and measured,
    the mechanism alone changes this number by four. The F1 live-timing feed itself
    publishes `nGear` of 128 (151 raw samples on this race) and every value between
    10 and 127, and three resampling stages carried it to **967 served frames at
    128 and 1,840 above gear 8**. PITWALL's GEAR lane is locked to [0, 9], so each
    is a full-height spike.

    Gear 0 is asserted PRESENT in the same breath. It is a real reading, and a
    validity predicate written `g < 1 or g > 8` would erase 4,773 frames of a
    stationary car while making the first assertion pass.
    """
    frames = _active_frames(_melbourne_or_skip())
    gears = {frame.gear for frame in frames}
    assert max(gears) <= 8, f"gears above 8 are served: {sorted(g for g in gears if g > 8)}"
    assert min(gears) >= 0
    neutral = sum(1 for frame in frames if frame.gear == 0)
    assert neutral > 1000, f"only {neutral} neutral frames: is 0 being filtered as invalid?"


def test_no_served_frame_carries_a_drs_code_the_feed_never_emits():
    """The raw channel holds {0, 1, 2, 3, 8, 10, 12, 14}; the wire used to hold more.

    Linear interpolation between two real codes manufactures the ones in between,
    and 9, 11 and 13 are read as CLOSED by `DRS_OPEN_CODES` while sitting between
    two open frames - so an open wing drew as a flicker. Measured before the fix:
    **1,775 served frames** on 4, 5, 6, 7, 9, 11 or 13.
    """
    frames = _active_frames(_melbourne_or_skip())
    served = {frame.drs for frame in frames}
    manufactured = served - {0, 1, 2, 3, 8, 10, 12, 14}
    assert not manufactured, f"codes FastF1 never emits are on the wire: {sorted(manufactured)}"
    # The open codes must still be REACHED, or a channel stuck at 0 would pass.
    assert served & DRS_OPEN_CODES, "no open-wing frame survives at all"


def test_the_brake_channel_is_the_boolean_it_was_measured_as():
    """`Brake` is `{'type': 'discrete'}` to FastF1 and False/True in the raw stream.

    It lived in the resampler's CONTINUOUS set and was multiplied by 100, so
    **86,925 served frames (3.49 %) sat strictly between 2 and 98** across 10,976
    distinct values, none of which any car ever published.
    """
    frames = _active_frames(_melbourne_or_skip())
    served = {round(frame.brake, 6) for frame in frames}
    assert served <= {0.0, 100.0}, f"interpolated brake pressures are served: {sorted(served)[:8]}"
    assert served == {0.0, 100.0}, "both states must occur, or the channel is stuck"


def test_a_tyre_age_is_a_whole_number_of_laps():
    """`tyre_life` is a per-lap count with a reset at every stop, not a ramp.

    It was in the continuous set beside `brake`, so it interpolated ACROSS the reset:
    758 served frames carried a fractional age and **25 sat more than half a lap from
    either neighbouring value, the worst 16.4 laps out**. The TimingTower renders this
    number, and a pit exit is exactly where it was wrong.
    """
    frames = _active_frames(_melbourne_or_skip())
    fractional = [f.tyre_life for f in frames if abs(f.tyre_life - round(f.tyre_life)) > 1e-9]
    assert not fractional, f"{len(fractional)} frames carry a fractional tyre age"


# --- the same rules on a hand-built driver, so CI can see them ------------------
#
# Every guard above reads the cached 382 MB pickle, which no CI runner has, so all of
# them SKIP there. Worse, on a machine that HAS a current pickle they pass against the
# artefact rather than the code: reverting `_resample_driver` to `np.interp` leaves the
# already-built v13 frames untouched and every one of them stays green. The tests below
# run the resampler itself on four samples and are the ones that go red for that.


def _one_driver_two_samples() -> dict:
    """Two raw samples a second apart, chosen so linear interpolation is VISIBLE.

    At the midpoint every discrete channel would take a value that is not in the raw
    array: gear 5 between 2 and 8, DRS 6 between 0 and 12 (a code FastF1 never emits),
    brake 0.5 on a boolean channel, tyre 2 (MEDIUM) between SOFT and HARD, and a tyre age
    of 2.5 on a channel counted in whole laps.
    """
    import numpy as np

    return {
        "t": np.array([0.0, 1.0]),
        "x": np.array([0.0, 100.0]),
        "y": np.array([0.0, 50.0]),
        "speed": np.array([100.0, 200.0]),
        "throttle": np.array([0.0, 100.0]),
        "brake": np.array([0.0, 1.0]),
        "dist": np.array([0.0, 200.0]),
        "tyre_life": np.array([4.0, 1.0]),
        "gear": np.array([2.0, 8.0]),
        "drs": np.array([0.0, 12.0]),
        "lap": np.array([1.0, 2.0]),
        "tyre": np.array([1.0, 3.0]),
    }


def _resampled_midpoint():
    """The frame the resampler builds exactly half way between the two samples."""
    import numpy as np

    from src.arcade.data import SessionLoader

    data = _one_driver_two_samples()
    timeline = np.array([0.0, 0.5, 1.0])
    frames = SessionLoader()._resample_driver(
        data, data["t"], timeline, 1.0, {"throttle": 1.0, "brake": 100.0}, 5000.0
    )
    assert len(frames) == 3, "the fixture must produce the midpoint frame it is about"
    return frames[1]


def test_a_discrete_channel_only_ever_takes_a_value_the_car_published():
    """The mechanism, on four samples, without the 382 MB artefact.

    This is the guard that goes red on a revert to `np.interp`, which the pickle-backed
    guards above cannot: they would still be reading frames built by the fixed code.
    """
    midpoint = _resampled_midpoint()
    assert midpoint.gear == 2, "gear 5 is what linear interpolation invents here"
    assert midpoint.drs == 0, "DRS 6 is a code FastF1 never emits"
    assert midpoint.brake == 0.0, "brake is boolean; 50.0 is a pressure nobody published"
    assert midpoint.tyre == 1, "tyre 2 is MEDIUM, invented between SOFT and HARD"
    assert midpoint.tyre_life == 4.0, "an age is a whole number of laps"
    assert midpoint.lap == 1, "the lap does not advance until the sample that carries it"


def test_a_continuous_channel_still_interpolates():
    """The other half, or the fix could be 'resample nothing' and pass.

    Speed, throttle and position are genuinely continuous and must still take the value
    between two samples: a stepped speed trace is the defect this change must not create.
    """
    midpoint = _resampled_midpoint()
    assert midpoint.speed == 150.0
    assert midpoint.throttle == 50.0
    assert (midpoint.x, midpoint.y) == (50.0, 25.0)


def test_the_nearest_sample_pick_is_correct_at_both_ends_and_on_a_tie():
    """Exact hits, extrapolation past either end, and a deterministic tie.

    The tie matters on real data: Melbourne 2025 carries 907 duplicate-`t` pairs, and a
    pick that depended on floating-point noise would resample the same race differently
    on two machines.
    """
    import numpy as np

    from src.arcade.data import _nearest_sample

    t = np.array([0.0, 1.0, 2.0])
    picked = _nearest_sample(t, np.array([-5.0, 0.0, 0.4, 0.5, 0.6, 1.0, 2.0, 9.0]))
    assert list(picked) == [0, 0, 0, 0, 1, 1, 2, 2]


def test_an_impossible_gear_is_replaced_by_the_last_real_one():
    """`> 8` only, a leading invalid filled backwards, and neutral left alone."""
    import numpy as np

    from src.arcade.data import _drop_impossible_gears

    assert list(_drop_impossible_gears(np.array([3.0, 128.0, 128.0, 5.0]))) == [3.0, 3.0, 3.0, 5.0]
    assert list(_drop_impossible_gears(np.array([128.0, 12.0, 3.0, 4.0]))) == [3.0, 3.0, 3.0, 4.0]
    # Neutral is a real reading, so a two-sided predicate would erase these.
    untouched = np.array([0.0, 0.0, 2.0, 8.0])
    assert list(_drop_impossible_gears(untouched)) == [0.0, 0.0, 2.0, 8.0]


def test_a_channel_that_is_entirely_impossible_is_left_as_published():
    """There is nothing to fill from, and a NaN column would raise at frame build."""
    import numpy as np

    from src.arcade.data import _drop_impossible_gears

    published = np.array([128.0, 128.0])
    assert list(_drop_impossible_gears(published)) == [128.0, 128.0]
