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

    roots = ("src/arcade", "src/pitwall", "src/simulation", "scripts")
    allowed = Path("src/arcade/config.py").resolve()
    offenders: list[str] = []

    for root in roots:
        for path in Path(root).rglob("*.py"):
            if path.resolve() == allowed:
                continue
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if not isinstance(node, (ast.Set, ast.Tuple, ast.List)):
                    continue
                values = [
                    element.value
                    for element in node.elts
                    if isinstance(element, ast.Constant) and isinstance(element.value, int)
                ]
                if len(values) == len(node.elts) and set(values) == set(DRS_OPEN_CODES):
                    offenders.append(f"{path}:{node.lineno}")

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
