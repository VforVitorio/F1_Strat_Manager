"""Race order and at-the-line intervals in the arcade replay (#844).

Two defects lived here and compounded:

1. the field was ranked by `dist` (after adding `(lap - 1) * track_len` to a
   value that already contained the completed laps), and `dist` is not a
   race-progress axis at all: each car accumulates the distance IT drove;
2. the gap was that distance divided by a hardcoded 55.56 m/s, an assumed
   200 km/h for every car, everywhere, in every condition.

**The fixture below carries per-car accumulation drift on purpose.** An
earlier version of this file did not, and an adversarial gate showed that 8
of its 14 tests passed with the fix reverted: `dist` and true progress were
the same quantity by construction, which is the one property the real data
does not have. `_car(drift_per_lap_m=...)` reproduces it, and the ranking
tests below fail without the fix.

Accuracy against real data is a separate question, measured on Melbourne
2025 and recorded in `src/arcade/gaps.py` together with the coordinates
that were refuted on the way.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest

pytest.importorskip("arcade", reason="the arcade replay is an optional surface")

from src.arcade.config import DT  # noqa: E402
from src.arcade.data import FrameData, SessionData  # noqa: E402
from src.arcade.gaps import RaceGapCalculator  # noqa: E402
from src.arcade.overlays import DriverInfoPanel, LeaderboardPanel  # noqa: E402

CIRCUIT_LENGTH_M = 5000.0
LAP_SECONDS = 100.0  # at 50 m/s, so a lap is a round number of frames
N_FRAMES = 8000  # 320 s at 25 Hz


def _car(
    speed_mps: float,
    head_start_laps: float = 0.0,
    drift_per_lap_m: float = 0.0,
    dead_from: int | None = None,
) -> list[FrameData]:
    """A car at constant speed, with its own per-lap distance accumulation.

    `rel_dist` is the true fraction of the lap and is therefore comparable
    across cars; `dist` accumulates `CIRCUIT_LENGTH_M + drift_per_lap_m`
    per lap and is therefore NOT. That asymmetry is the whole defect: on
    Melbourne 2025 the real drift reaches 1877 m on a 5220 m circuit.
    """
    lap_length = CIRCUIT_LENGTH_M + drift_per_lap_m
    frames = []
    for i in range(N_FRAMES):
        live = dead_from is None or i <= dead_from
        idx = i if live else dead_from
        progress = head_start_laps + (speed_mps * idx * DT) / CIRCUIT_LENGTH_M
        completed = int(progress)
        fraction = progress - completed
        frames.append(
            FrameData(
                t=i * DT,
                x=0.0,
                y=0.0,
                speed=speed_mps * 3.6,
                gear=7,
                drs=0,
                throttle=100.0,
                brake=0.0,
                lap=1 + completed,
                dist=(completed + fraction) * lap_length,
                rel_dist=fraction,
                tyre=1,
                tyre_life=5.0,
                active=live,
            )
        )
    return frames


def _blind_car(speed_mps: float) -> list[FrameData]:
    """A car FastF1 gives no `RelativeDistance` for. Real: HAD, Melbourne 2025."""
    return [FrameData(**{**vars(f), "rel_dist": float("nan")}) for f in _car(speed_mps)]


def _session(*, max_lap_number: int = 0, **cars) -> SessionData:
    """A session of synthetic cars.

    `max_lap_number` defaults to 0, which is what `SessionData` itself
    defaults to and means "the loader recorded no final lap". Nobody can
    then be known to have finished, so the tests that predate finishers
    keep exercising the same code path they always did.
    """
    return SessionData(
        gp_name="Test",
        location="Test",
        year=2025,
        frames_by_driver=dict(cars),
        circuit_length_m=CIRCUIT_LENGTH_M,
        max_lap_number=max_lap_number,
        total_frames=N_FRAMES,
    )


# A three-lap race. A car takes the flag by running out of telemetry while
# still ON its final lap: that is what the real data does, because the
# resampler stops at the last sample, which is at the line, before the lap
# field would have incremented. Frame `dead_from + 1` is therefore the flag,
# and is the first frame carrying the car's frozen final distance.
RACE_LAPS = 3
_FLAG_FRAME = {"P1": 7450, "P2": 7475, "P3": 7500}


def _finisher(code: str) -> list[FrameData]:
    """One of three cars that cross the line 1.0 s apart on the final lap."""
    flag = _FLAG_FRAME[code]
    head_start = (7500 - flag) / 2500 * 1.0  # laps of head start, so they stagger
    return _car(50.0, head_start_laps=head_start, dead_from=flag - 1)


def _frame(session: SessionData, idx: int) -> dict:
    """The internal frame dict `on_draw` builds, for the drawing panels."""
    return {
        "t": idx * DT,
        "lap": 1,
        "drivers": {
            code: {
                "lap": frames[idx].lap,
                "dist": frames[idx].dist,
                "speed": frames[idx].speed,
                "tyre": frames[idx].tyre,
                "active": frames[idx].active,
            }
            for code, frames in session.frames_by_driver.items()
        },
    }


def _order(session: SessionData, idx: int) -> list[str]:
    gaps = RaceGapCalculator(session)
    return [code for code, _, _ in LeaderboardPanel._rank_drivers(_frame(session, idx), gaps, idx)]


def _panel(code: str):
    """A stand-in for DriverInfoPanel: `_neighbor_gaps` reads only `self.code`.

    Constructing the real panel allocates `arcade.Text`, which wants a GL
    context CI does not have. Duck-typing also keeps the test honest the
    other way: a method that starts reading another attribute raises here
    instead of passing.
    """
    stand_in = SimpleNamespace(code=code, _gap_label=DriverInfoPanel._gap_label)
    stand_in._neighbor_gaps = lambda *a: DriverInfoPanel._neighbor_gaps(stand_in, *a)
    return stand_in


def _labels(session: SessionData, code: str, idx: int) -> tuple[str, str]:
    gaps = RaceGapCalculator(session)
    frame = _frame(session, idx)
    ranked = [(c, p) for c, _, p in LeaderboardPanel._rank_drivers(frame, gaps, idx)]
    return _panel(code)._neighbor_gaps(ranked, gaps, frame, idx)


# --- Defect 1: what the field is ranked BY ----------------------------------


def test_the_order_is_right_when_dist_and_true_progress_disagree():
    """The test the old fixture could not express, because it had no drift.

    SLOW runs long per lap, so its `dist` exceeds FAST's while FAST is
    genuinely ahead on track. Ranking on `dist` inverts them; ranking on
    laps completed plus fraction of the lap does not.
    """
    session = _session(FAST=_car(52.0), SLOW=_car(50.0, drift_per_lap_m=400.0))
    idx = N_FRAMES - 1
    fast = session.frames_by_driver["FAST"][idx]
    slow = session.frames_by_driver["SLOW"][idx]

    # The premise: FAST is ahead on track, and `dist` says the opposite.
    assert (fast.lap, fast.rel_dist) > (slow.lap, slow.rel_dist), "FAST must be ahead on track"
    assert fast.dist < slow.dist, "the fixture must reproduce the dist inversion"

    assert _order(session, idx) == ["FAST", "SLOW"]


def test_the_leader_is_the_car_that_has_covered_the_most_track():
    """Three cars, three different drifts, sampled across the race.

    P3's drift is 700 m per lap and not the 160 an earlier version used.
    At 160 the three cars' `dist` never actually inverted, so this test
    stayed green against a plain `dist` sort and only one test in the file
    guarded the defect it was written for. The premise assert below is
    what stops the fixture drifting back into agreement.
    """
    session = _session(
        P1=_car(56.0, drift_per_lap_m=-80.0),
        P2=_car(53.0, drift_per_lap_m=0.0),
        P3=_car(50.0, drift_per_lap_m=700.0),
    )
    frames = session.frames_by_driver
    assert frames["P3"][6000].dist > frames["P1"][6000].dist, (
        "the fixture must invert the dist order, or a dist sort passes this test"
    )

    for idx in (2000, 4000, 6000, N_FRAMES - 1):
        assert _order(session, idx) == ["P1", "P2", "P3"], f"wrong at frame {idx}"


def test_the_fraction_is_normalised_by_the_cars_own_lap():
    """The mechanism the whole coordinate rests on, and nothing was pinning it.

    Replace `_current_lap_bounds`'s body with `return start, circuit_length`
    and every other test in this file still passes: the fixture laps and
    the [0, 1] clamp conspire to hide it. So the one sentence both
    docstrings call "what makes it comparable between cars" could be
    deleted with the suite green.

    A car that runs 5400 m per lap, exactly half way through its second
    lap, is at 1.5. Normalised by the circuit constant instead it reads
    1.54, and every car with a different drift reads differently wrong.
    """
    session = _session(SOLO=_car(50.0, drift_per_lap_m=400.0))
    gaps = RaceGapCalculator(session)
    idx = 3750  # 150 s at 50 m/s: one lap and a half of true progress

    assert gaps.progress("SOLO", idx) == pytest.approx(1.5, abs=0.005)


def test_a_crossing_is_the_first_frame_of_the_increment_and_not_the_last():
    """The other unpinned invariant: `setdefault`, not assignment.

    The resampled `lap` field can flicker for a few frames around the
    line, so the same lap increments more than once. Keeping the FIRST
    occurrence measured p95 error 83 ms -> 68 ms; keeping the last is a
    one-character change that no test could see.
    """
    frames = _car(50.0)
    # The flicker: lap goes 2, back to 1, then 2 again over three frames.
    for offset, lap in ((0, 2), (1, 1), (2, 2)):
        frames[2500 + offset] = FrameData(**{**vars(frames[2500 + offset]), "lap": lap})
    gaps = RaceGapCalculator(_session(SOLO=frames))

    crossing = gaps._crossings["SOLO"][1]

    assert crossing == pytest.approx(2500 * DT, abs=DT / 2), "the first rise, not the third"


def test_a_car_whose_rel_dist_is_missing_is_still_ranked():
    """The coordinate does not read `rel_dist`, so a NaN one cannot break it.

    On Melbourne 2025 FastF1 leaves `RelativeDistance` NaN for 100% of
    HAD's frames. Ranking on it would have dropped a whole car off the
    order; ranking on distance-since-the-last-crossing does not notice.
    """
    session = _session(SLOW=_car(50.0), BLIND=_blind_car(60.0))
    idx = 4000
    gaps = RaceGapCalculator(session)

    ranked = LeaderboardPanel._rank_drivers(_frame(session, idx), gaps, idx)

    assert [code for code, _, _ in ranked] == ["BLIND", "SLOW"], "the faster car leads"
    assert gaps.progress("BLIND", idx) is not None


def test_a_stopped_car_is_ranked_where_its_distance_says_it_stopped():
    """The defect that made `rel_dist` unusable, reproduced.

    The 25 Hz resampler clamps past a driver's last real sample, and DOO's
    `rel_dist` saturates at 1.000 from frame 1500 while his distance shows
    him stopped 1717 m into lap 1. Ranking on `rel_dist` drew a car that
    had crashed at turn 1 as the race leader for 68 seconds of replay.
    """
    session = _session(RUNNING=_car(50.0), CRASHED=_car(50.0, dead_from=600))
    idx = 4000
    gaps = RaceGapCalculator(session)
    stuck = session.frames_by_driver["CRASHED"][idx]

    assert stuck.rel_dist == pytest.approx(0.24, abs=0.01), "frozen a quarter into lap 1"
    assert gaps.progress("CRASHED", idx) == pytest.approx(0.24, abs=0.01)
    assert _order(session, idx) == ["RUNNING", "CRASHED"]


# --- Defect 2: the interval itself ------------------------------------------


def test_the_interval_is_the_difference_of_two_line_crossings():
    """Two cars at 50 m/s, a tenth of a lap apart, is a 10.0 s interval.

    The old formula divided the same 500 m by 55.56 and returned 9.00 s: a
    10% error on a case where both cars travel at very nearly the speed
    the constant was chosen to approximate.
    """
    session = _session(FRONT=_car(50.0, head_start_laps=0.1), BACK=_car(50.0))
    gaps = RaceGapCalculator(session)

    assert gaps.interval_at_line("FRONT", "BACK", lap=1) == pytest.approx(10.0, abs=DT)
    assert gaps.interval_at_line("FRONT", "BACK", lap=1) != pytest.approx(500.0 / 55.56, abs=0.1)


def test_a_slow_field_reads_slow_and_a_fast_field_reads_fast():
    """The interval follows the field's real speed, which is the whole point.

    A tenth of a lap is 20 s apart under a Safety Car at 25 m/s and 7.1 s
    at 70 m/s. A hardcoded 55.56 answers 9.0 s to both, over-reading the
    Safety Car case by more than half.
    """
    for speed, expected in ((25.0, 20.0), (70.0, 500.0 / 70.0)):
        session = _session(FRONT=_car(speed, head_start_laps=0.1), BACK=_car(speed))

        seconds = RaceGapCalculator(session).interval_at_line("FRONT", "BACK", lap=1)

        assert seconds == pytest.approx(expected, abs=2 * DT)
        assert seconds != pytest.approx(500.0 / 55.56, abs=0.5)


def test_no_assumed_speed_constant_survives_in_the_gap_path():
    """The literal itself, hunted structurally rather than by grep.

    Parsed as an AST and not searched as text, because both modules
    explain in prose why the constant is gone and a text search cannot
    tell an executable 55.56 from a sentence about one. This repo has a
    standing lesson that a grep is not an audit.
    """
    import ast
    from pathlib import Path

    import src.arcade.gaps as gaps_module
    import src.arcade.overlays as overlays_module

    for module in (gaps_module, overlays_module):
        tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
        numbers = [
            node.value
            for node in ast.walk(tree)
            if isinstance(node, ast.Constant) and isinstance(node.value, (int, float))
        ]
        assert 55.56 not in numbers, f"{module.__name__} still carries the 200 km/h divisor"


# --- Being a lap down -------------------------------------------------------


def test_laps_down_is_positional_and_survives_the_accumulation_drift():
    """The case the `dist` form gets wrong.

    The rate carried here used to be "4.9% of same-corner pairs", which
    is a number nothing in this repo measured: the figure #862 actually
    published for same-corner disagreement is 3.4% over n=4,934, under a
    convention this docstring never stated. An unsourced percentage in a
    test is how a wrong number gets quoted back as evidence, so it is gone
    rather than swapped for one whose population may differ.

    Both cars sit at the same point on track, exactly one lap apart, and
    the lapped car runs long enough per lap that its `dist` deficit is
    less than a circuit length. The old `dist // circuit_length` form
    therefore answered 0 and the panel showed a seconds interval for a car
    a full lap down.
    """
    # Half the speed, so at 200 s LEAD is on lap 3 and LAPPED on lap 2, both
    # exactly at the line. A head start would not do: `laps_completed` counts
    # crossings the replay actually observed, and a car that begins mid-race
    # never crossed the line for the laps it was handed.
    session = _session(LEAD=_car(50.0), LAPPED=_car(25.0, drift_per_lap_m=800.0))
    gaps = RaceGapCalculator(session)
    idx = 5000
    lead = session.frames_by_driver["LEAD"][idx]
    lapped = session.frames_by_driver["LAPPED"][idx]

    assert lead.rel_dist == pytest.approx(lapped.rel_dist, abs=0.01), "same point on track"
    assert lead.lap - lapped.lap == 1, "exactly one lap apart"
    assert lead.dist - lapped.dist < CIRCUIT_LENGTH_M, "the dist form would answer 0 here"

    assert gaps.laps_down(gaps.progress("LEAD", idx), gaps.progress("LAPPED", idx)) == 1


def test_a_lap_number_that_differs_by_one_is_not_being_lapped():
    """One car past the line and the other not is most of every lap.

    `laps_down` is therefore positional: the two cars below are a fraction
    of a second apart and must read in seconds.
    """
    session = _session(FRONT=_car(50.0, head_start_laps=0.01), BACK=_car(50.0))
    gaps = RaceGapCalculator(session)
    idx = 2499  # FRONT has crossed the line, BACK has not
    frames = session.frames_by_driver

    assert frames["FRONT"][idx].lap - frames["BACK"][idx].lap == 1
    assert gaps.laps_down(gaps.progress("FRONT", idx), gaps.progress("BACK", idx)) == 0


def test_a_lapped_car_renders_as_a_lap_down_not_as_seconds():
    session = _session(LEAD=_car(60.0), LAPPED=_car(25.0))
    idx = N_FRAMES - 1

    ahead, _ = _labels(session, "LAPPED", idx)

    assert ahead.startswith("LEAD +")
    assert ahead.endswith("LAP") or ahead.endswith("LAPS")
    assert "(L)" not in ahead and "s" not in ahead.removesuffix("LAPS")


def test_cars_on_the_same_lap_read_in_seconds_and_say_they_are_at_the_line():
    """The "(L)" suffix is not decoration: it is what stops a lap-quantised
    number being read as a live one on a fidelity surface."""
    session = _session(FRONT=_car(50.0, head_start_laps=0.06), BACK=_car(50.0))

    ahead, _ = _labels(session, "BACK", 3000)

    assert ahead.startswith("FRONT +")
    assert ahead.endswith("s (L)")
    assert float(ahead.split("+")[1].split("s")[0]) == pytest.approx(6.0, abs=2 * DT)


# --- Unknown, retired and inverted stay visible as such ---------------------


def test_a_retired_neighbour_reads_OUT_rather_than_a_stale_interval():
    """`np.interp` clamps past a driver's last sample, so a parked car keeps
    reporting its final state forever. Without this branch the panel showed
    an interval up to 22 minutes old naming a car that had stopped."""
    session = _session(ALIVE=_car(50.0), STOPPED=_car(50.0, head_start_laps=0.2, dead_from=3000))
    idx = 6000

    assert session.frames_by_driver["STOPPED"][idx].active is False
    ahead, behind = _labels(session, "ALIVE", idx)

    assert "STOPPED OUT" in (ahead, behind)


def test_a_retired_car_keeps_the_laps_it_actually_completed():
    """Its crossings stop where its telemetry does, and nothing extrapolates."""
    session = _session(ALIVE=_car(50.0), STOPPED=_car(50.0, dead_from=3000))
    gaps = RaceGapCalculator(session)

    # 3000 frames at 50 m/s is 120 s: one full lap plus a fifth, so exactly
    # one line crossing. The fifth it was part-way through is not a lap.
    assert gaps.laps_completed("STOPPED", N_FRAMES - 1) == 1
    assert gaps.laps_completed("ALIVE", N_FRAMES - 1) == 3
    assert gaps.interval_at_line("ALIVE", "STOPPED", lap=2) is None


def test_an_interval_nobody_has_driven_yet_is_none():
    session = _session(FRONT=_car(50.0, head_start_laps=0.1), BACK=_car(50.0))
    gaps = RaceGapCalculator(session)

    assert gaps.interval_at_line("FRONT", "BACK", lap=0) is None
    assert gaps.interval_at_line("FRONT", "BACK", lap=99) is None
    assert gaps.interval_at_line("NOBODY", "BACK", lap=1) is None
    assert gaps.last_shared_lap("FRONT", "BACK", frame_idx=100) == 0


def test_an_inverted_call_is_none_and_never_a_plausible_zero():
    """Zero is a real answer for both methods, so it cannot double as "invalid".

    `laps_down` used to return 0 for an inverted pair, which is also what
    it returns for "same lap" — the twin of a discipline `interval_at_line`
    spends five docstring lines defending twenty lines above it.
    """
    session = _session(FRONT=_car(50.0, head_start_laps=0.1), BACK=_car(50.0))
    gaps = RaceGapCalculator(session)
    idx = 3000
    front, back = gaps.progress("FRONT", idx), gaps.progress("BACK", idx)

    assert gaps.interval_at_line("FRONT", "BACK", lap=1) == pytest.approx(10.0, abs=DT)
    assert gaps.interval_at_line("BACK", "FRONT", lap=1) is None
    assert gaps.laps_down(front, back) == 0
    assert gaps.laps_down(back, front) is None
    assert gaps.laps_down(None, back) is None


def test_the_panel_labels_both_neighbours_from_its_own_position():
    session = _session(
        P1=_car(50.0, head_start_laps=0.2),
        P2=_car(50.0, head_start_laps=0.1),
        P3=_car(50.0),
    )

    ahead, behind = _labels(session, "P2", 3000)

    assert ahead.startswith("P1 +") and ahead.endswith("s (L)")
    assert behind.startswith("P3 -") and behind.endswith("s (L)")
    assert float(ahead.split("+")[1].split("s")[0]) == pytest.approx(10.0, abs=2 * DT)
    assert float(behind.split("-")[1].split("s")[0]) == pytest.approx(10.0, abs=2 * DT)


def test_the_leader_and_the_last_car_have_no_neighbour_to_measure():
    session = _session(P1=_car(50.0, head_start_laps=0.1), P2=_car(50.0))

    assert _labels(session, "P1", 3000)[0] == "LEADER"
    assert _labels(session, "P2", 3000)[1] == "LAST"


# --- The crossings themselves -----------------------------------------------


# --- Taking the flag is not retiring (#855) ---------------------------------


def _finish_session(**extra) -> SessionData:
    return _session(
        max_lap_number=RACE_LAPS,
        # Inserted in the WRONG order on purpose: every finisher sits at
        # exactly RACE_LAPS, so a plain sort keeps whatever order the dict
        # was built in and the test would pass without the tie-break.
        P3=_finisher("P3"),
        P1=_finisher("P1"),
        P2=_finisher("P2"),
        **extra,
    )


def test_the_final_classification_is_the_order_the_cars_crossed_the_line():
    """The last seconds of the replay, which is the state a viewer reads as THE result.

    Before #855 the flag was not a crossing, so every finisher stayed on
    the lap before their last and their fraction of the final lap was
    measured against the length of the PREVIOUS one. Whoever's final lap
    ran long clamped at exactly 1.0 and jumped above everyone else.
    Measured on Melbourne 2025 the board agreed with the official
    classification on 1 of 20 positions: VER (P2) was drawn 8th.
    """
    session = _finish_session()
    idx = N_FRAMES - 1
    gaps = RaceGapCalculator(session)

    # The premise: all three are level on progress, so only the tie-break
    # can separate them.
    assert [gaps.progress(c, idx) for c in ("P1", "P2", "P3")] == [float(RACE_LAPS)] * 3

    assert _order(session, idx) == ["P1", "P2", "P3"]


def test_a_car_that_took_the_flag_is_not_rendered_as_retired():
    """`active` goes False the instant a finisher's telemetry ends.

    Reading it alone put "OUT" on the winner from the moment he won: 19 of
    20 rows at the final frame of Melbourne 2025, the whole podium among
    them.
    """
    session = _finish_session(RETIRED=_car(50.0, dead_from=3000))
    idx = N_FRAMES - 1
    gaps = RaceGapCalculator(session)

    assert session.frames_by_driver["P1"][idx].active is False, "the premise: it looks retired"
    assert gaps.has_finished("P1") is True
    assert gaps.has_finished("RETIRED") is False

    ahead, behind = _labels(session, "P2", idx)
    assert ahead == "P1 +1.00s (L)", ahead
    assert behind.startswith("P3 -"), behind
    assert "OUT" not in ahead + behind

    assert "RETIRED OUT" in _labels(session, "P3", idx)


def test_the_flag_is_a_crossing_so_the_final_lap_finally_has_an_interval():
    """The three cross 1.0 s apart, and that is what the last lap must read.

    On Melbourne 2025 this reproduces the published result: NOR-VER comes
    out at 0.880 s against an official 0.895 s, and NOR-RUS at 8.480 s
    against an official 8.481 s - an external check, unlike the rest of
    this module's accuracy work, which compares the replay against the
    table that sliced it.
    """
    gaps = RaceGapCalculator(_finish_session())

    assert gaps.laps_completed("P1", N_FRAMES - 1) == RACE_LAPS
    assert gaps.interval_at_line("P1", "P2", lap=RACE_LAPS) == pytest.approx(1.0, abs=DT)
    assert gaps.interval_at_line("P1", "P3", lap=RACE_LAPS) == pytest.approx(2.0, abs=DT)
    assert gaps.interval_at_line("P2", "P1", lap=RACE_LAPS) is None, "inverted stays None"


def test_a_lapped_car_that_takes_the_flag_is_a_finisher_on_its_own_lap():
    """The case that kills the obvious rule, and that Melbourne 2025 cannot show.

    "A finisher is a car that reached the final lap" measures perfectly on
    a race where every finisher was on the lead lap, and calls a lapped
    finisher a retirement. The rule shipped is the racing one instead: when
    the leader takes the flag, everyone still running takes it at their
    next line - so a finisher's telemetry ends at or after the leader's.

    LAPPED is on lap 2 when P1 wins and crosses its own line 2 s later. It
    must be credited lap 2, not the leader's lap 3.
    """
    session = _finish_session(LAPPED=_car(33.34, dead_from=7498))
    idx = N_FRAMES - 1
    gaps = RaceGapCalculator(session)

    assert session.frames_by_driver["LAPPED"][idx].lap == RACE_LAPS - 1, "never reached lap 3"
    assert gaps.has_finished("LAPPED") is True
    assert gaps.laps_completed("LAPPED", idx) == RACE_LAPS - 1
    assert gaps.progress("LAPPED", idx) == pytest.approx(float(RACE_LAPS - 1))
    assert _order(session, idx) == ["P1", "P2", "P3", "LAPPED"]


def test_a_car_that_retires_on_the_final_lap_does_not_take_the_flag():
    """It set the flag time for everyone and was then drawn P1, ahead of the winner.

    "The leader is the first car to reach the final lap" reads a car that
    crashes a fifth of the way into it as the winner: its telemetry ends
    first, so it defined the flag, it was credited a crossing of a lap it
    never completed, and the tie-break put it top for the whole
    flag-to-end window. Executed before the fix, the order was
    `[CRASH, WIN, P2]`.

    The rule that holds is that the flag is taken by the car LEADING when
    its telemetry ends. Melbourne 2025 has no final-lap retirement, so the
    broken version measured perfectly on the only race on disk.
    """
    crash = _car(50.0, head_start_laps=-0.30, dead_from=6200)
    session = _finish_session(CRASH=crash)
    idx = N_FRAMES - 1
    gaps = RaceGapCalculator(session)

    assert session.frames_by_driver["CRASH"][idx].lap == RACE_LAPS, "it DID reach the final lap"
    assert gaps.has_finished("CRASH") is False
    assert gaps.has_finished("P1") is True
    assert gaps.laps_completed("CRASH", idx) == RACE_LAPS - 1, (
        "no crossing of the lap it stopped on"
    )
    assert _order(session, idx) == ["P1", "P2", "P3", "CRASH"]
    assert "CRASH OUT" in _labels(session, "P3", idx)


def test_a_lapped_car_neither_sets_the_flag_nor_loses_its_finish():
    """The guard must not turn a lapped finisher into a retirement.

    A car a lap down runs to the end on a lap the leader has already
    passed. It is never leading, so it must not set the flag; and it took
    the chequered flag all the same, so it must still read as a finisher.
    Both halves have to hold at once.

    The name and the docstring used to say the opposite of the asserts -
    "ends first" for a car that ends 49 frames LATER, and "not a finisher"
    directly above `has_finished("LAPPED") is True`. A test whose prose
    contradicts its body is how the wrong behaviour gets defended later.
    """
    session = _finish_session(LAPPED=_car(33.34, dead_from=7498))
    gaps = RaceGapCalculator(session)

    assert gaps.has_finished("P1") is True, "the leader still takes the flag"
    assert gaps.has_finished("LAPPED") is True, "and a lapped finisher still finishes"
    assert gaps.laps_completed("P1", N_FRAMES - 1) == RACE_LAPS


def test_without_a_known_final_lap_nobody_is_a_finisher():
    """`max_lap_number` of 0 is what an older cache carries.

    The honest answer is then that nothing is known about who finished,
    which is exactly how this replay behaved before finishers existed.
    """
    session = _session(P1=_finisher("P1"), P2=_finisher("P2"))
    gaps = RaceGapCalculator(session)

    assert gaps.has_finished("P1") is False
    assert gaps.laps_completed("P1", N_FRAMES - 1) == RACE_LAPS - 1, "no flag crossing"


def test_a_crossing_is_keyed_by_the_lap_it_ends_and_only_real_laps_count():
    """Off by one here would put every interval on the wrong lap silently.

    A synthetic entry for the lap in progress would credit every
    retirement with a lap it never drove: a car that stops 1700 m into
    lap 1 would rank a full lap up the field.
    """
    session = _session(SOLO=_car(50.0))  # 100 s per lap, 320 s of frames

    crossings = RaceGapCalculator(session)._crossings["SOLO"]

    # 320 s of frames at 100 s per lap: three lines crossed, and the fourth
    # lap still running. The chequered flag is not a crossing.
    assert sorted(crossings) == [1, 2, 3]
    for lap in (1, 2, 3):
        assert crossings[lap] == pytest.approx(lap * LAP_SECONDS, abs=DT)


# --- Defect 3: who took the flag, when the official result is on disk --------
#
# #879: there is no threshold-free telemetry rule that separates a noisy
# winner from a final-lap crasher. The replay replays a COMPLETED session,
# so the loader now caches FastF1's official classification and the
# calculator consumes it; the derived anchor stays only as the fallback for
# sessions that have none (synthetic tests, a future live feed).


def _own_lengths_frame(i: int, lap: int, dist: float, live: bool) -> FrameData:
    return FrameData(
        t=i * DT,
        x=0.0,
        y=0.0,
        speed=180.0,
        gear=7,
        drs=0,
        throttle=100.0,
        brake=0.0,
        lap=lap,
        dist=dist,
        rel_dist=0.0,
        tyre=1,
        tyre_life=5.0,
        active=live,
    )


def _own_lengths_car(lap_lengths_m: list[float]) -> list[FrameData]:
    """A car whose laps have different OWN lengths, at a constant 50 m/s.

    This is the noise #879 measured on Melbourne 2025 (adjacent-lap
    own-length deltas of median 9.2 m, up to 340 m): `dist` accumulates
    what the car actually drove, so at its own telemetry end a car's
    final-lap fraction is measured against the PREVIOUS lap's own length.
    `_car` cannot express it — all its laps are the same length.
    """
    speed = 50.0
    frames: list[FrameData] = []
    dist = 0.0
    for lap_no, length in enumerate(lap_lengths_m, start=1):
        for i in range(int(round(length / speed / DT))):
            frames.append(_own_lengths_frame(len(frames), lap_no, dist + speed * i * DT, True))
        dist += length
    while len(frames) < N_FRAMES:
        frames.append(_own_lengths_frame(len(frames), len(lap_lengths_m), dist, False))
    return frames


def test_the_wall_clock_winner_finishes_when_the_official_result_says_so():
    """#879 miss 2, closed: adjacent-lap noise no longer decides the flag.

    WIN's previous lap ran 100 m long, so at its own end its fraction reads
    5000/5100 < 1.0; P2's ran 50 m short, so P2 clamps to 1.0 while still
    short of its line. The derived anchor then crowns P2 and classifies the
    wall-clock winner — who crossed 1.0 s earlier — as a retirement. The
    official result knows both finished, and it is on disk.
    """
    session = _session(
        max_lap_number=3,
        WIN=_own_lengths_car([5000.0, 5100.0, 5000.0]),
        P2=_own_lengths_car([5000.0, 4950.0, 5200.0]),
    )
    session.official_status = {"WIN": "Finished", "P2": "Finished"}
    idx = N_FRAMES - 1
    gaps = RaceGapCalculator(session)

    assert gaps.has_finished("WIN") is True, "the wall-clock winner took the flag"
    assert gaps.has_finished("P2") is True
    assert _order(session, idx) == ["WIN", "P2"]
    assert gaps.interval_at_line("WIN", "P2", lap=3) == pytest.approx(1.0, abs=DT)


def test_a_leading_final_lap_retiree_is_out_and_takes_nobody_with_it():
    """#879 miss 1, closed: the retiring leader neither wins nor launders others.

    CRASH leads by 0.3 laps and stops ~100 m short of its final line; RET2
    is a genuine retirement that stops after the winner's flag. The derived
    anchor let CRASH define the flag: it was drawn P1 with the full race
    distance and RET2 inherited a finish. The official result says both
    retired.
    """
    session = _finish_session(
        CRASH=_car(50.0, head_start_laps=0.30, dead_from=6700),
        RET2=_car(50.0, head_start_laps=-0.50, dead_from=7600),
    )
    session.official_status = {
        "P1": "Finished",
        "P2": "Finished",
        "P3": "Finished",
        "CRASH": "Retired",
        "RET2": "Retired",
    }
    idx = N_FRAMES - 1
    gaps = RaceGapCalculator(session)

    assert gaps.has_finished("CRASH") is False
    assert gaps.has_finished("RET2") is False
    assert gaps.has_finished("P1") is True
    assert _order(session, idx) == ["P1", "P2", "P3", "CRASH", "RET2"]
    assert "CRASH OUT" in _labels(session, "P3", idx)


@pytest.mark.parametrize("lapped_status", ["Lapped", "+1 Lap"])
def test_official_truth_survives_a_telemetry_dropout(lapped_status):
    """A car FastF1 loses mid-race but that officially finished stays FIN.

    Telemetry absence is not retirement: a dropout that ends a car's frames
    before the leader's flag made the derived rule call it a retirement.
    The official row knows better. 'Lapped' is the spelling every 2023-2025
    race actually serves (executed against jolpica; China 2025 classifies
    BOR/HUL/TSU P14-P16 with it); '+1 Lap' is the deep-historical Ergast
    form. NOT fastf1's own `DriverResult.dnf`, which calls 'Lapped' a DNF.
    """
    session = _finish_session(DROP=_car(33.34, dead_from=7000))
    session.official_status = {
        "P1": "Finished",
        "P2": "Finished",
        "P3": "Finished",
        "DROP": lapped_status,
    }
    gaps = RaceGapCalculator(session)

    assert gaps.has_finished("DROP") is True


def test_a_disagreement_between_official_and_derived_is_logged(caplog):
    """Two code paths answering one question is this repo's own defect class.

    The official result wins, but a divergence from the derived rule must
    become evidence in the log — it is the net that catches a vocabulary
    surprise (a DSQ spelling, a status FastF1 renames) instead of letting
    the replay and a future live feed drift apart silently.
    """
    session = _finish_session(CRASH=_car(50.0, head_start_laps=0.30, dead_from=6700))
    session.official_status = {
        "P1": "Finished",
        "P2": "Finished",
        "P3": "Finished",
        "CRASH": "Retired",
    }
    with caplog.at_level(logging.WARNING):
        RaceGapCalculator(session)

    flagged = [r for r in caplog.records if "Flag classification" in r.message]
    assert flagged, "the official-vs-derived disagreement must be logged"
    assert any("CRASH" in r.message and "Retired" in r.message for r in flagged)


def test_no_warning_when_official_and_derived_agree(caplog):
    """Melbourne 2025 in miniature: agreement on every driver stays silent."""
    session = _finish_session(RETIRED=_car(50.0, dead_from=3000))
    session.official_status = {
        "P1": "Finished",
        "P2": "Finished",
        "P3": "Finished",
        "RETIRED": "Retired",
    }
    with caplog.at_level(logging.WARNING):
        RaceGapCalculator(session)

    assert not [r for r in caplog.records if "Flag classification" in r.message]


def test_a_car_with_no_position_data_has_no_progress_rather_than_zero():
    """The absence must not wear the value a car on the grid also reads (#886).

    FastF1 delivers nothing for one whole driver on Melbourne 2025: 154,173
    frames with `dist`, `x` and `y` all at 0.0. `progress` computed 0.0 from
    that and returned it, so "we have no data for this car" and "this car is
    at the start line" were the same number - on the coordinate the entire
    field is ordered by, one sprint before the wire starts publishing it.

    The panel already partitions `None` out and appends it last. It simply
    never received one.
    """
    session = _session(
        max_lap_number=RACE_LAPS,
        RUNNING=_car(50.0),
        BLIND=[FrameData(**{**vars(f), "dist": 0.0, "x": 0.0, "y": 0.0}) for f in _car(50.0)],
    )
    session.has_position["BLIND"] = False
    gaps = RaceGapCalculator(session)

    assert gaps.progress("BLIND", 4000) is None, "no position data is an absence"
    assert gaps.progress("RUNNING", 4000) is not None, "and the running car still has one"

    order = _order(session, 4000)
    assert order[-1] == "BLIND", "the unknown car sorts last rather than leading from the grid"
