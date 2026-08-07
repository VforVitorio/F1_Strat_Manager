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


def _session(**cars) -> SessionData:
    return SessionData(
        gp_name="Test",
        location="Test",
        year=2025,
        frames_by_driver=dict(cars),
        circuit_length_m=CIRCUIT_LENGTH_M,
        total_frames=N_FRAMES,
    )


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
    """Three cars, three different drifts, sampled across the race."""
    session = _session(
        P1=_car(56.0, drift_per_lap_m=-80.0),
        P2=_car(53.0, drift_per_lap_m=0.0),
        P3=_car(50.0, drift_per_lap_m=160.0),
    )

    for idx in (2000, 4000, 6000, N_FRAMES - 1):
        assert _order(session, idx) == ["P1", "P2", "P3"], f"wrong at frame {idx}"


def test_a_car_whose_rel_dist_is_missing_is_still_ranked():
    """The coordinate does not read `rel_dist`, so a NaN one cannot break it.

    On Melbourne 2025 FastF1 leaves `RelativeDistance` NaN for 100 % of
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
    10 % error on a case where both cars travel at very nearly the speed
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
    """The case the `dist` form got wrong on 4.9 % of same-corner pairs.

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
