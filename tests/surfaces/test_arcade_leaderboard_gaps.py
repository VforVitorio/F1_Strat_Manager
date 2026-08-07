"""Leaderboard ranking and at-the-line intervals in the arcade replay (#844).

Two independent defects lived here and compounded:

1. `_rank_drivers` added `(lap - 1) * track_len` to `dist`, which is
   already race-cumulative, so a lapped car's progress was inflated by a
   full circuit length per lap of difference.
2. the gap was that distance divided by a hardcoded 55.56 m/s, an assumed
   200 km/h for every car, everywhere, in every condition.

The ranking survived (1) because the error is monotone in true progress,
which is exactly why nobody noticed: a true sentence, "the order is
right", sitting on top of a false one, "so the gaps are right".

The fixtures build cars at constant speed on a synthetic circuit, so every
expected interval is arithmetic checkable by hand rather than a number read
back off the implementation. The method's accuracy against real data is a
separate question and was measured on Melbourne 2025 against
`laps.parquet`: median 17 ms over 13,854 driver-pair comparisons. That
measurement is recorded in `src/arcade/gaps.py`, along with the four
coordinates that were tried and refuted before it.
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
N_FRAMES = 8000  # 320 s at 25 Hz: long enough that even a 25 m/s car finishes a lap


def _car(speed_mps: float, head_start_m: float = 0.0, dead_from: int | None = None):
    """A car at constant speed: `dist`, `lap` and `t` all relate by hand-checkable maths."""
    frames = []
    for i in range(N_FRAMES):
        live = dead_from is None or i <= dead_from
        idx = i if live else dead_from
        dist = head_start_m + speed_mps * idx * DT
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
                lap=1 + int(dist // CIRCUIT_LENGTH_M),
                dist=dist,
                rel_dist=(dist % CIRCUIT_LENGTH_M) / CIRCUIT_LENGTH_M,
                tyre=1,
                tyre_life=5.0,
                active=live,
            )
        )
    return frames


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


def _ranked(session: SessionData, idx: int) -> list[tuple[str, float]]:
    return [(code, prog) for code, _, prog in LeaderboardPanel._rank_drivers(_frame(session, idx))]


def _panel(code: str):
    """A stand-in for DriverInfoPanel: `_neighbor_gaps` reads only `self.code`.

    Constructing the real panel allocates `arcade.Text` objects, which want
    a GL context CI does not have. Duck-typing keeps the test honest in the
    other direction too: if the method starts reading another attribute it
    raises here instead of passing.
    """
    stand_in = SimpleNamespace(code=code, _gap_label=DriverInfoPanel._gap_label)
    stand_in._neighbor_gaps = lambda *a: DriverInfoPanel._neighbor_gaps(stand_in, *a)
    return stand_in


# --- Defect 1: the ranking's progress term ----------------------------------


def test_progress_is_dist_and_is_not_inflated_by_the_lap_term():
    """A lapped car must not gain a circuit length of phantom progress.

    LAP has driven 9600 m (lap 2); SLO has driven 3200 m (lap 1). True
    separation is 6400 m. The old term added `(2-1) * 5000` to LAP alone,
    which on Melbourne worked out to about 95 s of fabricated gap per lap
    of difference once divided by the 55.56 constant.
    """
    session = _session(LAP=_car(60.0), SLO=_car(20.0))
    ranked = LeaderboardPanel._rank_drivers(_frame(session, N_FRAMES - 1))

    order = [code for code, _, _ in ranked]
    progress = {code: prog for code, _, prog in ranked}
    real_separation = (
        session.frames_by_driver["LAP"][-1].dist - session.frames_by_driver["SLO"][-1].dist
    )

    assert order == ["LAP", "SLO"]
    assert progress["LAP"] - progress["SLO"] == pytest.approx(real_separation)
    assert progress["LAP"] - progress["SLO"] != pytest.approx(real_separation + CIRCUIT_LENGTH_M)


def test_the_order_is_unchanged_by_the_fix():
    """The bug never broke the order, and the fix must not either."""
    session = _session(A=_car(70.0), B=_car(60.0), C=_car(50.0), D=_car(20.0))

    for idx in (100, 1000, N_FRAMES - 1):
        order = [code for code, _, _ in LeaderboardPanel._rank_drivers(_frame(session, idx))]
        assert order == ["A", "B", "C", "D"]


# --- Defect 2: the interval itself ------------------------------------------


def test_the_interval_is_the_difference_of_two_line_crossings():
    """Two cars at 50 m/s, 500 m apart, is a 10.0 s interval. Nothing else is.

    The old formula divided the same 500 m by 55.56 and returned 9.00 s: a
    10 % error on a case where both cars are travelling at very nearly the
    speed the constant was chosen to approximate.
    """
    session = _session(FRONT=_car(50.0, head_start_m=500.0), BACK=_car(50.0))
    gaps = RaceGapCalculator(session)

    assert gaps.interval_at_line("FRONT", "BACK", lap=1) == pytest.approx(10.0, abs=DT)
    assert gaps.interval_at_line("FRONT", "BACK", lap=1) != pytest.approx(500.0 / 55.56, abs=0.1)


def test_a_slow_field_reads_slow_and_a_fast_field_reads_fast():
    """The whole point: the interval follows the field's real speed.

    The same 500 m of track separation is 20 s under a Safety Car at
    25 m/s and 7.1 s at 70 m/s. A hardcoded 55.56 answers 9.0 s to both,
    over-reading the Safety Car case by more than half.
    """
    for speed, expected in ((25.0, 20.0), (70.0, 500.0 / 70.0)):
        session = _session(FRONT=_car(speed, head_start_m=500.0), BACK=_car(speed))

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


# --- The lapped car, where defect 2 was largest -----------------------------


def test_a_lapped_car_reads_as_a_lap_down_not_as_fabricated_seconds():
    """One circuit length of race distance behind is "+1 LAP" on any timing screen.

    This is the case cause 2 hit hardest: the old code added a whole
    circuit length to the distance AND divided it by 55.56, so on
    Melbourne it fabricated roughly 95 s per lap of difference on top of
    the real deficit.
    """
    session = _session(LEAD=_car(60.0), LAPPED=_car(25.0))
    gaps = RaceGapCalculator(session)
    idx = N_FRAMES - 1
    lead = ("LEAD", session.frames_by_driver["LEAD"][idx].dist)
    lapped = ("LAPPED", session.frames_by_driver["LAPPED"][idx].dist)
    laps_by_code = {code: session.frames_by_driver[code][idx].lap for code in ("LEAD", "LAPPED")}

    assert lead[1] - lapped[1] > CIRCUIT_LENGTH_M
    assert gaps.laps_down(lead[1], lapped[1]) >= 1

    label = DriverInfoPanel._gap_label("+", lead, lapped, gaps, laps_by_code)
    assert label.endswith("LAP") or label.endswith("LAPS")
    assert "s (L)" not in label


def test_a_lap_number_that_differs_by_one_is_not_being_lapped():
    """One car past the line and the other not is most of every lap, not a lapping.

    `laps_down` therefore reads race distance, not lap numbers: the two
    cars below are seconds apart and must read in seconds.
    """
    session = _session(FRONT=_car(50.0, head_start_m=200.0), BACK=_car(50.0))
    idx = 2450  # FRONT crossed at 96 s, BACK crosses at 100 s
    front_lap = session.frames_by_driver["FRONT"][idx].lap
    back_lap = session.frames_by_driver["BACK"][idx].lap
    gaps = RaceGapCalculator(session)

    assert front_lap - back_lap == 1, "the fixture must straddle a line crossing"
    assert (
        gaps.laps_down(
            session.frames_by_driver["FRONT"][idx].dist,
            session.frames_by_driver["BACK"][idx].dist,
        )
        == 0
    )


def test_cars_on_the_same_lap_read_in_seconds_and_say_they_are_at_the_line():
    """The "(L)" suffix is not decoration: it is what stops a lap-quantised
    number being read as a live one on a fidelity surface."""
    session = _session(FRONT=_car(50.0, head_start_m=300.0), BACK=_car(50.0))
    gaps = RaceGapCalculator(session)
    # Past 100 s, so BACK has crossed the line too and lap 1 is shared.
    idx = 3000
    front = ("FRONT", session.frames_by_driver["FRONT"][idx].dist)
    back = ("BACK", session.frames_by_driver["BACK"][idx].dist)
    laps_by_code = {code: session.frames_by_driver[code][idx].lap for code in ("FRONT", "BACK")}

    label = DriverInfoPanel._gap_label("+", front, back, gaps, laps_by_code)

    assert label.startswith("FRONT +")
    assert label.endswith("s (L)")
    assert float(label.split("+")[1].split("s")[0]) == pytest.approx(6.0, abs=2 * DT)


# --- Unknown stays unknown --------------------------------------------------


def test_an_interval_nobody_has_driven_yet_is_none():
    """Before both cars have finished a lap there is no interval to show."""
    session = _session(FRONT=_car(50.0, head_start_m=500.0), BACK=_car(50.0))
    gaps = RaceGapCalculator(session)

    assert gaps.interval_at_line("FRONT", "BACK", lap=0) is None
    assert gaps.interval_at_line("FRONT", "BACK", lap=99) is None
    assert gaps.interval_at_line("NOBODY", "BACK", lap=1) is None
    assert gaps.last_shared_lap(1, 1) == 0, "nobody has completed a lap on lap 1"
    assert gaps.last_shared_lap(21, 20) == 19, "the last lap BOTH have finished"


def test_a_retired_car_has_no_interval_past_the_lap_it_stopped_on():
    """Its crossings stop where its telemetry does; nothing extrapolates them."""
    session = _session(DEAD=_car(50.0, dead_from=100), ALIVE=_car(50.0))
    gaps = RaceGapCalculator(session)

    assert gaps.interval_at_line("ALIVE", "DEAD", lap=1) is None
    label = DriverInfoPanel._gap_label(
        "+", ("ALIVE", 6000.0), ("DEAD", 5000.0), gaps, {"ALIVE": 2, "DEAD": 2}
    )
    assert label == "ALIVE N/A"


def test_an_inverted_call_is_none_and_never_a_plausible_zero():
    """Zero is a real interval, two cars level, so it cannot double as "invalid".

    Measured on Melbourne 2025, an inverted call under the previous
    clamp-to-zero returned `0.000` on five of six sampled laps while the
    truth was 1 to 25 seconds the other way. This repo has a scar about
    sentinels that collide with legitimate values.
    """
    session = _session(FRONT=_car(50.0, head_start_m=500.0), BACK=_car(50.0))
    gaps = RaceGapCalculator(session)

    assert gaps.interval_at_line("FRONT", "BACK", lap=1) == pytest.approx(10.0, abs=DT)
    assert gaps.interval_at_line("BACK", "FRONT", lap=1) is None


def test_the_panel_labels_both_neighbours_from_its_own_position():
    session = _session(P1=_car(50.0, 1000.0), P2=_car(50.0, 500.0), P3=_car(50.0))
    gaps = RaceGapCalculator(session)
    idx = 3000  # all three have crossed the line once (P3, the last, at 100 s)
    panel = _panel("P2")

    ahead, behind = panel._neighbor_gaps(_ranked(session, idx), gaps, _frame(session, idx))

    assert ahead.startswith("P1 +") and ahead.endswith("s (L)")
    assert behind.startswith("P3 -") and behind.endswith("s (L)")
    assert float(ahead.split("+")[1].split("s")[0]) == pytest.approx(10.0, abs=2 * DT)
    assert float(behind.split("-")[1].split("s")[0]) == pytest.approx(10.0, abs=2 * DT)


def test_the_leader_and_the_last_car_have_no_neighbour_to_measure():
    session = _session(P1=_car(50.0, 500.0), P2=_car(50.0))
    gaps = RaceGapCalculator(session)
    idx = 3000

    leader = _panel("P1")
    last = _panel("P2")

    assert leader._neighbor_gaps(_ranked(session, idx), gaps, _frame(session, idx))[0] == "LEADER"
    assert last._neighbor_gaps(_ranked(session, idx), gaps, _frame(session, idx))[1] == "LAST"


# --- The crossings themselves -----------------------------------------------


def test_a_crossing_is_keyed_by_the_lap_it_ends():
    """Off by one here would put every interval on the wrong lap silently."""
    session = _session(SOLO=_car(50.0))  # 5000 m lap at 50 m/s = 100 s, 320 s of frames

    crossings = RaceGapCalculator(session)._crossings["SOLO"]

    assert sorted(crossings) == [1, 2, 3]
    for lap in (1, 2, 3):
        assert crossings[lap] == pytest.approx(lap * LAP_SECONDS, abs=DT)
