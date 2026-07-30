"""#728 — RaceStateManager must reconstruct N04's Prev_LapTime for a RAW frame.

The featured parquet carries a ``Prev_LapTime`` column. The RAW per-race parquet this
class is normally built from does not, and ``get_driver_state`` had no fallback, so it
emitted ``None`` on every lap. The pace agent then substituted ``90.0``, and since
``_predict`` uses that value as an ANCHOR — the prediction is ``prev + delta``, with no
NaN branch — every absolute lap-time prediction was pinned near 90 s regardless of
circuit.

Measured on the real raw parquets, median |anchor − this lap|:

    Monaco  78 laps   14.686 s  ->  0.480 s
    Monza   53 laps    6.889 s  ->  0.191 s
    Lusail  57 laps    4.955 s  ->  0.158 s

WHY THE TESTS BELOW LOOK THE WAY THEY DO
----------------------------------------
The naive fix — take ``lap_number - 1`` — is worse than the bug on the lap after a
stop. N04 computes the shift AFTER ``filter_baseline_laps``
(``IsAccurate & ~Deleted & LapTime_s < 180 & LapNumber > 1``) and groups by ``Stint``,
so the previous lap is the previous **surviving** one. An out-lap is excluded by
``IsAccurate``, and measured at Lusail 2025 NOR's out-lap is 107.6 s against 85.3 s on
the lap that follows: anchoring there would be 22 s wrong, against the 5 s the 90.0
default was wrong by at that circuit.

So most of these tests are about what must NOT become the anchor.

Hermetic: the frames are built here, since the cases that matter (an out-lap, a
stint boundary, a deleted lap) are structural and a real slice may contain none of
them. One data-gated test at the bottom checks the real raw schema, because a
synthetic frame cannot prove the column names match production.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.simulation.race_state_manager import RaceStateManager

ROOT = Path(__file__).parent.parent.parent
RACE_DIR = ROOT / "data" / "raw" / "2025" / "Monaco"

OUR_DRIVER = "NOR"


def _raw_frame(laps: list[dict]) -> pd.DataFrame:
    """A minimal RAW-schema frame: the required columns, and no ``Prev_LapTime``.

    ``LapTime`` and ``Time`` are timedeltas because ``validate_laps_df`` requires that
    of a raw frame — the featured parquet's float ``LapTime_s`` is a different schema
    and a different code path.
    """
    rows = []
    for lap in laps:
        rows.append(
            {
                "Driver": OUR_DRIVER,
                "Team": "McLaren",
                "LapNumber": float(lap["lap"]),
                "LapTime": pd.to_timedelta(lap["seconds"], unit="s"),
                "Time": pd.to_timedelta(lap["lap"] * 90.0, unit="s"),
                "Position": 1.0,
                "Compound": "MEDIUM",
                "TyreLife": float(lap.get("tyre_life", lap["lap"])),
                "TrackStatus": lap.get("status", "1"),
                "Stint": float(lap.get("stint", 1)),
                "IsAccurate": lap.get("accurate", True),
                "Deleted": lap.get("deleted", False),
            }
        )
    return pd.DataFrame(rows)


def _manager(frame: pd.DataFrame) -> RaceStateManager:
    return RaceStateManager(
        frame, driver_code=OUR_DRIVER, team="McLaren", gp_name="Monaco", year=2025
    )


# ---------------------------------------------------------------------------
# The reconstruction
# ---------------------------------------------------------------------------


def test_a_raw_frame_gets_the_previous_lap_instead_of_nothing():
    """The whole point: no Prev_LapTime column, and the value still arrives."""
    rsm = _manager(
        _raw_frame(
            [
                {"lap": 1, "seconds": 80.0},
                {"lap": 2, "seconds": 76.0},
                {"lap": 3, "seconds": 75.5},
                {"lap": 4, "seconds": 75.2},
            ]
        )
    )

    assert rsm.get_driver_state(3)["prev_lap_time"] == pytest.approx(76.0)
    assert rsm.get_driver_state(4)["prev_lap_time"] == pytest.approx(75.5)


def test_lap_one_and_the_lap_after_it_stay_unknown():
    """N04's filter drops ``LapNumber <= 1``, so lap 2 has no surviving predecessor.

    Reproduced rather than smoothed over: lap 1 of a race is a standing start and is
    not a lap the model was trained to read as "previous".
    """
    rsm = _manager(
        _raw_frame(
            [
                {"lap": 1, "seconds": 80.0},
                {"lap": 2, "seconds": 76.0},
                {"lap": 3, "seconds": 75.5},
            ]
        )
    )

    assert rsm.get_driver_state(1)["prev_lap_time"] is None
    assert rsm.get_driver_state(2)["prev_lap_time"] is None


def test_an_out_lap_never_becomes_the_anchor():
    """The trap that makes ``lap_number - 1`` worse than the bug.

    Lap 11 is the out-lap: a real one is ~22 s slower than the lap after it. It is
    excluded by ``IsAccurate``, so lap 12 has no surviving predecessor inside its own
    stint and stays unknown — which falls back to the pre-existing 90.0 rather than
    introducing a new, larger error.
    """
    rsm = _manager(
        _raw_frame(
            [
                {"lap": 9, "seconds": 75.5, "stint": 1, "tyre_life": 9},
                {"lap": 10, "seconds": 95.0, "stint": 1, "tyre_life": 10, "accurate": False},
                {"lap": 11, "seconds": 97.6, "stint": 2, "tyre_life": 1, "accurate": False},
                {"lap": 12, "seconds": 75.3, "stint": 2, "tyre_life": 2},
                {"lap": 13, "seconds": 75.1, "stint": 2, "tyre_life": 3},
            ]
        )
    )

    assert rsm.get_driver_state(12)["prev_lap_time"] is None, (
        "anchored on the out-lap: 22 s wrong, worse than the 90.0 it replaces"
    )
    assert rsm.get_driver_state(13)["prev_lap_time"] == pytest.approx(75.3)


def test_the_shift_does_not_cross_a_stint_boundary():
    """Even when both laps survive the filter, the previous stint is out of reach.

    N04 groups by ``Stint``, and the reason is measured rather than formal: the last
    lap of a stint is the in-lap, so it is slow in exactly the way an out-lap is.
    """
    rsm = _manager(
        _raw_frame(
            [
                {"lap": 5, "seconds": 75.5, "stint": 1, "tyre_life": 5},
                {"lap": 6, "seconds": 75.4, "stint": 1, "tyre_life": 6},
                {"lap": 7, "seconds": 75.0, "stint": 2, "tyre_life": 1},
                {"lap": 8, "seconds": 74.8, "stint": 2, "tyre_life": 2},
            ]
        )
    )

    assert rsm.get_driver_state(7)["prev_lap_time"] is None
    assert rsm.get_driver_state(8)["prev_lap_time"] == pytest.approx(75.0)


def test_a_deleted_lap_is_skipped_and_the_shift_reaches_past_it():
    """``Deleted`` is the other half of N04's filter, and the shift reaches back.

    This is the case that distinguishes "previous surviving lap" from "previous lap":
    lap 8's anchor is lap 6, not the deleted lap 7.
    """
    rsm = _manager(
        _raw_frame(
            [
                {"lap": 6, "seconds": 75.5},
                {"lap": 7, "seconds": 75.4, "deleted": True},
                {"lap": 8, "seconds": 75.2},
            ]
        )
    )

    assert rsm.get_driver_state(8)["prev_lap_time"] == pytest.approx(75.5)


def test_the_featured_column_wins_when_it_carries_a_value():
    """Featured-parquet callers must be untouched: their own column takes precedence."""
    frame = _raw_frame(
        [
            {"lap": 6, "seconds": 75.5},
            {"lap": 7, "seconds": 75.4},
            {"lap": 8, "seconds": 75.2},
        ]
    )
    frame["Prev_LapTime"] = [float("nan"), 61.0, 62.0]

    rsm = _manager(frame)

    assert rsm.get_driver_state(8)["prev_lap_time"] == pytest.approx(62.0)
    # A NaN in the column is not a value, so lap 6 falls through to the
    # reconstruction rather than being trusted. `Series.get` returns the stored NaN
    # rather than a default, which is the trap this branch exists to avoid.
    assert rsm.get_driver_state(6)["prev_lap_time"] is None


def test_a_frame_without_stint_yields_no_reconstruction_rather_than_a_wrong_one():
    """``Stint`` is not a required column, and without it the grouping is unknowable.

    An empty map is the honest answer; the consumer keeps its existing default. The
    alternative — group everything as one stint — would silently anchor across pit
    stops, which is the error this whole helper exists to avoid.
    """
    frame = _raw_frame([{"lap": 6, "seconds": 75.5}, {"lap": 7, "seconds": 75.4}])
    frame = frame.drop(columns=["Stint"])

    rsm = _manager(frame)

    assert rsm._derived_prev_lap == {}
    assert rsm.get_driver_state(7)["prev_lap_time"] is None


# ---------------------------------------------------------------------------
# The real raw schema — a synthetic frame cannot prove the column names match
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not (RACE_DIR / "laps.parquet").exists(),
    reason="needs data/raw/2025/Monaco/ (HF, not git)",
)
def test_the_real_monaco_parquet_gets_a_plausible_anchor_on_most_laps():
    """Monaco on purpose: it is where the 90.0 anchor is visible.

    Real Monaco pace is ~75 s, so the old default was ~15 s out on every lap. Lusail
    would be a false negative — its pace is ~85.7 s, so the same bug reads as ~5 s and
    could pass for noise.

    Asserts a band and a coverage floor, not values: a model or data refresh may move
    the numbers, but if the anchor stops landing near the real lap time the
    reconstruction has broken.
    """
    raw = pd.read_parquet(RACE_DIR / "laps.parquet")
    rsm = RaceStateManager(raw, OUR_DRIVER, "McLaren", gp_name="Monaco", year=2025)

    anchored, errors = 0, []
    for lap in range(1, rsm.total_laps + 1):
        state = rsm.get_driver_state(lap)
        if not state or state.get("lap_time_s") is None:
            continue
        previous = state.get("prev_lap_time")
        if previous is None:
            continue
        anchored += 1
        errors.append(abs(previous - state["lap_time_s"]))

    assert anchored >= 50, f"only {anchored} laps anchored; the reconstruction is not firing"
    # 8.0 is chosen against the data, not picked to pass. Measured worst anchor error
    # on the real parquets: Monaco 4.522, Silverstone 4.266, Melbourne 2.716, Monza
    # 0.884 — so this leaves ~1.8x headroom over the worst real lap while still
    # killing the mutant it exists for. A naive `lap n - 1` shift anchors the lap
    # after a stop on the out-lap, which is ~22 s out; the band this replaced was
    # 25.0 and let that mutant through, so the assertion passed for the wrong reason
    # until the S4 gate mutated it.
    assert max(errors) < 8.0, "an out-lap or a cross-stint lap leaked in as the anchor"
    assert sorted(errors)[len(errors) // 2] < 2.0, (
        "median anchor error above 2 s at Monaco, where consecutive green laps sit "
        "within a few tenths — the shift is reaching the wrong lap"
    )
