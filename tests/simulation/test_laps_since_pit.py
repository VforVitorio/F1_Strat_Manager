"""`laps_since_pit` is its own quantity, not the tyre's age (#800).

Inference passed `TyreLife` for N06's `LapsSincePitStop`. They are different things:
`TyreLife` is the age of the tyre SET and counts laps it ran before this race, so the
two coincide only when the set was fitted at the last stop. Measured against the trained
column they agreed on 100% of NOR's laps at Lusail and **20%** at Melbourne.

The rule is N01's, reproduced rather than approximated: `lap - max(pit laps strictly
before lap)`, falling back to `lap` while the driver has not stopped.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).parent.parent.parent
_RACES = ("Lusail", "Melbourne", "Monza", "Silverstone")
_HAS_DATA = (ROOT / "data" / "processed" / "laps_featured_2025.parquet").exists() and all(
    (ROOT / "data" / "raw" / "2025" / race / "laps.parquet").exists() for race in _RACES
)


def _manager(pit_laps: list[int], total: int = 10):
    """A RaceStateManager over a hand-built race where our driver pits on given laps."""
    from src.simulation.race_state_manager import RaceStateManager

    rows = []
    for lap in range(1, total + 1):
        rows.append(
            {
                "Driver": "NOR",
                "DriverNumber": "4",
                "LapNumber": lap,
                "LapTime_s": 90.0,
                "LapTime": pd.Timedelta(seconds=90),
                "Time": pd.Timedelta(seconds=90 * lap),
                "TrackStatus": "1",
                "Position": 1,
                "Compound": "MEDIUM",
                "TyreLife": lap,
                "Stint": 1,
                "PitInTime": pd.Timedelta(seconds=1) if lap in pit_laps else pd.NaT,
                "Team": "McLaren",
            }
        )
    return RaceStateManager(pd.DataFrame(rows), "NOR", "McLaren")


# --- the rule itself, hermetic -----------------------------------------------


def test_before_the_first_stop_it_counts_from_the_start():
    """N01's fallback: no previous stop means the lap number itself."""
    rsm = _manager(pit_laps=[])

    assert rsm.laps_since_pit(1) == 1
    assert rsm.laps_since_pit(7) == 7


def test_after_a_stop_it_counts_from_that_stop():
    rsm = _manager(pit_laps=[4])

    assert rsm.laps_since_pit(4) == 4, "the pit lap itself has no EARLIER stop"
    assert rsm.laps_since_pit(5) == 1
    assert rsm.laps_since_pit(9) == 5


def test_only_the_most_recent_stop_counts():
    rsm = _manager(pit_laps=[3, 7])

    assert rsm.laps_since_pit(6) == 3
    assert rsm.laps_since_pit(8) == 1


def test_a_rivals_stop_does_not_reset_our_counter():
    """The single-driver boundary: this is our car's pit history, nobody else's."""
    from src.simulation.race_state_manager import RaceStateManager

    rows = []
    for lap in range(1, 9):
        for drv in ("NOR", "VER"):
            rows.append(
                {
                    "Driver": drv,
                    "DriverNumber": "4" if drv == "NOR" else "1",
                    "LapNumber": lap,
                    "LapTime_s": 90.0,
                    "LapTime": pd.Timedelta(seconds=90),
                    "Time": pd.Timedelta(seconds=90 * lap),
                    "TrackStatus": "1",
                    "Position": 1 if drv == "NOR" else 2,
                    "Compound": "MEDIUM",
                    "TyreLife": lap,
                    "Stint": 1,
                    # Only VER stops.
                    "PitInTime": pd.Timedelta(seconds=1) if (drv == "VER" and lap == 3) else pd.NaT,
                    "Team": "McLaren" if drv == "NOR" else "Red Bull",
                }
            )
    rsm = RaceStateManager(pd.DataFrame(rows), "NOR", "McLaren")

    assert rsm.laps_since_pit(6) == 6, "VER's stop must not reset NOR's counter"


# --- the value that actually reaches the model -------------------------------


@pytest.mark.data
@pytest.mark.skipif(not _HAS_DATA, reason="featured parquet or raw races absent")
def test_the_emitted_value_matches_the_trained_column_and_tyre_life_does_not():
    """Both halves matter, and the second is why this change exists.

    Asserting only that the new value is right would pass just as happily if the old
    one had been right too, and would say nothing about whether the change was needed.
    Melbourne is the case that makes it concrete: `TyreLife` matched on 20% of laps
    there, so two thirds of the race fed N06 a number from a different quantity.
    """
    from src.f1_strat_manager.data_cache import get_data_root
    from src.simulation.replay_engine import RaceReplayEngine

    featured = pd.read_parquet(
        get_data_root() / "processed" / "laps_featured_2025.parquet",
        columns=["GP_Name", "Driver", "LapNumber", "LapsSincePitStop", "TyreLife"],
    )

    total = matched_new = matched_old = 0
    for race in _RACES:
        engine = RaceReplayEngine(
            get_data_root() / "raw" / "2025" / race, "NOR", "McLaren", interval_seconds=0
        )
        emitted = {
            state["lap_number"]: (
                state["driver"].get("laps_since_pit"),
                state["driver"].get("tyre_life"),
            )
            for state in engine.replay()
        }
        ours = featured[(featured.GP_Name == race) & (featured.Driver == "NOR")]
        for _, row in ours.iterrows():
            pair = emitted.get(int(row.LapNumber))
            if pair is None:
                continue
            total += 1
            matched_new += pair[0] == row.LapsSincePitStop
            matched_old += pair[1] == row.LapsSincePitStop

    assert total > 0, "no laps compared: this would hold vacuously"
    assert matched_new == total, (
        f"laps_since_pit matches the trained column on {matched_new}/{total} laps; "
        f"the whole point is that it reproduces N01's rule exactly"
    )
    assert matched_old < total, (
        "TyreLife matched the trained column everywhere, so this change fixed nothing "
        "on the measured races and the claim behind it needs re-checking"
    )
