"""The replay serves the weather N04 trained on, joined by session time (W-F7).

`get_weather_state` used to pick its row by mapping the lap fraction onto the weather
frame's row index. That ignores session time, and neither side is evenly spaced: a Safety
Car stretches the gap between laps while the samples keep their own cadence, so the two
indices drift apart exactly when conditions are moving.

Measured over 79,032 driver-laps of all 71 races, the proportional lookup disagreed with
N04's join on **94.3%** of laps, mean 1.49 C and up to 17.3 C on TrackTemp, and **flipped
the rain flag on 3,399 laps**. N06's weather block is 39.7% of that model's gain.

These assert the EFFECT: what the replay serves must equal what N04 computed, to the cell.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).parent.parent.parent
_RAW = ROOT / "data" / "raw" / "2025"

# One dry evening race, one desert night race, and Las Vegas, whose weather frame is the
# one this repo has already been bitten by (its SpeedI2 trap is missing for the whole race).
_RACES = ("Shanghai", "Lusail", "Las_Vegas")
_HAS_DATA = all((_RAW / race / "weather.parquet").exists() for race in _RACES)

pytestmark = pytest.mark.skipif(not _HAS_DATA, reason="raw 2025 races absent")


@pytest.mark.data
@pytest.mark.parametrize("race", _RACES)
def test_the_served_weather_equals_n04s_join_cell_for_cell(race):
    """Not "close to". The alignment IS the trained contract."""
    from src.f1_strat_manager.weather_restore import weather_for_race
    from src.simulation.replay_engine import RaceReplayEngine

    race_dir = _RAW / race
    laps = pd.read_parquet(race_dir / "laps.parquet")
    weather = pd.read_parquet(race_dir / "weather.parquet")
    ours = laps[laps["Driver"] == "NOR"]
    truth = weather_for_race(ours, weather)

    engine = RaceReplayEngine(race_dir, "NOR", "McLaren", interval_seconds=0)

    compared = 0
    for state in engine.replay():
        row = ours[ours["LapNumber"] == state["lap_number"]]
        if row.empty:
            continue
        idx = row.index[0]
        for served_key, trained_col in (
            ("air_temp", "AirTemp"),
            ("track_temp", "TrackTemp"),
            ("humidity", "Humidity"),
        ):
            expected = truth.loc[idx, trained_col]
            served = state["weather"].get(served_key)
            if pd.isna(expected) or served is None:
                continue
            compared += 1
            assert served == pytest.approx(float(expected)), (
                f"{race} lap {state['lap_number']} {trained_col}: replay serves {served}, "
                f"N04's join says {expected}"
            )

    assert compared > 0, f"{race}: nothing was compared, so this held vacuously"


@pytest.mark.data
def test_the_rain_flag_follows_the_same_join():
    """The flag that used to flip on 3,399 laps, and the one a wrong value most changes."""
    from src.f1_strat_manager.weather_restore import weather_for_race
    from src.simulation.replay_engine import RaceReplayEngine

    race_dir = _RAW / "Shanghai"
    laps = pd.read_parquet(race_dir / "laps.parquet")
    weather = pd.read_parquet(race_dir / "weather.parquet")
    ours = laps[laps["Driver"] == "NOR"]
    truth = weather_for_race(ours, weather)

    engine = RaceReplayEngine(race_dir, "NOR", "McLaren", interval_seconds=0)

    compared = 0
    for state in engine.replay():
        row = ours[ours["LapNumber"] == state["lap_number"]]
        if row.empty:
            continue
        expected = truth.loc[row.index[0], "Rainfall"]
        if pd.isna(expected):
            continue
        compared += 1
        assert bool(state["weather"]["rainfall"]) == bool(expected), (
            f"lap {state['lap_number']}: replay says rain={state['weather']['rainfall']}, "
            f"N04's join says {bool(expected)}"
        )

    assert compared > 0, "no rain flags compared: this held vacuously"


@pytest.mark.data
def test_the_lookup_is_not_the_proportional_index_it_replaced():
    """Guard against a silent revert, by proving the two rules genuinely differ here.

    A test that only checks agreement with `weather_for_race` would still pass if someone
    restored the old lookup on a race where the two happen to coincide. This asserts that
    Shanghai is NOT such a race, so the test above is doing real work on it.
    """
    from src.f1_strat_manager.weather_restore import weather_for_race

    race_dir = _RAW / "Shanghai"
    laps = pd.read_parquet(race_dir / "laps.parquet")
    weather = pd.read_parquet(race_dir / "weather.parquet")
    ours = laps[laps["Driver"] == "NOR"]
    truth = weather_for_race(ours, weather)
    total = int(laps["LapNumber"].max())

    differing = 0
    for idx, lap in ours["LapNumber"].items():
        expected = truth.loc[idx, "TrackTemp"] if idx in truth.index else None
        if expected is None or pd.isna(expected):
            continue
        fraction = (int(lap) - 1) / max(total - 1, 1)
        old_row = weather.iloc[int(fraction * (len(weather) - 1))]
        if abs(float(old_row["TrackTemp"]) - float(expected)) > 1e-9:
            differing += 1

    assert differing > 0, (
        "the proportional index agrees with N04's join on every lap of this race, so the "
        "tests above would pass under the old rule too: pick a different race"
    )
