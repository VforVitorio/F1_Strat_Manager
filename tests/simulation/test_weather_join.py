"""The replay serves the weather N04 trained on, joined by session time (W-F7).

`get_weather_state` used to pick its row by mapping the lap fraction onto the weather
frame's row index. That ignores session time, and neither side is evenly spaced: a Safety
Car stretches the gap between laps while the samples keep their own cadence, so the two
indices drift apart exactly when conditions are moving.

Measured over 79,032 driver-laps of the then-71-race tree, the proportional lookup
disagreed with
N04's join on **94.3%** of laps, mean 1.49 C and up to 17.3 C on TrackTemp, and **flipped
the rain flag on 3,399 laps**. N06's weather block is 39.7% of that model's gain.

The race count is left as measured rather than restated as 70: the 2023 Spanish GP was
de-duplicated on 2026-08-06 (#823) and this measurement predates it, so 71 is what the
run actually saw. Editing the scope of a number nobody re-ran is how a figure stops
matching its own evidence.

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


# --- the two branches the first version of this file never touched ------------


@pytest.mark.data
def test_a_second_weather_frame_is_not_served_the_first_ones_alignment():
    """The cache is keyed on the FRAME, not on `id()`, which a dead object releases.

    Reproduced on the second trial before the fix: read frame A, serve a lap, let A die,
    read a shifted frame B, and CPython hands B the same id, so B's lap came back with A's
    temperature. Silent, no error, no log. Today's callers hold one frame for the race so
    it could not fire, but the promise this method makes is per-ARGUMENT.
    """
    import gc

    from src.simulation.race_state_manager import RaceStateManager

    race_dir = _RAW / "Lusail"
    laps = pd.read_parquet(race_dir / "laps.parquet")
    manager = RaceStateManager(laps, "NOR", "McLaren", "Lusail", 2025)

    first = pd.read_parquet(race_dir / "weather.parquet")
    served_first = manager.get_weather_state(10, first)["track_temp"]
    del first
    gc.collect()

    second = pd.read_parquet(race_dir / "weather.parquet")
    second["TrackTemp"] = second["TrackTemp"] + 50.0
    served_second = manager.get_weather_state(10, second)["track_temp"]

    assert served_second == pytest.approx(served_first + 50.0), (
        f"the second frame was served {served_second} where {served_first + 50.0} is "
        f"correct: the cache is keyed on something a dead object can release"
    )


@pytest.mark.data
def test_a_lap_our_driver_never_ran_reports_no_weather_at_all():
    """After a retirement the replay keeps running; it must not invent a reading.

    This branch used to serve `weather_df.iloc[0]`, the session's FIRST sample, under a
    docstring calling it "a real gap rather than a substituted reading". Over 405 real
    fallback laps it was 2.67 C off on average and up to 8.9, with the rain flag wrong on
    92 of them: worse, on that territory, than the lookup this whole fix replaced.
    """
    from src.simulation.race_state_manager import RaceStateManager

    race_dir = _RAW / "Lusail"
    laps = pd.read_parquet(race_dir / "laps.parquet")
    weather = pd.read_parquet(race_dir / "weather.parquet")

    total = int(laps["LapNumber"].max())
    retired = [
        drv for drv, rows in laps.groupby("Driver") if int(rows["LapNumber"].max()) < total - 1
    ]
    assert retired, "no retirement in this race, so this test would hold vacuously"

    manager = RaceStateManager(laps, retired[0], "Unknown", "Lusail", 2025)
    state = manager.get_weather_state(total, weather)

    assert state["track_temp"] is None
    assert state["air_temp"] is None
