"""RaceStateManager lap_state contract tests, on the committed mini race fixture.

Pins the single most load-bearing architectural constraint of the thesis: the
**single-driver perspective**. Our driver gets full car + timing telemetry; every
rival gets timing-screen-only data (what a real pit wall sees). Every agent and
both UIs consume this ``lap_state`` shape, so freezing it here guards the whole
system against a silent boundary leak.

Hermetic: ``RaceStateManager`` imports with no model weights, and the fixture
(``tests/fixtures/mini_race.parquet``, a 9-lap Qatar 2025 slice over 6 drivers) is
committed, so this runs on CI runners. Laps 7-10 sit under the Safety Car, so the
fixture also exercises the SC track-status the Qatar V7 case study depends on.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.simulation.race_state_manager import RaceStateManager

FIXTURE = Path(__file__).parent.parent / "fixtures" / "mini_race.parquet"

OUR_DRIVER = "VER"
SC_LAP = 9  # inside the Safety Car window (laps 7-10)
FIRST_LAP, LAST_LAP = 5, 13

# Fields that only the car's own data link + full timing exposes.
_RICH_ONLY_FIELDS = {"sector1_s", "sector2_s", "sector3_s", "fuel_load", "compound_id", "speed_i1"}


@pytest.fixture(scope="module")
def rsm():
    df = pd.read_parquet(FIXTURE)
    team = df.loc[df["Driver"] == OUR_DRIVER, "Team"].iloc[0]
    return RaceStateManager(df, driver_code=OUR_DRIVER, team=team, gp_name="Lusail", year=2025)


def test_total_laps_matches_the_fixture_window(rsm):
    assert rsm.total_laps == LAST_LAP


def test_lap_state_has_the_canonical_structure(rsm):
    lap_state = rsm.get_lap_state(SC_LAP)
    assert {"driver", "rivals", "weather", "session_meta"} <= lap_state.keys()
    assert isinstance(lap_state["rivals"], list)


def test_driver_gets_full_telemetry(rsm):
    """Our driver exposes the rich fields (sector times, fuel, compound id)."""
    driver = rsm.get_lap_state(SC_LAP)["driver"]
    assert _RICH_ONLY_FIELDS <= driver.keys()
    assert driver["driver"] == OUR_DRIVER


def test_rivals_get_timing_screen_only(rsm):
    """The single-driver boundary: rivals never carry the rich fields."""
    rivals = rsm.get_lap_state(SC_LAP)["rivals"]
    assert rivals, "expected rivals in the fixture window"
    for rival in rivals:
        leaked = _RICH_ONLY_FIELDS & rival.keys()
        assert not leaked, f"rival {rival.get('driver')} leaked rich fields: {leaked}"
        # but they DO carry the timing-screen essentials
        assert {"position", "gap_to_leader_s", "tyre_life", "compound"} <= rival.keys()


def test_our_driver_is_never_among_the_rivals(rsm):
    rivals = rsm.get_lap_state(SC_LAP)["rivals"]
    assert OUR_DRIVER not in {r.get("driver") for r in rivals}


def test_safety_car_lap_is_reflected_in_track_status(rsm):
    """Lap 9 sits under the SC (TrackStatus '4') — the Qatar V7 case signal."""
    driver = rsm.get_lap_state(SC_LAP)["driver"]
    assert driver["track_status"] == "4"


def test_edge_laps_resolve_and_out_of_range_is_empty(rsm):
    assert rsm.get_lap_state(FIRST_LAP)["driver"]
    assert rsm.get_lap_state(LAST_LAP)["driver"]
    # A lap past the race end yields an empty driver state (race-ended signal).
    assert rsm.get_lap_state(999)["driver"] == {}
