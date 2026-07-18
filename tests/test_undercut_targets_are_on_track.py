"""N28 must not score an undercut against a car that is no longer racing.

The bug that opened this epic: the pit wall recommended "undercut HUL" at Lusail lap 7,
having watched HUL crash on lap 6 and cause the Safety Car it was reacting to. That was
fixed in the rivals list, and N28 does not use the rivals list: it queries ``laps_df``
directly, so it kept its own copy of the bug.

Two distinct failures, and only one of them is loud:

* **HUL at lap 40**: his last row carries a NaN position, and ``pos_gap`` (N16's single
  strongest feature, gain 0.690) comes out NaN. LightGBM routes it down the missing
  branch and answers anyway.
* **BEA at lap 50**: he retired on lap 41, and ``_get_lap_row``'s unbounded prior-lap
  fallback hands back that complete row, position 19.0 and all. **No NaN, no warning**:
  a confident probability against a car in the garage, from 9-lap-stale telemetry.

No staleness threshold separates them. Measured across 2025, a car that FINISHED can
have its last known lap lag by 20 (RUS at Sakhir: the featured frame drops SC, pit and
out laps), while this bug fires at 9. The ranges overlap. The only sound signal is
presence: RaceStateManager builds ``rivals`` from the per-lap rows, so a car that is
gone is simply absent, which is the same answer a timing screen gives.
"""

from __future__ import annotations

from itertools import islice
from pathlib import Path

import pandas as pd
import pytest

RACE_DIR = Path("data/raw/2025/Lusail")
ROOT = Path(__file__).parent.parent
_HAS_MODELS = (ROOT / "data" / "models" / "tire_degradation" / "routing_config.json").exists()

# Importing N28 reads model configs at import time, and the fixture needs the raw
# parquet. data/ is pulled from Hugging Face, so the CI runner has neither.
pytestmark = pytest.mark.skipif(
    not (_HAS_MODELS and (RACE_DIR / "laps.parquet").exists()),
    reason="needs data/models/ and the raw Lusail parquet (data/ comes from HF, not git)",
)


@pytest.fixture(scope="module")
def agent_at_lap_50():
    """N28 wired exactly as ``run_from_state`` wires it, at Lusail lap 50."""
    from src.agents.pit_strategy_agent import PitStrategyAgent
    from src.simulation.replay_engine import RaceReplayEngine

    replay = RaceReplayEngine(RACE_DIR, driver_code="NOR", team="McLaren")
    lap_state = next(islice(replay.replay(), 49, 50))

    agent = PitStrategyAgent()
    agent.laps_df = pd.read_parquet(RACE_DIR / "laps.parquet")
    agent.laps_df["LapNumber"] = agent.laps_df["LapNumber"].astype(int)
    agent.session_meta = {"gp_name": "Lusail", "year": 2025, "total_laps": 57, "team_lookup": {}}
    agent._live_drivers = {r["driver"] for r in lap_state["rivals"] if r.get("driver")} | {"NOR"}
    return agent


def test_a_car_that_retired_is_absent_from_the_lap_state(agent_at_lap_50):
    """The premise: RSM already knows who is racing. HUL crashed L7, BEA retired L41."""
    live = agent_at_lap_50._live_drivers
    assert "HUL" not in live
    assert "BEA" not in live
    assert "PIA" in live, "the fixture is wrong if the leader is not on track"


@pytest.mark.parametrize("target", ["HUL", "BEA"])
def test_undercut_features_refuse_a_car_that_is_not_racing(agent_at_lap_50, target):
    """Refuse, do not default. A number here reads as a finding."""
    assert agent_at_lap_50._build_undercut_features("NOR", target, 50) is None


def test_a_live_rival_still_scores(agent_at_lap_50):
    """The guard must not buy safety by refusing everything."""
    features = agent_at_lap_50._build_undercut_features("NOR", "PIA", 50)
    assert features is not None
    assert not pd.isna(features["pos_gap"].iloc[0])


def test_the_position_default_cannot_mask_a_missing_value():
    """``Series.get(k, default)`` returns the STORED value, including NaN.

    The default only fires when the COLUMN is absent, never when the VALUE is. That is
    why five call sites believed they had a safety net that could not fire, and it is
    the #428 sentinel bug wearing a different column.
    """
    row = pd.Series({"Position": float("nan"), "Driver": "HUL"})
    assert pd.isna(row.get("Position", 10)), (
        "if this ever returns 10, pandas changed and the guards can be simplified"
    )


def test_the_orchestrator_llm_cannot_ship_a_retired_undercut_target():
    """N28's validated target wins; the LLM's free text only fills a gap, and only if live.

    `score_undercut_tool` checks its target against the cars on track. The orchestrator
    LLM then writes the SAME field from free text, and the prompt seeds it with a literal
    example ("e.g. SAI"). Preferring the LLM's value made that check dead code: the
    validated answer was computed and then overwritten by an unchecked string, which the
    arcade renders as "UCUT: <name>".
    """
    from itertools import islice

    from src.agents.strategy_orchestrator import _assemble_recommendation, _live_drivers_from
    from src.simulation.replay_engine import RaceReplayEngine
    from src.strategy.inference.no_llm import _deterministic_synthesis

    replay = RaceReplayEngine(RACE_DIR, driver_code="NOR", team="McLaren")
    live = _live_drivers_from(next(islice(replay.replay(), 49, 50)))
    mc_results = {"UNDERCUT": {"score": 0.4}}

    # HUL crashed on lap 7. The LLM names him anyway.
    synthesis = _deterministic_synthesis("UNDERCUT", None)
    synthesis.undercut_target = "HUL"
    rec = _assemble_recommendation(synthesis, None, mc_results, "", live_drivers=live)
    assert rec.undercut_target is None, "a car that crashed 43 laps ago reached the pit wall"

    # A car that is racing survives.
    synthesis_live = _deterministic_synthesis("UNDERCUT", None)
    synthesis_live.undercut_target = "PIA"
    rec_live = _assemble_recommendation(synthesis_live, None, mc_results, "", live_drivers=live)
    assert rec_live.undercut_target == "PIA"
