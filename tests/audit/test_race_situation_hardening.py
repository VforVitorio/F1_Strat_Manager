"""Race Situation Agent (N27) hardening — #450 tuned thresholds, #476 unvalidated LLM input.

Two independent bugs closed here, both silent (no exception, just a wrong or
unused number):

- **#450**: ``RaceSituationConfig.high_overtake``/``high_sc`` were dataclass
  literals (0.80 / 0.30) that ``__post_init__`` loaded the REAL tuned thresholds
  (``overtake_threshold`` from N12's ``model_config.json``, ``sc_threshold`` from
  N14's ``feature_list_v1.json``) right next to, and then never used. Every
  ``threat_level`` ever computed by ``RaceSituationOutput.__post_init__`` was
  thresholded against the untuned placeholder — most visibly for SC, where the
  hardcoded 0.30 sat nowhere near the tuned 0.2335.
- **#476**: ``predict_overtake_tool``/``predict_sc_tool`` take free-text driver
  codes and a free-int lap number straight from the LLM. The only guard was an
  empty-dataframe check, which does not catch a driver who is not racing THIS
  lap but still has stale rows elsewhere in ``laps_df`` (the FastF1 path loads
  the WHOLE session), nor a lap number beyond the one the agent was actually
  loaded for.

Doctrine: no-LLM assertions only — real Lusail 2025 parquet, the actual loaded
model config, no LangGraph ReAct / live orchestrator run anywhere below.
"""

from __future__ import annotations

import json
from itertools import islice
from pathlib import Path

import pandas as pd
import pytest

RACE_DIR = Path("data/raw/2025/Lusail")
ROOT = Path(__file__).parent.parent.parent

_HAS_OVERTAKE_MODEL = (
    ROOT / "data" / "models" / "overtake_probability" / "model_config.json"
).exists()
_HAS_SC_MODEL = (
    ROOT / "data" / "models" / "safety_car_probability" / "feature_list_v1.json"
).exists()
_HAS_RACE_DATA = (RACE_DIR / "laps.parquet").exists()

# Importing race_situation_agent builds the module-level CFG singleton at import
# time (loads both LightGBM models + calibrators + the circuit parquets), and the
# fixture below needs the raw Lusail parquet. data/ comes from Hugging Face, not
# git, so CI runners without it must skip rather than fail on import.
pytestmark = [
    pytest.mark.data,
    pytest.mark.skipif(
        not (_HAS_OVERTAKE_MODEL and _HAS_SC_MODEL and _HAS_RACE_DATA),
        reason="needs data/models/{overtake_probability,safety_car_probability}/ "
        "and the raw Lusail parquet (data/ comes from HF, not git)",
    ),
]


def test_thresholds_loaded_from_model_config_not_hardcoded():
    """#450: CFG.high_overtake/high_sc must be the TUNED thresholds, not 0.80/0.30.

    Reads the two on-disk configs independently of the agent module (so the test
    does not just compare the singleton against itself) and checks CFG matches
    what was actually written to disk by N12/N14 — and does NOT match the old
    dataclass literals, which is exactly the bug: they were never overwritten.
    """
    from src.agents.race_situation_agent import CFG

    with open(ROOT / "data" / "models" / "overtake_probability" / "model_config.json") as f:
        ov_cfg = json.load(f)
    with open(ROOT / "data" / "models" / "safety_car_probability" / "feature_list_v1.json") as f:
        sc_cfg = json.load(f)

    loaded_overtake_threshold = ov_cfg["optimal_threshold"]
    loaded_sc_threshold = sc_cfg["best_threshold"]

    assert CFG.high_overtake == pytest.approx(loaded_overtake_threshold)
    assert CFG.high_sc == pytest.approx(loaded_sc_threshold)

    # The pre-fix hardcoded literals. If CFG ever equals these again while the
    # loaded config differs, #450 has regressed (the loaded value stopped being
    # the one threat_level actually thresholds against).
    if loaded_overtake_threshold != pytest.approx(0.80):
        assert CFG.high_overtake != pytest.approx(0.80)
    if loaded_sc_threshold != pytest.approx(0.30):
        assert CFG.high_sc != pytest.approx(0.30)


@pytest.fixture(scope="module")
def agent_at_lap_50():
    """RaceSituationAgent wired exactly as run_from_state wires it, Lusail lap 50.

    Reuses the same real-bug scenario as tests/mc/test_undercut_targets_are_on_track.py:
    HUL crashed on lap 7, so he is absent from the RaceStateManager ``rivals`` this
    replay actually produces at lap 50 — a real retired-driver case, not a made-up one.
    """
    from src.agents.race_situation_agent import RaceSituationAgent, _ensure_timedelta_laps
    from src.simulation.replay_engine import RaceReplayEngine

    replay = RaceReplayEngine(RACE_DIR, driver_code="NOR", team="McLaren")
    lap_state = next(islice(replay.replay(), 49, 50))

    agent = RaceSituationAgent()
    agent.laps_df = _ensure_timedelta_laps(pd.read_parquet(RACE_DIR / "laps.parquet"))
    agent.laps_df["LapNumber"] = agent.laps_df["LapNumber"].astype(int)
    agent.session_meta = {
        "session": None,
        "gp_name": "Lusail",
        "event_name": "Lusail",
        "year": 2025,
        "circuit_cluster": 0,
        "circuit_sc_rate": 0.10,
        "total_laps": 57,
        "AirTemp": 28.0,
        "TrackTemp": 38.0,
        "Humidity": 50.0,
        "track_temp_start": 38.0,
    }
    agent._live_drivers = {r["driver"] for r in lap_state["rivals"] if r.get("driver")} | {"NOR"}
    agent._current_lap = 50
    return agent


def _tool(agent, name: str):
    """Look up one of the agent's built LangChain tools by name."""
    tools = {t.name: t for t in agent._tools}
    if name not in tools:
        pytest.skip("LangGraph/LangChain not installed — _build_tools() returned no tools")
    return tools[name]


def test_a_car_that_retired_is_absent_from_the_lap_state(agent_at_lap_50):
    """The premise the guard relies on: HUL crashed on Lusail lap 7."""
    assert "HUL" not in agent_at_lap_50._live_drivers
    assert "PIA" in agent_at_lap_50._live_drivers, "fixture is wrong if the leader is not racing"


def test_predict_overtake_tool_refuses_a_car_not_on_track(agent_at_lap_50):
    """#476: an impossible pair (one driver retired 43 laps ago) must error, not score.

    Mirrors the task's example (predict_overtake_tool('VER','HAM', lap=12) when they
    are not racing each other) with the real retired-driver case this repo already
    has a verified fixture for.
    """
    result = _tool(agent_at_lap_50, "predict_overtake_tool").invoke(
        {"driver_x": "NOR", "driver_y": "HUL", "lap_number": 50}
    )
    assert "REFUSED" in result, f"expected a refusal, got: {result}"
    assert "HUL" in result


def test_predict_overtake_tool_refuses_a_future_lap(agent_at_lap_50):
    """#476: a lap far beyond the one the agent was actually loaded for must error."""
    result = _tool(agent_at_lap_50, "predict_overtake_tool").invoke(
        {"driver_x": "NOR", "driver_y": "PIA", "lap_number": 9999}
    )
    assert "REFUSED" in result, f"expected a refusal, got: {result}"


def test_predict_sc_tool_refuses_a_future_lap(agent_at_lap_50):
    """#476: predict_sc_tool takes only a lap number — it needs the same lap-range guard."""
    result = _tool(agent_at_lap_50, "predict_sc_tool").invoke({"lap_number": 9999})
    assert "REFUSED" in result, f"expected a refusal, got: {result}"


def test_a_live_pair_still_scores(agent_at_lap_50):
    """The guard must not buy safety by refusing everything — a real pair still answers."""
    result = _tool(agent_at_lap_50, "predict_overtake_tool").invoke(
        {"driver_x": "NOR", "driver_y": "PIA", "lap_number": 50}
    )
    assert "REFUSED" not in result, f"a live, in-range pair was refused: {result}"
    assert "P(overtake)" in result
