"""Race Situation Agent (N27) hardening — #665 threat bands, #476 unvalidated LLM input.

Two independent bugs closed here, both silent (no exception, just a wrong or
unused number):

- **#665**: ``RaceSituationConfig`` compared the CALIBRATED probabilities against
  thresholds tuned on the RAW model output. N14 tunes ``best_threshold`` on
  ``proba_test`` (cell 20, ``m.predict_proba(X_test)[:,1]``) and only calibrates in
  cell 32; N12 does the same in cells 22/25/26 vs cell 36. #450 wired those raw
  operating points onto ``high_overtake``/``high_sc``, which made both bands
  unreachable — measured over real 2025 laps, SC HIGH fired 0/1420 and overtake
  HIGH 0/8171, so the SC model contributed nothing to ``threat_level`` at any
  level. The bands are now pit-wall alert levels set on the served calibrated
  scale, kept separate from the classifier operating points.
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


def test_bands_are_never_the_raw_classifier_operating_points():
    """#665: the threat bands must NOT be the tuned thresholds, which are raw-scale.

    #450 assigned overtake_threshold/sc_threshold onto high_overtake/high_sc. Both
    are tuned on the RAW model output (N14 cell 20/23, N12 cells 22/25/26) while
    threat_level compares the CALIBRATED probability, so neither could ever fire.
    Reads the on-disk configs independently of the agent module so the test cannot
    pass by comparing the singleton against itself.
    """
    from src.agents.race_situation_agent import CFG

    with open(ROOT / "data" / "models" / "overtake_probability" / "model_config.json") as f:
        ov_cfg = json.load(f)
    with open(ROOT / "data" / "models" / "safety_car_probability" / "feature_list_v1.json") as f:
        sc_cfg = json.load(f)

    assert CFG.high_overtake != pytest.approx(ov_cfg["optimal_threshold"])
    assert CFG.high_sc != pytest.approx(sc_cfg["best_threshold"])

    # The loads themselves stay — they are exported for anyone who wants to
    # binarise a RAW score — but they must remain separate from the bands.
    assert CFG.overtake_threshold == pytest.approx(ov_cfg["optimal_threshold"])
    assert CFG.sc_threshold == pytest.approx(sc_cfg["best_threshold"])


def test_every_band_is_reachable_on_the_calibrated_scale():
    """#665: a band above the calibrator's ceiling is a constant False, not a band.

    The pre-fix high_overtake (0.7976) sat above what the Platt calibrator can
    emit at raw=1.0, so overtake could never reach HIGH by construction. Assert the
    property directly: push raw 1.0 through each calibrator and require every band
    to sit strictly below that ceiling.
    """
    import numpy as np

    from src.agents.race_situation_agent import CFG

    overtake_ceiling = float(CFG.overtake_calibrator.predict_proba(np.array([[1.0]]))[:, 1][0])
    sc_ceiling = float(CFG.sc_calibrator.predict_proba(np.array([[1.0]]))[:, 1][0])

    for band, ceiling, name in (
        (CFG.high_overtake, overtake_ceiling, "high_overtake"),
        (CFG.medium_overtake, overtake_ceiling, "medium_overtake"),
        (CFG.high_sc, sc_ceiling, "high_sc"),
        (CFG.medium_sc, sc_ceiling, "medium_sc"),
    ):
        assert band < ceiling, (
            f"{name}={band} is at or above the calibrator ceiling {ceiling:.4f} — "
            "no calibrated probability can ever reach it"
        )


def test_sc_bands_track_the_models_base_rate():
    """#665: high_sc/medium_sc are defined as multiples of N14's own base rate.

    N14 is too weak (AUC-PR 0.072, lift 1.67x) for its absolute probability to mean
    much, so the bands are anchored on its base rate instead: MEDIUM at 1x, HIGH at
    2x. Retraining N14 moves that baseline, and this test is what makes the bands
    move with it rather than silently drifting into meaninglessness.
    """
    from src.agents.race_situation_agent import CFG

    with open(ROOT / "data" / "models" / "safety_car_probability" / "feature_list_v1.json") as f:
        sc_cfg = json.load(f)
    base_rate = sc_cfg["target_comparison"]["3-lap"]["baseline"]

    assert CFG.medium_sc == pytest.approx(base_rate, abs=1e-4)
    assert CFG.high_sc == pytest.approx(2 * base_rate, abs=1e-4)


def test_bands_are_ordered():
    """A MEDIUM band above its HIGH band makes threat_level unreachable at the top."""
    from src.agents.race_situation_agent import CFG

    assert CFG.medium_overtake < CFG.high_overtake
    assert CFG.medium_sc < CFG.high_sc


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
