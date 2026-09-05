"""
Agent import + contract smoke tests.

These tests verify that:
  1. Every src/agents/ module is importable
  2. The public entry-point functions exist with the expected signature
  3. Per-agent request schemas in strategy.py are correctly structured

No LLM calls are made — model loading is NOT triggered (lazy imports / singleton).
Run with:
    pytest tests/test_agents.py -v
"""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest

from tests.conftest import skip_no_tire_models as _skip_no_models

ROOT = Path(__file__).parent.parent.parent


# Guard: skip backend tests when telemetry backend is not installed.
_HAS_BACKEND = (ROOT / "src" / "telemetry" / "backend").exists()
_skip_no_backend = pytest.mark.skipif(
    not _HAS_BACKEND,
    reason="src/telemetry/backend not present",
)


# ---------------------------------------------------------------------------
# Import checks — verify module-level imports don't blow up
# ---------------------------------------------------------------------------


@_skip_no_models
@pytest.mark.parametrize(
    "module_path",
    [
        "src.agents.pace_agent",
        "src.agents.tire_agent",
        "src.agents.race_situation_agent",
        "src.agents.pit_strategy_agent",
        "src.agents.radio_agent",
        "src.agents.rag_agent",
        "src.agents.strategy_orchestrator",
    ],
)
def test_agent_module_importable(module_path):
    """Each agent module must import without errors."""
    mod = importlib.import_module(module_path)
    assert mod is not None


# ---------------------------------------------------------------------------
# Entry-point function existence
# ---------------------------------------------------------------------------
#
# One test, not seven. The six per-agent versions asserted that a name imported
# and was callable, which test_agent_module_importable above already covers for
# them: strategy_orchestrator imports all six entry points at module level, so
# renaming any one of them turns that test red on its own (measured, one rename
# at a time). The orchestrator's own entry point is the exception, because
# nothing imports it, so a rename there leaves the import test green and this is
# the only guard that catches it.


@_skip_no_models
def test_orchestrator_entry_points():
    from src.agents.strategy_orchestrator import RaceState, run_strategy_orchestrator_from_state

    assert callable(run_strategy_orchestrator_from_state)
    assert RaceState is not None


# ---------------------------------------------------------------------------
# Output dataclass fields
# ---------------------------------------------------------------------------


@_skip_no_models
def test_pace_output_fields():
    import dataclasses

    from src.agents.pace_agent import PaceOutput

    fields = {f.name for f in dataclasses.fields(PaceOutput)}
    assert {"lap_time_pred", "ci_p10", "ci_p90", "delta_vs_median", "reasoning"} <= fields


@_skip_no_models
def test_tire_output_fields():
    import dataclasses

    from src.agents.tire_agent import TireOutput

    fields = {f.name for f in dataclasses.fields(TireOutput)}
    assert {
        "laps_to_cliff_p10",
        "laps_to_cliff_p50",
        "laps_to_cliff_p90",
        "warning_level",
    } <= fields


@_skip_no_models
def test_race_situation_output_fields():
    import dataclasses

    from src.agents.race_situation_agent import RaceSituationOutput

    fields = {f.name for f in dataclasses.fields(RaceSituationOutput)}
    assert {"overtake_prob", "sc_prob_3lap", "threat_level"} <= fields


@_skip_no_models
def test_pit_output_fields():
    import dataclasses

    from src.agents.pit_strategy_agent import PitStrategyOutput

    fields = {f.name for f in dataclasses.fields(PitStrategyOutput)}
    assert {
        "stop_duration_p05",
        "stop_duration_p50",
        "stop_duration_p95",
        "undercut_prob",
    } <= fields


@_skip_no_models
def test_strategy_recommendation_fields():
    from src.agents.strategy_orchestrator import StrategyRecommendation

    # StrategyRecommendation is a Pydantic BaseModel, not a dataclass
    fields = set(StrategyRecommendation.model_fields.keys())
    assert {"action", "confidence", "reasoning", "scenario_scores"} <= fields


# ---------------------------------------------------------------------------
# Strategy endpoint Pydantic schemas
# ---------------------------------------------------------------------------


@_skip_no_backend
def test_strategy_schemas_importable():
    """Per-agent request models must exist in strategy.py (no StrategyRequest)."""
    import sys

    sys.path.insert(0, str(ROOT / "src" / "telemetry"))
    # StrategyRequest must no longer exist
    import backend.api.v1.endpoints.strategy as strategy_module

    assert not hasattr(strategy_module, "StrategyRequest"), (
        "StrategyRequest was not removed — split into per-agent schemas"
    )


@_skip_no_backend
def test_pace_request_schema():
    import sys

    sys.path.insert(0, str(ROOT / "src" / "telemetry"))
    from backend.api.v1.endpoints.strategy import PaceRequest

    req = PaceRequest(lap_state={"driver": {}, "session_meta": {}})
    assert req.lap_state == {"driver": {}, "session_meta": {}}


@_skip_no_backend
def test_radio_request_defaults():
    """RadioRequest must default radio_msgs and rcm_events to empty lists."""
    import sys

    sys.path.insert(0, str(ROOT / "src" / "telemetry"))
    from backend.api.v1.endpoints.strategy import RadioRequest

    req = RadioRequest(lap_state={})
    assert req.radio_msgs == []
    assert req.rcm_events == []


@_skip_no_backend
def test_tire_request_defaults():
    import sys

    sys.path.insert(0, str(ROOT / "src" / "telemetry"))
    from backend.api.v1.endpoints.strategy import TireRequest

    req = TireRequest(lap_state={"driver": {}})
    assert "driver" in req.lap_state


# ---------------------------------------------------------------------------
# RCM context override (SC-active patch) — tests for the post-hoc safeguard
# that flips N27/N28/N31 routing when an RCM confirms a deployed Safety Car.
# ---------------------------------------------------------------------------


@_skip_no_models
def test_race_situation_sc_override_pure():
    """The pure helper recognises SC deploy, dict shape, and release-wins ordering."""
    from src.agents.race_situation_agent import _sc_active_from_rcm
    from src.agents.radio_agent import RCMEvent

    assert _sc_active_from_rcm([]) is False
    assert _sc_active_from_rcm(None) is False

    sc_ev = RCMEvent(message="SAFETY CAR DEPLOYED", flag="", category="SafetyCar", lap=7)
    assert _sc_active_from_rcm([sc_ev]) is True

    end_ev = RCMEvent(message="SAFETY CAR ENDING", flag="", category="SafetyCar", lap=7)
    # A lap carrying the SC-ending message is still neutralised: the car returns at the
    # end of that lap and overtaking stays banned until a driver passes the Line after it
    # returns (Art. 55.15 + 55.8). The flag clears the next lap, not on the release lap.
    assert _sc_active_from_rcm([sc_ev, end_ev]) is True

    # Raw FastF1-shaped dict is also accepted (auto-classified inside the helper).
    assert (
        _sc_active_from_rcm(
            [
                {
                    "message": "VIRTUAL SAFETY CAR DEPLOYED",
                    "flag": "",
                    "category": "SafetyCar",
                    "lap": 7,
                }
            ]
        )
        is True
    )


@_skip_no_models
def test_race_situation_output_has_sc_active_field():
    """The dataclass exposes sc_currently_active and bumps threat_level to HIGH."""
    import dataclasses

    from src.agents.race_situation_agent import RaceSituationOutput

    fields = {f.name for f in dataclasses.fields(RaceSituationOutput)}
    assert "sc_currently_active" in fields

    out = RaceSituationOutput(overtake_prob=0.1, sc_prob_3lap=0.05, sc_currently_active=True)
    assert out.threat_level == "HIGH"


@_skip_no_models
def test_pit_prompt_sc_deployed_banner():
    """The pit prompt must replace the 'SC probability' line with the deploy banner."""
    from src.agents.pit_strategy_agent import _build_pit_prompt

    prompt = _build_pit_prompt(
        driver="PIA",
        lap_number=7,
        tyre_life=6,
        compound="MEDIUM",
        team="McLaren",
        position=3,
        rival_str="VER",
        sc_prob=0.10,
        laps_to_cliff_p10=15.0,
        sc_currently_active=True,
    )
    assert "SAFETY CAR DEPLOYED RIGHT NOW" in prompt
    assert "SC probability (next 3 laps)" not in prompt


@_skip_no_models
def test_orchestrator_routing_sc_active_forces_n28():
    """sc_currently_active must force both N28 (pit) and N30 (RAG) into the active set."""
    from src.agents.strategy_orchestrator import _decide_agents_to_call

    active = _decide_agents_to_call(
        tire_warning="OK",
        sc_prob_3lap=0.05,
        radio_alerts=[],
        sc_currently_active=True,
    )
    assert "N28" in active
    assert "N30" in active
