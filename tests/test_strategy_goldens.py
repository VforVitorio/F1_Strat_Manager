"""Golden tests for the strategy orchestrator's deterministic spine.

These freeze the thesis-defended numeric behaviour — the Layer-2 Monte Carlo
scoring (seed 42, n=500) and the Layer-1 MoE routing truth table — against the
P2b engine refactor (#169), which is about to rebuild this logic into
``src/strategy/inference/engine.py``. Same canned inputs + the same asserts must
survive that extraction, so this is the regression bed the engine is born against.

Import note: ``src.agents`` modules load model *configs* at import time (e.g.
``tire_agent`` reads ``routing_config.json`` when it builds its module-level
``CFG``), so these tests carry the same ``_skip_no_models`` guard as the other
agent tests. They run locally and in the data tier as a regression bed; they skip
on CI runners that lack ``data/models/``. The MC math itself needs no model
weights — only the import chain does — and numpy's ``default_rng(42)`` (PCG64) is
reproducible across numpy versions, so the frozen values are stable.

Unblocks the per-lap agent-call-count spy deferred in #180: once the P2b engine
exposes a single inference path, the spy asserts each ``run_*_from_state`` fires
once per lap against these same canned outputs.
"""

from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
_HAS_MODELS = (ROOT / "data" / "models" / "tire_degradation" / "routing_config.json").exists()
_skip_no_models = pytest.mark.skipif(
    not _HAS_MODELS,
    reason="data/models/ not present (CI runner without model weights)",
)


def _canned_outputs():
    """Four hand-built sub-agent outputs for a near-cliff pit-window lap.

    Only the fields the Monte Carlo layer reads matter numerically (pace CI,
    laps-to-cliff triangular, sc probability, pit-duration triangular, undercut
    probability); the rest are filled with plausible values so the dataclasses
    construct.
    """
    from src.agents.pace_agent import PaceOutput
    from src.agents.pit_strategy_agent import PitStrategyOutput
    from src.agents.race_situation_agent import RaceSituationOutput
    from src.agents.tire_agent import TireOutput

    pace = PaceOutput(
        lap_time_pred=91.0, delta_vs_prev=-0.2, delta_vs_median=0.3, ci_p10=90.5, ci_p90=91.8
    )
    tire = TireOutput(
        compound="MEDIUM",
        current_tyre_life=18,
        deg_rate=0.05,
        laps_to_cliff_p10=3.0,
        laps_to_cliff_p50=5.0,
        laps_to_cliff_p90=8.0,
        gp_name="",
    )
    situation = RaceSituationOutput(overtake_prob=0.2, sc_prob_3lap=0.10)
    pit = PitStrategyOutput(
        action="PIT_NOW",
        recommended_lap=20,
        compound_recommendation="HARD",
        stop_duration_p05=2.2,
        stop_duration_p50=2.8,
        stop_duration_p95=3.6,
        undercut_prob=0.55,
        undercut_target="VER",
        sc_reactive=False,
        reasoning="",
    )
    return pace, tire, situation, pit


# The exact MC output for the canned scenario at alpha=0.5 (seed 42, n=500),
# rounded to 3 decimals by _run_mc_simulation. This IS the thesis-defended math:
# any drift in simulate_lap_window or the sampling breaks this assert.
_GOLDEN_ALPHA_05 = {
    "STAY_OUT": {"E": -0.149, "P10": -0.529, "P90": 0.0, "score": -0.339},
    "PIT_NOW": {"E": -0.574, "P10": -1.332, "P90": -0.695, "score": -0.953},
    "UNDERCUT": {"E": 0.01, "P10": -1.217, "P90": 0.26, "score": -0.604},
    "OVERCUT": {"E": 0.893, "P10": 0.333, "P90": 0.333, "score": 0.613},
}


# ---------------------------------------------------------------------------
# Monte Carlo — Layer 2
# ---------------------------------------------------------------------------


@_skip_no_models
def test_mc_scores_match_the_frozen_golden():
    from src.agents.strategy_orchestrator import _run_mc_simulation

    pace, tire, situation, pit = _canned_outputs()
    result = _run_mc_simulation(pace, tire, situation, pit, alpha=0.5)
    assert result == _GOLDEN_ALPHA_05


@_skip_no_models
def test_mc_is_deterministic_across_calls():
    from src.agents.strategy_orchestrator import _run_mc_simulation

    pace, tire, situation, pit = _canned_outputs()
    first = _run_mc_simulation(pace, tire, situation, pit, alpha=0.5)
    second = _run_mc_simulation(pace, tire, situation, pit, alpha=0.5)
    assert first == second


@_skip_no_models
def test_alpha_1_reduces_score_to_expected_value():
    """score = alpha*E + (1-alpha)*P10, so alpha=1 must collapse score to E."""
    from src.agents.strategy_orchestrator import _run_mc_simulation

    pace, tire, situation, pit = _canned_outputs()
    result = _run_mc_simulation(pace, tire, situation, pit, alpha=1.0)
    for strategy, scores in result.items():
        assert scores["score"] == pytest.approx(scores["E"]), strategy


@_skip_no_models
def test_alpha_0_reduces_score_to_p10():
    """alpha=0 must collapse score to the worst-case P10 (pure risk aversion)."""
    from src.agents.strategy_orchestrator import _run_mc_simulation

    pace, tire, situation, pit = _canned_outputs()
    result = _run_mc_simulation(pace, tire, situation, pit, alpha=0.0)
    for strategy, scores in result.items():
        assert scores["score"] == pytest.approx(scores["P10"]), strategy


@_skip_no_models
def test_mc_returns_all_four_strategies_with_full_score_shape():
    from src.agents.strategy_orchestrator import _run_mc_simulation

    pace, tire, situation, pit = _canned_outputs()
    result = _run_mc_simulation(pace, tire, situation, pit, alpha=0.5)
    assert set(result) == {"STAY_OUT", "PIT_NOW", "UNDERCUT", "OVERCUT"}
    for scores in result.values():
        assert set(scores) == {"E", "P10", "P90", "score"}


@_skip_no_models
def test_mc_pit_out_none_falls_back_to_conservative_prior():
    """A missing PitStrategyOutput must not crash the MC (uses the built-in prior)."""
    from src.agents.strategy_orchestrator import _run_mc_simulation

    pace, tire, situation, _ = _canned_outputs()
    result = _run_mc_simulation(pace, tire, situation, None, alpha=0.5)
    assert set(result) == {"STAY_OUT", "PIT_NOW", "UNDERCUT", "OVERCUT"}


# ---------------------------------------------------------------------------
# MoE routing — Layer 1 (_decide_agents_to_call decides N28 + N30 only)
# ---------------------------------------------------------------------------


@_skip_no_models
@pytest.mark.parametrize(
    "tire_warning,sc_prob,alerts,sc_active,expected",
    [
        # quiet lap: no conditional agent fires
        ("OK", 0.05, [], False, set()),
        # near-cliff tyre → pit agent, which drags in the regulation check
        ("PIT_SOON", 0.05, [], False, {"N28", "N30"}),
        # high SC probability → regulation check only
        ("OK", 0.90, [], False, {"N30"}),
        # radio PROBLEM → unplanned-stop risk → pit + regulation
        ("OK", 0.05, [{"intent": "PROBLEM"}], False, {"N28", "N30"}),
        # RCM red-flag ruling (event_type) → FIA-facing → regulation only. Radio
        # transcripts only ever carry PROBLEM/WARNING intents (never a PENALTY
        # intent), so penalty/red-flag rulings reach the orchestrator as RCM
        # alerts with an event_type — the routing now keys on that (NR-04, #398).
        ("OK", 0.05, [{"event_type": "RED_FLAG"}], False, {"N30"}),
        # SC physically deployed (RCM-confirmed) → force pit + regulation
        ("OK", 0.05, [], True, {"N28", "N30"}),
    ],
)
def test_routing_truth_table(tire_warning, sc_prob, alerts, sc_active, expected):
    from src.agents.strategy_orchestrator import _decide_agents_to_call

    active = _decide_agents_to_call(tire_warning, sc_prob, alerts, sc_active)
    assert set(active) == expected
