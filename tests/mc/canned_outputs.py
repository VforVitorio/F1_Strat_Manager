"""Shared canned sub-agent outputs for the MC/golden test tier (seed-42 regression bed).

Extracted 2026-08-01: four call sites (test_mc_state_helpers.py, test_strategy_goldens.py,
and their respective importers test_mc_is_a_real_decision.py / test_projection_golden.py)
either duplicated this exact function body or imported it from whichever of the first two
files happened to define it first -- two different import paths reaching two separately
maintained copies of the same fixture. A single source closes both problems at once.
"""

from __future__ import annotations


def canned_outputs():
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
