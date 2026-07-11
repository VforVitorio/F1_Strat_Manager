"""Shared strategy inference engine — one lap, one code path, three surfaces.

Why this module exists
----------------------
Before this, three places ran the N31 pipeline as hand-mirrored body copies:
``src/arcade/strategy_pipeline.py`` (verbose, returns agent outputs),
``scripts/run_simulation_cli.py`` (probe + orchestrator, runs every agent twice),
and the backend simulator's ``_run_no_llm_path``. The audit (AUDIT_P2B_CORE_COMPUTE
F1/F10) showed this drift is a real bug source: the ``_run_conditional_agents``
3-tuple change (commit bfe5b46) was mirrored into arcade but not the CLI, so every
``--no-llm`` lap crashes (#166). This module is the single additive home the audit
recommends (§7): CLI/Arcade/backend consume one ``run_lap`` instead of three copies.

Design (per documents/audits/P2B_ENGINE_DESIGN.md, #169 Phases 1.1 + 1.2)
--------------------------------------------------------------------------
``run_lap`` dispatches on ``profile``:
  * ``rich``   — reproduces ``run_strategy_orchestrator_from_state`` byte-for-byte
                 (same ``action`` / ``scenario_scores``) by IMPORTING the exact
                 orchestrator layer functions and re-driving their five-step
                 sequence, but RETURNS the per-agent outputs the orchestrator's
                 public API discards. This is arcade's proven pattern, promoted.
  * ``no-llm`` — the deterministic, zero-LLM-client path (see ``no_llm.py``); fixes
                 #166 by construction (it never calls ``_run_conditional_agents``).

Untouchability: nothing in ``src/agents/`` is modified. Every strategy layer is the
SAME code object the orchestrator runs (imported, never copied); the only
engine-owned code is the call sequence itself and the default-lap_state builder.

Anti-drift guard: ``tests/test_engine_parity.py`` asserts the engine's rich output
equals the orchestrator's on a fixture lap, down to the byte-level LLM prompts.
"""

from __future__ import annotations

import time
from typing import Any, Literal

import pandas as pd

from src.agents.strategy_orchestrator import (
    RaceState,
    StrategyRecommendation,
    _assemble_recommendation,
    _build_orchestrator_prompt,
    _decide_agents_to_call,
    _get_orchestrator_llm,
    _run_always_on_agents_from_state,
    _run_conditional_agents,
    _run_mc_simulation,
)

# The profiles #169 delivers. ``fast`` (direct-mode sub-agents + event-triggered
# N31, audit F3/F11) is a later phase and is rejected with a pointing error so
# consumers can already write a three-valued switch without silent fallthrough.
PROFILES: tuple[str, ...] = ("rich", "no-llm")

Profile = Literal["rich", "no-llm"]


class _StageTimer:
    """``perf_counter`` context manager that records one stage's wall time.

    Kept tiny and side-effect-local: it only writes its own key into the shared
    ``timings`` dict on exit, so a stage's cost is captured even if it raises.
    """

    def __init__(self, timings: dict[str, float], key: str) -> None:
        self._timings = timings
        self._key = key

    def __enter__(self) -> "_StageTimer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc: object) -> None:
        self._timings[self._key] = time.perf_counter() - self._start


def run_lap(
    race_state: RaceState,
    laps_df: pd.DataFrame,
    lap_state: dict[str, Any] | None = None,
    *,
    profile: Profile = "rich",
    return_agent_outputs: bool = True,
) -> tuple[StrategyRecommendation, dict[str, Any] | None, dict[str, float]]:
    """Run one lap of the N31 strategy pipeline and return everything a surface needs.

    Args:
        race_state: The orchestrator's Pydantic ``RaceState`` (unchanged).
        laps_df: Full laps frame (featured parquet slice or an RSM-fed frame).
        lap_state: RSM-shaped per-lap dict. ``None`` builds the minimal default
            (``_build_default_lap_state``), matching the orchestrator's own inline
            fallback so callers that only have a ``race_state`` still work.
        profile: ``"rich"`` (LLM synthesis, full fidelity) or ``"no-llm"``
            (deterministic, zero LLM clients). ``"fast"`` is reserved (raises).
        return_agent_outputs: When ``False``, slot 2 is ``None`` (compute is
            unchanged; only the assembly of the outputs dict is skipped).

    Returns:
        ``(recommendation, agent_outputs | None, stage_timings)`` where
        ``recommendation`` is always a real ``StrategyRecommendation`` (never an
        ad-hoc dict), ``agent_outputs`` carries the six sub-agent dataclasses +
        routing/RAG context (see ``_assemble_agent_outputs``), and
        ``stage_timings`` maps each stage name to its seconds.

    Raises:
        ValueError: on an unknown profile, or on the reserved ``"fast"`` profile.
    """
    if profile == "rich":
        return _run_rich(race_state, laps_df, lap_state, return_agent_outputs)
    if profile == "no-llm":
        # Imported lazily so the rich path never pays the no_llm module's agent
        # class imports, and so a circular import can never form.
        from src.strategy.inference.no_llm import run_no_llm_lap

        return run_no_llm_lap(race_state, laps_df, lap_state, return_agent_outputs)
    if profile == "fast":
        raise ValueError("profile 'fast' is P2b Phase 2 (audit F3/F11) — use 'rich' or 'no-llm'")
    raise ValueError(f"unknown profile {profile!r}; expected one of {PROFILES}")


def _run_rich(
    race_state: RaceState,
    laps_df: pd.DataFrame,
    lap_state: dict[str, Any] | None,
    return_agent_outputs: bool,
) -> tuple[StrategyRecommendation, dict[str, Any] | None, dict[str, float]]:
    """The rich profile: the orchestrator's five-step sequence, outputs retained.

    Statement for statement this mirrors ``run_strategy_orchestrator_from_state``
    (always-on agents -> routing -> conditional agents -> Monte Carlo -> LLM
    synthesis), but every step is an IMPORTED orchestrator function, so the result
    is byte-identical to the orchestrator's while the intermediate agent outputs
    are returned instead of discarded.
    """
    timings: dict[str, float] = {}
    if lap_state is None:
        lap_state = _build_default_lap_state(race_state, laps_df)

    with _StageTimer(timings, "always_on"):
        pace_out, tire_out, situation_out, radio_out = _run_always_on_agents_from_state(
            race_state, laps_df, lap_state
        )

    with _StageTimer(timings, "routing"):
        active = _decide_agents_to_call(
            tire_warning=tire_out.warning_level,
            sc_prob_3lap=situation_out.sc_prob_3lap,
            radio_alerts=radio_out.alerts,
            sc_currently_active=situation_out.sc_currently_active,
        )

    with _StageTimer(timings, "conditional"):
        pit_out, regulation_context, rag_dict = _run_conditional_agents(
            active=active,
            lap_state=lap_state,
            tire_out=tire_out,
            situation_out=situation_out,
            race_state=race_state,
            laps_df=laps_df,
        )
        regulation_context = regulation_context or ""

    with _StageTimer(timings, "mc"):
        mc_results = _run_mc_simulation(
            pace_out=pace_out,
            tire_out=tire_out,
            situation_out=situation_out,
            pit_out=pit_out,
            alpha=race_state.risk_tolerance,
        )
        best_mc = max(mc_results, key=lambda s: mc_results[s]["score"])

    with _StageTimer(timings, "synthesis"):
        prompt = _build_orchestrator_prompt(
            race_state=race_state,
            mc_results=mc_results,
            best_mc=best_mc,
            pace_out=pace_out,
            tire_out=tire_out,
            situation_out=situation_out,
            pit_out=pit_out,
            radio_out=radio_out,
            regulation_context=regulation_context,
        )
        synth = _get_orchestrator_llm().invoke(prompt)
        rec = _assemble_recommendation(synth, pit_out, mc_results, regulation_context)

    timings["total"] = sum(timings.values())

    agent_outputs = None
    if return_agent_outputs:
        agent_outputs = _assemble_agent_outputs(
            pace_out=pace_out,
            tire_out=tire_out,
            situation_out=situation_out,
            radio_out=radio_out,
            pit_out=pit_out,
            regulation_context=regulation_context,
            rag_dict=rag_dict,
            active=active,
            guardrail_reason=None,  # rich mode applies rails via the LLM prompt, not post-hoc
        )
    return rec, agent_outputs, timings


def _assemble_agent_outputs(
    *,
    pace_out: Any,
    tire_out: Any,
    situation_out: Any,
    radio_out: Any,
    pit_out: Any,
    regulation_context: str,
    rag_dict: dict[str, Any] | None,
    active: Any,
    guardrail_reason: str | None,
) -> dict[str, Any]:
    """Build the per-agent outputs dict consumers (arcade dashboard) render.

    The key set is identical to arcade's historical contract
    (``strategy_pipeline.py``) so its formatters keep working the day it delegates,
    plus ``guardrail_reason`` (``None`` in rich mode, populated in no-llm mode).
    """
    outputs = {
        "pace_out": pace_out,
        "tire_out": tire_out,
        "situation_out": situation_out,
        "radio_out": radio_out,
        "pit_out": pit_out,
        "regulation_context": regulation_context,
        "rag": rag_dict,
        "active": list(active),
        "guardrail_reason": guardrail_reason,
    }
    return outputs


def _build_default_lap_state(race_state: RaceState, laps_df: pd.DataFrame) -> dict[str, Any]:
    """Build the minimal ``lap_state`` every sub-agent expects from a bare RaceState.

    Lifted verbatim from ``src/arcade/strategy_pipeline._build_default_lap_state``
    (itself a mirror of the orchestrator's inline fallback). The engine is now the
    single non-orchestrator home for it; arcade's copy is deleted when it delegates.
    The parity test's ``lap_state=None`` case guards this against the orchestrator's
    inline block.
    """
    driver_rows = laps_df[laps_df["Driver"] == race_state.driver]
    lap_row = driver_rows[driver_rows["LapNumber"] == race_state.lap]
    year = int(laps_df["Year"].iloc[0]) if "Year" in laps_df.columns else 2025
    gp_name = str(laps_df["GP_Name"].iloc[0]) if "GP_Name" in laps_df.columns else ""
    stint = int(lap_row["Stint"].iloc[0]) if not lap_row.empty else 1
    team = str(lap_row["Team"].iloc[0]) if not lap_row.empty and "Team" in lap_row else "Unknown"
    lap_state = {
        "lap_number": race_state.lap,
        "driver": {
            "driver": race_state.driver,
            "driver_number": 0,
            "team": team,
            "position": race_state.position,
            "compound": race_state.compound,
            "tyre_life": race_state.tyre_life,
            "stint": stint,
            "lap_time_s": None,
            "speed_st": 300.0,
            "fuel_load": 1 - race_state.lap / max(race_state.total_laps, 1),
        },
        "session_meta": {
            "gp_name": gp_name,
            "gp": gp_name,
            "year": year,
            "driver": race_state.driver,
            "team": team,
            "total_laps": race_state.total_laps,
        },
        "weather": {
            "air_temp": race_state.air_temp,
            "track_temp": race_state.track_temp,
            "rainfall": race_state.rainfall,
            "humidity": 50.0,
        },
        "rivals": [],
    }
    return lap_state
