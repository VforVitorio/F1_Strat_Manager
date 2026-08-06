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
  * ``rich``   — re-drives ``run_strategy_orchestrator_from_state``'s five-step
                 sequence by importing the orchestrator layer functions (never
                 copying them) and returns the per-agent outputs the public API
                 discards. Importing the functions removes body drift, but not
                 call drift: re-driving a sequence means every argument is
                 threaded by hand. This is why the docstring no longer claims
                 "byte-for-byte" parity. It has now happened twice: `live_drivers`
                 was missed, disabling #462's guard, and `cliff_p50`/`total_laps`
                 were missed, leaving #433's stint-end guard with no anchor. Both
                 on this profile, the default for every surface.
  * ``no-llm`` — the deterministic, zero-LLM-client path (see ``no_llm.py``); fixes
                 #166 by construction (it never calls ``_run_conditional_agents``).

Untouchability: nothing in ``src/agents/`` is modified. Every strategy layer is the
SAME code object the orchestrator runs (imported, never copied); the only
engine-owned code is the call sequence itself and the default-lap_state builder.

Decision memory
---------------
``run_lap`` accepts an optional ``DecisionMemory`` and renders its block into the Layer 3
prompt. The accumulator belongs to the CALLER — the CLI loop, the arcade connector, the
backend simulator's stream — because this function has to stay pure per lap. The engine
only ever calls ``block()``; the caller records the recommendation afterwards. The two
stateless surfaces (``/recommend``, the MCP tool) get no memory by design, which
``tests/engine/test_memory_scope_is_deliberate.py`` enforces.

Anti-drift guards: ``tests/engine/test_engine.py``, ``tests/engine/test_engine_no_llm.py``,
``tests/engine/test_engine_threads_every_argument.py`` (which checks, by AST, that this path
passes ``_assemble_recommendation`` every argument the orchestrator does) and
``tests/engine/test_memory_scope_is_deliberate.py``.

These do not assert byte-level parity with the orchestrator. An earlier docstring cited
a ``tests/test_engine_parity.py`` that does not exist; the argument-threading test above
replaces that claim with a real one. The two ``lap_state is None`` fallbacks are still
kept in step by hand, so changing one without the other is not caught by a test.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any, Literal

import pandas as pd

if TYPE_CHECKING:
    # Type-only: importing it for real would put a strategy-layer import in the
    # engine's import graph for a value the engine never constructs.
    from src.strategy.inference.decision_memory import DecisionMemory

from src.agents.strategy_orchestrator import (
    RaceState,
    StrategyRecommendation,
    _assemble_recommendation,
    _build_orchestrator_prompt,
    _decide_agents_to_call,
    _get_orchestrator_llm,
    _live_drivers_from,
    _run_always_on_agents_from_state,
    _run_conditional_agents,
    _run_mc_simulation,
    best_mc_candidate,
    race_context_from_lap_state,
)
from src.f1_strat_manager.gp_slugs import resolve_gp_key

# The profiles #169 delivers. ``fast`` (direct-mode sub-agents + event-triggered
# N31, audit F3/F11) is a later phase and is rejected with a pointing error so
# consumers can already write a three-valued switch without silent fallthrough.
PROFILES: tuple[str, ...] = ("rich", "no-llm")

Profile = Literal["rich", "no-llm"]

logger = logging.getLogger(__name__)


# Re-exported from the leaf module `scoping`, where it now lives: importing THIS module
# instantiates the radio agent's three transformer models, and the backend needs the
# scoping rule without paying for them. Kept importable under the old name because
# callers and tests already reference `engine._scope_laps_to_gp`.
from src.strategy.inference.scoping import _scope_laps_to_gp  # noqa: E402


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
    memory: "DecisionMemory | None" = None,
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
        memory: This race's ``DecisionMemory``, owned by the CALLER. Read here,
            never written: the caller records each recommendation after the lap
            returns, which is what keeps this function pure and
            ``tests/engine/test_engine_no_llm.py``'s twice-on-lap-6 assertion true.
            Ignored on ``no-llm``, which builds no prompt to put it in — passing
            one there is harmless, and silent, so it is stated here.

    Returns:
        ``(recommendation, agent_outputs | None, stage_timings)`` where
        ``recommendation`` is always a real ``StrategyRecommendation`` (never an
        ad-hoc dict), ``agent_outputs`` carries the six sub-agent dataclasses +
        routing/RAG context (see ``_assemble_agent_outputs``), and
        ``stage_timings`` maps each stage name to its seconds.

    Raises:
        ValueError: on an unknown profile, or on the reserved ``"fast"`` profile.
    """
    # Scope BEFORE dispatch so both profiles, and therefore every surface that routes
    # through this engine (CLI PMV, arcade), get the single-race frame (#429). Passing
    # `race_state` lets scoping resolve a GP even when `lap_state` is None (#465), so the
    # `_build_default_lap_state` fallback below/inside `_run_rich`/`run_no_llm_lap` always
    # runs against an already-scoped frame instead of the season-wide one.
    laps_df = _scope_laps_to_gp(laps_df, lap_state, race_state)

    if profile == "rich":
        # Rendered here rather than inside _run_rich so the engine's only contact with
        # the accumulator is one read, at one place, of a method that cannot mutate it.
        memory_block = memory.block() if memory is not None else ""
        return _run_rich(race_state, laps_df, lap_state, return_agent_outputs, memory_block)
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
    memory_block: str | None = "",
) -> tuple[StrategyRecommendation, dict[str, Any] | None, dict[str, float]]:
    """The rich profile: the orchestrator's five-step sequence, outputs retained.

    Statement for statement this mirrors ``run_strategy_orchestrator_from_state``
    (always-on agents -> routing -> conditional agents -> Monte Carlo -> LLM
    synthesis), and every step is an IMPORTED orchestrator function rather than a
    copy, so the bodies cannot drift. The intermediate agent outputs are returned
    instead of discarded.

    It does NOT produce byte-identical output, and this docstring used to say it
    did. Importing the functions removes body drift, not CALL drift: the arguments
    still have to be threaded here, and twice they were not. ``live_drivers`` was
    missed, which disabled #462's guard on this profile, and ``cliff_p50`` with
    ``total_laps`` were missed, which left #433's stint-end guard with no anchor.
    Both bugs were invisible precisely because the docstring promised parity.
    ``tests/engine/test_engine_threads_every_argument.py`` is the real claim now: it
    checks the arguments, which is the thing that actually breaks.
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
        _ctx = race_context_from_lap_state(lap_state, race_state)
        mc_results = _run_mc_simulation(
            pace_out=pace_out,
            tire_out=tire_out,
            situation_out=situation_out,
            pit_out=pit_out,
            alpha=race_state.risk_tolerance,
            rivals=(lap_state or {}).get("rivals"),
            position=_ctx.get("position"),
            laps_remaining=_ctx.get("laps_remaining"),
            pit_context=_ctx.get("pit_context"),
        )
        best_mc = best_mc_candidate(mc_results)

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
            # The one argument in this call the orchestrator deliberately does NOT
            # pass: /recommend and the MCP tool are stateless per request and have no
            # race to accumulate over. tests/engine/test_memory_scope_is_deliberate.py
            # holds that asymmetry open, because the threading guard next to it only
            # looks the other way and is green on this by construction.
            memory_block=memory_block,
        )
        synth = _get_orchestrator_llm().invoke(prompt)
        rec = _assemble_recommendation(
            synth,
            pit_out,
            mc_results,
            regulation_context,
            sc_currently_active=situation_out.sc_currently_active,
            # Without this the LLM's free-text `undercut_target` ships unvalidated:
            # `_assemble_recommendation` reads a missing `live_drivers` as "unknown" and
            # lets it through by design. The orchestrator threads it; this path did not,
            # so #462's guard was dead on the `rich` profile — which is the DEFAULT for
            # /simulate, the arcade and the CLI. That is the whole class this engine
            # exists to prevent: one call sequence, or every caller drifts.
            live_drivers=_live_drivers_from(lap_state),
            # Same class, same profile, two arguments further along. Without
            # these `_clamp_expected_stint_end` has no physical anchor and
            # returns the LLM's `expected_stint_end` unclamped, so #433's guard
            # was dead everywhere the default profile runs. The fix for
            # `live_drivers` above landed and these two were missed, which is
            # the argument for the threading test rather than for reading the
            # call twice.
            cliff_p50=tire_out.laps_to_cliff_p50,
            total_laps=race_state.total_laps,
        )

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

    ``laps_df`` is expected to already be scoped to a single GP by the time this runs
    (``run_lap`` calls ``_scope_laps_to_gp(..., race_state)`` BEFORE dispatching to
    ``_run_rich``/``run_no_llm_lap``, #465) — otherwise ``gp_name``/``year`` below read
    ``iloc[0]`` of a season-wide frame and can pick a GP unrelated to ``race_state``.
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
            # A RaceState carries no lap history, so the stint's opening TyreLife is
            # genuinely unknowable here. None makes N06 emit NaN FuelEffect plus a
            # warning, which is in-distribution (2% of the training parquet is null)
            # and cannot be mistaken for a reading. Keep in lockstep with the same
            # key in strategy_orchestrator's lap_state fallback (#446).
            "stint_baseline_tyre_life": None,
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
