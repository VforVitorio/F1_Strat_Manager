"""src/agents/strategy_orchestrator.py

Strategy Orchestrator: extraction from N31_strategy_orchestrator.ipynb.

End-to-end multi-agent supervisor that integrates N25-N30 sub-agents through
three processing layers:

  Layer 1 (MoE routing): deterministic if-else rules decide which conditional
             agents (N28, N30) to activate based on N26/N27/N29 outputs.
  Layer 2 (Monte Carlo simulation): draws CFG.n_sim samples from sub-agent
             probability distributions and evaluates four strategy candidates.
  Layer 3 (LLM synthesis): structured-output LLM aggregates all reasoning strings
             and MC scores into a StrategyRecommendation.

Entry points
------------
run_strategy_orchestrator(race_state, lap_state)
    Primary entry point. Accepts a RaceState Pydantic model and a lap_state dict
    (compatible with the FastF1 entry points of the sub-agents). Every sub-agent
    is self-contained through the arguments passed to it here. Nothing needs to
    be set up on any sub-agent module beforehand.

run_strategy_orchestrator_from_state(race_state, laps_df)
    RSM adapter. Calls the *_from_state entry points of each sub-agent so no
    FastF1 session is required. laps_df is the full lap DataFrame from
    RaceStateManager. lap_state is built internally from race_state + laps_df.

References
----------
Heilmeier et al. (2020) ApplSci 10/4229: MC motorsport simulation
Wang et al. (2024) arXiv:2406.04692: MoA reasoning aggregation
Liu et al. (2024) arXiv:2402.02392: DeLLMa decision under uncertainty with LLM
"""

import json
import logging
import math
import importlib.util
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Literal, Optional

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from src.agents._shared_defaults import LLM_MAX_RETRIES

logger = logging.getLogger(__name__)

# ── Repo root (with root-stop guard for uv tool install) ─────────────────────
_REPO_ROOT = Path(__file__).resolve()
while not (_REPO_ROOT / ".git").exists():
    if _REPO_ROOT.parent == _REPO_ROOT:
        break
    _REPO_ROOT = _REPO_ROOT.parent

if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# ── Optional LangChain imports ──────────────────────────────────────
# Probed, not imported. `import langchain_openai` costs 14.3 s measured: it drags
# the langgraph stack and transformers in behind it, and every consumer of the
# name sits inside a factory that already builds its client on first call. So an
# eager import charged that to `f1-sim --help`, to a --no-llm run, and to every
# surface that merely touches this module. find_spec answers the only question
# asked here, is it installed, in about a millisecond and executes nothing.
_LC_OK = importlib.util.find_spec("langchain_openai") is not None

# ── Sub-agent imports ──────────────────────────────────────────────────────────
from src.agents.pace_agent         import (
    MISSING_PREV_LAP_TIME_S,
    run_pace_agent,
    run_pace_agent_from_state,
)
from src.agents.tire_agent         import run_tire_agent, run_tire_agent_from_state
from src.agents.race_situation_agent import (
    run_race_situation_agent,
    run_race_situation_agent_from_state,
)
from src.agents.pit_strategy_agent import (
    run_pit_strategy_agent,
    run_pit_strategy_agent_from_state,
    _MEDIUM_SUITABILITY_FLOOR_LAPS,
    _STINT_CAPACITY_LAPS,
)

# From the leaf guard-rails module, so this prompt states the same bounds the rails
# enforce rather than a prose copy of them. Every one of these was a literal here and a
# constant there, which is how the SOFT capacity ended up telling the LLM to refuse what
# the selector had just allowed (#741, #766).
from src.strategy.inference.guard_rails import (
    _CLIFF_P10_SAFE,
    _MIN_STINT_LAPS,
    _NO_PIT_BEFORE_LAP,
    _NO_PIT_LAST_N_LAPS,
)
from src.agents.radio_agent        import (
    run_radio_agent,
    run_radio_agent_from_state,
    RadioMessage,
    RCMEvent,
)
from src.agents.rag_agent          import run_rag_agent


# ==============================================================================
# Configuration
# ==============================================================================

@dataclass
class OrchestratorCFG:
    """Runtime configuration for the Strategy Orchestrator (N31).

    n_sim controls Monte Carlo draws per strategy candidate in Layer 2. 500 draws
    keep variance of the mean below 0.01 position units within lap-level latency.

    sc_prob_threshold is the N27.sc_prob_3lap cutoff above which N30 is activated
    to retrieve safety-car regulation context for the pit decision.

    risk_tolerance_default (α) weights expected value vs worst-case in the MC
    score: score(S) = α·E[S] + (1−α)·P10[S]. α=1.0 aggressive, α=0.0 conservative.

    temperature is REQUESTED, not guaranteed. Whether it survives depends on the
    model family: langchain_openai keeps it for gpt-4.1-mini (what the sub-agents
    run) and silently discards it for the gpt-5.x family. The client attribute
    comes back None and the key never reaches the request payload. So with the
    default model_name below, Layer 3 samples at the provider default and this
    value does nothing. Measured 2026-07-27 over 41 laps: two identical passes
    disagreed on confidence in 36, on pit_lap_target in 23, and produced opposite
    actions on the one lap where the call actually changed. The setting does not
    guarantee determinism for any model this project has shipped with.

    _get_orchestrator_llm warns when the request is dropped rather than papering
    over it. Do NOT read a surviving temperature as a promise of determinism
    either: it narrows sampling, it does not remove it.
    See documents/audits/AUDIT_ORCHESTRATOR_MEMORY.md, section 1.1.
    """

    model_name:             str   = "gpt-5.4-mini"
    base_url:               str   = "http://localhost:1234/v1"
    temperature:            float = 0.0
    n_sim:                  int   = 500
    sc_prob_threshold:      float = 0.30
    risk_tolerance_default: float = 0.5


CFG = OrchestratorCFG()

# Lazy LLM singleton, created on first call to avoid connection at import time
_orchestrator_llm = None


def _temperature_was_dropped(llm, requested: float | None) -> bool:
    """True when the client did not keep the requested temperature.

    Split out from _get_orchestrator_llm so the check is testable without an API
    key, a network call or a provider: it only reads an attribute off whatever
    object the client library handed back.

    langchain_openai does not raise when a model family rejects the parameter, it
    nulls the attribute and omits the key from the payload. So `None` on a client
    given an explicit number is the signal, and it is the only one available
    short of inspecting the request.
    """
    return requested is not None and getattr(llm, "temperature", None) is None


def _get_orchestrator_llm():
    """Return the cached structured-output LLM, creating it on first call.

    Checks F1_LLM_PROVIDER env var: 'openai' uses the real OpenAI API
    (requires OPENAI_API_KEY); anything else defaults to LM Studio at CFG.base_url.

    Warns once, on creation, when CFG.temperature does not survive into the
    client (see OrchestratorCFG). The warning is deliberately not a raise: the
    project ships on a model that drops it, so raising would take the whole
    orchestrator down over a setting Layer 3 has never actually had.

    Returns a Runnable that produces StrategyRecommendation Pydantic objects.
    Raises ImportError when langchain_openai is not installed.
    """
    import os
    global _orchestrator_llm
    if _orchestrator_llm is None:
        if not _LC_OK:
            raise ImportError(
                "langchain_openai is not installed. "
                "Install with: pip install langchain-openai"
            )
        from langchain_openai import ChatOpenAI
        provider = os.environ.get("F1_LLM_PROVIDER", "lmstudio")
        if provider == "openai":
            # No parallel_tool_calls: OpenAI rejects it when no tools are specified
            llm = ChatOpenAI(model=CFG.model_name, temperature=CFG.temperature, timeout=120, max_retries=LLM_MAX_RETRIES)
        else:
            llm = ChatOpenAI(
                model=CFG.model_name,
                base_url=CFG.base_url,
                api_key="lm-studio",
                temperature=CFG.temperature,
                model_kwargs={"parallel_tool_calls": False},
                timeout=120,
                max_retries=LLM_MAX_RETRIES,
            )
        if _temperature_was_dropped(llm, CFG.temperature):
            # ASCII only: this line lands on Windows terminals, where the repo's
            # cp1252 console renders an em-dash or a section sign as '?'.
            logger.warning(
                "Layer 3 temperature=%s was discarded by the client for model %r. The "
                "orchestrator is sampling at the provider default, not running "
                "deterministically: consecutive laps will disagree on confidence, "
                "pit_lap_target and reasoning even when the prompt is identical. "
                "See documents/audits/AUDIT_ORCHESTRATOR_MEMORY.md, section 1.1.",
                CFG.temperature, CFG.model_name,
            )
        # _LLMSynthesis only has the 3 fields the LLM actually fills.
        # scenario_scores (dict) and regulation_context are attached in code after.
        _orchestrator_llm = llm.with_structured_output(_LLMSynthesis)
    return _orchestrator_llm


# ==============================================================================
# Input / output dataclasses
# ==============================================================================

class RaceState(BaseModel):
    """Per-lap context slice passed to the Strategy Orchestrator.

    driver identifies the driver whose strategy is being evaluated. All gap
    and pace features are relative to this driver.

    lap and total_laps enable race-percentage features used by N28 (lap_race_pct)
    and the MC simulation for fuel load estimation.

    compound and tyre_life are the current stint values forwarded to N26.

    gap_ahead_s and pace_delta_s reach the LLM synthesis prompt's RACE CONTEXT
    block. They do NOT feed the overtake model: N27 computes its own pair gap
    from laps_df, and this line claimed otherwise for as long as it existed.
    gap_ahead_s is None when there is no car directly ahead to measure: the
    driver leads, or the pos-1 car is not classified this lap (#878). Render the
    absence; never substitute a number for it.

    weather fields (air_temp, track_temp, rainfall) are forwarded to N14 (SC model)
    as contextual features.

    radio_msgs and rcm_events are pre-filtered to the current lap ±1 window by the
    caller before passing to N29. The orchestrator does not filter them itself.
    Items may be RadioMessage/RCMEvent instances or dicts with matching fields;
    the orchestrator converts dicts automatically before passing to N29.

    risk_tolerance (α) is the MC score weight: score = α·E[S] + (1−α)·P10[S].
    Validated in [0, 1] by Pydantic; default 0.5 is neutral risk stance.
    """

    driver:         str
    lap:            int
    total_laps:     int
    position:       int
    compound:       str
    tyre_life:      int
    gap_ahead_s:    Optional[float]
    pace_delta_s:   float
    air_temp:       float
    track_temp:     float
    rainfall:       bool  = False
    radio_msgs:     list  = Field(default_factory=list)
    rcm_events:     list  = Field(default_factory=list)
    risk_tolerance: float = Field(default=0.5, ge=0.0, le=1.0)

    model_config = {"arbitrary_types_allowed": True}


_ACTION_VALUES    = Literal["STAY_OUT", "PIT_NOW", "UNDERCUT", "OVERCUT", "ALERT"]
_PACE_MODE_VALUES = Literal["PUSH", "NEUTRAL", "MANAGE", "LIFT_AND_COAST"]
_RISK_VALUES      = Literal["AGGRESSIVE", "BALANCED", "DEFENSIVE"]
_PRIORITY_VALUES  = Literal["HIGH", "MEDIUM", "LOW"]
_COMPOUND_VALUES  = Literal["SOFT", "MEDIUM", "HARD"]


def _gap_ahead_context_line(race_state: RaceState) -> str:
    """The RACE CONTEXT gap fragment, honest about an absent car ahead.

    None at P1 is the race leader: say so, and say what leading means for
    strategy, because defending a gap behind is a different problem from
    chasing one. None below P1 means the car one place ahead has no
    classified row this lap. Both used to render as a number - 0.00s from
    the builder, or a fabricated 2.00s once a falsy ``or`` had eaten the
    zero - and the LLM narrated overtaking pressure that did not exist
    (#878).
    """
    if race_state.gap_ahead_s is not None:
        return f"Gap ahead: {race_state.gap_ahead_s:.2f}s"
    if race_state.position == 1:
        return "Gap ahead: none - LEADING the race (defending a gap behind, not chasing)"
    return "Gap ahead: none - no car classified directly ahead this lap"


class Contingency(BaseModel):
    """A single conditional branch planned by the LLM for upcoming laps.

    Contingencies encode the if-then-else logic that a real F1 strategist keeps
    in their head: "stay out now, BUT if a Safety Car is deployed in the next
    three laps, switch to PIT_NOW immediately". Without this field the
    orchestrator is forced to collapse every lap decision into a single myopic
    action, discarding any plan B. Exposing contingencies lets N31 communicate
    a genuine multi-lap plan to the UI and to downstream consumers.

    trigger:
        Plain-language description of the event that activates this branch.
        Examples: "SC deployed within 3 laps", "gap to SAI drops below 0.8 s",
        "rain intensity increases". Must be specific enough for a human to
        recognise the trigger condition in live telemetry. Vague triggers
        ("things go wrong") are rejected implicitly by the LLM prompt.
    switch_to:
        The replacement action to execute when the trigger fires. Restricted
        to the same five-value enum as the primary action so that downstream
        MC grounding and UI rendering logic stays consistent whether the
        orchestrator executes the primary plan or a contingency.
    priority:
        Ordering signal when multiple contingencies fire in the same lap.
        HIGH contingencies pre-empt MEDIUM which pre-empt LOW. Used by the
        future execution layer to resolve conflicts deterministically.
    rationale:
        One-line justification linking the trigger to a sub-agent input or
        regulation clause. Kept short (ideally under 100 chars) so the UI
        can render a full contingency list without wrapping.
    """

    model_config = ConfigDict(extra="forbid")

    trigger:   str              = Field(description="When does this branch activate?")
    switch_to: _ACTION_VALUES   = Field(description="Replacement action to execute")
    priority:  _PRIORITY_VALUES = Field(description="Resolution order when multiple fire")
    rationale: str              = Field(description="Brief link to sub-agent data or regulation")


class _LLMSynthesis(BaseModel):
    """Strict-schema model passed to with_structured_output: only the fields the LLM fills.

    OpenAI structured output requires additionalProperties=false on all objects,
    which free-form Dict fields violate. scenario_scores and regulation_context
    are attached in code after the LLM call and live on StrategyRecommendation.

    The schema expanded in v2 adds execution detail (pit_lap_target,
    compound_next, undercut_target), driver-side instructions (pace_mode,
    target_lap_time_s, risk_posture), and multi-lap planning (contingencies,
    key_risks, expected_stint_end). These fields let the LLM surface reasoning
    that previously lived only inside the narrative string or was discarded
    entirely from N28's output. See StrategyRecommendation for per-field prose.
    """
    model_config = ConfigDict(extra="forbid")

    # ── Primary decision: kept for MC grounding + backward compatibility ─────
    action:             _ACTION_VALUES   = Field(
        description="STAY_OUT | PIT_NOW | UNDERCUT | OVERCUT | ALERT",
    )
    reasoning:          str              = Field(
        description="Narrative synthesis of all agent inputs and MC scores",
    )
    confidence:         float            = Field(
        ge=0.0, le=1.0,
        description="LLM self-assessed certainty",
    )

    # ── Pit execution details: recovers N28 data that was previously discarded
    pit_lap_target:     Optional[int]              = Field(
        default=None,
        description="Absolute lap number of the planned stop (None when STAY_OUT)",
    )
    compound_next:      Optional[_COMPOUND_VALUES] = Field(
        default=None,
        description="Compound chosen for the next stint (None when STAY_OUT)",
    )
    undercut_target:    Optional[str]              = Field(
        default=None,
        description="Three-letter code of the rival targeted by an undercut/overcut",
    )

    # ── Driver-side instructions: new dimension for pace management ──────────
    pace_mode:          _PACE_MODE_VALUES = Field(
        description="PUSH | NEUTRAL | MANAGE | LIFT_AND_COAST — what to tell the driver now",
    )
    target_lap_time_s:  Optional[float]   = Field(
        default=None,
        description="Target lap time in seconds, grounded in PaceOutput CI bounds",
    )
    risk_posture:       _RISK_VALUES      = Field(
        description="AGGRESSIVE | BALANCED | DEFENSIVE — championship-aware risk stance",
    )

    # ── Multi-lap planning: the big new reasoning surface ────────────────────
    contingencies:      list[Contingency] = Field(
        default_factory=list,
        max_length=4,
        description="Conditional branches activated by specific triggers",
    )
    key_risks:          list[str]         = Field(
        default_factory=list,
        max_length=5,
        description="Short bullet list of the top risks the LLM wants to flag",
    )
    expected_stint_end: Optional[int]     = Field(
        default=None,
        description="Lap at which the current stint is planned to end (for STAY_OUT plans)",
    )


class StrategyRecommendation(BaseModel):
    """Final structured output of the Strategy Orchestrator (N31).

    v2 schema rationale: the v1 schema collapsed ~30 sub-agent fields into a
    single five-value action, discarding rich execution detail from N28
    (recommended_lap, compound_recommendation, undercut_target) and leaving
    no room for multi-lap planning. v2 keeps the discrete action as the
    primary decision so Monte Carlo grounding via simulate_lap_window and
    existing downstream renderers continue to work, but surrounds it with
    execution detail, driver-side instructions, and a contingency list that
    together turn the orchestrator from a myopic per-lap selector into a
    planner that communicates a genuine strategy.

    action:
        Primary decision: one of five values. STAY_OUT defers the pit stop,
        PIT_NOW calls an immediate box, UNDERCUT pits before the target rival
        to gain track position, OVERCUT stays out to exploit fresh-tyre pace
        later, and ALERT flags a critical event (radio PROBLEM, SC deployed)
        that overrides standard strategy logic. The enum is kept intentionally
        small so each value maps directly to an MC candidate scored in
        simulate_lap_window and to a colour badge in the UI.
    reasoning:
        The LLM's narrative synthesis of all sub-agent inputs, MC scores, and
        regulation constraints, forwarded verbatim to the UI and post-race
        analysis. Use this for the human-readable "why", use the structured
        fields below for machine-readable decisions.
    confidence:
        The LLM's self-assessed certainty in [0, 1]. Treat it as a qualitative
        signal rather than a calibrated probability. Models tend to
        over-report certainty on borderline decisions.
    pit_lap_target:
        Absolute lap number of the planned stop. Populated whenever action is
        PIT_NOW / UNDERCUT / OVERCUT, and optionally for STAY_OUT when the
        LLM wants to communicate a forward-looking plan ("stay out, pit on
        lap 28"). None means "no pit stop planned within the visible horizon".
        Prefer this field over parsing the reasoning string.
    compound_next:
        Compound chosen for the next stint (SOFT / MEDIUM / HARD). Populated
        whenever a stop is planned. Lets the UI render the full stint plan
        without having to re-query N28. None is valid for STAY_OUT decisions.
    undercut_target:
        Three-letter code of the rival targeted by an undercut or overcut
        (e.g. "SAI"). Non-None only for UNDERCUT / OVERCUT actions. Recovers
        N28.undercut_target which v1 silently discarded.
    pace_mode:
        Driving instruction for the immediate next laps: PUSH, NEUTRAL,
        MANAGE, or LIFT_AND_COAST. This is a new dimension introduced in v2:
        previously the orchestrator only answered "when to pit" without any
        signal on how to drive between pit stops. A neutral default keeps
        backward compatibility with consumers that ignore this field.
    target_lap_time_s:
        Concrete target lap time for the driver, grounded in PaceOutput CI
        bounds so the LLM cannot invent values far outside what the N06 model
        predicts. Rendered by the UI as a radio-style instruction. None when
        the LLM prefers not to commit to a precise number.
    risk_posture:
        AGGRESSIVE / BALANCED / DEFENSIVE: captures the championship context
        the LLM is reasoning under. An AGGRESSIVE posture relaxes cliff risk
        tolerance and favours undercut attempts; DEFENSIVE prioritises track
        position over potential gain. Exposing this field makes the stance
        auditable instead of buried inside the narrative.
    contingencies:
        Conditional branches the LLM has planned for upcoming laps. Each
        Contingency bundles a trigger, a replacement action, a priority, and
        a one-line rationale. Capped at four entries so the UI can render the
        full list without scrolling. An empty list means the primary action
        is executed unconditionally.
    key_risks:
        Short bullet list (max five) of the top risks the LLM wants to flag:
        e.g. "cliff P10 at lap 22 is uncomfortably close", "SC probability
        rising in the last three laps". Surfaces reasoning that would
        otherwise be buried inside the prose narrative.
    expected_stint_end:
        Lap number at which the current stint is planned to end. Populated
        primarily for STAY_OUT decisions to communicate "this is a one-stop
        plan, stop on lap 32". Lets the UI render a stint-plan bar without
        parsing the reasoning string.
    scenario_scores:
        Full MC output dict per strategy: {"STAY_OUT": {"E", "P10", "P90",
        "score"}, ...}. Attached in code after the LLM call, not filled by
        the LLM itself. Downstream consumers can inspect the distribution
        without re-running the simulation.
    regulation_context:
        The N30 RAG answer string when activated, empty string otherwise.
        Attached in code after the LLM call. Included on the recommendation
        so the UI can surface the regulatory basis for the action without
        re-querying N30.
    """

    # ── Primary decision ──────────────────────────────────────────────────────
    action:             _ACTION_VALUES   = Field(
        description="STAY_OUT | PIT_NOW | UNDERCUT | OVERCUT | ALERT",
    )
    reasoning:          str              = Field(
        description="Narrative synthesis of all agent inputs and MC scores",
    )
    confidence:         float            = Field(
        ge=0.0, le=1.0,
        description="LLM self-assessed certainty",
    )

    # ── Pit execution details ─────────────────────────────────────────────────
    pit_lap_target:     Optional[int]              = Field(
        default=None,
        description="Absolute lap of the planned stop",
    )
    compound_next:      Optional[_COMPOUND_VALUES] = Field(
        default=None,
        description="Compound chosen for the next stint",
    )
    undercut_target:    Optional[str]              = Field(
        default=None,
        description="Rival code targeted by UNDERCUT/OVERCUT",
    )

    # ── Driver-side instructions ──────────────────────────────────────────────
    pace_mode:          _PACE_MODE_VALUES = Field(
        default="NEUTRAL",
        description="PUSH | NEUTRAL | MANAGE | LIFT_AND_COAST",
    )
    target_lap_time_s:  Optional[float]   = Field(
        default=None,
        description="Target lap time (s), grounded in PaceOutput CI",
    )
    risk_posture:       _RISK_VALUES      = Field(
        default="BALANCED",
        description="AGGRESSIVE | BALANCED | DEFENSIVE",
    )

    # ── Multi-lap planning ────────────────────────────────────────────────────
    contingencies:      list[Contingency] = Field(
        default_factory=list,
        description="Conditional branches planned for upcoming laps",
    )
    key_risks:          list[str]         = Field(
        default_factory=list,
        description="Top risks the LLM wants to flag",
    )
    expected_stint_end: Optional[int]     = Field(
        default=None,
        description="Lap at which the current stint is planned to end",
    )

    # ── Post-hoc grounding (attached in code, not filled by the LLM) ──────────
    scenario_scores:    dict  = Field(
        default_factory=dict,
        description="MC scores per strategy",
    )
    regulation_context: str   = Field(
        default="",
        description="N30 answer if activated, else empty",
    )


# ==============================================================================
# Layer 1: MoE routing
# ==============================================================================

# RCM event_type values (radio_agent._classify_rcm_event naming) that count as
# a FIA-facing penalty or red-flag ruling for N30 routing purposes. RED_FLAG is
# reachable today: it is one of radio_agent._SAFETY_FLAGS, so radio_agent
# ._build_alerts forwards it into RadioOutput.alerts as {'source': 'rcm',
# 'event_type': 'RED_FLAG', ...}. TIME_PENALTY is listed for when an RCM event
# actually reaches this set; as of writing, radio_agent._build_alerts only
# forwards RCM events whose event_type is in _SAFETY_FLAGS, which excludes
# TIME_PENALTY, so it currently never arrives in radio_alerts. That is a
# separate, upstream gap in src/agents/radio_agent.py, not fixed here.
_RCM_PENALTY_EVENT_TYPES = {"RED_FLAG", "TIME_PENALTY"}


def _decide_agents_to_call(
    tire_warning:  str,
    sc_prob_3lap:  float,
    radio_alerts:  list,
    sc_currently_active: bool = False,
) -> set:
    """Layer 1 MoE routing: returns set of conditional agent keys to activate.

    N25, N26, N27, N29 are always called by run_strategy_orchestrator and are
    not returned here. This function only decides N28 and N30.

    tire_warning is TireOutput.warning_level ("OK" | "MONITOR" | "PIT_SOON").
    sc_prob_3lap is RaceSituationOutput.sc_prob_3lap from N27.
    radio_alerts is RadioOutput.alerts: each dict has keys 'source' and 'intent'
    or 'event_type'.

    **N28 (pit strategy agent)** activates when the tyre is near the cliff
    (tire_warning == PIT_SOON) or when the radio signals a car problem that
    could force an unplanned stop (PROBLEM / WARNING intent). A firing N28 is
    the signal that a compound change is imminent, which is the canonical
    trigger for the regulation check.

    **N30 (RAG regulation check)** activates only when the upcoming decision
    actually touches sporting-regulation territory, so Qdrant is not queried
    on quiet cruise laps where no rule is relevant:

    * N28 is active, meaning an imminent pit or compound change: query
      tyre-compound and pit-lane rules (mandatory dry compound, unsafe
      release, pit window).
    * sc_prob_3lap > threshold, meaning SC deployment is likely: query SC
      procedure (delta lap time, pit-lane closure, double-yellow restart).
    * Radio carries a FIA-facing alert: an RCM alert whose event_type is in
      _RCM_PENALTY_EVENT_TYPES (e.g. RED_FLAG), or a radio-transcript alert
      with a WARNING intent, triggers a regulation lookup for the
      infringement the steward is flagging.

    Any of the three conditions independently activates N30. The orchestrator
    LLM then sees the retrieved regulation snippet in its prompt and must
    reconcile its proposed action with the rules before committing.
    """
    activate: set = set()

    if tire_warning == "PIT_SOON":
        activate.add("N28")

    alert_intents = {a.get("intent", "") for a in radio_alerts}
    if alert_intents & {"PROBLEM", "WARNING"}:
        activate.add("N28")

    if sc_prob_3lap > CFG.sc_prob_threshold:
        activate.add("N30")

    # RCM-sourced alerts (source='rcm') carry no 'intent' key at all: only
    # radio-transcript alerts (source='radio') do, and radio_agent.CFG
    # .alert_intents is ("PROBLEM", "WARNING"); no producer ever emits
    # intent == "PENALTY". `alert_intents & {"PENALTY", "WARNING"}` was
    # therefore unreachable on its PENALTY half. Route penalty/red-flag RCM
    # alerts on their real 'event_type' field instead, unioned with the
    # still-valid WARNING intent check so nothing that used to fire stops
    # firing (#398).
    alert_event_types = {a.get("event_type", "") for a in radio_alerts}
    if (alert_intents & {"WARNING"}) or (alert_event_types & _RCM_PENALTY_EVENT_TYPES):
        activate.add("N30")

    if "N28" in activate:
        activate.add("N30")

    # SC physically deployed (confirmed by RCM, not just predicted): force
    # N28 so the pit decision is re-evaluated, and N30 so the sporting
    # regulations covering pitting under SC and pit-lane closure are consulted.
    if sc_currently_active:
        activate.add("N28")
        activate.add("N30")

    return activate


# ==============================================================================
# Layer 2: Monte Carlo simulation
# ==============================================================================

# Simulation constants. Cite per constant, not as a block: the earlier blanket
# attribution "Heilmeier et al. 2020 section 3.2" was inaccurate. Section 3.2 of the
# Virtual Strategy Engineer paper (ApplSci 10/7805) covers pit-stop decisions with a
# neural network and holds none of these values. The safety-car pit-loss material is in
# section 3.5 of the race-simulation paper (ApplSci 10/4229), and there it is a
# per-circuit table rather than a single constant.
WINDOW_LAPS  = 5     # lap horizon for each strategy evaluation.
                     # Not drawn from the literature. Published F1 strategy models
                     # optimise over the full remaining race (Heilmeier minimises total
                     # race time; van Kampen et al. 2024 argues against short horizons).
                     # A short window is a deliberate simplification, noted as a
                     # limitation rather than derived.
FRESH_GAIN   = 0.25  # s/lap advantage of fresh vs degraded tyre
CLIFF_LOSS   = 0.80  # s/lap lost when tyre passes the cliff.
                     # No counterpart in Heilmeier, who models degradation as linear
                     # (t_tire = k0 + k1*age) with no cliff term. About 14x the
                     # degradation rate measured on this repo's 70 races (HARD .052 /
                     # MEDIUM .059 / SOFT .072 s/lap), so it is a cliff parameter,
                     # not a Heilmeier citation.
POS_GAP_S    = 1.50  # seconds per position gap (midfield approximation).
                     # LEGACY PATH ONLY. The projection scoring counts the actual
                     # cars and needs no such constant. Measured over this repo's
                     # 70 races, the median gap between consecutive cars is 2.23 s
                     # while racing and 1.48 s under a Safety Car, so a single
                     # figure cannot serve both regimes: 1.5 is close to the
                     # bunched-field value and was being applied to green-flag
                     # racing, where most decisions are taken. Kept unchanged
                     # because the goldens pin the legacy output to the digit.
SC_PIT_BONUS = 8.0   # seconds saved by pitting under a full Safety Car (Art. 55, no
                     # delta-lap loss). Measured on this repo's 70 races: 5.75 s, 95% CI
                     # [3.14, 8.25] (n=124), so 8.0 sits inside the interval. Close to the
                     # mean of Heilmeier's four published circuits (8.18 s, section 3.5 of
                     # 10/4229). The N28 prompt's earlier "~12 s" is outside that CI.
                     # A per-circuit value would fit better: Heilmeier's spread runs
                     # 5.24 s (Melbourne) to 11.16 s (Catalunya), and #448 already builds
                     # the per-circuit table it could read from.
VSC_PIT_BONUS = 3.0  # seconds saved by pitting under a Virtual Safety Car (Art. 56).
                     # Materially less than a full SC: a VSC preserves gaps and restarts
                     # near-instantly (56.5 / 56.7), so the field is NOT queued and the
                     # relative saving is roughly half. NOT measured: the repo ships no
                     # pit-stop dataset labelled by track status (data/processed/pit_labeled
                     # is empty), so this is a conservative placeholder to calibrate once
                     # such data exists (#470/#471). Roughly half the SC saving measured
                     # elsewhere (~5.75 s) rounds to ~3.0; conservative (understates the VSC
                     # benefit) so it biases against over-recommending a VSC stop until
                     # measured.


def _tyre_term(deg_cost_s: float | None, old_laps: int, fresh_laps: int) -> float:
    """Seconds the tyre state is worth over one plan's window.

    Two mutually exclusive ways of pricing the same physical thing, so this is a
    REPLACEMENT and not an addition, and the double-count trap is avoided by that
    fact rather than by a correction term.

    With a measured signal, charge the laps spent on the OLD set: a tyre 0.4 s off
    the pace costs 0.4 s every lap it is kept, whether or not the cliff is near.
    Without one, fall back to ``FRESH_GAIN``, the hardcoded 0.25 s/lap credit for
    the laps spent on the NEW set, which is the same quantity guessed at instead of
    read. Both are seconds, applied before the conversion to positions.

    ``None`` is a real state, not a defect: a stint whose early laps are outside the
    replay has no reference to subtract, and the fallback keeps that case scoring
    exactly as it did before #744b.
    """
    if deg_cost_s is None:
        return FRESH_GAIN * fresh_laps
    return -deg_cost_s * old_laps


def simulate_lap_window(
    strategy: str,
    cliff_i:  float,
    sc_i:     bool,
    pit_i:    float,
    ucut_i:   bool,
    window:   int = WINDOW_LAPS,
    sc_pit_bonus: float = SC_PIT_BONUS,
    deg_cost_s: float | None = None,
) -> float:
    """Estimate position gain vs STAY_OUT baseline over a W-lap window.

    Returns a position-equivalent score (positive = positions gained).
    STAY_OUT is the reference. All other strategies are scored relative to it.

    strategy:
        One of STAY_OUT / PIT_NOW / UNDERCUT / OVERCUT.
    cliff_i:
        Laps remaining before tyre cliff (from Triangular N26 draw). Laps
        beyond the cliff contribute CLIFF_LOSS s/lap of time loss, converted
        to position units using POS_GAP_S.
    sc_i:
        Whether a neutralisation (SC or VSC) occurs in the window (Bernoulli N27 draw).
        Pitting under one avoids the delta-lap penalty (sc_pit_bonus saved). Under a
        neutralisation, OVERCUT scores the same as PIT_NOW (same stop, same bonus, same
        fresh-tyre window), so the model is indifferent between them and the tie breaks
        elsewhere. OVERCUT previously scored higher because its branch took the bonus
        without subtracting the stop.
    pit_i:
        Pit stop duration sample in seconds (Triangular N28 / prior draw). Physical stop
        only, not the pit-lane traversal. The two-compound rule makes a stop mandatory,
        so pit-now and pit-later both pay the traversal and it cancels in a comparison
        scored relative to STAY_OUT. Adding it to one side puts PIT_NOW near -14 positions
        against a worst-case STAY_OUT of about -2.7, which would suppress pitting entirely.
    ucut_i:
        Whether the undercut succeeds (Bernoulli N16 draw). Gates the extra
        +POS_GAP_S bonus of UNDERCUT vs PIT_NOW.
    window:
        Lap horizon for the evaluation. Default is WINDOW_LAPS=5.
    sc_pit_bonus:
        Seconds a pit stop saves when sc_i is True. Defaults to SC_PIT_BONUS (a full
        Safety Car). The caller passes VSC_PIT_BONUS under a Virtual Safety Car, which
        saves materially less because the field is not queued (Art. 56, #471). Only the
        magnitude of the neutralisation's pit benefit differs; every other term is
        unchanged, so a green-flag call (default) is byte-identical to before.
    """
    if strategy == "STAY_OUT":
        cliff_laps = max(0.0, window - cliff_i)
        time_delta = -cliff_laps * CLIFF_LOSS + _tyre_term(deg_cost_s, window, 0)

    elif strategy == "PIT_NOW":
        sc_saving  = sc_pit_bonus if sc_i else 0.0
        time_delta = -pit_i + sc_saving + _tyre_term(deg_cost_s, 0, window)

    elif strategy == "UNDERCUT":
        sc_saving  = sc_pit_bonus if sc_i else 0.0
        ucut_bonus = POS_GAP_S if ucut_i else 0.0
        time_delta = -pit_i + sc_saving + _tyre_term(deg_cost_s, 0, window) + ucut_bonus

    elif strategy == "OVERCUT":
        # An overcut still makes the stop, just later, so it pays for it like the others.
        # These branches previously omitted `-pit_i`, so OVERCUT collected FRESH_GAIN and
        # the full SC_PIT_BONUS without cost and took the argmax on 92.5% of a 160-state
        # sweep, leaving the layer effectively constant. Charging the stop restores a real
        # choice: STAY_OUT 42.5%, UNDERCUT 31.2%, PIT_NOW 26.2% on that sweep (PIT_NOW's
        # share is all tie-break: it ties UNDERCUT whenever the undercut fails, and the
        # dict order lists it first). OVERCUT no longer wins, which is the known
        # limitation in test_mc_is_a_real_decision.py and #470.
        #
        # Do not add the pit-lane traversal (~20 s) here. A stop is mandatory under the
        # two-compound rule, so pit-now and pit-later both pay it and it cancels in a
        # comparison scored relative to STAY_OUT; charging one side only puts PIT_NOW near
        # -14 positions against a worst-case STAY_OUT of about -2.7 and suppresses pitting.
        # See tests/mc/test_mc_is_a_real_decision.py.
        if sc_i:
            time_delta = -pit_i + sc_pit_bonus + _tyre_term(deg_cost_s, 0, window)
        else:
            cliff_laps = max(0.0, (window // 2) - cliff_i)
            # The only branch that splits the window: half on the old set, half on the
            # new one. Getting these two counts wrong makes the wear term a constant
            # offset that cancels in the argmax and looks like it did nothing.
            time_delta = (
                -pit_i
                + _tyre_term(deg_cost_s, window // 2, window // 2)
                - cliff_laps * CLIFF_LOSS
            )

    else:
        time_delta = 0.0

    return time_delta / POS_GAP_S


# PIT-LANE TRAVERSAL, NOT TOTAL PIT LOSS. This is the time spent transiting the
# lane, to which the sampled physical stop is added: D = traversal + stop. The
# per-circuit table (#448) spans 19.7 s at Budapest to 27.5 s at Marina Bay and a
# caller that knows the GP must pass its own figure through pit_context, because
# a 7.9 s spread is the difference between a stop that costs a place and one that
# does not. The 20.0 fallback is a traversal, so adding a ~2.6 s stop lands near
# the 22.6 s pooled green pit loss measured over this repo's 70 races (n=1746).
# Do NOT pass that 22.6 s figure in as traversal or the stop is counted twice.
DEFAULT_PIT_TRAVERSAL_S = 20.0

# Physical-stop prior used for a RIVAL when the caller cannot supply one: the mode
# of N15's conservative Triangular(2.2, 2.8, 3.8). A rival who is in the pit lane
# has to lose the same kind of time a real stop would cost, and defaulting that to zero made
# their stop free: a car ahead could serve a stop and stay ahead, which is a
# sentinel wearing a plausible number (the #428 lesson in a new place).
RIVAL_STOP_PRIOR_S = 2.8


def _rival_states_from_lap_state(rivals: list[dict], pit_context: dict | None, traversal_s: float):
    """Adapt the lap_state rivals list into the projection's own value type.

    The one place that knows the RaceStateManager field names, so a rename there
    lands here and nowhere else. A rival whose interval is unknown keeps a None
    gap and the projection drops it from the count rather than inventing a zero.

    ``traversal_s`` is threaded in so a pitting rival is charged the same kind of
    pit loss charged for the driver's own stop. Callers can override per rival through
    ``pit_context['rival_pit_loss_s']`` once that figure is available per car.
    """
    from src.agents.position_projection import RivalState

    context = pit_context or {}
    per_rival_pending: dict = context.get("rival_stop_pending") or {}
    rival_loss = float(context.get("rival_pit_loss_s") or (traversal_s + RIVAL_STOP_PRIOR_S))

    states = []
    for rival in rivals:
        driver = str(rival.get("driver", ""))
        states.append(
            RivalState(
                driver=driver,
                gap_s=rival.get("interval_to_driver_s"),
                is_pitting=bool(rival.get("is_pitting", False)),
                stop_pending=per_rival_pending.get(driver),
                stop_loss_s=rival_loss,
            )
        )
    return states


def race_context_from_lap_state(lap_state: dict | None, race_state=None) -> dict:
    """Assemble the projection's race context from a lap_state, or an empty dict.

    Built here rather than at each surface so the CLI, the arcade and the backend
    cannot drift on what "race context" means: three hand-mirrored copies of a
    payload is how this codebase acquired most of its cross-surface bugs.

    Everything is optional and everything degrades honestly: no rivals means the
    caller stays on the legacy scoring, an unknown circuit means the traversal
    falls back with a warning rather than to a silent average, and an unsettled
    stop obligation stays None so the terminal liability makes no claim.
    """
    if not lap_state:
        return {}

    from src.agents.position_projection import traversal_seconds

    driver = lap_state.get("driver") or {}
    meta = lap_state.get("session_meta") or {}
    gp_name = meta.get("gp_name")

    total_laps = meta.get("total_laps") or (getattr(race_state, "total_laps", 0) or 0)
    current_lap = lap_state.get("lap_number") or getattr(race_state, "lap", 0) or 0
    traversal = traversal_seconds(gp_name)
    if traversal is None and gp_name:
        logger.warning(
            "no pit-lane traversal for GP %r — the projection falls back to %.1f s, "
            "which is right to within a few seconds but wrong per circuit (#448)",
            gp_name,
            DEFAULT_PIT_TRAVERSAL_S,
        )

    context = {
        "gp_name": gp_name,
        "laps_remaining": max(0, int(total_laps) - int(current_lap)),
        "position": driver.get("position"),
        "pit_context": {
            "gp_name": gp_name,
            "traversal_s": traversal,
            "mandatory_stop_pending": (lap_state.get("stint_flags") or {}).get(
                "mandatory_stop_pending"
            ),
            "rival_stop_pending": lap_state.get("rival_stop_pending") or {},
        },
    }
    return context


def _has_usable_gaps(rivals: list[dict] | None) -> bool:
    """Whether any rival carries a gap the projection can actually use.

    A list of cars whose intervals are all unknown is truthy but carries no
    geometry: projecting from it counted zero rivals and reported P1 with no
    uncertainty, which is a confident "finishes first, guaranteed" assembled from
    no information at all. Unknown is not zero, and a list of unknowns is not a
    race state. Those runs belong on the legacy path.

    NaN counts as unknown, not as a number. A pandas frame yields NaN where a
    dict yields None, and a single NaN gap is not a small error: it propagates
    through every arithmetic step, so all four candidates come back ``nan``
    while still claiming ``eligible: true``, the argmax collapses to whichever
    key happens to be first, and the payload serialises to invalid JSON.
    """
    return any(_finite_or_none(rival.get("interval_to_driver_s")) is not None for rival in rivals or ())


def _finite_or_none(value) -> float | None:
    """A real number, or None for anything that cannot be arithmetic.

    None, NaN and infinity all mean "no usable value" to this layer, and they
    arrive from different places: a dict gives None, a pandas frame gives NaN,
    and a division on an empty slice gives inf. Collapsing all three to None at
    the boundary is what keeps a single bad reading from turning every
    downstream number into ``nan`` while the payload still claims to be scored.
    """
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(number) or math.isinf(number) else number


def _ordered_by(choices: list[str], preference: list[str]) -> list[str]:
    """``choices`` sorted to follow ``preference``, keeping anything it omits.

    Used to put the nearest post-pit-cycle rival at the head of an eligibility
    list, so ``target`` names the car actually being raced rather than
    whichever one the rivals list happened to mention first.
    """
    rank = {driver: index for index, driver in enumerate(preference)}
    return sorted(choices, key=lambda driver: rank.get(driver, len(rank)))


def _clean_air_available(rival_states: list, gp_name: str | None) -> float:
    """This circuit's clean-air gain, but only if a car ahead is boxing from the driver's wake.

    Two conditions, both necessary. Someone directly ahead has to be entering the
    pit lane, because that is what vacates the road, and the driver has to be inside the
    band the measurement was taken at, because the number describes what a car
    within two seconds gains and says nothing about a car eight seconds back.
    Fail either and the gain is zero, which is what makes an overcut a real move
    at Suzuka and merely a late stop at Monza.
    """
    from src.agents.position_projection import CLEAN_AIR_BAND_S, measured_clean_air_s

    in_our_wake = any(
        rival.is_ahead
        and rival.is_pitting
        and rival.gap_ahead_s is not None
        and rival.gap_ahead_s <= CLEAN_AIR_BAND_S
        for rival in rival_states
    )
    return measured_clean_air_s(gp_name) if in_our_wake else 0.0


def _bounded_by_race_end(racing_laps: float, laps_remaining: int) -> float:
    """Racing laps the window can actually contain, given the race ends.

    ``laps_remaining`` of 0 means unknown here, not "the race is over": several
    callers cannot supply it, and clamping an unknown to zero would silence the
    whole window. Only a positive, smaller count clamps.
    """
    if laps_remaining <= 0:
        return racing_laps
    return min(racing_laps, float(laps_remaining))


def _position_or(reported, counted: int) -> int:
    """The reported classification position, or the one counted from the gaps.

    Positions start at P1, so anything at or below zero is not a position: it is
    the NaN-coerced sentinel that once let a leader "find" the car that had just
    crashed (#428). Such a value falls through to the counted figure rather than
    being trusted.
    """
    number = _finite_or_none(reported)
    if number is None or number < 1:
        return counted
    return int(number)


def _lap_count_or_zero(reported) -> int:
    """A lap count as a whole number, or zero when it is unknown or nonsensical."""
    number = _finite_or_none(reported)
    if number is None or number < 0:
        return 0
    return int(number)


# Which candidate wins when two score EXACTLY the same is decided by this stated
# order, not by `max`, which would return whichever item happens to come first in
# `_run_mc_simulation`'s dict. Relying on `max` would produce the right answer
# (STAY_OUT is first, and conservative is the right default for a genuine tie) for
# the wrong reason, invisibly: nothing would record that a tie had happened.
#
# It is not a rare corner either. Over 415 real laps there were six near-ties,
# ALL SIX on decision laps, three of them exact and falling on real pit stops (#645).
# The margin is smallest exactly where the call is hardest, which is the opposite of
# the usual case, so this list decides real races and deserves to be a decision.
#
# Conservative first: doing nothing is recoverable next lap, a stop is not.
_TIE_BREAK_ORDER: tuple[str, ...] = ("STAY_OUT", "PIT_NOW", "UNDERCUT", "OVERCUT")


def _scoreable(mc_results: dict) -> dict:
    """The candidates that actually carry a finite score.

    A candidate with no valid target scores None by design, and it must not be
    silently treated as zero: that is the sentinel-collision shape this codebase
    keeps paying for.
    """
    return {
        name: cell
        for name, cell in mc_results.items()
        if _finite_or_none(cell.get("score")) is not None
    }


def mc_decision_margin(mc_results: dict) -> float | None:
    """How far the winner beat the runner-up, or None when there is no contest.

    Exists because a caller could previously not tell "the model preferred this by
    a clear margin" from "two candidates scored identically and one was picked".
    Both arrive as the same bare string, and the second is a materially different
    statement about the lap: a tie means the decision is balanced, which
    is information a strategist wants rather than noise to hide.

    Returns None when fewer than two candidates are scoreable, because there is no
    margin to report and zero would read as "a dead tie" (#645).
    """
    scores = sorted(
        (cell["score"] for cell in _scoreable(mc_results).values()), reverse=True
    )
    if len(scores) < 2:
        return None
    return float(scores[0] - scores[1])


def best_mc_candidate(mc_results: dict) -> str:
    """The argmax over scored candidates, skipping the ones never offered.

    Four call sites used to do ``max(results, key=lambda s: results[s]["score"])``
    directly, which raises the moment a score is None. None is exactly what
    the projection engine emits for a candidate with no valid target. Sharing one
    helper is also what stops the four from drifting apart, which is how this
    codebase acquired most of its duplicate-logic bugs.

    An exact tie is broken by ``_TIE_BREAK_ORDER`` rather than by whichever key the
    caller inserted first. Same answer as before on today's dict order, different
    reason: it is now a stated rule instead of an accident, and the tie is logged
    so it stops being invisible. ``mc_decision_margin`` is the companion a caller
    should read when it needs to know how close the call was.

    Falls back to the first key when nothing is scoreable at all (no rivals, every
    candidate ineligible): callers need a string, and an arbitrary-but-stable pick
    beats raising inside a race.
    """
    scored = _scoreable(mc_results)
    if not scored:
        return next(iter(mc_results), "STAY_OUT")

    best_score = max(cell["score"] for cell in scored.values())
    tied = [name for name, cell in scored.items() if cell["score"] == best_score]
    if len(tied) == 1:
        return tied[0]

    winner = min(tied, key=lambda name: _tie_break_rank(name))
    logger.info(
        "MC tie at %.6f between %s; %s wins by the stated precedence, not by dict "
        "order. A tie means the lap is genuinely balanced (#645).",
        best_score,
        ", ".join(sorted(tied)),
        winner,
    )
    return winner


def _tie_break_rank(name: str) -> int:
    """Position in the tie-break precedence; unknown candidates sort last."""
    try:
        return _TIE_BREAK_ORDER.index(name)
    except ValueError:
        return len(_TIE_BREAK_ORDER)


def _format_mc_row(name: str, cell: dict) -> str:
    """One line of the Monte Carlo table for the LLM prompt.

    An ineligible candidate is stated as such rather than formatted: ``%+.3f``
    raises on None, and a candidate with no target is information the model
    should have: "there is nobody to undercut" is a reason, not a gap.
    """
    if cell.get("score") is None:
        reason = "no valid target" if cell.get("eligible") is False else "not scored"
        return f"  {name}: not offered ({reason})"

    target = cell.get("target")
    suffix = f"  target={target}" if target else ""
    return (
        f"  {name}: E={cell['E']:+.3f}  P10={cell['P10']:+.3f}  "
        f"P90={cell['P90']:+.3f}  score={cell['score']:+.3f}{suffix}"
    )


def _run_projection_mc(
    *,
    rivals: list[dict],
    position: int | None,
    laps_remaining: int | None,
    pit_context: dict | None,
    cliff_s,
    sc_s,
    pit_s,
    ucut_s,
    alpha: float,
    neutralisation_saving_s: float,
    deg_cost_s: float | None = None,
    capture: dict | None = None,
) -> dict:
    """Score the four candidates in projected track position instead of seconds.

    Same draws, same window, same alpha·E + (1−alpha)·P10 as the legacy path; what
    changes is the currency. Each candidate is projected against the ACTUAL cars on
    track (``position_projection``), so the pit lane, the traffic rejoined into and
    the still-owed stop all enter the score as cars rather than as constants.

    Draws are shared across candidates (common random numbers) and across the two
    neutralisation regimes: a draw that samples a Safety Car is scored with the
    measured racing-lap count for one, and the same draw under green with the full
    window. That per-draw split is why the Art. 55.17 endgame needs no rail: with
    the race finishing behind the Safety Car there are no racing laps left, so a
    stop buys nothing and staying out wins on the numbers.

    Returns the usual four keys, each with E / P10 / P90 / score, plus ``eligible``
    and ``target``. A candidate with no valid target is ``eligible: false`` with a
    ``score`` of None, never a numeric sentinel, which is the 0.5 coin-flip that
    used to hand UNDERCUT a bonus with no target at all (#434).
    """
    import numpy as _np

    from src.agents.position_projection import (
        DriverPlan,
        ProjectionConfig,
        future_neutralisation_probability,
        measured_neutralisation_rate,
        measured_racing_laps,
        measured_undercut_band_s,
        overcut_targets,
        payoff,
        project_positions,
        rank_targets,
        undercut_targets,
    )

    context = pit_context or {}
    traversal_s = float(context.get("traversal_s") or DEFAULT_PIT_TRAVERSAL_S)
    rival_states = _rival_states_from_lap_state(rivals, pit_context, traversal_s)

    # Total pit loss per draw: the lane traversal plus the physical stop. The legacy
    # scoring charged only the stop and argued in a comment that the traversal
    # cancels; here it is charged per car, so the cancellation happens exactly when
    # the rival really pays it too, and not otherwise.
    pit_loss_s = traversal_s + _np.asarray(pit_s, dtype=float)

    # The driver's position, preferably as reported, otherwise counted from the same gaps
    # the projection is about to use. It has to be derived rather than defaulted:
    # the featured parquet drops laps run under a Safety Car, so on exactly the
    # laps this layer matters most the driver row is missing and the position
    # arrives as None. Defaulting that to 1 claimed the driver was leading the race.
    # Counting rivals ahead is not a guess: it is the same arithmetic that
    # produces the projected position, so both sides of the delta agree.
    counted_position = 1 + sum(1 for state in rival_states if state.is_ahead)
    current_position = _position_or(position, counted_position)
    remaining = _lap_count_or_zero(laps_remaining)

    # An optional window onto the draws, for a surface that has to answer a
    # DIFFERENT question about the SAME stop. The rejoin readout is the first: it
    # prices where a stop taken now puts us, and a readout that sampled its own
    # pit loss would describe a stop the scored candidates never considered,
    # which is two stories about one lap on one screen.
    #
    # A dict passed IN, never a return value: the return is pinned by an
    # exact-equality golden, and widening it to carry a second product would make
    # every future consumer a golden change. Nothing here reads `capture` back,
    # so a caller that passes nothing pays nothing and the scored output cannot
    # depend on whether anyone was looking.
    if capture is not None:
        capture["pit_loss_s"] = pit_loss_s
        capture["rival_states"] = rival_states
        capture["current_position"] = current_position

    # Every constant below comes from the committed measurements
    # (data/mc_measured_v1.json, regenerated by scripts/measure_mc_tables.py). A
    # caller may override any of them through pit_context; what it may not do is
    # leave the onset rate at zero, which would tell the layer that no future
    # Safety Car can ever cover a stop and bias the terminal liability upward on
    # every lap of every race.
    onset_rate = context.get("neutralisation_rate")
    if onset_rate is None:
        onset_rate = measured_neutralisation_rate(context.get("gp_name"))
    q_f = future_neutralisation_probability(float(onset_rate), remaining)

    # ONE source for which neutralisation is active: the saving passed by the
    # caller already encodes it (VSC_PIT_BONUS vs SC_PIT_BONUS), so reading a
    # separate `vsc_active` here as well let the two disagree: a VSC saving
    # scored against a full-SC racing window.
    is_vsc = neutralisation_saving_s <= VSC_PIT_BONUS
    # `is None`, not `or`: zero racing laps is the Art. 55.17 endgame (the race
    # finishes behind the Safety Car), and it is the single most important value
    # a caller can pass here. Under `or` it was falsy and got replaced by the
    # measured average, so the case could not be expressed at all and a test that
    # passed 0.0 silently received 2.61 and went green for the wrong reason.
    override_racing_laps = _finite_or_none(context.get("racing_laps_neutralised"))
    if override_racing_laps is not None:
        racing_when_neutralised = float(override_racing_laps)
    else:
        racing_when_neutralised = measured_racing_laps("vsc" if is_vsc else "sc")

    # The window cannot outlast the race. A decision three laps from the flag
    # cannot bank five laps of racing, and under a neutralisation that runs to
    # the end it banks none at all, which is the Art. 55.17 endgame the docs
    # describe, and until this clamp existed the code could not express it: the
    # racing-lap count was always the measured average, so a stop always looked
    # as though it had laps left to pay itself back over.
    racing_when_racing = _bounded_by_race_end(float(WINDOW_LAPS), remaining)
    racing_when_neutralised = _bounded_by_race_end(racing_when_neutralised, remaining)

    # Clean air is worth something only to a car that was actually in the wake.
    # The measurement covers followers inside CLEAN_AIR_BAND_S of the car ahead,
    # so a rival boxing eight seconds up the road earns nothing here, and the
    # gain is zero under a neutralisation, where everyone runs to a delta and
    # clear track buys no lap time at all.
    clean_air_s = _clean_air_available(rival_states, context.get("gp_name"))

    def _config(racing_laps: float, clean_air_gain_s: float) -> ProjectionConfig:
        return ProjectionConfig(
            window_laps=WINDOW_LAPS,
            racing_laps=racing_laps,
            fresh_gain_s=FRESH_GAIN,
            # Set on BOTH configs the closure builds, green and neutralised. Setting
            # only the green one would leave every Safety Car lap on the old constant
            # while the report claimed the channel was connected.
            deg_cost_s=deg_cost_s,
            cliff_loss_s=CLIFF_LOSS,
            neutralisation_saving_s=neutralisation_saving_s,
            undercut_band_s=measured_undercut_band_s(),
            future_neutralisation_prob=q_f,
            laps_remaining=remaining,
            mandatory_stop_pending=context.get("mandatory_stop_pending"),
            clean_air_gain_s=clean_air_gain_s,
            neutralisation_onset_rate=float(onset_rate),
        )

    green_config = _config(racing_when_racing, clean_air_s)
    neutralised_config = _config(racing_when_neutralised, 0.0)

    # Eligible targets, ordered by where they will be once BOTH pit cycles have
    # played out rather than by where they sit on the timing screen now. That
    # ordering is the whole point of the far-field ranker (#439): the car actually
    # raced is not always the car currently in front, and picking the first
    # entry of an unordered list made "target" a coincidence of iteration order.
    ranking = rank_targets(rival_states, green_config, our_pit_loss_s=float(pit_loss_s.mean()))
    by_post_cycle_proximity = [target.driver for target in ranking]

    undercut_choices = _ordered_by(undercut_targets(rival_states, green_config), by_post_cycle_proximity)
    overcut_choices = _ordered_by(overcut_targets(rival_states), by_post_cycle_proximity)

    plans = {
        "STAY_OUT": DriverPlan("STAY_OUT", stops_in_window=False),
        "PIT_NOW": DriverPlan("PIT_NOW", stops_in_window=True, stop_offset_laps=0),
        "UNDERCUT": DriverPlan("UNDERCUT", stops_in_window=True, stop_offset_laps=0),
        # An overcut runs on for a lap while the target serves their stop, then boxes.
        "OVERCUT": DriverPlan("OVERCUT", stops_in_window=True, stop_offset_laps=1),
    }
    targets = {"UNDERCUT": undercut_choices, "OVERCUT": overcut_choices}

    neutralised = _np.asarray(sc_s, dtype=bool)
    results: dict = {}

    for name, plan in plans.items():
        choices = targets.get(name)
        if choices is not None and not choices:
            results[name] = {
                "E": None,
                "P10": None,
                "P90": None,
                "score": None,
                "eligible": False,
                "target": None,
            }
            continue

        green = project_positions(
            rival_states, plan, green_config, pit_loss_s, cliff_s, stop_is_neutralised=False
        )
        under_sc = project_positions(
            rival_states,
            plan,
            neutralised_config,
            pit_loss_s,
            cliff_s,
            stop_is_neutralised=True,
        )
        outcomes = _np.where(
            neutralised,
            payoff(under_sc, current_position, neutralised_config),
            payoff(green, current_position, green_config),
        )

        if name == "UNDERCUT":
            # N16 answers the one question it was trained on: does the undercut
            # actually clear the target? A success is worth the place, and the
            # projection supplies everything else. No new constant is introduced:
            # the alternative was inventing an out-lap delta nobody measured.
            #
            # Only on RACING draws. Under a neutralisation the move does not
            # exist: overtaking is prohibited (Art. 55.8), the field is queued
            # and everyone reaches the pit lane on the same delta, so there is no
            # advantage to arriving first. Granting it there was worth about half
            # a position on a fully neutralised state, a place awarded for a
            # manoeuvre the regulations forbid.
            landed = _np.asarray(ucut_s, dtype=float) * (~neutralised).astype(float)
            outcomes = outcomes + landed

        e_val = float(_np.mean(outcomes))
        p10_val = float(_np.percentile(outcomes, 10))
        p90_val = float(_np.percentile(outcomes, 90))
        results[name] = {
            "E": round(e_val, 3),
            "P10": round(p10_val, 3),
            "P90": round(p90_val, 3),
            "score": round(alpha * e_val + (1 - alpha) * p10_val, 3),
            "eligible": True,
            "target": choices[0] if choices else None,
        }

    return results


def _run_mc_simulation(
    pace_out,
    tire_out,
    situation_out,
    pit_out=None,
    alpha: float = 0.5,
    *,
    rivals: list[dict] | None = None,
    position: int | None = None,
    laps_remaining: int | None = None,
    pit_context: dict | None = None,
    capture: dict | None = None,
) -> dict:
    """Layer 2 Monte Carlo simulation over strategy candidates.

    Draws CFG.n_sim samples from the probability distributions exposed by the
    sub-agent outputs and evaluates each strategy over WINDOW_LAPS laps.

    pace_out:
        PaceOutput from N25, used to derive pace sigma from the bootstrap CI.
        σ = (ci_p90 − ci_p10) / (2 × 1.645). pace_i is sampled but not yet
        used inside simulate_lap_window, available for future extensions.
    tire_out:
        TireOutput from N26, provides P10/P50/P90 of laps-to-cliff for the
        Triangular distribution.
    situation_out:
        RaceSituationOutput from N27, sc_prob_3lap drives the Bernoulli SC draw.
    pit_out:
        PitStrategyOutput from N28, or None. When None, pit duration falls back
        to a conservative Triangular(2.2, 2.8, 3.8) prior and undercut_prob=0.5.
    alpha:
        RaceState.risk_tolerance. score = alpha·E[S] + (1−alpha)·P10[S].
        α=1.0 is pure expected value (aggressive); α=0.0 is worst-case only.
    capture:
        An optional dict the projection path deposits its draws into
        (``pit_loss_s``, ``rival_states``, ``current_position``). Write-only and
        defaulted, so the scored return is identical whether or not a caller
        passes one, and the goldens that pin that return by exact equality never
        see it. It exists because the rejoin readout has to price the SAME stop
        the candidates were scored on: a second sampling would put two answers
        about one lap on one screen. The legacy path deposits nothing, which is
        what makes an absent key the honest idle signal downstream.
    rivals / position / laps_remaining / pit_context:
        Race-context state for the projection-based scoring (#550). A TRUTHY
        ``rivals`` list routes to ``_run_projection_mc``, which scores in
        projected track position; a falsy value (None, or the ``[]`` the
        default lap_state builders emit) means "no per-rival gap data" and
        keeps the legacy seconds-based body below, byte-identical (the strategy
        goldens pin it to the digit). None means unknown, never zero rivals.
    """
    rng = np.random.default_rng(seed=42)
    n   = CFG.n_sim

    sigma_pace = (pace_out.ci_p90 - pace_out.ci_p10) / (2 * 1.645)

    # Clamp Triangular(left, mode, right) inputs so left < right always holds.
    # numpy raises ValueError("left == right") when the bounds collapse, which
    # happens at lap 1 because the N26 tire model and the N15 pit-duration
    # quantile regressors can return identical p10/p50/p90 when there is no
    # historical lap data yet (TyreLife=0). The clamp keeps the strategy
    # orchestrator running on the opening lap with a degenerate but valid
    # distribution instead of crashing the whole MC layer.
    def _clamp_triangular(p10: float, p50: float, p90: float,
                          eps: float = 1e-3) -> tuple[float, float, float]:
        left  = float(p10)
        mode  = float(p50)
        right = float(p90)
        if right <= left:
            right = left + eps
        if mode < left:
            mode = left
        if mode > right:
            mode = right
        return left, mode, right

    p10_cliff, p50_cliff, p90_cliff = _clamp_triangular(
        tire_out.laps_to_cliff_p10,
        tire_out.laps_to_cliff_p50,
        tire_out.laps_to_cliff_p90,
    )

    # `getattr` because a caller may still pass a stub TireOutput built before this
    # field existed; `None` then keeps both scorers on the FRESH_GAIN fallback, which
    # is the pre-#744b behaviour rather than a zero cost.
    deg_cost_s = getattr(tire_out, "deg_cost_s", None)

    sc_prob = situation_out.sc_prob_3lap

    # A VSC is a neutralisation too (N27 forces sc_prob_3lap to 1.0, so every draw sees
    # it), but it does NOT bunch the field the way a full SC does: gaps are preserved and
    # the restart is instant (Art. 56.5 / 56.7), so the relative pit-time saving is much
    # smaller. Charge the VSC its own (smaller) bonus instead of the full SC one; every
    # other draw is unchanged, so a green or full-SC state is byte-identical to before
    # (#471, and the "same 8 s for a VSC" bug in #470). vsc_active is False on any
    # RaceSituationOutput that predates the split, so old callers keep the SC bonus.
    sc_pit_bonus = VSC_PIT_BONUS if getattr(situation_out, "vsc_active", False) else SC_PIT_BONUS

    # A tool-parse failure inside N28 leaves the durations at 0.0 (its `or 0.0`
    # defaults), and 0.0 is not a stop time: it means "unknown". Simulating it makes
    # a pit stop FREE, so PIT_NOW (~+1.25 s) wins essentially every draw: a silent
    # parse miss becomes a confident recommendation to box. Treat a non-positive P50
    # as unavailable and fall through to the same prior the pit_out-is-None branch
    # already uses (#436). Durations and undercut_prob degrade INDEPENDENTLY: a
    # failed duration parse says nothing about the undercut probability.
    _durations_known = pit_out is not None and (pit_out.stop_duration_p50 or 0.0) > 0.0
    if _durations_known:
        pit_p05, pit_p50, pit_p95 = _clamp_triangular(
            pit_out.stop_duration_p05,
            pit_out.stop_duration_p50,
            pit_out.stop_duration_p95,
        )
    else:
        pit_p05, pit_p50, pit_p95 = 2.2, 2.8, 3.8

    ucut_prob = (
        pit_out.undercut_prob
        if pit_out is not None and pit_out.undercut_prob is not None
        else 0.5
    )

    # pace_s is drawn and never read, and deleting it is NOT free. F841 says the
    # NAME is unused; the DRAW is not. It takes n values off rng, so removing the
    # line shifts cliff_s, sc_s, pit_s and ucut_s to different numbers, which
    # re-rolls every candidate score and every golden in tests/mc. The line stays
    # until someone re-blesses the projection path on purpose. To see it, draw the
    # same triangular twice off one seed, once after an n-sized normal and once
    # without: the two triples share no value.
    pace_s  = rng.normal(pace_out.lap_time_pred, sigma_pace, n)  # noqa: F841
    cliff_s = rng.triangular(p10_cliff, p50_cliff, p90_cliff, n)
    sc_s    = rng.random(n) < sc_prob
    pit_s   = rng.triangular(pit_p05, pit_p50, pit_p95, n)
    ucut_s  = rng.random(n) < ucut_prob

    # Common random numbers: every candidate is scored on the SAME draw vectors, so
    # the comparison between them carries no sampling noise of its own. That is what
    # an argmax over 500 draws needs: the variance that matters is the variance of
    # the DIFFERENCES, and sharing the draws collapses it.
    if _has_usable_gaps(rivals):
        return _run_projection_mc(
            rivals=rivals,
            position=position,
            laps_remaining=laps_remaining,
            pit_context=pit_context,
            cliff_s=cliff_s,
            sc_s=sc_s,
            pit_s=pit_s,
            ucut_s=ucut_s,
            alpha=alpha,
            neutralisation_saving_s=sc_pit_bonus,
            deg_cost_s=deg_cost_s,
            capture=capture,
        )

    strategies = ["STAY_OUT", "PIT_NOW", "UNDERCUT", "OVERCUT"]
    results    = {}

    for s in strategies:
        outcomes = np.array([
            simulate_lap_window(s, cliff_s[i], sc_s[i], pit_s[i], ucut_s[i],
                                sc_pit_bonus=sc_pit_bonus, deg_cost_s=deg_cost_s)
            for i in range(n)
        ])
        e_val   = float(np.mean(outcomes))
        p10_val = float(np.percentile(outcomes, 10))
        p90_val = float(np.percentile(outcomes, 90))
        score   = alpha * e_val + (1 - alpha) * p10_val
        results[s] = {
            "E":     round(e_val, 3),
            "P10":   round(p10_val, 3),
            "P90":   round(p90_val, 3),
            "score": round(score, 3),
        }

    return results


# ==============================================================================
# Layer 3: LLM synthesis
# ==============================================================================

# Lock detection lives in `src/rag/store_lock.py`, not here, and moving it IS the
# fix for a defect this guard already carried twice. Reached through this module
# it can only be tested by importing the whole agent stack, which reads four
# artefacts at IMPORT time - so the tests pinning the behaviour skipped on CI and
# ERRORED on the tree `scripts/download_data.py` actually produces. See that
# module's header for the qdrant behaviour itself.
from src.rag.store_lock import LOCK_EXCEPTIONS as _LOCK_EXCEPTIONS
from src.rag.store_lock import QDRANT_LOCK_SIGNATURE as _QDRANT_LOCK_SIGNATURE  # noqa: F401
from src.rag.store_lock import is_store_locked as _is_store_locked



# Logged once per process, not once per lap. A locked store fails every lap
# identically, and 60 identical WARNING lines is how a configuration problem
# disguises itself as flaky data.
_rag_unavailable_logged = False


def _run_rag_agent_or_degrade(question: str):
    """N30's answer, or None when the regulation store cannot be opened.

    The store is local single-writer (`QdrantClient(path=...)`), so a second
    concurrent process holds an exclusive lock and every retrieval in this one
    fails. Before #827 that exception propagated out of `run_lap`, and the CLI's
    per-lap `except Exception` rendered it as a red row, for an entire race, with
    the cause named nowhere: three races in one measurement run produced zero rows
    while their batches walked every window and reported success.

    Degrading is the right direction because N30 is an ENRICHMENT: it is one of
    fourteen recommendation fields and the orchestrator already has a path for it
    not being routed at all. An unavailable store should cost the regulation block,
    not the recommendation.

    Only the lock is caught, via :func:`_is_store_locked`. A corrupt collection, a
    missing embedding model or a bad question must still surface: those are not
    "another process is running", and swallowing them would trade one silent
    failure for another.
    """
    global _rag_unavailable_logged
    try:
        return run_rag_agent(question)
    except (RuntimeError, *_LOCK_EXCEPTIONS) as exc:
        if not _is_store_locked(exc):
            raise
        if not _rag_unavailable_logged:
            logger.warning(
                "N30 disabled for this process: the regulation store is locked by "
                "another process (%s). Recommendations continue WITHOUT regulation "
                "context. Give each process its own store via F1_STRAT_DATA_ROOT, or "
                "run one at a time. This is logged once, not per lap.",
                exc,
            )
            _rag_unavailable_logged = True
        return None


def _build_rag_question(
    sc_active:  bool,
    pit_action: str | None,
    compound:   str,
) -> str:
    """Generate a targeted FIA regulation query based on active race conditions.

    sc_active triggers a safety car procedure query. pit_action drives a
    compound-change or undercut-specific query. Falls back to a generic
    pit stop regulation question when neither condition is specific.
    """
    if sc_active:
        # Asks what the PROCEDURE is, not what the RESTRICTIONS are. The old
        # wording presupposed that restrictions exist, and under an ordinary
        # mid-race Safety Car they do not: the pit lane is open and stopping is
        # the whole point. A question that assumes a constraint pulls the
        # retriever toward the one article mentioning Safety Cars and tyre
        # specification together (Art. 30.5 n), a wet-start and resumption rule)
        # and invites the answer to drop its condition (#826).
        return (
            "What is the procedure for drivers and teams when the safety car is "
            "deployed during a race? Describe what drivers must do, and state any "
            "conditions that limit when each requirement applies."
        )
    if pit_action == "UNDERCUT":
        return (
            f"Are there any restrictions on changing to {compound} compound "
            "tyres mid-race?"
        )
    return "What are the mandatory tyre compound regulations for a dry race?"


def _build_tire_block(tire_out) -> str:
    """N26's numeric block for the prompt, verbatim so the LLM can cite it.

    Extracted from the prompt builder when the wear term was added (#727), so the
    formatting decision below can be tested without assembling a whole prompt.

    ``wear`` leads because it is the model's actual output: seconds per lap this
    set is slower than it was fresh, fuel-corrected. ``deg_rate`` stays because
    the LLM has cited it for two years and removing it silently would change the
    synthesis for reasons unrelated to this change; it is the raw, uncorrected
    lap-to-lap derivative and is close to noise (median +0.006 s/lap over 110
    measured laps, negative on 43 of them).

    A missing prediction prints ``unknown``, never ``0.000``. Zero is a real
    reading for this quantity (a set sitting at its fresh baseline), so printing
    it for an absent one would tell the LLM the tyre is pristine at the exact
    moment the model failed to say anything.
    """
    if tire_out is None:
        return "  [N26 Tire]      not activated"

    cumulative = getattr(tire_out, "cumulative_deg_s", None)
    wear = f"{cumulative:+.3f}s/lap vs fresh" if cumulative is not None else "unknown"
    return (
        f"  [N26 Tire]      wear={wear}  "
        f"deg_rate={tire_out.deg_rate:.3f}s/lap  "
        f"cliff P10={tire_out.laps_to_cliff_p10:.1f}  "
        f"P50={tire_out.laps_to_cliff_p50:.1f}  "
        f"P90={tire_out.laps_to_cliff_p90:.1f}  "
        f"[{tire_out.warning_level}]\n"
        f"                  reasoning: {tire_out.reasoning or '(empty)'}"
    )


def _build_orchestrator_prompt(
    race_state:          "RaceState",
    mc_results:          dict,
    best_mc:             str,
    pace_out             = None,
    tire_out             = None,
    situation_out        = None,
    pit_out              = None,
    radio_out            = None,
    regulation_context:  str = "",
    memory_block:        str = "",
) -> str:
    """Build the LLM synthesis prompt for Layer 3.

    Assembles every sub-agent's structured numeric output plus its reasoning
    string, the Monte Carlo scenario scores, and the N30 regulation context
    into a single prompt. The v2 prompt exposes the full numeric grounding
    (Pace CI, Tire cliff percentiles, N27 probabilities, N28 stop and undercut
    data) so the LLM can fill the expanded StrategyRecommendation schema
    without having to reverse-engineer values from the reasoning strings.

    N30 regulation context is injected as a hard constraint block: the LLM is
    told explicitly which actions are regulation-compliant before it decides,
    so illegal options cannot appear in the output.

    best_mc is the MC argmax passed as a hint. The LLM may override it if
    regulation context, radio alerts, or a planned contingency justify a
    different action.

    memory_block is what this race's own previous recommendations looked like,
    rendered by DecisionMemory and empty by default. Empty is not a degraded
    mode: two of the three call sites (/recommend and the MCP tool) are
    stateless per request and have no race-scoped object to accumulate on, so
    they keep this prompt exactly as it was. The default therefore has to
    produce a byte-identical prompt, which tests/agents/test_orchestrator_prompt
    asserts.

    Measured effect when it IS supplied, on a client that honours temperature=0:
    under a Safety Car at Lusail 2025 lap 42, the model acted on a contingency it
    had itself declared one lap earlier on 8 of 8 runs, against 0 of 8 without
    the block (Fisher p=0.000155). It changes whether consecutive laps are the
    same plan, not what any single lap decides.
    """
    mc_table = "\n".join(_format_mc_row(name, cell) for name, cell in mc_results.items())

    reg_block = (
        f"REGULATION CONSTRAINT (hard — exclude non-compliant actions):\n"
        f"{regulation_context}"
        if regulation_context
        else "REGULATION CONSTRAINT: none flagged — all four actions are compliant."
    )

    # Pace CI bounds rendered into the prompt guidance: keep a safe fallback
    # when pace_out is unavailable so the format string never crashes.
    pace_ci_lo = pace_out.ci_p10 if pace_out is not None else 0.0
    pace_ci_hi = pace_out.ci_p90 if pace_out is not None else 0.0

    # ── Sub-agent numeric blocks: verbatim numbers so the LLM can cite them ──
    if pace_out is not None:
        pace_block = (
            f"  [N25 Pace]      pred={pace_out.lap_time_pred:.3f}s  "
            f"Δprev={pace_out.delta_vs_prev:+.3f}s  "
            f"Δmedian={pace_out.delta_vs_median:+.3f}s  "
            f"CI=[{pace_out.ci_p10:.3f}, {pace_out.ci_p90:.3f}]\n"
            f"                  reasoning: {pace_out.reasoning or '(empty)'}"
        )
    else:
        pace_block = "  [N25 Pace]      not activated"

    tire_block = _build_tire_block(tire_out)

    if situation_out is not None:
        # Spelled out rather than rendered as a number, because this line is read by the
        # LLM that writes the recommendation. N11 never saw a labelled pair beyond 2.5 s,
        # so on those laps there is no probability to quote, and a synthesised one here
        # does not get averaged away, it gets ARGUED from. Saying "unknown, the cars are
        # too far apart for the model" is a fact the model can reason with; a fabricated
        # 0.07 is one it cannot tell from a measurement.
        overtake_text = (
            "unknown (cars farther apart than the model's trained range)"
            if situation_out.overtake_prob is None
            else f"{situation_out.overtake_prob:.2f}"
        )
        sit_block = (
            f"  [N27 Situation] overtake={overtake_text}  "
            f"sc_3lap={situation_out.sc_prob_3lap:.2f}  "
            f"threat={situation_out.threat_level}\n"
            f"                  reasoning: {situation_out.reasoning or '(empty)'}"
        )
    else:
        sit_block = "  [N27 Situation] not activated"

    if pit_out is not None:
        ucut_str = (
            f"{pit_out.undercut_prob:.2f}→{pit_out.undercut_target}"
            if pit_out.undercut_prob is not None and pit_out.undercut_target
            else "n/a"
        )
        pit_block = (
            f"  [N28 Pit] ★     action={pit_out.action}  "
            f"rec_lap={pit_out.recommended_lap}  "
            f"compound_next={pit_out.compound_recommendation}  "
            f"stop=[{pit_out.stop_duration_p05:.2f}, "
            f"{pit_out.stop_duration_p50:.2f}, "
            f"{pit_out.stop_duration_p95:.2f}]s  "
            f"undercut={ucut_str}  "
            f"sc_reactive={pit_out.sc_reactive}\n"
            f"                  reasoning: {pit_out.reasoning or '(empty)'}"
        )
    else:
        pit_block = "  [N28 Pit]       not activated (no cliff pressure, no radio problem)"

    if radio_out is not None:
        n_radio  = len(getattr(radio_out, "radio_events", []) or [])
        n_rcm    = len(getattr(radio_out, "rcm_events",   []) or [])
        n_alerts = len(getattr(radio_out, "alerts",       []) or [])
        radio_block = (
            f"  [N29 Radio]     radio={n_radio}  rcm={n_rcm}  alerts={n_alerts}\n"
            f"                  reasoning: {radio_out.reasoning or '(empty)'}"
        )
    else:
        radio_block = "  [N29 Radio]     not activated"

    return (
        f"You are the F1 Strategy Orchestrator. Synthesise the sub-agent outputs below\n"
        f"into a single StrategyRecommendation. Choose the primary action that maximises\n"
        f"risk-adjusted position gain while respecting the regulation constraint, and fill\n"
        f"every structured field so the strategy can be executed without parsing your prose.\n\n"
        f"CRITICAL: the Monte Carlo score is ONE input among many. Your decision must\n"
        f"weigh it against the tire cliff distance, the specific rival gap and pace delta,\n"
        f"any radio or RCM alerts, and the regulation constraint. Never justify a call\n"
        f"with MC numbers alone — cite at least one tire, one situation or radio, and\n"
        f"(if present) one regulation signal. If evidence across agents disagrees, say so\n"
        f"and explain which signal you trusted and why.\n\n"
        f"STRATEGIC GUARD-RAILS (HARD — override any sub-agent or MC signal that conflicts):\n"
        f"  1. NO pit action (PIT_NOW / UNDERCUT / OVERCUT) before lap {_NO_PIT_BEFORE_LAP} "
        f"unless SC deployed\n"
        f"     or damage/puncture confirmed by radio. Fresh tyres cannot degrade in 1-4 laps;\n"
        f"     pit lane costs ~22-25s which is unrecoverable this early. Force STAY_OUT.\n"
        f"  2. NO pit action when remaining laps <= {_NO_PIT_LAST_N_LAPS} unless tyre failure "
        f"imminent\n"
        # ~9, not ~13. The 13 was 20 s / POS_GAP_S(1.50), and POS_GAP_S is the legacy
        # scoring constant `measure_mc_tables.py` records as retired. The MEASURED
        # median gap between consecutive cars under green flag is 2.227 s (n=69,487),
        # which puts 20 s at about nine places; 1.4795 s, and therefore thirteen, is
        # the SAFETY CAR bunched-field figure (n=3,658) and is the exception this rule
        # excludes. The pit agent's prompt already said exactly that, so the two
        # prompts were teaching one orchestrating LLM contradictory numbers in the
        # same run (#766).
        f"     (cliff P10 < {_CLIFF_P10_SAFE} laps). Pit cost ~22s vs ~1.5s recovery = "
        f"~9 positions lost\n"
        f"     under green flag (measured median gap 2.227s). The ~13 figure is the\n"
        f"     Safety Car bunched field (1.4795s), which is the exception above.\n"
        f"  3. REACTIVE_SC only when SC IS deployed (confirmed). High sc_prob is a\n"
        f"     contingency trigger, not a primary action — use STAY_OUT with SC contingency.\n"
        f"  4. Minimum stint before pit: SOFT >= {_MIN_STINT_LAPS['SOFT']} laps, "
        f"MEDIUM >= {_MIN_STINT_LAPS['MEDIUM']}, HARD >= {_MIN_STINT_LAPS['HARD']}.\n"
        f"     If tyre_life is below minimum, override to STAY_OUT (current set has life left).\n"
        # SOFT bound derived from _STINT_CAPACITY_LAPS (imported above), not restated:
        # this line and the pit agent's own system prompt used to disagree with the
        # deterministic selector's table (18 vs 15), so laps_remaining=16-18 had the
        # selector pass SOFT while both prompts told the LLM to refuse it (#741).
        f"  5. Compound must fit remaining laps: SOFT only if <= {_STINT_CAPACITY_LAPS['SOFT']} laps remain,\n"
        f"     MEDIUM for {_MEDIUM_SUITABILITY_FLOOR_LAPS}-{_STINT_CAPACITY_LAPS['MEDIUM']}, "
        f"HARD for 20+. Wrong compound forces an extra stop.\n"
        f"  6. Opening laps 1-3: threat levels from N27 are inflated by start chaos.\n"
        f"     Discount them one tier (HIGH→MEDIUM, MEDIUM→LOW) for decision-making.\n"
        f"  If a sub-agent recommends an action that violates these rules, override to\n"
        f"  STAY_OUT and explain why in reasoning.\n\n"
        # Everything above reaches STAY_OUT by subtraction: "force", "override to",
        # "no pit action". Measured on real races it is the call on the large majority
        # of green-flag laps (39/41 at Lusail 2025, 414/415 across the projection set),
        # so the prompt was teaching the LLM to treat its own most frequent output as a
        # failure to decide.
        #
        # What stays here is what is true on EVERY lap. The instruction to treat a
        # repeated STAY_OUT as a continuing plan used to sit in this block, where it
        # fired on lap 1, when there was nothing to continue, and on the lap the call
        # changed, when continuing was wrong. It now lives in the memory block, which
        # is the only place that knows whether anything is being repeated.
        f"WHEN THE CALL IS STAY_OUT (the majority call, by design):\n"
        f"  STAY_OUT is an ACTIVE monitoring posture, not the absence of a decision and\n"
        f"  not merely what is left when a pit is blocked. State (a) what you are watching,\n"
        f"  (b) the concrete threshold that would change the call, and (c) how far the\n"
        f"  current numbers sit from it.\n"
        f"  Carry the lap you are watching towards in pit_lap_target whenever the tyre\n"
        f"  data supports one, so a hold reads as a plan with a horizon, not indecision.\n\n"
        # Placed after the framing and before the per-lap facts, so the model reads
        # what it decided before it reads this lap's numbers. `or ""` because the
        # producer, DecisionMemory.block(), returns None before there is any history
        # and an unguarded f-string would render the literal "None" into the prompt.
        f"{memory_block or ''}"
        f"RACE CONTEXT:\n"
        f"  Driver: {race_state.driver} | Lap: {race_state.lap}/{race_state.total_laps}\n"
        f"  Position: P{race_state.position} | Compound: {race_state.compound} "
        f"TyreLife {race_state.tyre_life}\n"
        f"  {_gap_ahead_context_line(race_state)} | "
        f"Pace delta: {race_state.pace_delta_s:+.3f}s\n"
        f"  Air {race_state.air_temp:.1f}°C | Track {race_state.track_temp:.1f}°C | "
        f"Rain {race_state.rainfall}\n"
        f"  Risk tolerance α: {race_state.risk_tolerance}\n\n"
        f"SUB-AGENT OUTPUTS:\n"
        f"{pace_block}\n"
        f"{tire_block}\n"
        f"{sit_block}\n"
        f"{pit_block}\n"
        f"{radio_block}\n\n"
        f"MONTE CARLO SCENARIO SCORES "
        f"(N_SIM={CFG.n_sim}, α={race_state.risk_tolerance}, window={WINDOW_LAPS} laps):\n"
        f"{mc_table}\n"
        f"  → Best MC candidate: {best_mc}\n\n"
        f"{reg_block}\n\n"
        f"REASONING RUBRIC — your ``reasoning`` field MUST follow this structure:\n"
        f"  1. Open with the chosen action and WHY the tire/pace numbers drive it\n"
        f"     (cite the cliff P50 lap AND the pace delta explicitly).\n"
        f"  2. Add a situational line: overtake prob, SC prob, or radio alert intent\n"
        f"     if any of those meaningfully shaped the decision — name the signal.\n"
        # Conditional on an article actually being there. The unconditional form
        # manufactured a citation on every lap where the RAG answered without one,
        # and it is the instruction that made a re-scoped rule sound authoritative
        # enough to override the Monte Carlo (#826).
        f"  3. If regulation_context cites an article, quote it AND the condition\n"
        f"     under which that article applies. If it states no condition, say the\n"
        f"     regulation context was inconclusive rather than inventing one, and\n"
        f"     do NOT let it override the numeric evidence.\n"
        f"  4. Close with how the MC score either confirms or is overridden by the\n"
        f"     evidence above. Never start with MC.\n"
        f"Example of a rich reasoning paragraph (do NOT copy verbatim, use as shape):\n"
        f"  \"PIT_NOW on lap 22: tire cliff P50 sits at lap 24 and pace has already\n"
        f"  lost +0.42s/lap vs session median, so the window is closing. SC\n"
        f"  probability (0.38) is below the threshold so we cannot wait for a free\n"
        f"  stop. Radio intent 'BLISTER' on RUS confirms the front-left is gone.\n"
        # No article number in this example on purpose. The two-compound rule is
        # renumbered between seasons (30.5(n) in 2023, 30.5(m) in 2024, 30.5(i) in 2025,
        # from the corpus PDFs), and this block asks the LLM to cite article numbers, so a
        # hardcoded one would be echoed into the output and wrong for most years. N30
        # reads the season's own regulations; the article should come from that context.
        # The example must OBEY rule 3, because an exemplar marked "use as shape"
        # beats a rule it contradicts. It used to state the two-compound rule with no
        # condition at all - and that rule is itself conditional (it does not bind if
        # wet-weather tyres were used, and it is a dry-race rule), so the example was
        # unconditioned about a CONDITIONAL rule, which is the exact failure #826 is
        # about, shipped inside the fix for it.
        f"  The mandatory two-compound rule (see the regulation context above for the\n"
        f"  article, it is renumbered between seasons) requires no fewer than two dry\n"
        f"  compounds used IN A DRY RACE, and this is a dry race with no wet tyre so\n"
        f"  far, so the condition holds and switching to HARD satisfies it. MC ranks PIT_NOW first\n"
        f"  (score +0.81) and the tire and radio evidence reinforce that call.\"\n\n"
        f"Return a StrategyRecommendation filling EVERY field:\n"
        f"  action:             one of STAY_OUT / PIT_NOW / UNDERCUT / OVERCUT / ALERT.\n"
        f"                       Do not invent new values.\n"
        f"  reasoning:          3-5 sentences following the rubric above. At minimum\n"
        f"                       reference tire cliff P50, pace delta, one situation\n"
        f"                       or radio signal, and (when present) one regulation\n"
        f"                       article. MC score is the closing confirmation, not\n"
        f"                       the opening argument.\n"
        f"  confidence:         your certainty in [0, 1] after weighing MC and regulation.\n"
        f"  pit_lap_target:     absolute lap number of the planned stop. None only if the\n"
        f"                       plan is to stay out beyond the visible horizon.\n"
        f"  compound_next:      SOFT / MEDIUM / HARD for the next stint. None only for\n"
        f"                       STAY_OUT with no planned stop.\n"
        f"  undercut_target:    rival code (e.g. SAI). Non-None only for UNDERCUT/OVERCUT.\n"
        f"                       Prefer N28.undercut_target when available.\n"
        f"  pace_mode:          PUSH | NEUTRAL | MANAGE | LIFT_AND_COAST. Choose PUSH when\n"
        f"                       attacking a close rival, MANAGE when defending a gap,\n"
        f"                       LIFT_AND_COAST only when tyre warning is PIT_SOON.\n"
        f"  target_lap_time_s:  concrete target lap time, inside PaceOutput CI "
        f"[{pace_ci_lo:.3f}, {pace_ci_hi:.3f}] if available. None if you\n"
        f"                       prefer not to commit to a number.\n"
        f"  risk_posture:       AGGRESSIVE / BALANCED / DEFENSIVE. Align with the\n"
        f"                       position and gap — leaders defend, midfield balances,\n"
        f"                       chasers attack.\n"
        f"  contingencies:      up to 4 Contingency entries. Each must have a concrete\n"
        f"                       trigger (e.g. 'SC deployed within 3 laps', 'gap to "
        f"<rival> drops below 0.8 s'), a switch_to action, a HIGH/MEDIUM/LOW priority,\n"
        f"                       and a one-line rationale tied to a sub-agent number.\n"
        f"  key_risks:          up to 5 short bullets flagging the top risks (tyre cliff\n"
        f"                       timing, SC probability spikes, regulation gaps, etc.).\n"
        f"  expected_stint_end: lap at which you expect the current stint to end. Use the\n"
        f"                       tyre cliff P50 as a baseline and adjust for strategy.\n"
    )


# ==============================================================================
# Helpers: input coercion
# ==============================================================================

def _to_radio_message(item) -> RadioMessage:
    """Convert a dict or RadioMessage instance to a RadioMessage.

    Accepts both RadioMessage dataclass instances (passed through unchanged)
    and dicts with keys driver, lap, text. Used so callers can pass either
    type in RaceState.radio_msgs without explicit conversion.
    """
    if isinstance(item, RadioMessage):
        return item
    return RadioMessage(
        driver=item.get("driver", "UNK"),
        lap=item.get("lap", 0),
        text=item.get("text", ""),
        timestamp=item.get("timestamp"),
    )


def _to_rcm_event(item) -> RCMEvent:
    """Convert a dict or RCMEvent instance to a RCMEvent.

    Accepts both RCMEvent dataclass instances (passed through unchanged) and
    dicts with keys message, flag, category, lap. Used so callers can pass
    FastF1 RCM row dicts directly into RaceState.rcm_events.
    """
    if isinstance(item, RCMEvent):
        return item
    return RCMEvent(
        message=str(item.get("message", "")),
        flag=str(item.get("flag", "") or ""),
        category=str(item.get("category", "")),
        lap=int(item.get("lap", 0) or 0),
        racing_number=item.get("racing_number") or item.get("RacingNumber"),
        scope=str(item.get("scope", "") or ""),
    )


# ==============================================================================
# Entry point helpers
# ==============================================================================

def _run_always_on_agents(race_state: "RaceState", lap_state: dict) -> tuple:
    """Run N25, N26, N27, N29: always activated regardless of race state.

    race_state:
        Current RaceState with all lap and session fields.
    lap_state:
        Dict of scalar lap features consumed by the sub-agent entry points.
        Must contain: driver_number, stint, team, year, gp_name and optionally
        laps_since_pit, fuel_load, prev_lap_time, prev_speed_st, humidity.

    Returns (pace_out, tire_out, situation_out, radio_out), typed dataclass
    outputs from N25, N26, N27, N29 respectively.
    """
    pace_out = run_pace_agent(
        driver_number  = lap_state["driver_number"],
        lap_number     = race_state.lap,
        stint          = lap_state["stint"],
        tyre_life      = race_state.tyre_life,
        compound       = race_state.compound,
        position       = race_state.position,
        team           = lap_state["team"],
        laps_since_pit = lap_state.get("laps_since_pit", race_state.tyre_life),
        fuel_load      = lap_state.get(
            "fuel_load", 1 - race_state.lap / race_state.total_laps
        ),
        year           = lap_state["year"],
        # `or`, NOT the two-arg get(key, default): `RaceStateManager` emits this key
        # PRESENT with a None value whenever no surviving predecessor exists, and the
        # two-arg form substitutes only for an absent KEY. `_predict` reads the value
        # straight into `prev + delta` with no NaN branch, so a None reaching it turns
        # the prediction into NaN and then raises on the subtraction. The twin of this
        # line in `pace_agent` carries a fifteen-line comment saying exactly that; this
        # one carried a bare 92.0 and got neither the form nor the same number (#766).
        prev_lap_time  = lap_state.get("prev_lap_time") or MISSING_PREV_LAP_TIME_S,
        prev_tyre_life = race_state.tyre_life - 1,
        prev_speed_st  = lap_state.get("prev_speed_st", 300.0),
        air_temp       = race_state.air_temp,
        track_temp     = race_state.track_temp,
        humidity       = lap_state.get("humidity", 50.0),
        rainfall       = race_state.rainfall,
        total_laps     = race_state.total_laps,
        gp_name        = lap_state["gp_name"],
    )

    tire_out = run_tire_agent(lap_state)

    # Build the RCM list once so it can feed BOTH the radio agent
    # (its primary consumer) and the situation agent (so the SC override
    # in N27 can flip sc_currently_active when a SAFETY_CAR_DEPLOYED is
    # active in this lap window).
    radio_msgs    = [_to_radio_message(m) for m in race_state.radio_msgs]
    rcm_events    = [_to_rcm_event(e) for e in race_state.rcm_events]
    situation_out = run_race_situation_agent({**lap_state, "rcm_events": rcm_events})
    radio_out     = run_radio_agent({
        **lap_state,
        "lap":        race_state.lap,
        "radio_msgs": radio_msgs,
        "rcm_events": rcm_events,
    })

    return pace_out, tire_out, situation_out, radio_out


def _run_always_on_agents_from_state(
    race_state: "RaceState",
    laps_df:    pd.DataFrame,
    lap_state:  dict,
) -> tuple:
    """RSM adapter version of _run_always_on_agents.

    N25 (pace, XGBoost) and N27 (situation, LightGBM + HTTP LLM) are I/O-bound
    and share no mutable state, so they run in parallel threads. N26 (tire, TCN +
    HTTP LLM) and N29 (radio, NLP + HTTP LLM) run sequentially to avoid potential
    PyTorch/MLX thread-safety issues with their model inference layers.

    Returns (pace_out, tire_out, situation_out, radio_out).
    """
    radio_msgs      = [_to_radio_message(m) for m in race_state.radio_msgs]
    rcm_events      = [_to_rcm_event(e) for e in race_state.rcm_events]
    radio_lap_state = {**lap_state, "lap": race_state.lap,
                       "radio_msgs": radio_msgs, "rcm_events": rcm_events}
    # N27 needs the RCM events too so the SC override fires when the
    # SAFETY_CAR_DEPLOYED message is in the current lap window.
    sit_lap_state   = {**lap_state, "rcm_events": rcm_events}

    # N25 + N27 in parallel (no shared PyTorch state)
    with ThreadPoolExecutor(max_workers=2) as pool:
        fut_pace = pool.submit(run_pace_agent_from_state, lap_state)
        fut_sit  = pool.submit(run_race_situation_agent_from_state, sit_lap_state, laps_df)
        pace_out      = fut_pace.result()
        situation_out = fut_sit.result()

    # N26 + N29 sequential (PyTorch/MLX inference)
    tire_out  = run_tire_agent_from_state(lap_state, laps_df)
    radio_out = run_radio_agent_from_state(radio_lap_state, laps_df)

    return pace_out, tire_out, situation_out, radio_out


def _run_conditional_agents(
    active:       set,
    lap_state:    dict,
    tire_out,
    situation_out,
    race_state:   "RaceState",
    laps_df:      pd.DataFrame | None = None,
) -> tuple:
    """Run N28 and N30 when the routing layer activates them.

    active:
        Set of agent names from _decide_agents_to_call ('N28', 'N30').
    lap_state:
        Scalar lap feature dict, extended with laps_to_cliff and sc_prob
        before being forwarded to N28.
    tire_out:
        TireOutput from N26, provides cliff timing for N28.
    situation_out:
        RaceSituationOutput from N27, provides sc_prob for N28 and the N30
        routing decision.
    race_state:
        Full RaceState used to build the FIA regulation query for N30.
    laps_df:
        When provided, N28 is called via run_pit_strategy_agent_from_state.
        When None, run_pit_strategy_agent is used (FastF1 entry point).

    Returns ``(pit_out, regulation_context_str, rag_dict)``. ``pit_out`` and
    ``regulation_context_str`` may be ``None`` when the respective agent was
    not activated this lap; ``rag_dict`` is the structured payload from N30
    (``question`` / ``answer`` / ``articles`` / ``chunks``) for downstream
    consumers that need more than just the answer string (the arcade
    dashboard surfaces article references and chunk text in its RAG card).
    The legacy ``regulation_context_str`` is preserved verbatim for the
    orchestrator's own LLM prompt and for ``StrategyRecommendation``,
    neither of which depend on the structured shape.
    """
    pit_out = None
    if "N28" in active:
        pit_lap_state = {
            **lap_state,
            "laps_to_cliff":       tire_out.laps_to_cliff_p50,
            "sc_prob":             situation_out.sc_prob_3lap,
            # N28 reads this flag to (a) replace the "SC probability" line in
            # its prompt with the deploy banner, (b) bypass the minimum-stint
            # guard, and (c) trip the post-LLM STAY_OUT→PIT_NOW guard-rail.
            "sc_currently_active": situation_out.sc_currently_active,
            # ...and this one so the banner names a VSC as a VSC: under Art. 56 the
            # field is not queued, so a stop saves much less than under a full SC (#471).
            "vsc_active":          situation_out.vsc_active,
        }
        if laps_df is not None:
            pit_out = run_pit_strategy_agent_from_state(pit_lap_state, laps_df)
        else:
            pit_out = run_pit_strategy_agent(pit_lap_state)

    regulation_context: str | None = None
    rag_dict: dict | None          = None
    # None means "no regulation context this lap", which happens two ways that the
    # rest of the function does not need to tell apart: N30 was not routed, or it
    # was routed and the store was unavailable (#827).
    reg_out = None
    if "N30" in active:
        pit_action = pit_out.action if pit_out else None
        question   = _build_rag_question(
            # A deployed SC is a FACT, not a forecast. This used to key off
            # `sc_prob_3lap > threshold`, but N13/N14 predicts an SC *within the next 3
            # laps*: while one is already out, that forward probability can sit below the
            # threshold, and N30 would then ask the green-flag question and hand the
            # orchestrator a "hard regulation constraint" block describing the wrong
            # race. `sc_currently_active` is N27's observation of the RCM feed and it was
            # already in scope twelve lines above (`:1166`), passed to N28 and dropped
            # here, the same restored-datum-with-an-unswitched-consumer shape as #447.
            #
            # Keep the forecast as well: an SC that is merely likely still changes which
            # articles matter.
            sc_active  = (
                situation_out.sc_currently_active
                or situation_out.sc_prob_3lap > CFG.sc_prob_threshold
            ),
            pit_action = pit_action,
            compound   = race_state.compound,
        )
        reg_out = _run_rag_agent_or_degrade(question)

    if reg_out is not None:
        regulation_context = reg_out.answer
        rag_dict = {
            "question": reg_out.question,
            "answer":   reg_out.answer,
            "articles": list(reg_out.articles),
            "chunks":   [
                {
                    "text":          c.text,
                    "article":       c.article,
                    "doc_type":      c.doc_type,
                    "year":          c.year,
                    "score":         c.score,
                    "section_title": c.section_title,
                }
                for c in reg_out.chunks
            ],
        }

    return pit_out, regulation_context, rag_dict


# ==============================================================================
# Post-LLM assembly helper
# ==============================================================================

def _live_drivers_from(lap_state: dict | None) -> set | None:
    """The cars on track this lap, or None when that cannot be told.

    RaceStateManager builds ``rivals`` from the per-lap rows, so a car that has retired
    is simply absent: the same answer a timing screen gives, and the only sound one.
    No staleness threshold works, because the featured frame drops SC, pit and out laps,
    so a car that FINISHED can have its last known lap lag by 20 while a retirement shows
    up at 9: the ranges overlap (#462).

    Returns None rather than an empty set when there is no lap_state, so callers can tell
    "nobody is racing" from "unknown" and fall through instead of rejecting all.
    """
    if not lap_state:
        return None
    rivals = lap_state.get('rivals') or []
    live = {r['driver'] for r in rivals if r.get('driver')}
    own = (lap_state.get('session_meta') or {}).get('driver')
    if own:
        live.add(own)
    return live or None


def _clamp_expected_stint_end(
    llm_stint_end:  int | None,
    pit_lap_target: int | None,
    compound_next:  str | None,
    cliff_p50:      float | None,
    total_laps:     int | None,
) -> int | None:
    """Ground the LLM's ``expected_stint_end`` against a physical anchor (#433).

    ``expected_stint_end`` is unvalidated LLM free text: nothing in the schema stops
    the LLM naming a lap far beyond what this stint can physically reach. Anchor it to
    ``pit_lap_target`` plus the shorter of the N26 cliff P50 and the Pirelli stint
    capacity for the NEXT compound (the same ``_STINT_CAPACITY_LAPS`` table
    ``recommend_compound_tool`` uses, not duplicated), bounded by ``total_laps``. Accept
    the LLM value only within +/-3 laps of the anchor; otherwise use the anchor. When
    ``pit_lap_target``, ``compound_next`` or ``cliff_p50`` is missing there is no anchor
    to ground against, so the LLM value passes through unclamped rather than inventing
    one. Pure (no I/O) so it is unit-testable without loading any model.
    """
    if pit_lap_target is None or compound_next is None or cliff_p50 is None:
        return llm_stint_end

    capacity = _STINT_CAPACITY_LAPS.get(compound_next, _STINT_CAPACITY_LAPS['MEDIUM'])
    anchor = pit_lap_target + min(cliff_p50, capacity)
    if total_laps is not None:
        anchor = min(anchor, total_laps)
    anchor = int(round(anchor))

    if llm_stint_end is not None and abs(llm_stint_end - anchor) <= 3:
        return llm_stint_end
    if llm_stint_end is not None:
        logger.warning(
            "Clamping LLM expected_stint_end %r to anchor %d "
            "(pit_lap_target=%s, compound_next=%s, cliff_p50=%.1f) — #433",
            llm_stint_end, anchor, pit_lap_target, compound_next, cliff_p50,
        )
    return anchor


def _assemble_recommendation(
    synth:              "_LLMSynthesis",
    pit_out,
    mc_results:         dict,
    regulation_context: str,
    sc_currently_active: bool = False,
    live_drivers:       set | None = None,
    cliff_p50:          Optional[float] = None,
    total_laps:         Optional[int] = None,
) -> "StrategyRecommendation":
    """Merge the LLM synthesis with N28 pit data and attach grounding fields.

    The LLM fills a _LLMSynthesis (action, reasoning, confidence, plus the
    v2 expansion fields). This helper builds the final StrategyRecommendation
    by combining that synthesis with:

    * pit_out from N28: used as a deterministic fallback whenever the LLM
      leaves pit_lap_target, compound_next, or undercut_target as None but
      N28 actually produced a value. This guarantees that richer execution
      detail never silently disappears because the LLM was lazy on a given
      lap. The LLM's explicit choice always wins; N28 only backfills nulls.
    * mc_results: attached as scenario_scores so downstream consumers can
      inspect the MC distribution without re-running the simulation. The LLM
      never writes this field directly (strict schema forbids dicts).
    * regulation_context: attached verbatim from N30 so the UI can surface
      the regulatory basis for the action without re-querying N30.

    --- WHERE TO CHANGE IF THE SC POLICY CHANGES ---
    **There is no action rail here, and there must not be one.** A rail may encode what
    the FIA regulation makes certain; it may never encode a strategy opinion. Whether to
    pit under a Safety Car depends on stops already made, laps remaining, gap behind and
    compounds used (race state, none of it a rule), so it belongs to the model.

    One forcing to a strategy opinion once lived here (STAY_OUT -> PIT_NOW on every SC
    lap, from the Qatar 2025 case). It was wrong: Art. 55.17 finishes the race behind a
    late SC with no overtaking, so the position a forced stop surrenders is unrecoverable
    BY REGULATION, and the pipeline shipped PIT_NOW carrying the guard-rail's own reason
    "too late to pit". It also silenced its evidence: with an SC deployed sc_prob is 1.0,
    so every MC draw already receives the full SC_PIT_BONUS, and a STAY_OUT argmax IS the
    model saying the cheap stop was outweighed.

    What `sc_currently_active` is for, then: the regulatory FACTS. Most live in N27, so
    every consumer inherits one number (overtake_prob = 0, Art. 55.8; drs_window = 0,
    Art. 22.1(c)). Only `target_lap_time_s` is forced here, because this is the layer
    that emits it (see the note at its assignment below). Add a new fact only if the
    regulation removes the field's SOURCE; if it merely makes a choice usually smart,
    it belongs in the prompt or the MC. See tests/mc/test_sc_regulatory_rails.py.

    Defaults to False so a caller that does not thread N27's output keeps the previous
    behaviour rather than silently changing it.

    `live_drivers` carries the cars on track so the LLM's free-text undercut_target can
    be checked. **None means "unknown", not "nobody"**: a caller without a lap_state
    cannot know, so its target passes rather than being silently discarded.

    `cliff_p50` (N26 TireOutput.laps_to_cliff_p50) and `total_laps` (RaceState.total_laps)
    ground `expected_stint_end`, see the clamp below (#433). Both default to None so a
    caller that has not been updated to pass them keeps the previous unclamped behaviour
    rather than crashing.

    Returns a fully-populated StrategyRecommendation ready for the UI layer.
    """
    # N28 fallbacks: only used when the LLM did not commit to a value
    fallback_lap     = pit_out.recommended_lap        if pit_out else None
    fallback_cmpd    = pit_out.compound_recommendation if pit_out else None
    fallback_target  = pit_out.undercut_target        if pit_out else None

    pit_lap_target  = synth.pit_lap_target  if synth.pit_lap_target  is not None else fallback_lap
    compound_next   = synth.compound_next   if synth.compound_next   is not None else fallback_cmpd
    # N28's target takes precedence: it is the one validated against the live drivers in
    # score_undercut_tool. The synthesis field is LLM free text (the prompt seeds it with
    # an example, "e.g. SAI"), so preferring it bypasses that check. The LLM value is used
    # only when N28 produced none, and only if it names a car currently racing; otherwise
    # an unvalidated code would surface on the pit wall as "UCUT: SAI".
    if fallback_target is not None:
        undercut_target = fallback_target
    elif synth.undercut_target is None:
        undercut_target = None
    elif live_drivers is None:
        # Who is racing cannot be told (no lap_state: the FastF1 path). Rejecting here
        # would silently discard every LLM target on that path, which is not the same
        # thing as knowing it is wrong. `None` means unknown; only a real roster may
        # reject. Collapsing it to an empty set is how a "do not know" turns into a "no".
        undercut_target = synth.undercut_target
    elif synth.undercut_target in live_drivers:
        undercut_target = synth.undercut_target
    else:
        logger.warning(
            "Discarding LLM undercut_target %r: not on track this lap (live: %s)",
            synth.undercut_target, sorted(live_drivers),
        )
        undercut_target = None

    # The action is the synthesis's, always. No SC rail forces STAY_OUT -> PIT_NOW
    # here, because that would be an opinion wearing a rail's clothes: one race
    # (Qatar 2025) generalised into a universal law. Under a real SC, staying out is
    # often right (having just pitted; leading while the pack must stop anyway; or
    # rejoining into traffic), and Art. 55.17 makes forcing the stop provably wrong in
    # the closing laps, where the race finishes behind the SC and the surrendered
    # track position is unrecoverable by regulation.
    #
    # It also silenced the very computation built to weigh it: with an SC deployed
    # sc_prob_3lap is 1.0, so every Monte Carlo draw already receives the full
    # SC_PIT_BONUS. A STAY_OUT argmax under those conditions IS the model saying the
    # cheap stop was outweighed.
    #
    # What a deployed SC forces are the REGULATORY facts, and they live in N27 where
    # every consumer inherits one consistent number: overtake_prob = 0 (Art. 55.8),
    # sc_prob_3lap = 1.0, drs_window = 0 (Art. 22.1(c)).
    #
    # One is forced here instead, because it is this layer that emits it:
    # `target_lap_time_s` is grounded in N06's PaceOutput CI, and N06 predicts
    # GREEN-FLAG pace. Art. 55.7 requires drivers to stay ABOVE the FIA ECU minimum
    # time while the SC is out, so a green-flag target is below the delta by
    # construction: the system would be instructing the driver to earn a penalty. The
    # real delta cannot be sourced (it is not in the telemetry), so the field has no
    # valid value. None is forced by ABSENCE OF A SOURCE, not by a strategy view, and
    # the schema already documents None as "the LLM prefers not to commit".
    # Inventing a delta would only launder the breach into looking authoritative.
    # See tests/mc/test_sc_regulatory_rails.py.
    action    = synth.action
    reasoning = synth.reasoning

    # #433: expected_stint_end is unvalidated LLM free text; clamp it against the
    # physical pit_lap + cliff/capacity anchor via the pure _clamp_expected_stint_end
    # helper (extracted so the clamp is CI-testable without loading any model).
    expected_stint_end = _clamp_expected_stint_end(
        synth.expected_stint_end, pit_lap_target, compound_next, cliff_p50, total_laps
    )

    return StrategyRecommendation(
        action             = action,
        reasoning          = reasoning,
        confidence         = synth.confidence,
        pit_lap_target     = pit_lap_target,
        compound_next      = compound_next,
        undercut_target    = undercut_target,
        pace_mode          = synth.pace_mode,
        target_lap_time_s  = None if sc_currently_active else synth.target_lap_time_s,
        risk_posture       = synth.risk_posture,
        contingencies      = synth.contingencies,
        key_risks          = synth.key_risks,
        expected_stint_end = expected_stint_end,
        scenario_scores    = mc_results,
        regulation_context = regulation_context,
    )


# ==============================================================================
# Entry points
# ==============================================================================

def run_strategy_orchestrator(
    race_state: "RaceState",
    lap_state:  dict,
) -> "StrategyRecommendation":
    """Run the Strategy Orchestrator for one lap and return a StrategyRecommendation.

    Primary entry point. Uses the FastF1-dependent entry points of each sub-agent
    (run_pace_agent, run_tire_agent, run_race_situation_agent, run_radio_agent,
    run_pit_strategy_agent). Each one is self-contained through the lap_state
    dict it receives here: no pre-populated globals or setup call is needed on
    any sub-agent module.

    race_state:
        Validated Pydantic RaceState for this lap. Contains driver, position,
        compound, tyre_life, weather fields, and pre-filtered radio/RCM events.
    lap_state:
        Dict of scalar lap features forwarded to sub-agent entry points. Must
        contain: driver_number, stint, team, year, gp_name, session and driver
        (both read by run_tire_agent and run_race_situation_agent), compound_id
        and tyre_life (run_tire_agent), lap_number and event_name
        (run_race_situation_agent). Optional keys: laps_since_pit, fuel_load,
        prev_lap_time, prev_speed_st, humidity, rivals (list of rival dicts for
        N27/N28).

    Returns a StrategyRecommendation with action, reasoning, confidence,
    scenario_scores, and regulation_context populated. scenario_scores and
    regulation_context are attached after the LLM call, not parsed from it.
    """
    # Layer 1a: always-on agents
    pace_out, tire_out, situation_out, radio_out = _run_always_on_agents(
        race_state, lap_state
    )

    # Layer 1b: routing
    active = _decide_agents_to_call(
        tire_warning        = tire_out.warning_level,
        sc_prob_3lap        = situation_out.sc_prob_3lap,
        radio_alerts        = radio_out.alerts,
        sc_currently_active = situation_out.sc_currently_active,
    )

    # Layer 1c: conditional agents. The structured RAG dict is only used by
    # the arcade dashboard; the orchestrator path keeps the answer string.
    pit_out, regulation_context, _rag_dict = _run_conditional_agents(
        active        = active,
        lap_state     = lap_state,
        tire_out      = tire_out,
        situation_out = situation_out,
        race_state    = race_state,
        laps_df       = None,
    )
    regulation_context = regulation_context or ""

    # Layer 2: MC simulation. Race context is threaded so this entry point
    # projects like the shared engine does; without it /recommend and the MCP
    # chat scored in the legacy currency while every other surface had moved on,
    # and the raw max() below raised on the first ineligible candidate.
    _ctx = race_context_from_lap_state(lap_state, race_state)
    mc_results = _run_mc_simulation(
        pace_out      = pace_out,
        tire_out      = tire_out,
        situation_out = situation_out,
        pit_out       = pit_out,
        alpha         = race_state.risk_tolerance,
        rivals        = (lap_state or {}).get("rivals"),
        position      = _ctx.get("position"),
        laps_remaining= _ctx.get("laps_remaining"),
        pit_context   = _ctx.get("pit_context"),
    )
    best_mc = best_mc_candidate(mc_results)

    # Layer 3: LLM synthesis
    prompt = _build_orchestrator_prompt(
        race_state         = race_state,
        mc_results         = mc_results,
        best_mc            = best_mc,
        pace_out           = pace_out,
        tire_out           = tire_out,
        situation_out      = situation_out,
        pit_out            = pit_out,
        radio_out          = radio_out,
        regulation_context = regulation_context,
    )

    synth: _LLMSynthesis = _get_orchestrator_llm().invoke(prompt)
    return _assemble_recommendation(
        synth, pit_out, mc_results, regulation_context,
        sc_currently_active = situation_out.sc_currently_active,
        live_drivers        = _live_drivers_from(lap_state),
        cliff_p50           = tire_out.laps_to_cliff_p50,
        total_laps          = race_state.total_laps,
    )


def _scope_laps_to_gp(
    laps_df: pd.DataFrame,
    lap_state: dict | None,
    race_state: "RaceState | None" = None,
) -> pd.DataFrame:
    """Narrow a season-wide laps frame to the Grand Prix being analysed (#429/#465).

    Thin delegator to the canonical implementation in
    ``src/strategy/inference/engine.py``. That module imports FROM this one, so a
    top-level ``import`` would be circular: the deferred import inside the body
    breaks the cycle while keeping ONE source of truth. A hand-kept duplicate is
    exactly the kind of drift the #429/#465 family of bugs came from, so no two
    copies are kept; the engine version also derives the GP from ``race_state`` when
    ``lap_state`` carries no ``gp_name`` yet.
    """
    from src.strategy.inference.engine import _scope_laps_to_gp as _engine_scope

    return _engine_scope(laps_df, lap_state, race_state)


def run_strategy_orchestrator_from_state(
    race_state: "RaceState",
    laps_df:    pd.DataFrame,
    lap_state:  dict | None = None,
) -> "StrategyRecommendation":
    """RSM adapter: run the orchestrator without a live FastF1 session.

    Calls the *_from_state entry points of every sub-agent so the orchestrator
    can run from a pre-loaded laps DataFrame (e.g. from RaceStateManager replay
    or offline backtesting) without any FastF1 session object.

    race_state:
        Validated Pydantic RaceState for this lap.
    laps_df:
        Full lap DataFrame from RaceStateManager. Forwarded to each sub-agent's
        RSM adapter (run_pace_agent_from_state, run_tire_agent_from_state, etc.),
        each of which derives its own laps_df / session_meta state from the call
        arguments: no shared globals involved.
    lap_state:
        Optional supplementary scalar dict. When None, a minimal lap_state is
        derived automatically from race_state and laps_df. Provide it when
        additional features (prev_lap_time, prev_speed_st, humidity, rivals)
        are available from the RaceStateManager.

    Returns a StrategyRecommendation identical to run_strategy_orchestrator().
    """
    if lap_state is None:
        driver_rows = laps_df[laps_df["Driver"] == race_state.driver]
        lap_row     = driver_rows[driver_rows["LapNumber"] == race_state.lap]
        year        = int(laps_df["Year"].iloc[0]) if "Year" in laps_df.columns else 2025
        # Derive the GP from the (driver, lap) row match, NOT laps_df.iloc[0] (the
        # first row of the whole-season frame): the latter blends one race's GP with
        # another race's stint/team: the #465 wrong-GP bug engine._build_default_lap_state
        # also has to avoid. Fall back to iloc[0] only when the row is absent.
        # Flattened from a nested ternary. The middle branch reads as redundant and
        # is not: lap_row can be non-empty and still lack GP_Name, which happens only
        # when laps_df lacks it too, since lap_row is a slice of it. Written as three
        # branches the reader sees that without having to re-derive it.
        if not lap_row.empty and "GP_Name" in lap_row:
            gp_name = str(lap_row["GP_Name"].iloc[0])
        elif "GP_Name" in laps_df.columns:
            gp_name = str(laps_df["GP_Name"].iloc[0])
        else:
            gp_name = ""
        stint = int(lap_row["Stint"].iloc[0]) if not lap_row.empty else 1
        team  = (
            str(lap_row["Team"].iloc[0]) if not lap_row.empty and "Team" in lap_row else "Unknown"
        )
        lap_state = {
            "lap_number": race_state.lap,
            "driver": {
                "driver":        race_state.driver,
                # None, not 0. This is the synthetic lap state built when no real one
                # exists, so the car number is genuinely unknown, and 0 is a value the
                # model can also legitimately find: it sorts below every real number
                # (1-81) and sends each DriverNumber split down its left branch. A
                # PRESENT key set to 0 also defeats the consumer's `.get`, which is why
                # fixing the reader alone was not enough (W-F2).
                "driver_number": None,
                "team":          team,
                "position":      race_state.position,
                "compound":      race_state.compound,
                "tyre_life":     race_state.tyre_life,
                "stint":         stint,
                # A RaceState carries no lap history, so the stint's opening TyreLife is
                # genuinely unknowable here. None makes N06 emit NaN FuelEffect plus a
                # warning, which is in-distribution (2% of the training parquet is null)
                # and cannot be mistaken for a reading. Keep in lockstep with the same
                # key in engine._build_default_lap_state (#446).
                "stint_baseline_tyre_life": None,
                "lap_time_s":    None,
                "speed_st":      300.0,
                "fuel_load":     1 - race_state.lap / max(race_state.total_laps, 1),
            },
            "session_meta": {
                "gp_name":    gp_name,
                "gp":         gp_name,
                "year":       year,
                "driver":     race_state.driver,
                "team":       team,
                "total_laps": race_state.total_laps,
            },
            "weather": {
                "air_temp":   race_state.air_temp,
                "track_temp": race_state.track_temp,
                "rainfall":   race_state.rainfall,
                "humidity":   50.0,
            },
            "rivals": [],
        }

    # Scope AFTER lap_state is resolved (built above or supplied by the caller),
    # never before: see _scope_laps_to_gp's docstring for the #465 ordering bug
    # this avoids. Passing race_state lets the canonical engine helper derive the GP
    # even when a caller supplies a lap_state without a gp_name. Every downstream use
    # of laps_df in this function (both agent calls below) sees the scoped frame.
    laps_df = _scope_laps_to_gp(laps_df, lap_state, race_state)

    # Layer 1a: always-on agents (RSM variants)
    pace_out, tire_out, situation_out, radio_out = _run_always_on_agents_from_state(
        race_state, laps_df, lap_state
    )

    # Layer 1b: routing
    active = _decide_agents_to_call(
        tire_warning        = tire_out.warning_level,
        sc_prob_3lap        = situation_out.sc_prob_3lap,
        radio_alerts        = radio_out.alerts,
        sc_currently_active = situation_out.sc_currently_active,
    )

    # Layer 1c: conditional agents (RSM variants). Same RAG-dict treatment
    # as the FastF1 entry point: discarded here, consumed only by the arcade.
    pit_out, regulation_context, _rag_dict = _run_conditional_agents(
        active        = active,
        lap_state     = lap_state,
        tire_out      = tire_out,
        situation_out = situation_out,
        race_state    = race_state,
        laps_df       = laps_df,
    )
    regulation_context = regulation_context or ""

    # Layer 2: MC simulation (same as primary entry point)
    _ctx = race_context_from_lap_state(lap_state, race_state)
    mc_results = _run_mc_simulation(
        pace_out      = pace_out,
        tire_out      = tire_out,
        situation_out = situation_out,
        pit_out       = pit_out,
        alpha         = race_state.risk_tolerance,
        rivals        = (lap_state or {}).get("rivals"),
        position      = _ctx.get("position"),
        laps_remaining= _ctx.get("laps_remaining"),
        pit_context   = _ctx.get("pit_context"),
    )
    best_mc = best_mc_candidate(mc_results)

    # Layer 3: LLM synthesis (same as primary entry point)
    prompt = _build_orchestrator_prompt(
        race_state         = race_state,
        mc_results         = mc_results,
        best_mc            = best_mc,
        pace_out           = pace_out,
        tire_out           = tire_out,
        situation_out      = situation_out,
        pit_out            = pit_out,
        radio_out          = radio_out,
        regulation_context = regulation_context,
    )

    synth: _LLMSynthesis = _get_orchestrator_llm().invoke(prompt)
    return _assemble_recommendation(
        synth, pit_out, mc_results, regulation_context,
        sc_currently_active = situation_out.sc_currently_active,
        live_drivers        = _live_drivers_from(lap_state),
        cliff_p50           = tire_out.laps_to_cliff_p50,
        total_laps          = race_state.total_laps,
    )
