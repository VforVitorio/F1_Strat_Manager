"""Deterministic, zero-LLM-client inference profile for the shared engine.

This is the ``no-llm`` half of ``engine.run_lap`` (P2b #169 Phase 1.2) and the real
fix for #166. The broken CLI ``--no-llm`` path crashed because it unpacked 2 values
from ``_run_conditional_agents`` (a 3-tuple since commit bfe5b46); here the crash is
unreachable by construction — the no-llm path never calls that function (N28/N30 are
LLM-backed and simply not run).

What "no-llm" means here (design doc §4, gate decision Q6 = ship all four deltas):
  * ZERO LLM clients are ever constructed (a test bombs ``ChatOpenAI.__init__`` to
    prove it) — no retry storm, and ``--no-llm`` can never silently become LLM mode.
  * N25/N26/N27/N29 produce REAL model numbers, not hardcoded stubs: pace via its
    public XGBoost entry, tire/situation by injecting a deterministic tool-runner
    (``_NullReActRunner``) into ENGINE-PRIVATE agent instances at the ``_react_agent``
    cache seam, so ``run_from_state`` runs end to end (priming, tool execution, output
    parsing, and the RCM Safety-Car override all reused, not reinvented).
  * Routing sees ``sc_currently_active`` (the Qatar V7 lesson), so under a deployed SC
    N28/N30 are correctly *routed* (reported in ``active``) even though, being
    LLM-backed, they are not executed (``pit_out`` stays ``None``).
  * The guard-rail decision policy is applied deterministically to the MC argmax.

Untouchability: nothing in ``src/agents/`` is edited — the injection sets a private
attribute on the engine's OWN agent instances (never the module singletons, so an
LLM-mode call in the same process is never contaminated).
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.agents.pace_agent import run_pace_agent_from_state
from src.agents.race_situation_agent import RaceSituationAgent
from src.agents.radio_agent import RadioOutput, _build_alerts, run_pipeline, run_rcm_pipeline
from src.agents.strategy_orchestrator import (
    RaceState,
    StrategyRecommendation,
    _assemble_recommendation,
    _decide_agents_to_call,
    _LLMSynthesis,
    _run_mc_simulation,
    _to_radio_message,
    _to_rcm_event,
)
from src.agents.tire_agent import TireAgent, _compound_name_to_id
from src.strategy.inference.engine import (
    _assemble_agent_outputs,
    _build_default_lap_state,
    _StageTimer,
)

# ---------------------------------------------------------------------------
# Guard-rails — canonical re-host of the backend's apply_guard_rails.
# Identical rule set to src/telemetry/backend/services/simulation/guard_rails.py
# and the CLI's inline copy (run_simulation_cli.py L1504-L1535). Re-hosted rather
# than imported because the parent must never depend on the submodule; the backend
# copy retires when P1 migrates, the CLI copy when the P4 duplicate lands (3 -> 1).
# ---------------------------------------------------------------------------
_PIT_ACTIONS = frozenset({"PIT_NOW", "UNDERCUT", "OVERCUT", "REACTIVE_SC"})
_NO_PIT_BEFORE_LAP = 5
_NO_PIT_LAST_N_LAPS = 3
_CLIFF_P10_SAFE = 2
_MIN_STINT_LAPS = {"SOFT": 8, "MEDIUM": 12, "HARD": 15}
_DEFAULT_MIN_STINT = 10


def apply_guard_rails(
    action: str,
    lap: int,
    total_laps: int,
    compound: str,
    tyre_life: int,
    cliff_p10: float = 99.0,
) -> tuple[str, str | None]:
    """Override *action* with STAY_OUT when a hard strategic constraint fires.

    Rules: no pit before lap 5; no pit in the last 3 laps unless the cliff is
    imminent (cliff_p10 < 2); minimum stint SOFT 8 / MEDIUM 12 / HARD 15. Returns
    ``(action, reason)`` with ``reason=None`` when no rail fired.
    """
    if action not in _PIT_ACTIONS:
        return action, None

    remaining_laps = total_laps - lap

    if lap < _NO_PIT_BEFORE_LAP:
        return "STAY_OUT", f"guard-rail: pit window not open (lap < {_NO_PIT_BEFORE_LAP})"

    if remaining_laps <= _NO_PIT_LAST_N_LAPS and cliff_p10 >= _CLIFF_P10_SAFE:
        return "STAY_OUT", f"guard-rail: too late to pit (<={_NO_PIT_LAST_N_LAPS} laps left)"

    min_life = _MIN_STINT_LAPS.get(compound, _DEFAULT_MIN_STINT)
    if tyre_life < min_life:
        return (
            "STAY_OUT",
            f"guard-rail: minimum stint not reached ({compound} {tyre_life}/{min_life} laps)",
        )

    return action, None


# ---------------------------------------------------------------------------
# The deterministic tool-runner injected at the agents' _react_agent seam.
# ---------------------------------------------------------------------------
class _NullReActRunner:
    """Stand-in for the LangGraph ReAct graph — runs pre-bound tools, no LLM.

    ``get_react_agent`` returns ``self._react_agent`` early when it is set, and
    ``_run_core`` calls ``.invoke({'messages': [...]}) -> {'messages': [...]}`` then
    regex-parses the tool outputs from the message contents. So this runner executes
    the agent's real tool closures (the same TCN / LightGBM inference the LLM path
    would trigger), wraps each returned string in a ``ToolMessage`` the parser reads,
    and appends a final ``AIMessage`` that becomes the ``reasoning`` field.
    """

    def __init__(self, tool_calls: list[tuple[Any, dict[str, Any]]]) -> None:
        # tool_calls: ordered list of (LangChain tool, kwargs) to execute this lap.
        self._tool_calls = tool_calls

    def invoke(self, payload: dict[str, Any]) -> dict[str, Any]:
        from langchain_core.messages import AIMessage, ToolMessage

        messages: list[Any] = []
        for i, (tool, kwargs) in enumerate(self._tool_calls):
            output = tool.invoke(kwargs)
            messages.append(ToolMessage(content=str(output), tool_call_id=f"nullrun-{i}"))
        messages.append(AIMessage(content="[no-llm — deterministic tool pass]"))
        return {"messages": messages}


# Engine-private agent instances (lazy). Kept separate from the module singletons
# (_get_default_*), so injecting the null runner here never leaks into an LLM-mode
# call served by the same process (the backend serves both modes).
_tire_agent: TireAgent | None = None
_situation_agent: RaceSituationAgent | None = None


def _get_tire_agent() -> TireAgent:
    global _tire_agent
    if _tire_agent is None:
        _tire_agent = TireAgent()
    return _tire_agent


def _get_situation_agent() -> RaceSituationAgent:
    global _situation_agent
    if _situation_agent is None:
        _situation_agent = RaceSituationAgent()
    return _situation_agent


def _tire_no_llm(lap_state: dict[str, Any], laps_df: pd.DataFrame):
    """Run N26 with real TCN numbers and no LLM, via the injected null runner.

    Pre-binds both tire tools (predict_tire_deg + estimate_laps_to_cliff) with the
    same (driver, compound_id, tyre_life) args ``run_from_state`` derives, then calls
    the instance's public ``run_from_state`` so all its priming/parsing is reused.
    """
    agent = _get_tire_agent()
    driver = lap_state["session_meta"]["driver"]
    d = lap_state["driver"]
    meta = lap_state["session_meta"]
    compound = d.get("compound", "MEDIUM")
    tyre_life = d.get("tyre_life", 1)
    gp_name = meta.get("gp_name", "")
    year = meta.get("year", 2025)
    compound_id = (
        compound if compound.startswith("C") else _compound_name_to_id(compound, gp_name, year)
    )
    args = {"driver": driver, "compound_id": compound_id, "tyre_life": tyre_life}
    agent._react_agent = _NullReActRunner([(tool, args) for tool in agent._tools])
    return agent.run_from_state(lap_state, laps_df)


def _situation_no_llm(sit_lap_state: dict[str, Any], laps_df: pd.DataFrame):
    """Run N27 with real LightGBM numbers + the RCM SC override, no LLM.

    Executes predict_sc_tool always, predict_overtake_tool only when a rival ahead is
    derivable (parser defaults overtake fields to 0.0 otherwise). ``run_from_state``
    then applies the SafetyCar RCM override exactly as in LLM mode.
    """
    agent = _get_situation_agent()
    meta = sit_lap_state["session_meta"]
    d = sit_lap_state["driver"]
    driver = meta["driver"]
    lap_number = sit_lap_state["lap_number"]
    rivals = sit_lap_state.get("rivals", []) or []
    driver_pos = d.get("position", 20)
    rival = next((r["driver"] for r in rivals if r.get("position") == driver_pos - 1), None)

    tools = {tool.name: tool for tool in agent._tools}
    calls: list[tuple[Any, dict[str, Any]]] = [
        (tools["predict_sc_tool"], {"lap_number": lap_number})
    ]
    if rival:
        calls.append(
            (
                tools["predict_overtake_tool"],
                {"driver_x": driver, "driver_y": rival, "lap_number": lap_number},
            )
        )
    agent._react_agent = _NullReActRunner(calls)
    return agent.run_from_state(sit_lap_state, laps_df)


def _run_radio_no_llm(radio_lap_state: dict[str, Any]) -> RadioOutput:
    """Run N29 stages 1+2 (NLP inference + deterministic alerts), skipping the LLM.

    ``radio_msgs`` / ``rcm_events`` are already coerced (RadioMessage / RCMEvent) by
    the caller, mirroring run_radio_agent's own stages 1-2; stage 3 (synthesis) is the
    only LLM step and is dropped, matching the wording run_radio_agent uses on failure.
    """
    radio_msgs = radio_lap_state.get("radio_msgs", []) or []
    rcm_events = radio_lap_state.get("rcm_events", []) or []
    radio_results = [run_pipeline(msg.text) for msg in radio_msgs]
    rcm_results = [run_rcm_pipeline(ev) for ev in rcm_events]
    alerts = _build_alerts(radio_results, rcm_results, radio_msgs)
    return RadioOutput(
        radio_events=radio_results,
        rcm_events=rcm_results,
        alerts=alerts,
        reasoning="[no-LLM mode — radio synthesis skipped, NLP stages 1+2 still applied]",
        corrections=[],
    )


def _deterministic_synthesis(action: str, guardrail_reason: str | None) -> "_LLMSynthesis":
    """Build the synthesis stand-in fed to the real ``_assemble_recommendation``.

    Using the schema object (not a hand-built StrategyRecommendation) means any future
    field flows through automatically and the no-llm rec can never structurally diverge
    from the rich rec.
    """
    reasoning = "[no-llm mode — MC argmax + guard-rails, no LLM synthesis]"
    if guardrail_reason:
        reasoning += " " + guardrail_reason
    synth = _LLMSynthesis(
        action=action,
        reasoning=reasoning,
        confidence=0.0,
        pace_mode="NEUTRAL",
        risk_posture="BALANCED",
        contingencies=[],
        key_risks=[],
    )
    return synth


def run_no_llm_lap(
    race_state: RaceState,
    laps_df: pd.DataFrame,
    lap_state: dict[str, Any] | None,
    return_agent_outputs: bool,
) -> tuple[StrategyRecommendation, dict[str, Any] | None, dict[str, float]]:
    """Deterministic no-LLM lap — the ``no-llm`` profile of ``engine.run_lap``.

    Mirrors the orchestrator's always-on feeding (radio/rcm coercion, sit_lap_state
    carrying rcm_events for the SC override) but runs every agent with zero LLM
    clients, then routes, runs the MC (pit_out=None -> conservative prior), guard-rails
    the argmax, and assembles the recommendation through the real helper.
    """
    timings: dict[str, float] = {}
    if lap_state is None:
        lap_state = _build_default_lap_state(race_state, laps_df)

    radio_msgs = [_to_radio_message(m) for m in race_state.radio_msgs]
    rcm_events = [_to_rcm_event(e) for e in race_state.rcm_events]
    radio_lap_state = {
        **lap_state,
        "lap": race_state.lap,
        "radio_msgs": radio_msgs,
        "rcm_events": rcm_events,
    }
    sit_lap_state = {**lap_state, "rcm_events": rcm_events}

    with _StageTimer(timings, "always_on"):
        pace_out = run_pace_agent_from_state(lap_state)
        tire_out = _tire_no_llm(lap_state, laps_df)
        situation_out = _situation_no_llm(sit_lap_state, laps_df)
        radio_out = _run_radio_no_llm(radio_lap_state)

    with _StageTimer(timings, "routing"):
        active = _decide_agents_to_call(
            tire_warning=tire_out.warning_level,
            sc_prob_3lap=situation_out.sc_prob_3lap,
            radio_alerts=radio_out.alerts,
            sc_currently_active=situation_out.sc_currently_active,
        )

    with _StageTimer(timings, "conditional"):
        # N28 (pit) and N30 (RAG) are LLM-backed, so they are never run in no-llm.
        # `active` above still reports what WOULD have routed, so the panels stay honest.
        pit_out, regulation_context, rag_dict = None, "", None

    with _StageTimer(timings, "mc"):
        mc_results = _run_mc_simulation(
            pace_out=pace_out,
            tire_out=tire_out,
            situation_out=situation_out,
            pit_out=pit_out,  # None -> conservative prior (Triangular 2.2/2.8/3.8, ucut 0.5)
            alpha=race_state.risk_tolerance,
        )
        best_mc = max(mc_results, key=lambda s: mc_results[s]["score"])

    with _StageTimer(timings, "synthesis"):
        action, guardrail_reason = apply_guard_rails(
            best_mc,
            race_state.lap,
            race_state.total_laps,
            race_state.compound,
            race_state.tyre_life,
            tire_out.laps_to_cliff_p10,
        )
        synth = _deterministic_synthesis(action, guardrail_reason)
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
            guardrail_reason=guardrail_reason,
        )
    return rec, agent_outputs, timings
