"""Arcade-local strategy pipeline: thin delegate over the shared engine.

The arcade process runs the full N31 multi-agent pipeline in-process (no backend
SSE hop) so its dashboard subprocess can subscribe to the arcade TCP stream and
receive both the synthesised ``StrategyRecommendation`` and the raw per-sub-agent
outputs on the same wire.

This module was a body-copy of
``src.agents.strategy_orchestrator.run_strategy_orchestrator_from_state`` with a
"mirror the change here" warning: the exact drift the audit (AUDIT_P2B_CORE_COMPUTE
F10) flagged and the #166 crash proved real. It now delegates to the single shared
engine ``src.strategy.inference.engine.run_lap`` (``rich`` profile), which reproduces
the orchestrator byte-for-byte AND returns the agent outputs. One code path, three
surfaces; nothing to keep in sync by hand.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import pandas as pd

from src.strategy.inference.engine import run_lap

logger = logging.getLogger(__name__)

if TYPE_CHECKING:  # pragma: no cover, only for type hints
    from src.agents.strategy_orchestrator import RaceState, StrategyRecommendation
    from src.strategy.inference.decision_memory import DecisionMemory


def run_strategy_pipeline(
    race_state: "RaceState",
    laps_df: pd.DataFrame,
    lap_state: dict | None = None,
    memory: "DecisionMemory | None" = None,
    no_llm: bool = False,
) -> tuple["StrategyRecommendation", dict]:
    """Run the full N31 pipeline; return the recommendation and per-agent outputs.

    Public signature stays backward compatible: ``memory`` and ``no_llm`` are
    both optional and default to the pre-existing behaviour (LLM synthesis),
    so ``src/arcade/strategy.py`` and the dashboard formatters that already
    call this positionally keep working unchanged. The ``agent_outputs`` dict
    carries the same keys as before (``pace_out``/``tire_out``/
    ``situation_out``/``radio_out``/``pit_out``/``regulation_context``/
    ``rag``/``active``) plus ``guardrail_reason``.

    ``no_llm`` (#1155) selects ``run_lap``'s deterministic profile instead of
    the LLM-synthesised one: ``True`` routes through ``"no-llm"`` (zero LLM
    clients, ``src/strategy/inference/no_llm.py``), ``False`` keeps the
    existing ``"rich"`` profile. The caller carried this flag through two
    dataclasses and a constructor call without it ever reaching here, so
    setting it on the arcade's request changed nothing and the arcade always
    ran the paid path.

    **The stage timings are LOGGED, not returned and not broadcast (#1045).**
    This docstring used to say they were dropped here and that "a future arcade
    change may forward it on the TCP stream", which stayed a promise rather than
    a plan while the measurement was paid for on every lap and thrown away on the
    same line. All three callers of ``run_lap`` discarded it, so nothing has ever
    consumed it.

    They go to the log rather than to the wire because they are DIAGNOSTIC: a pit
    wall does not show model latency, this window has no diagnostics tier, and
    putting six numbers on a surface built for a decision is how a panel becomes a
    drawer. In a log they cost nothing per tick and they are there when a lap is
    slow, which is the only moment anyone wants them.

    DEBUG and not INFO: this runs once per lap of every arcade race.
    """
    profile = "no-llm" if no_llm else "rich"
    rec, agent_outputs, timings = run_lap(
        race_state, laps_df, lap_state, profile=profile, memory=memory
    )
    if logger.isEnabledFor(logging.DEBUG):
        breakdown = " ".join(f"{stage}={seconds:.3f}s" for stage, seconds in timings.items())
        logger.debug("lap %s pipeline stages: %s", getattr(race_state, "lap", "?"), breakdown)
    return rec, agent_outputs
