"""Arcade-local strategy pipeline: thin delegate over the shared engine.

The arcade process runs the full N31 multi-agent pipeline in-process (no backend
SSE hop) so its dashboard subprocess can subscribe to the arcade TCP stream and
receive both the synthesised ``StrategyRecommendation`` and the raw per-sub-agent
outputs on the same wire.

This module used to be a body-copy of
``src.agents.strategy_orchestrator.run_strategy_orchestrator_from_state`` with a
"mirror the change here" warning: the exact drift the audit (AUDIT_P2B_CORE_COMPUTE
F10) flagged and the #166 crash proved real. It now delegates to the single shared
engine ``src.strategy.inference.engine.run_lap`` (``rich`` profile), which reproduces
the orchestrator byte-for-byte AND returns the agent outputs. One code path, three
surfaces; nothing to keep in sync by hand.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

from src.strategy.inference.engine import run_lap

if TYPE_CHECKING:  # pragma: no cover — only for type hints
    from src.agents.strategy_orchestrator import RaceState, StrategyRecommendation


def run_strategy_pipeline(
    race_state: "RaceState",
    laps_df: pd.DataFrame,
    lap_state: dict | None = None,
) -> tuple["StrategyRecommendation", dict]:
    """Run the full N31 pipeline; return the recommendation and per-agent outputs.

    Public signature is unchanged so ``src/arcade/strategy.py`` and the dashboard
    formatters keep working. The ``agent_outputs`` dict carries the same keys as
    before (``pace_out``/``tire_out``/``situation_out``/``radio_out``/``pit_out``/
    ``regulation_context``/``rag``/``active``) plus ``guardrail_reason``; the
    engine's third return value (stage timings) is dropped here (a future arcade
    change may forward it on the TCP stream).
    """
    rec, agent_outputs, _timings = run_lap(race_state, laps_df, lap_state, profile="rich")
    return rec, agent_outputs
