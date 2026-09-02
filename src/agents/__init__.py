"""src/agents: Multi-Agent Strategy System (v0.9)

Public re-exports for the six sub-agents and the orchestrator, resolved LAZILY.

Importing this package (or any submodule under it) no longer eagerly loads every
agent. The heavy agents, the radio NLP stack (Whisper/RoBERTa/SetFit/NER) and
the RAG embedder, used to load at import time simply because they were listed
here, so touching ANY single agent (e.g. the Model Lab asking only for the Pace
XGBoost model) pulled the whole family into memory and VRAM. Now each name below
resolves on first access via module ``__getattr__`` (PEP 562), so a single-agent
call loads only that agent.

The public surface is unchanged: ``from src.agents import run_pace_agent`` still
works (it loads pace_agent on first use), and direct submodule imports
(``from src.agents.pace_agent import ...``) are unaffected and remain the fastest
path. Import from the individual modules for full type information.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

# Public name -> the submodule that defines it. Loaded on first attribute access.
_EXPORTS: dict[str, str] = {
    # Pace (N25)
    "run_pace_agent": "pace_agent",
    "run_pace_agent_from_state": "pace_agent",
    # Tire (N26)
    "run_tire_agent": "tire_agent",
    "run_tire_agent_from_state": "tire_agent",
    # Race situation (N27)
    "run_race_situation_agent": "race_situation_agent",
    "run_race_situation_agent_from_state": "race_situation_agent",
    # Pit strategy (N28)
    "run_pit_strategy_agent": "pit_strategy_agent",
    "run_pit_strategy_agent_from_state": "pit_strategy_agent",
    # Radio (N29)
    "run_radio_agent": "radio_agent",
    "run_radio_agent_from_state": "radio_agent",
    "RadioMessage": "radio_agent",
    "RCMEvent": "radio_agent",
    # RAG (N30)
    "run_rag_agent": "rag_agent",
    "run_rag_agent_from_state": "rag_agent",
    # Orchestrator (N31)
    "RaceState": "strategy_orchestrator",
    "StrategyRecommendation": "strategy_orchestrator",
    "run_strategy_orchestrator": "strategy_orchestrator",
    "run_strategy_orchestrator_from_state": "strategy_orchestrator",
}

__all__ = list(_EXPORTS)


def __getattr__(name: str):
    """Lazily import the submodule that defines ``name`` on first access (PEP 562).

    Caches the resolved value into module globals so subsequent access skips
    this hook. Raises AttributeError for unknown names, matching normal module
    attribute semantics.
    """
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    resolved = getattr(importlib.import_module(f"{__name__}.{module}"), name)
    globals()[name] = resolved
    return resolved


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))


# Static-analysis + IDE resolution: these are the same names the lazy hook
# serves at runtime. Guarded so they never execute (and never eager-load) at
# import time.
if TYPE_CHECKING:  # pragma: no cover
    from src.agents.pace_agent import run_pace_agent, run_pace_agent_from_state
    from src.agents.tire_agent import run_tire_agent, run_tire_agent_from_state
    from src.agents.race_situation_agent import (
        run_race_situation_agent,
        run_race_situation_agent_from_state,
    )
    from src.agents.pit_strategy_agent import (
        run_pit_strategy_agent,
        run_pit_strategy_agent_from_state,
    )
    from src.agents.radio_agent import (
        run_radio_agent,
        run_radio_agent_from_state,
        RadioMessage,
        RCMEvent,
    )
    from src.agents.rag_agent import run_rag_agent, run_rag_agent_from_state
    from src.agents.strategy_orchestrator import (
        RaceState,
        StrategyRecommendation,
        run_strategy_orchestrator,
        run_strategy_orchestrator_from_state,
    )
