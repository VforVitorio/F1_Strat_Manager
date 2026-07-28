"""The decision-memory block reaches three surfaces and deliberately not the other two.

`run_lap`'s callers own a race — the CLI loop, the arcade connector, the backend
simulator's stream — so each can hold one `DecisionMemory` for its lifetime and pass
the rendered block down. `/recommend` and the MCP tool cannot: both are stateless per
request, with no race-scoped object to accumulate on, so the webapp Strategy tab keeps
the memoryless prompt. That is a declared limitation, not an oversight.

WHY THIS NEEDS ITS OWN FILE, when an anti-drift guard already exists next door.
`test_engine_threads_every_argument` asserts `orch_kwargs - engine_kwargs`: the
ORCHESTRATOR has an argument the ENGINE lacks. That is the direction both real
failures took (#462, and the prompt builder in #675), and it is the direction that
file was widened to cover.

Memory goes the other way. The engine passes it; the orchestrator's stateless entry
point deliberately does not. The existing guard is green on that by construction — its
own docstring says so — so without this file the asymmetry is protected by nothing, and
a later "make these two call sites consistent" cleanup would pass the whole suite while
silently giving `/recommend` a memory it has no way to populate.

The failure this prevents is not hypothetical in shape: `_rcm_events_for_lap` already
exists twice with two signatures because a second surface grew its own copy of
something the engine already had.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.engine.ast_helpers import kwargs_passed_by

ROOT = Path(__file__).parent.parent.parent
_HAS_MODELS = (ROOT / "data" / "models" / "tire_degradation" / "routing_config.json").exists()
pytestmark = pytest.mark.skipif(
    not _HAS_MODELS,
    reason="data/models/ not present (CI runner without model weights)",
)

_PROMPT_BUILDER = "_build_orchestrator_prompt"
_MEMORY_ARG = "memory_block"


def test_the_stateless_entry_point_does_not_pass_a_memory_block():
    """`/recommend` and the MCP tool must keep the memoryless prompt.

    Both reach the orchestrator through `run_strategy_orchestrator_from_state`, one
    request at a time, with nothing that survives between laps. A memory block there
    could only ever be empty or, worse, filled from something request-scoped that
    looks like a race and is not.

    If this test fails because someone threaded memory into the orchestrator, the fix
    is NOT to update the assertion. It is to give that surface a real race-scoped
    accumulator or leave it alone.
    """
    import src.agents.strategy_orchestrator as orch

    passed = kwargs_passed_by(orch.run_strategy_orchestrator_from_state, _PROMPT_BUILDER)

    assert passed, (
        f"no call to {_PROMPT_BUILDER} found in run_strategy_orchestrator_from_state; "
        f"this guard is inspecting a call site that no longer exists"
    )
    assert _MEMORY_ARG not in passed, (
        f"run_strategy_orchestrator_from_state now passes {_MEMORY_ARG!r}, but it is the "
        f"entry point for /recommend and the MCP tool, which are stateless per request "
        f"and have no race-scoped object to accumulate one on. Whatever it is passing "
        f"is not this race's history. See src/strategy/inference/decision_memory.py."
    )


def test_the_prompt_builder_still_accepts_a_memory_block():
    """The parameter has to exist for the asymmetry to mean anything.

    Without this, deleting `memory_block` from the builder entirely would leave the
    test above passing green and reporting a scope decision about an argument that no
    longer exists.
    """
    import inspect

    import src.agents.strategy_orchestrator as orch

    signature = inspect.signature(orch._build_orchestrator_prompt)

    assert _MEMORY_ARG in signature.parameters
    assert signature.parameters[_MEMORY_ARG].default == "", (
        "the memory block must default to empty: the stateless surfaces call the "
        "builder without it and have to keep the prompt they had"
    )
