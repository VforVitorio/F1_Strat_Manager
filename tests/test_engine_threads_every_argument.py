"""The engine must pass `_assemble_recommendation` everything the orchestrator does.

The engine's whole purpose is to remove duplication: it IMPORTS the orchestrator's layer
functions rather than copying their bodies. But it re-drives the call *sequence*, which
means every argument is threaded by hand — and importing the functions removes **body**
drift, not **call** drift.

That distinction cost a real bug. `_assemble_recommendation` grew a `live_drivers`
argument (#462, so the LLM cannot name a retired car as an undercut target). The
orchestrator threads it at both call sites; the engine did not. `live_drivers=None` is
documented to mean "unknown" and therefore passes the LLM's value unchecked, so the guard
was dead on the `rich` profile — which is the DEFAULT for /simulate, the arcade and the
CLI. The fix shipped and reached nothing.

Nobody caught it because the module docstring promised an anti-drift guard,
`tests/test_engine_parity.py`, that **has never existed**. This file is the smallest
thing that would have.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
_HAS_MODELS = (ROOT / "data" / "models" / "tire_degradation" / "routing_config.json").exists()
pytestmark = pytest.mark.skipif(
    not _HAS_MODELS,
    reason="data/models/ not present (CI runner without model weights)",
)


def _kwargs_passed_by(func) -> set[str]:
    """The keyword names `func` passes to `_assemble_recommendation`."""
    import ast

    tree = ast.parse(inspect.getsource(func).lstrip())
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "_assemble_recommendation"
        ):
            return {kw.arg for kw in node.keywords if kw.arg}
    return set()


def test_the_engine_threads_every_argument_the_orchestrator_does():
    """A new argument on the assembly must reach the engine, or its guard is dead.

    Compares the keywords the engine passes against the ones the orchestrator passes.
    Anything the orchestrator threads and the engine does not is, by construction, a
    feature that works everywhere except the default path.
    """
    import src.agents.strategy_orchestrator as orch
    from src.strategy.inference import engine

    engine_kwargs = _kwargs_passed_by(engine._run_rich)
    orch_kwargs = _kwargs_passed_by(orch.run_strategy_orchestrator_from_state)

    missing = orch_kwargs - engine_kwargs
    assert not missing, (
        f"the engine's rich profile does not thread {sorted(missing)} into "
        f"_assemble_recommendation, so whatever those arguments guard is dead on every "
        f"surface that uses the default profile"
    )


def test_live_drivers_reaches_the_assembly_from_the_engine():
    """The specific argument this test file was written for (#462)."""
    from src.strategy.inference import engine

    assert "live_drivers" in _kwargs_passed_by(engine._run_rich)


def test_the_docstring_does_not_promise_a_test_that_does_not_exist():
    """The promise of a guard is worse than no guard: it stops people looking.

    The engine's docstring cited `tests/test_engine_parity.py` as its anti-drift guard
    for months. The file never existed, and its absence is why `live_drivers` went
    unthreaded while its author believed a test covered it.

    Only the line that ASSERTS a guard is checked. Prose explaining a file's absence may
    name it — that is history, not a claim.
    """
    import re

    from src.strategy.inference import engine

    for line in (engine.__doc__ or "").splitlines():
        if not line.strip().startswith("Anti-drift guard:"):
            continue
        for name in re.findall(r"tests/test_[a-z_]+\.py", line):
            assert (ROOT / name).exists(), (
                f"the engine docstring names {name} as its anti-drift guard, and it does not exist"
            )
