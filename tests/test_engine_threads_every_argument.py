"""The engine must pass `_assemble_recommendation` everything the orchestrator does.

The engine removes duplication by importing the orchestrator's layer functions rather
than copying their bodies, but it re-drives the call sequence, so every argument is
threaded by hand. Importing the functions prevents body drift, not call drift.

That gap is a real failure mode: when `_assemble_recommendation` gained a `live_drivers`
argument (#462, to keep the LLM from naming a retired car as an undercut target), the
orchestrator threaded it at both call sites and the engine did not. `live_drivers=None`
means "unknown" and passes the LLM value unchecked, so the guard was inactive on the
`rich` profile, which is the default for /simulate, the arcade and the CLI.

The guarantee this file provides is that the engine passes every keyword the orchestrator
passes. The engine docstring previously cited a `tests/test_engine_parity.py` that does
not exist; this test replaces that claim with an enforced one.
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


def test_the_docstring_does_not_name_a_test_that_does_not_exist():
    """Every test file the engine docstring names as a guard must exist.

    The docstring cited a `tests/test_engine_parity.py` that was never written. Naming a
    guard that does not exist is worse than naming none, because it discourages checking.
    This scans the "Anti-drift guards" section and asserts each file it points at is real.
    """
    import re

    from src.strategy.inference import engine

    doc = engine.__doc__ or ""
    section = doc.split("Anti-drift guard", 1)[-1].split("\n\n", 1)[0]
    for name in re.findall(r"tests/test_[a-z_]+\.py", section):
        assert (ROOT / name).exists(), (
            f"the engine docstring names {name} as an anti-drift guard, but it does not exist"
        )
