"""The engine must pass the orchestrator's layer functions everything the orchestrator does.

The engine removes duplication by importing the orchestrator's layer functions rather
than copying their bodies, but it re-drives the call sequence, so every argument is
threaded by hand. Importing the functions prevents body drift, not call drift.

That gap is a real failure mode: when `_assemble_recommendation` gained a `live_drivers`
argument (#462, to keep the LLM from naming a retired car as an undercut target), the
orchestrator threaded it at both call sites and the engine did not. `live_drivers=None`
means "unknown" and passes the LLM value unchecked, so the guard was inactive on the
`rich` profile, which is the default for /simulate, the arcade and the CLI.

Two functions take hand-threaded arguments, and for a long time only one was checked.
`_build_orchestrator_prompt` is the other, and it is the more exposed of the two: three
production call sites (`strategy_orchestrator.py:2164`, `:2339`, `engine.py:277`) against
the assembly's two. Every argument added to the prompt builder is therefore a chance to
repeat #462 on the surface where it is hardest to see, because a prompt that silently
lost a block still returns a perfectly well-formed recommendation.

The guarantee this file provides is that the engine passes every keyword the orchestrator
passes, for both callees. Read the direction note on the test itself before trusting a
green run for anything else. The engine docstring previously cited a
`tests/test_engine_parity.py` that does not exist; this test replaces that claim with an
enforced one.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent.parent
_HAS_MODELS = (ROOT / "data" / "models" / "tire_degradation" / "routing_config.json").exists()
pytestmark = pytest.mark.skipif(
    not _HAS_MODELS,
    reason="data/models/ not present (CI runner without model weights)",
)


# The engine re-drives the orchestrator's sequence, so it threads arguments by hand
# into BOTH of the layer functions that take them. Only the assembly was covered until
# the memory audit pointed out the omission: `_build_orchestrator_prompt` has three
# production call sites (strategy_orchestrator.py:2164, :2339, engine.py:277), the same
# shape that produced the two failures in this file's docstring.
_THREADED_CALLEES = ("_assemble_recommendation", "_build_orchestrator_prompt")


def _kwargs_passed_by(func, callee: str = "_assemble_recommendation") -> set[str]:
    """The keyword names `func` passes to `callee`."""
    import ast

    tree = ast.parse(inspect.getsource(func).lstrip())
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == callee
        ):
            return {kw.arg for kw in node.keywords if kw.arg}
    return set()


@pytest.mark.parametrize("callee", _THREADED_CALLEES)
def test_the_engine_threads_every_argument_the_orchestrator_does(callee: str):
    """A new argument on a layer function must reach the engine, or its guard is dead.

    Compares the keywords the engine passes against the ones the orchestrator passes.
    Anything the orchestrator threads and the engine does not is, by construction, a
    feature that works everywhere except the default path.

    KNOW WHICH DIRECTION THIS COVERS. It asserts `orch_kwargs - engine_kwargs`, i.e.
    the orchestrator has something the engine lacks. That is the direction both real
    failures took. An argument the ENGINE passes and the orchestrator deliberately does
    not is invisible here and passes green - which is exactly the shape a per-lap memory
    block would have, since /recommend and the MCP tool are stateless per request and
    cannot carry one. Do not read a green run as "both prompt paths are equivalent".
    """
    import src.agents.strategy_orchestrator as orch
    from src.strategy.inference import engine

    engine_kwargs = _kwargs_passed_by(engine._run_rich, callee)
    orch_kwargs = _kwargs_passed_by(orch.run_strategy_orchestrator_from_state, callee)

    assert orch_kwargs, f"no call to {callee} found in run_strategy_orchestrator_from_state"

    missing = orch_kwargs - engine_kwargs
    assert not missing, (
        f"the engine's rich profile does not thread {sorted(missing)} into "
        f"{callee}, so whatever those arguments guard is dead on every "
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
