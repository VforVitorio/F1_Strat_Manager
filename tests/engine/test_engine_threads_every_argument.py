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

from pathlib import Path

import pytest

from tests.conftest import HAS_TIRE_MODELS as _HAS_MODELS
from tests.engine.ast_helpers import kwargs_passed_by

ROOT = Path(__file__).parent.parent.parent
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
    """The keyword names `func` passes to `callee`, with this file's default callee."""
    return kwargs_passed_by(func, callee)


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

    The pattern allows a subdirectory. It did not, and every guard the docstring names
    now lives in `tests/engine/`, so this test matched NOTHING and passed green while
    asserting about zero files - the same defect it was written to catch, one level up.
    The count assertion is what stops that recurring: a pattern that stops matching is
    indistinguishable from a docstring with nothing to check.
    """
    import re

    from src.strategy.inference import engine

    doc = engine.__doc__ or ""
    section = doc.split("Anti-drift guard", 1)[-1].split("\n\n", 1)[0]
    named = re.findall(r"tests/[a-z_/]*test_[a-z_]+\.py", section)

    assert named, "the engine docstring's anti-drift section names no test files at all"
    for name in named:
        assert (ROOT / name).exists(), (
            f"the engine docstring names {name} as an anti-drift guard, but it does not exist"
        )


def test_both_profiles_open_the_draw_capture_and_forward_the_rejoin():
    """The rich profile and the no-LLM one must agree about the PIT EXIT readout.

    The capture kwarg landed on the rich profile alone. The no-LLM one still ran
    the same projection and still deposited nothing, so `_rejoin_from` saw an
    empty dict and the card rendered its "no rejoin geometry" idle on laps where
    the projection HAD run. That is worse than an absent card: the copy claims
    there is nothing to compute while the geometry is sitting one call away.

    This is the same drift the rest of this file guards, one layer down. The
    engine imports the orchestrator's functions so their BODIES cannot diverge,
    and then re-drives the sequence twice, once per profile, by hand.

    Static, like its siblings: the question is about the call site, and calling
    either profile for real needs the weights and a race state.
    """
    from src.strategy.inference import engine, no_llm

    for name, func in (("rich", engine._run_rich), ("no_llm", no_llm.run_no_llm_lap)):
        mc_kwargs = kwargs_passed_by(func, "_run_mc_simulation")
        assert mc_kwargs, f"{name} no longer calls _run_mc_simulation; this guard is dead"
        assert "capture" in mc_kwargs, (
            f"the {name} profile scores the candidates without opening the draw "
            f"capture, so the rejoin readout has nothing to price the stop with"
        )

        outputs_kwargs = kwargs_passed_by(func, "_assemble_agent_outputs")
        assert outputs_kwargs, f"{name} no longer calls _assemble_agent_outputs"
        assert "rejoin" in outputs_kwargs, (
            f"the {name} profile captured the draws and then dropped the readout "
            f"on the floor before the surfaces could see it"
        )
