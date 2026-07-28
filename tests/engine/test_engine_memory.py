"""The memory block the caller supplies must actually reach the Layer 3 prompt.

Nothing else in the suite can tell you this. `run_lap`'s parameter defaults to `None`,
so every existing engine test passes whatever the memory path does — including doing
nothing at all. That is the price of keeping the engine pure, and it is why this file
exists rather than being folded into `test_engine_no_llm`.

HOW IT IS CHECKED, and why not end to end. Driving `run_lap(profile="rich")` for real
does not work as a test of this: the rich profile builds LLM clients for the always-on
sub-agents long before the prompt is assembled, so the first attempt at this file failed
on a connection error rather than on anything about memory. The chain is therefore
verified as its three links instead, each one hermetically:

  1. `run_lap` turns a `DecisionMemory` into a block string and hands it to `_run_rich`
  2. `_run_rich` passes that string to `_build_orchestrator_prompt`
  3. the builder renders it into the prompt — `tests/agents/test_orchestrator_prompt.py`

Link 2 is a static check for the same reason `test_engine_threads_every_argument` is:
the question is about a call site, and that guard cannot answer this one because it
compares the engine against the orchestrator, which deliberately does not pass memory.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from tests.engine.ast_helpers import kwargs_passed_by

ROOT = Path(__file__).parent.parent.parent
_HAS_MODELS = (ROOT / "data" / "models" / "tire_degradation" / "routing_config.json").exists()
pytestmark = pytest.mark.skipif(
    not _HAS_MODELS,
    reason="data/models/ not present (CI runner without model weights)",
)

FIXTURE = ROOT / "tests" / "fixtures" / "mini_race.parquet"
TRIGGER = "SC deployed within 3 laps"


def _memory_with_one_lap(lap: int = 5):
    from src.strategy.inference.decision_memory import DecisionMemory

    memory = DecisionMemory()
    memory.record(
        lap,
        SimpleNamespace(
            action="STAY_OUT",
            pit_lap_target=22,
            contingencies=[SimpleNamespace(trigger=TRIGGER, switch_to="PIT_NOW", priority="HIGH")],
        ),
    )
    return memory


def _race_state(lap: int = 6):
    from src.agents.strategy_orchestrator import RaceState

    return RaceState(
        driver="NOR",
        lap=lap,
        total_laps=57,
        position=5,
        compound="MEDIUM",
        tyre_life=10,
        gap_ahead_s=2.0,
        pace_delta_s=0.0,
        risk_tolerance=0.5,
        air_temp=25.0,
        track_temp=35.0,
    )


@pytest.fixture
def rich_call(monkeypatch):
    """Capture the arguments `run_lap` hands to `_run_rich`, without running it."""
    from src.strategy.inference import engine

    seen: dict = {}

    def _capture(*args, **kwargs):
        seen["args"] = args
        seen["kwargs"] = kwargs
        return ("rec", None, {})

    monkeypatch.setattr(engine, "_run_rich", _capture)
    return seen


@pytest.mark.unit
def test_run_lap_renders_the_memory_into_a_block_for_the_rich_path(rich_call):
    """Link 1: a `DecisionMemory` in, the rendered block out.

    Asserted on the contingency trigger, because the echo is the load-bearing field —
    it is what took the Safety Car experiment from 0 of 8 to 8 of 8. A block that
    rendered its header and dropped its contingencies would still look like memory
    in a diff.
    """
    from src.strategy.inference.engine import run_lap

    run_lap(_race_state(), pd.read_parquet(FIXTURE), None, memory=_memory_with_one_lap())

    block = rich_call["args"][-1]
    assert "DECISION MEMORY" in block
    assert TRIGGER in block


@pytest.mark.unit
def test_run_lap_without_memory_sends_an_empty_block(rich_call):
    """The default must be indistinguishable from before the parameter existed.

    Empty string, not `None`: the two are equivalent inside the builder, but only
    because it guards for it, and this pins which one the engine actually sends.
    """
    from src.strategy.inference.engine import run_lap

    run_lap(_race_state(), pd.read_parquet(FIXTURE), None)

    assert rich_call["args"][-1] == ""


@pytest.mark.unit
def test_the_rich_path_passes_the_block_to_the_prompt_builder():
    """Link 2, statically: the argument survives the last hop.

    `test_engine_threads_every_argument` cannot cover this. It asserts the engine
    passes everything the ORCHESTRATOR passes, and the orchestrator deliberately does
    not pass memory (see `test_memory_scope_is_deliberate`), so the block reaching the
    builder is asserted by nothing but this line.
    """
    from src.strategy.inference import engine

    passed = kwargs_passed_by(engine._run_rich, "_build_orchestrator_prompt")

    assert "memory_block" in passed, (
        "the rich path builds the prompt without the memory block, so every surface "
        "that wires an accumulator is accumulating into nothing"
    )


@pytest.mark.unit
def test_run_lap_does_not_record_into_the_memory_it_is_given(rich_call):
    """Recording is the CALLER's job, and this is what keeps it that way.

    If the engine recorded, `run_lap` would stop being pure per lap and
    `test_engine_no_llm`'s twice-on-lap-6 equality would start depending on call
    order — a failure that would surface far from its cause.
    """
    from src.strategy.inference.engine import run_lap

    memory = _memory_with_one_lap()
    before = memory.block()

    run_lap(_race_state(), pd.read_parquet(FIXTURE), None, memory=memory)

    assert memory.block() == before, (
        "run_lap mutated the caller's DecisionMemory; the accumulator is caller-owned "
        "and the engine must only read it"
    )


def test_the_no_llm_profile_accepts_a_memory_and_ignores_it():
    """The documented behaviour, made executable.

    `no-llm` builds no prompt, so there is nowhere for a block to go. Passing one has
    to be harmless rather than a `TypeError` — surfaces switch profiles at runtime and
    would otherwise need a conditional at every call site.
    """
    from src.strategy.inference.engine import run_lap

    laps_df = pd.read_parquet(FIXTURE)
    memory = _memory_with_one_lap()

    with_memory = run_lap(_race_state(), laps_df, None, profile="no-llm", memory=memory)[0]
    without = run_lap(_race_state(), laps_df, None, profile="no-llm")[0]

    assert with_memory.action == without.action
    assert memory.block() is not None
