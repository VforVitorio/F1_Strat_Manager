"""Guards on the Layer 3 synthesis prompt.

The prompt is the whole of the LLM's instruction surface: 12 of the 14
``StrategyRecommendation`` fields come from it (6 verbatim from the synthesis,
6 synthesis-then-clamped) and only ``scenario_scores`` and
``regulation_context`` are assembled deterministically. A silent edit here
changes every recommendation the system emits and no other test would notice,
because nothing else asserts on prompt text.

--- WHERE TO CHANGE IF THE PROMPT CHANGES ---
``_build_orchestrator_prompt`` in ``src/agents/strategy_orchestrator.py`` is
the only producer. If a block below is deliberately reworded, update the
assertion rather than deleting it.
"""

from pathlib import Path

import pytest

from tests.conftest import HAS_TIRE_MODELS as _HAS_MODELS

# Importing the orchestrator pulls in the tire agent, which reads its routing
# config at import time, so the whole module is unimportable without the HF
# weights. That is why the import below lives inside the fixture rather than at
# module scope: a module-level import raises during COLLECTION, which no skipif
# can catch. Same guard as tests/agents/test_agents.py, same reason.
ROOT = Path(__file__).parent.parent.parent

pytestmark = pytest.mark.skipif(
    not _HAS_MODELS,
    reason="data/models/ not present (CI environment without model weights)",
)


@pytest.fixture
def race_state():
    """The one RaceState every prompt in this file is built from."""
    from src.agents.strategy_orchestrator import RaceState

    return RaceState(
        driver="NOR",
        lap=25,
        total_laps=57,
        position=2,
        compound="MEDIUM",
        tyre_life=25,
        gap_ahead_s=2.4,
        pace_delta_s=0.15,
        risk_tolerance=0.5,
        air_temp=22.7,
        track_temp=28.1,
    )


@pytest.fixture
def prompt(race_state) -> str:
    """A prompt with no sub-agent outputs, so only the static blocks remain."""
    from src.agents.strategy_orchestrator import _build_orchestrator_prompt

    return _build_orchestrator_prompt(race_state, {}, "STAY_OUT")


@pytest.mark.unit
def test_stay_out_is_framed_as_an_active_posture_not_a_fallback(prompt: str) -> None:
    """STAY_OUT is the majority call, so the prompt must not define it by subtraction.

    Measured before the fix: 39 of 41 green-flag laps at Lusail 2025 and 414 of
    415 across the projection set resolve to STAY_OUT, while every mention of it
    in the guard-rails arrived as "force" or "override to". The model was being
    taught to read its own most frequent output as a failure to decide (#646).
    """
    assert "ACTIVE monitoring posture" in prompt
    assert "not merely what is left when a pit is blocked" in prompt


@pytest.mark.unit
def test_a_held_stay_out_asks_for_a_threshold_rather_than_the_same_case_again(
    prompt: str,
) -> None:
    """Consecutive prompts are 99.0% identical text and carry no memory of the last lap.

    Nothing in the prompt says an action has been held, so the LLM re-argues the
    same case in fresh prose every lap. The block cannot add memory the prompt
    does not have, but it can ask for the shape that makes a hold legible: what
    is watched, what would change it, and how far away that is.
    """
    # Matched as fragments, not as one phrase: the block is a wrapped f-string
    # and every demand below straddles a line break in the source.
    for demand in ("what you are watching", "the concrete threshold", "would change the call"):
        assert demand in prompt, f"the held-STAY_OUT block no longer asks for: {demand}"


@pytest.mark.unit
def test_the_continuation_instruction_is_not_in_the_unconditioned_prompt(prompt: str) -> None:
    """ "Do not re-argue the same case" is only true when there IS a previous case.

    It used to sit in the static block, so it fired on lap 1, when there was nothing
    to continue, and on the lap the call changed, when continuing was the wrong
    answer. It now lives in `DecisionMemory`'s block, which is the only place that
    knows whether anything is being repeated. `/recommend` and the MCP tool are
    stateless per request, so for them this instruction is now correctly absent
    rather than unconditionally present.

    If this fails, check whether the sentence was reintroduced here instead of being
    conditioned there.
    """
    assert "CONTINUING a plan" not in prompt
    assert "re-argue the same case" not in prompt


@pytest.mark.unit
def test_the_default_memory_block_leaves_the_prompt_byte_identical(race_state) -> None:
    """The memoryless surfaces must get exactly the prompt they got before the parameter existed.

    ``/recommend`` and the MCP tool are stateless per request, so they will never
    pass a block and must not be changed by the fact that others can. Byte
    equality is the only assertion that proves it: a prompt that gained a stray
    blank line still returns a well-formed recommendation, so nothing else in the
    suite would notice.

    Both spellings of "no memory" are checked, because ``DecisionMemory.block()``
    returns ``None`` before there is any history and a caller passing it straight
    through is the natural way to use it. Without the guard in the builder that
    renders the literal string "None" above ``RACE CONTEXT:``, which equality with
    the empty-block prompt is what catches. (A bare "None" not in prompt would not
    work here: the field-schema instructions use the word legitimately.)
    """
    from src.agents.strategy_orchestrator import _build_orchestrator_prompt

    baseline = _build_orchestrator_prompt(race_state, {}, "STAY_OUT")

    assert _build_orchestrator_prompt(race_state, {}, "STAY_OUT", memory_block="") == baseline
    assert _build_orchestrator_prompt(race_state, {}, "STAY_OUT", memory_block=None) == baseline


@pytest.mark.unit
def test_a_supplied_memory_block_lands_between_the_framing_and_the_facts(race_state) -> None:
    """Position is the point: the model reads what it decided before it reads this lap.

    Above ``RACE CONTEXT:`` and below the held-STAY_OUT framing, which is the anchor
    the A/B harness spliced at while measuring this. If the block moves, the measured
    result stops applying to the shipped prompt.
    """
    from src.agents.strategy_orchestrator import _build_orchestrator_prompt

    block = "DECISION MEMORY (your own previous calls this race):\n  Last call: STAY_OUT.\n\n"
    prompt = _build_orchestrator_prompt(race_state, {}, "STAY_OUT", memory_block=block)

    assert block in prompt
    assert prompt.index("ACTIVE monitoring posture") < prompt.index(block)
    assert prompt.index(block) < prompt.index("RACE CONTEXT:")


@pytest.mark.unit
def test_the_real_decision_memory_block_renders_into_the_prompt(race_state) -> None:
    """Against the shipping producer, not a hand-written string.

    The two objects are wired together by nothing but a string, so a change to
    ``DecisionMemory``'s rendering that broke the placement would otherwise be
    caught by no test in either file.
    """
    from types import SimpleNamespace

    from src.agents.strategy_orchestrator import _build_orchestrator_prompt
    from src.strategy.inference.decision_memory import DecisionMemory

    memory = DecisionMemory()
    memory.record(
        5,
        SimpleNamespace(
            action="STAY_OUT",
            pit_lap_target=22,
            contingencies=[
                SimpleNamespace(
                    trigger="SC deployed within 3 laps", switch_to="PIT_NOW", priority="HIGH"
                )
            ],
        ),
    )
    prompt = _build_orchestrator_prompt(race_state, {}, "STAY_OUT", memory_block=memory.block())

    assert "DECISION MEMORY" in prompt
    assert "SC deployed within 3 laps" in prompt
    # The counterweight is not optional decoration: without it the block measurably
    # anchored the model at the decision lap.
    assert "NOT a commitment" in prompt
    assert prompt.index("DECISION MEMORY") < prompt.index("RACE CONTEXT:")


@pytest.mark.unit
def test_the_regulation_example_still_cites_no_hardcoded_article(prompt: str) -> None:
    """The two-compound article is renumbered between seasons, so the example must not name one.

    30.5(n) in 2023, 30.5(m) in 2024, 30.5(i) in 2025. The prompt asks the LLM to
    quote article numbers, so any number hardcoded in the worked example gets
    echoed into the output and is wrong for two seasons out of three.
    """
    assert "30.5(" not in prompt
