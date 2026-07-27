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

import pytest

from src.agents.strategy_orchestrator import RaceState, _build_orchestrator_prompt


@pytest.fixture
def prompt() -> str:
    """A prompt with no sub-agent outputs, so only the static blocks remain."""
    race_state = RaceState(
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
def test_the_regulation_example_still_cites_no_hardcoded_article(prompt: str) -> None:
    """The two-compound article is renumbered between seasons, so the example must not name one.

    30.5(n) in 2023, 30.5(m) in 2024, 30.5(i) in 2025. The prompt asks the LLM to
    quote article numbers, so any number hardcoded in the worked example gets
    echoed into the output and is wrong for two seasons out of three.
    """
    assert "30.5(" not in prompt
