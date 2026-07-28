"""Contract for ``DecisionMemory``, the per-race accumulator.

Hermetic: no models, no data, no LLM. ``decision_memory`` imports nothing from
``src.agents``, which is deliberate - it is the only piece of the memory work that
can be tested on a CI runner with no weights, so it carries the whole behavioural
contract and the surfaces only have to be checked for wiring.

Every assertion here traces to a measurement in
``documents/audits/AUDIT_ORCHESTRATOR_MEMORY.md``; the docstrings name which.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.strategy.inference.decision_memory import (
    CONTINUATION,
    COUNTERWEIGHT,
    MAX_CONTINGENCIES,
    DecisionMemory,
)


def _contingency(trigger: str, switch_to: str = "PIT_NOW", priority: str = "HIGH"):
    return SimpleNamespace(trigger=trigger, switch_to=switch_to, priority=priority)


def _rec(action="STAY_OUT", pit_lap_target=None, contingencies=()):
    """A stand-in for StrategyRecommendation with only the fields memory reads."""
    return SimpleNamespace(
        action=action, pit_lap_target=pit_lap_target, contingencies=list(contingencies)
    )


@pytest.mark.unit
def test_no_history_renders_no_block():
    """First lap of a race: there is no previous call, so do not invent one.

    A "held for 0 laps" line would be a statement about a decision that does not
    exist, which is how a prompt teaches a model something untrue.
    """
    assert DecisionMemory().block() is None


@pytest.mark.unit
def test_a_clean_run_reports_the_span_in_laps():
    memory = DecisionMemory()
    for lap in range(5, 13):
        memory.record(lap, _rec(pit_lap_target=30))

    block = memory.block()
    assert "held since lap 5 (8 laps)" in block
    assert "decisions across" not in block, "no gap here, so do not report one"


@pytest.mark.unit
def test_a_gap_is_reported_as_a_gap_not_hidden_in_a_count():
    """The bare-count bug, demonstrated in audit section 2.4.

    Every surface `continue`s past laps it cannot decide on (a retired car, an
    incomplete lap, a lap that raised). A bare "held for 7 laps" while 16 elapsed
    is a false statement the model has no way to check, so both numbers are stated.
    """
    memory = DecisionMemory()
    for lap in range(5, 11):  # 6 decisions
        memory.record(lap, _rec())
    memory.record(20, _rec())  # laps 11-19 skipped

    block = memory.block()
    assert "7 decisions across 16 laps" in block
    assert "held for 7 laps" not in block


@pytest.mark.unit
def test_an_action_change_resets_the_run():
    memory = DecisionMemory()
    for lap in range(5, 10):
        memory.record(lap, _rec(action="STAY_OUT"))
    memory.record(10, _rec(action="UNDERCUT"))
    memory.record(11, _rec(action="UNDERCUT"))

    block = memory.block()
    assert "Last call: UNDERCUT, held since lap 10" in block


@pytest.mark.unit
def test_target_drift_is_reported_with_its_sign():
    """A target that moves every lap is a plan that does not exist (audit 1.7).

    Without memory the target moved 311 laps in total across a 57-lap race, so the
    drift is the point of the field, not the latest value.
    """
    memory = DecisionMemory()
    for lap, target in zip(range(5, 10), [30, 31, 33, 35, 36]):
        memory.record(lap, _rec(pit_lap_target=target))

    assert "30, 31, 33, 35, 36 (net drift +6 laps)" in memory.block()


@pytest.mark.unit
def test_a_missing_target_is_rendered_as_none_not_as_a_number():
    """`None` means "no stop planned in the visible horizon" and must stay distinct.

    This repo has a scar from a sentinel colliding with a real value; rendering an
    absent target as 0 would put a searchable lap number in the prompt.
    """
    memory = DecisionMemory()
    memory.record(5, _rec(pit_lap_target=None))
    memory.record(6, _rec(pit_lap_target=None))

    block = memory.block()
    assert "none, none" in block
    assert "drift" not in block, "two unknowns cannot produce a drift"


@pytest.mark.unit
def test_only_the_last_lap_of_contingencies_is_echoed_and_it_is_capped():
    """The block must not grow with race length (audit 1.5 / 3.3).

    A trigger is free text with no evaluator, so no code can retire one. The
    cumulative reading would have carried 80 stale lines by lap 45.
    """
    memory = DecisionMemory()
    memory.record(5, _rec(contingencies=[_contingency("old trigger from lap 5")]))
    memory.record(
        6,
        _rec(contingencies=[_contingency(f"trigger {i}") for i in range(MAX_CONTINGENCIES + 3)]),
    )

    block = memory.block()
    assert "old trigger from lap 5" not in block
    assert block.count("    - [") == MAX_CONTINGENCIES


@pytest.mark.unit
def test_the_counterweight_is_always_present():
    """Without it, memory ANCHORED the model on the one lap that mattered.

    At Lusail 2025 lap 44, agreement with the deterministic Monte Carlo was 6/10
    with no memory, 4/10 with memory, and 10/10 with memory plus this sentence.
    It is part of the block, not a caller's option.
    """
    memory = DecisionMemory()
    memory.record(5, _rec())
    assert COUNTERWEIGHT in memory.block()


@pytest.mark.unit
def test_a_single_call_is_not_a_continuation():
    """One STAY_OUT is a call, not a plan being carried.

    This is the case that made the instruction wrong in the static prompt: it also
    fired on the very first decision of a race, telling the model not to re-argue a
    case it had never argued.
    """
    memory = DecisionMemory()
    memory.record(5, _rec())

    block = memory.block()
    assert CONTINUATION not in block
    assert COUNTERWEIGHT in block, "the counterweight is unconditional; only this line is not"


@pytest.mark.unit
def test_a_repeated_stay_out_is_told_it_is_continuing_a_plan():
    """The one case the instruction was always meant for (audit section 3.1)."""
    memory = DecisionMemory()
    memory.record(5, _rec())
    memory.record(6, _rec())

    assert CONTINUATION in memory.block()


@pytest.mark.unit
def test_a_changed_call_is_not_a_continuation():
    """The other case it was wrong in: the lap the plan actually changed.

    A STAY_OUT that follows an UNDERCUT is a new decision. Telling the model not to
    re-argue it is telling it not to think about the only lap that moved.
    """
    memory = DecisionMemory()
    memory.record(5, _rec())
    memory.record(6, _rec(action="UNDERCUT"))
    memory.record(7, _rec())

    assert CONTINUATION not in memory.block()


@pytest.mark.unit
def test_a_repeated_pit_call_is_not_a_continuing_plan():
    """STAY_OUT is the only action a race can sit on.

    A repeated PIT_NOW is not a plan being carried, it is a stop that has not
    happened, and it is the last thing that should be discouraged from re-arguing.
    """
    memory = DecisionMemory()
    memory.record(5, _rec(action="PIT_NOW"))
    memory.record(6, _rec(action="PIT_NOW"))

    assert CONTINUATION not in memory.block()


@pytest.mark.unit
def test_the_span_reads_as_english_for_a_single_lap():
    """It rendered "(1 laps)" on the first held lap of every real run."""
    memory = DecisionMemory()
    memory.record(20, _rec())

    assert "held since lap 20 (1 lap)." in memory.block()


@pytest.mark.unit
def test_the_first_call_of_a_race_is_not_a_change():
    """Nothing to compare against, so a surface must not open the panel on lap one."""
    memory = DecisionMemory()
    memory.record(5, _rec())

    assert memory.last_call_changed() is False


@pytest.mark.unit
def test_a_repeated_action_is_not_a_change():
    memory = DecisionMemory()
    memory.record(5, _rec())
    memory.record(6, _rec())

    assert memory.last_call_changed() is False


@pytest.mark.unit
def test_a_new_action_is_a_change():
    """The lap the call actually moved: this is the one worth explaining."""
    memory = DecisionMemory()
    memory.record(5, _rec())
    memory.record(6, _rec(action="PIT_NOW"))

    assert memory.last_call_changed() is True


@pytest.mark.unit
def test_a_drifting_target_under_an_unchanged_call_is_NOT_a_change():
    """Measured, not stylistic: this is what stops the signal becoming wallpaper.

    Over 40 lap pairs of a real race the action changed on 0 and `pit_lap_target`
    moved on 25 (62%). Counting the target would open the panel on two laps in
    three. A target drifting under a held call is what the block's drift line
    reports; it is not a change of plan.
    """
    memory = DecisionMemory()
    memory.record(5, _rec(pit_lap_target=30))
    memory.record(6, _rec(pit_lap_target=44))

    assert memory.last_call_changed() is False


@pytest.mark.unit
def test_recording_backwards_raises_rather_than_producing_a_false_span():
    memory = DecisionMemory()
    memory.record(10, _rec())

    with pytest.raises(ValueError, match="forward-only"):
        memory.record(9, _rec())
    with pytest.raises(ValueError, match="forward-only"):
        memory.record(10, _rec())


@pytest.mark.unit
def test_a_recommendation_with_no_contingencies_says_so_explicitly():
    """Silence and "none declared" are different messages to the model."""
    memory = DecisionMemory()
    memory.record(5, _rec())
    assert "Contingencies you declared last lap: none." in memory.block()
