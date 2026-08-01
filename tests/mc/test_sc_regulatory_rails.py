"""What a deployed Safety Car forces, and what it must not.

The governing rule, set after the SC rail was found to be an opinion rather than a
rule: **a deterministic rail may encode what the FIA Sporting Regulations make
certain. It must never encode a strategy opinion.** Facts are forced; the pit/stay
decision stays with the model, which is the only layer that sees the race state the
decision actually depends on (stops made, laps remaining, gap behind, compounds used).

The rail these tests replace forced ``STAY_OUT -> PIT_NOW`` on every SC lap. That was
a single race (Qatar 2025) generalised into a universal law, and Art. 55.17 makes it
provably wrong in the case it fired hardest on.

Every assertion here runs without an LLM, against the real helpers.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.conftest import HAS_TIRE_MODELS as _HAS_MODELS

# Importing the orchestrator pulls the sub-agent modules, which read model configs at
# import time, so this carries the same guard as the other agent-touching tests.
ROOT = Path(__file__).parent.parent.parent
pytestmark = pytest.mark.skipif(
    not _HAS_MODELS,
    reason="data/models/ not present (CI runner without model weights)",
)


def _recommend_under_sc(action: str, reason: str | None):
    """Assemble a recommendation the way the no-LLM path does, with an SC deployed."""
    from src.agents.strategy_orchestrator import _assemble_recommendation
    from src.strategy.inference.no_llm import _deterministic_synthesis

    synthesis = _deterministic_synthesis(action, reason)
    mc_results = {"STAY_OUT": {"score": 0.4}, "PIT_NOW": {"score": -1.2}}
    return _assemble_recommendation(synthesis, None, mc_results, "", sc_currently_active=True)


def test_sc_in_the_last_laps_does_not_force_a_stop():
    """Art. 55.17: an SC still out on the last lap ends the race behind it.

    Laps under the SC count as race laps (Art. 55.16), and if the SC is still deployed
    at the start of the last lap the cars take the flag behind it "without overtaking"
    (Art. 55.17). So a stop in the closing laps surrenders track position that is not
    merely hard to recover: it is unrecoverable by regulation.

    The guard-rail already knows this and returns STAY_OUT. This asserts that nothing
    downstream overrides it. Before the rail was removed, the pipeline shipped
    ``action=PIT_NOW`` carrying the reason "too late to pit".

    ``sc_active=True`` is passed explicitly since #716 gave the bounds that parameter.
    Until then this test named a Safety Car it never actually declared, so it was
    asserting about a green lap while claiming to be about a neutralised one.
    """
    from src.strategy.inference.no_llm import apply_guard_rails

    action, reason = apply_guard_rails(
        action="PIT_NOW",
        lap=55,
        total_laps=57,
        compound="MEDIUM",
        tyre_life=30,
        cliff_p10=99.0,
        sc_active=True,
    )
    assert action == "STAY_OUT", "the guard-rail itself changed; this test's premise is gone"
    assert "too late to pit" in (reason or "")

    rec = _recommend_under_sc(action, reason)

    assert rec.action == "STAY_OUT", (
        "an SC in the closing laps must not force a stop: Art. 55.17 guarantees the race "
        "finishes behind the SC, so pitting here is a certain loss"
    )
    assert "OVERRIDE" not in rec.reasoning


def test_the_shipped_action_never_contradicts_its_own_reason():
    """A recommendation may not carry a reason that argues against its action.

    This is the shape the rail produced: it rewrote ``action`` at final assembly while
    every field derived from it, ``guardrail_reason`` included, kept describing the
    decision it had just overridden. A deterministic layer may only write a field it is
    the authority for, and ``action`` is not a leaf.
    """
    from src.strategy.inference.no_llm import apply_guard_rails

    action, reason = apply_guard_rails(
        action="PIT_NOW",
        lap=55,
        total_laps=57,
        compound="MEDIUM",
        tyre_life=30,
        cliff_p10=99.0,
        sc_active=True,
    )
    rec = _recommend_under_sc(action, reason)

    says_too_late = "too late to pit" in (rec.reasoning or "")
    if says_too_late:
        assert rec.action != "PIT_NOW", (
            f"shipped action={rec.action!r} while its own reasoning says it is too late "
            f"to pit: {rec.reasoning!r}"
        )


def test_no_green_flag_lap_time_target_under_a_safety_car():
    """Art. 55.7 removes the field's only source, so it has no valid value.

    ``target_lap_time_s`` is grounded in N06's PaceOutput CI, and N06 predicts
    green-flag pace. While the SC is out, drivers must stay ABOVE the FIA ECU minimum
    time, so a green-flag target sits below the delta by construction: shipping it
    instructs the driver to earn a penalty. The delta itself is not in our telemetry,
    so there is nothing to substitute. None is forced by absence of a source, which is
    what separates this from a strategy opinion.
    """
    from src.agents.strategy_orchestrator import _assemble_recommendation
    from src.strategy.inference.no_llm import _deterministic_synthesis

    synthesis = _deterministic_synthesis("STAY_OUT", None)
    synthesis.target_lap_time_s = 84.5  # a green-flag target
    mc_results = {"STAY_OUT": {"score": 0.4}}

    under_sc = _assemble_recommendation(synthesis, None, mc_results, "", sc_currently_active=True)
    assert under_sc.target_lap_time_s is None, (
        "a green-flag lap-time target under an SC tells the driver to break Art. 55.7"
    )

    green = _assemble_recommendation(synthesis, None, mc_results, "", sc_currently_active=False)
    assert green.target_lap_time_s == 84.5, "the target must survive when the track is green"


def test_drs_window_is_shut_on_neutralised_laps():
    """Art. 22.1(c): no DRS under SC/VSC.

    The feature is a plain ``gap < 1.0`` test, so a neutralised lap reported an open DRS
    window purely because the field had bunched up: live and lying on exactly the laps
    where the regulation shuts it.
    """
    from src.agents.race_situation_agent import _is_neutralised

    assert _is_neutralised("4") is True, "code 4 is the Safety Car"
    assert _is_neutralised("6") is True, "code 6 is the VSC"
    assert _is_neutralised("41") is True, "FastF1 packs codes: a lap that saw the SC at all"
    assert _is_neutralised("1") is False, "code 1 is green"
    assert _is_neutralised(None) is False, "unknown status must not fake a neutralisation"


@pytest.mark.parametrize("mc_favours_staying_out", [True, False])
def test_the_model_keeps_the_pit_decision_under_a_safety_car(mc_favours_staying_out):
    """The stop/stay call is race state, not regulation, so no rail may take it.

    Under an SC the Monte Carlo already receives the full ``SC_PIT_BONUS`` on every
    draw (``sc_prob_3lap`` is forced to 1.0, so every sample sees an SC). If it still
    scores STAY_OUT highest, that is the model saying the cheap stop was outweighed.
    Overriding it silences the exact computation built to weigh it.
    """
    chosen = "STAY_OUT" if mc_favours_staying_out else "PIT_NOW"
    rec = _recommend_under_sc(chosen, None)
    assert rec.action == chosen, (
        "an active SC must not rewrite the chosen action; it is a strategy call that "
        "depends on stops made, laps remaining and gap behind, none of which is a rule"
    )
