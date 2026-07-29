"""The pit bounds, and exactly which of them a Safety Car suspends.

These run everywhere. The rails moved to a leaf module in #708 precisely so that
reading a bound costs no model load, and this file is the payoff: until now the
only tests covering them sat behind a `data/models/` gate and therefore never ran
on a push.

The subject under test is a **proscriptive** bound — it forbids an action so a
generative model cannot emit nonsense. That is a different object from a
prescriptive rail that forces one, and the two are judged differently: a
proscriptive bound needs calibration, not a regulation. See #716.
"""

from __future__ import annotations

import pytest

from src.strategy.inference.guard_rails import (
    _MIN_STINT_LAPS,
    _NO_PIT_BEFORE_LAP,
    _NO_PIT_LAST_N_LAPS,
    apply_guard_rails,
)

_GREEN_TOTAL_LAPS = 57


def _pit(
    lap, *, total_laps=_GREEN_TOTAL_LAPS, compound="MEDIUM", tyre_life=30, cliff=99.0, sc=False
):
    """Ask the bounds whether a PIT_NOW on this lap survives."""
    return apply_guard_rails("PIT_NOW", lap, total_laps, compound, tyre_life, cliff, sc_active=sc)


# --- the bounds fire on green ----------------------------------------------


def test_the_three_bounds_fire_on_a_green_lap():
    """The baseline the exceptions are exceptions to."""
    assert _pit(_NO_PIT_BEFORE_LAP - 1)[0] == "STAY_OUT"
    assert _pit(_GREEN_TOTAL_LAPS - _NO_PIT_LAST_N_LAPS)[0] == "STAY_OUT"
    assert _pit(30, tyre_life=_MIN_STINT_LAPS["MEDIUM"] - 1)[0] == "STAY_OUT"


def test_a_non_pit_action_is_never_touched():
    """The bounds only ever veto a stop; they never manufacture one."""
    for lap in (1, 30, 57):
        assert apply_guard_rails("STAY_OUT", lap, 57, "MEDIUM", 1)[0] == "STAY_OUT"


# --- what a neutralisation suspends -----------------------------------------


def test_a_safety_car_suspends_the_early_race_bound():
    """The bound defends against a ~22-25 s cost that a neutralisation removes.

    A first-lap stop under a Safety Car is ordinary racing, not a hallucination.
    """
    assert _pit(2)[0] == "STAY_OUT"
    assert _pit(2, sc=True)[0] == "PIT_NOW"


def test_a_safety_car_suspends_the_minimum_stint_bound():
    """A cheap stop makes a short stint affordable, which is the whole premise.

    This was the divergence found in #716: the N28 prompt has always exempted an
    active SC here and the deterministic mirror did not, so the offline path
    refused the cheapest stop in racing.
    """
    short = {"tyre_life": _MIN_STINT_LAPS["MEDIUM"] - 1}
    assert _pit(30, **short)[0] == "STAY_OUT"
    assert _pit(30, sc=True, **short)[0] == "PIT_NOW"


def test_a_safety_car_does_NOT_suspend_the_end_of_race_bound():
    """Deliberate divergence from the prompt, and the one that must not regress.

    Art. 55.17 ends the race behind the Safety Car if it is still deployed on the
    final lap, so track position surrendered in the closing laps is unrecoverable
    **by regulation** rather than merely expensive. A neutralisation makes that
    more true, not less. Suspending this bound would re-create the #464 defect,
    where the pipeline shipped PIT_NOW carrying the reason "too late to pit".
    """
    late = _GREEN_TOTAL_LAPS - _NO_PIT_LAST_N_LAPS
    assert _pit(late)[0] == "STAY_OUT"

    action, reason = _pit(late, sc=True)
    assert action == "STAY_OUT", (
        "a Safety Car must not unlock a stop Art. 55.17 makes unrecoverable"
    )
    assert "too late to pit" in (reason or "")


def test_an_imminent_cliff_still_overrides_the_end_of_race_bound():
    """The one exception the end-of-race bound does keep: the tyre is about to fail."""
    late = _GREEN_TOTAL_LAPS - _NO_PIT_LAST_N_LAPS
    assert _pit(late, cliff=99.0)[0] == "STAY_OUT"
    assert _pit(late, cliff=1.0)[0] == "PIT_NOW"


# --- the exception is opt-in ------------------------------------------------


def test_sc_active_defaults_to_false_so_existing_callers_are_unchanged():
    """Adding the parameter must not have moved behaviour for anyone not passing it."""
    with_default = apply_guard_rails("PIT_NOW", 2, 57, "MEDIUM", 30)
    explicit_green = apply_guard_rails("PIT_NOW", 2, 57, "MEDIUM", 30, 99.0, sc_active=False)
    assert with_default == explicit_green


@pytest.mark.parametrize("compound", ["SOFT", "MEDIUM", "HARD"])
def test_every_compound_minimum_is_suspended_by_a_neutralisation(compound):
    """No compound keeps a minimum stint while the field is queued behind the SC."""
    one_short = _MIN_STINT_LAPS[compound] - 1
    assert _pit(30, compound=compound, tyre_life=one_short)[0] == "STAY_OUT"
    assert _pit(30, compound=compound, tyre_life=one_short, sc=True)[0] == "PIT_NOW"
