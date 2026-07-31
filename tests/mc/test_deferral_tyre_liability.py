"""#763 — an elective stop's full pit loss used to stand against nothing at all.

`_terminal_gaps` carries every KNOWN outstanding stop to a common horizon. What it did
not carry is what the rubber costs while you defer, and for a car whose mandatory stop
is already discharged that was the only future cost it had. So PIT_NOW paid ~22.8 s and
STAY_OUT paid zero, and the layer declined **69.9%** of elective stops against 26.7% of
first ones.

WHY THE FIX IS NOT A WIDER WINDOW
----------------------------------
That was this issue's first diagnosis and a design gate refuted it with executed
evidence. Widening is monotone in favour of PIT with today's arithmetic, pushes first
calls even earlier (the direction that had just cost five exact agreements), and would
need W around 37 to touch the elective declines at all.

THE MEASURED BASIS
-------------------
Over 694 real elective stops in 2023-24, the horizon a stop takes to repay itself from
its own pace advantage has a **median of 13 laps** (95% CI [12, 14]); only **15.0%**
repay inside five. A five-lap window can price at most a seventh of the decision.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.agents.position_projection import (
    DriverPlan,
    ProjectionConfig,
    RivalState,
    _deferral_tyre_liability_s,
    _terminal_gaps,
)

DRAWS = 4


def _config(**overrides) -> ProjectionConfig:
    base = {
        "window_laps": 5,
        "racing_laps": 5.0,
        "deg_cost_s": 0.5,
        "cliff_loss_s": 0.0,
        "neutralisation_saving_s": 0.0,
        "future_neutralisation_prob": 0.0,
        "laps_remaining": 25,
        "mandatory_stop_pending": False,
    }
    base.update(overrides)
    return ProjectionConfig(**base)


def _pit_loss(value: float = 22.8) -> np.ndarray:
    return np.full(DRAWS, value, dtype=float)


def test_a_discharged_obligation_still_pays_for_the_rubber_it_defers():
    """The whole point: before this, staying out on an elective stop cost zero."""
    liability = _deferral_tyre_liability_s(_pit_loss(), _config())

    assert (liability > 0).all()


def test_the_liability_is_the_cheaper_of_the_two_futures():
    """A car chooses; it does not pay both. Running 20 laps at 0.5 s is 10 s, which is
    cheaper than the 22.8 s a later stop still costs, so the run-it-out branch wins."""
    liability = _deferral_tyre_liability_s(_pit_loss(22.8), _config())

    assert liability == pytest.approx(np.full(DRAWS, 20 * 0.5))


def test_a_cheap_later_stop_wins_over_running_a_long_way_on_old_rubber():
    """The other branch, so the minimum is pinned in both directions rather than one.

    Without this, a term that always took the run-it-out branch would pass the test
    above and be wrong whenever the stop is the cheaper future.
    """
    liability = _deferral_tyre_liability_s(_pit_loss(4.0), _config(laps_remaining=45))

    assert liability == pytest.approx(np.full(DRAWS, 4.0))


def test_no_tyre_reading_means_no_claim():
    """`deg_cost_s` is None on a stint the model had no reference for. Charging a
    guessed liability there would be inventing the very number this epic refused to
    invent twice already."""
    assert _deferral_tyre_liability_s(_pit_loss(), _config(deg_cost_s=None)) == pytest.approx(
        np.zeros(DRAWS)
    )


def test_a_car_that_still_owes_its_stop_is_untouched(monkeypatch):
    """E3, the invariance criterion, at the unit level.

    A deferral term for elective stops has no business moving mandatory-stop timing.
    That population already measures balanced (26.7% declines) and charging it there
    pushes first calls earlier, which is what #744b just paid five exact agreements
    for. If this ever fires on pending=True, the scoping leaked.
    """
    called = []
    import src.agents.position_projection as pp

    monkeypatch.setattr(
        pp, "_deferral_tyre_liability_s", lambda *a, **k: called.append(1) or np.zeros(DRAWS)
    )

    rival = RivalState(driver="VER", gap_s=3.0, stop_pending=False)
    plan = DriverPlan(name="STAY_OUT", stops_in_window=False)
    _terminal_gaps(
        [rival], plan, np.zeros((DRAWS, 1)), _pit_loss(), _config(mandatory_stop_pending=True)
    )

    assert called == []


def test_an_unknown_obligation_buys_no_correction_either_way():
    """`None` means the compound history could not settle it, and this module's rule is
    that a claim needs a fact. Treating unknown as discharged would charge a liability
    on evidence nobody has."""
    plan = DriverPlan(name="STAY_OUT", stops_in_window=False)
    gaps = np.full((DRAWS, 1), 3.0)
    rival = RivalState(driver="VER", gap_s=3.0, stop_pending=False)

    unknown = _terminal_gaps([rival], plan, gaps, _pit_loss(), _config(mandatory_stop_pending=None))

    assert unknown == pytest.approx(gaps)


def test_a_plan_that_stops_owes_no_deferral_at_all():
    """It is not deferring. Charging it here would double-count against its own pit loss."""
    plan = DriverPlan(name="PIT_NOW", stops_in_window=True)
    gaps = np.full((DRAWS, 1), 3.0)
    rival = RivalState(driver="VER", gap_s=3.0, stop_pending=False)

    assert _terminal_gaps([rival], plan, gaps, _pit_loss(), _config()) == pytest.approx(gaps)


def test_the_liability_shrinks_as_the_flag_approaches():
    """With three laps left there is almost nothing left to defer, and the term must
    say so rather than charging a full race of rubber. This is the Art. 55.17 endgame
    shape the rest of the module already expresses."""
    late = _deferral_tyre_liability_s(_pit_loss(), _config(laps_remaining=6))
    early = _deferral_tyre_liability_s(_pit_loss(), _config(laps_remaining=40))

    assert (late < early).all()
    assert _deferral_tyre_liability_s(_pit_loss(), _config(laps_remaining=5)) == pytest.approx(
        np.zeros(DRAWS)
    )
