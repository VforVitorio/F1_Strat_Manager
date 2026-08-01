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


def _no_cliff() -> np.ndarray:
    """Cliff far past the flag, so the cliff half of the liability contributes nothing.

    Explicit rather than defaulted: an earlier version of the production code assumed
    everything past the window was past the cliff, and charged a median 19 laps of it
    on real laps -- including 189 where the tyre model said the set cost nothing at all.
    Tests that leave the onset implicit cannot see that.
    """
    return np.full(DRAWS, 999.0, dtype=float)


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
    liability = _deferral_tyre_liability_s(_pit_loss(), _no_cliff(), _config())

    assert (liability > 0).all()


def test_the_liability_is_the_cheaper_of_the_two_futures():
    """A car chooses; it does not pay both. Running 20 laps at 0.5 s is 10 s, which is
    cheaper than the 22.8 s a later stop still costs, so the run-it-out branch wins."""
    liability = _deferral_tyre_liability_s(_pit_loss(22.8), _no_cliff(), _config())

    assert liability == pytest.approx(np.full(DRAWS, 20 * 0.5))


def test_a_cheap_later_stop_wins_over_running_a_long_way_on_old_rubber():
    """The other branch, so the minimum is pinned in both directions rather than one.

    Without this, a term that always took the run-it-out branch would pass the test
    above and be wrong whenever the stop is the cheaper future.
    """
    liability = _deferral_tyre_liability_s(_pit_loss(4.0), _no_cliff(), _config(laps_remaining=45))

    assert liability == pytest.approx(np.full(DRAWS, 4.0))


def test_no_tyre_reading_means_no_claim():
    """`deg_cost_s` is None on a stint the model had no reference for. Charging a
    guessed liability there would be inventing the very number this epic refused to
    invent twice already."""
    assert _deferral_tyre_liability_s(
        _pit_loss(), _no_cliff(), _config(deg_cost_s=None)
    ) == pytest.approx(np.zeros(DRAWS))


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
        [rival],
        plan,
        np.zeros((DRAWS, 1)),
        _pit_loss(),
        _no_cliff(),
        _config(mandatory_stop_pending=True),
    )

    assert called == []


def test_an_unknown_obligation_buys_no_correction_either_way():
    """`None` means the compound history could not settle it, and this module's rule is
    that a claim needs a fact. Treating unknown as discharged would charge a liability
    on evidence nobody has."""
    plan = DriverPlan(name="STAY_OUT", stops_in_window=False)
    gaps = np.full((DRAWS, 1), 3.0)
    rival = RivalState(driver="VER", gap_s=3.0, stop_pending=False)

    unknown = _terminal_gaps(
        [rival], plan, gaps, _pit_loss(), _no_cliff(), _config(mandatory_stop_pending=None)
    )

    assert unknown == pytest.approx(gaps)


def test_a_plan_that_stops_owes_no_deferral_at_all():
    """It is not deferring. Charging it here would double-count against its own pit loss."""
    plan = DriverPlan(name="PIT_NOW", stops_in_window=True)
    gaps = np.full((DRAWS, 1), 3.0)
    rival = RivalState(driver="VER", gap_s=3.0, stop_pending=False)

    assert _terminal_gaps(
        [rival], plan, gaps, _pit_loss(), _no_cliff(), _config()
    ) == pytest.approx(gaps)


def test_the_liability_shrinks_as_the_flag_approaches():
    """With three laps left there is almost nothing left to defer, and the term must
    say so rather than charging a full race of rubber. This is the Art. 55.17 endgame
    shape the rest of the module already expresses."""
    late = _deferral_tyre_liability_s(_pit_loss(), _no_cliff(), _config(laps_remaining=6))
    early = _deferral_tyre_liability_s(_pit_loss(), _no_cliff(), _config(laps_remaining=40))

    assert (late < early).all()
    assert _deferral_tyre_liability_s(
        _pit_loss(), _no_cliff(), _config(laps_remaining=5)
    ) == pytest.approx(np.zeros(DRAWS))


def test_the_terminal_gap_of_a_deferring_car_actually_MOVES():
    """The gate's HIGH: deleting the wiring left all 173 `tests/mc` green.

    Every other test here either exercises the helper directly, or asserts the term
    does NOT apply. Not one asserted the thing the feature exists to do — that a car
    with a discharged obligation, deferring on worn rubber, ends the race further back
    than its window-end gap says. So the whole feature could be disconnected and the
    suite would not notice, which is this project's most-repeated defect and the third
    time a gate has found it in my own work.
    """
    plan = DriverPlan(name="STAY_OUT", stops_in_window=False)
    gaps = np.full((DRAWS, 1), 3.0)
    rival = RivalState(driver="VER", gap_s=3.0, stop_pending=False)

    terminal = _terminal_gaps([rival], plan, gaps, _pit_loss(), _no_cliff(), _config())

    # Deferring costs us seconds, and our loss pushes every gap DOWN (the module's
    # stated sign convention), so the terminal gap is strictly below the projected one.
    assert (terminal < gaps).all()


def test_the_cliff_half_uses_the_onset_the_model_supplied_not_an_assumed_one():
    """Measured on real elective laps, the assumed onset charged a median 19 laps of
    cliff — and on 189 of them the tyre model reported the set cost NOTHING per lap
    versus fresh while the liability still charged a median 8.41 s of pure invention.

    Same quantity, same rule, both horizons: `driver_time_delta` has always used the
    per-draw onset for the in-window half.
    """
    early_cliff = np.full(DRAWS, 2.0, dtype=float)
    late_cliff = np.full(DRAWS, 999.0, dtype=float)
    config = _config(cliff_loss_s=0.8, deg_cost_s=0.0)

    # With no per-lap wear at all, the only thing left is the cliff. A set that never
    # reaches it must owe nothing; the earlier code charged it anyway.
    assert _deferral_tyre_liability_s(_pit_loss(), late_cliff, config) == pytest.approx(
        np.zeros(DRAWS)
    )
    assert (_deferral_tyre_liability_s(_pit_loss(), early_cliff, config) > 0).all()


def test_a_safety_car_cheapens_the_stop_and_not_the_rubber():
    """The exit gate's q_f blind spot: nothing caught discounting BOTH branches.

    A neutralisation makes a STOP cheaper. It does not make worn tyres cheaper. So the
    option value belongs to the stop-later branch, where `_stop_residual_s` applies it,
    and the `min` is what lets a cheapened stop win once it becomes the better future.
    Discounting the run-it-out branch as well credited the same Safety Car twice and
    handed a car that never stops a saving it has no way to collect.

    Pinned as an invariance: raising the neutralisation odds must not move a liability
    whose cheaper future is holding the set to the flag.
    """
    # Heavy wear over a long run, so holding the set to the flag costs far more than
    # the stop and the STOP branch wins. That is where the option value belongs, so
    # this is where raising the Safety Car odds must show.
    heavy = dict(deg_cost_s=1.5, laps_remaining=45, neutralisation_saving_s=8.0)
    calm = _deferral_tyre_liability_s(_pit_loss(), _no_cliff(), _config(**heavy))
    likely_sc = _deferral_tyre_liability_s(
        _pit_loss(), _no_cliff(), _config(**heavy, future_neutralisation_prob=0.5)
    )
    assert (likely_sc < calm).all(), "the stop branch must carry the option value"

    # Now a regime where running it out is the cheaper future. The Safety Car odds must
    # not touch it: no neutralisation reduces what the rubber costs.
    light = dict(deg_cost_s=0.5, laps_remaining=25, neutralisation_saving_s=8.0)
    calm_run = _deferral_tyre_liability_s(_pit_loss(), _no_cliff(), _config(**light))
    sc_run = _deferral_tyre_liability_s(
        _pit_loss(), _no_cliff(), _config(**light, future_neutralisation_prob=0.5)
    )
    assert sc_run == pytest.approx(calm_run)
    # And the regimes really are different, or both halves would be asserting about
    # whichever branch happens to win everywhere.
    assert calm[0] != calm_run[0]
