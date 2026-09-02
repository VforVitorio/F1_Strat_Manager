"""#744b — the tyre channel reaches BOTH scorers, and it is charged on the right laps.

Until now the Monte Carlo's only tyre signal was the cliff, a discrete event that
falls inside the 5-lap window on 4 of 110 measured laps. A set 0.4 s off the pace but
ten laps from the cliff scored identically to a fresh one, which is most of why the
decision layer declines to call a large share of real stops. The level is deliberately
not quoted here, because it moves with every harness revision and every widening of the
sample. `documents/eval_reports/decision_modes.md` regenerates it, and a copy in a test
docstring only ever goes stale.

WHAT IS BEING REPLACED, AND WHY IT IS A REPLACEMENT
----------------------------------------------------
``FRESH_GAIN = 0.25  # s/lap advantage of fresh vs degraded tyre`` is the same
quantity, hardcoded and applied identically at tyre life 3 and tyre life 25. So the
measured cost REPLACES it rather than adding to it, and the double-count trap is
avoided by that fact rather than by a correction term. ``FRESH_GAIN`` survives as the
no-signal fallback, which is what keeps a stint with no reference scoring exactly as
it did before.

THE LAP COUNTS ARE THE WHOLE THING
-----------------------------------
The two prices sit on opposite sides of the stop: the old credit paid for laps on the
NEW set, the new charge bills laps on the OLD one. Get those counts wrong and the term
becomes a constant offset that cancels in the argmax and looks like it did nothing at
all. Per candidate, with ``window = 5``:

    STAY_OUT              5 old, 0 fresh
    PIT_NOW / UNDERCUT    0 old, 5 fresh
    OVERCUT under SC      0 old, 5 fresh
    OVERCUT green         2 old, 2 fresh   <- the only split, and window // 2

The projection scorer counts differently because it knows when in the window the stop
falls: ``laps_before_stop`` on the old set, ``laps_after_stop`` on the new one, and the
whole window on the old set for a plan that does not stop at all.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.agents.position_projection import DriverPlan, ProjectionConfig, driver_time_delta
from tests.conftest import HAS_TIRE_MODELS as _HAS_MODELS

ROOT = Path(__file__).parent.parent.parent

DRAWS = 4


def _config(**overrides) -> ProjectionConfig:
    """A config with every term but the tyre ones zeroed, so the charge is readable."""
    base = {
        "window_laps": 5,
        "racing_laps": 5.0,
        "fresh_gain_s": 0.25,
        "cliff_loss_s": 0.0,
        "neutralisation_saving_s": 0.0,
        "clean_air_gain_s": 0.0,
        "neutralisation_onset_rate": 0.0,
    }
    base.update(overrides)
    return ProjectionConfig(**base)


def _zeros() -> np.ndarray:
    return np.zeros(DRAWS, dtype=float)


def _no_cliff() -> np.ndarray:
    """Cliff far outside the window, so the cliff term contributes nothing."""
    return np.full(DRAWS, 99.0, dtype=float)


# ---------------------------------------------------------------------------
# The projection scorer — hermetic, so CI sees these
# ---------------------------------------------------------------------------


def test_staying_out_pays_the_wear_on_every_racing_lap():
    """A plan that never stops runs the whole window on the old set."""
    config = _config(deg_cost_s=0.4)

    delta = driver_time_delta(
        DriverPlan(name="STAY_OUT", stops_in_window=False, stop_offset_laps=0),
        _zeros(),
        _no_cliff(),
        config,
    )

    assert delta == pytest.approx(np.full(DRAWS, 5.0 * 0.4))


def test_pitting_immediately_pays_no_wear_at_all():
    """Zero laps on the old set, so the charge vanishes without needing a branch."""
    config = _config(deg_cost_s=0.4)

    delta = driver_time_delta(
        DriverPlan(name="PIT_NOW", stops_in_window=True, stop_offset_laps=0),
        _zeros(),
        _no_cliff(),
        config,
    )

    assert delta == pytest.approx(_zeros())


def test_an_overcut_pays_for_the_laps_it_waits_and_no_more():
    """The term has to track WHEN the stop falls, not just whether it happens.

    A stop deferred two laps runs two laps on the old set. If this were charged over
    the whole window instead, every candidate would shift by the same amount and the
    argmax would not move — the term would be invisible while looking connected.
    """
    config = _config(deg_cost_s=0.4)

    delta = driver_time_delta(
        DriverPlan(name="OVERCUT", stops_in_window=True, stop_offset_laps=2),
        _zeros(),
        _no_cliff(),
        config,
    )

    assert delta == pytest.approx(np.full(DRAWS, 2.0 * 0.4))


def test_a_worn_tyre_makes_staying_out_worse_than_stopping():
    """The behaviour the whole issue exists for, stated as an inequality.

    Nothing here says by how much; it says the sign is right. A term wired with the
    wrong sign still moves the numbers and would pass every count test above.
    """
    config = _config(deg_cost_s=0.4)
    stay = driver_time_delta(
        DriverPlan(name="STAY_OUT", stops_in_window=False, stop_offset_laps=0),
        _zeros(),
        _no_cliff(),
        config,
    )
    pit = driver_time_delta(
        DriverPlan(name="PIT_NOW", stops_in_window=True, stop_offset_laps=0),
        _zeros(),
        _no_cliff(),
        config,
    )

    assert stay.mean() > pit.mean()


def test_a_fresh_set_costs_nothing_and_is_not_the_same_as_no_reading():
    """0.0 is a real reading: the wear term vanishes, and the FRESH_GAIN fallback
    must NOT come back to life instead. That distinction is the sentinel rule this
    field is built on, checked here at the consuming end rather than the producing
    one."""
    measured_zero = _config(deg_cost_s=0.0)
    no_reading = _config(deg_cost_s=None)
    plan = DriverPlan(name="PIT_NOW", stops_in_window=True, stop_offset_laps=0)

    with_zero = driver_time_delta(plan, _zeros(), _no_cliff(), measured_zero)
    without = driver_time_delta(plan, _zeros(), _no_cliff(), no_reading)

    assert with_zero == pytest.approx(_zeros())
    assert without == pytest.approx(np.full(DRAWS, -5.0 * 0.25))


def test_with_no_reading_the_scorer_is_byte_identical_to_the_old_credit():
    """The fallback is not an approximation of the previous behaviour, it IS it."""
    config = _config(deg_cost_s=None)

    for plan in (
        DriverPlan(name="STAY_OUT", stops_in_window=False, stop_offset_laps=0),
        DriverPlan(name="PIT_NOW", stops_in_window=True, stop_offset_laps=0),
        DriverPlan(name="OVERCUT", stops_in_window=True, stop_offset_laps=2),
    ):
        delta = driver_time_delta(plan, _zeros(), _no_cliff(), config)
        laps_after_stop = 0.0 if not plan.stops_in_window else 5.0 - plan.stop_offset_laps
        assert delta == pytest.approx(np.full(DRAWS, -laps_after_stop * 0.25))


def test_the_cliff_term_stacks_on_top_of_the_wear_rather_than_replacing_it():
    """They price different THINGS, and past the cliff they are deliberately additive.

    The wear is what an old set costs on every lap it is held. The cliff is an EXTRA
    penalty once the set is past it. So before the cliff only the wear applies, and past
    it both do — 5 laps charged 0.4 plus the same 5 charged 0.8.

    Named for what it pins. An earlier version of this test was called "do_not_overlap"
    and its docstring claimed the two bill disjoint laps, which is the opposite of the
    second assertion below: they bill the SAME laps for different reasons. A test name
    stating the reverse of its own assertion is worse than no name (gate G2, LOW).

    The property that matters for #744b is not disjointness, it is that a tyre ten laps
    from the cliff now costs something at all, where it used to cost exactly nothing.
    """
    config = _config(deg_cost_s=0.4, cliff_loss_s=0.8)
    plan = DriverPlan(name="STAY_OUT", stops_in_window=False, stop_offset_laps=0)

    before_cliff = driver_time_delta(plan, _zeros(), _no_cliff(), config)
    past_cliff = driver_time_delta(plan, _zeros(), np.zeros(DRAWS), config)

    assert before_cliff == pytest.approx(np.full(DRAWS, 5.0 * 0.4))
    assert past_cliff == pytest.approx(np.full(DRAWS, 5.0 * 0.4 + 5.0 * 0.8))


# ---------------------------------------------------------------------------
# The legacy scorer — reached whenever a caller has no usable rival gaps
# ---------------------------------------------------------------------------
#
# Not a relic. TWO shipping builders hardcode `"rivals": []`, which routes here:
# `engine.py::_build_default_lap_state` and `strategy_orchestrator.py`, both serving a
# caller that supplies only a RaceState. `_has_usable_gaps` also demotes to this branch
# whenever every rival interval is NaN.
#
# NOT the backend endpoint, despite what #744 says and despite what the first version of
# this comment repeated. `api/v1/endpoints/strategy.py:882` also hardcodes an empty rival
# list, but its only consumer calls `run_pace_agent_from_state`, so it never reaches the
# Monte Carlo; the backend's own MC paths pass real rivals. Traced twice, once here and
# once by the gate that found this comment still carrying the uncorrected claim after the
# commit message had been fixed — one copy corrected, its twin not, inside the very PR
# that was correcting it.


@pytest.mark.skipif(
    not _HAS_MODELS,
    reason="strategy_orchestrator imports the agent stack, which loads model bundles",
)
class TestLegacyScorer:
    """Charged in seconds, before the conversion to positions."""

    def test_each_candidate_is_charged_for_its_own_old_set_laps(self):
        from src.agents.strategy_orchestrator import _tyre_term

        assert _tyre_term(0.4, old_laps=5, fresh_laps=0) == pytest.approx(-2.0)
        assert _tyre_term(0.4, old_laps=0, fresh_laps=5) == pytest.approx(0.0)
        assert _tyre_term(0.4, old_laps=2, fresh_laps=2) == pytest.approx(-0.8)

    def test_without_a_reading_it_is_the_old_fresh_credit_unchanged(self):
        from src.agents.strategy_orchestrator import FRESH_GAIN, _tyre_term

        assert _tyre_term(None, old_laps=5, fresh_laps=0) == pytest.approx(0.0)
        assert _tyre_term(None, old_laps=0, fresh_laps=5) == pytest.approx(FRESH_GAIN * 5)
        assert _tyre_term(None, old_laps=2, fresh_laps=2) == pytest.approx(FRESH_GAIN * 2)

    def test_the_two_prices_are_never_charged_together(self):
        """The double-count trap, closed by construction rather than by a correction:
        with a reading, the fresh credit is not merely offset, it is absent."""
        from src.agents.strategy_orchestrator import _tyre_term

        assert _tyre_term(0.4, old_laps=0, fresh_laps=5) == pytest.approx(0.0)

    def test_a_worn_set_makes_staying_out_score_worse(self):
        """Same inequality as the projection branch, on the other implementation.

        The two scorers have opposite sign conventions — this one accumulates a gain,
        the other a loss — so a term copied across without flipping it would pass the
        count tests on both and fail exactly here.
        """
        from src.agents.strategy_orchestrator import simulate_lap_window

        kwargs = dict(cliff_i=99.0, sc_i=False, pit_i=22.0, ucut_i=False)
        fresh = simulate_lap_window("STAY_OUT", deg_cost_s=0.0, **kwargs)
        worn = simulate_lap_window("STAY_OUT", deg_cost_s=0.4, **kwargs)

        assert worn < fresh

    def test_the_overcut_green_branch_splits_the_window(self):
        """The only candidate that runs on both sets, and the one whose counts are
        easiest to get wrong: `window // 2` on each side, not the whole window."""
        from src.agents.strategy_orchestrator import WINDOW_LAPS, _tyre_term

        half = WINDOW_LAPS // 2
        assert _tyre_term(0.4, old_laps=half, fresh_laps=half) == pytest.approx(-0.4 * half)


def _rivals_with_usable_gaps() -> list[dict]:
    """Rivals the projection branch will actually accept.

    The key is `interval_to_driver_s`, and getting it wrong is not a loud failure:
    `_has_usable_gaps` simply returns False and `_run_mc_simulation` falls through to
    the LEGACY scorer. A test that thought it was exercising the projection branch
    would then pass while proving nothing about it, which is what the first version of
    this file did — caught by mutating the projection call site and watching all
    fifteen tests stay green.
    """
    return [
        {"driver": "VER", "interval_to_driver_s": -2.0, "is_pitting": False},
        {"driver": "LEC", "interval_to_driver_s": 3.5, "is_pitting": False},
    ]


# ---------------------------------------------------------------------------
# The wiring itself — the part a comment cannot assert
# ---------------------------------------------------------------------------
#
# Everything above tests the two leaf functions. Neither one proves the value
# actually TRAVELS from TireOutput through `_run_mc_simulation` into each branch,
# and that is precisely the defect gate G1 found in the previous PR of this epic:
# the mechanism a commit calls essential, protected by a comment and nothing else.
#
# `_run_projection_mc` did not receive `tire_out` in any form before #744b, so the
# projection branch needed a new kwarg. A change that added it to one branch and not
# the other would leave every existing test green: the goldens run with no reading at
# all and take the fallback.


@pytest.mark.skipif(
    not _HAS_MODELS,
    reason="_run_mc_simulation imports the agent stack, which loads model bundles",
)
class TestTheValueReachesBothBranches:
    """One canned scenario, scored twice, differing only in the tyre reading."""

    @staticmethod
    def _score(rivals, deg_cost_s):
        from dataclasses import replace

        from src.agents.strategy_orchestrator import _run_mc_simulation
        from tests.mc.canned_outputs import canned_outputs as _canned_outputs

        pace, tire, situation, pit = _canned_outputs()
        return _run_mc_simulation(
            pace_out=pace,
            tire_out=replace(tire, deg_cost_s=deg_cost_s),
            situation_out=situation,
            pit_out=pit,
            alpha=0.5,
            rivals=rivals,
            position=5,
            laps_remaining=25,
            pit_context=None,
        )

    def test_the_legacy_branch_receives_it(self):
        """Reached whenever a caller passes no usable gaps, which two shipping builders
        do by hardcoding an empty rival list, and which `_has_usable_gaps` also forces
        whenever every rival interval is NaN."""
        without = self._score([], None)
        with_wear = self._score([], 0.4)

        assert with_wear["STAY_OUT"]["E"] != without["STAY_OUT"]["E"]
        assert with_wear["STAY_OUT"]["E"] < without["STAY_OUT"]["E"]

    def test_the_projection_branch_receives_it(self):
        """Reached when rivals carry usable gaps: arcade, the CLI, the eval harness."""
        rivals = _rivals_with_usable_gaps()
        without = self._score(rivals, None)
        with_wear = self._score(rivals, 0.4)

        assert with_wear["STAY_OUT"]["E"] != without["STAY_OUT"]["E"]

    def test_neither_branch_moves_when_there_is_no_reading(self):
        """The fallback keeps every pre-#744b caller on exactly its old numbers,
        which is why the frozen goldens did not need re-freezing."""
        rivals = _rivals_with_usable_gaps()

        assert self._score([], None) == self._score([], None)
        assert self._score(rivals, None) == self._score(rivals, None)

    def test_every_candidate_pays_at_its_own_call_site_not_just_the_helper(self):
        """Gate G2's surviving mutant: zeroing OVERCUT's old-lap count AT THE CALL SITE
        left all 232 tests green.

        `test_the_overcut_green_branch_splits_the_window` above tests `_tyre_term` with
        the right arguments. It cannot see whether `simulate_lap_window` passes them,
        and nothing else evaluated OVERCUT with a reading present. So this reads the
        four candidates through the public function and pins each one's charge as the
        DIFFERENCE between a worn reading and a zero one, which isolates the tyre term
        from the pit loss and the cliff without having to restate either.
        """
        from src.agents.strategy_orchestrator import WINDOW_LAPS, simulate_lap_window

        deg = 0.4
        # Cliff far away and no neutralisation, so only the tyre term differs.
        kwargs = dict(cliff_i=99.0, sc_i=False, pit_i=22.0, ucut_i=False)

        def charge(strategy: str) -> float:
            """Positions the reading costs this candidate, positive = worse."""
            return simulate_lap_window(strategy, deg_cost_s=0.0, **kwargs) - simulate_lap_window(
                strategy, deg_cost_s=deg, **kwargs
            )

        from src.agents.strategy_orchestrator import POS_GAP_S

        expected = {
            "STAY_OUT": deg * WINDOW_LAPS / POS_GAP_S,
            "PIT_NOW": 0.0,
            "UNDERCUT": 0.0,
            "OVERCUT": deg * (WINDOW_LAPS // 2) / POS_GAP_S,
        }
        for strategy, want in expected.items():
            assert charge(strategy) == pytest.approx(want), strategy

    def test_under_a_neutralisation_the_overcut_runs_no_old_set_laps(self):
        """The SC arm of OVERCUT pits immediately, so it owes nothing for the old set.

        Pinned separately from the green arm because the two are different branches of
        the same `if`, and the green one is the only candidate in either scorer whose
        laps split across the stop.
        """
        from src.agents.strategy_orchestrator import simulate_lap_window

        kwargs = dict(cliff_i=99.0, sc_i=True, pit_i=22.0, ucut_i=False)

        assert simulate_lap_window("OVERCUT", deg_cost_s=0.4, **kwargs) == pytest.approx(
            simulate_lap_window("OVERCUT", deg_cost_s=0.0, **kwargs)
        )

    def test_the_neutralised_config_carries_the_reading_too(self):
        """Gate G2's other surviving mutant: setting the value on the green config but
        NOT the neutralised one left all 232 tests green.

        The mutant I claimed to have run was not that mutant. It disabled the value
        whenever `clean_air_gain_s == 0.0`, which is also true of the green config in
        that fixture, so it took out both arms — which is why it went red for the wrong
        reason. A genuine green-only mutant survived.

        Catching it needs a FULLY neutralised scenario (`sc_prob_3lap = 1.0`, so every
        draw takes the neutralised config) and rivals close enough that the neutralised
        window's wear actually crosses a gap. Under a Safety Car the window is worth
        ~2.6 racing laps, so the charge is small and a wide gap absorbs it silently —
        which is exactly how the previous fixture passed while blind.
        """
        from dataclasses import replace

        from src.agents.strategy_orchestrator import _run_mc_simulation
        from tests.mc.canned_outputs import canned_outputs as _canned_outputs

        pace, tire, situation, pit = _canned_outputs()
        rivals = [
            {"driver": "VER", "interval_to_driver_s": -0.4, "is_pitting": False},
            {"driver": "LEC", "interval_to_driver_s": 0.5, "is_pitting": False},
        ]

        def score(deg_cost_s):
            return _run_mc_simulation(
                pace_out=pace,
                tire_out=replace(tire, deg_cost_s=deg_cost_s),
                situation_out=replace(situation, sc_prob_3lap=1.0),
                pit_out=pit,
                alpha=0.5,
                rivals=rivals,
                position=5,
                laps_remaining=25,
                pit_context=None,
            )

        assert score(0.8)["STAY_OUT"]["E"] != score(None)["STAY_OUT"]["E"]
        assert score(0.8)["STAY_OUT"]["E"] != score(0.0)["STAY_OUT"]["E"]
