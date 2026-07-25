"""Invariants on the Monte Carlo decision layer.

These check structure, not accuracy, so they need no ground truth: a decision layer
whose argmax is constant across its inputs carries no information, and that is provable
from the code alone.

Before the OVERCUT fix, over 160 plausible race states:

    OVERCUT  wins 92.5%   (its branch never subtracted pit_i)
    UNDERCUT wins  6.9%
    PIT_NOW  wins  0.6%   (never strictly; dominated by UNDERCUT)
    STAY_OUT wins  0%

One candidate was free, one dominated, one unreachable.

Two paths are swept here. The LEGACY path (no rivals) scores in seconds and is what
the strategy goldens pin; the PROJECTION path (rivals present) scores in projected
track position and is what the redesign added. Both must be real decisions.
"""

from __future__ import annotations

import itertools
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).parent.parent
_HAS_MODELS = (ROOT / "data" / "models" / "tire_degradation" / "routing_config.json").exists()
pytestmark = pytest.mark.skipif(
    not _HAS_MODELS,
    reason="data/models/ not present (CI runner without model weights)",
)

STRATEGIES = ("STAY_OUT", "PIT_NOW", "UNDERCUT", "OVERCUT")

# A sweep of race states that actually occur. `pit_i` spans N15's real distribution
# (~2.2-3.8 s) plus a slow stop and a disaster; `cliff_i` spans fresh tyres to falling off.
_CLIFFS = (0, 1, 3, 5, 10, 20, 50, 99)
_PITS = (2.2, 2.8, 3.8, 6.0, 11.0)
_STATES = tuple(itertools.product(_CLIFFS, (True, False), _PITS, (True, False)))


def _score(strategy: str, state: tuple):
    from src.agents.strategy_orchestrator import simulate_lap_window

    cliff_i, sc_i, pit_i, ucut_i = state
    return simulate_lap_window(strategy, cliff_i=cliff_i, sc_i=sc_i, pit_i=pit_i, ucut_i=ucut_i)


def _argmax(state: tuple) -> str:
    scores = {s: _score(s, state) for s in STRATEGIES}
    return max(scores, key=scores.get)


# ---------------------------------------------------------------------------
# Legacy path — seconds, no rivals. Frozen behaviour, still swept.
# ---------------------------------------------------------------------------


def test_the_argmax_is_not_a_constant():
    """A layer that answers the same thing regardless of its inputs is not computing.

    This is the cheapest possible check that the MC is a function of the race state, and
    it is the one that would have caught OVERCUT collecting fresh tyres without ever
    paying for the stop.
    """
    winners = {_argmax(state) for state in _STATES}
    assert len(winners) > 1, (
        f"the argmax is {winners.pop()!r} for all {len(_STATES)} race states: the Monte "
        f"Carlo is a constant function and its output carries no information"
    )


def test_no_strategy_wins_almost_everything():
    """A candidate that wins nearly always is arbitraging the scoring, not the race.

    The threshold is deliberately loose. This is not a claim about what the right
    distribution is: it is a smoke alarm for a candidate that has been handed an
    advantage the others pay for.
    """
    wins = Counter(_argmax(state) for state in _STATES)
    top, count = wins.most_common(1)[0]
    share = count / len(_STATES)
    assert share < 0.85, (
        f"{top} is the argmax on {share:.1%} of race states ({count}/{len(_STATES)}); "
        f"a candidate that dominates this hard is being scored on different terms than "
        f"the rest. Full tally: {dict(wins)}"
    )


def test_staying_out_is_reachable():
    """The reference must still be choosable.

    `STAY_OUT` is the baseline and cannot score positive: that is its definition, not a
    defect. But if nothing else can ever score *below* it, it is not a reference, it is
    an option the model has been forbidden to take. Staying out is the right call often
    enough in a real race that never choosing it is a bug on its own.
    """
    assert any(_argmax(state) == "STAY_OUT" for state in _STATES), (
        "STAY_OUT is never the argmax across the whole state sweep: the model can never "
        "recommend staying out, whatever the race is doing"
    )


def test_every_strategy_that_pits_pays_for_the_stop():
    """The structural cause, asserted directly.

    PIT_NOW, UNDERCUT and OVERCUT all put the car in the pit lane, so a slower stop must
    make each of them worse. OVERCUT's branch never read `pit_i`, so it collected the
    fresh-tyre gain and the full SC bonus for free: an arbitrage rather than a strategy.
    """
    fast = (99, True, 2.2, False)
    slow = (99, True, 11.0, False)
    for strategy in ("PIT_NOW", "UNDERCUT", "OVERCUT"):
        assert _score(strategy, slow) < _score(strategy, fast), (
            f"{strategy} scores the same with a 2.2 s stop and an 11.0 s one, so it is "
            f"not paying for the pit stop it makes"
        )


# ---------------------------------------------------------------------------
# Projection path — positions, real rivals
# ---------------------------------------------------------------------------

_DRAWS = 200
_GAPS_AHEAD = (-1.2, -4.5, -15.0)
_GAPS_BEHIND = (1.5, 8.0, 26.0)
_PROJECTION_STATES = tuple(
    itertools.product(
        _GAPS_AHEAD,
        _GAPS_BEHIND,
        (True, False),  # a car ahead is serving its stop right now
        (1.0, 6.0, 20.0),  # laps to the cliff
        (True, False),  # neutralisation in the window
        (True, False),  # we still owe the mandatory stop
        (2.4, 3.6, 9.0),  # physical stop seconds
    )
)


def _projection_scores(state: tuple, gp_name: str | None = None) -> dict:
    """Score one projected race state, returning the four candidates' dicts.

    ``gp_name`` is left unset for the main sweep so the invariants below hold on
    the pooled measurements rather than on any one circuit's quirks. The
    clean-air tests pass it, because that is the only term that is per circuit.
    """
    from src.agents.strategy_orchestrator import _run_projection_mc

    gap_ahead, gap_behind, ahead_pitting, cliff, neutralised, owes_stop, stop_s = state
    rivals = [
        {"driver": "A", "interval_to_driver_s": gap_ahead, "is_pitting": ahead_pitting},
        {"driver": "B", "interval_to_driver_s": gap_behind, "is_pitting": False},
        {"driver": "C", "interval_to_driver_s": gap_behind + 18.0, "is_pitting": False},
    ]
    return _run_projection_mc(
        rivals=rivals,
        position=2,
        laps_remaining=22,
        pit_context={
            "gp_name": gp_name,
            "traversal_s": 21.0,
            "mandatory_stop_pending": owes_stop,
            # No neutralisation_rate override: with no circuit the layer falls
            # back to the pooled measurement anyway, and pinning it here meant a
            # named circuit silently kept the pooled hazard instead of its own.
            "rival_stop_pending": {"B": False, "C": False},
            "rival_pit_loss_s": 23.8,
        },
        cliff_s=np.full(_DRAWS, cliff),
        sc_s=np.full(_DRAWS, neutralised),
        pit_s=np.full(_DRAWS, stop_s),
        ucut_s=(np.arange(_DRAWS) % 2 == 0),
        alpha=0.5,
        neutralisation_saving_s=8.0,
    )


def _projection_argmax(scores: dict) -> str:
    live = {name: cell["score"] for name, cell in scores.items() if cell["score"] is not None}
    return max(live, key=live.get)


@pytest.fixture(scope="module")
def projection_sweep() -> list[dict]:
    return [_projection_scores(state) for state in _PROJECTION_STATES]


def test_the_projection_argmax_is_not_a_constant(projection_sweep):
    winners = {_projection_argmax(scores) for scores in projection_sweep}
    assert len(winners) > 1, (
        f"the projection argmax is {winners.pop()!r} across all "
        f"{len(_PROJECTION_STATES)} race states"
    )


def test_no_projected_candidate_wins_almost_everything(projection_sweep):
    wins = Counter(_projection_argmax(scores) for scores in projection_sweep)
    top, count = wins.most_common(1)[0]
    share = count / len(projection_sweep)
    assert share < 0.85, f"{top} takes {share:.1%} of the projection sweep: {dict(wins)}"


def test_staying_out_and_pitting_are_both_reachable_on_the_projection(projection_sweep):
    """Both must be choosable, and here they are choosable for opposite reasons.

    Staying out wins when the cars behind still owe their own stop or sit beyond our
    pit loss; pitting wins when the deferred stop would cost more later than the
    places it costs now. That trade is the terminal liability doing its job, and it
    is why neither needs a bonus constant to be reachable.
    """
    wins = Counter(_projection_argmax(scores) for scores in projection_sweep)
    assert wins["STAY_OUT"] > 0, f"STAY_OUT never wins on the projection: {dict(wins)}"
    assert wins["PIT_NOW"] > 0, f"PIT_NOW never wins on the projection: {dict(wins)}"


def test_an_undercut_without_a_target_is_ineligible_and_never_scored(projection_sweep):
    """#434's sentinel, dead: no target means no number at all, never a 0.5 coin flip."""
    ineligible = [s for s in projection_sweep if not s["UNDERCUT"]["eligible"]]
    assert ineligible, "the sweep must contain states with no reachable undercut target"
    for scores in ineligible:
        assert scores["UNDERCUT"]["score"] is None
        assert scores["UNDERCUT"]["target"] is None
        assert _projection_argmax(scores) != "UNDERCUT"


def test_an_eligible_undercut_names_the_car_it_is_attacking(projection_sweep):
    eligible = [s for s in projection_sweep if s["UNDERCUT"]["eligible"]]
    assert eligible, "the sweep must contain reachable undercut targets"
    for scores in eligible:
        assert scores["UNDERCUT"]["target"] == "A"
        assert scores["UNDERCUT"]["score"] is not None


def test_an_overcut_is_only_offered_when_a_car_ahead_is_in_the_pit_lane(projection_sweep):
    for scores in projection_sweep:
        if scores["OVERCUT"]["eligible"]:
            assert scores["OVERCUT"]["target"] == "A"
        else:
            assert scores["OVERCUT"]["score"] is None


# ---------------------------------------------------------------------------
# Why a strategist delays a stop, and where it actually pays
#
# An overcut IS a pit stop, taken one lap later. Both candidates pay the same pit
# lane, so the overcut forfeits exactly one lap of fresh rubber and buys one lap
# of running on. It pays only if that lap was worth more than the rubber, and two
# measured quantities say whether it was: the circuit's clean-air gain, and its
# neutralisation hazard times what a neutralised stop saves.
#
# So "can the overcut win?" is not a yes or no about the code. It is arithmetic
# per circuit, and the tests below assert it in both directions — because a
# version that credited either term everywhere would pass a one-sided check.
# ---------------------------------------------------------------------------

# Suzuka measures the largest clean-air gain in the sample and Monza one of the
# smallest, and neither was chosen for that: they are the archetypal
# high-downforce and slipstream circuits, and the measurement independently put
# them at the two ends.
DIRTY_AIR_EXPENSIVE = "Suzuka"
TOW_MATTERS = "Monza"
# Melbourne holds the two terms apart. Its clean-air gain is +0.008 s, which is
# nothing, but it throws more neutralisations per lap than any circuit measured.
# An overcut paying there can only be the waiting term, so this case fails the
# moment the two are conflated or one of them is dropped.
SAFETY_CARS_LIKELY = "Melbourne"


def _live_racing_states() -> list[tuple]:
    """Swept states where an overcut exists at all and the race is running.

    A car ahead must be in the pit lane, and inside the band the clean-air gain
    was measured at. Neutralised states are excluded because both terms are zero
    there by construction: the field is queued, clear track buys no lap time, and
    a Safety Car cannot arrive when it is already out.
    """
    live = [s for s in _PROJECTION_STATES if s[2] and abs(s[0]) <= 2.0 and not s[4]]
    assert live, "the sweep must contain a car pitting from inside the dirty-air band"
    return live


def _overcut_minus_plain_stop(gp_name: str) -> list[float]:
    """Score difference between the overcut and the same stop taken now.

    The head-to-head is the precise question. Comparing argmaxes over four
    candidates answers a coarser one, because position is a whole number: a real
    sub-position edge is invisible unless those seconds happen to cross a car.
    """
    scored = (_projection_scores(state, gp_name) for state in _live_racing_states())
    return [
        s["OVERCUT"]["score"] - s["PIT_NOW"]["score"]
        for s in scored
        if s["OVERCUT"]["score"] is not None
    ]


def test_the_overcut_beats_a_plain_stop_where_dirty_air_is_expensive():
    """At Suzuka the lap in clean air outweighs the lap of fresh rubber it costs."""
    margins = _overcut_minus_plain_stop(DIRTY_AIR_EXPENSIVE)
    assert max(margins) > 0, (
        f"the overcut never scores above a plain stop at {DIRTY_AIR_EXPENSIVE}, where "
        f"clean air measures more than a lap of fresh rubber is worth"
    )
    assert min(margins) >= 0, (
        f"the overcut scores BELOW a plain stop somewhere at {DIRTY_AIR_EXPENSIVE}: the "
        f"clean-air term cannot be reaching every draw"
    )


def test_the_overcut_beats_a_plain_stop_on_safety_car_odds_alone():
    """Melbourne proves the waiting term exists independently of clean air.

    Clean air is worth nothing there, so anything the overcut gains has to be the
    option value of one more lap spent waiting for a neutralisation that makes
    the stop cheap. Same call as Suzuka, entirely different reason.
    """
    margins = _overcut_minus_plain_stop(SAFETY_CARS_LIKELY)
    assert max(margins) > 0, (
        f"the overcut never scores above a plain stop at {SAFETY_CARS_LIKELY}, whose "
        f"clean-air gain is nil but whose onset hazard is the highest measured: the "
        f"waiting term is not being priced"
    )


def test_the_overcut_loses_to_a_plain_stop_where_the_tow_is_worth_more():
    """At Monza it must lose, and that is a result rather than a limitation.

    Losing the car ahead there costs a slipstream worth more than the clear track
    gives back, and the hazard is too low to make up the difference. A version
    that credited either term uniformly would fail here, which is the point.
    """
    margins = _overcut_minus_plain_stop(TOW_MATTERS)
    assert max(margins) <= 0, (
        f"the overcut scores above a plain stop at {TOW_MATTERS}, where the measured "
        f"clean-air gain is at or below zero: a term is being applied without its circuit"
    )
    assert min(margins) < 0, (
        f"the overcut and a plain stop are indistinguishable at {TOW_MATTERS}: the lap "
        f"of fresh rubber it gives up is not being charged"
    )


def test_the_overcut_is_reachable_as_the_final_answer():
    """Beating PIT_NOW is not enough — it has to be able to win outright."""
    winners = [
        _projection_argmax(_projection_scores(state, DIRTY_AIR_EXPENSIVE))
        for state in _live_racing_states()
    ]
    assert "OVERCUT" in winners, (
        f"OVERCUT is never the argmax at {DIRTY_AIR_EXPENSIVE}: it can out-score a plain "
        f"stop but never becomes the recommendation"
    )


def test_a_slower_stop_is_worse_for_every_projected_candidate_that_pits():
    fast = (-4.5, 8.0, False, 20.0, False, True, 2.4)
    slow = (-4.5, 8.0, False, 20.0, False, True, 11.0)
    fast_scores, slow_scores = _projection_scores(fast), _projection_scores(slow)
    for name in ("PIT_NOW", "UNDERCUT"):
        if fast_scores[name]["score"] is None or slow_scores[name]["score"] is None:
            continue
        assert slow_scores[name]["score"] <= fast_scores[name]["score"], (
            f"{name} does not get worse with a slower stop on the projection path"
        )


def test_the_race_finishing_behind_the_safety_car_favours_staying_out():
    """Art. 55.17, emergent: with no racing laps left a stop cannot pay itself back.

    No rail forces this. The measured racing-lap count under a neutralisation drops
    the window to almost nothing, the fresh tyres have nowhere to earn their cost,
    and the cars queued behind become places surrendered for free.
    """
    from src.agents.strategy_orchestrator import _run_projection_mc

    rivals = [
        {"driver": "B", "interval_to_driver_s": 1.2, "is_pitting": False},
        {"driver": "C", "interval_to_driver_s": 3.0, "is_pitting": False},
    ]
    scores = _run_projection_mc(
        rivals=rivals,
        position=1,
        laps_remaining=1,
        pit_context={
            "traversal_s": 21.0,
            "mandatory_stop_pending": False,
            "neutralisation_rate": 0.0179,
            "racing_laps_neutralised": 0.0,
            "rival_stop_pending": {"B": False, "C": False},
            "rival_pit_loss_s": 23.8,
        },
        cliff_s=np.full(_DRAWS, 2.0),
        sc_s=np.full(_DRAWS, True),
        pit_s=np.full(_DRAWS, 2.8),
        ucut_s=np.zeros(_DRAWS, dtype=bool),
        alpha=0.5,
        neutralisation_saving_s=8.0,
    )
    assert _projection_argmax(scores) == "STAY_OUT"
    assert scores["STAY_OUT"]["score"] > scores["PIT_NOW"]["score"]


def test_the_legacy_path_is_taken_whenever_the_rivals_list_is_falsy():
    """None and [] both mean "no per-rival data", and both must keep the old scoring.

    Three lap_state builders default the list to ``[]``, so a None-only check would
    have routed them into a projection with no cars in it.
    """
    from src.agents.strategy_orchestrator import _run_mc_simulation
    from tests.test_mc_state_helpers import _canned_outputs

    pace, tire, situation, pit = _canned_outputs()
    baseline = _run_mc_simulation(pace, tire, situation, pit, alpha=0.5)
    for falsy in (None, [], ()):
        assert _run_mc_simulation(pace, tire, situation, pit, alpha=0.5, rivals=falsy) == baseline


def test_an_undercut_earns_nothing_under_a_neutralisation():
    """Art. 55.8 forbids overtaking under a Safety Car, so the move does not exist.

    The field is queued and everyone reaches the pit lane on the same delta, so
    arriving first buys nothing. The bonus used to be granted regardless of
    regime, which awarded roughly half a position for a manoeuvre the
    regulations prohibit.
    """
    racing = _projection_scores((-1.2, 8.0, False, 20.0, False, True, 2.4))
    neutralised = _projection_scores((-1.2, 8.0, False, 20.0, True, True, 2.4))

    assert racing["UNDERCUT"]["score"] > racing["PIT_NOW"]["score"]
    assert neutralised["UNDERCUT"]["score"] == pytest.approx(
        neutralised["PIT_NOW"]["score"], abs=1e-9
    )


def test_the_window_cannot_outlast_the_race():
    """With one lap left behind the Safety Car a stop cannot pay itself back.

    This is the Art. 55.17 endgame, and until the racing-lap count was bounded
    by the laps that actually remain, the code could not express it: the count
    was always the measured average, so a stop always appeared to have laps left
    to earn its cost over. The docs described the mechanism; the code did not
    have it.
    """
    from src.agents.strategy_orchestrator import _run_projection_mc

    rivals = [
        {"driver": "B", "interval_to_driver_s": 1.2, "is_pitting": False},
        {"driver": "C", "interval_to_driver_s": 3.0, "is_pitting": False},
    ]

    def _scores(laps_remaining: int) -> dict:
        return _run_projection_mc(
            rivals=rivals,
            position=1,
            laps_remaining=laps_remaining,
            pit_context={
                "traversal_s": 21.0,
                "mandatory_stop_pending": False,
                "neutralisation_rate": 0.0179,
                "rival_stop_pending": {"B": False, "C": False},
                "rival_pit_loss_s": 23.8,
            },
            cliff_s=np.full(_DRAWS, 2.0),
            sc_s=np.full(_DRAWS, True),
            pit_s=np.full(_DRAWS, 2.8),
            ucut_s=np.zeros(_DRAWS, dtype=bool),
            alpha=0.5,
            neutralisation_saving_s=8.0,
        )

    endgame = _scores(1)
    assert _projection_argmax(endgame) == "STAY_OUT"
    assert endgame["STAY_OUT"]["score"] > endgame["PIT_NOW"]["score"]


def test_the_racing_lap_clamp_only_fires_near_the_flag():
    """The bound belongs to the end of the race, not to every lap of it.

    Asserted on the helper rather than through a score, because with one lap
    left or thirty the stop loses the same two cars — the sub-second difference
    in fresh-tyre laps does not move a whole position, so the score cannot show
    the clamp working. Zero means unknown here, not "the race is over": several
    callers cannot supply a lap count, and clamping an unknown to zero would
    silence the window entirely.
    """
    from src.agents.strategy_orchestrator import _bounded_by_race_end

    assert _bounded_by_race_end(5.0, 30) == 5.0
    assert _bounded_by_race_end(5.0, 3) == 3.0
    assert _bounded_by_race_end(2.61, 1) == 1.0
    assert _bounded_by_race_end(2.61, 0) == 2.61


def test_the_named_target_is_the_car_we_will_be_racing_not_the_first_in_the_list():
    """#439 delivered: eligibility is ordered by post-pit-cycle proximity.

    Two cars are inside the undercut band. The one further away on the timing
    screen is the one still owing a stop, so once both cycles play out it is the
    one we come out racing. Picking the first entry of an unordered list made
    `target` a coincidence of iteration order.
    """
    from src.agents.strategy_orchestrator import _run_projection_mc

    rivals = [
        # Listed first, but it has already stopped, so it keeps gaining on us.
        {"driver": "GONE", "interval_to_driver_s": -1.0, "is_pitting": False},
        # Listed second and further away, but it still owes the stop we are
        # about to take, so after both cycles it is right next to us.
        {"driver": "RACING_US", "interval_to_driver_s": -3.5, "is_pitting": False},
    ]
    scores = _run_projection_mc(
        rivals=rivals,
        position=3,
        laps_remaining=25,
        pit_context={
            "traversal_s": 21.0,
            "mandatory_stop_pending": True,
            "neutralisation_rate": 0.0179,
            "rival_stop_pending": {"GONE": False, "RACING_US": True},
            "rival_pit_loss_s": 23.8,
        },
        cliff_s=np.full(_DRAWS, 8.0),
        sc_s=np.zeros(_DRAWS, dtype=bool),
        pit_s=np.full(_DRAWS, 2.8),
        ucut_s=np.zeros(_DRAWS, dtype=bool),
        alpha=0.5,
        neutralisation_saving_s=8.0,
    )
    assert scores["UNDERCUT"]["eligible"]
    assert scores["UNDERCUT"]["target"] == "RACING_US"
