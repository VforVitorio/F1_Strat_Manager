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

STAY_OUT scoring at most zero is by design: simulate_lap_window documents it as the
reference the others are scored against, so a zero ceiling is expected. What these tests
require is that it can still be the argmax; a reference that can never be chosen is not a
reference.
"""

from __future__ import annotations

import itertools
from pathlib import Path

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
    from collections import Counter

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


@pytest.mark.xfail(
    strict=True,
    reason=(
        "Known limitation, not a regression. The MC has no term for what an overcut "
        "buys, which is track position while the rival pits. Charging OVERCUT its stop "
        "(an overcut still pits) leaves it with the same cost as PIT_NOW and half the "
        "fresh-tyre laps, so it cannot strictly win; PIT_NOW is likewise a tie-alias of "
        "UNDERCUT when the undercut fails. Tracked in the MC issue. When this test "
        "starts passing, the model has gained the missing term and the xfail can go."
    ),
)
def test_no_candidate_dominates_another_by_construction():
    """If A >= B for every state, B can never strictly win and the choice is not a choice.

    `UNDERCUT` is `PIT_NOW + ucut_bonus` with `ucut_bonus >= 0`, so PIT_NOW cannot
    strictly beat it anywhere. And OVERCUT, once charged for its stop, is PIT_NOW with
    half the fresh laps. The invariant is right and the model does not meet it: that is
    worth stating out loud rather than deleting the assertion.
    """
    for a, b in itertools.permutations(STRATEGIES, 2):
        a_always_at_least_b = all(_score(a, s) >= _score(b, s) for s in _STATES)
        b_strictly_wins_somewhere = any(_score(b, s) > _score(a, s) for s in _STATES)
        assert not (a_always_at_least_b and not b_strictly_wins_somewhere), (
            f"{a} >= {b} on every one of {len(_STATES)} race states, so {b} can never "
            f"strictly win: it is dominated by construction and the choice between them "
            f"is not a choice"
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
