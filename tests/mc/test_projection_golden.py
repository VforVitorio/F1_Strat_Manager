"""A frozen golden for the PROJECTION branch — the one every real surface takes.

``test_strategy_goldens.py`` pins ``simulate_lap_window``, the legacy seconds
path, and says so: *"This IS the thesis-defended math."* It reaches that path by
calling ``_run_mc_simulation`` with **no** ``rivals`` kwarg, so
``_has_usable_gaps(None)`` is False and the dispatch at
``strategy_orchestrator.py:1396`` routes away from the projection entirely.

The consequence went unnoticed for as long as the branch has existed: **the
projection path had no golden at all.** It was guarded only by structural
invariants with wide tolerance bands and by unit asserts on the primitive. An
edit that broke its VALUES failed loudly on the legacy side and silently on this
one — which is exactly the asymmetry that let a distribution collapse into a
point mass without a single test noticing.

WHAT THIS GOLDEN FREEZES, AND WHAT IT ALREADY CAUGHT
-----------------------------------------------------
It was created (#725) pinning the defect on purpose: ``STAY_OUT`` was a strict
point mass, ``E == P10 == P90 == 1.3``. That was the whole design — a golden
recording what we wished were true would prove nothing, while one recording the
defect turns the next fix into a visible diff.

It worked. Race-end residual netting (#726) moved ``STAY_OUT``'s expected value
off the point mass to 1.276 and lifted PIT_NOW from 0.582 to 0.678, and the
tripwire below fired rather than the change slipping through.

``P10`` and ``P90`` are still equal for STAY_OUT here, and that is honest rather
than a leftover: in THIS geometry only a minority of draws see the correction,
so the middle of the distribution does not move even though its mean does. The
remaining flatness belongs to the tyre channel, which is #727's subject: the
layer's only tyre signal is still a cliff that almost never falls inside the
five-lap window.

WHERE TO CHANGE IF THE PROJECTION CHANGES:
- Regenerate by calling the same function with the same arguments and pasting
  the result. Do NOT hand-edit an entry to make a test pass; the whole value of
  this file is that it moves only when someone decided it should.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.conftest import HAS_TIRE_MODELS as _HAS_MODELS
from tests.mc.canned_outputs import canned_outputs as _canned_outputs

ROOT = Path(__file__).parent.parent.parent

# The rival geometry is chosen, not arbitrary, and the roles are the opposite of
# the intuitive reading — measured, not assumed:
#
#   AHEAD  (-2.4 s)  a live undercut target, which is why UNDERCUT names it.
#   BEHIND ( 4.6 s)  charged by the terminal liability on 100% of draws. Being
#                    charged CONSTANTLY it contributes no spread at all; it pins
#                    the level, not the distribution.
#   FAR    (22.6 s)  sits inside the total pit-loss support and is crossed on
#                    exactly 50% of draws. THIS is what gives PIT_NOW and
#                    UNDERCUT their P10/P90 spread.
#
# So the pit-cycle-behind car is not here "so the liability has something to
# charge" — that is BEHIND's job, and it is a constant. Stating it the wrong way
# round would teach a false mechanism to whoever edits this next.
#
# Far-field control, executed: move BEHIND to 40 s and FAR to 60 s and STAY_OUT
# and PIT_NOW both collapse to point masses (2.3/2.3/2.3). UNDERCUT does not,
# because the N16 bonus is a Bernoulli draw that spreads regardless of geometry
# and is not part of the projection. So the geometry is load-bearing for the
# projection channels specifically, which is the claim worth making.
_RIVALS = [
    {"driver": "AHEAD", "interval_to_driver_s": -2.4, "is_pitting": False},
    {"driver": "BEHIND", "interval_to_driver_s": 4.6, "is_pitting": False},
    {"driver": "FAR", "interval_to_driver_s": 22.6, "is_pitting": False},
]

# AHEAD still owes its stop, and that is the point of the entry: the race-end
# netting only does anything when at least one rival carries an outstanding
# obligation. An all-settled map would leave the branch that changed in #726
# unpinned by the only golden that covers this path.
_PIT_CONTEXT = {
    "gp_name": None,
    "traversal_s": 21.0,
    "mandatory_stop_pending": True,
    "rival_stop_pending": {"AHEAD": True, "BEHIND": False, "FAR": False},
    "rival_pit_loss_s": 23.8,
}

_GOLDEN_PROJECTION_ALPHA_05 = {
    # STAY_OUT moved when the 2023 Spanish GP duplicate left the dataset: E 1.276 -> 1.28,
    # score 1.288 -> 1.29. The other three candidates are identical to the digit. The race
    # was in the featured files and the raw tree twice, so the per-circuit and per-team
    # aggregates the projection reads were computed over one weekend counted double; this
    # is the same correction that moved the published sample from 1,810 stops to 1,768.
    "STAY_OUT": {
        "E": 1.28,
        "P10": 1.3,
        "P90": 1.3,
        "score": 1.29,
        "eligible": True,
        "target": None,
    },
    "PIT_NOW": {
        "E": 0.678,
        "P10": 0.0,
        "P90": 1.056,
        "score": 0.339,
        "eligible": True,
        "target": None,
    },
    "UNDERCUT": {
        "E": 1.21,
        "P10": 0.0,
        "P90": 2.049,
        "score": 0.605,
        "eligible": True,
        "target": "AHEAD",
    },
    "OVERCUT": {
        "E": None,
        "P10": None,
        "P90": None,
        "score": None,
        "eligible": False,
        "target": None,
    },
}


def _projection_scores(alpha: float = 0.5) -> dict:
    """Score the frozen state through the real production entry point."""
    from src.agents.strategy_orchestrator import _run_mc_simulation

    pace, tire, situation, pit = _canned_outputs()
    return _run_mc_simulation(
        pace,
        tire,
        situation,
        pit,
        alpha=alpha,
        rivals=_RIVALS,
        position=4,
        laps_remaining=22,
        pit_context=_PIT_CONTEXT,
    )


# ---------------------------------------------------------------------------
# Shape — runs everywhere, including a CI runner with no model weights
# ---------------------------------------------------------------------------


def test_the_golden_carries_the_projection_only_keys():
    """The projection returns two keys the legacy branch never does.

    Asserted on the frozen dict rather than on a live call so it survives
    without ``data/models/``. Both MC test files are gated on model weights, so
    without this the entire projection branch would again have zero coverage on
    CI — the precise gap this file exists to close.
    """
    assert set(_GOLDEN_PROJECTION_ALPHA_05) == {"STAY_OUT", "PIT_NOW", "UNDERCUT", "OVERCUT"}
    for cell in _GOLDEN_PROJECTION_ALPHA_05.values():
        assert set(cell) == {"E", "P10", "P90", "score", "eligible", "target"}


def test_an_ineligible_candidate_carries_no_numbers_at_all():
    """``None`` everywhere, never 0.0 — a zero score is a real, findable value."""
    overcut = _GOLDEN_PROJECTION_ALPHA_05["OVERCUT"]
    assert overcut["eligible"] is False
    assert all(overcut[key] is None for key in ("E", "P10", "P90", "score"))


def test_the_frozen_state_records_how_far_the_collapse_has_been_undone():
    """A tripwire on the defect, updated once it partly fired.

    It began as ``E == P10 == P90``, the strict point mass #726 set out to
    break. Netting broke it: the mean now differs from the quantiles. The
    quantiles themselves are still equal, because in this geometry the
    correction reaches only a minority of draws — that residual flatness is the
    tyre channel, and it belongs to #727.

    Kept as a tripwire rather than deleted, so the NEXT change to the collapse
    also has to arrive as a deliberate golden update rather than as a surprise.
    """
    stay_out = _GOLDEN_PROJECTION_ALPHA_05["STAY_OUT"]
    assert stay_out["E"] != stay_out["P10"], "the netting's effect on the mean has vanished"
    assert stay_out["P10"] == stay_out["P90"], (
        "the quantile band has moved: the tyre channel (#727) is the expected cause, "
        "and it should arrive as a deliberate update here"
    )


# ---------------------------------------------------------------------------
# Values — needs the model weights the canned outputs are shaped against
# ---------------------------------------------------------------------------

pytestmark_values = pytest.mark.skipif(
    not _HAS_MODELS,
    reason="data/models/ not present (CI runner without model weights)",
)


@pytestmark_values
def test_projection_scores_match_the_frozen_golden():
    assert _projection_scores() == _GOLDEN_PROJECTION_ALPHA_05


@pytestmark_values
def test_the_projection_is_deterministic_across_calls():
    assert _projection_scores() == _projection_scores()


@pytestmark_values
@pytest.mark.parametrize(("alpha", "key"), [(1.0, "E"), (0.0, "P10")])
def test_alpha_collapses_the_score_onto_a_single_quantile(alpha, key):
    """Alpha 1 is pure expected value, alpha 0 is pure worst case."""
    for name, cell in _projection_scores(alpha=alpha).items():
        if cell["score"] is None:
            continue
        assert cell["score"] == pytest.approx(cell[key]), name
