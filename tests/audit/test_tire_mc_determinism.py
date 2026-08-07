"""#735 — the tyre model's MC Dropout must give the same answer twice.

N26 samples MC Dropout at inference (``model.train()`` kept on for ``n_mc``
forward passes) and never seeded it, while its evaluation twin,
``src/strategy/eval/tire_holdout.py::mc_dropout_global_sigma``, does. Same
operation, near-identical comment, one seeded and one not — this repo's
dominant defect class in its purest form.

WHY IT SURVIVED TWO YEARS
-------------------------
The quantiles are rounded to one decimal before they leave the tool, which hides
most of the wobble. ``laps_to_cliff_p90`` was measured moving between 3.00 and
3.10 across five identical runs of the same lap; a downstream Monte Carlo score
moved by 0.493 on one lap between two identical captures. Small enough to look
like nothing, large enough to change a published census: the same command at the
same commit reported the argmax as STAY_OUT on 53/57 laps once and 52/57 the
next time.

WHY IT MATTERS MORE THAN THE WOBBLE
-----------------------------------
Everything downstream is built for determinism — the strategy Monte Carlo seeds
its own RNG at 42, shares draws across candidates, and is pinned by
byte-identical goldens. A stochastic step upstream of all that means two
identical race states can produce two different recommendations, and it means no
measurement of the decision layer has a knowable error bar.

WHAT THIS TEST IS, EXACTLY
--------------------------
The assertion is repeatability at the PRODUCTION seam, not a frozen value. It
calls ``_tire_no_llm`` — the deterministic path ``engine.py`` runs, which injects
a null ReAct runner and then goes through the agent's own ``run_from_state`` — so
it exercises the real tool, the real TCN, and the real parser rather than a
re-implementation of them. Pinning a NUMBER here would fail on any legitimate
model or data refresh while saying nothing about the property that broke.
"""

from __future__ import annotations

from itertools import islice
from pathlib import Path

import pandas as pd
import pytest

from tests.conftest import HAS_TIRE_MODELS as _HAS_MODELS

ROOT = Path(__file__).parent.parent.parent
RACE_DIR = ROOT / "data" / "raw" / "2025" / "Lusail"
FEATURED = ROOT / "data" / "processed" / "laps_featured_2025.parquet"
_HAS_DATA = (RACE_DIR / "laps.parquet").exists() and FEATURED.exists()

pytestmark = pytest.mark.skipif(
    not (_HAS_MODELS and _HAS_DATA),
    reason="needs data/models/tire_degradation/ and data/{raw/2025/Lusail,processed} (HF, not git)",
)


# Lusail 2025, NOR. A WORN tyre 12 laps into its stint, and both halves matter.
#
# Mid-stint, because the first laps of a stint are where the degradation rate's own floor
# fires and a lap chosen there could be repeatable for a reason that has nothing to do with
# the dropout seed.
#
# WORN, because the projection saturates at the race distance on a fresh one. This fixture
# used to sit on lap 30, five laps into a stint, and the fix for the tyre serving frame
# (#816) changed that lap's answer from `deg_rate -0.184, cliff 12.8/13.6/14.4` to
# `+0.0231, cliff 57/57/57` — the new value being the plausible one, since a sustained
# -0.18 s/lap after fuel correction does not happen. But a cliff pinned at the race end has
# no spread left to observe, so six unseeded runs came out identical and the mutation check
# below could no longer fail. It was guarding nothing, exactly as its own message said.
#
# Measured on this lap: p90 20.4 laps against a 57-lap race, and four of four unseeded runs
# distinct. Barcelona lap 12 and Monza lap 14 behave the same way if this one ever goes flat.
_DEMONSTRATOR_LAP = 12


@pytest.fixture(scope="module")
def lusail_worn_tyre():
    """A real lap_state plus the featured frame the agents are fed, Lusail 2025.

    Scoped to the GP the way ``engine.py`` scopes it (#429) before any agent sees
    the frame. The first version of this fixture passed the season-wide parquet
    straight in — the assertions held either way, because determinism does not
    care, but it was measuring a configuration nothing runs. Unscoped, the lap it
    then used reported a cumulative degradation of -33.840 s/lap against a real
    range of roughly -0.9 to +0.5.
    """
    from src.f1_strat_manager.laps_augment import augment_featured_laps
    from src.simulation.replay_engine import RaceReplayEngine
    from src.strategy.inference.engine import _scope_laps_to_gp

    replay = RaceReplayEngine(RACE_DIR, driver_code="NOR", team="McLaren", interval_seconds=0.0)
    lap_state = next(islice(replay.replay(), _DEMONSTRATOR_LAP - 1, _DEMONSTRATOR_LAP))
    laps = augment_featured_laps(pd.read_parquet(FEATURED), 2025)
    return lap_state, _scope_laps_to_gp(laps, lap_state)


def test_two_identical_calls_return_an_identical_tyre_output(lusail_worn_tyre):
    """The whole dataclass, not a chosen field — the wobble moved p90, not deg_rate."""
    from src.strategy.inference.no_llm import _tire_no_llm

    lap_state, laps = lusail_worn_tyre

    first = _tire_no_llm(lap_state, laps)
    second = _tire_no_llm(lap_state, laps)

    assert first == second


def test_the_seed_is_what_makes_it_repeatable(lusail_worn_tyre, monkeypatch):
    """Mutation check: with the seed removed, this test must be able to fail.

    Without it the assertion above could pass for an unrelated reason — a model
    with dropout disabled, a cached tool result, a stint too short to sample —
    and a test that cannot fail is exactly the defect it was written to catch.

    ``torch.manual_seed`` is neutralised rather than the config changed, because
    a config with a ``None`` seed is a state production cannot reach; the thing
    worth proving is that the SEEDING is load-bearing.
    """
    import torch

    from src.strategy.inference.no_llm import _tire_no_llm

    lap_state, laps = lusail_worn_tyre
    monkeypatch.setattr(torch, "manual_seed", lambda _seed: None)

    outputs = [_tire_no_llm(lap_state, laps) for _ in range(6)]

    assert any(o != outputs[0] for o in outputs[1:]), (
        "unseeded MC Dropout produced six identical outputs, so this lap cannot "
        "demonstrate the property. The usual cause is a SATURATED projection: on a fresh "
        "tyre the cliff pins to the race distance and the quantiles have no spread left to "
        "show. Pick a lap with a worn tyre whose p90 sits inside the race — Barcelona 12 "
        "and Monza 14 both work — and update _DEMONSTRATOR_LAP with the measurement"
    )
