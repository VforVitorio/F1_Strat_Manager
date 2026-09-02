"""N25's bootstrap interval is one forward pass, and the same one (#1118).

`_bootstrap_ci` used to call `model.predict()` once per sample, 200 times a lap.
XGBoost pays a fixed pandas-to-native conversion on every call, so at n=200 that
conversion was 61% of an offline lap: 660 ms of the 1.08 s the lap loop spent in
the pace agent, measured on Lusail 2025.

Batching it is only safe if the samples are the same numbers, not merely close
ones, because the P10/P90 it returns feed the Monte Carlo layer and every
`tests/mc` golden downstream. They are: numpy computes `normal(0, sigma)` as
`sigma * standard_normal()`, and a C-order `(n, len(noise_cols))` block draws
sample-major then column, which is the order the per-sample loop drew in.

The two things that can break, and what catches each:

- the perturbation matrix stops matching the loop's, because a rewrite draws
  column-major, reuses one sigma across columns, or perturbs the base row in
  place. `GOLDEN_*` pins the exact percentiles, computed against the loop.
- the batching quietly comes back apart, because someone reintroduces a
  per-sample call. `test_the_model_is_called_once` is the only assertion here
  that would still pass on a correct loop, which is why it is separate.

The model is a stub rather than the shipped N06 booster on purpose. The property
under test is the sampling, not the regression, and a stub keeps this runnable on
a CI runner with no model weights.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.agents.pace_agent import _NOISE_PCT, PaceAgent

NOISE_COLS = (
    "Prev_LapTime",
    "Prev_SpeedST",
    "mean_sector_speed",
    "AirTemp",
    "TrackTemp",
    "TyreLife",
)

# One extra column that must never be perturbed, so a rewrite that widens the
# noise set fails here rather than in a Monte Carlo golden six files away.
COLUMNS = (*NOISE_COLS, "Stint")

BASE_ROW = {
    "Prev_LapTime": 86.4,
    "Prev_SpeedST": 291.0,
    "mean_sector_speed": 210.0,
    "AirTemp": 24.0,
    "TrackTemp": 31.0,
    "TyreLife": 12.0,
    "Stint": 2.0,
}

# Pinned against the pre-batch loop, run on this stub with seed 42 and n=200.
GOLDEN_P10 = 84.28866555452777
GOLDEN_P90 = 89.13041064831987


class RecordingModel:
    """A predict() that is deterministic, row-dependent and counts its calls.

    Row-dependent matters: a stub returning a constant would pass every
    assertion below while the perturbation matrix was silently wrong.
    """

    def __init__(self) -> None:
        self.frames: list[pd.DataFrame] = []

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        self.frames.append(frame.copy())
        return (
            frame["Prev_SpeedST"].to_numpy() * 1e-3
            + frame["TyreLife"].to_numpy() * 1e-2
            + frame["TrackTemp"].to_numpy() * 1e-4
        )


class StubAgent:
    """The only attribute `_bootstrap_ci` reads off self."""

    def __init__(self, model: RecordingModel) -> None:
        self.model = model


def feature_row(**overrides: float) -> pd.DataFrame:
    return pd.DataFrame([{**BASE_ROW, **overrides}], columns=list(COLUMNS))


def bootstrap(model: RecordingModel, frame: pd.DataFrame | None = None, **kwargs: object):
    return PaceAgent._bootstrap_ci(
        StubAgent(model), feature_row() if frame is None else frame, **kwargs
    )


def reference_loop(
    model: RecordingModel,
    frame: pd.DataFrame | None = None,
    n: int = 200,
    seed: int = 42,
) -> tuple[float, float]:
    """The implementation that shipped before the batch, kept as the oracle.

    Reading it next to the assertion is the point: the golden constants above
    mean nothing on their own, and this is where they come from.
    """
    feature_df = feature_row() if frame is None else frame
    rng = np.random.default_rng(seed)
    base = feature_df.values.copy().astype(float)
    col_idx = {c: feature_df.columns.get_loc(c) for c in NOISE_COLS}
    preds = []
    for _ in range(n):
        row = base.copy()
        for _col, idx in col_idx.items():
            sigma = abs(base[0, idx]) * _NOISE_PCT
            row[0, idx] += rng.normal(0, sigma)
        df_row = pd.DataFrame(row, columns=feature_df.columns)
        preds.append(float(df_row["Prev_LapTime"].iloc[0]) + float(model.predict(df_row)[0]))
    return float(np.percentile(preds, 10)), float(np.percentile(preds, 90))


def test_the_batch_reproduces_the_loop_exactly() -> None:
    """Bit-identical, not approximately equal.

    `pytest.approx` would hide precisely the failure this exists to catch: a
    perturbation matrix drawn in a different order is still statistically the
    same distribution, and would land within any tolerance worth writing.
    """
    assert bootstrap(RecordingModel()) == reference_loop(RecordingModel())


def test_the_percentiles_are_pinned() -> None:
    """So a rewrite of both sides at once cannot agree its way past the test."""
    assert bootstrap(RecordingModel()) == (GOLDEN_P10, GOLDEN_P90)


def test_the_model_is_called_once() -> None:
    """The efficiency property, which the equality assertions do not cover.

    A restored per-sample loop returns the same numbers and fails only here.
    """
    model = RecordingModel()
    bootstrap(model)
    assert len(model.frames) == 1


def test_every_sample_reaches_the_model() -> None:
    """One call, n rows: batching must not have dropped or deduplicated any."""
    model = RecordingModel()
    bootstrap(model, n=200)
    assert len(model.frames[0]) == 200


def test_n_is_honoured() -> None:
    """A hardcoded 200 inside the batch would pass every test above."""
    model = RecordingModel()
    bootstrap(model, n=37)
    assert len(model.frames[0]) == 37


def test_the_columns_survive_in_order() -> None:
    """The frame handed to the booster keeps the training column order.

    XGBoost matches features by position when handed a bare frame, so a
    reordered block would score every sample against the wrong feature and
    still return a plausible number.
    """
    model = RecordingModel()
    bootstrap(model)
    assert list(model.frames[0].columns) == list(COLUMNS)


def test_only_the_noise_columns_move() -> None:
    """Stint is in the frame and must arrive untouched on all 200 rows."""
    model = RecordingModel()
    bootstrap(model)
    assert (model.frames[0]["Stint"].to_numpy() == BASE_ROW["Stint"]).all()


@pytest.mark.parametrize("column", NOISE_COLS)
def test_each_noise_column_actually_moves(column: str) -> None:
    """The mirror of the test above: a column dropped from the noise set is a
    narrower interval, which reads as a more confident agent rather than a bug.
    """
    model = RecordingModel()
    bootstrap(model)
    assert model.frames[0][column].nunique() > 1


def test_a_zero_valued_feature_still_consumes_its_draw() -> None:
    """sigma == 0 yields no perturbation but must not skip the sample.

    The loop called `rng.normal(0, 0)`, which returns 0.0 and still advances the
    stream, so an optimisation that skips zero-sigma columns shifts every later
    draw. It is invisible on a normal row, which is why the comparison runs on a
    row that has one, and why asserting that the column stayed 0.0 is not enough:
    it stays 0.0 under the skip too. Only the interval moves.
    """
    zeroed = feature_row(AirTemp=0.0)
    assert bootstrap(RecordingModel(), zeroed) == reference_loop(RecordingModel(), zeroed)
    model = RecordingModel()
    bootstrap(model, zeroed)
    assert (model.frames[0]["AirTemp"].to_numpy() == 0.0).all()


def test_the_seed_decides_the_answer() -> None:
    """Reproducibility, and that the seed argument is still wired through."""
    assert bootstrap(RecordingModel(), seed=1) == bootstrap(RecordingModel(), seed=1)
    assert bootstrap(RecordingModel(), seed=1) != bootstrap(RecordingModel(), seed=2)
