"""N06's trained range, now declared as an operating envelope (#710).

The pace agent had no range check of any kind. These pin the two things that
makes it worth adding and the one thing that would make it dangerous:

1. It LABELS. Not a single value N06 is fed may change, or the strategy goldens
   move and a real regression becomes indistinguishable from this commit.
2. It fires on a value outside the trained range, and stays silent inside it. A
   check that never speaks and a check that always speaks are equally useless.
3. The bounds are MEASURED, not typed. The data-tier test at the bottom rebuilds
   the training seasons through N06's own feature recipe and fails if any declared
   bound has drifted from the range it claims to describe.

Most of this file needs no model weights: the labelling step is a staticmethod
over a plain frame, so it runs on a bare CI checkout, which is where a silent
regression would otherwise hide.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).parent.parent.parent
_HAS_TRAINING_DATA = all(
    (ROOT / "data" / "processed" / f"laps_featured_{year}.parquet").exists()
    for year in (2023, 2024)
)


def _row(**overrides) -> pd.DataFrame:
    """A one-row frame sitting comfortably inside every declared bound."""
    from src.agents.pace_agent import _N06_TRAINED_BOUNDS

    row = {name: (lo + hi) / 2 for name, (lo, hi) in _N06_TRAINED_BOUNDS.items()}
    row.update(overrides)
    return pd.DataFrame([row])


def _label(frame: pd.DataFrame) -> None:
    from src.agents.pace_agent import PaceAgent

    PaceAgent._label_against_envelope(frame)


# --- it labels, it never touches ---------------------------------------------


def test_labelling_does_not_alter_a_single_value():
    """The whole licence for this change: the model is fed exactly what it was fed."""
    frame = _row(TyreLife=500.0, AirTemp=-40.0)
    before = frame.copy(deep=True)

    _label(frame)

    pd.testing.assert_frame_equal(frame, before)


# --- it fires outside, and only outside --------------------------------------


def test_a_value_above_the_trained_range_is_announced(caplog):
    from src.agents.pace_agent import _N06_TRAINED_BOUNDS

    _lower, upper = _N06_TRAINED_BOUNDS["TyreLife"]
    with caplog.at_level(logging.WARNING):
        _label(_row(TyreLife=upper + 1))

    assert any("outside its trained range" in r.message for r in caplog.records)
    assert any("TyreLife" in str(r.args) for r in caplog.records)


def test_a_value_below_the_trained_range_is_announced(caplog):
    """Both directions. A lower bound that is never checked is a lower bound in name only."""
    from src.agents.pace_agent import _N06_TRAINED_BOUNDS

    lower, _upper = _N06_TRAINED_BOUNDS["Prev_LapTime"]
    with caplog.at_level(logging.WARNING):
        _label(_row(Prev_LapTime=lower - 1))

    assert any("outside its trained range" in r.message for r in caplog.records)


def test_an_in_range_call_says_nothing(caplog):
    """A normal lap must be silent, or the signal is worth nothing on a 57-lap race."""
    with caplog.at_level(logging.WARNING):
        _label(_row())

    assert not [r for r in caplog.records if "trained range" in r.message]


def test_a_missing_feature_is_not_reported_as_out_of_range(caplog):
    """Unknown is not a violation, and this file must not be the place that conflates them.

    A NaN feature was given no value, not a bad one. The producers that can emit one
    already warn for themselves, so reporting it again here would bury the case this
    check exists for under the cases something else already covers.
    """
    with caplog.at_level(logging.WARNING):
        _label(_row(FuelEffect=float("nan")))

    assert not [r for r in caplog.records if "trained range" in r.message]


# --- the bounds describe features that have a range at all -------------------


@pytest.mark.parametrize("labelled", ["DriverNumber", "TeamID", "CompoundID", "Cluster", "Year"])
def test_no_bound_is_declared_over_an_identifier(labelled):
    """A code is not a quantity: `TeamID` between 0 and 10 says nothing about range."""
    from src.agents.pace_agent import _N06_TRAINED_BOUNDS

    assert labelled not in _N06_TRAINED_BOUNDS


@pytest.mark.parametrize(
    "constant", ["Prev_DegradationRate", "Prev_CumulativeDeg", "Prev_DegAcceleration"]
)
def test_the_hardcoded_degradation_features_are_deliberately_unbounded(constant):
    """`run_from_state` pins all three at 0.0, and 0.0 is INSIDE the trained range.

    This reads like the classic out-of-range defect and is not one: roughly 42-47% of
    training rows fall below zero for each, so a bound could never fire on the pinned
    value. Declaring one would ship a check that looks like coverage and is not.
    Feeding a constant where the model saw a distribution is a real problem with its
    own shape, and this envelope is not the instrument for it.
    """
    from src.agents.pace_agent import _N06_TRAINED_BOUNDS

    assert constant not in _N06_TRAINED_BOUNDS


# --- the one test that checks the bounds against the data they claim to describe ---


@pytest.mark.data
@pytest.mark.skipif(not _HAS_TRAINING_DATA, reason="laps_featured_2023/2024 absent")
def test_every_declared_bound_matches_the_measured_training_range():
    """Re-measure, do not re-read. A declared bound is a claim about the training data.

    This is the assertion that keeps `_N06_TRAINED_BOUNDS` sourced rather than typed.
    It rebuilds 2023 + 2024 exactly the way `pace_holdout.py` rebuilds the holdout
    (augment, encode, lag, drop) and compares every bound against the range that
    rebuild produces, so retraining N06 or restating a number fails here and names
    the feature instead of silently widening what the agent will vouch for.
    """
    import json

    from src.agents.pace_agent import _N06_TRAINED_BOUNDS
    from src.f1_strat_manager.data_cache import get_data_root
    from src.f1_strat_manager.laps_augment import augment_featured_laps
    from src.strategy.eval.pace_holdout import (
        _DROPNA,
        _add_lag_deg_features,
        _encode_categoricals,
    )

    root = get_data_root()
    manifest = json.loads(
        (root / "processed" / "feature_manifest_laptime.json").read_text(encoding="utf-8")
    )
    encoding = manifest["categorical_encoding"]

    frames = []
    for year in (2023, 2024):
        featured = augment_featured_laps(
            pd.read_parquet(root / "processed" / f"laps_featured_{year}.parquet"), year
        )
        encoded = _encode_categoricals(featured, encoding["Compound"], encoding["race_phase"])
        frames.append(_add_lag_deg_features(encoded).dropna(subset=_DROPNA))
    train = pd.concat(frames, ignore_index=True)

    assert len(train) > 0, "the rebuilt training frame is empty: nothing below would mean anything"

    for feature, (lower, upper) in _N06_TRAINED_BOUNDS.items():
        values = pd.to_numeric(train[feature], errors="coerce").dropna()
        assert values.size > 0, f"{feature} carried no values in the rebuilt training frame"
        assert lower == pytest.approx(values.min()), (
            f"{feature} declares a lower bound of {lower}, but N06 trained down to {values.min()}"
        )
        assert upper == pytest.approx(values.max()), (
            f"{feature} declares an upper bound of {upper}, but N06 trained up to {values.max()}"
        )
