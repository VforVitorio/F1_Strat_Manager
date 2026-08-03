"""Tests for the guard-rail stint-length evidence tier (#708 family).

Two layers, the same split ``test_decision_modes.py`` and
``test_position_projection.py`` use:

1. Pure tests that pin the contract: percentile maths, the strict "shorter than
   threshold" comparison, the wet/dry/unknown compound split, and that the
   thresholds are imported from the guard rail rather than retyped. These run
   everywhere, no dataset required.

2. One data-tier test that refuses to believe a sample it has not counted. It
   needs ``data/raw`` and skips without it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.strategy.eval.stint_lengths import (
    _MIN_STINT_LAPS,
    _PERCENTILE_POINTS,
    CompoundStints,
    StintLengthSample,
    _bound_for,
    _compound_bucket,
    _render_table,
)
from src.strategy.inference.guard_rails import _CALIBRATION_CEILING
from src.strategy.inference.guard_rails import _MIN_STINT_LAPS as RAIL_MIN_STINT_LAPS

ROOT = Path(__file__).parent.parent.parent
_HAS_RAW = (ROOT / "data" / "raw" / "2025").is_dir()


def _stints(compound: str, lengths: list[float], threshold: int) -> CompoundStints:
    return CompoundStints(
        compound=compound, lengths=np.array(lengths, dtype=float), threshold=threshold
    )


# --- percentile maths --------------------------------------------------------


def test_summary_matches_numpy_percentile_directly():
    """The report's own percentile points must agree with numpy, point for point."""
    lengths = [3, 5, 8, 9, 11, 14, 15, 18, 22, 30]
    stats = _stints("MEDIUM", lengths, threshold=12)
    summary = stats.summary()
    for label, q in _PERCENTILE_POINTS:
        assert summary[label] == pytest.approx(float(np.percentile(lengths, q))), label


def test_min_and_max_are_the_extremes_of_the_sample():
    stats = _stints("HARD", [15, 20, 40, 9], threshold=15)
    summary = stats.summary()
    assert summary["min"] == 9.0
    assert summary["max"] == 40.0


def test_percentile_over_an_empty_sample_is_zero_not_nan():
    """An empty compound must read as 'no data' via sample_size, never as NaN."""
    stats = _stints("SOFT", [], threshold=8)
    assert stats.sample_size == 0
    for value in stats.summary().values():
        assert value == 0.0
        assert not np.isnan(value)


# --- the headline: share strictly shorter than the guard rail's bound -------


def test_share_below_threshold_is_strict_not_inclusive():
    """A stint exactly on the boundary is one the rail would allow, not block."""
    stats = _stints("SOFT", [5, 8, 8, 10, 15], threshold=8)
    # Only the single 5-lap stint is strictly shorter than 8.
    assert stats.share_below_threshold == pytest.approx(0.2)


def test_share_below_threshold_is_zero_when_nothing_runs_that_short():
    stats = _stints("HARD", [15, 20, 25, 30], threshold=15)
    assert stats.share_below_threshold == 0.0


def test_share_below_threshold_is_one_when_every_stint_undercuts_the_rail():
    stats = _stints("MEDIUM", [3, 4, 5], threshold=12)
    assert stats.share_below_threshold == 1.0


def test_share_below_threshold_over_an_empty_sample_is_zero():
    stats = _stints("SOFT", [], threshold=8)
    assert stats.share_below_threshold == 0.0


# --- classifying a raw Compound reading --------------------------------------


@pytest.mark.parametrize(
    ("compound", "expected"),
    [
        ("SOFT", "SOFT"),
        ("MEDIUM", "MEDIUM"),
        ("HARD", "HARD"),
        ("INTERMEDIATE", "WET"),
        ("WET", "WET"),
        ("UNKNOWN_TYRE", "unknown"),
        ("", "unknown"),
    ],
)
def test_compound_bucket_classifies_dry_wet_and_unknown(compound, expected):
    """The wet/dry split is pure and dataframe-free: no parquet needed to check it."""
    assert _compound_bucket(compound) == expected


# --- the thresholds must come from the guard rail, never be retyped ---------


def test_thresholds_are_imported_from_the_guard_rail_not_restated():
    """A retyped boundary has shipped wrong in this codebase before (#708).

    Asserting identity, not just equal values, is what actually catches a
    future edit that hardcodes ``{"SOFT": 8, ...}`` here instead of importing it:
    two equal-but-separate dicts would pass an equality check and still drift the
    day only one of them is edited.
    """
    assert _MIN_STINT_LAPS is RAIL_MIN_STINT_LAPS


def test_every_dry_compound_has_a_rail_threshold():
    for compound in ("SOFT", "MEDIUM", "HARD"):
        assert compound in _MIN_STINT_LAPS


# --- rendering ----------------------------------------------------------------


def _sample(
    by_compound: dict[str, CompoundStints], dropped_missing=0, races=1
) -> StintLengthSample:
    return StintLengthSample(
        by_compound=by_compound,
        dropped_missing=dropped_missing,
        races=races,
    )


def test_render_without_data_says_so_instead_of_printing_zeros():
    body = _render_table(None)
    assert "Not measured" in body
    assert "0.0%" not in body


def test_render_reports_the_headline_share_and_the_drop_counts():
    sample = _sample(
        {
            "SOFT": _stints("SOFT", [5, 8, 10, 20], threshold=8),
            "MEDIUM": _stints("MEDIUM", [10, 12, 20], threshold=12),
            "HARD": _stints("HARD", [15, 18, 22], threshold=15),
            "WET": _stints("WET", [11, 14], threshold=10),
        },
        dropped_missing=2,
        races=3,
    )
    body = _render_table(sample)

    assert "SOFT" in body and "MEDIUM" in body and "HARD" in body
    # One of four SOFT stints (the 5-lap one) undercuts the 8-lap rail: 25.0%.
    assert "25.0%" in body
    assert "dropped" in body
    assert "2" in body


def test_render_states_the_field_it_measures():
    """The 'not race laps, TyreLife' choice is part of the artifact, not a comment."""
    sample = _sample(
        {
            "SOFT": _stints("SOFT", [10], threshold=8),
            "MEDIUM": _stints("MEDIUM", [], threshold=12),
            "HARD": _stints("HARD", [], threshold=15),
            "WET": _stints("WET", [], threshold=10),
        }
    )
    body = _render_table(sample)
    assert "TyreLife" in body
    assert "PitInTime" in body


# --- the one test that checks against the world -----------------------------


@pytest.mark.data
@pytest.mark.skipif(not _HAS_RAW, reason="data/raw absent (CI runner without the dataset)")
def test_measured_sample_is_non_empty_before_any_figure_is_believed():
    """Guard against a green run that quietly counted zero real stints.

    A report iterating a DISCOVERED set can pass every assertion about the empty
    set. This asserts the set exists first, the same guard
    ``test_decision_modes.py`` keeps for its own discovered sample.
    """
    from src.strategy.eval.stint_lengths import measure_stint_lengths

    sample = measure_stint_lengths(years=(2025,))

    assert sample.races > 0, "no races found under data/raw/2025"
    assert sample.total_counted > 0, "no real green-flag stints were counted: sample is empty"
    for bucket, stats in sample.by_compound.items():
        assert stats.threshold == _bound_for(bucket)
        # Not every compound need appear at every circuit, but at least one must,
        # or this "distribution by compound" report has nothing to report.
    assert any(stats.sample_size > 0 for stats in sample.by_compound.values())


@pytest.mark.data
@pytest.mark.skipif(not _HAS_RAW, reason="data/raw absent (CI runner without the dataset)")
def test_no_minimum_stint_bound_vetoes_more_than_the_calibration_ceiling():
    """Every minimum-stint bound must still sit where real strategy essentially never goes.

    This is the test #716 exists to leave behind, and it deliberately asserts the
    EFFECT rather than the values. A test reading ``_MIN_STINT_LAPS["SOFT"] == 2``
    would pass forever while saying nothing: the repo has shipped exactly that
    before, a green assertion pinning a threshold that no longer did anything (see
    #450, where a wired-in bound was compared against the wrong probability scale
    and its test asserted the constant rather than a single firing).

    What actually went wrong here is what this catches: SOFT 8 / MEDIUM 12 / HARD 15
    vetoed 15.5% / 17.0% / 12.2% of real green-flag stops, and the wet fallback 20.0%
    -- one real stop in six overridden by a bound whose entire licence to exist is
    that real strategy does not go there. Raise any bound back above the ceiling and
    this fails naming the compound and the share.

    Measured over all three seasons rather than 2025 alone, because that is the
    sample the bounds were set from; a single season would grade them against
    evidence they were not calibrated on.
    """
    from src.strategy.eval.stint_lengths import measure_stint_lengths

    sample = measure_stint_lengths()

    graded = {
        bucket: stats for bucket, stats in sample.by_compound.items() if stats.sample_size > 0
    }
    assert graded, "no bucket carried a sample: the ceiling would hold vacuously"

    for bucket, stats in graded.items():
        assert stats.share_below_threshold <= _CALIBRATION_CEILING, (
            f"the {bucket} minimum-stint bound of {stats.threshold} laps vetoes "
            f"{stats.share_below_threshold:.1%} of {stats.sample_size} real green-flag "
            f"stops, above the {_CALIBRATION_CEILING:.0%} ceiling: it is separating "
            f"unusual from usual, not absurd from sane"
        )
