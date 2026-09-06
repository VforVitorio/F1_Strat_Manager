"""Guard the permutation-safe flip count that the circuit_cluster leak figure rests on.

k-means cluster ids are arbitrary, so the same partition can come back with the ids
shuffled. On the real data that is not a corner case: raw label equality calls 22 of
24 circuits changed where the matched count is 5, so a regression from Hungarian
matching back to equality would inflate the leak figure in
src/strategy/eval/hygiene.py by more than four times.

These run on hand-built labellings, no parquet and no fit, so they hold on CI where
the data tree is absent.
"""

import pandas as pd
import pytest

from scripts.measure_cluster_leak import compare_labellings

CIRCUITS = ["Monaco", "Monza", "Spa", "Silverstone", "Baku", "Suzuka"]


def _labels(values: list[int]) -> pd.Series:
    return pd.Series(values, index=CIRCUITS)


def test_a_relabelled_identical_partition_counts_zero_flips():
    """The same grouping with every id renamed is not a change."""
    original = _labels([0, 0, 1, 1, 2, 2])
    permuted = _labels([2, 2, 0, 0, 1, 1])

    result = compare_labellings(original, permuted)

    assert result["flips"] == 0
    assert result["ari"] == pytest.approx(1.0)
    # Anti-vacuity: without the matching this pair looks like a total change, which
    # is the failure the Hungarian assignment exists to prevent.
    assert result["raw_flips"] == len(CIRCUITS)


def test_one_circuit_moving_counts_one_flip():
    """A single circuit changing group is one flip, not a whole relabelling."""
    original = _labels([0, 0, 1, 1, 2, 2])
    moved = _labels([0, 0, 1, 2, 2, 2])

    result = compare_labellings(original, moved)

    assert result["flips"] == 1
    assert result["moved"] == ["Silverstone"]
    assert result["ari"] < 1.0


def test_identical_labellings_count_zero_flips():
    """The trivial case, so a matching that always reports movement cannot pass."""
    labels = _labels([0, 1, 2, 0, 1, 2])

    result = compare_labellings(labels, labels)

    assert result["flips"] == 0
    assert result["raw_flips"] == 0
    assert result["moved"] == []
