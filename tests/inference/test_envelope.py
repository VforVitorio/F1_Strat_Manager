"""Tests for the operating-envelope contract (#709).

Pure, hermetic tests only: no ``data/models/`` weights, no agent imports. The
contract is small on purpose (one dataclass, one check function), so the
tests pin the three-way feature state (in-range / out-of-range / unknown),
the inclusive boundary, multiple simultaneous violations, and the import-time
guarantee that makes this module usable before ``data/`` even exists on disk.
"""

from __future__ import annotations

import dataclasses
import math
from pathlib import Path

import pytest

from src.strategy.inference.envelope import (
    EnvelopeVerdict,
    FeatureViolation,
    OperatingEnvelope,
)

ROOT = Path(__file__).parent.parent.parent


def _tyre_life_envelope() -> OperatingEnvelope:
    """The N15-shaped example used across most tests: one bounded feature."""
    return OperatingEnvelope(name="n15_pit_duration", bounds={"tyre_life_in": (0, 50)})


# --- in-range / out-of-range / boundary -------------------------------------


def test_value_inside_bounds_is_in_range():
    verdict = _tyre_life_envelope().check({"tyre_life_in": 25})
    assert verdict.in_range
    assert bool(verdict) is True
    assert verdict.violations == {}
    assert verdict.unknown == frozenset()


def test_value_outside_bounds_is_a_violation_with_the_offending_bound():
    verdict = _tyre_life_envelope().check({"tyre_life_in": 63})
    assert not verdict
    assert not verdict.in_range
    assert set(verdict.violations) == {"tyre_life_in"}
    violation = verdict.violations["tyre_life_in"]
    assert violation == FeatureViolation(value=63, lower=0, upper=50)


def test_lower_and_upper_bounds_are_inclusive():
    envelope = _tyre_life_envelope()
    assert envelope.check({"tyre_life_in": 0}).in_range
    assert envelope.check({"tyre_life_in": 50}).in_range


def test_just_outside_each_bound_is_a_violation():
    envelope = _tyre_life_envelope()
    assert not envelope.check({"tyre_life_in": -1}).in_range
    assert not envelope.check({"tyre_life_in": 51}).in_range


# --- the unknown state: never coerced to a number ---------------------------


def test_missing_key_is_unknown_not_a_violation():
    """A feature absent from the dict is UNKNOWN, distinct from out-of-range."""
    verdict = _tyre_life_envelope().check({})
    assert not verdict
    assert verdict.violations == {}
    assert verdict.unknown == frozenset({"tyre_life_in"})


def test_explicit_none_is_unknown():
    verdict = _tyre_life_envelope().check({"tyre_life_in": None})
    assert verdict.unknown == frozenset({"tyre_life_in"})
    assert verdict.violations == {}


def test_nan_is_unknown_not_silently_in_range():
    """NaN must never be compared numerically: NaN <= x is always False."""
    verdict = _tyre_life_envelope().check({"tyre_life_in": math.nan})
    assert verdict.unknown == frozenset({"tyre_life_in"})
    assert verdict.violations == {}


def test_unknown_alone_makes_the_verdict_falsy():
    """Unknown is not a violation, but it still means 'do not trust this call'."""
    verdict = _tyre_life_envelope().check({})
    assert verdict.violations == {}
    assert not verdict


# --- multiple features at once ----------------------------------------------


def test_multiple_simultaneous_violations_are_all_reported():
    envelope = OperatingEnvelope(
        name="two_feature_example",
        bounds={"gap_ahead_s": (0.0, 5.0), "tyre_life_x": (0, 40)},
    )
    verdict = envelope.check({"gap_ahead_s": 9.4, "tyre_life_x": -3})
    assert set(verdict.violations) == {"gap_ahead_s", "tyre_life_x"}
    assert verdict.violations["gap_ahead_s"] == FeatureViolation(value=9.4, lower=0.0, upper=5.0)
    assert verdict.violations["tyre_life_x"] == FeatureViolation(value=-3, lower=0, upper=40)


def test_mixed_violation_and_unknown_are_kept_in_their_own_buckets():
    envelope = OperatingEnvelope(
        name="mixed_example",
        bounds={"gap_ahead_s": (0.0, 5.0), "tyre_life_x": (0, 40)},
    )
    verdict = envelope.check({"gap_ahead_s": 9.4})  # tyre_life_x missing entirely
    assert set(verdict.violations) == {"gap_ahead_s"}
    assert verdict.unknown == frozenset({"tyre_life_x"})


def test_extra_undeclared_features_are_ignored():
    verdict = _tyre_life_envelope().check({"tyre_life_in": 10, "some_other_field": -999})
    assert verdict.in_range


# --- construction-time validation -------------------------------------------


def test_inverted_bound_is_rejected_at_construction():
    with pytest.raises(ValueError):
        OperatingEnvelope(name="broken", bounds={"tyre_life_in": (50, 0)})


# --- the contract is a label, and it is frozen -------------------------------


def test_envelope_and_verdict_types_are_frozen():
    envelope = _tyre_life_envelope()
    with pytest.raises(dataclasses.FrozenInstanceError):
        envelope.name = "mutated"  # type: ignore[misc]

    verdict = envelope.check({"tyre_life_in": 10})
    with pytest.raises(dataclasses.FrozenInstanceError):
        verdict.envelope_name = "mutated"  # type: ignore[misc]


def test_verdict_carries_the_envelope_name_for_logging():
    verdict = _tyre_life_envelope().check({"tyre_life_in": 10})
    assert verdict.envelope_name == "n15_pit_duration"
    assert isinstance(verdict, EnvelopeVerdict)


# --- the import surface -----------------------------------------------------


def test_importing_the_module_pulls_in_nothing_heavy(tmp_path):
    """Importing envelope.py must need neither model weights nor the agent stack.

    Mirrors ``tests/eval/test_decision_modes.py`` ::
    ``test_importing_the_module_does_not_load_the_agent_stack``. Run with cwd
    pointed at an empty temp directory (no ``data/models/`` present) so a
    hidden filesystem read at import time would fail loudly instead of
    silently succeeding because the real data happens to already be on disk
    on this machine.
    """
    import subprocess
    import sys

    assert not (tmp_path / "data" / "models").exists()

    probe = (
        "import sys; sys.path.insert(0, r'{root}'); "
        "import src.strategy.inference.envelope; "
        "heavy = [m for m in sys.modules if m.split('.')[0] in "
        "('pandas', 'numpy', 'torch', 'lightgbm', 'xgboost')]; "
        "agents = [m for m in sys.modules if m.startswith('src.agents')]; "
        "print(','.join(sorted(heavy + agents)))"
    ).format(root=str(ROOT))
    out = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        cwd=tmp_path,
        check=True,
    )
    assert out.stdout.strip() == "", f"unexpected imports: {out.stdout.strip()}"
