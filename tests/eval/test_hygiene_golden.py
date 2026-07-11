"""Golden tests for the #207 threshold-provenance / leakage report.

The verdict assertions are hermetic (the audit findings are authored data).
The overtake threshold-correction test is data-tier (it re-runs the model on
the 2024/2025 holdout) and is skipped on CI without weights.
"""

from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent.parent
_HAS_MODELS = (ROOT / "data" / "models" / "overtake_probability" / "model_config.json").exists()


def test_hygiene_verdicts_match_the_audit():
    """Both contaminated thresholds, undercut clean, aggregates clean bar circuit_cluster."""
    from src.strategy.eval.hygiene import CLEAN, CONTAMINATED, UNDERDOCUMENTED, audit_findings

    findings = audit_findings()
    thresholds = {f.model: f for f in findings if f.kind == "threshold"}

    assert thresholds["overtake"].verdict == CONTAMINATED
    assert thresholds["safety_car"].verdict == CONTAMINATED
    assert thresholds["undercut"].verdict == CLEAN

    aggregates = [f for f in findings if f.kind == "aggregate_feature"]
    cluster = [f for f in aggregates if "circuit_cluster" in f.item]
    assert cluster and cluster[0].verdict == UNDERDOCUMENTED
    assert all(f.verdict == CLEAN for f in aggregates if "circuit_cluster" not in f.item)


@pytest.mark.data
@pytest.mark.skipif(not _HAS_MODELS, reason="data/models/ absent (CI runner without weights)")
def test_overtake_threshold_correction_is_honest():
    """Re-selecting on val-2024 yields a threshold whose test F1 <= the leaked (test-fit) one."""
    from src.strategy.eval.hygiene import correct_overtake_threshold

    correction = correct_overtake_threshold()
    assert correction is not None
    assert 0.0 < correction["corrected_threshold"] < 1.0
    leaked_f1 = correction["leaked_test_operating_point"]["f1"]
    corrected_f1 = correction["corrected_test_operating_point"]["f1"]
    assert corrected_f1 <= leaked_f1 + 1e-9, (
        "leaked threshold was fit on test, so its F1 is maximal there"
    )


@pytest.mark.data
@pytest.mark.skipif(
    not (ROOT / "data" / "models" / "safety_car_probability" / "lgbm_sc_v1.pkl").exists(),
    reason="SC model absent (CI runner without weights)",
)
def test_sc_threshold_correction_does_not_beat_test_fit():
    """The val-selected SC F2 cannot exceed the leaked (test-fit) F2 on test."""
    from src.strategy.eval.hygiene import correct_sc_threshold

    correction = correct_sc_threshold()
    assert correction is not None
    assert correction["val_positive_count"] >= 0
    leaked_f2 = correction["leaked_test_operating_point"]["f2"]
    corrected_f2 = correction["corrected_test_operating_point"]["f2"]
    assert corrected_f2 <= leaked_f2 + 1e-9


@pytest.mark.data
@pytest.mark.skipif(
    not (ROOT / "data" / "models" / "safety_car_probability" / "lgbm_sc_v1.pkl").exists(),
    reason="SC model absent (CI runner without weights)",
)
def test_sc_window_is_not_retro_selectable():
    """Only the 3-lap SC model is persisted, so the window cannot be re-selected."""
    from src.strategy.eval.hygiene import sc_window_sensitivity

    window = sc_window_sensitivity()
    assert window is not None
    assert window["retro_selectable"] is False
    assert set(window["per_window_auc_pr"]) == {
        "sc_within_3_laps",
        "sc_within_5_laps",
        "sc_within_7_laps",
    }
