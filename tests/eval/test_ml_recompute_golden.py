"""Golden tests for the SC + undercut recomputes (#364).

Data-tier: the loaders rebuild the engineered features in-memory and run the
LightGBM, so they need the models + holdouts on disk and skip on CI.
"""

from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent.parent
_HAS_SC = (ROOT / "data" / "models" / "safety_car_probability" / "lgbm_sc_v1.pkl").exists()
_HAS_UNDERCUT = (ROOT / "data" / "models" / "pit_prediction" / "lgbm_undercut_v1.pkl").exists()
_HAS_PIT = (ROOT / "data" / "models" / "pit_prediction" / "hist_pit_p50_v1.pkl").exists() and bool(
    list((ROOT / "data" / "raw").glob("*/*/laps.parquet"))
)


@pytest.mark.data
@pytest.mark.skipif(not _HAS_SC, reason="SC model absent (CI runner without weights)")
def test_sc_auc_pr_reproduces_exactly():
    """SC AUC-PR reproduces its 0.0723 headline via the in-memory feature rebuild."""
    from src.strategy.eval.reproduce import _sc_auc_pr

    result = _sc_auc_pr()
    assert result.status == "reproduced"
    assert result.reproduced is not None and abs(result.reproduced - 0.0723) < 0.01


@pytest.mark.data
@pytest.mark.skipif(not _HAS_SC, reason="SC model absent (CI runner without weights)")
def test_sc_loader_supports_windows_and_years():
    """The SC loader slices any year and any of the 3 target windows (for #363)."""
    from src.strategy.eval.calibration import load_sc_predictions

    for target in ("sc_within_3_laps", "sc_within_5_laps", "sc_within_7_laps"):
        loaded = load_sc_predictions(year=2024, target=target)
        assert loaded is not None
        y, proba_raw, proba_cal = loaded
        assert len(y) == len(proba_raw) == len(proba_cal) > 0


@pytest.mark.data
@pytest.mark.skipif(not _HAS_UNDERCUT, reason="undercut model absent (CI runner without weights)")
def test_undercut_auc_pr_reproduces_exactly():
    """Undercut AUC-PR reproduces its 0.6739 headline with train-fit target encodings."""
    from src.strategy.eval.reproduce import _undercut_auc_pr

    result = _undercut_auc_pr()
    assert result.status == "reproduced"
    assert result.reproduced is not None and abs(result.reproduced - 0.6739) < 0.01


@pytest.mark.data
@pytest.mark.skipif(not _HAS_PIT, reason="pit models or raw laps absent (CI runner without data)")
def test_pit_p50_mae_reproduces_from_raw_laps():
    """The pit holdout rebuilt from raw laps reproduces the 0.487 P50 MAE within tolerance."""
    from src.strategy.eval.reproduce import _pit_p50_mae

    result = _pit_p50_mae()
    assert result.status == "reproduced"
    assert result.reproduced is not None and abs(result.reproduced - 0.487) < 0.01
