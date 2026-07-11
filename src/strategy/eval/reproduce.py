"""Headline-metric reproduction check (the ``f1-eval models`` deliverable).

The registry consolidates the published numbers; this module re-derives them
from the frozen model + holdout and reports reproduced-vs-config deltas. Where
the on-disk data does not support a clean re-derivation, it documents the delta
explicitly rather than inventing a number - which is the deliverable's own
"reproduces every headline number within tolerance, or documents the delta".

Reproduced on-disk: overtake / safety_car / undercut AUC-PR (in-memory feature
rebuild), pit P50 MAE (raw-laps regen), pace MAE (featured-laps rebuild of the
N06 delta model), and tire MAE (TCN forward pass over the rebuilt 2025
sequences). Every headline number now re-derives from the frozen artifacts.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any

from src.f1_strat_manager.data_cache import get_models_root
from src.strategy.eval.calibration import (
    load_overtake_predictions,
    load_sc_predictions,
    load_undercut_predictions,
)
from src.strategy.eval.report import build_header, write_report

REPRO_NAME = "reproduction"
TOLERANCE = 0.01  # |reproduced - published| within this counts as reproduced


@dataclass
class ReproResult:
    """One headline-metric reproduction check.

    ``status`` is ``reproduced`` (within tolerance of the published value),
    ``delta`` (re-derived but diverges), or ``pending`` (cannot re-derive
    on-disk this phase).
    """

    model: str
    metric: str
    published: float
    reproduced: float | None
    status: str
    detail: str


def _overtake_auc_pr() -> ReproResult:
    """Re-derive overtake AUC-PR on the 2025 holdout and compare to the config.

    AUC-PR is threshold-free, so it is computed on the raw model scores (the
    same quantity N12 reported). ``sklearn.average_precision_score`` is the
    AUC-PR estimator.
    """
    cfg = json.loads(
        (get_models_root() / "overtake_probability" / "model_config.json").read_text(
            encoding="utf-8"
        )
    )
    published = float(cfg["auc_pr_test"])

    loaded = load_overtake_predictions()
    if loaded is None:
        return ReproResult(
            "overtake",
            "auc_pr_test",
            published,
            None,
            "pending",
            "holdout or artifacts absent on disk",
        )

    from sklearn.metrics import average_precision_score

    y, proba_raw, _ = loaded
    reproduced = float(average_precision_score(y, proba_raw))
    delta = abs(reproduced - published)
    status = "reproduced" if delta <= TOLERANCE else "delta"
    return ReproResult(
        "overtake",
        "auc_pr_test",
        published,
        reproduced,
        status,
        f"|delta| {delta:.4f} vs tol {TOLERANCE}",
    )


def _classifier_auc_pr(model: str, published: float, loader: "Any") -> ReproResult:
    """Re-derive a classifier's AUC-PR on the 2025 holdout via a shared loader.

    AUC-PR is threshold-free, so it is computed on the raw model scores. Generic
    over safety_car / undercut, mirroring ``_overtake_auc_pr``.
    """
    loaded = loader()
    if loaded is None:
        return ReproResult(
            model, "auc_pr_test", published, None, "pending", "holdout or artifacts absent on disk"
        )

    from sklearn.metrics import average_precision_score

    y, proba_raw, _ = loaded
    reproduced = float(average_precision_score(y, proba_raw))
    delta = abs(reproduced - published)
    status = "reproduced" if delta <= TOLERANCE else "delta"
    return ReproResult(
        model,
        "auc_pr_test",
        published,
        reproduced,
        status,
        f"|delta| {delta:.4f} vs tol {TOLERANCE}",
    )


def _sc_auc_pr() -> ReproResult:
    """SC AUC-PR reproduction against its published test number (feature_list_v1)."""
    cfg = json.loads(
        (get_models_root() / "safety_car_probability" / "feature_list_v1.json").read_text(
            encoding="utf-8"
        )
    )
    return _classifier_auc_pr(
        "safety_car", float(cfg["metrics"]["test_auc_pr"]), load_sc_predictions
    )


def _undercut_auc_pr() -> ReproResult:
    """Undercut AUC-PR reproduction against its published test number (model_config)."""
    cfg = json.loads(
        (get_models_root() / "pit_prediction" / "model_config_undercut_v1.json").read_text(
            encoding="utf-8"
        )
    )
    return _classifier_auc_pr(
        "undercut", float(cfg["metrics"]["auc_pr_test"]), load_undercut_predictions
    )


_PIT_PUBLISHED_P50_MAE = 0.487  # registry headline (N15 P50 physical-stop MAE, s)


def _pit_p50_mae() -> ReproResult:
    """Re-derive the pit P50 MAE on the holdout rebuilt from raw laps (#364).

    ``pit_labeled`` is empty on disk, so the holdout is regenerated from the raw
    laps by ``pit_holdout.load_pit_holdout``. MAE of the P50 quantile prediction
    vs physical_stop_est on the 2025 slice.
    """
    from src.strategy.eval.pit_holdout import load_pit_holdout

    loaded = load_pit_holdout()
    if loaded is None:
        return ReproResult(
            "pit_duration",
            "p50_mae_test_s",
            _PIT_PUBLISHED_P50_MAE,
            None,
            "pending",
            "raw laps or pit models absent on disk",
        )

    import numpy as np

    test_slice, models, features = loaded
    predicted = models["p50"].predict(test_slice[features])
    actual = test_slice["physical_stop_est"].to_numpy()
    reproduced = float(np.mean(np.abs(predicted - actual)))
    delta = abs(reproduced - _PIT_PUBLISHED_P50_MAE)
    status = "reproduced" if delta <= TOLERANCE else "delta"
    return ReproResult(
        "pit_duration",
        "p50_mae_test_s",
        _PIT_PUBLISHED_P50_MAE,
        reproduced,
        status,
        f"n={len(actual)}; |delta| {delta:.4f} vs tol {TOLERANCE}",
    )


_PACE_PUBLISHED_MAE = 0.4104  # thesis-final N06 delta-model absolute-lap MAE (s)


def _pace_mae() -> ReproResult:
    """Re-derive the pace MAE on the 2025 holdout rebuilt in-memory from the featured laps.

    The N06 delta model predicts ``LapTime_Delta``; the reported MAE is on the
    reconstructed absolute lap time (``Prev_LapTime + delta`` vs ``LapTime_s``).
    ``pace_holdout.load_pace_predictions`` re-applies the two N06 feature steps
    and scores the frozen model, so no leaky session feature enters.
    """
    from src.strategy.eval.pace_holdout import load_pace_predictions

    loaded = load_pace_predictions()
    if loaded is None:
        return ReproResult(
            "pace",
            "mae_test_s",
            _PACE_PUBLISHED_MAE,
            None,
            "pending",
            "featured laptime holdout or delta model absent on disk",
        )

    import numpy as np

    y_true, y_pred = loaded
    reproduced = float(np.mean(np.abs(y_pred - y_true)))
    delta = abs(reproduced - _PACE_PUBLISHED_MAE)
    status = "reproduced" if delta <= TOLERANCE else "delta"
    return ReproResult(
        "pace",
        "mae_test_s",
        _PACE_PUBLISHED_MAE,
        reproduced,
        status,
        f"n={len(y_true)}; |delta| {delta:.4f} vs tol {TOLERANCE}",
    )


_TIRE_PUBLISHED_MAE = 0.7078  # thesis-final N09 global Model A test MAE (s), cumulative deg


def _tire_mae() -> ReproResult:
    """Re-derive the tire MAE on the 2025 holdout rebuilt from the featured laps.

    The headline 0.7078 is the global TireDegTCN (Model A) evaluated on all 2025
    test sequences (per-compound best is C2 fine-tuned at 0.5501). A deterministic
    ``model.eval()`` forward pass over the rebuilt sequences reproduces it.
    """
    from src.strategy.eval.tire_holdout import load_tire_predictions

    loaded = load_tire_predictions()
    if loaded is None:
        return ReproResult(
            "tire_degradation",
            "mae_test_s",
            _TIRE_PUBLISHED_MAE,
            None,
            "pending",
            "tire model bundle or featured holdout absent on disk",
        )

    import numpy as np

    y_true, y_pred = loaded
    reproduced = float(np.mean(np.abs(y_pred - y_true)))
    delta = abs(reproduced - _TIRE_PUBLISHED_MAE)
    status = "reproduced" if delta <= TOLERANCE else "delta"
    return ReproResult(
        "tire_degradation",
        "mae_test_s",
        _TIRE_PUBLISHED_MAE,
        reproduced,
        status,
        f"n={len(y_true)} seqs; global Model A; |delta| {delta:.4f} vs tol {TOLERANCE}",
    )


def collect_results() -> list[ReproResult]:
    """All reproduction checks, reproduced/delta rows first."""
    results = [
        _overtake_auc_pr(),
        _sc_auc_pr(),
        _undercut_auc_pr(),
        _pit_p50_mae(),
        _pace_mae(),
        _tire_mae(),
    ]
    order = {"delta": 0, "reproduced": 1, "pending": 2}
    return sorted(results, key=lambda r: order.get(r.status, 3))


def _render_table(results: list[ReproResult]) -> str:
    """Render reproduction results as a markdown table."""
    header = "| model | metric | published | reproduced | status | detail |"
    rule = "|---|---|---|---|---|---|"
    rows = []
    for r in results:
        reproduced = "-" if r.reproduced is None else f"{r.reproduced:.4f}"
        rows.append(
            f"| {r.model} | {r.metric} | {r.published:g} | {reproduced} | {r.status} | {r.detail} |"
        )
    return "\n".join([header, rule, *rows])


def build_reproduction_report() -> dict[str, Any]:
    """Regenerate the reproduction report and write the versioned output."""
    results = collect_results()
    models = get_models_root()
    artifacts = {"overtake_model": models / "overtake_probability" / "lgbm_overtake_v1.pkl"}
    header = build_header(dataset="2025 holdout vs published model_configs", artifacts=artifacts)
    payload = {"results": [asdict(r) for r in results]}
    md_path, json_path = write_report(REPRO_NAME, header, _render_table(results), payload)
    return {
        "header": asdict(header),
        "md_path": str(md_path),
        "json_path": str(json_path),
        **payload,
    }
