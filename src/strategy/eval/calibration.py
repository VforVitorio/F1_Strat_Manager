"""E-03 calibration verification for the ML predictors.

The audit's finding is "calibration asserted, never verified": the pit
P05-P95 interval is already published broken (coverage 0.7047 vs 0.90
nominal), yet nothing re-checks it. This module makes calibration a measured
quantity so the paper can state it as fact, and it deliberately "finds" the
known pit breakage as a retro-validation of the harness itself.

Scope this phase (#206):
- **overtake** - reliability (Brier + ECE) recomputed on the 2025 holdout,
  wiring N33 Section A's proven loader + interaction-feature logic.
- **pit_duration** - the P05-P95 quantile coverage, flagged as drift vs the
  0.90 nominal.
- **tire_degradation** - the deployed MC-Dropout sigma per compound.
- **safety_car / undercut** - recomputed on the 2025 holdout by rebuilding their
  engineered features in-memory (SC: the 5 derived features from N14 Step 2;
  undercut: the two train-fit target encodings from N16 Step 3), the same
  ``_add_overtake_features`` pattern. Both reproduce their published AUC-PR
  exactly (#364).

ponytail: pit coverage + the MC sigma are read from frozen artifacts, not
re-run from a holdout (``pit_labeled`` is empty on disk; the MC pass needs a
torch forward pass x N). Upgrade path when the holdouts land: recompute both
from data, the same way N33 Section D already does for the TCN.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from src.f1_strat_manager.data_cache import get_data_root, get_models_root
from src.strategy.eval.report import build_header, write_report

CAL_NAME = "calibration"
NOMINAL_COVERAGE = 0.90
ECE_BINS = 10
ECE_DRIFT = 0.05  # expected-calibration-error above this is flagged as drift

# Reproduced verbatim from N33 Section A / the overtake model_config so the
# harness feeds the LightGBM the exact 15-feature vector it was trained on.
_OVERTAKE_FEATURES = [
    "gap_ahead_s",
    "pace_delta_s",
    "tyre_life_x",
    "tyre_life_y",
    "tyre_life_diff",
    "speed_trap_delta",
    "LapNumber",
    "drs_window",
    "compound_x",
    "compound_y",
    "circuit_cluster",
    "gap_pace_product",
    "drs_ready_gap",
    "gap_trend",
    "pace_delta_rolling3",
]
_OVERTAKE_CAT = ["compound_x", "compound_y", "circuit_cluster"]
_OVERTAKE_PAIR_KEYS = ["Year", "GP_Name", "driver_x", "driver_y"]

# SC + undercut recompute: the model feature lists are read from disk; the
# engineered features are rebuilt in-memory from the base parquet, the same
# pattern _add_overtake_features uses.
_SC_DIR = "safety_car_probability"
_SC_LAPTIME_Z_COLS = ["lap_time_mean", "lap_time_std", "lap_time_min"]  # per-race z-scored
_SC_TARGETS = ("sc_within_3_laps", "sc_within_5_laps", "sc_within_7_laps")
_UNDERCUT_DIR = "pit_prediction"
_UNDERCUT_TRAIN_YEARS = (2023, 2024)
_UNDERCUT_TARGET = "undercut_success"


@dataclass
class CalibrationResult:
    """One calibration measurement.

    ``status`` is ``ok`` (within nominal), ``drift`` (measured worse than
    nominal - the harness caught a miscalibration), or ``pending`` (cannot be
    recomputed on-disk this phase; deferred to #207 / holdout availability).
    """

    model: str
    metric: str
    value: float | None
    nominal: float | None
    status: str
    detail: str


def _ece(y_true: np.ndarray, proba: np.ndarray, n_bins: int = ECE_BINS) -> float:
    """Expected Calibration Error: bin-weighted gap between confidence and accuracy.

    A perfectly calibrated classifier has ECE 0 (in each confidence bin, the
    predicted probability equals the empirical positive rate). Standard
    equal-width binning over [0, 1].
    """
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    total = len(y_true)
    error = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        in_bin = (proba > lo) & (proba <= hi)
        count = int(in_bin.sum())
        if count == 0:
            continue
        confidence = float(proba[in_bin].mean())
        accuracy = float(y_true[in_bin].mean())
        error += (count / total) * abs(confidence - accuracy)
    return error


def _add_overtake_features(df: "Any") -> "Any":
    """Reproduce the N12 interaction + rolling features from the base columns.

    The labeled parquet stores only the base columns; the four derived
    features (products + per-pair rolling/diff) are rebuilt here exactly as the
    training notebook did, or the LightGBM sees a different feature space and
    the numbers do not reproduce.
    """
    df = df.copy()
    df["gap_pace_product"] = df["gap_ahead_s"] * df["pace_delta_s"]
    df["drs_ready_gap"] = df["gap_ahead_s"] * df["drs_window"]
    df = df.sort_values(_OVERTAKE_PAIR_KEYS + ["LapNumber"]).copy()
    grp = df.groupby(_OVERTAKE_PAIR_KEYS)
    df["gap_trend"] = grp["gap_ahead_s"].diff().fillna(0.0)
    df["pace_delta_rolling3"] = grp["pace_delta_s"].transform(
        lambda x: x.rolling(3, min_periods=1).mean()
    )
    for col in _OVERTAKE_CAT:
        df[col] = df[col].astype("category")
    return df


def load_overtake_predictions(year: int = 2025) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Load an overtake season holdout (default 2025 test) -> ``(y, proba_raw, proba_cal)``.

    ``year`` selects the season slice: 2025 = test (calibration/reproduction),
    2024 = validation (the #207 hygiene threshold re-selection).

    Shared seam: both the calibration metrics here and the headline-metric
    reproduction (``reproduce.py``) need the same holdout + model + calibrator,
    so the load lives in one place. Returns ``None`` when the artifacts or the
    holdout parquet are absent, letting each caller degrade to a ``pending``
    result instead of crashing.
    """
    import joblib
    import pandas as pd

    model_dir = get_models_root() / "overtake_probability"
    parquet = (
        get_data_root() / "processed" / "overtake_labeled" / "overtake_pairs_2023_2025.parquet"
    )
    model_path = model_dir / "lgbm_overtake_v1.pkl"
    calib_path = model_dir / "calibrator.pkl"
    if not (parquet.exists() and model_path.exists() and calib_path.exists()):
        return None

    df = _add_overtake_features(pd.read_parquet(parquet))
    test = df[df["Year"] == year].copy()
    model = joblib.load(model_path)
    calibrator = joblib.load(calib_path)

    x = test[_OVERTAKE_FEATURES]
    y = test["overtake"].astype(int).to_numpy()
    proba_raw = model.predict_proba(x)[:, 1]
    proba_cal = calibrator.predict_proba(proba_raw.reshape(-1, 1))[:, 1]
    return y, proba_raw, proba_cal


def _add_sc_features(df: "Any") -> "Any":
    """Rebuild the SC model's engineered features in-memory (N14 Step 2, verbatim).

    The persisted sc_labeled parquet stores base columns; the model consumes 5
    derived features - per-race z-scores of lap_time_{mean,std,min} plus the
    anomaly_and_yellow and lap1_chaos interactions - rebuilt here exactly as N14
    did (per-race median imputation first, then z-scores, then weather fill), or
    the LightGBM sees a different feature space and the number moves.

    --- WHERE TO CHANGE IF THE SC FEATURES CHANGE ---
    notebooks/strategy/sc_probability/N14_sc_model.ipynb (Step 2) is the source
    of truth; mirror any edit here.
    """
    df = df.copy()
    race_key = "race_id" if "race_id" in df.columns else "event_name"
    for col in ["lap_time_mean", "lap_time_std", "lap_time_min", "lap_time_cv", "lap_time_trend_5"]:
        if col in df.columns:
            per_race_median = df.groupby(race_key)[col].transform("median")
            df[col] = df[col].fillna(per_race_median).fillna(df[col].median())
    for col in _SC_LAPTIME_Z_COLS:
        if col in df.columns:
            race_mu = df.groupby(race_key)[col].transform("mean")
            race_sigma = df.groupby(race_key)[col].transform("std").clip(lower=0.01)
            df[f"{col}_z"] = (df[col] - race_mu) / race_sigma
    for col in ["track_temp", "air_temp", "humidity", "track_temp_delta"]:
        if col in df.columns:
            df[col] = df.groupby(race_key)[col].transform(lambda s: s.ffill().bfill())
            df[col] = df[col].fillna(df[col].median())
    has_anomaly = {"driver_anomaly_hard_count", "yellow_escalation_count"} <= set(df.columns)
    df["anomaly_and_yellow"] = (
        ((df["driver_anomaly_hard_count"] > 0) & (df["yellow_escalation_count"] > 0)).astype(int)
        if has_anomaly
        else 0
    )
    has_lap1 = {"is_lap1", "n_drivers_delta"} <= set(df.columns)
    df["lap1_chaos"] = (df["is_lap1"] * df["n_drivers_delta"].abs()).astype(int) if has_lap1 else 0
    return df


def _sc_frame_and_proba(year: int) -> tuple["Any", np.ndarray, np.ndarray] | None:
    """Load the SC year-slice with engineered features + model probabilities.

    Returns ``(df_slice, proba_raw, proba_cal)`` or ``None``. Shared seam: the
    calibration recompute (3-lap target) and the #363 threshold/window correction
    both need the same slice + model output, so the load lives here once. The
    feature order is read from the model's ``feature_list_v1.json``.
    """
    import json

    import joblib
    import pandas as pd

    model_dir = get_models_root() / _SC_DIR
    parquet = get_data_root() / "processed" / "sc_labeled" / "sc_labeled_2023_2025.parquet"
    model_path = model_dir / "lgbm_sc_v1.pkl"
    calib_path = model_dir / "calibrator_sc_v1.pkl"
    feature_list_path = model_dir / "feature_list_v1.json"
    if not all(p.exists() for p in (parquet, model_path, calib_path, feature_list_path)):
        return None

    features = json.loads(feature_list_path.read_text(encoding="utf-8"))["features"]
    df = _add_sc_features(pd.read_parquet(parquet))
    df_slice = df[df["year"] == year].copy()
    model = joblib.load(model_path)
    calibrator = joblib.load(calib_path)
    proba_raw = model.predict_proba(df_slice[features])[:, 1]
    proba_cal = calibrator.predict_proba(proba_raw.reshape(-1, 1))[:, 1]
    return df_slice, proba_raw, proba_cal


def load_sc_predictions(
    year: int = 2025, target: str = "sc_within_3_laps"
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """``(y, proba_raw, proba_cal)`` for a SC target window, or ``None`` if artifacts absent.

    ``year`` selects the season slice (2025 = test, 2024 = validation for the
    #363 threshold re-selection); ``target`` selects the SC window (3/5/7 laps),
    which the #363 window correction sweeps.
    """
    loaded = _sc_frame_and_proba(year)
    if loaded is None:
        return None
    df_slice, proba_raw, proba_cal = loaded
    y = df_slice[target].astype(int).to_numpy()
    return y, proba_raw, proba_cal


def load_undercut_predictions(
    year: int = 2025,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """``(y, proba_raw, proba_cal)`` for the undercut holdout with train-fit target encodings.

    The two aggregate features (circuit_undercut_rate, team_x_undercut_rate) are
    target encodings fit on the 2023-2024 train slice and applied to the requested
    year (N16 Step 3), so the split precedes the encoding and no train<-test leak
    occurs. Returns ``None`` when the artifacts or holdout are absent.
    """
    import json

    import joblib
    import pandas as pd

    model_dir = get_models_root() / _UNDERCUT_DIR
    parquet = get_data_root() / "processed" / "undercut_labeled" / "undercut_clean.parquet"
    model_path = model_dir / "lgbm_undercut_v1.pkl"
    calib_path = model_dir / "calibrator_undercut_v1.pkl"
    cfg_path = model_dir / "model_config_undercut_v1.json"
    if not all(p.exists() for p in (parquet, model_path, calib_path, cfg_path)):
        return None

    features = json.loads(cfg_path.read_text(encoding="utf-8"))["features"]
    df = pd.read_parquet(parquet)
    train = df[df["Year"].isin(_UNDERCUT_TRAIN_YEARS)]
    target_slice = df[df["Year"] == year].copy()
    for group_col, feat in (
        ("circuit_key", "circuit_undercut_rate"),
        ("Team_X", "team_x_undercut_rate"),
    ):
        encoding = train.groupby(group_col)[_UNDERCUT_TARGET].mean()
        fallback = train[_UNDERCUT_TARGET].mean()
        target_slice[feat] = target_slice[group_col].map(encoding).fillna(fallback)

    model = joblib.load(model_path)
    calibrator = joblib.load(calib_path)
    proba_raw = model.predict_proba(target_slice[features])[:, 1]
    proba_cal = calibrator.predict_proba(proba_raw.reshape(-1, 1))[:, 1]
    y = target_slice[_UNDERCUT_TARGET].astype(int).to_numpy()
    return y, proba_raw, proba_cal


def _overtake_calibration() -> list[CalibrationResult]:
    """Recompute Brier + ECE for the overtake classifier on the 2025 holdout.

    Uses the calibrated probabilities (Platt on val-2024) - the quantity the
    production agent actually consumes. Returns a ``pending`` result instead of
    raising if the artifacts are absent, so a partial install still reports.
    """
    loaded = load_overtake_predictions()
    if loaded is None:
        return [
            CalibrationResult(
                "overtake", "ece", None, ECE_DRIFT, "pending", "artifacts or holdout absent on disk"
            )
        ]
    y, proba_raw, proba_cal = loaded

    brier_raw = float(np.mean((proba_raw - y) ** 2))
    brier_cal = float(np.mean((proba_cal - y) ** 2))
    ece_cal = _ece(y, proba_cal)
    status = "drift" if ece_cal > ECE_DRIFT else "ok"
    detail = f"n={len(y)}; brier raw {brier_raw:.4f} -> cal {brier_cal:.4f} (Platt val-2024)"

    return [
        CalibrationResult("overtake", "brier_calibrated", brier_cal, None, "ok", detail),
        CalibrationResult(
            "overtake",
            "ece_calibrated",
            ece_cal,
            ECE_DRIFT,
            status,
            f"n={len(y)}; {ECE_BINS}-bin equal-width",
        ),
    ]


def _pit_quantile_coverage() -> list[CalibrationResult]:
    """Surface the pit P05-P95 empirical coverage and flag it vs nominal.

    ponytail: read from the frozen ``model_config`` (the value is published
    there) rather than re-run - ``pit_labeled`` is empty on disk so there is no
    holdout to predict on. Upgrade path: recompute empirical coverage from the
    hist_pit P05/P95 models over the N15 holdout once it lands. This is the
    retro-validation target: the harness must surface the known 0.7047 break.
    """
    cfg_path = get_models_root() / "pit_prediction" / "model_config.json"
    if not cfg_path.exists():
        return [
            CalibrationResult(
                "pit_duration",
                "p05_p95_coverage",
                None,
                NOMINAL_COVERAGE,
                "pending",
                "model_config absent",
            )
        ]

    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    coverage = float(cfg["eval"]["p05_p95_coverage_test"])
    status = "drift" if coverage < NOMINAL_COVERAGE else "ok"
    detail = f"config-declared (recompute pending N15 holdout); {coverage:.4f} vs {NOMINAL_COVERAGE:.2f} nominal"
    return [
        CalibrationResult(
            "pit_duration", "p05_p95_coverage", coverage, NOMINAL_COVERAGE, status, detail
        )
    ]


def _tcn_mc_sigma() -> list[CalibrationResult]:
    """Report the deployed MC-Dropout sigma per compound from the frozen JSON.

    The stored ``mean_sigma_s`` is the epistemic band the tire agent uses at
    runtime. Empirical coverage validation (predicted vs residual sigma) needs
    the torch MC pass N33 Section D runs and is wired-pending here.
    """
    calib_path = get_models_root() / "tire_degradation" / "mc_dropout_calibration.json"
    if not calib_path.exists():
        return [
            CalibrationResult(
                "tire_degradation",
                "mc_mean_sigma_s",
                None,
                None,
                "pending",
                "mc_dropout_calibration.json absent",
            )
        ]

    calib = json.loads(calib_path.read_text(encoding="utf-8"))
    results = []
    for compound in sorted(calib):
        sigma = float(calib[compound]["mean_sigma_s"])
        detail = "deployed epistemic sigma; empirical coverage wired-pending (N33-D)"
        results.append(
            CalibrationResult(
                f"tire_degradation[{compound}]", "mc_mean_sigma_s", sigma, None, "ok", detail
            )
        )
    return results


def _classifier_calibration(model_name: str, loader: "Any") -> list[CalibrationResult]:
    """Brier + ECE on the 2025 holdout for a binary classifier via its loader.

    Generic over safety_car / undercut: both rebuild their engineered features
    in-memory (the loaders in this module) and report the calibrated Brier + ECE,
    the same quantities ``_overtake_calibration`` reports for overtake. Returns a
    ``pending`` row when the loader cannot find its artifacts.
    """
    loaded = loader()
    if loaded is None:
        return [
            CalibrationResult(
                model_name,
                "ece_calibrated",
                None,
                ECE_DRIFT,
                "pending",
                "artifacts or holdout absent on disk",
            )
        ]
    y, _proba_raw, proba_cal = loaded
    brier_cal = float(np.mean((proba_cal - y) ** 2))
    ece_cal = _ece(y, proba_cal)
    status = "drift" if ece_cal > ECE_DRIFT else "ok"
    return [
        CalibrationResult(
            model_name,
            "brier_calibrated",
            brier_cal,
            None,
            "ok",
            f"n={len(y)}; recomputed on 2025 holdout",
        ),
        CalibrationResult(
            model_name,
            "ece_calibrated",
            ece_cal,
            ECE_DRIFT,
            status,
            f"n={len(y)}; {ECE_BINS}-bin equal-width",
        ),
    ]


def collect_results() -> list[CalibrationResult]:
    """All calibration results across the predictors."""
    return [
        *_overtake_calibration(),
        *_classifier_calibration("safety_car", load_sc_predictions),
        *_classifier_calibration("undercut", load_undercut_predictions),
        *_pit_quantile_coverage(),
        *_tcn_mc_sigma(),
    ]


def _render_table(results: list[CalibrationResult]) -> str:
    """Render calibration results as a markdown table, drift rows first."""
    order = {"drift": 0, "pending": 1, "ok": 2}
    ordered = sorted(results, key=lambda r: order.get(r.status, 3))
    header = "| model | metric | value | nominal | status | detail |"
    rule = "|---|---|---|---|---|---|"
    rows = []
    for r in ordered:
        value = "-" if r.value is None else f"{r.value:.4f}"
        nominal = "-" if r.nominal is None else f"{r.nominal:g}"
        rows.append(f"| {r.model} | {r.metric} | {value} | {nominal} | {r.status} | {r.detail} |")
    return "\n".join([header, rule, *rows])


def build_calibration_report() -> dict[str, Any]:
    """Regenerate the calibration report and write the versioned output.

    Returns the payload dict (also written to
    ``documents/eval_reports/calibration.json``). The report always contains
    the pit P05-P95 coverage row flagged as drift - the retro-validation that
    proves the harness finds a known-broken calibration.
    """
    results = collect_results()
    models = get_models_root()
    artifacts = {
        "overtake_model": models / "overtake_probability" / "lgbm_overtake_v1.pkl",
        "pit_cfg": models / "pit_prediction" / "model_config.json",
        "tcn_mc_calib": models / "tire_degradation" / "mc_dropout_calibration.json",
    }
    header = build_header(
        dataset="2025 holdout + frozen calibration artifacts",
        seed_policy="deterministic",
        artifacts=artifacts,
    )
    payload = {"results": [asdict(r) for r in results]}
    md_path, json_path = write_report(CAL_NAME, header, _render_table(results), payload)
    return {
        "header": asdict(header),
        "md_path": str(md_path),
        "json_path": str(json_path),
        **payload,
    }
