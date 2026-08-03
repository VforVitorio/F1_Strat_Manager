"""Pace (lap-time delta) holdout reconstruction from the featured laps (#372).

The N06 XGBoost model predicts ``LapTime_Delta`` (the change from the previous
lap); the headline MAE 0.4104 s is on the RECONSTRUCTED absolute lap time
(``Prev_LapTime + predicted_delta`` vs ``LapTime_s``) over the 2025 test slice,
not on the delta directly.

Unlike pit (which regenerates from raw laps), the pace holdout is derivable
in-memory: the featured 2025 parquet already carries every base column, so this
only re-applies the two N06 feature steps the delta model needs - the
categorical encoding and the lag-1 degradation features - and loads the frozen
model to score them.

The model's saved ``xgb_laptime_delta_feature_names.json`` IS ``FEATURES_DELTA``:
the session-leaky features (``year_circuit_median``, ``team_pace_rank``,
``lap_time_pct_of_race_fastest``) and the same-lap degradation/speed columns were
already excluded at train time (hygiene.py flags them LEAKY), so the 25
prediction columns come straight from that list and no leaky feature enters.

--- WHERE TO CHANGE IF THE PACE PIPELINE CHANGES ---
notebooks/strategy/lap_time_prediction/N06_laptime_model.ipynb (cells 6 encode,
25 load-2025, 29 lag features, 32 test reconstruction) is the source of truth;
mirror any edit here. Validated: MAE 0.4104 on 2025, exact to the thesis-final
headline.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from src.f1_strat_manager.data_cache import get_data_root, get_models_root

_LAP_DIR = "lap_time"
_MANIFEST = "feature_manifest_laptime.json"
# groupby keys for the lag-1 shift: one stint of one driver in one race (N06 cell 29)
_STINT_KEYS = ["GP_Name", "Year", "DriverNumber", "Stint"]
_LAG_SOURCES = {
    "Prev_DegradationRate": "DegradationRate",
    "Prev_CumulativeDeg": "CumulativeDeg",
    "Prev_DegAcceleration": "DegAcceleration",
}
_DROPNA = ["LapTime_Delta", "Prev_LapTime"]  # N06 drops first-lap-of-stint NaNs before scoring
_PREV_LAP = "Prev_LapTime"


def _encode_categoricals(df: Any, compound_map: dict, racephase_map: dict) -> Any:
    """Map the string categoricals to the int codes the model was trained on (N06 encode_features).

    Only ``FreshTyre`` feeds the delta model directly; ``Compound`` / ``race_phase``
    are encoded for parity with the notebook (the model consumes the pre-numeric
    ``CompoundID``), so the transform is applied verbatim.
    """
    out = df.copy()
    out["Compound"] = out["Compound"].map(compound_map).fillna(-1).astype(int)
    out["race_phase"] = out["race_phase"].astype(str).map(racephase_map).fillna(-1).astype(int)
    out["FreshTyre"] = out["FreshTyre"].astype(int)
    return out


def _add_lag_deg_features(df: Any) -> Any:
    """Add the lag-1 degradation features the delta model uses (N06 add_lag_deg_features).

    The same-lap ``DegradationRate`` / ``CumulativeDeg`` / ``DegAcceleration`` are
    leaky for a next-lap delta target, so the model consumes their previous-lap
    value: a shift(1) within each (race, year, driver, stint) group. The first
    lap of each stint becomes NaN and is dropped before scoring.
    """
    out = df.copy()
    grp = out.groupby(_STINT_KEYS, sort=False)
    for prev_col, source_col in _LAG_SOURCES.items():
        out[prev_col] = grp[source_col].shift(1)
    return out


def load_pace_predictions(year: int = 2025) -> tuple[np.ndarray, np.ndarray] | None:
    """Rebuild the pace holdout and return ``(y_true_abs, y_pred_abs)`` lap times in seconds.

    Loads the featured ``year`` parquet, re-applies the two N06 feature steps,
    scores the frozen delta model, and reconstructs the absolute lap time from
    the predicted delta plus ``Prev_LapTime`` - the exact quantity the headline
    MAE is measured on. Returns ``None`` when the model, feature list, manifest,
    or holdout parquet is absent, so the caller degrades to a ``pending`` result.
    """
    import json

    import pandas as pd
    from xgboost import XGBRegressor

    model_dir = get_models_root() / _LAP_DIR
    model_path = model_dir / "xgb_laptime_delta_final.json"
    features_path = model_dir / "xgb_laptime_delta_feature_names.json"
    manifest_path = get_data_root() / "processed" / _MANIFEST
    parquet = get_data_root() / "processed" / f"laps_featured_{year}.parquet"
    if not all(p.exists() for p in (model_path, features_path, manifest_path, parquet)):
        return None

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    compound_map = manifest["categorical_encoding"]["Compound"]
    racephase_map = manifest["categorical_encoding"]["race_phase"]
    target = manifest["target"]  # LapTime_s
    features_delta = json.loads(features_path.read_text(encoding="utf-8"))

    # Through augment_featured_laps, never straight from the parquet: the 2025 artefact
    # ships without the four weather columns N06 was trained on, so a direct read raises
    # a KeyError here (#782). That module's own docstring has said "every consumer must
    # call it" since the third time this happened; this was the fourth.
    from src.f1_strat_manager.laps_augment import augment_featured_laps

    featured = augment_featured_laps(pd.read_parquet(parquet), year)
    df = _add_lag_deg_features(_encode_categoricals(featured, compound_map, racephase_map))
    df = df.dropna(subset=_DROPNA).copy()
    if df.empty:
        return None

    model = XGBRegressor()
    model.load_model(str(model_path))

    pred_delta = model.predict(df[features_delta])
    y_pred_abs = df[_PREV_LAP].to_numpy() + pred_delta
    y_true_abs = df[target].to_numpy()
    return y_true_abs, y_pred_abs
