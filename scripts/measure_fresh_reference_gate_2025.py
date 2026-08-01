"""Held-out check: does the fresh-reference quality gate (threshold chosen on
2023-24, shipped in `TireAgentConfig.fresh_reference_max_pct_of_fastest`) still
help on 2025 -- the season the system actually runs in?

`scripts/measure_fresh_reference_gate.py` measured a 33% mean-error / 60% bias
reduction, but only ever on `laps_tiredeg.parquet` (2023-24, the only parquet
carrying N04's training target). That is a real number about the seasons the
threshold was CHOSEN on, not evidence about what ships -- see
`documents/audits/MEASURE_fresh_reference_quality_gate.md`'s 2025 addendum.

This script measures the identical diagnostic on `laps_featured_2025.parquet`
(the real, full 24-race 2025 season), reusing the actual production functions
(`_add_compound_cols`, `_compound_name_to_id`, `_reject_contaminated_laps`)
rather than reimplementing the compound-resolution or gating logic -- the
featured parquet does not carry `AbsoluteCompoundID`/`CompoundHardness`, so
those have to come from the same code path production uses, not a guess.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.agents.tire_agent import (  # noqa: E402
    CFG,
    TireDegTCN,
    _add_compound_cols,
    _compound_name_to_id,
    _reject_contaminated_laps,
)

FEATURED_2025 = Path("data/processed/laps_featured_2025.parquet")
MODEL_DIR = Path("data/models/tire_degradation")
FRESH_MAX_TYRE_LIFE = 3
STINT_KEYS = ["Year", "GP_Name", "DriverNumber", "Stint"]
TARGET = "FuelAdjustedDegAbsolute"


def load_bundles() -> dict:
    routing = json.loads((MODEL_DIR / "routing_config.json").read_text(encoding="utf-8"))
    by_file: dict = {}
    by_compound: dict = {}
    for compound, entry in routing.items():
        filename = entry["bundle"]
        if filename not in by_file:
            bundle = torch.load(MODEL_DIR / filename, map_location="cpu", weights_only=False)
            model = TireDegTCN(bundle["n_features"], **bundle["model_hparams"])
            model.load_state_dict(bundle["state_dict"])
            model.eval()
            bundle["model"] = model
            by_file[filename] = bundle
        by_compound[compound] = by_file[filename]
    return by_compound


def stint_sequences(stint: pd.DataFrame, bundle: dict) -> np.ndarray:
    """One scaled (window, n_features) sequence per lap -- same transform as
    `scripts/measure_tyre_reference.py::stint_sequences`, kept in sync by hand."""
    window = bundle["window"]
    raw = stint[bundle["feature_names"]].astype(float).fillna(0)
    scaled = bundle["scaler"].transform(raw)
    sequences = []
    for length in range(1, len(scaled) + 1):
        prefix = scaled[:length]
        if len(prefix) >= window:
            sequences.append(prefix[-window:])
            continue
        padding = np.zeros((window - len(prefix), prefix.shape[1]), dtype=prefix.dtype)
        sequences.append(np.concatenate([padding, prefix], axis=0))
    return np.stack(sequences)


def predict_2025_stints() -> pd.DataFrame:
    """Run every 2025 stint through its compound's TCN, lap by lap."""
    laps = pd.read_parquet(FEATURED_2025)
    laps = laps.dropna(subset=["Compound", "GP_Name", "Year"])
    # A handful of rows carry the literal string "None"/"nan" rather than a real
    # null (a known quirk of this parquet's Compound column) -- dropna misses those.
    laps = laps[~laps["Compound"].isin(["None", "nan"])]
    bundles = load_bundles()

    rows = []
    for keys, stint in laps.groupby(STINT_KEYS, sort=False):
        year, gp_name = int(keys[0]), keys[1]
        compound_id = _compound_name_to_id(str(stint["Compound"].iloc[0]), gp_name, year)
        if compound_id not in bundles:
            continue

        bundle = bundles[compound_id]
        # The real production transform, not a reimplementation: this is the
        # same call `_build_stint_features` makes, so AbsoluteCompoundID and
        # CompoundHardness land on the identical scale the model was trained on.
        ordered = _add_compound_cols(stint.sort_values("TyreLife").copy(), compound_id)
        sequences = stint_sequences(ordered, bundle)
        with torch.no_grad():
            batch = torch.tensor(sequences, dtype=torch.float32)
            predictions = bundle["model"](batch).numpy().reshape(-1)

        stint_id = "|".join(str(k) for k in keys)
        for tyre_life, pred, target, lap_time_s in zip(
            ordered["TyreLife"].to_numpy(),
            predictions,
            ordered[TARGET].to_numpy(),
            ordered["LapTime_s"].to_numpy(),
            strict=True,
        ):
            rows.append(
                (
                    stint_id,
                    gp_name,
                    year,
                    float(tyre_life),
                    float(pred),
                    float(target),
                    float(lap_time_s),
                )
            )

    columns = ["stint", "gp_name", "year", "tyre_life", "pred", "target", "lap_time_s"]
    return pd.DataFrame(rows, columns=columns)


def bound(scored: pd.DataFrame, label: str) -> dict:
    absolute = scored["error"].abs()
    result = {
        "label": label,
        "n": len(scored),
        "mean_abs_error": round(float(absolute.mean()), 3),
        "median_abs_error": round(float(absolute.median()), 3),
        "signed_bias": round(float(scored["error"].mean()), 3),
    }
    print(
        f"\n=== {label} (n={result['n']}) ===\n"
        f"  mean abs error   {result['mean_abs_error']:7.3f} s/lap\n"
        f"  median abs error {result['median_abs_error']:7.3f} s/lap\n"
        f"  bias (signed)    {result['signed_bias']:+7.3f} s/lap"
    )
    return result


def main() -> None:
    frame = predict_2025_stints().sort_values(["stint", "tyre_life"])
    print(
        f"2025 laps predicted: {len(frame)}, stints: {frame['stint'].nunique()}, "
        f"GPs: {frame['gp_name'].nunique()}"
    )

    fastest_by_gp = frame.groupby(["year", "gp_name"])["lap_time_s"].transform("min")
    frame["fastest_lap_s"] = fastest_by_gp

    fresh = frame[frame["tyre_life"] <= FRESH_MAX_TYRE_LIFE]

    def scored_bound(gated: bool) -> pd.DataFrame:
        if gated:
            candidates = fresh.rename(columns={"lap_time_s": "LapTime_s"})
            kept = _reject_contaminated_laps(
                candidates,
                fastest_lap_s=candidates["fastest_lap_s"],
                max_pct=CFG.fresh_reference_max_pct_of_fastest,
            )
        else:
            kept = fresh

        ref_pred = kept.groupby("stint")["pred"].last()
        ref_true = kept.groupby("stint")["target"].last()
        print(f"  stints with a reference: {ref_pred.notna().sum()} of {fresh['stint'].nunique()}")

        scored = frame[frame["tyre_life"] > FRESH_MAX_TYRE_LIFE].copy()
        scored["ref_pred"] = scored["stint"].map(ref_pred)
        scored["ref_true"] = scored["stint"].map(ref_true)
        scored = scored.dropna(subset=["ref_pred", "ref_true"])
        scored["error"] = (scored["pred"] - scored["ref_pred"]) - (
            scored["target"] - scored["ref_true"]
        )
        return scored

    bound(scored_bound(gated=False), "2025 BASELINE -- no gate")
    bound(scored_bound(gated=True), f"2025 GATED @ {CFG.fresh_reference_max_pct_of_fastest}")


if __name__ == "__main__":
    main()
