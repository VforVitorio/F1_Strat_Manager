"""Measure what gating the fresh-reference lap on race pace buys `deg_cost_s`.

`deg_cost_s = pred(now) - pred(fresh reference)`, both from the TCN. Its TRUE
counterpart is the same difference taken on N04's own target column, and the
error is the difference of the two differences -- the same bound
`scripts/measure_deg_error_bound.py` (branch `feat/deferral-tyre-liability`,
issue #763) measured at 0.650 s/lap mean absolute, +0.351 s/lap signed bias.

That investigation traced the growth to a small number of STINTS whose
fresh-reference lap (`TyreLife <= FRESH_MAX_TYRE_LIFE`) was itself a
Safety-Car- or red-flag-affected lap: `track_status_clean` is supposed to flag
this but is a constant 0 across the whole featured parquet (see
`_add_session_cols`'s docstring), because N04's `IsAccurate` gate does not
catch every neutralised lap. Measured counter-example: Mexico City 2023 car 4,
lap 36 reads 137.757 s on a circuit whose green-flag pace is ~83 s, and every
later lap in that nominal stint then priced tens of seconds "faster than
fresh" -- not tyre wear, an artefact of a contaminated zero point.

`TireAgent._fresh_reference` now rejects a candidate lap slower than
`fresh_reference_max_pct_of_fastest` times the race's fastest lap via
`_reject_contaminated_laps`, imported here rather than reimplemented so this
measures what actually ships.

Training seasons only (2023-24), matching `measure_tyre_reference.py` and
`src/strategy/eval/hygiene.py`'s reasoning against fitting on the test season.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.measure_tyre_reference import (  # noqa: E402
    FRESH_MAX_TYRE_LIFE,
    predict_training_stints,
)
from src.agents.tire_agent import CFG, _reject_contaminated_laps  # noqa: E402

LAPS_PATH = Path("data/processed/laps_tiredeg.parquet")
TRAINING_YEARS = (2023, 2024)
STINT_KEYS = ["Year", "GP_Name", "DriverNumber", "Stint"]


def attach_pct_of_fastest(frame: pd.DataFrame) -> pd.DataFrame:
    """Join `lap_time_pct_of_race_fastest`'s two inputs from the raw parquet.

    `predict_training_stints()` returns only `stint`/`tyre_life`/`pred`/`target`,
    so the lap time and the race's fastest lap have to be pulled back in.
    `TyreLife` is not unique within a (stint) group for a handful of red-flag
    affected stints -- the same nominal `Stint` number restarts `TyreLife` after
    the restart -- so duplicates are collapsed to the WORST (slowest) candidate:
    missing a contaminated lap is exactly the failure mode being measured.
    """
    laps = pd.read_parquet(LAPS_PATH)
    laps = laps[laps["Year"].isin(TRAINING_YEARS)].copy()
    laps["stint_id"] = laps[STINT_KEYS].astype(str).agg("|".join, axis=1)
    laps["fastest_lap_s"] = laps.groupby(["Year", "GP_Name"])["LapTime_s"].transform("min")

    lap_time = laps.groupby(["stint_id", "TyreLife"])["LapTime_s"].max()
    fastest = laps.groupby(["stint_id", "TyreLife"])["fastest_lap_s"].first()

    key = frame.set_index(["stint", "tyre_life"]).index
    frame = frame.copy()
    frame["lap_time_s"] = key.map(lap_time)
    frame["fastest_lap_s"] = key.map(fastest)
    return frame


def score(scored: pd.DataFrame, label: str) -> dict:
    """The bound: mean/median absolute error, signed bias, and error by band."""
    absolute = scored["error"].abs()
    bands = pd.cut(scored["tyre_life"], [3, 10, 20, 30, 100])
    by_band = scored.groupby(bands, observed=True)["error"].mean()

    result = {
        "label": label,
        "n": len(scored),
        "mean_abs_error": round(float(absolute.mean()), 3),
        "median_abs_error": round(float(absolute.median()), 3),
        "signed_bias": round(float(scored["error"].mean()), 3),
        "by_band_mean_error": {str(k): round(float(v), 3) for k, v in by_band.items()},
    }
    print(
        f"\n=== {label} (n={result['n']}) ===\n"
        f"  mean abs error   {result['mean_abs_error']:7.3f} s/lap\n"
        f"  median abs error {result['median_abs_error']:7.3f} s/lap\n"
        f"  bias (signed)    {result['signed_bias']:+7.3f} s/lap"
    )
    for band, value in result["by_band_mean_error"].items():
        print(f"    {band:<12} {value:+.3f}")
    return result


def build_reference(fresh: pd.DataFrame, gated: bool) -> tuple[pd.Series, pd.Series]:
    """The fresh-reference prediction and target, gated or not, per stint."""
    if gated:
        clean = _reject_contaminated_laps(
            fresh.rename(columns={"lap_time_s": "LapTime_s"}),
            fastest_lap_s=fresh["fastest_lap_s"],
            max_pct=CFG.fresh_reference_max_pct_of_fastest,
        )
    else:
        clean = fresh
    return clean.groupby("stint")["pred"].last(), clean.groupby("stint")["target"].last()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, help="write the measured scores to this JSON path")
    args = parser.parse_args()

    frame = predict_training_stints().sort_values(["stint", "tyre_life"])
    frame = attach_pct_of_fastest(frame)
    fresh = frame[frame["tyre_life"] <= FRESH_MAX_TYRE_LIFE].dropna(
        subset=["lap_time_s", "fastest_lap_s"]
    )

    def scored_bound(gated: bool) -> pd.DataFrame:
        ref_pred, ref_true = build_reference(fresh, gated)
        n_stints_before = fresh["stint"].nunique()
        n_stints_after = ref_pred.notna().sum()

        scored = frame[frame["tyre_life"] > FRESH_MAX_TYRE_LIFE].copy()
        scored["ref_pred"] = scored["stint"].map(ref_pred)
        scored["ref_true"] = scored["stint"].map(ref_true)
        scored = scored.dropna(subset=["ref_pred", "ref_true"])
        scored["error"] = (scored["pred"] - scored["ref_pred"]) - (
            scored["target"] - scored["ref_true"]
        )
        print(
            f"  stints with a reference: {n_stints_after} of {n_stints_before} "
            f"({n_stints_before - n_stints_after} lost to the gate)"
            if gated
            else f"  stints with a reference: {n_stints_after} of {n_stints_before}"
        )
        return scored

    before = score(scored_bound(gated=False), "BASELINE -- no quality gate (current main)")
    after = score(scored_bound(gated=True), "GATED -- fresh_reference_max_pct_of_fastest")

    if args.out:
        payload = {
            "generated_by": "scripts/measure_fresh_reference_gate.py",
            "years": list(TRAINING_YEARS),
            "gate_threshold": CFG.fresh_reference_max_pct_of_fastest,
            "before": before,
            "after": after,
        }
        args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
