"""Measure what a fresh-tyre reference for the TCN's own predictions is worth.

The tyre channel needs *seconds per lap that staying out costs versus a fresh set*.
The TCN supplies ``cumulative_deg_s`` — N04's ``FuelAdjustedDegAbsolute``, measured
against **this stint's own baseline lap**. Turning that level into a cost needs a
reference for zero, and #744 proposed a pooled per-compound one measured on the
model's own scale.

This script measures whether that reference is worth building, by scoring three
candidates against the two criteria #744 set (non-negative on the large majority of
laps, monotonic by tyre-life band) plus a rank correlation with tyre life:

* ``pooled``      — one median per compound, the artefact #744a proposed
* ``stint_first`` — the model's prediction on this stint's first lap
* ``stint_le3``   — the median of this stint's predictions at tyre life <= 3
* ``stint_live``  — the prediction at the LAST lap with tyre life <= 3, which is the
  only one the live path can produce: ``_get_driver_stint(driver, 3)`` returns the
  whole prefix and ``_build_stint_tensor`` predicts from it once. It is what #744b
  ships, so it is measured rather than assumed equivalent to ``stint_le3``.

Training seasons only (2023-24). ``src/strategy/eval/hygiene.py`` documents why a
constant fitted on the 2025 test season would repeat a leak this project already paid
for once.

--- WHERE TO CHANGE IF THE MODEL CHANGES ---
The transform below (bundle scaler, then N09's left-ZERO-pad) mirrors
``TireAgent._build_stint_tensor``. If that changes, this changes with it, or the
reference lands on a different scale than the value it is subtracted from — the exact
defect CLAUDE.md records for 2026-07-27. ``--self-check`` is the guard: it correlates
predictions against the training target and fails below the threshold.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.agents.tire_agent import TireDegTCN  # noqa: E402

MODEL_DIR = Path("data/models/tire_degradation")
LAPS_PATH = Path("data/processed/laps_tiredeg.parquet")
TRAINING_YEARS = (2023, 2024)
STINT_KEYS = ["Year", "GP_Name", "DriverNumber", "Stint"]
TARGET = "FuelAdjustedDegAbsolute"

# The band that stands in for "fresh". N04's baseline is the stint's lowest-tyre-life
# lap, which is an out-lap or a standing start, so anything tighter than this leaves
# the reference resting on a single cold lap.
FRESH_MAX_TYRE_LIFE = 3

# Below this, the transform above no longer reproduces the model and every number the
# script prints is measuring the harness rather than the reference.
SELF_CHECK_MIN_CORR = 0.90

TYRE_LIFE_BANDS = [3, 5, 10, 15, 20, 25, 100]


def load_bundles() -> dict[str, dict]:
    """Load one bundle per compound, sharing the file when the routing does.

    Five of the six compounds route to the same global model, so loading per compound
    key without the cache would instantiate it five times.
    """
    routing = json.loads((MODEL_DIR / "routing_config.json").read_text(encoding="utf-8"))
    by_file: dict[str, dict] = {}
    by_compound: dict[str, dict] = {}
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
    """One scaled (window, n_features) sequence per lap, each from that lap's prefix.

    Every feature in the manifest is row-local or backward-looking — verified by
    AST-scanning the ten ``_add_*`` builders in ``tire_agent`` for frame aggregates —
    so slicing the stint's precomputed features gives the same rows that recomputing
    them over the prefix would. That equivalence is what lets this run from the
    training parquet instead of reloading every FastF1 session.
    """
    window = bundle["window"]
    raw = stint[bundle["feature_names"]].astype(float).fillna(0)
    scaled = bundle["scaler"].transform(raw)

    sequences = []
    for length in range(1, len(scaled) + 1):
        prefix = scaled[:length]
        if len(prefix) >= window:
            sequences.append(prefix[-window:])
            continue
        # N09's `_pad_or_truncate`, verbatim: zeros, never a tiled first lap (#443).
        padding = np.zeros((window - len(prefix), prefix.shape[1]), dtype=prefix.dtype)
        sequences.append(np.concatenate([padding, prefix], axis=0))
    return np.stack(sequences)


def predict_training_stints() -> pd.DataFrame:
    """Run every training-season stint through its compound's TCN, lap by lap."""
    laps = pd.read_parquet(LAPS_PATH)
    training = laps[laps["Year"].isin(TRAINING_YEARS)]
    bundles = load_bundles()

    rows = []
    for keys, stint in training.groupby(STINT_KEYS, sort=False):
        absolute_id = stint["AbsoluteCompoundID"].iloc[0]
        if pd.isna(absolute_id):
            continue
        compound = f"C{int(absolute_id)}"
        if compound not in bundles:
            continue

        bundle = bundles[compound]
        ordered = stint.sort_values("TyreLife")
        sequences = stint_sequences(ordered, bundle)
        with torch.no_grad():
            batch = torch.tensor(sequences, dtype=torch.float32)
            predictions = bundle["model"](batch).numpy().reshape(-1)

        stint_id = "|".join(str(k) for k in keys)
        for tyre_life, prediction, target in zip(
            ordered["TyreLife"].to_numpy(), predictions, ordered[TARGET].to_numpy()
        ):
            rows.append((stint_id, compound, float(tyre_life), float(prediction), float(target)))

    return pd.DataFrame(rows, columns=["stint", "compound", "tyre_life", "pred", "target"])


def self_check(predictions: pd.DataFrame) -> float:
    """Correlation against the training target — the guard on the whole harness.

    A transform that no longer reproduces the model still returns numbers, and every
    reference measured from them would look plausible. This is the one check that
    fails loudly instead.
    """
    correlation = predictions["pred"].corr(predictions["target"])
    if correlation < SELF_CHECK_MIN_CORR:
        raise RuntimeError(
            f"predictions correlate {correlation:.3f} with the training target, below "
            f"{SELF_CHECK_MIN_CORR}: the tensor transform no longer reproduces the model, "
            "so every reference below would be measured on the wrong scale"
        )
    return correlation


def add_candidate_references(predictions: pd.DataFrame) -> pd.DataFrame:
    """Attach the three candidate references as columns, one per proposal."""
    scored = predictions.sort_values(["stint", "tyre_life"]).copy()
    fresh = scored[scored["tyre_life"] <= FRESH_MAX_TYRE_LIFE]

    scored["ref_pooled"] = scored["compound"].map(fresh.groupby("compound")["pred"].median())
    scored["ref_stint_first"] = scored["stint"].map(scored.groupby("stint")["pred"].first())
    scored["ref_stint_le3"] = scored["stint"].map(fresh.groupby("stint")["pred"].median())
    scored["ref_stint_live"] = scored["stint"].map(fresh.groupby("stint")["pred"].last())
    return scored


def score_candidate(scored: pd.DataFrame, reference: str | None) -> dict:
    """The two acceptance criteria plus a rank correlation, for one reference.

    Spearman rather than Pearson because the quantity has a heavy tail — the training
    target itself reaches -65 s/lap — and a handful of stints whose baseline lap was a
    Safety Car or an out-lap would otherwise decide the number.
    """
    wear = scored["pred"] if reference is None else scored["pred"] - scored[reference]
    bands = pd.cut(scored["tyre_life"], TYRE_LIFE_BANDS)
    by_band = wear.groupby(bands, observed=True).median()

    return {
        "non_negative_pct": round(100 * float((wear >= 0).mean()), 1),
        "spearman": round(float(spearmanr(wear, scored["tyre_life"]).statistic), 3),
        "pearson": round(float(wear.corr(scored["tyre_life"])), 3),
        "median": round(float(wear.median()), 3),
        "monotonic_bands": bool(by_band.is_monotonic_increasing),
        "band_medians": {str(k): round(float(v), 3) for k, v in by_band.items()},
        # The bound a consumer needs, measured rather than chosen. The raw quantity
        # reaches +-15 s/lap on real laps because a handful of stints have a Safety Car
        # or an out-lap as their N04 baseline, and a scorer fed one of those prices a
        # single lap like ten positions. Do NOT bound this at CLIFF_LOSS = 0.80: the
        # measured median at 20-25 laps of tyre life is already above it, so that would
        # delete the signal instead of the outliers.
        "p1": round(float(wear.quantile(0.01)), 3),
        "p99": round(float(wear.quantile(0.99)), 3),
    }


def report(results: dict, n_laps: int, correlation: float) -> str:
    """A copy-pasteable summary, ordered so the baseline reads last."""
    lines = [
        f"laps scored: {n_laps}  (tyre life > {FRESH_MAX_TYRE_LIFE}, seasons {TRAINING_YEARS})",
        f"harness self-check: corr(pred, target) = {correlation:.3f}",
        "",
        f"{'reference':<16}{'non-neg':>9}{'spearman':>10}{'monotone':>10}{'p1':>8}{'p99':>8}",
    ]
    for name, scores in results.items():
        lines.append(
            f"{name:<16}{scores['non_negative_pct']:>8.1f}%{scores['spearman']:>10.3f}"
            f"{str(scores['monotonic_bands']):>10}{scores['p1']:>8.2f}{scores['p99']:>8.2f}"
        )
    lines += [
        "",
        "`monotone` is one of #744's two acceptance criteria and it reads False on the",
        "training seasons for EVERY candidate, including having no reference at all: the",
        "(25, 100] band dips because stints that reach 25 laps are the low-degradation ones,",
        "so that band draws from a different population. It reads True on 2025. The column is",
        "printed rather than argued away, because an earlier version of this script dropped it",
        "in the same commit that shipped a candidate it reads False for.",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, help="write the measured scores to this JSON path")
    args = parser.parse_args()

    predictions = predict_training_stints()
    correlation = self_check(predictions)

    scored = add_candidate_references(predictions)
    past_fresh = scored["tyre_life"] > FRESH_MAX_TYRE_LIFE
    has_stint_reference = scored["ref_stint_le3"].notna()
    usable = scored[past_fresh & has_stint_reference]

    results = {
        "pooled": score_candidate(usable, "ref_pooled"),
        "stint_first": score_candidate(usable, "ref_stint_first"),
        "stint_le3": score_candidate(usable, "ref_stint_le3"),
        "stint_live": score_candidate(usable, "ref_stint_live"),
        "none": score_candidate(usable, None),
    }

    summary = report(results, len(usable), correlation)
    print(summary)

    if args.out:
        payload = {
            "generated_by": "scripts/measure_tyre_reference.py",
            "years": list(TRAINING_YEARS),
            "laps_scored": len(usable),
            "self_check_corr": round(correlation, 4),
            "fresh_max_tyre_life": FRESH_MAX_TYRE_LIFE,
            "candidates": results,
        }
        args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
