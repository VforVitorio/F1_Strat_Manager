"""The per-lap error bound for `deg_cost_s` that E4 has been formally unevaluable without.

`deg_cost_s = pred(now) - pred(fresh)`, both from the TCN. Its TRUE counterpart is the
same difference taken on N04's own target column, which is what the model was trained
to reproduce. The error is the difference of the two differences, and its distribution
is the bound.

Training seasons only, so the bound is measured where the model was fitted rather than
on the season it serves -- the conservative direction for a bound.
"""

import sys

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from scripts.measure_tyre_reference import (  # noqa: E402
    FRESH_MAX_TYRE_LIFE,
    predict_training_stints,
)

# The threshold E4's amplification table is quoted at. If the typical error sits well
# under this, the 7.1% flip rate describes a perturbation larger than the model makes.
E4_PERTURBATION_S = 0.1


def main() -> None:
    frame = predict_training_stints().sort_values(["stint", "tyre_life"])
    fresh = frame[frame["tyre_life"] <= FRESH_MAX_TYRE_LIFE]

    # The live path's reference: the prediction at the last lap with life <= 3.
    frame["ref_pred"] = frame["stint"].map(fresh.groupby("stint")["pred"].last())
    frame["ref_true"] = frame["stint"].map(fresh.groupby("stint")["target"].last())

    scored = frame[(frame["tyre_life"] > FRESH_MAX_TYRE_LIFE) & frame["ref_pred"].notna()].copy()
    scored["deg_pred"] = scored["pred"] - scored["ref_pred"]
    scored["deg_true"] = scored["target"] - scored["ref_true"]
    scored["error"] = scored["deg_pred"] - scored["deg_true"]

    absolute = scored["error"].abs()
    print(f"laps: {len(scored)}  (2023-24, tyre life > {FRESH_MAX_TYRE_LIFE})\n")
    print("PER-LAP ERROR of deg_cost_s against its own training target")
    for q in (0.5, 0.75, 0.9, 0.95, 0.99):
        print(f"  |error| p{int(q * 100):<3} {absolute.quantile(q):7.3f} s/lap")
    print(f"  mean absolute  {absolute.mean():7.3f} s/lap")
    print(f"  bias (signed)  {scored['error'].mean():+7.3f} s/lap")
    print()
    print(
        f"  share under the E4 perturbation ({E4_PERTURBATION_S} s/lap): "
        f"{100 * (absolute < E4_PERTURBATION_S).mean():.1f}%"
    )
    print(f"  share under 0.2 s/lap: {100 * (absolute < 0.2).mean():.1f}%")
    print(f"  share under 0.5 s/lap: {100 * (absolute < 0.5).mean():.1f}%")

    print("\n=== by tyre-life band, because the term integrates to the flag ===")
    bands = pd.cut(scored["tyre_life"], [3, 10, 20, 30, 100])
    print(absolute.groupby(bands, observed=True).agg(["median", "mean", "count"]).round(3))

    # A bound that only holds where the liability is small is not a bound. The term
    # multiplies deg by the laps remaining, so the error that matters is weighted by
    # how many laps it is charged over.
    print("\n=== bootstrap 95% CI on the mean absolute error ===")
    boot = [absolute.sample(len(absolute), replace=True, random_state=s).mean() for s in range(400)]
    print(f"  [{np.percentile(boot, 2.5):.3f}, {np.percentile(boot, 97.5):.3f}] s/lap")


if __name__ == "__main__":
    main()
