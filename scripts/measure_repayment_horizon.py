"""E1 — how many laps does a real elective stop take to repay itself?

The constant's basis, and a fact about the sport rather than a fit to any eval. If the
median sits well beyond the 5-lap window, that is the measured proof that no fixed
short horizon can price an elective stop.

Training seasons only (2023-24). 2025 stays held out, the same hygiene the tyre
reference measurement used.

METHOD, stated before running so it cannot be adjusted afterwards
-----------------------------------------------------------------
For each real green-flag stop:
  - the pit loss is the OBSERVED cost: the stop lap's time minus the driver's median
    green-flag lap in the stint it just left. That is what the stop actually cost on
    the road, not a modelled figure.
  - the advantage is the OBSERVED per-lap gain: the median lap in the new stint's
    early window against the median of the old stint's closing laps.
  - the repayment horizon is ceil(pit_loss / advantage), capped at the laps that
    actually remained.

A stop with no positive advantage never repays and is reported as such rather than
dropped, because dropping it would bias the median toward the stops that worked.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, ".")

RAW = Path("data/raw")
TRAINING_YEARS = (2023, 2024)
# Laps either side of the stop used to characterise pace. Three is the shortest window
# that survives a single bad lap; the out-lap itself is excluded on both counts.
PACE_WINDOW = 3


def _seconds(series: pd.Series) -> pd.Series:
    return pd.to_timedelta(series).dt.total_seconds()


def _green(laps: pd.DataFrame) -> pd.DataFrame:
    """Laps clean enough to characterise pace, using N04's own quality gates."""
    keep = laps["LapTime"].notna()
    if "IsAccurate" in laps.columns:
        keep &= laps["IsAccurate"].fillna(False).astype(bool)
    if "Deleted" in laps.columns:
        keep &= ~laps["Deleted"].fillna(False).astype(bool)
    return laps[keep]


def _stop_rows(driver_laps: pd.DataFrame) -> list[int]:
    """Lap numbers where the stint number increases, i.e. a real stop happened."""
    stints = driver_laps["Stint"].to_numpy()
    laps = driver_laps["LapNumber"].to_numpy()
    return [int(laps[i]) for i in range(1, len(stints)) if stints[i] > stints[i - 1]]


def measure_race(path: Path) -> list[dict]:
    laps = pd.read_parquet(path / "laps.parquet")
    if "Stint" not in laps.columns:
        return []
    laps = laps.dropna(subset=["Stint", "LapNumber", "Driver"])
    laps["_s"] = _seconds(laps["LapTime"])

    rows = []
    for driver, driver_laps in laps.groupby("Driver"):
        ordered = driver_laps.sort_values("LapNumber")
        total = int(ordered["LapNumber"].max())
        stops = _stop_rows(ordered)

        for index, stop_lap in enumerate(stops):
            old = _green(ordered[ordered["LapNumber"] < stop_lap]).tail(PACE_WINDOW)
            # Skip the out-lap on the new set; it is not representative pace.
            fresh = _green(ordered[ordered["LapNumber"] > stop_lap]).head(PACE_WINDOW)
            if len(old) < 2 or len(fresh) < 2:
                continue

            old_pace = float(old["_s"].median())
            fresh_pace = float(fresh["_s"].median())
            stop_row = ordered[ordered["LapNumber"] == stop_lap]
            if stop_row.empty or pd.isna(stop_row["_s"].iloc[0]):
                continue

            pit_loss = float(stop_row["_s"].iloc[0]) - old_pace
            advantage = old_pace - fresh_pace
            remaining = total - stop_lap
            if pit_loss <= 0 or remaining <= 0:
                continue

            repays = advantage > 0
            horizon = float(np.ceil(pit_loss / advantage)) if repays else np.inf
            rows.append(
                {
                    "driver": driver,
                    "stop_lap": stop_lap,
                    "elective": index > 0,
                    "pit_loss_s": pit_loss,
                    "advantage_s_per_lap": advantage,
                    "horizon_laps": horizon,
                    "laps_remaining": remaining,
                    "repaid_before_flag": repays and horizon <= remaining,
                }
            )
    return rows


def main() -> None:
    rows = []
    for year in TRAINING_YEARS:
        for race in sorted((RAW / str(year)).iterdir()):
            if (race / "laps.parquet").exists():
                rows += measure_race(race)

    df = pd.DataFrame(rows)
    print(f"stops measured: {len(df)}  ({TRAINING_YEARS})")

    for label, subset in (("ALL", df), ("ELECTIVE", df[df["elective"]])):
        finite = subset[np.isfinite(subset["horizon_laps"])]
        print(f"\n=== {label} (n={len(subset)}) ===")
        print(
            f"  never repays (no positive advantage): {100 * (1 - len(finite) / max(len(subset), 1)):.1f}%"
        )
        if not len(finite):
            continue
        print(f"  pit loss  median {finite['pit_loss_s'].median():.1f} s")
        print(f"  advantage median {finite['advantage_s_per_lap'].median():.3f} s/lap")
        q = finite["horizon_laps"].quantile([0.25, 0.5, 0.75]).round(1).to_dict()
        print(f"  repayment horizon  p25 {q[0.25]}  MEDIAN {q[0.5]}  p75 {q[0.75]} laps")
        boot = [
            np.median(finite["horizon_laps"].sample(len(finite), replace=True, random_state=s))
            for s in range(400)
        ]
        print(f"  median 95% CI: [{np.percentile(boot, 2.5):.1f}, {np.percentile(boot, 97.5):.1f}]")
        print(
            f"  horizon <= 5 laps (what the window prices): {100 * (finite['horizon_laps'] <= 5).mean():.1f}%"
        )
        print(f"  repaid before the flag: {100 * finite['repaid_before_flag'].mean():.1f}%")


if __name__ == "__main__":
    main()
