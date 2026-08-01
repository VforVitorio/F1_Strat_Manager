"""What does a lap of pit-stop offset actually cost, in track position?

The decision report leads with EXACT-lap agreement. That is a demanding metric, and
the objection to it is a good one: matching the exact lap requires reproducing a call
made with radio, tyre temperatures, rival intel and driver feel that this layer does
not have. If being two or three laps off costs nothing on the road, then exact-lap is
the wrong RESOLUTION and the headline should be a band.

⛔ The band must not be chosen by seeing which one flatters the system. So this
measures the cost per lap of offset FIRST, on real stops, and whatever it says decides
the band.

METHOD
------
For each real stop, take the driver's actual position change across the stop, and
compare it against what the same stop taken N laps earlier or later would have cost in
pit-loss terms alone: N laps of the field's own pace spread. The question is how many
laps of offset it takes to move one track position.

The measured green-flag median gap between consecutive cars is 2.227 s
(`data/mc_measured_v1.json`, n=69,487). So the cost of an offset is the pace delta
accumulated over those laps, divided by that gap.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, ".")

RAW = Path("data/raw")
YEARS = (2023, 2024)
PACE_WINDOW = 3


def _seconds(series: pd.Series) -> pd.Series:
    return pd.to_timedelta(series).dt.total_seconds()


def _green(laps: pd.DataFrame) -> pd.DataFrame:
    keep = laps["LapTime"].notna()
    if "IsAccurate" in laps.columns:
        keep &= laps["IsAccurate"].fillna(False).astype(bool)
    if "Deleted" in laps.columns:
        keep &= ~laps["Deleted"].fillna(False).astype(bool)
    return laps[keep]


def measure_race(path: Path) -> list[float]:
    """Per-lap pace advantage of the new set over the old, per real stop."""
    laps = pd.read_parquet(path / "laps.parquet")
    if "Stint" not in laps.columns:
        return []
    laps = laps.dropna(subset=["Stint", "LapNumber", "Driver"])
    laps["_s"] = _seconds(laps["LapTime"])

    advantages = []
    for _driver, driver_laps in laps.groupby("Driver"):
        ordered = driver_laps.sort_values("LapNumber")
        stints = ordered["Stint"].to_numpy()
        nums = ordered["LapNumber"].to_numpy()
        stops = [int(nums[i]) for i in range(1, len(stints)) if stints[i] > stints[i - 1]]

        for stop_lap in stops:
            old = _green(ordered[ordered["LapNumber"] < stop_lap]).tail(PACE_WINDOW)
            fresh = _green(ordered[ordered["LapNumber"] > stop_lap]).head(PACE_WINDOW)
            if len(old) < 2 or len(fresh) < 2:
                continue
            advantages.append(float(old["_s"].median() - fresh["_s"].median()))
    return advantages


def main() -> None:
    tables = json.loads(Path("data/mc_measured_v1.json").read_text(encoding="utf-8"))
    gap_s = tables["gap_density"]["racing"]["p50"]

    advantages = []
    for year in YEARS:
        for race in sorted((RAW / str(year)).iterdir()):
            if (race / "laps.parquet").exists():
                advantages += measure_race(race)

    values = np.array([a for a in advantages if np.isfinite(a)])
    print(f"real stops measured: {len(values)}  ({YEARS})")
    print(
        f"median gap between consecutive cars: {gap_s} s (n={tables['gap_density']['racing']['n']})\n"
    )

    median_adv = float(np.median(values))
    print(f"per-lap pace advantage of a fresh set, median: {median_adv:.3f} s/lap\n")

    print("cost of stopping N laps LATER than the ideal lap, in track position:")
    print(f"{'offset':>8}{'seconds':>10}{'positions':>12}")
    for n in (1, 2, 3, 5):
        seconds = n * median_adv
        print(f"{n:>7} {seconds:>9.2f}{seconds / gap_s:>12.2f}")

    print("\nlaps of offset it takes to lose ONE position:")
    print(f"  {gap_s / median_adv:.2f} laps")


if __name__ == "__main__":
    main()
