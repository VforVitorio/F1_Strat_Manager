"""Rebuild the featured-laps artefacts, reproducing N04 rather than improving on it.

WHY A SCRIPT AND NOT THE NOTEBOOK
---------------------------------
`notebooks/**` is read-only by project rule, and more to the point the producer on disk no
longer reproduces its own output. Measured against the shipped `laps_featured_2025.parquet`,
by taking that file's own `GP_Name -> Cluster` mapping and checking each candidate source
against it:

    circuit_clusters_k4.parquet         (what N04 reads today)   17 / 24 GPs agree
    circuit_clusters_k4_2025.parquet    (the pre-11a7ffa wiring)  24 / 24 GPs agree

So a straight re-run of the notebook would move seven of twenty-four races to a different
cluster, and with them `lap_time_vs_cluster_mean` and `mean_sector_speed`. Two further
defects live in the producer and survive any re-run: `finalize_and_save` writes the combined
file from whatever frame it is handed, so the 2025 pass clobbers it to 2025-only, and the
2025 loader's Miami alias never reaches the 2023-24 pass.

This script therefore lifts N04's feature functions VERBATIM — they are the training
contract and must not drift — and replaces only the orchestration around them.

WHAT IT DELIBERATELY DOES NOT DO
--------------------------------
It does not "improve" a feature. Every model in the stack was trained on what N04 produced,
so a better join, a filled gap or a widened filter here is a silent distribution shift
wearing a fix's clothes. Las Vegas keeps its missing `mean_sector_speed` unless the imputed
column is written explicitly and flagged; the season-broadcast `mean_sector_speed` in the
combined file is reproduced, not corrected.

The one thing it adds is what N04 already computed and the published split dropped: the four
weather columns. `augment_featured_laps` restores them at load time today, and
`tests/agents/test_weather_restore.py` asserts this rebuild agrees with that restore.

--- WHERE TO CHANGE IF THE PIPELINE CHANGES ---
The lifted block below is a copy of `.nb_py/N04_feature_engineering.py`. If N04 changes, this
must be re-lifted, and the acceptance diff in `--verify` is what proves the two still agree.
"""

# ruff: noqa: E712, F541
# Both rules fire only inside the lifted block, and both must stay. `== True` on a pandas
# column is NOT the same as a truth check when the column is object dtype carrying NaN, so
# "fixing" it would change which laps survive the baseline filter — the one thing this file
# exists to keep identical. The empty f-strings are N04's, and rewriting them would make the
# block stop being a copy.

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.f1_strat_manager.data_cache import get_data_root  # noqa: E402

# Seasons in the order they are built. 2025 is last because it is the only one whose cluster
# sources differ, and building it separately is what keeps that difference visible.
_TRAINING_SEASONS = (2023, 2024)
_HOLDOUT_SEASON = 2025

_WEATHER_COLUMNS = ("AirTemp", "TrackTemp", "Humidity", "Rainfall")


# ─────────────────────────────────────────────────────────────────────────────
# Lifted verbatim from .nb_py/N04_feature_engineering.py — do not edit here
# ─────────────────────────────────────────────────────────────────────────────

_COLS_TO_DROP = [
    # Raw FastF1 timedeltas — already converted to *_s during this notebook
    "Time",
    "PitOutTime",
    "PitInTime",
    "Sector1Time",
    "Sector2Time",
    "Sector3Time",
    "LapStartTime",
    "LapStartDate",
    "Sector1SessionTime",
    "Sector2SessionTime",
    "Sector3SessionTime",
    "Time_s",
    # Superseded by derived columns
    "LapTime",  # -> LapTime_s
    "TrackStatus",  # -> track_status_clean
    # Zero variance after baseline filter (all True / all False)
    "Deleted",
    "DeletedReason",
    "IsAccurate",
    # Metadata not useful as ML features
    "FastF1Generated",
    "IsPersonalBest",
]


_GP_NAME_ALIASES_2025 = {
    "Miami Gardens": "Miami",
    "Miami_Gardens": "Miami",
}


def load_single_gp(gp_path: Path, year: str, gp_name: str) -> dict:
    try:
        laps_file = gp_path / "laps.parquet"
        intervals_file = gp_path / "intervals.parquet"
        weather_file = gp_path / "weather.parquet"
        pitstops_file = gp_path / "pitstops.parquet"

        if not all(
            [
                laps_file.exists(),
                intervals_file.exists(),
                weather_file.exists(),
                pitstops_file.exists(),
            ]
        ):
            print(f"  WARNING: {gp_name}: Missing files, skipping...")
            return None

        laps_df = pd.read_parquet(laps_file)
        intervals_df = pd.read_parquet(intervals_file)
        weather_df = pd.read_parquet(weather_file)
        pitstops_df = pd.read_parquet(pitstops_file)

        for df in [laps_df, intervals_df, weather_df, pitstops_df]:
            df["GP_Name"] = gp_name
            df["Year"] = int(year)

        return {
            "laps": laps_df,
            "intervals": intervals_df,
            "weather": weather_df,
            "pitstops": pitstops_df,
        }

    except Exception as e:
        print(f"  ERROR: {gp_name}: {str(e)}")
        return None


def combine_master_dataframes(all_laps, all_intervals, all_weather, all_pitstops):
    laps_master = pd.concat(all_laps, ignore_index=True)
    intervals_master = pd.concat(all_intervals, ignore_index=True)
    weather_master = pd.concat(all_weather, ignore_index=True)
    pitstops_master = pd.concat(all_pitstops, ignore_index=True)

    laps_master = laps_master.sort_values(
        ["Year", "GP_Name", "DriverNumber", "LapNumber"]
    ).reset_index(drop=True)

    intervals_master = intervals_master.sort_values(["Year", "GP_Name", "date"]).reset_index(
        drop=True
    )

    return laps_master, intervals_master, weather_master, pitstops_master


def filter_baseline_laps(laps_master):
    """Convert LapTime to seconds and apply baseline quality filters."""
    laps = laps_master.copy()
    laps["LapTime_s"] = laps["LapTime"].dt.total_seconds()

    laps_clean = (
        laps[
            (laps["IsAccurate"] == True)
            & (laps["Deleted"] == False)
            & (laps["LapTime_s"] < 180)
            & (laps["LapNumber"] > 1)
        ]
        .copy()
        .reset_index(drop=True)
    )

    removed = len(laps_master) - len(laps_clean)
    print(f"Original laps : {len(laps_master):,}")
    print(f"Removed       : {removed:,}  ({removed / len(laps_master) * 100:.1f}%)")
    print(f"Clean laps    : {len(laps_clean):,}")
    print(f"\nBy year:")
    print(laps_clean.groupby("Year").size().rename("laps").to_string())
    return laps_clean


def add_fuel_corrected_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add fuel-corrected degradation features to lap-level DataFrame.

    Methodology: legacy N01_tire_prediction.ipynb + N02_model_tire_predictions.ipynb
    Constant: 0.055 s/lap (empirical F1 fuel effect, midpoint of 0.05-0.06 s/lap range)

    Features added:
        FuelEffect              - seconds recovered from fuel burn since stint start
        FuelAdjustedLapTime     - LapTime_s with fuel advantage removed
        FuelAdjustedDegAbsolute - seconds slower vs first stint lap (fuel-corrected)
        FuelAdjustedDegPercent  - % slower vs first stint lap (fuel-corrected)
    """
    FUEL_EFFECT_PER_LAP = 0.055  # s/lap

    result = df.copy()
    result["FuelEffect"] = np.nan
    result["FuelAdjustedLapTime"] = np.nan
    result["FuelAdjustedDegAbsolute"] = np.nan
    result["FuelAdjustedDegPercent"] = np.nan

    groups = result.groupby(["Year", "GP_Name", "DriverNumber", "Stint"], sort=False)

    for name, group in groups:
        if group["LapTime_s"].isna().all():
            continue

        # Baseline: first lap of the stint
        baseline_tyrelife = group["TyreLife"].min()
        baseline_mask = group["TyreLife"] == baseline_tyrelife
        baseline_laptime = group.loc[baseline_mask, "LapTime_s"].mean()

        if pd.isna(baseline_laptime):
            continue

        # Fuel effect: laps since stint start × 0.055 s/lap
        laps_from_baseline = group["TyreLife"] - baseline_tyrelife
        fuel_effect = laps_from_baseline * FUEL_EFFECT_PER_LAP
        adjusted = group["LapTime_s"] + fuel_effect

        result.loc[group.index, "FuelEffect"] = fuel_effect
        result.loc[group.index, "FuelAdjustedLapTime"] = adjusted
        result.loc[group.index, "FuelAdjustedDegAbsolute"] = adjusted - baseline_laptime
        result.loc[group.index, "FuelAdjustedDegPercent"] = (adjusted / baseline_laptime - 1) * 100

    return result


def add_sequential_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add previous-lap and delta features within each driver stint.

    Groups by Year + GP_Name + DriverNumber + Stint, sorted by LapNumber.
    First lap of each stint → NaN (no predecessor exists).
    NaN is left intentionally: XGBoost handles it natively via missing value branch.
    TCN masking handled in model notebooks (v0.4.0).

    Features added: Prev_LapTime, Prev_SpeedI1/I2/FL/ST, Prev_TyreLife,
                    LapTime_Delta, Speed*_Delta, LapTime_Trend
    """
    result = df.copy()
    speed_cols = ["SpeedI1", "SpeedI2", "SpeedFL", "SpeedST"]

    result = result.sort_values(
        ["Year", "GP_Name", "DriverNumber", "Stint", "LapNumber"]
    ).reset_index(drop=True)

    grp = result.groupby(["Year", "GP_Name", "DriverNumber", "Stint"], sort=False)

    # Previous values
    result["Prev_LapTime"] = grp["LapTime_s"].shift(1)
    result["Prev_TyreLife"] = grp["TyreLife"].shift(1)
    for col in speed_cols:
        result[f"Prev_{col}"] = grp[col].shift(1)

    # Deltas
    result["LapTime_Delta"] = result["LapTime_s"] - result["Prev_LapTime"]
    for col in speed_cols:
        result[f"{col}_Delta"] = result[col] - result[f"Prev_{col}"]

    # Trend (second derivative of lap time)
    result["LapTime_Trend"] = grp["LapTime_Delta"].shift(1)
    result["LapTime_Trend"] = result["LapTime_Delta"] - result["LapTime_Trend"]

    # No fillna — NaN on first lap of each stint is meaningful signal.
    return result


def add_degradation_rate_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add rolling degradation rate features using 3-lap linear regression per stint.

    Uses FuelAdjustedLapTime (Step 2) to isolate pure tire degradation.
    Requires minimum 2 points per window (min_periods=2).

    Features added: DegradationRate, CumulativeDeg, DegAcceleration

    NaN policy: lap 1 of each stint has no predecessor window → left as NaN.
    XGBoost handles NaN natively via its missing value branch.
    """
    result = df.copy()
    result["DegradationRate"] = np.nan
    result["CumulativeDeg"] = np.nan
    result["DegAcceleration"] = np.nan

    groups = result.groupby(["Year", "GP_Name", "DriverNumber", "Stint"], sort=False)

    for name, group in groups:
        idx = group.index
        adj_times = group["FuelAdjustedLapTime"].values
        tyre_lives = group["TyreLife"].values
        n = len(group)

        if n < 2:
            continue

        # Rolling 3-lap slope (window=3, min_periods=2)
        deg_rates = np.full(n, np.nan)
        for i in range(1, n):  # start at index 1 (need ≥2 points)
            start = max(0, i - 2)  # up to 3 laps back
            x = tyre_lives[start : i + 1]
            y = adj_times[start : i + 1]
            if len(x) >= 2 and not np.isnan(y).any():
                slope = np.polyfit(x, y, 1)[0]
                deg_rates[i] = slope

        result.loc[idx, "DegradationRate"] = deg_rates

        # Cumulative degradation since stint start
        base = group["FuelAdjustedDegAbsolute"].iloc[0]
        result.loc[idx, "CumulativeDeg"] = (group["FuelAdjustedDegAbsolute"] - base).values

        # Degradation acceleration (change in rate)
        accel = np.full(n, np.nan)
        for i in range(1, n):
            if not np.isnan(deg_rates[i]) and not np.isnan(deg_rates[i - 1]):
                accel[i] = deg_rates[i] - deg_rates[i - 1]
        result.loc[idx, "DegAcceleration"] = accel

    # No fillna — NaN on first lap of each stint is meaningful signal.
    return result


def clip_degradation_outliers(laps_clean, clip_range=(-2.0, 2.0)):
    """Clip DegradationRate and DegAcceleration to a realistic s/lap range."""
    lo, hi = clip_range
    outside = (laps_clean["DegradationRate"] < lo) | (laps_clean["DegradationRate"] > hi)
    print(
        f"DegradationRate outside [{lo}, {hi}] s/lap: "
        f"{outside.sum()} laps  ({outside.mean() * 100:.2f}%)"
    )

    laps = laps_clean.copy()
    laps["DegradationRate"] = laps["DegradationRate"].clip(lo, hi)
    laps["DegAcceleration"] = laps["DegAcceleration"].clip(lo, hi)

    print(f"\nAfter clipping:")
    print(laps["DegradationRate"].describe().to_string())
    return laps


def add_weather_features(laps_df: pd.DataFrame, weather_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge weather conditions to lap-level data using nearest-time join.

    For each race (Year + GP_Name), matches each lap to the closest weather
    sample using pd.merge_asof(direction='nearest') on session Time.

    Features added: AirTemp, TrackTemp, Humidity, Rainfall

    NaN policy: Rainfall filled with 0 (no rain); other weather cols left NaN
    if no weather data available for a race (rare edge case).
    """
    WEATHER_COLS = ["AirTemp", "TrackTemp", "Humidity", "Rainfall"]
    result = laps_df.copy()

    for col in WEATHER_COLS:
        result[col] = np.nan

    for (year, gp), lap_group in result.groupby(["Year", "GP_Name"]):
        wth = (
            weather_df[(weather_df["Year"] == year) & (weather_df["GP_Name"] == gp)][
                ["Time"] + WEATHER_COLS
            ]
            .dropna(subset=["Time"])
            .sort_values("Time")
        )
        if wth.empty:
            continue

        # Sort laps by session time for merge_asof
        laps_sorted = lap_group[["Time"]].sort_values("Time")

        # Nearest-time join: each lap gets the closest weather sample
        merged = pd.merge_asof(laps_sorted, wth, on="Time", direction="nearest")
        merged.index = laps_sorted.index

        for col in WEATHER_COLS:
            result.loc[merged.index, col] = merged[col].values

    # Rainfall: fill NaN as 0 (no rain) and cast to int flag
    result["Rainfall"] = result["Rainfall"].fillna(0).astype(int)

    return result


def add_race_context_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add race context features derivable directly from existing columns.
    No joins or external data required.
    """
    result = df.copy()

    # --- Sector times as float (seconds) ---
    for col, new_col in [
        ("Sector1Time", "Sector1_s"),
        ("Sector2Time", "Sector2_s"),
        ("Sector3Time", "Sector3_s"),
    ]:
        if col in result.columns:
            result[new_col] = result[col].dt.total_seconds()

    # --- Max laps per race ---
    max_laps = result.groupby(["Year", "GP_Name"])["LapNumber"].transform("max")

    # --- Race phase (early / mid / late) ---
    lap_fraction = result["LapNumber"] / max_laps
    result["race_phase"] = pd.cut(
        lap_fraction,
        bins=[0, 0.33, 0.67, 1.0],
        labels=["early", "mid", "late"],
        include_lowest=True,
    )

    # --- Laps remaining ---
    result["laps_remaining"] = (max_laps - result["LapNumber"]).astype(int)

    # --- Track status simplified ---
    # FastF1 codes: 1=clear, 2=yellow, 3=SC deployed, 4=SC ending,
    #               5=red flag, 6=VSC deployed, 7=VSC ending
    status_map = {1: 0, 2: 1, 3: 2, 4: 2, 5: 2, 6: 1, 7: 1}
    result["track_status_clean"] = result["TrackStatus"].map(status_map).fillna(0).astype(int)

    return result


def add_cluster_features(
    df: pd.DataFrame, clusters: pd.DataFrame, circuit_features: pd.DataFrame
) -> pd.DataFrame:
    """
    Merge N03 circuit clustering artifacts into lap-level data.

    Features added:
        Cluster                  - Circuit archetype (0-3) from K-Means
        lap_time_vs_cluster_mean - Delta from cluster mean lap time (s)
        mean_sector_speed        - Circuit-level mean sector speed (km/h)
    """
    result = df.copy()

    # Cluster mean lap time (from circuit_features)
    cluster_mean = (
        circuit_features.groupby("Cluster")["mean_laptime"]
        .mean()
        .reset_index()
        .rename(columns={"mean_laptime": "cluster_mean_laptime"})
    )

    # Merge Cluster + mean_sector_speed on GP_Name
    circuit_lookup = clusters.merge(
        circuit_features[["GP_Name", "mean_sector_speed"]], on="GP_Name", how="left"
    )
    result = result.merge(circuit_lookup, on="GP_Name", how="left")

    # Merge cluster mean lap time on Cluster
    result = result.merge(cluster_mean, on="Cluster", how="left")

    # Delta from cluster mean
    result["lap_time_vs_cluster_mean"] = result["LapTime_s"] - result["cluster_mean_laptime"]

    # Drop helper column
    result = result.drop(columns=["cluster_mean_laptime"])

    return result


def fix_spain_cluster_artefact(laps_clean, circuit_clusters, circuit_features):
    """
    Map the 'Spain' duplicate (N01 test run) to the correct 'Barcelona' cluster values.
    The cluster assignment, mean_sector_speed, and lap_time_vs_cluster_mean are patched
    to match Barcelona so downstream models see consistent data.
    """
    laps = laps_clean.copy()
    spain_mask = laps["GP_Name"] == "Spain"
    print(f"Spain laps without cluster: {spain_mask.sum()}")

    if spain_mask.sum() > 0:
        barcelona_cluster = circuit_clusters.loc[
            circuit_clusters["GP_Name"] == "Barcelona", "Cluster"
        ].iloc[0]
        barcelona_speed = circuit_features.loc[
            circuit_features["GP_Name"] == "Barcelona", "mean_sector_speed"
        ].iloc[0]
        cluster_mean_barcelona = circuit_features.groupby("Cluster")["mean_laptime"].mean()[
            barcelona_cluster
        ]

        laps.loc[spain_mask, "Cluster"] = barcelona_cluster
        laps.loc[spain_mask, "mean_sector_speed"] = barcelona_speed
        laps.loc[spain_mask, "lap_time_vs_cluster_mean"] = (
            laps.loc[spain_mask, "LapTime_s"] - cluster_mean_barcelona
        )

    print(f"Nulls remaining in Cluster : {laps['Cluster'].isna().sum()}")
    print(f"Total laps with cluster    : {laps['Cluster'].notna().sum():,}")
    return laps


def add_temporal_normalization_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add lap_time_pct_of_race_fastest: ratio of each lap time to the
    fastest clean lap in the same race (Year + GP_Name).

    Formula: LapTime_s / min(LapTime_s per race)

    Purpose: normalizes inter-year pace drift (~0.5–1.5 s/lap/year on same circuit)
    so the model can compare equivalent race pace across seasons without
    conflating year-over-year car development with tire degradation signal.

    References:
    - Liu et al. (2023) arXiv:2304.01512 §4.1 — relative intra-series normalization
      reduces GFM error under gradual concept drift up to 8%.
    - concept_drift_strategy.md — Acción 1
    """
    result = df.copy()

    race_fastest = result.groupby(["Year", "GP_Name"])["LapTime_s"].transform("min")

    result["lap_time_pct_of_race_fastest"] = result["LapTime_s"] / race_fastest

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Orchestration — the part that replaces N04's, and the only part that differs
# ─────────────────────────────────────────────────────────────────────────────


def _load_season(raw_root: Path, year: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Every race directory of ONE season, combined the way N04's loader combines them.

    Per season rather than N04's glob over every year, because the two eras take different
    cluster sources and building them together is what hid that difference.
    """
    season_dir = raw_root / str(year)
    if not season_dir.is_dir():
        raise SystemExit(f"no raw directory for {year}: {season_dir}")

    laps, intervals, weather, pitstops = [], [], [], []
    for race_dir in sorted(p for p in season_dir.glob("*") if p.is_dir()):
        race = load_single_gp(race_dir, str(year), race_dir.name.replace("_", " "))
        if race is None:
            continue
        laps.append(race["laps"])
        intervals.append(race["intervals"])
        weather.append(race["weather"])
        pitstops.append(race["pitstops"])

    if not laps:
        raise SystemExit(f"no readable races under {season_dir}")

    laps_master, _intervals, weather_master, _pitstops = combine_master_dataframes(
        laps, intervals, weather, pitstops
    )
    return laps_master, weather_master


def _finalize_frame(laps_clean: pd.DataFrame) -> pd.DataFrame:
    """N04's `finalize_and_save` column drop and dtype cast, WITHOUT its writing.

    Its writer is where the combined artefact gets clobbered: it derives the combined file
    and the per-year splits from whatever frame it is given, so N04's 2025 pass rewrote the
    three-season file with one season. Here the writing is the caller's job and the combined
    file is a concat of the parts, which cannot disagree with them by construction.
    """
    droppable = [column for column in _COLS_TO_DROP if column in laps_clean.columns]
    laps_featured = laps_clean.drop(columns=droppable)
    laps_featured["Cluster"] = laps_featured["Cluster"].astype(int)
    laps_featured["DriverNumber"] = laps_featured["DriverNumber"].astype(int)
    return laps_featured


def build_season(
    year: int,
    raw_root: Path,
    clustering_root: Path,
    *,
    drop_spain: bool = False,
) -> pd.DataFrame:
    """One season, through N04's own feature functions in N04's own order."""
    laps, weather = _load_season(raw_root, year)

    # The Miami rename is N04's, and it applies to the 2025 loader only: the raw folder is
    # `Miami_Gardens` that season and `Miami` in the two before it, so without this the
    # combined file carries the same circuit under two names.
    #
    # THE WEATHER FRAME TOO, which N04 does not do (`_load_raw_2025` renames `laps` and
    # returns `weather` untouched). `add_weather_features` merges per GP_Name, so with only
    # one side renamed Miami matches nothing and the whole race comes out with no weather.
    # Caught by running the real simulation: the regenerated file had 857 NaN AirTemp rows
    # where the runtime restore had none, and one Lusail... one MIAMI lap changed its call.
    #
    # This is the one place the script knowingly diverges from N04, and it can: the weather
    # columns were never in the published artefact, so there is no trained-on value to
    # preserve. Their correct value is what `augment_featured_laps` restores at load time,
    # which is what every model has actually been fed, and that is what this now matches.
    if year == _HOLDOUT_SEASON:
        laps["GP_Name"] = laps["GP_Name"].replace(_GP_NAME_ALIASES_2025)
        weather["GP_Name"] = weather["GP_Name"].replace(_GP_NAME_ALIASES_2025)

    laps = filter_baseline_laps(laps)
    laps = add_fuel_corrected_features(laps)
    laps = add_sequential_features(laps)
    laps = add_degradation_rate_features(laps)
    laps = clip_degradation_outliers(laps)
    laps = add_weather_features(laps, weather)
    laps = add_race_context_features(laps)

    # THE SOURCE SPLIT, and the reason this script exists. Measured against the shipped
    # artefact's own GP->Cluster mapping: the 2025 file agrees with the 2025 sources on
    # 24/24 races and with the pooled ones on 17/24. N04 reads the pooled files for both
    # eras today, which is the post-11a7ffa wiring and not what produced what is published.
    suffix = "_2025" if year == _HOLDOUT_SEASON else ""
    clusters = pd.read_parquet(clustering_root / f"circuit_clusters_k4{suffix}.parquet")
    features = pd.read_parquet(
        clustering_root / f"circuit_features_with_clusters_k4{suffix}.parquet"
    )
    laps = add_cluster_features(laps, clusters, features)

    # The 2023 duplicate. Patched to Barcelona's values while it exists, per N04; skipped
    # when it is being removed, since patching a race that is about to be dropped only
    # makes the drop harder to verify.
    if year == 2023 and not drop_spain:
        laps = fix_spain_cluster_artefact(laps, clusters, features)
    elif year == 2023 and drop_spain:
        before = len(laps)
        laps = laps[laps["GP_Name"] != "Spain"].reset_index(drop=True)
        removed = before - len(laps)
        # Says what happened, not what was asked for. Once the raw directory is gone the
        # filter has nothing to remove, and a line reading "Spain dropped: 20,908 -> 20,908"
        # claims an action it did not perform — which is how a no-op gets read as a success.
        if removed:
            print(f"Spain dropped: {before:,} -> {len(laps):,} rows ({removed:,} removed)")
        else:
            print("Spain: no rows to drop (the raw directory is already gone)")

    laps = add_temporal_normalization_features(laps)
    return _finalize_frame(laps)


_IMPUTED_FLAG = "mean_sector_speed_imputed"


def _circuit_trap_means(raw_root: Path, gp_name: str, year: int) -> tuple[float, float]:
    """That race's three-trap and two-trap mean speeds, by N03's own rule.

    N03 builds `mean_sector_speed` as the row-wise mean of SpeedI1, SpeedI2 and SpeedFL over
    the laps where ALL THREE are present, averaged per GP
    (`.nb_py/N03_circuit_clustering.py:689-694`). The two-trap figure drops I2 from both the
    filter and the mean, and is what remains when that sensor is gone.

    Through `laps_augment._raw_race_dir`, not `gp_name.replace(" ", "_")`. The first version
    of this used the naive form, which resolves `Marina Bay` and fails on the circuits that
    were RENAMED on disk — `Miami` lives in `Miami_Gardens` for 2025. Dormant, because Las
    Vegas is the only hole today and its name happens to be the easy case; it would have
    raised FileNotFoundError the first time a renamed circuit lost a trap. That resolver
    already exists and already carries the three forms, which is the whole reason not to
    write a fourth one here.
    """
    from src.f1_strat_manager.laps_augment import _raw_race_dir

    laps = pd.read_parquet(_raw_race_dir(raw_root.parent, year, gp_name) / "laps.parquet")
    three = laps.dropna(subset=["SpeedI1", "SpeedI2", "SpeedFL"])
    two = laps.dropna(subset=["SpeedI1", "SpeedFL"])
    return (
        three[["SpeedI1", "SpeedI2", "SpeedFL"]].mean(axis=1).mean() if len(three) else np.nan,
        two[["SpeedI1", "SpeedFL"]].mean(axis=1).mean() if len(two) else np.nan,
    )


def impute_circuit_speed(frame: pd.DataFrame, raw_root: Path) -> pd.DataFrame:
    """Fill a circuit's missing `mean_sector_speed` from its own trap-offset, and SAY SO.

    FastF1 has no SpeedI2 reading for the whole 2025 Las Vegas race — 0% of 886 raw laps,
    against 80% for I1 and 97% for FL — so N03's all-three-traps filter yields zero valid
    laps and the circuit's speed comes out NaN on all 760 featured rows. It is the only hole
    of its shape in 71 races, and no re-run can recover it: the reading does not exist.

    THE ESTIMATOR, AND WHY THIS ONE
    -------------------------------
    The missing trap's contribution is a property of the TRACK LAYOUT, which is stable across
    seasons, while the speeds themselves are season-true. So: take the season's own two-trap
    mean and add the (three-trap minus two-trap) gap measured at THAT circuit in its other
    seasons.

    Scored leave-era-out over every circuit-season that has a real three-trap value — hide
    it, impute, compare:

        offset from the circuit's own other seasons   MAE 1.22 km/h   p95 3.40   n=68
        offset averaged across all circuits           MAE 9.44 km/h   p95 20.82  n=70

    The second is the same idea with the layout term thrown away, and it is nearly eight
    times worse. That difference IS the argument for the first.

    Las Vegas 2025: two-trap mean 245.977, its own offset −13.150 → **232.83 km/h**.

    (An earlier audit reported 239.14 for this. Neither reading of its stated method
    reproduces that: the circuit's own offset gives 232.83 and a global offset 241.54. The
    value here is the one the validation above actually scores.)

    NEVER SILENTLY. Every row carries `mean_sector_speed_imputed`, because a fabricated
    number that looks like a measurement is how a model ends up trained on one.
    """
    result = frame.copy()
    if _IMPUTED_FLAG not in result.columns:
        result[_IMPUTED_FLAG] = False

    for (year, gp), group in result.groupby(["Year", "GP_Name"], sort=False):
        if group["mean_sector_speed"].notna().any():
            continue

        from src.f1_strat_manager.laps_augment import _raw_race_dir

        seasons = sorted(p.name for p in raw_root.iterdir() if p.is_dir() and p.name.isdigit())
        offsets = []
        for other in seasons:
            if int(other) == int(year):
                continue
            if not (_raw_race_dir(raw_root.parent, int(other), gp) / "laps.parquet").exists():
                continue
            three, two = _circuit_trap_means(raw_root, gp, int(other))
            if pd.notna(three) and pd.notna(two):
                offsets.append(three - two)

        _three_now, two_now = _circuit_trap_means(raw_root, gp, int(year))
        if not offsets or pd.isna(two_now):
            print(f"  {gp} {year}: no other season to take an offset from; left NaN")
            continue

        imputed = float(two_now + np.mean(offsets))
        rows = (result["Year"] == year) & (result["GP_Name"] == gp)
        result.loc[rows, "mean_sector_speed"] = imputed
        result.loc[rows, _IMPUTED_FLAG] = True
        print(
            f"  {gp} {year}: two-trap {two_now:.3f} + offset {np.mean(offsets):+.3f} "
            f"-> {imputed:.2f} km/h over {int(rows.sum()):,} rows (FLAGGED)"
        )
    return result


def _compare(new: pd.DataFrame, old: pd.DataFrame, label: str) -> bool:
    """Value diff against the shipped file: the anti-'silently worse' gate.

    Reports the columns the rebuild ADDS separately from the ones it CHANGES, because the
    weather four are expected to be new and anything else changing is the failure this gate
    exists to catch.
    """
    added = [c for c in new.columns if c not in old.columns]
    removed = [c for c in old.columns if c not in new.columns]
    print(f"\n--- {label} ---")
    print(f"  rows      {len(old):,} -> {len(new):,}")
    print(f"  added     {added}")
    print(f"  removed   {removed}")

    if len(new) != len(old):
        print("  rows differ; per-cell diff skipped (expected only when Spain is dropped)")
        return not removed

    differing = {}
    for column in old.columns:
        if column not in new.columns:
            continue
        left = old[column].reset_index(drop=True)
        right = new[column].reset_index(drop=True)
        if pd.api.types.is_numeric_dtype(left) and pd.api.types.is_numeric_dtype(right):
            # rtol=0: `np.isclose`'s default relative term is 1e-5, which on a lap time
            # near 100 s tolerates a millisecond and on an elapsed `Time_s` near 9,000 s
            # tolerates nine hundredths. An absolute bound is what "value-identical" means
            # here, and leaving the default in place would have made this gate looser
            # exactly where the numbers are biggest.
            unequal = ~np.isclose(
                pd.to_numeric(left, errors="coerce"),
                pd.to_numeric(right, errors="coerce"),
                rtol=0.0,
                atol=1e-6,
                equal_nan=True,
            )
        else:
            unequal = (left.astype(str) != right.astype(str)) & ~(left.isna() & right.isna())
        count = int(unequal.sum())
        if count:
            differing[column] = count

    if differing:
        print(f"  CHANGED   {differing}")
    else:
        print("  CHANGED   none - every shipped column reproduces to 1e-6")
    return not differing and not removed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--verify",
        action="store_true",
        help="rebuild in memory and diff against the files on disk, writing nothing",
    )
    parser.add_argument(
        "--drop-spain",
        action="store_true",
        help="remove the duplicated 2023 Spanish GP (one-way; the raw folder is a copy of "
        "Barcelona, OpenF1 session 9102)",
    )
    parser.add_argument(
        "--impute-circuit-speed",
        action="store_true",
        help="fill a circuit whose mean_sector_speed is entirely missing from its own "
        "trap-offset, flagging every affected row (2025 Las Vegas is the only one)",
    )
    args = parser.parse_args()

    data_root = get_data_root()
    raw_root = data_root / "raw"
    processed_root = data_root / "processed"
    clustering_root = processed_root / "circuit_clustering"

    seasons = {}
    for year in (*_TRAINING_SEASONS, _HOLDOUT_SEASON):
        print(f"\n{'=' * 70}\nBuilding {year}\n{'=' * 70}")
        seasons[year] = build_season(year, raw_root, clustering_root, drop_spain=args.drop_spain)

    # AFTER the seasons are built and BEFORE anything is compared or written, so the
    # untouched rebuild is what Gate A diffs and the imputation is a visible, opt-in step
    # rather than something the regeneration does on the way past.
    if args.impute_circuit_speed:
        print(f"\n{'=' * 70}\nImputing absent circuit speeds\n{'=' * 70}")
        seasons = {year: impute_circuit_speed(frame, raw_root) for year, frame in seasons.items()}

    combined = pd.concat(seasons.values(), ignore_index=True)

    reproduces = True
    for year, frame in seasons.items():
        shipped = processed_root / f"laps_featured_{year}.parquet"
        if shipped.exists():
            reproduces &= _compare(frame, pd.read_parquet(shipped), f"laps_featured_{year}.parquet")
    shipped_combined = processed_root / "laps_featured.parquet"
    if shipped_combined.exists():
        reproduces &= _compare(combined, pd.read_parquet(shipped_combined), "laps_featured.parquet")

    missing_weather = [c for c in _WEATHER_COLUMNS if c not in combined.columns]
    if missing_weather:
        print(f"\nFAIL: the rebuild did not produce {missing_weather}")
        return 1

    # THE GATE GATES, in both modes. It used to report the diff and then write anyway, and
    # `--drop-spain` returned 0 unconditionally — so the one run that changes rows on
    # purpose could never fail, which is the run where an unnoticed second change is most
    # likely. When rows change deliberately the per-cell diff is not comparable, so the
    # check that still applies is the one about columns: nothing may be LOST.
    rows_change_on_purpose = args.drop_spain
    acceptable = reproduces or rows_change_on_purpose
    if not acceptable:
        print("\nFAIL: a shipped column changed value and no row-count change explains it.")
        print("      Nothing was written. Diff the columns listed above against the backups.")
        return 1

    if args.verify:
        print("\n--verify: nothing written.")
        return 0

    for year, frame in seasons.items():
        target = processed_root / f"laps_featured_{year}.parquet"
        frame.to_parquet(target, index=False)
        print(f"wrote {target.name}  ({len(frame):,} rows, {frame.shape[1]} cols)")
    combined.to_parquet(shipped_combined, index=False)
    print(f"wrote {shipped_combined.name}  ({len(combined):,} rows, {combined.shape[1]} cols)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
