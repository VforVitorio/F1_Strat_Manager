"""Measure how much the pooled 2023-2025 k-means fit moves a circuit's cluster.

`circuit_cluster` is a k=4 bucket fit in N03 over per-circuit OUTCOME aggregates
(mean_laptime, degradation_rate, mean_sector_speed), and the fit pooled all three
seasons, so the label a 2025 race is served carries test-season information. The
question this script answers is how COARSE that leak is in practice: refit k-means
on 2023-2024 only and count how many of the 24 circuits land in a different bucket.

Two labellings are deployed and different models consume them, so the answer is two
numbers, not one:

    lookup   the fit's own label for the circuit, which is how tire, overtake and
             safety car read it (circuit_clusters_k4.parquet)
    predict  the frozen fit applied to that season's aggregates, which is how the
             pace model's labels were made (circuit_clusters_k4_2025.parquet)

Nothing is retrained and no notebook runs. N03's extractors are copied rather than
imported: importing .nb_py/N03 executes save_clustering_artifacts at module level,
which would overwrite the deployed artefacts with a fresh fit.

Contents:
    load_years / build_features   N03's loader and feature merge, over explicit seasons
    fit_clean                     StandardScaler + KMeans over one year set only
    compare_labellings            Hungarian-matched flip count and ARI between two labellings
    deployed_labellings           the two label sets that are served today
    main                          the comparisons, the lap weighting and the seed sweep

--- WHERE TO CHANGE IF X CHANGES ---
The extractors are a verbatim copy of N03 cells 2.1 to 3.1. If that notebook's
feature set changes this file drifts silently and the numbers quoted in
src/strategy/eval/hygiene.py stop describing the shipped model. The two parquets
read here are written by N03 Step 5 (pooled) and Step 7 (2025).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_PATH = REPO_ROOT / "data" / "raw"
CLUSTERING_PATH = REPO_ROOT / "data" / "processed" / "circuit_clustering"
FEATURED_2025 = REPO_ROOT / "data" / "processed" / "laps_featured_2025.parquet"

TRAIN_YEARS = (2023, 2024)
TEST_YEAR = 2025
N_CLUSTERS = 4
FIT_SEED = 42
SEED_SWEEP = range(10)

# N03 renames the 2025 folder to the 2023-24 spelling at Step 7 only, so the pooled
# fit at Step 1 carries both spellings as two separate circuits.
GP_NAME_ALIASES = {"Miami Gardens": "Miami"}

# N03:740 drops these two before scaling: min_laptime is collinear with mean_laptime
# and stint_variance is NaN-heavy.
DROPPED_FEATURES = ["min_laptime", "stint_variance"]

# n_laps is a lap COUNT and max_stint_length a MAXIMUM over the fitted seasons, so
# both scale with how many seasons a circuit contributed. A one-season 2025 row reads
# about two sigma low on n_laps through any multi-season scaler, which moves clusters
# for a reason that is not the leak, so the predict comparison also runs without them.
SEASON_COUNT_FEATURES = ["n_laps", "max_stint_length"]


# ---------------------------------------------------------------------------
# N03 extractors, copied verbatim from .nb_py/N03_circuit_clustering.py
# ---------------------------------------------------------------------------


def load_single_gp(gp_path: Path, year: int, gp_name: str) -> dict[str, pd.DataFrame] | None:
    """Read one GP directory's four parquets, or None when any of them is missing.

    Args:
        gp_path: the directory holding the laps, intervals, weather and pitstops parquets.
        year: the season the directory belongs to, stamped onto every frame.
        gp_name: the circuit name, the folder name with underscores as spaces.

    Returns:
        The four frames, each carrying GP_Name and Year, or None when the directory
        is not a complete GP, which is how the radio corpus directory is skipped.
    """
    files = {k: gp_path / f"{k}.parquet" for k in ("laps", "intervals", "weather", "pitstops")}
    if not all(f.exists() for f in files.values()):
        return None
    frames = {k: pd.read_parquet(f) for k, f in files.items()}
    for frame in frames.values():
        frame["GP_Name"] = gp_name
        frame["Year"] = int(year)
    return frames


def load_years(years: tuple[int, ...]) -> tuple[int, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Concatenate laps, weather and pitstops across an explicit set of seasons.

    N03 globs every directory under data/raw/, which also visits radio_audio/. The
    seasons are named explicitly here so the year set is the experiment's variable
    rather than whatever the directory happens to hold.

    Args:
        years: the seasons to read.

    Returns:
        The GP count actually loaded and the three concatenated frames, sorted the
        way N03 sorts them so the groupby aggregates come out identical.
    """
    laps_parts, weather_parts, pit_parts = [], [], []
    loaded = 0
    for year in years:
        for gp_path in sorted((RAW_DATA_PATH / str(year)).glob("*")):
            if not gp_path.is_dir():
                continue
            frames = load_single_gp(gp_path, year, gp_path.name.replace("_", " "))
            if frames is None:
                continue
            laps_parts.append(frames["laps"])
            weather_parts.append(frames["weather"])
            pit_parts.append(frames["pitstops"])
            loaded += 1
    laps = pd.concat(laps_parts, ignore_index=True).sort_values(
        ["Year", "GP_Name", "DriverNumber", "LapNumber"]
    )
    weather = pd.concat(weather_parts, ignore_index=True).sort_values(["Year", "GP_Name", "Time"])
    pitstops = pd.concat(pit_parts, ignore_index=True).sort_values(["Year", "GP_Name", "LapNumber"])
    return (
        loaded,
        laps.reset_index(drop=True),
        weather.reset_index(drop=True),
        pitstops.reset_index(drop=True),
    )


def extract_speed_complexity_features(laps_df: pd.DataFrame) -> pd.DataFrame:
    """Per-circuit lap-time location, spread, floor and count (N03 cell 2.1)."""
    valid = laps_df[laps_df["LapTime"].notna()].copy()
    valid["LapTime_seconds"] = valid["LapTime"].dt.total_seconds()
    valid = valid[valid["LapTime_seconds"] < 180]
    features = valid.groupby("GP_Name").agg({"LapTime_seconds": ["mean", "std", "min", "count"]})
    features = features.reset_index()
    features.columns = ["GP_Name", "mean_laptime", "std_laptime", "min_laptime", "n_laps"]
    return features


def extract_tire_degradation_features(laps_df: pd.DataFrame) -> pd.DataFrame:
    """Per-circuit degradation slope, longest stint and within-stint variance (N03 cell 2.2)."""
    valid = laps_df[laps_df["LapTime"].notna() & laps_df["TyreLife"].notna()].copy()
    valid["LapTime_seconds"] = valid["LapTime"].dt.total_seconds()
    valid = valid[valid["LapTime_seconds"] < 180]
    rows = []
    for circuit in valid["GP_Name"].unique():
        circuit_laps = valid[valid["GP_Name"] == circuit]
        slope = np.nan
        if len(circuit_laps) > 50:
            slope = stats.linregress(circuit_laps["TyreLife"], circuit_laps["LapTime_seconds"])[0]
        stint_variance = (
            circuit_laps.groupby(["Year", "DriverNumber", "Stint"])["LapTime_seconds"].var().mean()
        )
        rows.append(
            {
                "GP_Name": circuit,
                "degradation_rate": slope,
                "max_stint_length": circuit_laps["TyreLife"].max(),
                "stint_variance": stint_variance,
            }
        )
    return pd.DataFrame(rows)


def extract_pitstop_strategy_features(pitstops_df: pd.DataFrame) -> pd.DataFrame:
    """Per-circuit stop count and stop lap, mean and spread (N03 cell 2.3)."""
    per_driver = pitstops_df.groupby(["Year", "GP_Name", "DriverNumber"]).size()
    per_driver = per_driver.reset_index(name="Pit_Stops")
    counts = per_driver.groupby("GP_Name").agg({"Pit_Stops": ["mean", "std"]}).reset_index()
    counts.columns = ["GP_Name", "mean_pitstops", "std_pitstops"]
    timing = pitstops_df.groupby("GP_Name")["LapNumber"].agg(["mean", "std"]).reset_index()
    timing.columns = ["GP_Name", "mean_pitstop_lap", "std_pitstop_lap"]
    return counts.merge(timing, on="GP_Name")


def extract_environmental_features(weather_df: pd.DataFrame) -> pd.DataFrame:
    """Per-circuit track and air temperature and pressure (N03 cell 2.4)."""
    features = weather_df.groupby("GP_Name").agg(
        {"TrackTemp": ["mean", "std"], "AirTemp": "mean", "Pressure": "mean"}
    )
    features = features.reset_index()
    features.columns = [
        "GP_Name",
        "mean_track_temp",
        "std_track_temp",
        "mean_air_temp",
        "mean_pressure",
    ]
    return features


def compute_sector_speed(laps_df: pd.DataFrame) -> pd.DataFrame:
    """Per-circuit mean of the three speed traps (N03 cell 3.1)."""
    valid = laps_df[
        laps_df["SpeedI1"].notna() & laps_df["SpeedI2"].notna() & laps_df["SpeedFL"].notna()
    ].copy()
    valid["mean_sector_speed"] = valid[["SpeedI1", "SpeedI2", "SpeedFL"]].mean(axis=1)
    return valid.groupby("GP_Name")["mean_sector_speed"].mean().reset_index()


def build_features(
    laps: pd.DataFrame, weather: pd.DataFrame, pitstops: pd.DataFrame
) -> pd.DataFrame:
    """Merge the four extractor outputs into N03's circuit feature matrix.

    Args:
        laps: concatenated laps for the year set.
        weather: concatenated weather for the year set.
        pitstops: concatenated pitstops for the year set.

    Returns:
        One row per circuit with the 16 aggregates, minus the `Spain` folder, a 2023
        test-session artefact that N03 drops at its own merge step.
    """
    features = (
        extract_speed_complexity_features(laps)
        .merge(extract_tire_degradation_features(laps), on="GP_Name")
        .merge(extract_pitstop_strategy_features(pitstops), on="GP_Name")
        .merge(extract_environmental_features(weather), on="GP_Name")
        .merge(compute_sector_speed(laps), on="GP_Name", how="left")
    )
    return features[features["GP_Name"] != "Spain"].reset_index(drop=True)


# ---------------------------------------------------------------------------
# The experiment
# ---------------------------------------------------------------------------


def feature_columns(frame: pd.DataFrame, drop_season_counts: bool = False) -> list[str]:
    """The columns k-means is fit on: everything but GP_Name and N03's two drops."""
    dropped = ["GP_Name", *DROPPED_FEATURES]
    if drop_season_counts:
        dropped = dropped + SEASON_COUNT_FEATURES
    return [c for c in frame.columns if c not in dropped]


def fit_clean(
    train: pd.DataFrame, columns: list[str], seed: int = FIT_SEED
) -> tuple[StandardScaler, KMeans, np.ndarray]:
    """Fit the scaler and k-means on one year set, with nothing from any other.

    The deployed scaler was fit on the pooled 25 rows, so reusing it would put the
    test season back into the experiment through the back door. This fits its own.

    Args:
        train: the circuit feature frame for the training seasons.
        columns: the feature columns to fit on.
        seed: the KMeans random_state, swept elsewhere to show partition stability.

    Returns:
        The fitted scaler, the fitted k-means and the training circuits' labels.
    """
    scaler = StandardScaler().fit(train[columns])
    kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=seed, n_init=20)
    labels = kmeans.fit_predict(scaler.transform(train[columns]))
    return scaler, kmeans, labels


def compare_labellings(left: pd.Series, right: pd.Series) -> dict:
    """Count how many circuits move between two labellings, permutation-safe.

    k-means cluster ids are arbitrary, so an identical partition with relabelled ids
    would read as a near-total flip under raw equality. The count reported is the
    disagreement under the best one-to-one matching of the two id sets, found by
    Hungarian assignment on the contingency table.

    Args:
        left: cluster ids indexed by circuit name.
        right: cluster ids for the same circuits, same index.

    Returns:
        The matched flip count, the circuit total, the raw unmatched count for
        contrast, the Adjusted Rand Index scoring the partitions themselves, the id
        mapping that was matched, and the names of the circuits that moved.
    """
    aligned = pd.DataFrame({"left": left, "right": right}).dropna()
    contingency = pd.crosstab(aligned["left"], aligned["right"])
    rows, cols = linear_sum_assignment(-contingency.to_numpy())
    matched = contingency.to_numpy()[rows, cols].sum()
    mapping = dict(zip(contingency.index[rows], contingency.columns[cols]))
    moved = aligned.index[aligned["left"].map(mapping) != aligned["right"]].tolist()
    return {
        "flips": int(len(aligned) - matched),
        "total": int(len(aligned)),
        "raw_flips": int((aligned["left"] != aligned["right"]).sum()),
        "ari": float(adjusted_rand_score(aligned["left"], aligned["right"])),
        "mapping": mapping,
        "moved": sorted(moved),
    }


def deployed_labellings() -> tuple[pd.Series, pd.Series]:
    """The two cluster labellings that are served today, both keyed by circuit.

    Returns:
        The pooled labels, which tire, overtake and safety car read, with the
        duplicate `Miami Gardens` row folded onto `Miami`, and the Step-7 labels,
        which the pace model reads. Both cover the same 24 circuits.

    Raises:
        ValueError: if the two Miami spellings carry different pooled clusters, in
            which case folding them would hide a real disagreement.
    """
    pooled = pd.read_parquet(CLUSTERING_PATH / "circuit_clusters_k4.parquet")
    miami = pooled[pooled["GP_Name"].isin(["Miami", "Miami Gardens"])]["Cluster"].unique()
    if len(miami) != 1:
        raise ValueError(
            f"the two Miami spellings carry different pooled clusters {miami}; folding "
            "them onto one row would hide a real disagreement"
        )
    pooled["GP_Name"] = pooled["GP_Name"].replace(GP_NAME_ALIASES)
    pooled_labels = pooled.drop_duplicates("GP_Name").set_index("GP_Name")["Cluster"]

    step7 = pd.read_parquet(CLUSTERING_PATH / "circuit_clusters_k4_2025.parquet")
    step7["GP_Name"] = step7["GP_Name"].replace(GP_NAME_ALIASES)
    step7_labels = step7.set_index("GP_Name")["Cluster"]
    return pooled_labels, step7_labels


def lap_share(circuits: list[str]) -> float:
    """The fraction of served 2025 laps run at the given circuits.

    A flip at Monaco and a flip at Silverstone are not the same amount of served
    data, so the circuit count on its own over or understates the impact.

    Args:
        circuits: the circuit names that changed cluster.

    Returns:
        Their share of laps_featured_2025.parquet rows, or NaN when that file is not
        present on this machine's data tree.
    """
    if not FEATURED_2025.exists():
        return float("nan")
    laps = pd.read_parquet(FEATURED_2025, columns=["GP_Name"])
    laps["GP_Name"] = laps["GP_Name"].replace(GP_NAME_ALIASES)
    return float(laps["GP_Name"].isin(circuits).mean())


def report(title: str, result: dict) -> None:
    """Print one comparison: the matched flip count, the raw count, ARI and lap share."""
    share = lap_share(result["moved"])
    print(f"\n{title}")
    print(
        f"  {result['flips']} / {result['total']} circuits change cluster (Hungarian-matched)"
        f" | raw label equality would say {result['raw_flips']} | ARI {result['ari']:.3f}"
    )
    print(f"  served 2025 laps affected: {share:.1%}")
    if result["moved"]:
        print(f"  moved: {', '.join(result['moved'])}")


def main() -> int:
    started = time.perf_counter()
    n_train, train_laps, train_weather, train_pits = load_years(TRAIN_YEARS)
    n_test, test_laps, test_weather, test_pits = load_years((TEST_YEAR,))
    train = build_features(train_laps, train_weather, train_pits)
    test = build_features(test_laps, test_weather, test_pits)
    test["GP_Name"] = test["GP_Name"].replace(GP_NAME_ALIASES)

    if set(train["GP_Name"]) != set(test["GP_Name"]):
        only_train = sorted(set(train["GP_Name"]) - set(test["GP_Name"]))
        only_test = sorted(set(test["GP_Name"]) - set(train["GP_Name"]))
        raise ValueError(
            "the year sets do not cover the same circuits, so the comparison would be "
            f"partial. Only in {TRAIN_YEARS}: {only_train}. Only in {TEST_YEAR}: {only_test}"
        )

    # Las Vegas 2025 has no speed-trap columns. N03 Step 7 imputes the POOLED scaler's
    # mean here, so the clean equivalent is the training seasons' own mean.
    imputed = test["mean_sector_speed"].isna()
    test.loc[imputed, "mean_sector_speed"] = train["mean_sector_speed"].mean()

    print(f"train {TRAIN_YEARS}: {n_train} GPs, {len(train_laps):,} laps, {len(train)} circuits")
    print(f"test  {TEST_YEAR}: {n_test} GPs, {len(test_laps):,} laps, {len(test)} circuits")
    if imputed.any():
        print(
            f"imputed mean_sector_speed from the {TRAIN_YEARS} mean for: "
            f"{', '.join(test.loc[imputed, 'GP_Name'])}"
        )

    pooled, step7 = deployed_labellings()
    columns = feature_columns(train)
    scaler, kmeans, train_labels = fit_clean(train, columns)
    clean_lookup = pd.Series(train_labels, index=train["GP_Name"])
    clean_predict = pd.Series(
        kmeans.predict(scaler.transform(test[columns])), index=test["GP_Name"]
    )

    print("\n" + "=" * 78)
    print(f"circuit_cluster leak: k-means refit on {TRAIN_YEARS[0]}-{TRAIN_YEARS[1]} only")
    print("=" * 78)

    report(
        "LOOKUP path (tire, overtake, safety car): clean fit vs deployed pooled fit",
        compare_labellings(clean_lookup, pooled),
    )
    report(
        "PREDICT path (pace): clean fit predicting 2025 vs deployed Step-7 labels",
        compare_labellings(clean_predict, step7),
    )

    # The two season-count features move circuits on their own: the pooled fit counted
    # three seasons of laps per circuit and a clean fit counts two, so a circuit whose
    # character did not change at all still shifts on n_laps. Dropping them from both
    # sides separates the leak from that artefact, and both paths need the control
    # because both compare a two-season fit against a three-season one.
    lean_columns = feature_columns(train, drop_season_counts=True)
    lean_scaler, lean_kmeans, lean_labels = fit_clean(train, lean_columns)
    lean_lookup = pd.Series(lean_labels, index=train["GP_Name"])
    lean_predict = pd.Series(
        lean_kmeans.predict(lean_scaler.transform(test[lean_columns])), index=test["GP_Name"]
    )
    report(
        "LOOKUP path without n_laps / max_stint_length (the season-count control)",
        compare_labellings(lean_lookup, pooled),
    )
    report(
        "PREDICT path without n_laps / max_stint_length (the season-count control)",
        compare_labellings(lean_predict, step7),
    )

    print("\nBaseline for scale: the two deployed labellings against each other")
    baseline = compare_labellings(pooled, step7)
    print(
        f"  pooled vs Step-7: {baseline['flips']} / {baseline['total']} differ "
        f"(ARI {baseline['ari']:.3f}) | moved: {', '.join(baseline['moved'])}"
    )

    # 24 points at silhouette 0.21 is not a stable partition, so one seed is a probe
    # rather than a measurement.
    sweep = []
    for seed in SEED_SWEEP:
        seed_scaler, seed_kmeans, seed_labels = fit_clean(train, columns, seed=seed)
        seed_lookup = pd.Series(seed_labels, index=train["GP_Name"])
        sweep.append(compare_labellings(seed_lookup, pooled)["flips"])
    print(
        f"\nSeed sweep (lookup path, random_state 0-{max(SEED_SWEEP)}): flips range "
        f"{min(sweep)}-{max(sweep)} / {len(train)}, median {int(np.median(sweep))}"
    )
    print(f"\nran in {time.perf_counter() - started:.1f}s, no network, no model retrained")
    return 0


if __name__ == "__main__":
    sys.exit(main())
