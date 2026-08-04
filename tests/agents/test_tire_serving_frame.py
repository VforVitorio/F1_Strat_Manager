"""The tire agent serves the TCN the frame it was trained on (W-F8/F9/F10/F11).

Four columns of the serving frame were a different quantity from the trained one, and the
TCN's output moved by a mean of 0.42 s and up to 4.99 s against cliff thresholds of ~2 s.

The one that needed the most digging was `lap_time_vs_cluster_mean`. N07 builds
`laps_tiredeg` by reading the COMBINED `laps_featured.parquet` and inheriting its cluster
columns (`.nb_py/N07_tiredeg_eda.py:92`), so the TCN trained on the POOLED clustering. The
serving frame kept the 2025 family's delta next to the pooled family's `Cluster`, a mix
neither model ever saw.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).parent.parent.parent
_TIREDEG = ROOT / "data" / "processed" / "laps_tiredeg.parquet"
_CLUSTERS = ROOT / "data" / "processed" / "circuit_clustering"


def _constants() -> dict[int, float]:
    from src.agents.tire_agent import TireAgentConfig

    return TireAgentConfig._TRAINED_CLUSTER_MEAN_LAP_S


# --- the trained constants, re-derived rather than re-read --------------------


@pytest.mark.data
@pytest.mark.skipif(not _TIREDEG.exists(), reason="laps_tiredeg absent")
def test_the_cluster_means_are_the_ones_n04_subtracted():
    """A hardcoded constant is a claim about the artefact; this re-derives it.

    Recovered as `LapTime_s - lap_time_vs_cluster_mean`, which must come out to ONE
    value per cluster. If it does not, the feature is not what this code believes and
    the constants are meaningless rather than merely stale.
    """
    laps = pd.read_parquet(
        _TIREDEG, columns=["Cluster", "LapTime_s", "lap_time_vs_cluster_mean"]
    ).dropna()
    implied = laps.assign(mean_lap=laps["LapTime_s"] - laps["lap_time_vs_cluster_mean"])

    by_cluster = implied.groupby("Cluster")["mean_lap"]
    assert by_cluster.nunique().eq(1).all(), (
        "the implied cluster mean is not constant within a cluster, so "
        "lap_time_vs_cluster_mean is not LapTime_s minus a per-cluster constant"
    )

    constants = _constants()
    for cluster, value in by_cluster.first().items():
        assert int(cluster) in constants, f"cluster {cluster} has no declared constant"
        assert constants[int(cluster)] == pytest.approx(float(value), abs=1e-6), (
            f"cluster {cluster}: declared {constants[int(cluster)]}, artefact says {value}"
        )


@pytest.mark.data
@pytest.mark.skipif(not _TIREDEG.exists(), reason="laps_tiredeg absent")
def test_the_served_cluster_delta_reproduces_the_trained_column_exactly():
    """The effect, over every 2025 row the artefact carries."""
    laps = pd.read_parquet(
        _TIREDEG, columns=["Year", "Cluster", "LapTime_s", "lap_time_vs_cluster_mean"]
    ).dropna(subset=["LapTime_s", "lap_time_vs_cluster_mean"])
    rows = laps[laps["Year"] == 2025]
    assert len(rows) > 0, "no 2025 rows: this would hold vacuously"

    served = rows["LapTime_s"] - rows["Cluster"].map(_constants())
    assert np.abs(served - rows["lap_time_vs_cluster_mean"]).max() == pytest.approx(0.0, abs=1e-9)


@pytest.mark.data
@pytest.mark.skipif(
    not (_CLUSTERS / "circuit_clusters_k4.parquet").exists() or not _TIREDEG.exists(),
    reason="cluster artefacts absent",
)
def test_the_agent_reads_the_cluster_family_the_tcn_trained_on():
    """Pooled, not 2025. The two disagree, and the disagreement is the bug.

    Asserting the agreement with the pooled map alone would pass if someone repointed the
    loader at the 2025 map on a day both happened to agree, so this also asserts that the
    two maps genuinely differ.
    """
    trained = (
        pd.read_parquet(_TIREDEG, columns=["GP_Name", "Year", "Cluster"])
        .query("Year == 2025")
        .drop_duplicates("GP_Name")[["GP_Name", "Cluster"]]
    )
    pooled = pd.read_parquet(
        _CLUSTERS / "circuit_clusters_k4.parquet", columns=["GP_Name", "Cluster"]
    )
    season = pd.read_parquet(
        _CLUSTERS / "circuit_clusters_k4_2025.parquet", columns=["GP_Name", "Cluster"]
    )

    against_pooled = trained.merge(pooled, on="GP_Name", suffixes=("_t", "_m"))
    against_season = trained.merge(season, on="GP_Name", suffixes=("_t", "_m"))

    assert (against_pooled["Cluster_t"] == against_pooled["Cluster_m"]).all(), (
        "the pooled map no longer matches what the TCN trained on"
    )
    assert not (against_season["Cluster_t"] == against_season["Cluster_m"]).all(), (
        "the two cluster families now agree everywhere, so this test can no longer tell "
        "them apart and the loader could be repointed unnoticed"
    )


# --- the three shift/alias fixes, hermetic -----------------------------------


def test_degradation_is_not_lagged():
    """Training computed both unshifted; a lag moved the TCN 0.185 s at cliff onset."""
    from src.agents.tire_agent import _add_degradation_rate

    frame = pd.DataFrame(
        {
            "TyreLife": range(1, 9),
            "FuelAdjustedLapTime": [90.0, 90.2, 90.5, 90.9, 91.4, 92.0, 92.7, 93.5],
        }
    )
    out = _add_degradation_rate(frame.copy())

    # A lagged series would repeat the previous row; an unshifted one rises immediately.
    assert out["DegradationRate"].iloc[1] != pytest.approx(0.0), (
        "the second lap still reads the first lap's zero, so the shift is back"
    )


def test_a_missing_predecessor_stays_missing():
    """N10's scaler does fillna(0), so training saw a raw zero, not a repeat of the lap."""
    from src.agents.tire_agent import _add_prev_cols

    frame = pd.DataFrame({"LapTime_s": [90.0, 91.0], "SpeedFL": [300.0, 301.0]})
    for col in ("SpeedI1", "SpeedI2", "SpeedST", "TyreLife"):
        frame[col] = [1.0, 2.0]

    out = _add_prev_cols(frame.copy())

    assert pd.isna(out["Prev_LapTime"].iloc[0]), (
        "the first row was filled with its own value, which is what training did not do"
    )
    assert out["Prev_LapTime"].iloc[1] == pytest.approx(90.0)


def test_an_existing_laps_since_pit_is_not_overwritten_by_tyre_life():
    """They are different quantities and the artefacts carry the real one."""
    from src.agents.tire_agent import _add_session_cols

    frame = pd.DataFrame(
        {
            "LapTime_s": [90.0, 91.0],
            "Sector1_s": [30.0, 30.0],
            "Sector2_s": [30.0, 30.0],
            "Sector3_s": [30.0, 31.0],
            "TyreLife": [12, 13],
            "LapsSincePitStop": [3, 4],
        }
    )
    meta = {
        "fastest_lap_s": 89.0,
        "cluster_mean_lap_s": 95.0,
        "total_laps": 50,
        "cluster_id": 1,
        "team_id": 0,
        "year": 2025,
    }
    frame["LapNumber"] = [1, 2]
    for col in ("SpeedI1", "SpeedI2", "SpeedFL"):
        frame[col] = [300.0, 301.0]

    out = _add_session_cols(frame.copy(), meta)

    assert list(out["LapsSincePitStop"]) == [3, 4], "the real column was stomped with TyreLife"
