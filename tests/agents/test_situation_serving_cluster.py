"""N27 serves N14 the circuit clustering N14 trained on (gate finding on #831).

The mirror of `test_tire_serving_frame.py::test_the_agent_reads_the_cluster_family_the_tcn_trained_on`,
and it exists because I got this exact question backwards.

Two clustering families live under `data/processed/circuit_clustering/`, and which one
a model trained on is a property OF THAT MODEL, not of the repository. My first attempt
at #831 repointed N27 at `circuit_clusters_k4_2025.parquet` on the strength of a table
showing it agreeing 24/24 with `laps_featured_2025.Cluster` — which is **N06's** training
column. N14 trains on `sc_labeled_2023_2025.parquet`, which N13 builds from the POOLED
map, so the fix would have handed N14 a clustering its own training rows never carried,
moving `sc_prob_3lap` on 15 of 15 Budapest laps.

WHY THIS ASSERTS ON SOURCES, NOT ON A JOIN
------------------------------------------
The obvious test — join `sc_labeled`'s `circuit_cluster` against each map and compare —
cannot be written directly: that frame keys on `race_id` (`"2023_1"`), not `GP_Name`, and
reconstructing the mapping would mean reimplementing N13's fuzzy `resolve_cluster` here.
A reimplementation is what this test is guarding against.

So it pins the pairing where it is unambiguous: N13's own source names the file it built
the labels from, and N27's source names the file it serves. Both must be the same file,
and the two families must still genuinely differ — otherwise this passes on a day they
coincide and the loader could be repointed unnoticed.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).parent.parent.parent
_CLUSTERS = ROOT / "data" / "processed" / "circuit_clustering"
_POOLED_NAME = "circuit_clusters_k4.parquet"
_SEASON_NAME = "circuit_clusters_k4_2025.parquet"

_N13_SOURCE = ROOT / ".nb_py" / "N13_sc_eda.py"
_N27_SOURCE = ROOT / "src" / "agents" / "race_situation_agent.py"


def test_the_sc_agent_loads_the_pooled_map_and_only_that_one():
    """The half that runs everywhere: N27's own source, which IS tracked in git."""
    n27 = _N27_SOURCE.read_text(encoding="utf-8")

    assert _POOLED_NAME in n27, (
        "N27 no longer loads the pooled clustering. Read this module's docstring "
        "before changing it: the obvious-looking repoint at the _2025 map is a "
        "regression, and it moved sc_prob_3lap on 15 of 15 Budapest laps"
    )
    assert _SEASON_NAME not in n27, (
        "N27 references the 2025 clustering family, which N14 never trained on"
    )


@pytest.mark.skipif(
    not _N13_SOURCE.exists(),
    reason=".nb_py/ is gitignored, so this half cannot run on a CI runner",
)
def test_the_map_n27_serves_is_the_one_n13_built_the_labels_from():
    """The other half of the pairing, checkable only where the notebooks are.

    `.nb_py/` is gitignored (`.gitignore:148`), so without this marker the test
    would fail on every CI runner rather than skip -- the mirror of the mistake
    that turned this branch's sibling red: guard on the artefact the test READS,
    and check whether git actually ships it.
    """
    n13 = _N13_SOURCE.read_text(encoding="utf-8")
    assert _POOLED_NAME in n13, (
        "N13 no longer builds sc_labeled from the pooled clustering; whatever it "
        "builds from is now what N27 must serve, so rewrite this against it "
        "rather than deleting it"
    )


@pytest.mark.skipif(
    not ((_CLUSTERS / _POOLED_NAME).exists() and (_CLUSTERS / _SEASON_NAME).exists()),
    reason="clustering artefacts absent (HF dataset not fetched)",
)
def test_the_two_cluster_families_still_disagree():
    """Without this, the test above passes on a day the two maps coincide."""
    pooled = pd.read_parquet(_CLUSTERS / _POOLED_NAME, columns=["GP_Name", "Cluster"])
    season = pd.read_parquet(_CLUSTERS / _SEASON_NAME, columns=["GP_Name", "Cluster"])

    merged = pooled.merge(season, on="GP_Name", suffixes=("_pooled", "_season"))
    assert not merged.empty, "no GP names joined; the key convention changed"

    disagreements = (merged["Cluster_pooled"] != merged["Cluster_season"]).sum()
    assert disagreements > 0, (
        "the two cluster families now agree everywhere, so the loader could be "
        "repointed unnoticed and the source assertion above has lost its teeth"
    )
