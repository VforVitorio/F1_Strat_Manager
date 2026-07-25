"""The measured MC tables must stay measured (#553).

Two layers, because the raw races are not in git:

1. Structural + sanity invariants on the COMMITTED ``data/mc_measured_v1.json``.
   These run everywhere, including a CI runner with no data, and are what stops a
   hand-edited or half-regenerated table from reaching the engine: every value
   carries an n, every interval brackets its point estimate, and the undercut
   band decays with distance instead of wandering.

2. A regeneration check that reruns the script and asserts the committed file is
   byte-identical. It needs ``data/raw`` and skips without it.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent.parent
TABLES_PATH = ROOT / "data" / "mc_measured_v1.json"

_HAS_RAW = (ROOT / "data" / "raw" / "2024").is_dir()
_skip_no_raw = pytest.mark.skipif(
    not _HAS_RAW,
    reason="data/raw/ not present (CI runner without the HF dataset)",
)


@pytest.fixture(scope="module")
def tables() -> dict:
    return json.loads(TABLES_PATH.read_text(encoding="utf-8"))


def _walk_measurements(node, path="") -> list[tuple[str, dict]]:
    """Collect every dict that looks like a measurement (has an ``n`` and a ci95)."""
    found: list[tuple[str, dict]] = []
    if isinstance(node, dict):
        if "n" in node and "ci95" in node:
            found.append((path, node))
        for key, value in node.items():
            found.extend(_walk_measurements(value, f"{path}.{key}" if path else key))
    return found


# ---------------------------------------------------------------------------
# Structure
# ---------------------------------------------------------------------------


def test_every_table_is_present(tables):
    for table in (
        "sc_window",
        "neutralisation_rate",
        "gap_density",
        "undercut_band",
        "stop_hazard",
    ):
        assert table in tables, f"missing table: {table}"
    assert tables["races_measured"] > 0
    assert tables["window_laps"] == 5


def test_every_measurement_carries_its_sample_size_and_interval(tables):
    measurements = _walk_measurements(tables)
    assert measurements, "no measurements found — the file shape changed"

    for path, node in measurements:
        assert isinstance(node["n"], int), f"{path}: n is not an int"
        point = node.get("mean", node.get("rate"))
        low, high = node["ci95"]
        if point is None or low is None or high is None:
            # An unmeasured cell is allowed; a cell with a value but no interval
            # is not, because that is how an unsourced constant sneaks back in.
            assert node["n"] <= 1, f"{path}: has n={node['n']} but no usable interval"
            continue
        assert low <= point <= high, f"{path}: {point} outside its own CI {[low, high]}"


def test_rates_are_probabilities(tables):
    for path, node in _walk_measurements(tables):
        rate = node.get("rate")
        if rate is None:
            continue
        assert 0.0 <= rate <= 1.0, f"{path}: rate {rate} is not a probability"


# ---------------------------------------------------------------------------
# Sanity — the numbers have to mean what the engine will read them as
# ---------------------------------------------------------------------------


def test_a_neutralised_window_holds_fewer_racing_laps_than_the_window(tables):
    for kind, stats in tables["sc_window"]["by_kind"].items():
        racing = stats["racing_laps_in_window"]["mean"]
        assert racing is not None, f"{kind}: no measurement"
        assert 0.0 <= racing < tables["window_laps"], (
            f"{kind}: {racing} racing laps inside a {tables['window_laps']}-lap window "
            "while neutralised — a neutralisation that costs no racing laps is not one"
        )


def test_safety_car_spells_outlast_virtual_ones(tables):
    kinds = tables["sc_window"]["by_kind"]
    assert kinds["sc"]["spell_length_laps"]["mean"] > kinds["vsc"]["spell_length_laps"]["mean"], (
        "a full Safety Car must run longer than a VSC on average; if this flips, the "
        "status parsing is conflating the two (#471)"
    )


def test_the_field_closes_up_under_a_safety_car(tables):
    racing_median = tables["gap_density"]["racing"]["p50"]
    sc_median = tables["gap_density"]["safety_car"]["p50"]
    assert sc_median < racing_median, (
        "cars must sit closer together under a Safety Car than while racing; "
        "this is the physics the pit-loss saving comes from"
    )


def test_the_racing_bucket_declares_what_is_inside_it(tables):
    mix = tables["status_mix"]
    assert mix["racing_is"] == ["clear", "yellow"], (
        "the racing bucket must state that it holds local-yellow laps too; a bucket "
        "named for something it does not contain is how #486 happened"
    )
    shares = {status: cell["share"] for status, cell in mix["by_status"].items()}
    assert abs(sum(shares.values()) - 1.0) < 1e-3, "status shares must cover every lap"
    assert shares.get("yellow", 0) > 0, (
        "yellow laps must be visible in the mix, not silently folded into clear"
    )


def test_the_undercut_decays_with_the_gap_to_the_target(tables):
    band = tables["undercut_band"]
    bins = band["by_gap_bin_seconds"]
    ordered = ["0-1", "1-2", "2-3", "3-5", "5-10", "10+"]
    rates = [
        bins[label]["rate"] for label in ordered if bins.get(label, {}).get("rate") is not None
    ]
    assert len(rates) >= 4, "too few populated gap bins to judge the band"
    assert rates == sorted(rates, reverse=True), (
        f"undercut success must fall as the target gets further away, got {rates}"
    )
    assert rates[0] > rates[-1] * 5, "the near bin should dominate the far bin by a wide margin"


def test_the_undercut_band_is_a_usable_number_of_seconds(tables):
    u_band = tables["undercut_band"]["u_band_s"]
    assert u_band is not None and 0.5 < u_band < 30.0, (
        f"u_band {u_band}s is not a plausible undercut range; it replaces the ad-hoc "
        "'within 5 positions' selector, so an implausible value would widen or brick it"
    )


def test_stop_hazard_rises_with_tyre_life_on_every_dry_compound(tables):
    cells = tables["stop_hazard"]["by_cell"]
    for compound in ("SOFT", "MEDIUM", "HARD"):
        young = cells.get(f"{compound}|0-9|racing", {}).get("rate")
        old = cells.get(f"{compound}|20-29|racing", {}).get("rate")
        if young is None or old is None:
            continue
        assert old > young, (
            f"{compound}: a car on older tyres must be likelier to stop within the "
            f"window ({old} vs {young})"
        )


def test_the_neutralisation_rate_is_keyed_by_circuit_slugs_agents_can_query(tables):
    from src.f1_strat_manager.gp_slugs import resolve_gp_slug

    per_circuit = tables["neutralisation_rate"]["per_circuit"]
    assert per_circuit, "no per-circuit rates"
    for slug in per_circuit:
        # Raises for a key nothing will ever look up — the #448 failure mode,
        # where a whole table sat in a keyspace no caller used.
        resolve_gp_slug(slug)


# ---------------------------------------------------------------------------
# Regeneration
# ---------------------------------------------------------------------------


@_skip_no_raw
def test_the_committed_tables_match_a_fresh_measurement():
    import subprocess
    import sys

    before = TABLES_PATH.read_text(encoding="utf-8")
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "measure_mc_tables.py")],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    assert result.returncode == 0, f"measurement script failed:\n{result.stderr}"

    after = TABLES_PATH.read_text(encoding="utf-8")
    assert after == before, (
        "data/mc_measured_v1.json drifted from what the script produces. Either the "
        "raw data changed (rerun and commit) or the file was edited by hand (do not)."
    )
