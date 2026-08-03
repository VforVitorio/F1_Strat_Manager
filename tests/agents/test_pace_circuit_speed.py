"""N06 is fed the circuit's own mean sector speed, not the speed trap (#797).

`mean_sector_speed` is a property of the CIRCUIT: the featured parquet holds exactly
one value per (year, GP). `_compute_derived` used to substitute `prev_speed_st` whenever
none was supplied, and `run_from_state` never supplied one, so on the path every real
race takes the model received a different physical quantity on every lap.

What these pin is the set of rules that makes the fix a fix rather than a new guess: the
value served must be the value the replayed lap carries, EVERY race must resolve, and a
circuit that genuinely does not resolve must reach the model as MISSING rather than as a
substituted reading.
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).parent.parent.parent
_HAS_ARTEFACTS = (
    ROOT / "data" / "models" / "lap_time" / "xgb_laptime_delta_final.json"
).exists() and (ROOT / "data" / "processed" / "laps_featured.parquet").exists()
_HAS_RAW = (ROOT / "data" / "raw").is_dir()

pytestmark = pytest.mark.skipif(
    not _HAS_ARTEFACTS, reason="N06 weights or the combined featured parquet are absent"
)


@pytest.fixture(scope="module")
def agent():
    from src.agents.pace_agent import PaceAgent

    return PaceAgent()


def _feature_row(agent, **overrides):
    """A real feature row, built through the real builder."""
    kwargs = dict(
        driver_number=4,
        lap_number=30,
        stint=2,
        tyre_life=14,
        compound="MEDIUM",
        position=4,
        team="McLaren",
        laps_since_pit=14,
        fuel_load=0.5,
        year=2025,
        prev_lap_time=84.0,
        prev_tyre_life=13,
        prev_speed_st=300.0,
        air_temp=25.0,
        track_temp=35.0,
        humidity=50.0,
        rainfall=0.0,
        total_laps=57,
        gp_name="Lusail",
        stint_baseline_tyre_life=1,
    )
    kwargs.update(overrides)
    return agent._build_feature_row(**kwargs)


# --- the value served is the value the replayed lap carries -------------------


def test_every_race_resolves_to_the_value_its_own_parquet_rows_carry(agent):
    """Not "close to", the same number, and for every (year, GP) rather than one.

    A lookup that works for the circuit somebody tested and silently misses the rest is
    the failure this asserts against: the first version of the loader read only the 2025
    parquet and served that one value for every season, so a 2023 Silverstone lap was fed
    a measurement taken two years after it, 18.4 km/h away.
    """
    from src.f1_strat_manager.data_cache import get_data_root

    laps = pd.read_parquet(
        get_data_root() / "processed" / "laps_featured.parquet",
        columns=["Year", "GP_Name", "mean_sector_speed"],
    ).dropna()
    per_race = laps.drop_duplicates(["Year", "GP_Name"])

    assert len(per_race) > 0, "the featured artefact carried no circuit speeds"
    for row in per_race.itertuples():
        served = agent._resolve_mean_sector_speed(str(row.GP_Name), int(row.Year))
        assert served == pytest.approx(float(row.mean_sector_speed)), f"{row.Year} {row.GP_Name}"


def test_the_training_seasons_are_one_artefact_generation_not_two(agent):
    """2023 and 2024 are identical per GP, which is why the map is not "per season".

    Pinned because the docstring says so and a wrong mechanism is how the next fix goes
    wrong: someone completing a per-season resolution between 2023 and 2024 would find
    nothing to resolve. The value is recomputed per artefact BUILD, and one build pooled
    both training seasons.
    """
    shared = {
        gp
        for (year, gp) in agent.circuit_mean_sector_speed
        if (2023, gp) in agent.circuit_mean_sector_speed
        and (2024, gp) in agent.circuit_mean_sector_speed
    }
    assert shared, "no GP appears in both training seasons"
    for gp in shared:
        assert agent.circuit_mean_sector_speed[(2023, gp)] == pytest.approx(
            agent.circuit_mean_sector_speed[(2024, gp)]
        ), gp


def test_the_map_has_no_holes(agent):
    """Las Vegas is the reason this reads the combined parquet, not the 2025 one.

    `laps_featured_2025.parquet` carries NaN on all 760 Las Vegas rows, so a `.dropna()`
    over it drops a circuit N06 was fitted on and whose value sits in three other
    artefacts. "Absent from the map" has to mean an unknown circuit, never an artefact
    with a hole in it.
    """
    assert ("2025" not in {str(y) for y, _ in agent.circuit_mean_sector_speed}) is False
    assert (2025, "Las Vegas") in agent.circuit_mean_sector_speed
    for year in (2023, 2024, 2025):
        assert agent._resolve_mean_sector_speed("Las Vegas", year) == pytest.approx(
            228.9645, abs=1e-3
        )


def test_the_row_fed_to_n06_carries_the_circuit_value_not_the_speed_trap(agent):
    """The regression itself: the trap reading must not reach the feature again."""
    row = _feature_row(agent, gp_name="Lusail", year=2025, prev_speed_st=300.0)

    served = float(row["mean_sector_speed"].iloc[0])
    assert served == pytest.approx(agent.circuit_mean_sector_speed[(2025, "Lusail")])
    assert served != pytest.approx(300.0)
    # Prev_SpeedST is still its own feature and still carries the trap.
    assert float(row["Prev_SpeedST"].iloc[0]) == pytest.approx(300.0)


# --- every keyspace a real caller uses ---------------------------------------


@pytest.mark.parametrize(
    ("name", "year"),
    [
        ("Lusail", 2025),  # parquet slug, what the CLI passes
        ("Qatar Grand Prix", 2025),  # FastF1 event name
        ("Miami Gardens", 2025),  # metadata.json, what the replay engine passes
        ("Miami_Gardens", 2025),  # raw folder name
        ("Spain", 2023),  # a training-season folder whose slug differs
    ],
)
def test_each_keyspace_a_real_caller_uses_resolves(agent, name, year):
    """Four names for one GP, and an earlier draft resolved only two of them.

    'Miami Gardens' with a SPACE is what `RaceReplayEngine` puts into `session_meta`, and
    it matches neither the parquet slug nor the underscore folder form, so every lap of
    the 2025 Miami race was served NaN while its value sat in the map. The #448/#450
    dual-keyspace trap, third occurrence.
    """
    assert not math.isnan(agent._resolve_mean_sector_speed(name, year))


@pytest.mark.data
@pytest.mark.skipif(not _HAS_RAW, reason="data/raw absent")
def test_every_race_on_disk_resolves(agent):
    """The enumeration, checked rather than assumed.

    Fixing the two names an audit happened to name would leave the next one to be found
    the same way. This walks every race under `data/raw/` and asserts the name its
    metadata actually carries resolves, which is the only form of this claim worth making.
    """
    from src.f1_strat_manager.data_cache import get_data_root

    unresolved = []
    checked = 0
    for year_dir in sorted((get_data_root() / "raw").iterdir()):
        if not year_dir.is_dir() or not year_dir.name.isdigit():
            continue
        for race_dir in sorted(p for p in year_dir.iterdir() if p.is_dir()):
            meta = race_dir / "metadata.json"
            if not meta.exists():
                continue
            checked += 1
            gp_name = json.loads(meta.read_text(encoding="utf-8")).get("gp_name", "")
            if math.isnan(agent._resolve_mean_sector_speed(gp_name, int(year_dir.name))):
                unresolved.append((year_dir.name, race_dir.name, gp_name))

    assert checked > 0, "no races found on disk: this would hold vacuously"
    assert unresolved == [], f"{len(unresolved)} of {checked} races resolve to NaN: {unresolved}"


# --- an unresolvable circuit stays unknown -----------------------------------


def test_an_unknown_circuit_yields_nan_and_never_a_substituted_reading(agent, caplog):
    """The rule the bug broke: unknown data must not become a number the model can use.

    NaN is in-distribution for XGBoost, which routes a missing feature through its
    sparse-aware split. A plausible-looking speed is not: it answers a different question
    with full confidence, which is exactly what #797 was.
    """
    with caplog.at_level(logging.WARNING):
        value = agent._resolve_mean_sector_speed("Nurburgring", 2025)

    assert math.isnan(value)
    assert any("no trained mean sector speed" in r.message for r in caplog.records)


def test_a_season_the_dataset_does_not_cover_is_unknown_too(agent):
    """A real circuit in a year with no artefact is still unknown, not the nearest year."""
    assert math.isnan(agent._resolve_mean_sector_speed("Lusail", 2019))


def test_an_unknown_circuit_reaches_the_model_as_missing(agent):
    """End to end: the NaN survives the frame build rather than being coerced."""
    row = _feature_row(agent, gp_name="Nurburgring", prev_speed_st=300.0)

    assert math.isnan(float(row["mean_sector_speed"].iloc[0]))


# --- an explicit measurement still wins --------------------------------------


def test_an_explicitly_supplied_value_is_not_overridden_by_the_circuit_constant(agent):
    """A caller that genuinely measured this lap must keep its number."""
    row = _feature_row(agent, gp_name="Lusail", mean_sector_speed=241.5)

    assert float(row["mean_sector_speed"].iloc[0]) == pytest.approx(241.5)


# --- the envelope bound is meaningful again ----------------------------------


def test_the_envelope_separates_a_circuit_n06_was_fitted_on_from_one_it_was_not(agent):
    """The bound is back and it discriminates, which is the point of restoring it.

    A first version of this test asserted that EVERY served value falls inside the bound,
    and Monza refuted it. That assertion was wrong, not the code: the bound is the
    2023-2024 range and a 2025 lap is served a 2025 measurement, so a 2025 circuit may
    legitimately sit outside. Monza 2025 does, at 317.24 against a fitted maximum of
    314.97, and saying so is exactly the job the bound was restored to do.

    So what is pinned is the discrimination, not a blanket. A bound that flagged
    everything, or nothing, would pass a coverage check and tell nobody anything.
    """
    from src.agents.pace_agent import _N06_ENVELOPE, _N06_TRAINED_BOUNDS

    lower, upper = _N06_TRAINED_BOUNDS["mean_sector_speed"]
    served = agent.circuit_mean_sector_speed
    inside = {k: v for k, v in served.items() if lower <= v <= upper}
    outside = {k: v for k, v in served.items() if not lower <= v <= upper}

    assert inside, "no served value is in range: the bound and the feed disagree"
    assert len(inside) > len(outside), (
        f"{len(outside)} of {len(served)} races fall outside; a bound that rejects most "
        f"of what it is fed is describing the wrong quantity"
    )
    # Every training-season value must be inside by construction: that is where the
    # bound came from. Only a later season may legitimately escape it.
    for (year, gp), value in outside.items():
        assert year == 2025, f"{year} {gp} is outside the range its own season defined"
        verdict = _N06_ENVELOPE.check({"mean_sector_speed": value})
        assert "mean_sector_speed" in verdict.violations, gp
