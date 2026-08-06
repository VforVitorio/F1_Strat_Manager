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
    from src.agents.pace_agent import _FEATURED_SEASONS
    from src.f1_strat_manager.data_cache import get_data_root

    checked = 0
    for year in _FEATURED_SEASONS:
        laps = pd.read_parquet(
            get_data_root() / "processed" / f"laps_featured_{year}.parquet",
            columns=["GP_Name", "mean_sector_speed"],
        ).dropna()
        for row in laps.drop_duplicates("GP_Name").itertuples():
            checked += 1
            served = agent._resolve_mean_sector_speed(str(row.GP_Name), year)
            assert served == pytest.approx(float(row.mean_sector_speed)), f"{year} {row.GP_Name}"

    assert checked > 0, "the featured artefacts carried no circuit speeds"


def test_a_2025_lap_is_not_served_the_training_seasons_measurement(agent):
    """The regression a whole gate round was spent on, pinned as an EFFECT.

    Switching the loader to the COMBINED `laps_featured.parquet` looked like a strict
    improvement, because that file has no missing values. It broadcasts the training-era
    number across all three seasons instead: its Silverstone row reads 249.71 for 2023 and
    for 2025 alike, so keying by year became decorative and every 2025 lap silently
    received the 2023 measurement.

    The raw laps settle it. Silverstone 2025's own speed traps average 232.32 km/h, which
    is the per-year artefact's 231.36 and not the combined file's 249.71.

    Asserting that the two seasons DIFFER is what catches this; asserting either value on
    its own passes happily against the wrong artefact.
    """
    served_2023 = agent._resolve_mean_sector_speed("Silverstone", 2023)
    served_2025 = agent._resolve_mean_sector_speed("Silverstone", 2025)

    assert abs(served_2025 - served_2023) > 1.0, (
        f"Silverstone reads {served_2025} in 2025 and {served_2023} in 2023; if they are "
        f"equal the loader is reading an artefact that broadcasts one season's value"
    )
    assert served_2025 == pytest.approx(231.36, abs=0.5)


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


def test_las_vegas_2025_is_answered_from_the_artefact_and_not_from_another_season(agent):
    """The one race FastF1 could not measure, and the two ways of getting it wrong.

    FastF1 has no SpeedI2 reading for the entire 2025 race, so the circuit's speed was NaN
    on all 760 rows. This test used to pin that NaN, with its own docstring saying the
    missing value "belongs in the artefact rather than in a fallback here" — which is where
    it now is: imputed from the circuit's own trap-offset (MAE 1.22 km/h leave-era-out) and
    carried with a `mean_sector_speed_imputed` flag on every affected row.

    The property that mattered has not moved. The tempting repair was always to serve the
    TRAINING-era value, 228.96, and that is still the defect this file exists to prevent:
    a 2025 lap answered with a different season's measurement. So the assertion is not
    "Vegas has a number" but "Vegas has ITS OWN number, and it is not 2023's".
    """
    training_era = agent._resolve_mean_sector_speed("Las Vegas", 2023)
    served = agent._resolve_mean_sector_speed("Las Vegas", 2025)

    assert training_era == pytest.approx(228.9645, abs=1e-3)
    assert not math.isnan(served), "the artefact no longer carries the hole"
    assert served != pytest.approx(training_era, abs=1e-3), (
        "2025 is being served the 2023 measurement — the cross-season substitution this "
        "whole fix removes"
    )
    assert served == pytest.approx(232.827, abs=1e-2)


@pytest.mark.data
@pytest.mark.skipif(not _HAS_RAW, reason="raw data absent")
def test_an_imputed_circuit_speed_is_flagged_in_the_artefact():
    """A fabricated number that looks like a measurement is how a model trains on one.

    Asserted on the artefact rather than through the agent, because the flag is the
    dataset's contract with every consumer, not this agent's private business.
    """
    import pandas as pd

    from src.f1_strat_manager.data_cache import get_data_root

    featured = pd.read_parquet(
        get_data_root() / "processed" / "laps_featured_2025.parquet",
        columns=["GP_Name", "mean_sector_speed", "mean_sector_speed_imputed"],
    )
    imputed = featured[featured["mean_sector_speed_imputed"]]

    assert set(imputed["GP_Name"]) == {"Las Vegas"}, "something else was quietly filled in"
    assert len(imputed) == 760
    unflagged_holes = (~featured["mean_sector_speed_imputed"]) & featured[
        "mean_sector_speed"
    ].isna()
    assert unflagged_holes.sum() == 0, "a missing circuit speed with no flag on it"


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
    # No exceptions left. Las Vegas 2025 was the one entry here — a hole in the artefact
    # rather than a resolution failure — and the regeneration filled it from the circuit's
    # own trap-offset, flagged. An empty list is the stronger assertion, and it is the one
    # that makes a NEW unresolved race fail loudly instead of joining a tolerated set.
    assert unresolved == [], (
        f"{len(unresolved)} of {checked} races resolve to NaN: {unresolved}"
    )


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
