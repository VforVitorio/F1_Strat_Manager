"""N06 is fed the circuit's own mean sector speed, not the speed trap (#797).

`mean_sector_speed` is a property of the CIRCUIT: the featured parquet holds exactly
one value per GP. `_compute_derived` used to substitute `prev_speed_st` whenever none
was supplied, and `run_from_state` never supplied one, so on the path every real race
takes the model received a different physical quantity on every lap.

What these pin is the pair of rules that makes the fix a fix rather than a new guess:
the value served must be the value fitted, and a circuit that does not resolve must
reach the model as MISSING rather than as a substituted reading.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).parent.parent.parent
_HAS_ARTEFACTS = (
    ROOT / "data" / "models" / "lap_time" / "xgb_laptime_delta_final.json"
).exists() and (ROOT / "data" / "processed" / "laps_featured_2025.parquet").exists()

pytestmark = pytest.mark.skipif(
    not _HAS_ARTEFACTS, reason="N06 weights or the featured parquet are absent"
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


# --- the value served is the value fitted ------------------------------------


def test_every_circuit_resolves_to_the_value_the_parquet_trained_on(agent):
    """Not "close to", the same number. The map is read from the training artefact.

    Asserted over every GP rather than one, because a lookup that works for the
    circuit somebody tested and silently misses the rest is the failure mode a
    single-case test invites.
    """
    from src.f1_strat_manager.data_cache import get_data_root

    trained = (
        pd.read_parquet(
            get_data_root() / "processed" / "laps_featured_2025.parquet",
            columns=["GP_Name", "mean_sector_speed"],
        )
        .dropna()
        .drop_duplicates("GP_Name")
        .set_index("GP_Name")["mean_sector_speed"]
    )

    assert len(trained) > 0, "the training artefact carried no circuit speeds"
    for gp, expected in trained.items():
        assert agent._resolve_mean_sector_speed(str(gp)) == pytest.approx(float(expected)), gp


def test_the_row_fed_to_n06_carries_the_circuit_value_not_the_speed_trap(agent):
    """The regression itself: the trap reading must not reach the feature again."""
    row = _feature_row(agent, gp_name="Lusail", prev_speed_st=300.0)

    served = float(row["mean_sector_speed"].iloc[0])
    assert served == pytest.approx(agent.circuit_mean_sector_speed["Lusail"])
    assert served != pytest.approx(300.0)
    # Prev_SpeedST is still its own feature and still carries the trap.
    assert float(row["Prev_SpeedST"].iloc[0]) == pytest.approx(300.0)


def test_a_full_event_name_resolves_through_the_slug(agent):
    """gp_name arrives in either keyspace depending on the caller (#448, #450)."""
    assert agent._resolve_mean_sector_speed("Qatar Grand Prix") == pytest.approx(
        agent.circuit_mean_sector_speed["Lusail"]
    )


# --- an unresolvable circuit stays unknown -----------------------------------


def test_an_unknown_circuit_yields_nan_and_never_a_substituted_reading(agent, caplog):
    """The rule the bug broke: unknown data must not become a number the model can use.

    NaN is in-distribution for XGBoost, which routes a missing feature through its
    sparse-aware split. A plausible-looking speed is not: it answers a different
    question with full confidence, which is exactly what #797 was.
    """
    with caplog.at_level(logging.WARNING):
        value = agent._resolve_mean_sector_speed("Nurburgring")

    assert math.isnan(value)
    assert any("no trained mean sector speed" in r.message for r in caplog.records)


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


def test_the_envelope_now_separates_a_circuit_n06_was_fitted_on_from_one_it_was_not(agent):
    """The bound is back and it discriminates, which is the point of restoring it.

    A first version of this test asserted that EVERY served circuit falls inside the
    bound, and Monza refuted it. That assertion was wrong, not the code: the bound is
    the 2023-2024 range and the served values are the 2025 measurements, so they are
    different sets and a 2025 circuit may legitimately sit outside. Monza 2025 does,
    at 317.24 against a fitted maximum of 314.97, and saying so is exactly the job the
    bound was restored to do.

    So what is pinned here is the discrimination, not a blanket. A bound that flagged
    everything, or nothing, would pass a coverage check and tell nobody anything.
    """
    from src.agents.pace_agent import _N06_ENVELOPE, _N06_TRAINED_BOUNDS

    lower, upper = _N06_TRAINED_BOUNDS["mean_sector_speed"]

    inside = {gp: v for gp, v in agent.circuit_mean_sector_speed.items() if lower <= v <= upper}
    outside = {
        gp: v for gp, v in agent.circuit_mean_sector_speed.items() if not lower <= v <= upper
    }

    assert inside, "no served circuit is in range: the bound and the feed disagree"
    assert len(inside) > len(outside), (
        f"{len(outside)} of {len(agent.circuit_mean_sector_speed)} circuits fall outside; "
        f"a bound that rejects most of what it is fed is describing the wrong quantity"
    )

    # And it must actually fire on the one that is out, through the real envelope.
    for gp, value in outside.items():
        verdict = _N06_ENVELOPE.check({"mean_sector_speed": value})
        assert "mean_sector_speed" in verdict.violations, gp
