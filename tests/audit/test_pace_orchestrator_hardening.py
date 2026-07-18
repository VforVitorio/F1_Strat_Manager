"""Hardening tests for the pace agent, RaceStateManager, and strategy orchestrator —
#435 (pace self-fulfilling prev_lap_time), #476 (unvalidated predict_pace_tool LLM
input), and #433 (unvalidated expected_stint_end LLM free text).

Doctrine: no-LLM assertions only.

- The #435 section builds a real ``RaceStateManager`` off the raw Lusail 2025 laps
  parquet, merged with ``Prev_LapTime`` from the real featured parquet (the same
  join ``src/f1_strat_manager/laps_augment.py`` performs in the opposite direction),
  and reads the real dataclass method — no mocks, no LLM.
- The #476 section calls ``PaceAgent.run()`` (the real N06 XGBoost model) through
  ``predict_pace_tool.invoke()`` directly — deterministic inference, never a
  LangGraph ReAct loop or a live LLM client.
- The #433 section calls the pure ``_clamp_expected_stint_end`` helper directly
  (never a live orchestrator), but skips when ``data/`` is absent: importing
  ``strategy_orchestrator`` instantiates the tire agent, which loads model files.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).parent.parent.parent
RAW_LUSAIL = ROOT / "data" / "raw" / "2025" / "Lusail" / "laps.parquet"
FEATURED_2025 = ROOT / "data" / "processed" / "laps_featured_2025.parquet"

_HAS_RSM_DATA = RAW_LUSAIL.exists() and FEATURED_2025.exists()

_HAS_PACE_MODEL = (
    (ROOT / "data" / "models" / "lap_time" / "xgb_laptime_delta_final.json").exists()
    and (ROOT / "data" / "processed" / "feature_manifest_laptime.json").exists()
    and (
        ROOT / "data" / "processed" / "circuit_clustering" / "circuit_clusters_k4_2025.parquet"
    ).exists()
    and FEATURED_2025.exists()
)


# =============================================================================
# #435 — RaceStateManager.get_driver_state must emit the REAL previous-lap
# time (Prev_LapTime from the featured parquet), not the current lap's time
# reused as a stand-in.
# =============================================================================


@pytest.fixture(scope="module")
def lusail_rsm_and_reference_row():
    """A real RaceStateManager for Lusail 2025, plus the row used to check it.

    ``data/raw/2025/Lusail/laps.parquet`` satisfies RaceStateManager's raw-shape
    contract (LapTime/Time/TrackStatus/... as REQUIRED_LAPS_COLUMNS expects) but
    has no Prev_LapTime column — that is an N04-engineered feature that only
    lives in the featured parquet. Merging it in on (Driver, LapNumber) mirrors
    laps_augment.augment_featured_laps's own raw<->featured join, just in the
    other direction, and produces the same laps_df shape the strategy pipeline
    relies on in production.
    """
    if not _HAS_RSM_DATA:
        pytest.skip(
            "needs data/raw/2025/Lusail/laps.parquet and "
            "data/processed/laps_featured_2025.parquet (data/ comes from HF, not git)"
        )

    from src.simulation.race_state_manager import RaceStateManager

    raw = pd.read_parquet(RAW_LUSAIL)
    raw["LapNumber"] = raw["LapNumber"].astype(int)

    featured = pd.read_parquet(
        FEATURED_2025, columns=["GP_Name", "Driver", "LapNumber", "Prev_LapTime"]
    )
    featured = featured[featured["GP_Name"] == "Lusail"].copy()
    featured["LapNumber"] = featured["LapNumber"].astype(int)

    merged = raw.merge(
        featured[["Driver", "LapNumber", "Prev_LapTime"]],
        on=["Driver", "LapNumber"],
        how="left",
    )

    candidates = merged.dropna(subset=["Prev_LapTime"])
    if candidates.empty:
        pytest.skip("no Lusail 2025 lap has a non-NaN Prev_LapTime after the join")

    # Prefer VER lap 10 (the task's suggested repro) when it survived the join;
    # otherwise any non-NaN candidate demonstrates the same fix.
    ver_lap10 = candidates[(candidates["Driver"] == "VER") & (candidates["LapNumber"] == 10)]
    row = ver_lap10.iloc[0] if not ver_lap10.empty else candidates.iloc[0]

    team = merged.loc[merged["Driver"] == row["Driver"], "Team"].iloc[0]
    rsm = RaceStateManager(
        merged, driver_code=row["Driver"], team=team, gp_name="Lusail", year=2025
    )
    return rsm, row


def test_prev_lap_time_matches_the_real_prev_laptime_and_differs_from_current(
    lusail_rsm_and_reference_row,
):
    """#435: prev_lap_time must be the row's real Prev_LapTime, not lap_time_s.

    Before the fix, PaceAgent.run_from_state fed lap_time_s (this lap's own time)
    back in as the "previous" lap, so the model chased its own last prediction.
    get_driver_state must now expose the real preceding-lap value so the agent
    can stop doing that.
    """
    rsm, row = lusail_rsm_and_reference_row
    state = rsm.get_lap_state(int(row["LapNumber"]))
    driver = state["driver"]

    assert driver["prev_lap_time"] == pytest.approx(float(row["Prev_LapTime"]), abs=1e-6)
    assert driver["prev_lap_time"] != driver["lap_time_s"]


# =============================================================================
# #476 — predict_pace_tool must refuse an off-roster driver number or an
# out-of-range lap instead of predicting from data that cannot exist.
# =============================================================================


@pytest.fixture(scope="module")
def pace_agent_and_valid_driver():
    """The real, singleton PaceAgent plus a genuinely-on-track Lusail 2025 car
    number, discovered from the agent's own loaded reference data rather than
    guessed — so the "still works for a real driver" control case cannot
    accidentally be wrong about who was racing.
    """
    if not _HAS_PACE_MODEL:
        pytest.skip(
            "needs data/models/lap_time/, data/processed/feature_manifest_laptime.json, "
            "data/processed/circuit_clustering/circuit_clusters_k4_2025.parquet and "
            "data/processed/laps_featured_2025.parquet (data/ comes from HF, not git)"
        )

    from src.agents.pace_agent import _get_default_pace_agent

    agent = _get_default_pace_agent()
    ref = agent.laps_ref
    lusail_2025 = ref[(ref["GP_Name"] == "Lusail") & (ref["Year"] == 2025)]
    valid_numbers = lusail_2025["DriverNumber"].dropna().astype(int)
    if valid_numbers.empty:
        pytest.skip("no DriverNumber found for Lusail 2025 in laps_ref")
    return agent, int(valid_numbers.iloc[0])


def _pace_tool_kwargs(**overrides) -> dict:
    """Base kwargs for predict_pace_tool.invoke() — all 19 params, overridable.

    Values are plausible but not required to be exact: everything except
    driver_number/lap_number/gp_name/year/total_laps degrades gracefully
    (unknown team/compound fall back to encoding defaults), so the guard under
    test is isolated from the rest of the feature row.
    """
    kwargs = dict(
        driver_number=1,
        lap_number=10,
        stint=1,
        tyre_life=5,
        compound="MEDIUM",
        position=5,
        team="Red Bull Racing",
        laps_since_pit=5,
        fuel_load=0.8,
        year=2025,
        prev_lap_time=85.0,
        prev_tyre_life=4,
        prev_speed_st=320.0,
        air_temp=28.0,
        track_temp=38.0,
        humidity=50.0,
        rainfall=False,
        total_laps=57,
        gp_name="Lusail",
    )
    kwargs.update(overrides)
    return kwargs


def test_predict_pace_tool_refuses_a_driver_not_on_track(pace_agent_and_valid_driver):
    """#476: an invented car number for a real GP/year must error, not predict."""
    from src.agents.pace_agent import predict_pace_tool

    _agent, _valid_number = pace_agent_and_valid_driver
    off_roster_number = -999  # no real car has ever carried this number

    result = predict_pace_tool.invoke(_pace_tool_kwargs(driver_number=off_roster_number))

    assert "error" in result, f"expected a refusal, got: {result}"
    assert "REFUSED" in result["error"]
    assert str(off_roster_number) in result["error"]


def test_predict_pace_tool_refuses_an_out_of_range_lap(pace_agent_and_valid_driver):
    """#476: a lap far beyond total_laps must error, not predict."""
    from src.agents.pace_agent import predict_pace_tool

    _agent, valid_number = pace_agent_and_valid_driver

    result = predict_pace_tool.invoke(
        _pace_tool_kwargs(driver_number=valid_number, lap_number=9999, total_laps=57)
    )

    assert "error" in result, f"expected a refusal, got: {result}"
    assert "REFUSED" in result["error"]


def test_predict_pace_tool_still_computes_for_a_real_driver_in_range(
    pace_agent_and_valid_driver,
):
    """The guard must not buy safety by refusing everyone — a real, in-range
    driver number must still run inference and return a prediction.
    """
    from src.agents.pace_agent import predict_pace_tool

    _agent, valid_number = pace_agent_and_valid_driver

    result = predict_pace_tool.invoke(
        _pace_tool_kwargs(driver_number=valid_number, lap_number=10, total_laps=57)
    )

    assert "error" not in result, f"a live, in-range driver was refused: {result}"
    assert "lap_time_pred" in result


# =============================================================================
# #433 — expected_stint_end must be grounded against pit_lap_target plus the
# N26 cliff / Pirelli stint capacity, not passed through as raw LLM free text.
# The clamp is the pure _clamp_expected_stint_end helper; these tests call it
# directly (no live orchestrator). They skip when data/ is absent because
# importing strategy_orchestrator instantiates the tire agent, which loads
# data/models/tire_degradation/ (fetched from HF Hub, not committed).
# =============================================================================


@pytest.fixture(scope="module")
def clamp_fn():
    """The pure _clamp_expected_stint_end helper, imported lazily.

    Importing it pulls in strategy_orchestrator, whose import chain instantiates the
    tire agent and loads data/models/tire_degradation/routing_config.json — absent in
    a bare checkout (data/ comes from HF Hub). Skip rather than fail there, like every
    other data-dependent test in this suite; the pure clamp still runs locally.
    """
    routing_cfg = ROOT / "data" / "models" / "tire_degradation" / "routing_config.json"
    if not routing_cfg.exists():
        pytest.skip(
            "importing strategy_orchestrator instantiates the tire agent, which needs "
            "data/models/tire_degradation/ (data/ comes from HF, not git)"
        )
    from src.agents.strategy_orchestrator import _clamp_expected_stint_end

    return _clamp_expected_stint_end


def test_expected_stint_end_is_clamped_to_the_pit_and_cliff_anchor(clamp_fn):
    """#433: an absurd 57 must be pulled back to the pit_lap + cliff/capacity anchor.

    pit_lap_target=7, HARD capacity=38 (pit_strategy_agent._STINT_CAPACITY_LAPS),
    cliff_p50=20.0 -> anchor = 7 + min(20.0, 38) = 27. |57 - 27| = 30 > 3, so the
    LLM's 57 must be discarded in favour of the anchor.
    """
    result = clamp_fn(
        llm_stint_end=57,
        pit_lap_target=7,
        compound_next="HARD",
        cliff_p50=20.0,
        total_laps=57,
    )

    assert result == 27
    assert result <= 45


def test_expected_stint_end_within_band_is_kept_as_the_llm_reported_it(clamp_fn):
    """A plausible LLM value close to the anchor must survive unchanged."""
    # anchor is 27; |29-27| = 2 <= 3
    result = clamp_fn(
        llm_stint_end=29,
        pit_lap_target=7,
        compound_next="HARD",
        cliff_p50=20.0,
        total_laps=57,
    )

    assert result == 29


def test_expected_stint_end_anchor_is_clamped_to_total_laps(clamp_fn):
    """The anchor itself must never exceed the race's actual lap count."""
    # 50 + min(38, 38) = 88 would exceed a 57-lap race
    result = clamp_fn(
        llm_stint_end=95,
        pit_lap_target=50,
        compound_next="HARD",
        cliff_p50=38.0,
        total_laps=57,
    )

    assert result == 57


def test_expected_stint_end_passes_through_unclamped_when_no_anchor_is_available(clamp_fn):
    """Without pit_lap_target/compound_next/cliff_p50 there is no anchor to
    ground against, so the LLM's raw value must pass through rather than
    inventing one.
    """
    result = clamp_fn(
        llm_stint_end=57,
        pit_lap_target=None,
        compound_next=None,
        cliff_p50=None,
        total_laps=57,
    )

    assert result == 57
