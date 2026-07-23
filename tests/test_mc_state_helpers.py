"""PR-1 of the MC projection redesign (#552): stint-history facts + MC kwargs.

Two guarantees:

1. ``stint_history_flags`` answers Art. 30.5(m) questions honestly — positive
   evidence flips the flag to False, full visibility is required for True, and
   anything an invisible stint could hide comes back None (never a guess).
   Synthetic frames cover the logic everywhere; the real featured parquet
   (skipped on data-less CI runners) pins three hand-checked histories.

2. ``_run_mc_simulation`` accepts the new race-context kwargs and IGNORES
   them: passing them must be byte-identical to not passing them, because the
   legacy scoring path is golden-pinned and no caller threads context yet.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.simulation.stint_history import stint_history_flags

ROOT = Path(__file__).parent.parent

_HAS_DATA = (ROOT / "data" / "processed" / "laps_featured_2024.parquet").exists()
_skip_no_data = pytest.mark.skipif(
    not _HAS_DATA,
    reason="data/processed/ not present (CI runner without the HF dataset)",
)

_HAS_MODELS = (ROOT / "data" / "models" / "tire_degradation" / "routing_config.json").exists()
_skip_no_models = pytest.mark.skipif(
    not _HAS_MODELS,
    reason="data/models/ not present (CI runner without model weights)",
)


def _frame(rows: list[tuple]) -> pd.DataFrame:
    """Build a minimal laps frame from (driver, lap, stint, compound) tuples."""
    frame = pd.DataFrame(rows, columns=["Driver", "LapNumber", "Stint", "Compound"])
    return frame


# ---------------------------------------------------------------------------
# Synthetic frames — the logic, exhaustively
# ---------------------------------------------------------------------------


def test_two_stints_same_dry_compound_keeps_the_obligation_pending():
    laps = _frame([("VER", 1, 1, "MEDIUM"), ("VER", 10, 1, "MEDIUM"), ("VER", 20, 2, "MEDIUM")])
    flags = stint_history_flags(laps, "VER", 25)
    assert flags["stops_made"] == 1
    assert flags["compounds_used"] == ["MEDIUM"]
    assert flags["mandatory_stop_pending"] is True


def test_two_different_dry_compounds_satisfy_the_obligation():
    laps = _frame([("VER", 1, 1, "MEDIUM"), ("VER", 20, 2, "HARD")])
    flags = stint_history_flags(laps, "VER", 57)
    assert flags["mandatory_stop_pending"] is False


def test_wet_weather_compound_exempts_even_with_one_dry_compound():
    laps = _frame([("VER", 1, 1, "INTERMEDIATE"), ("VER", 20, 2, "SOFT")])
    flags = stint_history_flags(laps, "VER", 30)
    assert flags["mandatory_stop_pending"] is False


def test_lap_bound_excludes_later_compounds():
    laps = _frame([("VER", 1, 1, "MEDIUM"), ("VER", 30, 2, "HARD")])
    flags = stint_history_flags(laps, "VER", 10)
    assert flags["compounds_used"] == ["MEDIUM"]
    assert flags["stops_made"] == 0
    assert flags["mandatory_stop_pending"] is True


def test_invisible_stint_downgrades_pending_true_to_unknown():
    # Stint 2 never appears in the frame (its laps all failed the quality
    # gate), so the second compound may be hiding there: the honest answer
    # is None, while stops_made still counts from the stint NUMBER.
    laps = _frame([("VER", 1, 1, "MEDIUM"), ("VER", 30, 3, "MEDIUM")])
    flags = stint_history_flags(laps, "VER", 40)
    assert flags["stops_made"] == 2
    assert flags["mandatory_stop_pending"] is None


def test_invisible_stint_cannot_hide_positive_evidence():
    laps = _frame([("VER", 1, 1, "MEDIUM"), ("VER", 30, 4, "HARD")])
    flags = stint_history_flags(laps, "VER", 40)
    assert flags["stops_made"] == 3
    assert flags["mandatory_stop_pending"] is False


def test_empty_history_is_unknown_not_false():
    laps = _frame([("VER", 1, 1, "MEDIUM")])
    flags = stint_history_flags(laps, "HAM", 10)
    assert flags == {"stops_made": None, "compounds_used": [], "mandatory_stop_pending": None}


def test_missing_stint_column_still_reads_compounds():
    laps = pd.DataFrame(
        [("VER", 1, "MEDIUM"), ("VER", 20, "HARD")],
        columns=["Driver", "LapNumber", "Compound"],
    )
    flags = stint_history_flags(laps, "VER", 30)
    assert flags["stops_made"] is None
    assert flags["compounds_used"] == ["MEDIUM", "HARD"]
    assert flags["mandatory_stop_pending"] is False


def test_unknown_compound_strings_are_neither_evidence_nor_crash():
    laps = _frame([("VER", 1, 1, ""), ("VER", 2, 1, "TEST_UNKNOWN")])
    flags = stint_history_flags(laps, "VER", 5)
    assert flags["compounds_used"] == []
    assert flags["mandatory_stop_pending"] is None


def test_none_frame_is_unknown():
    assert stint_history_flags(None, "VER", 10)["mandatory_stop_pending"] is None


# ---------------------------------------------------------------------------
# Real featured parquet — three hand-checked histories (probed 2026-07-23)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def laps_2024() -> pd.DataFrame:
    # Direct featured read is fine HERE only: these tests consume nothing but
    # Driver/LapNumber/Stint/Compound. The augment_featured_laps rule exists
    # for Time_s/TrackStatus consumers; stint history needs neither.
    frame = pd.read_parquet(ROOT / "data" / "processed" / "laps_featured_2024.parquet")
    return frame


_HAS_RAW = (ROOT / "data" / "raw" / "2024" / "Lusail" / "laps.parquet").exists()
_skip_no_raw = pytest.mark.skipif(
    not _HAS_RAW,
    reason="data/raw/ not present (CI runner without the HF dataset)",
)


@pytest.fixture(scope="module")
def raw_lusail_2024() -> pd.DataFrame:
    frame = pd.read_parquet(ROOT / "data" / "raw" / "2024" / "Lusail" / "laps.parquet")
    return frame


@_skip_no_data
def test_sao_paulo_2024_ver_wet_race_is_exempt(laps_2024):
    gp = laps_2024[laps_2024["GP_Name"] == "São Paulo"]
    flags = stint_history_flags(gp, "VER", 69)
    assert flags["compounds_used"] == ["INTERMEDIATE"]
    assert flags["stops_made"] == 1
    assert flags["mandatory_stop_pending"] is False


@_skip_no_data
def test_lusail_2024_ver_single_compound_early_race_is_pending(laps_2024):
    gp = laps_2024[laps_2024["GP_Name"] == "Lusail"]
    flags = stint_history_flags(gp, "VER", 10)
    assert flags["compounds_used"] == ["MEDIUM"]
    assert flags["stops_made"] == 0
    assert flags["mandatory_stop_pending"] is True


@_skip_no_data
def test_lusail_2024_ver_counts_stops_through_invisible_stints(laps_2024):
    # The featured frame shows VER's stints 1 and 4 only (2 and 3 dropped by
    # the quality gate). Stint NUMBERS still say three stops happened, and
    # MEDIUM + HARD is positive evidence the obligation is satisfied.
    gp = laps_2024[laps_2024["GP_Name"] == "Lusail"]
    flags = stint_history_flags(gp, "VER", 57)
    assert flags["stops_made"] == 3
    assert flags["compounds_used"] == ["MEDIUM", "HARD"]
    assert flags["mandatory_stop_pending"] is False


@_skip_no_raw
def test_rsm_get_stint_flags_matches_the_pure_helper(raw_lusail_2024):
    # The RSM consumes the RAW per-race laps parquet (all laps, full stint
    # visibility: VER shows stints 1-4 here where the featured frame keeps
    # only 1 and 4), so on this path the invisible-stint None rarely fires.
    from src.simulation.race_state_manager import RaceStateManager

    rsm = RaceStateManager(
        raw_lusail_2024, driver_code="VER", team="Red Bull Racing", gp_name="Lusail", year=2024
    )
    flags = rsm.get_stint_flags(10)
    assert flags == stint_history_flags(raw_lusail_2024, "VER", 10)
    assert flags["compounds_used"] == ["MEDIUM"]
    assert flags["mandatory_stop_pending"] is True
    rival_flags = rsm.get_stint_flags(10, driver_code="HAM")
    assert rival_flags == stint_history_flags(raw_lusail_2024, "HAM", 10)


# ---------------------------------------------------------------------------
# MC kwargs — accepted and ignored, byte-identical output
# ---------------------------------------------------------------------------


def _canned_outputs():
    """Same canned near-cliff fixture as tests/test_strategy_goldens.py."""
    from src.agents.pace_agent import PaceOutput
    from src.agents.pit_strategy_agent import PitStrategyOutput
    from src.agents.race_situation_agent import RaceSituationOutput
    from src.agents.tire_agent import TireOutput

    pace = PaceOutput(
        lap_time_pred=91.0, delta_vs_prev=-0.2, delta_vs_median=0.3, ci_p10=90.5, ci_p90=91.8
    )
    tire = TireOutput(
        compound="MEDIUM",
        current_tyre_life=18,
        deg_rate=0.05,
        laps_to_cliff_p10=3.0,
        laps_to_cliff_p50=5.0,
        laps_to_cliff_p90=8.0,
        gp_name="",
    )
    situation = RaceSituationOutput(overtake_prob=0.2, sc_prob_3lap=0.10)
    pit = PitStrategyOutput(
        action="PIT_NOW",
        recommended_lap=20,
        compound_recommendation="HARD",
        stop_duration_p05=2.2,
        stop_duration_p50=2.8,
        stop_duration_p95=3.6,
        undercut_prob=0.55,
        undercut_target="VER",
        sc_reactive=False,
        reasoning="",
    )
    return pace, tire, situation, pit


@_skip_no_models
def test_mc_race_context_kwargs_are_accepted_and_ignored():
    from src.agents.strategy_orchestrator import _run_mc_simulation

    pace, tire, situation, pit = _canned_outputs()
    baseline = _run_mc_simulation(pace, tire, situation, pit, alpha=0.5)
    with_context = _run_mc_simulation(
        pace,
        tire,
        situation,
        pit,
        alpha=0.5,
        rivals=[{"driver": "HAM", "interval_to_driver_s": -1.8}],
        position=5,
        laps_remaining=20,
        pit_context={"mandatory_stop_pending": True},
    )
    assert with_context == baseline
