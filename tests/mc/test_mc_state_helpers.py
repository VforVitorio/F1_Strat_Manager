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
from tests.conftest import skip_no_tire_models as _skip_no_models
from tests.mc.canned_outputs import canned_outputs as _canned_outputs

ROOT = Path(__file__).parent.parent.parent

_HAS_DATA = (ROOT / "data" / "processed" / "laps_featured_2024.parquet").exists()
_skip_no_data = pytest.mark.skipif(
    not _HAS_DATA,
    reason="data/processed/ not present (CI runner without the HF dataset)",
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


@_skip_no_models
def test_race_context_without_rivals_still_takes_the_legacy_path():
    """The kwargs alone change nothing — only a usable rivals list switches paths.

    This test pinned a contract that has since moved. When PR-1 added the kwargs
    they were accepted and ignored, so passing a rivals list had to be a no-op;
    PR-4 gave that list meaning and made it the dispatch key. The assertion now
    covers what is still true: race context with no rivals is legacy scoring.

    Worth remembering how it surfaced. It failed only on a machine holding the
    dataset, because ``_skip_no_models`` hides it on CI — the same shape as the
    voice-retirement test that pinned a retired contract and only broke on the
    promotion. A test that guards a contract must fail where the contract lives.
    """
    from src.agents.strategy_orchestrator import _run_mc_simulation

    pace, tire, situation, pit = _canned_outputs()
    baseline = _run_mc_simulation(pace, tire, situation, pit, alpha=0.5)
    with_context = _run_mc_simulation(
        pace,
        tire,
        situation,
        pit,
        alpha=0.5,
        position=5,
        laps_remaining=20,
        pit_context={"mandatory_stop_pending": True},
    )
    assert with_context == baseline


@_skip_no_models
def test_rivals_with_no_usable_gap_cannot_conjure_a_projection():
    """A list of cars whose intervals are all unknown is not race context.

    It is truthy, so it used to route into the projection, which then counted
    zero rivals and reported P1 with no uncertainty — "you will finish first",
    fabricated from nothing. Unknown gaps mean the projection has no geometry to
    work with, and the honest fallback is the legacy scoring it would have used
    had the list been empty.
    """
    from src.agents.strategy_orchestrator import _run_mc_simulation

    pace, tire, situation, pit = _canned_outputs()
    baseline = _run_mc_simulation(pace, tire, situation, pit, alpha=0.5)
    blind = _run_mc_simulation(
        pace,
        tire,
        situation,
        pit,
        alpha=0.5,
        rivals=[
            {"driver": "HAM", "interval_to_driver_s": None},
            {"driver": "VER", "interval_to_driver_s": None},
        ],
        position=5,
        laps_remaining=20,
    )
    assert blind == baseline


# ---------------------------------------------------------------------------
# Keyspace and hazard floor (final-audit F3-6, F3-7)
# ---------------------------------------------------------------------------


# Circuits this repo spells more than one way: underscored folder forms, the
# country name 2023 filed Barcelona under, Miami's three variants, and the two
# that carry diacritics and lose them through a non-UTF-8 console.
_CIRCUIT_SPELLINGS = (
    ("Barcelona", "Spain"),
    ("Miami", "Miami Gardens", "Miami_Gardens"),
    ("Lusail", "Qatar Grand Prix"),
    ("Yas Island", "Yas_Island"),
    ("São Paulo", "Sao Paulo", "Sao_Paulo"),
    ("Montréal", "Montreal"),
)


def test_every_spelling_of_a_circuit_finds_the_same_hazard():
    """Three keyspaces meet in this repo, and a hazard miss used to be silent.

    Unlike the traversal lookup, which at least warns when it falls back, a
    hazard miss quietly returned the pooled rate — a table that looks populated
    while every lookup misses is the #448 failure exactly. Reads the committed
    measured tables, so it runs everywhere.
    """
    from src.agents.position_projection import measured_neutralisation_rate

    for spellings in _CIRCUIT_SPELLINGS:
        hazards = {round(measured_neutralisation_rate(name), 6) for name in spellings}
        assert len(hazards) == 1, f"{spellings} disagree on hazard: {hazards}"


@_skip_no_models
def test_every_spelling_of_a_circuit_finds_the_same_traversal():
    """Same keyspace check for the per-circuit pit-lane traversal.

    Separate from the hazard test because this table lives in ``data/models/``,
    which is distributed through the Hugging Face Hub rather than git — so a CI
    runner without the weights has no table to look anything up in.
    """
    from src.agents.position_projection import traversal_seconds

    for spellings in _CIRCUIT_SPELLINGS:
        traversals = {traversal_seconds(name) for name in spellings}
        assert len(traversals) == 1, f"{spellings} disagree on traversal: {traversals}"
        assert None not in traversals, f"{spellings} has no traversal at all"


def test_a_circuit_that_has_never_thrown_a_safety_car_is_not_given_a_zero_rate():
    """Monza and Budapest measure exactly zero onsets, and zero is not a rate.

    A zero drives q_f to 0, which tells the decision layer that no future
    neutralisation can ever cover a stop and biases the terminal liability
    upward on every lap. Monza is also the archetypal Art. 55.17 circuit, so
    that is the worst possible place to lose the term. A zero count means "not
    seen here", not "cannot happen here".
    """
    from src.agents.position_projection import measured_neutralisation_rate

    pooled = measured_neutralisation_rate(None)
    for quiet_circuit in ("Monza", "Budapest"):
        assert measured_neutralisation_rate(quiet_circuit) == pooled

    # A circuit that HAS thrown them keeps its own measured, higher rate.
    assert measured_neutralisation_rate("Melbourne") > pooled
