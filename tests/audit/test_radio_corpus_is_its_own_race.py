"""Guards for #825: a race's radio corpus must be that race's, not its country's.

The defect: `resolve_session` queried OpenF1 by `country_name` and took the first
main-Race session it got back. Italy holds two races and the United States holds
three, so `italy_monza/` was written with Imola's messages and both
`united_states_austin/` and `united_states_las_vegas/` with Miami's. The output
PATH had carried a circuit disambiguator from the start; the FETCH had not.

The consequence was not cosmetic. Monza 2025 has ZERO neutralised laps of its own
and Imola's corpus deploys a Safety Car on laps 29 and 46, so the stack was served
`sc_currently_active=True`, `sc_prob_3lap=1.0`, `threat_level=HIGH`, an
`overtake_prob` forced to 0.0 under Art. 55.8, and two suspended guard rails, for
an event that did not happen.

Two guards, deliberately at different levels: the first is pure and always runs;
the second reads the corpora on disk and skips when they are absent, because it
is the only thing that can catch a rebuild that silently reintroduces the defect.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.data_extraction.openf1.radio_dataset_builder import RadioDatasetBuilder

ROOT = Path(__file__).resolve().parents[2]
CORPUS_ROOT = ROOT / "data" / "processed" / "race_radios" / "2025"
RAW_ROOT = ROOT / "data" / "raw" / "2025"

# Corpus slug -> the raw folder whose race it must describe. Only the multi-race
# countries are listed: they are the ones the country-keyed fetch could confuse.
_MULTI_RACE_COUNTRIES = {
    "italy_imola": "Imola",
    "italy_monza": "Monza",
    "united_states_miami": "Miami_Gardens",
    "united_states_austin": "Austin",
    "united_states_las_vegas": "Las_Vegas",
}

_ITALY_SESSIONS = [
    {"session_name": "Race", "circuit_short_name": "Imola", "session_key": 9987},
    {"session_name": "Race", "circuit_short_name": "Monza", "session_key": 9999},
]


def test_a_named_circuit_picks_its_own_session_not_the_first():
    """The whole defect in one assertion."""
    picked = RadioDatasetBuilder._disambiguate_by_circuit(_ITALY_SESSIONS, 2025, "Italy", "Monza")
    assert picked["session_key"] == 9999


def test_the_first_session_is_still_wrong_for_the_second_race():
    """Guards the guard: Imola is genuinely `[0]`, so a passing Monza case is not luck."""
    picked = RadioDatasetBuilder._disambiguate_by_circuit(_ITALY_SESSIONS, 2025, "Italy", "Imola")
    assert picked["session_key"] == 9987
    assert _ITALY_SESSIONS[0]["session_key"] == 9987


def test_an_ambiguous_country_raises_rather_than_guessing():
    """Silently picking one is what shipped three races with another race's corpus."""
    with pytest.raises(ValueError, match="Race sessions"):
        RadioDatasetBuilder._disambiguate_by_circuit(_ITALY_SESSIONS, 2025, "Italy", None)


def test_a_circuit_the_country_does_not_hold_raises():
    with pytest.raises(ValueError, match="No Race session at circuit"):
        RadioDatasetBuilder._disambiguate_by_circuit(_ITALY_SESSIONS, 2025, "Italy", "Interlagos")


def test_a_single_race_country_needs_no_disambiguator():
    """Every other country must keep working without the new argument."""
    only = [{"session_name": "Race", "circuit_short_name": "Lusail", "session_key": 9850}]
    assert (
        RadioDatasetBuilder._disambiguate_by_circuit(only, 2025, "Qatar", None)["session_key"]
        == 9850
    )


@pytest.mark.skipif(
    not CORPUS_ROOT.is_dir() or not RAW_ROOT.is_dir(),
    reason="data/processed/race_radios or data/raw absent (CI runner without the HF dataset)",
)
@pytest.mark.parametrize("slug,raw_folder", sorted(_MULTI_RACE_COUNTRIES.items()))
def test_each_corpus_describes_the_race_it_is_filed_under(slug: str, raw_folder: str):
    """`total_laps` is the cheapest discriminator and it fails today for three of five.

    Imola runs 63 laps and Monza 53; Miami 57, Austin 56, Las Vegas 50. A corpus
    carrying the wrong race's `total_laps` is carrying the wrong race's messages,
    and this is the assertion a rebuild has to pass before the parquets are trusted.
    """
    rcm_path = CORPUS_ROOT / slug / "rcm.parquet"
    laps_path = RAW_ROOT / raw_folder / "laps.parquet"
    if not rcm_path.exists() or not laps_path.exists():
        pytest.skip(f"{slug} or {raw_folder} not present in this checkout")

    corpus = pd.read_parquet(rcm_path)
    if "total_laps" not in corpus.columns or corpus.empty:
        pytest.skip(f"{slug} carries no total_laps to check")

    corpus_laps = sorted({int(v) for v in corpus["total_laps"].dropna()})
    race_laps = int(pd.read_parquet(laps_path)["LapNumber"].max())

    assert corpus_laps == [race_laps], (
        f"{slug} carries total_laps={corpus_laps} but {raw_folder} ran {race_laps} laps: "
        "this corpus is another race's (#825)"
    )
