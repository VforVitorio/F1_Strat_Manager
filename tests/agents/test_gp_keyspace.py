"""Every race directory resolves in every gp_name-keyed lookup.

The fourth occurrence of one defect (#448, #450, #797, and the 2026-08-04 sweep in
`documents/audits/PR3_GP_KEYSPACE_SWEEP.md`): a GP is spelled four ways across this project
and a lookup keyed by one of them is queried with another, so it misses and takes its
fallback without a word. Each previous fix repaired the site that hurt and left the rest.

This file is the enumeration that ends that: every race directory on disk, against every
consumer, so a FIFTH mismatch in any season fails the suite instead of shipping. Testing
Miami alone would pass the moment Miami is fixed and say nothing about the next race.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.f1_strat_manager.gp_slugs import normalise_gp_key, resolve_gp_key
from tests.conftest import HAS_TIRE_MODELS as _HAS_MODELS

ROOT = Path(__file__).parent.parent.parent
_DATA = ROOT / "data"
_RAW = _DATA / "raw"
_PROCESSED = _DATA / "processed"
_COMPOUNDS = _DATA / "tire_compounds_by_race.json"

# The three compounds a dry race can run. HARD and MEDIUM matter most here: the pit agent's
# SOFT fallback is 5, and 2025 Miami's SOFT is C5, so a SOFT-only check reports that site as
# healthy while it is broken. The first version of the sweep did exactly that.
_DRY_COMPOUNDS = ("SOFT", "MEDIUM", "HARD")


def _races() -> list[tuple[int, str, str]]:
    """(year, folder, gp_name) for every race directory carrying a metadata.json.

    `gp_name` is read the way `replay_engine._parse_meta` reads it, because that is the
    string the agents are actually handed.
    """
    found: list[tuple[int, str, str]] = []
    for year_dir in sorted(p for p in _RAW.iterdir() if p.is_dir() and p.name.isdigit()):
        for race_dir in sorted(p for p in year_dir.iterdir() if p.is_dir()):
            meta_path = race_dir / "metadata.json"
            if not meta_path.exists():
                continue
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            found.append((int(year_dir.name), race_dir.name, meta.get("gp_name", race_dir.name)))
    return found


# --- the resolver itself, hermetic -------------------------------------------


def test_neither_resolver_alone_covers_the_four_spellings():
    """Why `resolve_gp_key` chains both instead of picking one.

    Asserting the chain works is not enough: someone simplifying it back to a single
    resolver needs the reason in front of them, and both halves of that reason are here.
    """
    from src.f1_strat_manager.gp_slugs import slug_from_event_name

    assert slug_from_event_name("Miami Gardens") is None, "the slug resolver alone suffices"
    assert normalise_gp_key("Qatar Grand Prix") != "Lusail", "the normaliser alone suffices"


@pytest.mark.parametrize(
    "spelling",
    ["Miami", "Miami Gardens", "Miami_Gardens", "Miami Grand Prix"],
)
def test_every_spelling_resolves_to_the_stored_key(spelling):
    """All four forms find the one key a table actually holds."""
    assert resolve_gp_key({"Miami": 1}, spelling) == "Miami"


def test_an_unknown_name_passes_through_unchanged():
    """So the caller's own fallback and warning still fire, unchanged."""
    assert resolve_gp_key({"Miami": 1}, "Nowhere") == "Nowhere"
    assert resolve_gp_key({}, "") == ""


def test_a_table_holding_two_spellings_keeps_both_reachable():
    """The pooled clustering artefact carries Miami twice; re-keying would drop one.

    This is why the fix resolves on the QUERY side. If someone converts it to a load-time
    re-key, this assertion is what tells them a row went missing.
    """
    both = {"Miami": 1, "Miami Gardens": 2}
    assert resolve_gp_key(both, "Miami Gardens") == "Miami Gardens"
    assert resolve_gp_key(both, "Miami") == "Miami"


# --- the enumeration, over real data -----------------------------------------


@pytest.mark.data
@pytest.mark.skipif(not _RAW.exists() or not _COMPOUNDS.exists(), reason="raw data absent")
def test_every_race_resolves_in_the_compound_allocation():
    """The JSON behind three agents: tire, pit and race-situation.

    A miss is invisible from the return value (the fallback is itself a valid Cx), so this
    asks the artefact whether the key resolved, not what the function returned.
    """
    alloc = json.loads(_COMPOUNDS.read_text(encoding="utf-8"))

    unresolved = []
    for year, folder, gp_name in _races():
        year_data = alloc.get(str(year), {})
        if not year_data:
            continue  # a season the allocation does not cover is not a keyspace failure
        if resolve_gp_key(year_data, gp_name) not in year_data:
            unresolved.append(f"{year} {gp_name!r} (data/raw/{year}/{folder})")

    assert unresolved == [], (
        f"{len(unresolved)} race(s) miss tire_compounds_by_race.json and would silently "
        f"take the compound fallback: {unresolved}"
    )


@pytest.mark.data
@pytest.mark.skipif(
    not _RAW.exists() or not (_PROCESSED / "circuit_clustering").exists(),
    reason="clustering artefacts absent",
)
@pytest.mark.parametrize(
    "artefact", ["circuit_clusters_k4.parquet", "circuit_clusters_k4_2025.parquet"]
)
def test_every_race_resolves_in_the_cluster_maps(artefact):
    """Pooled for tire/race-situation, 2025 for pace. Both are queried by gp_name.

    The 2025 map covers one season, so only that season's races are asserted against it.
    """
    path = _PROCESSED / "circuit_clustering" / artefact
    if not path.exists():
        pytest.skip(f"{artefact} absent")
    keys = set(pd.read_parquet(path, columns=["GP_Name"])["GP_Name"].astype(str))
    seasons = {2025} if "2025" in artefact else None

    unresolved = [
        f"{year} {gp_name!r}"
        for year, _folder, gp_name in _races()
        if (seasons is None or year in seasons) and resolve_gp_key(keys, gp_name) not in keys
    ]
    assert unresolved == [], f"{artefact}: {len(unresolved)} race(s) unresolved: {unresolved}"


@pytest.mark.data
@pytest.mark.skipif(
    not _RAW.exists() or not (_PROCESSED / "laps_featured_2025.parquet").exists(),
    reason="featured parquet absent",
)
def test_every_2025_race_resolves_in_the_pace_reference_frame():
    """`_session_median` masks this frame by name; an unresolved name yields no laps."""
    names = set(
        pd.read_parquet(_PROCESSED / "laps_featured_2025.parquet", columns=["GP_Name"])[
            "GP_Name"
        ].astype(str)
    )
    unresolved = [
        f"2025 {gp_name!r}"
        for year, _folder, gp_name in _races()
        if year == 2025 and resolve_gp_key(names, gp_name) not in names
    ]
    assert unresolved == [], (
        f"{len(unresolved)} race(s) would give N31 delta_vs_median=None for a whole "
        f"race: {unresolved}"
    )


# --- the consumers, end to end -----------------------------------------------


@pytest.mark.data
@pytest.mark.skipif(
    not _HAS_MODELS or not _COMPOUNDS.exists(), reason="importing the agents reads data/models/"
)
def test_the_three_compound_consumers_agree_across_every_race_and_compound():
    """The functions themselves, not just their tables.

    All three read the same JSON and each degrades differently on a miss: tire returns the
    fallback Cx, pit an int that is 2 low on HARD, race-situation the RELATIVE name. Run
    against the metadata spelling, they must equal what the slug spelling gives.
    """
    from src.agents.pit_strategy_agent import _compound_to_id
    from src.agents.race_situation_agent import _abs_compound
    from src.agents.tire_agent import _compound_name_to_id

    alloc = json.loads(_COMPOUNDS.read_text(encoding="utf-8"))

    disagreements = []
    for year, _folder, gp_name in _races():
        year_data = alloc.get(str(year), {})
        stored = resolve_gp_key(year_data, gp_name)
        if stored not in year_data:
            continue  # covered by the allocation test above; not this one's job
        for compound in _DRY_COMPOUNDS:
            for label, fn in (
                ("tire", _compound_name_to_id),
                ("pit", _compound_to_id),
                ("rsa", _abs_compound),
            ):
                if fn(compound, gp_name, year) != fn(compound, stored, year):
                    disagreements.append(f"{label} {year} {gp_name!r} {compound}")

    assert disagreements == [], (
        f"{len(disagreements)} consumer/race/compound combinations answer differently for "
        f"the metadata spelling than for the stored one: {disagreements[:10]}"
    )
