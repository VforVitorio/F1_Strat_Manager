"""Every round the menu offers resolves to a folder the dataset really has (#1116).

`ensure_race` globbed the menu's own label, and `snapshot_download` on a pattern
that matches nothing returns without raising. So picking "Mexico City" fetched
zero files, raised nothing, and left the strategy layer degrading against an
empty directory: the same defect the lazy fetch was added to close, surviving it
for six of the 2025 rounds. A menu offering 24 races where 18 work is the same
defect wearing a different face.

`DATASET_FOLDERS` is the repository's real listing, read from the HuggingFace API
on 2026-08-27. It is recorded rather than fetched so this runs in CI without a
network, and `test_the_recorded_listing_matches_the_calendar_shape` is what fails
if the recording drifts from the calendars the menu reads.
"""

from __future__ import annotations

import pytest

from src.arcade.config import get_gp_names
from src.arcade.views import last_round
from src.f1_strat_manager.data_cache import race_folder

YEARS = (2023, 2024, 2025)

# `data/raw/{year}/{folder}/` in VforVitorio/f1-strategy-dataset. Note the
# underscores, which is what the label-verbatim glob missed, and Miami, which is
# `Miami` in 2023 and 2024 and `Miami_Gardens` in 2025.
DATASET_FOLDERS: dict[int, frozenset[str]] = {
    2023: frozenset(
        {
            "Austin",
            "Baku",
            "Barcelona",
            "Budapest",
            "Imola",
            "Jeddah",
            "Las_Vegas",
            "Lusail",
            "Marina_Bay",
            "Melbourne",
            "Mexico_City",
            "Miami",
            "Monaco",
            "Montréal",
            "Monza",
            "Sakhir",
            "Shanghai",
            "Silverstone",
            "Spa-Francorchamps",
            "Spielberg",
            "Suzuka",
            "São_Paulo",
            "Yas_Island",
            "Zandvoort",
        }
    ),
    2024: frozenset(
        {
            "Austin",
            "Baku",
            "Barcelona",
            "Budapest",
            "Imola",
            "Jeddah",
            "Las_Vegas",
            "Lusail",
            "Marina_Bay",
            "Melbourne",
            "Mexico_City",
            "Miami",
            "Monaco",
            "Montréal",
            "Monza",
            "Sakhir",
            "Shanghai",
            "Silverstone",
            "Spa-Francorchamps",
            "Spielberg",
            "Suzuka",
            "São_Paulo",
            "Yas_Island",
            "Zandvoort",
        }
    ),
    2025: frozenset(
        {
            "Austin",
            "Baku",
            "Barcelona",
            "Budapest",
            "Imola",
            "Jeddah",
            "Las_Vegas",
            "Lusail",
            "Marina_Bay",
            "Melbourne",
            "Mexico_City",
            "Miami_Gardens",
            "Monaco",
            "Montréal",
            "Monza",
            "Sakhir",
            "Shanghai",
            "Silverstone",
            "Spa-Francorchamps",
            "Spielberg",
            "Suzuka",
            "São_Paulo",
            "Yas_Island",
            "Zandvoort",
        }
    ),
}


@pytest.mark.parametrize("year", YEARS)
def test_every_round_the_menu_offers_resolves_to_a_real_folder(year: int) -> None:
    """The whole point, per season. RED for six 2025 rounds before the resolver.

    Named individually on failure, because "some races are broken" is the state
    this replaces and it took a live launch to notice.
    """
    folders = DATASET_FOLDERS[year]
    unresolved = [
        f"R{rnd} {label!r} -> {race_folder(year, label)!r}"
        for rnd, label in sorted(get_gp_names(year).items())
        if race_folder(year, label) not in folders
    ]
    assert not unresolved, f"{year}: {len(unresolved)} rounds fetch nothing: {unresolved}"


def test_all_seventy_rounds_resolve() -> None:
    """The number, stated so a regression is a number and not a feeling."""
    total = sum(len(get_gp_names(year)) for year in YEARS)
    resolved = sum(
        1
        for year in YEARS
        for label in get_gp_names(year).values()
        if race_folder(year, label) in DATASET_FOLDERS[year]
    )
    assert (total, resolved) == (70, 70)


def test_a_spaced_label_becomes_the_underscored_folder() -> None:
    """The rule that covers 68 of the 70, asserted on its own.

    This is what the label-verbatim glob got wrong, and every one of these is a
    real round of a real season.
    """
    assert race_folder(2025, "Mexico City") == "Mexico_City"
    assert race_folder(2025, "Las Vegas") == "Las_Vegas"
    assert race_folder(2025, "Marina Bay") == "Marina_Bay"
    assert race_folder(2025, "Yas Island") == "Yas_Island"
    assert race_folder(2025, "Melbourne") == "Melbourne", "a single word is left alone"


def test_the_alias_is_scoped_to_the_season_that_needs_it() -> None:
    """Miami is `Miami` in 2023 and 2024 and `Miami_Gardens` in 2025.

    An unscoped alias broke the two earlier seasons while fixing the later one,
    which is why the table is keyed by year: 68 of 70 became 68 of 70 again with
    a different pair failing.
    """
    assert race_folder(2023, "Miami") == "Miami"
    assert race_folder(2024, "Miami") == "Miami"
    assert race_folder(2025, "Miami") == "Miami_Gardens"


# Folders the dataset carries for a season whose calendar does not list them.
# 2023 ran 22 rounds: Imola was cancelled for flooding and China did not run, and
# the dataset keeps a folder for each anyway. Extra folders are harmless, so this
# is recorded rather than treated as drift.
UNRACED_FOLDERS: dict[int, frozenset[str]] = {
    2023: frozenset({"Imola", "Shanghai"}),
    2024: frozenset(),
    2025: frozenset(),
}


def test_the_recorded_listing_matches_the_calendar_shape() -> None:
    """Pins the recording to the calendars, so a rename fails here and says so.

    Only one direction is a requirement. Every ROUND has to resolve, which is
    what the tests above assert; a folder no round maps to costs nothing, and
    2023 has two of them because two of its races were cancelled. Naming them
    keeps the check useful: a folder appearing for any other reason is a rename
    the resolver has not been told about, and would otherwise surface months
    later as a silent empty fetch.
    """
    for year in YEARS:
        mapped = {race_folder(year, label) for label in get_gp_names(year).values()}
        unmatched = DATASET_FOLDERS[year] - mapped
        assert unmatched == UNRACED_FOLDERS[year], (
            f"{year}: unexpected dataset folders {sorted(unmatched - UNRACED_FOLDERS[year])}"
        )


@pytest.mark.parametrize("year", YEARS)
def test_the_menu_can_reach_the_last_round_of_the_season(year: int) -> None:
    """The stepper was clamped to 23, so 2025's finale was unreachable.

    2025 ran 24 rounds and Yas Island is round 24, a real published race the
    menu would not offer; 2023 ran 22 and the stepper walked into a round with
    no name.
    """
    calendar = get_gp_names(year)
    assert last_round(year) == max(calendar)
    assert calendar.get(last_round(year)), f"{year}: round {last_round(year)} has no name"


def test_the_seasons_are_not_all_the_same_length() -> None:
    """Which is why a constant cap could not be right for all three."""
    lengths = {year: last_round(year) for year in YEARS}
    assert lengths == {2023: 22, 2024: 24, 2025: 24}
