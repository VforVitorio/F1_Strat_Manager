"""A cached session is named after the race it holds (#1119).

`SessionLoader.load` fetches by `(year, round_)` and used to name its pickle
after `gp_name`, so the file name and the data inside it came from different
inputs. `data.py:86` documents the path where those two disagree: when
`data/tire_compounds_by_race.json` is missing or lacks the year, `get_gp_names`
falls back to a hardcoded 2024 table, and 2025 round 3 comes back as an
Australian label when the race is Suzuka. The pickle was then written under the
wrong name, and the cache hit checked only `version`, so every later load of
that name returned the wrong race and logged nothing about it.

It happened on this machine: `Melbourne_2025_race.pkl` held Suzuka, 53 laps, and
every measurement taken through it during the design pass was recorded against
the wrong race name.

A year and a round cannot disagree with the session they fetch, so that is what
the file is named after now.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.arcade.data import SessionLoader


@pytest.fixture
def loader(tmp_path: Path) -> SessionLoader:
    return SessionLoader(cache_dir=tmp_path)


def test_the_name_is_built_from_what_decides_the_contents(loader: SessionLoader) -> None:
    """Year and round, the two arguments FastF1 is actually given."""
    path = loader._cache_path(2025, 3)
    assert path.name == "2025_r03_race.pkl"


@pytest.mark.parametrize(("year", "round_"), [(2025, 1), (2025, 3), (2024, 24), (2023, 22)])
def test_a_different_race_is_a_different_file(
    loader: SessionLoader, year: int, round_: int
) -> None:
    """No two rounds may share a path, or one would serve the other."""
    others = [
        loader._cache_path(y, r)
        for y in (2023, 2024, 2025)
        for r in (1, 3, 22, 24)
        if (y, r) != (year, round_)
    ]
    assert loader._cache_path(year, round_) not in others


def test_the_round_is_zero_padded(loader: SessionLoader) -> None:
    """So a directory listing sorts by round rather than lexically.

    Cosmetic on its own, and the reason it is asserted is that changing it later
    would silently orphan every cache a user already has.
    """
    assert loader._cache_path(2025, 3).name < loader._cache_path(2025, 22).name


def test_the_name_carries_no_label_at_all(loader: SessionLoader) -> None:
    """A name holding a label could disagree with the data again.

    This is what makes the fix structural rather than a validation that has to
    be remembered: there is no field in the name for a wrong answer to sit in.
    """
    name = loader._cache_path(2025, 3).name
    for label in ("Melbourne", "Suzuka", "Australia", "Japan"):
        assert label.lower() not in name.lower()
