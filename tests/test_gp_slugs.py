"""Hermetic unit tests for the GP name/slug resolver (#243).

Before #243, ~6 GPs/season silently ran with **no radio and no compound
labels**: the resolver keyed on the friendly space-separated name
(``"Las Vegas"``) while callers passed the raw on-disk folder name with
underscores (``"Las_Vegas"``) or a renamed folder (``"Miami_Gardens"``).
These pin that every on-disk folder form now maps to the right slug, and
that the reentrant / raise-on-typo contract is preserved.
"""

from __future__ import annotations

import pytest

from src.f1_strat_manager.gp_slugs import (
    COUNTRY_SLUG_BY_GP,
    canonical_gp_name,
    resolve_gp_slug,
)

# The 6 GPs whose raw 2025 folder name did NOT match the friendly key before
# #243: (folder_name, expected_friendly_name, expected_slug).
_PREVIOUSLY_FAILING = [
    ("Las_Vegas", "Las Vegas", "united_states_las_vegas"),
    ("Marina_Bay", "Marina Bay", "singapore"),
    ("Mexico_City", "Mexico City", "mexico"),
    ("Miami_Gardens", "Miami", "united_states_miami"),
    ("São_Paulo", "São Paulo", "brazil"),
    ("Yas_Island", "Yas Island", "united_arab_emirates"),
]


@pytest.mark.parametrize("folder, friendly, slug", _PREVIOUSLY_FAILING)
def test_folder_names_resolve(folder, friendly, slug):
    """Underscore folder names (and the Miami_Gardens rename) now resolve."""
    assert canonical_gp_name(folder) == friendly
    assert resolve_gp_slug(folder) == slug


def test_friendly_names_still_resolve():
    """Every friendly key still maps to its slug — no regression."""
    for friendly, slug in COUNTRY_SLUG_BY_GP.items():
        assert resolve_gp_slug(friendly) == slug


def test_resolve_is_reentrant_on_slugs():
    """Passing an already-resolved slug is a no-op, not an error."""
    for slug in set(COUNTRY_SLUG_BY_GP.values()):
        assert resolve_gp_slug(slug) == slug


def test_unknown_gp_raises():
    """A genuine typo still raises so it does not silently drop radios."""
    with pytest.raises(ValueError):
        resolve_gp_slug("Nordschleife")


def test_canonical_returns_unknown_unchanged():
    """canonical_gp_name hands unknown input back verbatim (caller decides)."""
    assert canonical_gp_name("Nordschleife") == "Nordschleife"
    assert canonical_gp_name("Totally_Unknown") == "Totally_Unknown"
