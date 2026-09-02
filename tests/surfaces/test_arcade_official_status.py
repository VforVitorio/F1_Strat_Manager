"""SessionLoader._extract_official_status: the facts cached for #879's flag logic.

No arcade/pyglet dependency: data.py imports fastf1 and pandas only, so this
file needs no importorskip("arcade").
"""

from __future__ import annotations

from types import SimpleNamespace

import pandas as pd

from src.arcade.data import SessionLoader


def _extract(session, driver_codes):
    return SessionLoader()._extract_official_status(session, driver_codes)


def test_statuses_are_keyed_by_abbreviation_in_the_frames_keyspace():
    """Results are indexed by driver NUMBER; frames by ABBREVIATION (#448)."""
    df = pd.DataFrame({"Status": ["Finished", "Retired"]}, index=["4", "7"])
    out = _extract(SimpleNamespace(results=df), {"4": "NOR", "7": "DOO"})
    assert out == {"NOR": "Finished", "DOO": "Retired"}


def test_a_missing_or_empty_status_is_absent_not_guessed():
    """Unknown is absent: gaps.py falls back per driver, never a stored guess."""
    df = pd.DataFrame({"Status": ["Finished", float("nan"), ""]}, index=["4", "1", "63"])
    out = _extract(SimpleNamespace(results=df), {"4": "NOR", "1": "VER", "63": "RUS", "99": "GHO"})
    assert out == {"NOR": "Finished"}


def test_an_unreadable_results_table_degrades_to_empty(caplog):
    """DataNotLoadedError subclasses Exception directly (the weather scar)."""
    import fastf1.core

    class _Boom:
        @property
        def results(self):
            raise fastf1.core.DataNotLoadedError("not loaded")

    out = _extract(_Boom(), {"4": "NOR"})
    assert out == {}
    assert any("falls back to the derived rule" in r.message for r in caplog.records)


def test_a_none_or_empty_results_table_is_empty():
    assert _extract(SimpleNamespace(results=None), {"4": "NOR"}) == {}
    assert _extract(SimpleNamespace(results=pd.DataFrame()), {"4": "NOR"}) == {}
