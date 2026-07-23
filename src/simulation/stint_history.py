"""Art. 30.5(m) stint-history facts for the strategy decision layer.

Pure, frame-agnostic helpers that read a GP-scoped laps frame and answer three
questions about one driver at one lap: how many stops they have made, which
compounds they have used, and whether the mandatory two-dry-compound obligation
(F1 Sporting Regulations Art. 30.5(m), 2024/25 numbering; 30.5(n) in 2023) is
still pending. Using intermediate or wet tyres at any point exempts the driver.

Invariant these helpers protect: the featured parquet drops laps (SC / pit /
out-laps fail N04's ``IsAccurate`` gate) and can therefore hide WHOLE stints
(e.g. Lusail 2024 VER shows stints 1 and 4 only). So the helpers distinguish
"seen" from "certain": ``stops_made`` comes from the highest stint NUMBER (a
sequential FastF1 index), never from the count of visible stints, and when an
invisible stint could hide the second compound, ``mandatory_stop_pending`` is
``None`` (unknown) rather than a guess. None means unknown, never a default.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

DRY_COMPOUNDS = frozenset({"SOFT", "MEDIUM", "HARD"})
WET_COMPOUNDS = frozenset({"INTERMEDIATE", "WET"})

_REQUIRED_COLUMNS = frozenset({"Driver", "LapNumber", "Compound"})


def _unknown_flags() -> dict[str, Any]:
    """The honest empty answer: nothing observed, nothing asserted."""
    return {"stops_made": None, "compounds_used": [], "mandatory_stop_pending": None}


def _visible_compounds(rows: pd.DataFrame) -> list[str]:
    """Ordered unique compound names seen in ``rows``, unknown strings dropped.

    First-seen order is preserved so the list reads as the driver's compound
    history. Strings outside the five FIA compound names (empty cells, test
    placeholders) are excluded: they are neither dry nor wet evidence.
    """
    seen = rows["Compound"].dropna().astype(str).str.upper()
    known = seen[seen.isin(DRY_COMPOUNDS | WET_COMPOUNDS)]
    ordered_unique = list(dict.fromkeys(known))
    return ordered_unique


def _visible_stint_numbers(rows: pd.DataFrame) -> list[int]:
    """Sorted distinct stint numbers present in ``rows`` (empty if no column)."""
    if "Stint" not in rows.columns:
        return []
    numbers = sorted({int(s) for s in rows["Stint"].dropna()})
    return numbers


def stint_history_flags(
    gp_laps: pd.DataFrame | None,
    driver: str,
    lap_number: int,
) -> dict[str, Any]:
    """Stops made, compounds used and the Art. 30.5(m) flag up to ``lap_number``.

    Args:
        gp_laps:    GP-scoped laps frame (scoping is the caller's duty, #429
                    lesson: an unscoped season frame would blend races). Needs
                    Driver / LapNumber / Compound; Stint is optional.
        driver:     FIA three-letter code.
        lap_number: Inclusive upper bound — only laps at or before it count.

    Returns:
        ``stops_made``: highest visible stint number minus one (FastF1 numbers
        stints sequentially from 1, so this counts stops even when the featured
        frame hides intermediate stints), or None without Stint data.
        ``compounds_used``: first-seen-ordered unique compounds observed.
        ``mandatory_stop_pending``: False on positive evidence (a wet-weather
        compound used, or two different dry compounds seen); True only when
        EVERY stint so far is visible and still shows a single dry compound;
        None when the history is empty or an invisible stint could hide the
        second compound.
    """
    if gp_laps is None or gp_laps.empty or not _REQUIRED_COLUMNS <= set(gp_laps.columns):
        return _unknown_flags()

    in_window = (gp_laps["Driver"] == driver) & (gp_laps["LapNumber"] <= lap_number)
    rows = gp_laps[in_window]
    if rows.empty:
        return _unknown_flags()

    stints = _visible_stint_numbers(rows)
    stops_made = stints[-1] - 1 if stints else None

    compounds = _visible_compounds(rows)
    used_wet = any(c in WET_COMPOUNDS for c in compounds)
    dry_seen = {c for c in compounds if c in DRY_COMPOUNDS}
    all_stints_visible = bool(stints) and set(range(1, stints[-1] + 1)) <= set(stints)

    if not compounds:
        pending: bool | None = None
    elif used_wet or len(dry_seen) >= 2:
        pending = False
    elif all_stints_visible:
        pending = True
    else:
        pending = None

    flags = {
        "stops_made": stops_made,
        "compounds_used": compounds,
        "mandatory_stop_pending": pending,
    }
    return flags
