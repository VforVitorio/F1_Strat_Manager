"""Load-boundary validation for race data (F-02, #244).

Turns silent failure modes at the parquet-read boundary into loud, sourced
errors before they reach the agents:

- A **missing required column** used to surface as a cryptic ``KeyError`` deep
  inside ``RaceStateManager`` (or, worse, as a silent ``None`` in ``lap_state``
  that flowed into an agent's feature vector). Now it fails at ingestion,
  naming the artifact and the exact missing column.
- The **FastF1 quality flags** (``IsAccurate`` / ``Deleted``) already ship in
  every laps parquet but were never consulted. They are legitimate data, not
  corruption, so they are surfaced as a *warning*, not a hard stop.

Fail-loud vs warn split (per the P5 audit): structural breaks that make the
data unusable raise :class:`DataValidationError`; quality-flag findings only
warn.
"""

from __future__ import annotations

import sys

import pandas as pd

# The columns the simulation pipeline needs to produce a correct lap_state.
# Driver / LapNumber / Time / TrackStatus are indexed UNCONDITIONALLY by
# RaceStateManager (the per-driver split + total laps, the session-time gaps in
# _compute_session_times, and the weather/track-status snapshot), so a missing
# one is a hard crash deep in construction. LapTime / Position / Compound /
# TyreLife are read more defensively but a missing one would flow into lap_state
# as a SILENT None on an agent's feature vector - exactly the failure mode #244
# exists to catch - so they are required too.
REQUIRED_LAPS_COLUMNS: tuple[str, ...] = (
    "Driver",
    "LapNumber",
    "LapTime",
    "Time",
    "Position",
    "Compound",
    "TyreLife",
    "TrackStatus",
)


class DataValidationError(ValueError):
    """A loaded race artifact is structurally unusable.

    Raised for a missing required column or an empty laps table - conditions
    under which the simulation cannot produce a correct ``lap_state`` and must
    stop loudly rather than emit silent ``None`` fields.
    """


def validate_laps_df(df: pd.DataFrame, source: str) -> None:
    """Assert a laps DataFrame satisfies the RaceStateManager contract.

    Args:
        df:     The laps DataFrame just read from parquet.
        source: A human-readable identifier for the error message (the parquet
                path when known, else a ``"<gp> <year>"`` label).

    Raises:
        DataValidationError: if any :data:`REQUIRED_LAPS_COLUMNS` is absent or
            the table has zero rows. The message names the source and the exact
            missing columns so the fix is unambiguous.
    """
    missing = [c for c in REQUIRED_LAPS_COLUMNS if c not in df.columns]
    if missing:
        raise DataValidationError(
            f"{source}: laps parquet is missing required column(s) {missing}. "
            f"Present columns: {sorted(map(str, df.columns))}"
        )
    if len(df) == 0:
        raise DataValidationError(f"{source}: laps parquet has zero rows.")
    if not df["LapNumber"].notna().any():
        raise DataValidationError(
            f"{source}: laps parquet has no non-null LapNumber "
            "(RaceStateManager cannot index laps or compute total_laps)."
        )


def warn_low_quality_laps(df: pd.DataFrame, source: str) -> None:
    """Warn (never raise) when FastF1 flags some laps as inaccurate/deleted.

    ``IsAccurate=False`` and ``Deleted=True`` are legitimate FastF1 annotations
    (out-laps, track-limit deletions), not corruption - so this surfaces them
    for the operator's awareness without blocking the run. Absent flag columns
    make it a no-op.
    """
    notes: list[str] = []
    try:
        if "IsAccurate" in df.columns:
            inaccurate = int((~df["IsAccurate"].astype("boolean").fillna(True)).sum())
            if inaccurate:
                notes.append(f"{inaccurate} lap(s) flagged IsAccurate=False")
        if "Deleted" in df.columns:
            deleted = int(df["Deleted"].astype("boolean").fillna(False).sum())
            if deleted:
                notes.append(f"{deleted} lap(s) flagged Deleted")
    except (TypeError, ValueError):
        # A quality-flag column with an unexpected dtype (e.g. "True"/"False"
        # strings, mixed object) must never abort a load - this helper only ever
        # advises, so swallow the coercion error rather than raise. (#244 D4)
        return
    if notes:
        print(
            f"[warn] {source}: {'; '.join(notes)} (FastF1 quality flags).",
            file=sys.stderr,
        )
