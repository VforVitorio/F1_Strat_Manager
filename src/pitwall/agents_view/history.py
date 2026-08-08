"""The two rolling per-lap stores the AGENTS charts read from.

Ported from `src/arcade/dashboard/window.py:239-293`, near-verbatim and
on purpose: the Qt window is the acceptance reference for this port, and
an accumulator rewritten from its description is exactly how the two
surfaces start disagreeing about the same race.

Why the window owns these at all: `history_tail` on the broadcast strips
`per_agent` from every past decision, a deliberate wire-size trade-off, so
predicted lap times and cliff percentiles exist only on the `latest` block
of the tick they arrived on. Nobody can rebuild them later, which is also
why the rewind guard below **evicts the future and never truncates the
past**.

The store is keyed by LAP, not by frame. Gate A's D-11: a frame-indexed
truncate cannot address a lap-keyed map, and re-ingesting the same tick
must be idempotent, which a list would not be.
"""

from __future__ import annotations

from typing import Any

# The charts draw a rolling window, and an hour of replay would otherwise
# grow a 200-entry dict per store for laps nothing renders.
KEEP_LAPS: int = 40


class LapHistory:
    """Per-lap pace and tyre rows, accumulated from successive ticks.

    Responsibilities:

    - backfill actuals from `history_tail` on a mid-stream connect;
    - fold each tick's `latest` in, which is the only carrier of the
      per-agent predictions;
    - drop laps the user rewound past, so a lap that gets re-driven is
      re-observed rather than read from the timeline that was abandoned;
    - stay bounded.
    """

    def __init__(self, keep: int = KEEP_LAPS) -> None:
        self._keep = keep
        self._pace: dict[int, dict[str, Any]] = {}
        self._tire: dict[int, dict[str, Any]] = {}

    # --- Ingest -----------------------------------------------------------

    def seed_from_tail(self, tail: list[dict[str, Any]]) -> None:
        """Backfill lap-time and tyre actuals from the broadcast history tail.

        `setdefault`, not assignment: the tail is stripped of `per_agent`,
        so a value already observed on a `latest` block is richer than
        anything here and must not be overwritten by a later reconnect.
        """
        for row in tail:
            lap = row.get("lap_number")
            if not isinstance(lap, int):
                continue
            pace_row = self._pace.setdefault(lap, {})
            pace_row.setdefault("actual", row.get("lap_time_s"))
            tire_row = self._tire.setdefault(lap, {})
            tire_row.setdefault("tyre_life", row.get("tyre_life"))
            tire_row.setdefault("compound", row.get("compound"))
            tire_row.setdefault("lap_time_s", row.get("lap_time_s"))
        self.trim()

    def ingest_latest(self, latest: dict[str, Any] | None) -> None:
        """Fold this tick's decision in. The only source of predictions."""
        if not latest:
            return
        lap = latest.get("lap_number")
        if not isinstance(lap, int):
            return
        per = latest.get("per_agent") or {}
        pace = per.get("pace") or {}
        row = self._pace.setdefault(lap, {})
        if latest.get("lap_time_s") is not None:
            row["actual"] = latest.get("lap_time_s")
        if pace.get("lap_time_pred") is not None:
            row["pred"] = pace.get("lap_time_pred")
            row["ci_p10"] = pace.get("ci_p10")
            row["ci_p90"] = pace.get("ci_p90")
        trow = self._tire.setdefault(lap, {})
        if latest.get("tyre_life") is not None:
            trow["tyre_life"] = latest.get("tyre_life")
        if latest.get("compound"):
            trow["compound"] = latest.get("compound")
        if latest.get("lap_time_s") is not None:
            trow["lap_time_s"] = latest.get("lap_time_s")
        self.trim()

    def evict_after(self, lap: int) -> None:
        """Drop everything ahead of `lap`, for a backwards seek.

        The Qt window has no equivalent and its charts are wrong after a
        rewind: they hold laps the replay has not reached again. The guard
        belongs here rather than in the UI because the store is keyed by
        lap and the UI clock counts frames - a frame-indexed truncate
        cannot address this map at all.
        """
        for store in (self._pace, self._tire):
            for stored in [key for key in store if key > lap]:
                store.pop(stored, None)

    def trim(self, keep: int | None = None) -> None:
        """Keep only the most recent `keep` laps so memory stays bounded."""
        limit = self._keep if keep is None else keep
        for store in (self._pace, self._tire):
            if len(store) <= limit:
                continue
            for lap in sorted(store)[: len(store) - limit]:
                store.pop(lap, None)

    # --- Read -------------------------------------------------------------

    @property
    def pace(self) -> dict[int, dict[str, Any]]:
        """Lap -> {actual, pred, ci_p10, ci_p90}, as `PaceChart` reads it."""
        return self._pace

    def tire_rows(self) -> list[dict[str, Any]]:
        """Chronological rows, the shape `TireChart.update_from` takes."""
        return [{"lap": lap, **row} for lap, row in sorted(self._tire.items())]
