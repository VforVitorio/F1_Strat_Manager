"""The BULK channel: a race's lap table, read from disk, revealed lap by lap.

PITWALL has two data channels (design decision C7). The TICK carries the
instant at ~10 Hz. This is the other one: everything the timing table and the
bests panel show is already on disk before lap 1, because a replay's laps are
static parquet. So the panel is a **progressive reveal masked by the clock**,
not a stream to accumulate - which also makes it immune to the sample loss at
8x, since it is per lap rather than per frame.

Three properties this module exists to guarantee, each one measured before a
line was written (`~/.claude/plans/pitwall-sprint5/bulk-reader-design.md`):

1. **The mask is applied HERE, not in the renderer.** The reveal is the DATA
   window's load-bearing invariant, and host-side it lives in one testable
   function instead of being re-implemented by every band. The alternative
   put the whole race result in the renderer's memory, where one filter bug
   leaks the finishing order onto a live screen.
2. **`FastF1Generated` rows are excluded from every statistic.** They are the
   6 rows FastF1 synthesises for cars that did not finish a lap, and their
   `Time` stamps sort BEFORE the real field: a naive ranking puts the lap-1
   crashers P1-P2-P3, and a naive lap count shows a crashed car in the top 3
   for 172 seconds of replay. They are still returned as rows, so the table
   can render the car; they simply never count.
3. **Every missing value is `None`, never a number.** The bridge serialises
   with `allow_nan=False`, so a NaN is a 500 and a blank window - and a
   numeric default is worse than the crash: this repo has a scar where a NaN
   `Position` became `0` and the leader then "found" the car that had just
   crashed. Unknown data is `None` and every consumer handles it.

--- WHERE TO CHANGE IF THE PARQUET CHANGES ---
`_LapRow` names every column it reads. The columns come from FastF1's
`session.laps`, so a FastF1 upgrade that renames one lands here first.
`src/f1_strat_manager/laps_augment.py` reads the same file for the model
pipeline and is the other place to check.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any

import pandas as pd

from src.f1_strat_manager.tyre_stint_repair import repair_tyre_stints

logger = logging.getLogger(__name__)

# Columns read straight through, parquet name -> payload key. Speeds are the
# timing screen's four traps: I1/I2 are the sector-1 and sector-2 traps, FL is
# the finish line and ST is the longest straight.
_SPEED_COLUMNS = {"SpeedI1": "v1", "SpeedI2": "v2", "SpeedFL": "vfl", "SpeedST": "vst"}
_SECTOR_COLUMNS = {"Sector1Time": "s1", "Sector2Time": "s2", "Sector3Time": "s3"}

# Fields the bests panel ranks. `s1` is in here even though lap 1 never has
# one: the min simply ignores the Nones, which is why they must be None and
# not zero - a zero would win every ranking.
_BEST_FIELDS = ("lap_time", "s1", "s2", "s3")
_BEST_SPEED_FIELDS = ("v1", "v2", "vfl", "vst")


def _none_if_nan(value: Any) -> Any:
    """NaN out, `None` in - the only sentinel this module is allowed to emit."""
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def _seconds(value: Any) -> float | None:
    """A pandas timedelta as float seconds, or None when the cell is empty."""
    if value is None or pd.isna(value):
        return None
    return round(float(pd.Timedelta(value).total_seconds()), 3)


def race_dir(data_root: Path, year: int, location: str) -> Path | None:
    """The race folder for a wire `location`, or None when it is not on disk.

    **Resolve on `location`, never on `gp_name`.** The producer publishes both
    and they diverge: FastF1's 2025 Location is "Miami Gardens" and the folder
    is `Miami_Gardens`, while the canonical calendar's key - which is what
    names `gp_name` and the session pickle - is "Miami". A `gp_name` resolver
    misses that race on the happy path, with no fallback involved.

    Exact match, then the spaced-to-underscore variant. Checked against the
    complete folder inventory of all three published seasons: the two rules
    resolve all 70 races, including `Marina_Bay`, `Mexico_City`, `São_Paulo`,
    `Yas_Island` and `Las_Vegas`. No alias table is needed on this path.

    Returns None rather than raising, and never downloads. A missing race is
    the COMMON case - a curated install holds one of the seventy - so it is
    absent data, not a failed operation.
    """
    raw = data_root / "raw" / str(year)
    for name in (location, location.replace(" ", "_")):
        candidate = raw / name
        if (candidate / "laps.parquet").is_file():
            return candidate
    return None


def _lap_row(record: dict[str, Any]) -> dict[str, Any]:
    """One parquet row in the shape the tower renders.

    `time_s` is the raw FastF1 SessionTime of the line crossing. It stays raw
    here and becomes the wire's clock (`t`) only in `masked_view`, where
    `global_t_min` is known - the parquet has no idea what the replay chose
    as its origin.
    """
    row: dict[str, Any] = {
        "lap": int(record["LapNumber"]),
        "time_s": _seconds(record.get("Time")),
        "lap_time": _seconds(record.get("LapTime")),
        "position": _none_if_nan(record.get("Position")),
        "compound": _none_if_nan(record.get("Compound")),
        "tyre_life": _none_if_nan(record.get("TyreLife")),
        "stint": _none_if_nan(record.get("Stint")),
        "track_status": _none_if_nan(record.get("TrackStatus")),
        "pit_in": pd.notna(record.get("PitInTime")),
        "pit_out": pd.notna(record.get("PitOutTime")),
        "deleted": bool(record.get("Deleted", False)),
        # The poison flag. Rendered, never counted - see the module docstring.
        "generated": bool(record.get("FastF1Generated", False)),
        # The column holds {True, False, None}; only a literal True is a flag.
        # Letting the None cross the bridge would make it a third state that
        # every consumer has to know about.
        "pb": record.get("IsPersonalBest") is True,
    }
    row.update({key: _seconds(record.get(column)) for column, key in _SECTOR_COLUMNS.items()})
    row.update({key: _none_if_nan(record.get(column)) for column, key in _SPEED_COLUMNS.items()})
    if row["position"] is not None:
        row["position"] = int(row["position"])
    if row["stint"] is not None:
        row["stint"] = int(row["stint"])
    return row


def _best_of(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """The per-field minima the bests panel ranks, over countable rows only.

    Recomputed rather than read off `IsPersonalBest`, per the delivery plan.
    The two definitions agree on the final lap for every driver but differ on
    47 lap-flags mid-race, and mid-race is the only state a masked panel ever
    renders. Deleted laps are excluded because a deleted time does not count,
    and generated rows because they carry no times at all.
    """
    countable = [row for row in rows if not row["deleted"] and not row["generated"]]
    best: dict[str, Any] = {"lap": None}
    for field in (*_BEST_FIELDS, *_BEST_SPEED_FIELDS):
        values = [row[field] for row in countable if row[field] is not None]
        best[field] = min(values) if values else None
    fastest = [row for row in countable if row["lap_time"] is not None]
    if fastest:
        quickest = min(fastest, key=lambda row: row["lap_time"])
        best["lap"] = quickest["lap"]
        best["compound"] = quickest["compound"]
    else:
        best["compound"] = None
    return best


def _theoretical(best: dict[str, Any]) -> float | None:
    """Best S1 + S2 + S3, or None while any sector is still unknown."""
    sectors = [best["s1"], best["s2"], best["s3"]]
    if any(value is None for value in sectors):
        return None
    return round(sum(sectors), 3)


class SessionLaps:
    """One race's lap table, loaded once, sliced by the reveal on every read.

    Immutable after construction. The mask is a pure function of the tick's
    per-driver `laps_completed`, so the same clock always produces the same
    view and a rewind produces a strictly smaller one.
    """

    def __init__(
        self,
        year: int,
        location: str,
        by_driver: dict[str, list[dict[str, Any]]],
        numbers: dict[str, str],
    ):
        self._year = year
        self._location = location
        self._by_driver = by_driver
        self._numbers = numbers
        self._total_laps = max(
            (row["lap"] for rows in by_driver.values() for row in rows), default=0
        )

    @classmethod
    def load(cls, data_root: Path, year: int, location: str) -> SessionLaps | None:
        """Read `data_root/raw/{year}/{location}/laps.parquet`, or None if absent.

        The frame goes through `repair_tyre_stints` first. The live-timing
        feed sometimes drops a stint's records and then restarts the stint at
        the recovery lap, so at Miami 2025 every car reads `TyreLife 1` on a
        set that has done 24 racing laps - and this panel shows compound and
        stops. The repair is a no-op on a healthy race by construction, so it
        costs an import; it was previously called by exactly one consumer,
        which is how the arcade's own compound pills came to render the
        unrepaired values.
        """
        directory = race_dir(data_root, year, location)
        if directory is None:
            logger.info(
                "No lap data for %s %s. Fetch it with: uv run python -c "
                '"from src.f1_strat_manager.data_cache import ensure_race; '
                "ensure_race(%s, '%s')\"",
                year,
                location,
                year,
                location,
            )
            return None

        frame = pd.read_parquet(directory / "laps.parquet")
        frame, _report = repair_tyre_stints(frame)

        by_driver: dict[str, list[dict[str, Any]]] = {}
        numbers: dict[str, str] = {}
        for record in frame.to_dict("records"):
            code = record.get("Driver")
            if not isinstance(code, str) or pd.isna(record.get("LapNumber")):
                continue
            by_driver.setdefault(code, []).append(_lap_row(record))
            # Kept as the string the parquet holds. It is an identifier, not a
            # quantity: "07" is a real car number and int() would print it "7".
            number = record.get("DriverNumber")
            if isinstance(number, str):
                numbers[code] = number
        for rows in by_driver.values():
            rows.sort(key=lambda row: row["lap"])
        return cls(year, location, by_driver, numbers)

    @property
    def total_laps(self) -> int:
        return self._total_laps

    def masked_view(
        self, laps_completed: dict[str, int], global_t_min: float = 0.0
    ) -> dict[str, Any]:
        """The race as of the clock: every driver's laps up to what they finished.

        The reveal is **per driver and strict** - driver *d*'s lap *L* shows
        iff `L <= laps_completed[d]`. Not the main driver's lap: at 96 % of
        instants the running field spans two or three different laps, so one
        shared cut lags the leaders by a lap and leaks look-ahead for the cars
        behind, simultaneously.

        A driver the tick does not mention reveals nothing. That is the
        honest answer for a car with no position data (#886) and it is also
        what makes a rewind un-reveal: `laps_completed` falls, the view
        shrinks, and because this is recomputed rather than accumulated there
        is no cache to leak the future out of.
        """
        drivers: dict[str, Any] = {}
        for code, rows in self._by_driver.items():
            revealed_to = laps_completed.get(code, 0)
            revealed = [row for row in rows if row["lap"] <= revealed_to]
            view = self._driver_view(revealed, revealed_to, global_t_min)
            view["number"] = self._numbers.get(code)
            drivers[code] = view
        result = {
            "available": True,
            "race": {
                "year": self._year,
                "location": self._location,
                "total_laps": self._total_laps,
            },
            "drivers": drivers,
        }
        return result

    @staticmethod
    def _driver_view(
        revealed: list[dict[str, Any]], revealed_to: int, global_t_min: float
    ) -> dict[str, Any]:
        """One driver's revealed block: rows on the wire clock, stops, bests.

        `stops` counts in-laps rather than `max(stint) - 1`. The two agree on
        every driver of a healthy race, which is exactly how the wrong one
        gets chosen: Miami 2025's raw frame carries a 446-row NaN `Stint`
        block, and a stint-based count reads zero stops for most of the field
        late in the race.

        `crossings` is the lap-quantised clock the gap column subtracts. It
        holds real rows only - a generated row's `Time` stamp sorts before
        the entire field and would invert the interval it takes part in.
        """
        laps = [{**row, "t": SessionLaps._on_wire_clock(row, global_t_min)} for row in revealed]
        crossings = {
            row["lap"]: row["t"] for row in laps if not row["generated"] and row["t"] is not None
        }
        best = _best_of(revealed)
        view = {
            "laps_revealed": revealed_to,
            "stops": sum(1 for row in revealed if row["pit_in"]),
            "laps": laps,
            "crossings": crossings,
            "best": best,
            "theoretical": _theoretical(best),
        }
        return view

    @staticmethod
    def _on_wire_clock(row: dict[str, Any], global_t_min: float) -> float | None:
        """`Time` rebased onto the tick's origin, so both channels share a clock."""
        if row["time_s"] is None:
            return None
        return round(row["time_s"] - global_t_min, 3)


def unavailable(year: int | None = None, location: str | None = None) -> dict[str, Any]:
    """The payload for a race with no lap data, so the panels can say so.

    An explicit state rather than an empty one: "no lap data for this race"
    is information, and a tower rendering zero rows silently is the same
    pixel as a tower whose reveal is broken.
    """
    result = {
        "available": False,
        "race": {"year": year, "location": location, "total_laps": 0},
        "drivers": {},
    }
    return result
