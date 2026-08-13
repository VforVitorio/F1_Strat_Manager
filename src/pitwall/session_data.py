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
# When each sector was CROSSED, on FastF1's SessionTime. This is what lets a
# sector be revealed at the instant it happened rather than a whole lap later,
# which is the difference between a timing tower and a table of the previous
# lap. Same clock as the tick's `t + global_t_min`.
_SECTOR_AT_COLUMNS = {
    "Sector1SessionTime": "s1_at",
    "Sector2SessionTime": "s2_at",
    "Sector3SessionTime": "s3_at",
}

# The sector, its time, its trap speed, and the moment it opened. The speed is
# measured AT the trap, so it becomes known with its sector and not before.
_LIVE_SECTORS = (("s1", "v1", "s1_at"), ("s2", "v2", "s2_at"), ("s3", "vfl", "s3_at"))
_SECTOR_AT_KEYS = frozenset(_SECTOR_AT_COLUMNS.values())

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
    row.update({key: _seconds(record.get(column)) for column, key in _SECTOR_AT_COLUMNS.items()})
    row.update({key: _none_if_nan(record.get(column)) for column, key in _SPEED_COLUMNS.items()})
    if row["position"] is not None:
        row["position"] = int(row["position"])
    if row["stint"] is not None:
        row["stint"] = int(row["stint"])
    return row


def _extremes(
    rows: list[dict[str, Any]], fields: tuple[str, ...], pick: Any
) -> dict[str, float | None]:
    """`pick` over each field's known values, or None when a field has none.

    None because the value is genuinely unknown, never a number: a zero would
    win every speed ranking outright and lose every time ranking, which is the
    sentinel collision this repo has a scar from.
    """
    chosen: dict[str, float | None] = {}
    for field in fields:
        values = [row[field] for row in rows if row[field] is not None]
        chosen[field] = pick(values) if values else None
    return chosen


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
    # A best TIME is the smallest and a best SPEED is the largest, and the two
    # tuples exist to say so. They were once walked by one loop that minimised
    # both, which served NOR's best speed-trap as 180 km/h against a real
    # maximum of 289 - his slowest crawl through the trap, presented as his
    # best of the session, on all four speed columns.
    best.update(_extremes(countable, _BEST_FIELDS, min))
    best.update(_extremes(countable, _BEST_SPEED_FIELDS, max))
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

    def live_lap(
        self, laps_completed: dict[str, int], clock_s: float, global_t_min: float = 0.0
    ) -> dict[str, Any]:
        """The lap each driver is ON, with only the sectors he has already crossed.

        The tower's three sector columns show the lap IN PROGRESS, blank at
        the line and filling as the car crosses each sector - which is what a
        timing tower does and what showing the last COMPLETED lap for a whole
        lap afterwards does not.

        **This does not weaken the reveal rule; it applies it at a finer
        coordinate.** `masked_view`'s `L <= laps_completed` is the rule for
        lap ROWS, which only exist once the lap is over. A sector has its own
        timestamp, so its own moment: reveal it iff the replay clock has
        passed `SectorNSessionTime`. Nothing here is visible before it
        happened, and a rewind closes the sectors again because the clock
        goes back with it.

        It is a separate reader rather than part of `masked_view` because the
        two are masked by different things at different rates. A sector opens
        somewhere in the field every 2.22 s (measured: 2,744 crossings over
        6,103 s of Melbourne 2025), and re-sending the whole revealed race at
        that cadence is ~154 KB/s against the tick's own ~58. This block is
        2 KB for twenty drivers.
        """
        session_clock = clock_s + global_t_min
        drivers: dict[str, Any] = {}
        for code, rows in self._by_driver.items():
            in_progress = self._row_for_lap(rows, laps_completed.get(code, 0) + 1)
            if in_progress is None:
                continue
            drivers[code] = self._revealed_sectors(in_progress, session_clock)
        return drivers

    @staticmethod
    def _row_for_lap(rows: list[dict[str, Any]], lap: int) -> dict[str, Any] | None:
        """The driver's row for one lap, or None when he has no such lap.

        None covers three real cases and they all mean the same thing to the
        caller: a car that retired has no further row, a finisher has no lap
        past the flag, and a car whose only rows are `FastF1Generated` has
        nothing with times in it.
        """
        for row in rows:
            if row["lap"] == lap:
                return None if row["generated"] else row
        return None

    @staticmethod
    def _revealed_sectors(row: dict[str, Any], session_clock: float) -> dict[str, Any]:
        """One in-progress lap, with the sectors the clock has not reached left out.

        `None` for a sector that has not happened yet, exactly as for one that
        has no data - the renderer draws a dash either way, and inventing a
        distinction the tower cannot use would only be a third state for every
        consumer to carry.
        """
        live: dict[str, Any] = {"lap": row["lap"]}
        for sector, speed, crossed_at in _LIVE_SECTORS:
            moment = row[crossed_at]
            open_now = moment is not None and moment <= session_clock
            live[sector] = row[sector] if open_now else None
            live[speed] = row[speed] if open_now else None
        return live

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
        # The sector crossing instants stay OFF this payload. They exist for
        # `live_lap`, which reveals the lap in progress, and putting three
        # more floats on each of 927 rows would grow the whole-race worst case
        # for a field the tower never reads here.
        laps = [
            {
                **{key: value for key, value in row.items() if key not in _SECTOR_AT_KEYS},
                "t": SessionLaps._on_wire_clock(row, global_t_min),
            }
            for row in revealed
        ]
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
