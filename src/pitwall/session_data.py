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

from src.arcade.track_status import neutralised_label
from src.f1_strat_manager.tyre_stint_repair import is_real_compound, repair_tyre_stints

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


def _compound_or_none(value: Any) -> str | None:
    """The compound, or None when the value is a stringified absence."""
    cleaned = _none_if_nan(value)
    if cleaned is None or not is_real_compound(cleaned):
        return None
    return cleaned


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
        # **Filtered through the same sentinel rule the stop count uses.** The
        # extractor stringifies a missing compound, so `_none_if_nan` - which only
        # catches a float NaN - passes `"nan"` or `"unknown"` straight through, and
        # the tower's `tyreCell` prints `compound[0]`: an `n` or a `u` in the TYRE
        # column, wearing the shape of a compound letter. No instance exists on the
        # one race a curated install carries, so there is no wrong pixel to show;
        # what there is, is `tyre_stint_repair` making that rule public and the
        # consumer NEXT to the one that got it not getting it. Doing it here means
        # no TypeScript consumer has to know the rule at all.
        "compound": _compound_or_none(record.get("Compound")),
        "tyre_life": _none_if_nan(record.get("TyreLife")),
        "stint": _none_if_nan(record.get("Stint")),
        "track_status": _none_if_nan(record.get("TrackStatus")),
        # The digits DECODED, by the arcade's own rule, for the one question the
        # grid needs answered: was the field racing freely on this lap. Decoded
        # here for the same reason `track_status_label` is decoded for the tick -
        # the priority order and the labels are a project rule, and a client that
        # tested for a `4` would be the second copy of it in another language.
        # Non-null means a per-lap pace ranking over this lap ranks the safety
        # car's queue rather than pace; see `neutralised_label`.
        "neutralised": neutralised_label(_none_if_nan(record.get("TrackStatus"))),
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


def _tyre_stops(revealed: list[dict[str, Any]]) -> int:
    """How many times this driver changed tyres, from the sets themselves.

    **A pit entry is not a stop, and the rule is already written in this repo.**
    `tyre_stint_repair`'s docstring states it under *"Why a pit entry alone does
    not mean a new set"*: a stop-and-go or a drive-through sends the car down
    the pit lane with no work done. Melbourne 2025 is a stronger case than the
    penalty it names - the safety car led the field through the pit lane on laps
    2, 3 AND 4, so `PitInTime` is set for all seventeen runners on all three,
    while `Compound` stays INTERMEDIATE and `TyreLife` counts 2 -> 3 -> 4 -> 5
    unbroken. Counting in-laps reported THREE stops for the whole field from lap
    5 to the flag, next to a `TYRE` cell reading `I 23`.

    `Stint` is not the answer either: FastF1 opens a new one on each of those
    passes, and at Miami 2025 the column is a 446-row NaN block.

    Therefore, the evidence is the SET: the compound changed, or the published
    age dropped. Both ride the bulk already, so this costs one pass and no new field.

    Generated rows are dropped BEFORE pairing rather than skipped inside the
    loop - a car that did not finish a lap has no compound and no age, and
    leaving it between two real rows would hide a change that spans it.

    --- WHAT THIS DELIBERATELY DOES NOT CATCH ---
    * A set refitted with the SAME compound whose published age is HIGHER than
      the outgoing set's. `tyre_stint_repair` measures used sets starting
      anywhere from 2 to 16, so a short first stint replaced by a used set is
      invisible here. It errs toward missing a stop, never toward inventing one.
    * A set refitted with the same compound while the feed REPUBLISHES the age as
      1. That reads as a stop here and it is one of the two things it could be.
      It is not a gap this rule can close, because the two are identical in every
      field it reads; `tyre_stint_repair`'s `_republished_age_mask` nulls those
      ages upstream instead, so the age arrives as None and neither reading is
      published. Melbourne 2025's five cases (ALB and STR on lap 3, LAW on lap 4,
      BEA and OCO on lap 5) are repaired there, which is what takes the field
      from **82 in-laps to 31**.
    """
    rows = [row for row in revealed if not row["generated"]]
    changed = 0
    for previous, current in zip(rows, rows[1:]):
        compounds = (previous["compound"], current["compound"])
        both_named = all(is_real_compound(value) for value in compounds)
        refitted = both_named and compounds[0] != compounds[1]
        ages = (previous["tyre_life"], current["tyre_life"])
        both_aged = all(value is not None for value in ages)
        reset = both_aged and ages[1] < ages[0]
        if refitted or reset:
            changed += 1
    return changed


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
        iff `L <= laps_completed[d]`. Not the main driver's lap: at 96% of
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
        """Each sector's most recent value, and whether it belongs to this lap.

        A timing screen's sector cells do not go blank at the line - they
        **roll**. Each one shows the freshest number it has: this lap's if the
        car has crossed that sector, otherwise the lap before's, with the
        difference shown by dimming rather than by hiding.

        **The first version of this served only the lap in progress, and it
        made the S3 column permanently empty.** S3's crossing IS the end of
        the lap: measured over the 920 real rows of Melbourne 2025 carrying both stamps,
        `Sector3SessionTime` lands a median 55 ms AFTER the lap's own crossing
        `Time` and after it on 94.1% of laps. So S1 was visible for 60.3 s of
        its lap, S2 for 40.8 s, and S3 for -0.055 s. One of three columns
        showed nothing for the entire race.

        **The reveal rule still holds at the finer coordinate.** A sector is
        served only once the clock has passed its own crossing - this lap's or
        the previous one's - so nothing is visible before it happened, and a
        rewind takes it back because the clock goes back with it.

        It is a separate reader rather than part of `masked_view` because the
        two are masked by different things at different rates. A sector opens
        somewhere in the field every 2.22 s (measured: 2,744 crossings over
        6,103 s), and re-sending the whole revealed race at that cadence is
        ~154 KB/s against the tick's own ~58. This block is 2 KB for twenty
        drivers.
        """
        session_clock = clock_s + global_t_min
        drivers: dict[str, Any] = {}
        for code, rows in self._by_driver.items():
            lap = laps_completed.get(code, 0) + 1
            in_progress = self._row_for_lap(rows, lap)
            if in_progress is None:
                continue
            previous = self._row_for_lap(rows, lap - 1)
            drivers[code] = self._rolling_sectors(in_progress, previous, session_clock)
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
    def _rolling_sectors(
        row: dict[str, Any], previous: dict[str, Any] | None, session_clock: float
    ) -> dict[str, Any]:
        """Each sector's freshest crossed value, flagged with whose lap it is.

        `<sector>_fresh` is True when the value belongs to the lap in
        progress and False when it is carried over from the lap before, which
        is what the renderer dims. It is not a third state for a missing
        value: a null sector is simply null and its flag is False.

        **Both branches are gated on the clock, and the second one had to be
        told so.** It first checked only that the previous lap HAD a stamp,
        which made the sentence below false on the wire the window serves: the
        arcade's crossing map increments before that lap's own
        `Sector3SessionTime` on 837 of 921 laps (median 39 ms, max 0.463 s),
        so the just-ended S3 went out before its own official moment. It leaked
        nothing the bulk was not already revealing at the same tick, but a
        guard that looks like the reveal rule and checks something else is
        this repo's most expensive shape.

        So: nothing is served before its own crossing, in either lap. The cost
        is a cell that dashes for a median 39 ms after a line crossing, about
        one replay frame.

        A sector whose value exists but whose stamp does not - one driver-lap
        on Melbourne 2025 - dashes rather than being carried, because there is
        no moment to compare the clock against and the rule above is the one
        that matters.
        """
        live: dict[str, Any] = {"lap": row["lap"]}
        for sector, speed, crossed_at in _LIVE_SECTORS:
            moment = row[crossed_at]
            if moment is not None and moment <= session_clock:
                live[sector], live[speed], fresh = row[sector], row[speed], True
            elif (
                previous is not None
                and previous[crossed_at] is not None
                and previous[crossed_at] <= session_clock
            ):
                live[sector], live[speed], fresh = previous[sector], previous[speed], False
            else:
                live[sector], live[speed], fresh = None, None, False
            live[f"{sector}_fresh"] = fresh and live[sector] is not None
        return live

    @staticmethod
    def _driver_view(
        revealed: list[dict[str, Any]], revealed_to: int, global_t_min: float
    ) -> dict[str, Any]:
        """One driver's revealed block: rows on the wire clock, stops, bests.

        `stops` counts TYRE-SET TRANSITIONS; see `_tyre_stops`. It used to
        count in-laps, and the docstring that chose that read *"the two agree
        on every driver of a healthy race, which is exactly how the wrong one
        gets chosen"* - a true clause under a false headline, because on the
        one race a curated install carries NEITHER counts stops.

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
            "stops": _tyre_stops(revealed),
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
