"""Session data loading and 25 Hz resampling for the Arcade replay.

Ported from Tom Shaw's f1-race-replay reference (see
`c:/tmp/arcade_analysis/04_fastf1_data_loading.md`) with three concrete fixes
over the reference: a race-distance accumulator that actually accumulates, a
`CACHE_VERSION` tag to invalidate stale pickles, and an `active` flag that
stops DNF'd drivers from sitting as ghosts at their crash position.

Output is a `SessionData` dataclass holding per-driver lists of `FrameData`
plus the geometry of a single reference lap that `track.py` consumes for the
circuit outline. All telemetry is kept in raw FastF1 units (1/10 mm for X/Y,
km/h for speed, seconds for time); conversion happens at render boundaries.
"""

from __future__ import annotations

import gc
import logging
import pickle
from dataclasses import dataclass, field
from multiprocessing import Pool
from pathlib import Path
from typing import Any

import fastf1
import fastf1.plotting
import numpy as np
import pandas as pd

from src.arcade.config import (
    ARCADE_CACHE_DIR,
    CACHE_VERSION,
    DT,
    FASTF1_CACHE_DIR,
    POOL_SIZE,
)
from src.f1_strat_manager.tyre_stint_repair import repair_tyre_stints

logger = logging.getLogger(__name__)

_COMPOUND_TO_INT: dict[str, int] = {
    "SOFT": 0,
    "MEDIUM": 1,
    "HARD": 2,
    "INTERMEDIATE": 3,
    "WET": 4,
    "UNKNOWN": 1,
    "TEST_UNKNOWN": 1,
}


@dataclass
class FrameData:
    """One 40 ms slice of one driver's state.

    `x`, `y` are in FastF1 raw units (1/10 mm). `dist` is race-cumulative
    metres (fixed from the reference bug that held it at zero). `active`
    becomes False past the driver's last real sample so the renderer can skip
    ghost cars after a DNF."""

    t: float
    x: float
    y: float
    speed: float
    gear: int
    drs: int
    throttle: float
    brake: float
    lap: int
    dist: float
    rel_dist: float
    tyre: int
    tyre_life: float
    active: bool = True


@dataclass
class SessionData:
    """Top-level cache payload consumed by `F1ArcadeWindow`.

    `ref_lap_xy` is the raw (non-rotated) fastest-lap polyline used by
    `track.py` for circuit geometry; rotation is applied at render time via
    `circuit_rotation_deg`. `timeline` is the common 25 Hz grid shared by
    every driver; its length is the total frame count of the replay."""

    version: str = CACHE_VERSION
    gp_name: str = ""
    # FastF1 ``session.event['Location']``, matches the per-race folder name
    # under ``data/raw/<year>/`` (``Suzuka``, ``Melbourne``, …). Normally
    # identical to ``gp_name``, because both resolve from the same canonical
    # calendar. They diverge on one path: when
    # ``data/tire_compounds_by_race.json`` is missing or lacks the year,
    # ``get_gp_names`` falls back to a hardcoded 2024 table and 2025 round 3
    # comes back "Australia" when it is Suzuka. ``gp_name`` also names the
    # session pickle (``_cache_path``), so on that path the cache is
    # mislabelled too, which is why this field exists and why every path
    # that touches disk must read it instead.
    location: str = ""
    year: int = 0
    frames_by_driver: dict[str, list[FrameData]] = field(default_factory=dict)
    driver_colors: dict[str, tuple[int, int, int]] = field(default_factory=dict)
    min_lap_number: int = 1
    max_lap_number: int = 0
    circuit_length_m: float = 5300.0
    circuit_rotation_deg: float = 0.0
    total_frames: int = 0
    timeline: np.ndarray = field(default_factory=lambda: np.zeros(0))
    # Session-time origin of the frame clock, in seconds. `timeline` starts at
    # 0.0 by construction (`_build_timeline`), so a frame's `t` is an offset,
    # not a session time, and on its own it is just `frame_index * DT`. Adding
    # this back recovers FastF1 `SessionTime` seconds, the clock that
    # `laps.parquet` (`Time`, `LapStartTime`, `Sector*SessionTime`) and
    # `weather.parquet` are keyed on. Without it nothing on the broadcast can
    # be joined by time to anything on disk, which is why `intervals.parquet`
    # is downloaded for every race and read by nothing.
    global_t_min: float = 0.0
    ref_lap_xy: tuple[np.ndarray, np.ndarray] = field(
        default_factory=lambda: (np.zeros(0), np.zeros(0))
    )
    ref_lap_drs: np.ndarray = field(default_factory=lambda: np.zeros(0))
    events: list[dict[str, Any]] = field(default_factory=list)
    # Per-lap FastF1 ``TrackStatus`` codes (multi-digit strings, e.g. ``"1"``,
    # ``"24"``, ``"567"``).  Consumed by the arcade race-events HUD card so
    # the user sees a coloured "SAFETY CAR" / "VSC" / "YELLOW FLAG" /
    # "RED FLAG" pill the moment the lap window enters a non-clear status.
    # Empty dict means the loader did not populate it (e.g. older cache);
    # the panel treats unknown laps as clear and stays hidden.
    track_status_by_lap: dict[int, str] = field(default_factory=dict)
    # Per-lap weather reading built from FastF1's ``session.weather_data``
    # (see #616): ``{lap_number: {"air_temp", "track_temp", "humidity",
    # "wind_speed", "wind_direction", "rain_state"}}``. Before this field
    # existed the weather was already fetched (``session.load(weather=True)``)
    # and then thrown away; the arcade UI showed a hardcoded 45 C / 18 C / DRY
    # constant on every lap of every race while the strategy pipeline used the
    # real values from a different path. Empty dict means the loader could not
    # extract weather (older cache, or the session genuinely has none), and the
    # panel then renders "N/A" per field rather than a display constant, so a
    # session with no weather reads as one with no weather (#1087).
    weather_by_lap: dict[int, dict[str, Any]] = field(default_factory=dict)
    # Whether each driver's telemetry actually places the car. False when the
    # distance never advances: on Melbourne 2025 that is HAD, whose position
    # channel FastF1 does not deliver at all. Every panel that plots position
    # renders empty for such a driver, and without this flag it renders empty
    # with a populated header and a "live" status bar, which reads as a broken
    # chart rather than as absent data. An empty dict means an older cache;
    # consumers treat an unlisted driver as having position.
    has_position: dict[str, bool] = field(default_factory=dict)
    # FastF1's official classification ``Status`` per driver abbreviation
    # ({"NOR": "Finished", "DOO": "Retired", ...}), read off
    # ``session.results`` at load time. The replay always replays a
    # COMPLETED session, so who took the chequered flag is a recorded fact;
    # deriving it from telemetry noise misclassified a wall-clock winner as
    # a retirement and a final-lap crasher as the winner (#879).
    # ``RaceGapCalculator`` treats this as the source of truth and keeps the
    # derived anchor only as the fallback for sessions that have none. Only
    # the status string is stored: position columns would be cached unread,
    # which is how ``intervals.parquet`` came to be downloaded every race
    # and read by nothing. Empty dict means an older cache or a results
    # table FastF1 could not deliver.
    official_status: dict[str, str] = field(default_factory=dict)


def _pedal_multiplier(results: list[dict], channel: str) -> float:
    """Decide once per session whether a pedal channel is 0-1 or 0-100.

    FastF1 delivers throttle and brake on either scale depending on the
    session, and the old code guessed **per frame**: `if value <= 1.0:
    value *= 100`. That cannot tell "0-1 scale, full throttle" from
    "0-100 scale, barely lifting", and it resolves the ambiguity the wrong
    way for a lifting car. Measured on Melbourne 2025, where the throttle
    channel is 0-100 (max 104): **72,104 frames, 2.34% of the race**, were
    genuine sub-1% openings published as 80-odd per cent.

    The session maximum has no such ambiguity: a 0-100 channel exceeds 1.0
    somewhere in a race and a 0-1 channel never does. One look at the whole
    array replaces three million guesses.
    """
    # `max()` keeps a NaN when the NaN comes FIRST, because every later
    # `x > nan` is False. One driver whose whole channel is NaN and who
    # happens to sort first would then flip the multiplier for the entire
    # session - every throttle above 1% published as 100.0, for all twenty
    # cars, depending on nothing but driver order. Filtering the peaks makes
    # an all-NaN channel contribute nothing instead of deciding the answer.
    peaks: list[float] = []
    for result in results:
        samples = result["data"][channel]
        finite = samples[np.isfinite(samples)] if len(samples) else samples
        if len(finite):
            peaks.append(float(finite.max()))
    peak = max(peaks, default=0.0)
    return 1.0 if peak > 1.0 else 100.0


def _lap_fraction_from_distance(
    dist: np.ndarray, lap_numbers: np.ndarray, circuit_length_m: float
) -> np.ndarray:
    """Fraction of the current lap, derived from the driver's own distance.

    NOT FastF1's `RelativeDistance` resampled. That column is normalised
    per lap and then interpolated across the concatenation of every lap, so
    at a boundary `np.interp` draws a straight line down through the whole
    [0, 1] range: measured on Melbourne 2025, **594 frames where it falls
    while `dist` rises**, up to half a lap in a single step, on 18 of the
    20 drivers. A consumer placing a car from it drew the car running
    backwards around the circuit for up to a second.

    Deriving it from `dist` cannot do that, because `dist` is monotone. It
    also normalises each lap by that lap's own length, so an in-lap and an
    out-lap map onto [0, 1] correctly instead of against a constant taken
    from the fastest lap. The last lap has no end yet, so it borrows the
    previous lap's length, and lap 1 borrows the circuit length.
    """
    fraction = np.zeros_like(dist)
    starts = np.flatnonzero(np.diff(lap_numbers) > 0) + 1
    bounds = [0, *starts.tolist(), len(dist)]
    previous_length = float(circuit_length_m) or 1.0
    for index in range(len(bounds) - 1):
        lo, hi = bounds[index], bounds[index + 1]
        if lo >= hi:
            continue
        start = float(dist[lo])
        length = float(dist[hi]) - start if hi < len(dist) else previous_length
        if length <= 0.0:
            length = previous_length
        fraction[lo:hi] = np.clip((dist[lo:hi] - start) / length, 0.0, 1.0)
        previous_length = length
    return fraction


def _repair_session_stints(session) -> None:
    """Apply the shared tyre-stint repair to this session's laps, in place.

    The replay used to read `TyreLife` straight off FastF1 while the two other
    consumers of the same race repaired it first: `laps_augment` on the way into
    the models, `session_data` on the way into PITWALL's timing tower. Both
    surfaces launch from one `f1-arcade --strategy` and sit on screen together,
    so a race with corrupted stint data showed two different ages for one tyre
    and nothing said which to believe (#951). Two members of a trio had the fix
    and the third did not, which is this repo's most frequent defect wearing a
    cross-surface costume: each surface was self-consistent, so neither could
    catch it alone.

    Written back column by column rather than by replacing `session.laps`,
    because the repair returns a plain DataFrame and the code below needs the
    FastF1 `Laps` subclass: `pick_drivers` in the per-driver extraction and
    `pick_fastest` for the reference lap are both methods on it. The repair
    reindexes its result to the input's index, so the assignment aligns row for
    row.

    Silent on a healthy race, which is the repair's own contract rather than a
    check here: a frame that needs nothing comes back unchanged with an empty
    report.
    """
    laps = getattr(session, "laps", None)
    if laps is None or laps.empty:
        return

    repaired, report = repair_tyre_stints(pd.DataFrame(laps))
    if not report.changed_anything:
        return

    # The three columns the repair writes. Named rather than copying the whole
    # frame, so this cannot quietly overwrite a column FastF1 owns; a column the
    # repair starts touching later would simply not arrive, and the guard for
    # that is the parity check against session_data, not this loop.
    for column in ("TyreLife", "Stint", "Compound"):
        if column in repaired.columns:
            laps[column] = repaired[column]

    logger.info(
        "Arcade tyre-stint repair: %d driver(s) touched, %d lap(s) now carry an "
        "unknown age, which is what PITWALL's tower shows for the same race",
        len(report.drivers_touched),
        report.unknown_after,
    )


def _enable_fastf1_cache() -> None:
    """Point FastF1 at the repo-local cache. Idempotent, safe across spawn."""
    FASTF1_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(str(FASTF1_CACHE_DIR))


def _compound_to_int(compound: Any) -> int:
    """Map a FastF1 compound string to the int code, defaulting to MEDIUM on unknowns."""
    if compound is None or (isinstance(compound, float) and np.isnan(compound)):
        return 1
    return _COMPOUND_TO_INT.get(str(compound).upper(), 1)


def _hex_to_rgb(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def _nearest_sample(t: np.ndarray, timeline: np.ndarray) -> np.ndarray:
    """For each timeline instant, the index of the closest raw sample (#1002).

    The resampler used to run `np.interp` over channels whose values are LABELS,
    which manufactures labels nobody measured. Melbourne 2025, 2,491,006 served
    frames: **1,775 DRS frames** carried 4, 5, 6, 7, 9, 11 or 13, codes the raw
    stream never contains and `DRS_OPEN_CODES` reads as closed, so an open wing
    drew as a flicker; and **86,925 brake frames (3.49%)** sat strictly between
    2 and 98 on a channel whose raw form is BOOLEAN. Both go to zero here.

    `fastf1.core.Telemetry._CHANNELS` marks `DRS`, `nGear` and `Brake` as
    `{'type': 'discrete'}` and fills them rather than interpolating. This is that
    distinction, which the arcade had inverted for `brake`.

    Ties go to the EARLIER sample, which matters because the raw stream carries
    907 duplicate-`t` pairs on this race: `<=` makes the pick deterministic
    instead of dependent on floating-point noise. Those 907 all have one cause,
    found in #1069: FastF1's per-lap windows share their boundary sample, so
    every crossing is concatenated twice at the same instant. `<=` settles which
    of the pair a timeline instant reads; `_concat_sorted_by_time` settles which
    of them comes first, and until #1069 that half was left to quicksort.

    Leans on `len(t) >= 2`, which `_process_driver_data` guarantees - it skips
    any lap with fewer than two samples and returns None when a driver has no
    laps at all, so the shortest array reaching here is 862 samples long.
    """
    right = np.searchsorted(t, timeline).clip(1, len(t) - 1)
    closer_to_left = timeline - t[right - 1] <= t[right] - timeline
    return np.where(closer_to_left, right - 1, right)


# The eight forward gears plus neutral, and the validity test is ONE-SIDED because
# 0 is a real reading rather than a sentinel: `session.car_data` for Melbourne 2025
# carries 202,509 of them across the 20 drivers, and the PITWALL GEAR lane's [0, 9]
# range renders it.
#
# **None of them reaches a replay, and that is a property of the DATA, not a filter
# here.** Every one of those 202,509 samples falls OUTSIDE every lap window: the
# cars are stationary in the garage and on the grid, before lap 1 starts at 00:56:06
# session time, while the laps run to 02:19:37. `_process_driver_data` reads
# `lap.get_telemetry()`, so it only ever sees samples inside a lap, and across all
# 1,059 laps of this race not one carries gear 0. Measured 2026-08-26 (#1094).
#
# So a replay of this race legitimately serves no neutral frame. Do NOT widen the
# predicate below to `< 1` on the strength of that: the reading is real, other
# sessions can put a stationary car inside a lap window, and one-sided is the rule.
def _concat_sorted_by_time(arrays: dict[str, list[np.ndarray]]) -> dict[str, np.ndarray]:
    """Flatten the per-lap arrays into one time-ordered block per channel.

    **The sort has to be STABLE, and that is why this is a function** (#1069).
    FastF1's per-lap telemetry windows SHARE their boundary sample rather than
    abutting, so the concatenation carries every lap crossing twice at the same
    `SessionTime`: 907 duplicate pairs on Melbourne 2025, 866 on Las Vegas, 1047
    on Qatar, all 2820 of them a lap boundary and none anywhere else, every gap
    zero to the bit. `np.argsort` defaults to quicksort, which is not stable, so
    the tie broke arbitrarily and put the OLD lap's copy second on 158 rows of
    Melbourne, 190 of Las Vegas and 256 of Qatar.

    What that corrupted is the three channels written as per-lap CONSTANTS in
    `_process_driver_data`: `lap`, `tyre` and `tyre_life`. A per-SAMPLE reading
    cannot be hurt by the swap, because both copies are the same instant and hold
    bitwise-identical values - `gear`, `drs`, `brake`, `throttle` and `speed`
    moved by zero on all 2820 pairs. A per-lap constant is discontinuous at
    exactly that boundary, so it moved on every flipped one. `tyre_life` was the
    worst, reading a 34-lap-old INTERMEDIATE one frame after a fresh MEDIUM went
    on.

    **The consequence was not one frame.** `_lap_fraction_from_distance` finds
    lap starts with `np.diff(lap) > 0`, which fires twice on `23, 24, 23, 24`,
    leaving a segment two to four frames long. Mid-race that publishes a stale
    frame at half a lap. On a driver's FINAL crossing the last segment borrows
    that length as `previous_length`, so the whole last lap is normalised by
    about 10 m instead of 5220 and `rel_dist` saturates at 1.0: on the shipped
    Melbourne replay HAM and LEC sat on the start/finish line for 92.3 s and
    90.7 s of active frames while their speed and gear kept moving. Roughly two
    drivers a race.

    Stability keeps the concatenation order for equal times, and the
    concatenation follows `iterlaps()`, which is ascending for every driver on
    all three of those races. `np.lexsort((concat["lap"], concat["t"]))` would
    decide the tie explicitly rather than inherit it; it is not used because the
    ordering already holds and one keyword is smaller. Dropping the duplicate row
    instead was measured and rejected: it advances the lap counter early on about
    1100 frames a race, where keeping both copies reads the old lap right up to
    the crossing, which is the correct reading.
    """
    concat = {k: np.concatenate(v) for k, v in arrays.items()}
    order = np.argsort(concat["t"], kind="stable")
    return {k: v[order] for k, v in concat.items()}


_MAX_GEAR = 8


def _drop_impossible_gears(gears: np.ndarray) -> np.ndarray:
    """Replace a gear no car has with the last one it really was in (#1002).

    The F1 live-timing feed publishes `nGear` values that do not exist. Read from
    `session.car_data` for Melbourne 2025, before any resampling this repo does:
    **151 samples at 128**, plus one or two each of almost every value between 10
    and 127. FastF1's own two resampling stages carry that to 310 and then 570,
    and the arcade's carried it to 967 frames at 128. PITWALL's GEAR lane is
    locked to [0, 9], so each one is a full-height spike.

    Nearest-neighbour resampling does NOT fix this: measured, it moves the count
    from 1,840 frames above 8 to 1,836. The sentinel is upstream of the
    interpolation, so it has to be rejected as data rather than smoothed.

    The repair is FastF1's own discrete-channel fill idiom (`.ffill()` then
    `.bfill()`) applied to a validity check FastF1 does not itself perform: it
    fills gaps in the merge, it does not judge whether a published gear is
    possible. The `bfill` tail is for a driver whose FIRST sample is invalid,
    where there is nothing behind to carry forward.

    A dropout longer than a blink therefore renders as a frozen gear rather than
    a spike. PIA's lap 44 is the worst case on this race and freezes at gear 8
    for about 75 s at 0 km/h. That is a visible artefact and the intended one:
    the alternative is a gear the car cannot select.
    """
    impossible = gears > _MAX_GEAR
    if not impossible.any():
        return gears
    if impossible.all():
        # Nothing to carry from in either direction, so the fill would return a
        # column of NaN and `int(...)` on the frame would raise. Unreachable on
        # real telemetry, where the sentinel is a handful of samples in tens of
        # thousands, and handled rather than left because the alternative failure
        # is the whole session load, not one frame.
        logger.warning("Every nGear sample is out of range; leaving the channel as published")
        return gears
    # The fill also closes any pre-existing NaN in the channel, because `ffill` does not
    # distinguish the ones this mask made from the ones it found. That is inconsistent
    # with the no-invalid-samples path above, which returns such an array untouched for
    # `int()` to raise on. Left as it is: no raw channel on any race measured carries a
    # NaN, and the raising half is what the interpolating resampler did too.
    repaired = pd.Series(gears).mask(impossible)
    return repaired.ffill().bfill().to_numpy()


def _process_driver_data(args: tuple) -> dict | None:
    """Module-level worker: iterate a driver's laps and flatten telemetry.

    Must stay at module scope so `multiprocessing.Pool` can pickle it by
    qualified name on Windows spawn. Mirrors the reference per-driver loop
    but actually increments the race-distance accumulator each lap."""

    driver_no, session, driver_code = args
    _enable_fastf1_cache()

    laps_driver = session.laps.pick_drivers(driver_no)
    if laps_driver.empty:
        return None

    arrays: dict[str, list[np.ndarray]] = {
        k: []
        for k in (
            "t",
            "x",
            "y",
            "speed",
            "gear",
            "drs",
            "throttle",
            "brake",
            "lap",
            "dist",
            "tyre",
            "tyre_life",
        )
    }
    total_dist_so_far = 0.0
    max_lap = 0

    for _, lap in laps_driver.iterlaps():
        try:
            tel = lap.get_telemetry()
        except (KeyError, ValueError, AttributeError):
            continue
        if tel is None or tel.empty:
            continue

        t = tel["SessionTime"].dt.total_seconds().to_numpy()
        n = len(t)
        if n < 2:
            continue

        x = tel["X"].to_numpy().astype(float)
        y = tel["Y"].to_numpy().astype(float)
        speed = tel["Speed"].to_numpy().astype(float) if "Speed" in tel.columns else np.zeros(n)
        gear = tel["nGear"].to_numpy().astype(float) if "nGear" in tel.columns else np.zeros(n)
        drs = tel["DRS"].to_numpy().astype(float) if "DRS" in tel.columns else np.zeros(n)
        thr = tel["Throttle"].to_numpy().astype(float) if "Throttle" in tel.columns else np.zeros(n)
        brk = tel["Brake"].to_numpy().astype(float) if "Brake" in tel.columns else np.zeros(n)

        d_lap = (
            tel["Distance"].to_numpy().astype(float) if "Distance" in tel.columns else np.zeros(n)
        )
        # FastF1's `RelativeDistance` is deliberately NOT collected. The
        # resampler stopped consuming it when the fraction became a
        # derivation over the driver's own distance; extracting it was
        # three lines of work per lap per driver feeding nothing.
        race_dist = total_dist_so_far + d_lap
        total_dist_so_far += float(d_lap[-1]) if n else 0.0

        lap_no = int(lap.LapNumber) if not pd.isna(lap.LapNumber) else 0
        max_lap = max(max_lap, lap_no)
        tyre = _compound_to_int(lap.Compound)
        tyre_life = 0.0 if pd.isna(lap.TyreLife) else float(lap.TyreLife)

        arrays["t"].append(t)
        arrays["x"].append(x)
        arrays["y"].append(y)
        arrays["speed"].append(speed)
        arrays["gear"].append(gear)
        arrays["drs"].append(drs)
        arrays["throttle"].append(thr)
        arrays["brake"].append(brk)
        arrays["lap"].append(np.full(n, lap_no, dtype=float))
        arrays["dist"].append(race_dist)
        arrays["tyre"].append(np.full(n, tyre, dtype=float))
        arrays["tyre_life"].append(np.full(n, tyre_life, dtype=float))

    if not arrays["t"]:
        return None

    concat = _concat_sorted_by_time(arrays)
    # After the sort, because the fill carries a gear FORWARD in time and the
    # per-lap arrays are concatenated in lap order but the samples inside them
    # are only sorted here.
    concat["gear"] = _drop_impossible_gears(concat["gear"])

    return {
        "code": driver_code,
        "data": concat,
        "t_min": float(concat["t"][0]),
        "t_max": float(concat["t"][-1]),
        "max_lap": int(max_lap),
    }


class SessionLoader:
    """Cache-first FastF1 loader. Warm path <5 s, cold path <3 min."""

    def __init__(self, cache_dir: Path = ARCADE_CACHE_DIR, pool_size: int = POOL_SIZE) -> None:
        self.cache_dir = cache_dir
        self.pool_size = pool_size

    def load(self, year: int, round_: int, gp_name: str) -> SessionData:
        """Fetch a race session, resample every driver to 25 Hz, and cache."""
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = self._cache_path(year, round_)

        if cache_path.exists():
            try:
                with cache_path.open("rb") as f:
                    # The generational collector walks the container it is in
                    # the middle of filling, and this one ends up holding ~2.5
                    # million FrameData objects, so it walks them repeatedly for
                    # nothing: none of them can be garbage while the load runs.
                    gc.disable()
                    try:
                        sd: SessionData = pickle.load(f)
                    finally:
                        gc.enable()
                if sd.version == CACHE_VERSION:
                    logger.info(
                        "Loaded session from cache: %s (%s %d)",
                        cache_path,
                        sd.location or sd.gp_name or "?",
                        sd.year,
                    )
                    return sd
                logger.info(
                    "Cache version mismatch (got %s, want %s) — refetching",
                    sd.version,
                    CACHE_VERSION,
                )
            except (pickle.PickleError, EOFError, AttributeError) as exc:
                logger.warning("Cache unreadable (%s) — refetching", exc)

        _enable_fastf1_cache()
        logger.info("Loading FastF1 session: %d round %d", year, round_)
        session = fastf1.get_session(year, round_, "R")
        session.load(telemetry=True, weather=True, laps=True)
        _repair_session_stints(session)

        # Read FastF1's authoritative Location (``Suzuka``, ``Melbourne``, …)
        # so the strategy pipeline can find the right per-race folder
        # regardless of how the arcade's hardcoded GP_NAMES table maps the
        # round number.
        try:
            location = str(session.event.get("Location", "") or "")
        except (AttributeError, KeyError):
            # session.event is a pandas Series (fastf1.events.Event); its
            # .get() already returns the default instead of raising for a
            # missing key. The only realistic failure is session.event
            # itself not being Series-like (AttributeError on .get) or a
            # custom __getitem__ override raising KeyError internally.
            location = ""

        driver_nums = list(session.drivers)
        driver_codes = {n: session.get_driver(n)["Abbreviation"] for n in driver_nums}
        driver_colors = self._resolve_driver_colors(session, driver_codes)

        results = self._process_all_drivers(session, driver_nums, driver_codes)
        results = [r for r in results if r is not None]
        if not results:
            raise RuntimeError(f"No driver telemetry could be extracted for {gp_name} {year}")

        timeline, global_t_min = self._build_timeline(results)

        max_lap = max(r["max_lap"] for r in results)
        ref_x, ref_y, ref_drs = self._extract_reference_lap(session, year, round_)
        rotation_deg = self._safe_rotation(session)
        circuit_length = self._session_circuit_length(session, ref_x, ref_y)
        # Resampling needs the circuit length: lap 1 has no previous lap to
        # normalise its distance fraction against.
        frames_by_driver = self._resample_all(results, timeline, global_t_min, circuit_length)
        track_status_by_lap = self._extract_track_status_by_lap(session)
        weather_by_lap = self._extract_weather_by_lap(session)
        official_status = self._extract_official_status(session, driver_codes)

        has_position = {
            code: bool(len(frames)) and frames[-1].dist > frames[0].dist
            for code, frames in frames_by_driver.items()
        }

        sd = SessionData(
            version=CACHE_VERSION,
            gp_name=gp_name,
            location=location,
            year=year,
            frames_by_driver=frames_by_driver,
            driver_colors=driver_colors,
            min_lap_number=1,
            max_lap_number=max_lap,
            circuit_length_m=circuit_length,
            circuit_rotation_deg=rotation_deg,
            total_frames=len(timeline),
            timeline=timeline,
            global_t_min=float(global_t_min),
            ref_lap_xy=(ref_x, ref_y),
            ref_lap_drs=ref_drs,
            events=[],
            track_status_by_lap=track_status_by_lap,
            weather_by_lap=weather_by_lap,
            has_position=has_position,
            official_status=official_status,
        )

        with cache_path.open("wb") as f:
            pickle.dump(sd, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info(
            "Cached session: %s (%d drivers, %d frames, %d laps)",
            cache_path,
            len(frames_by_driver),
            len(timeline),
            max_lap,
        )
        return sd

    def _cache_path(self, year: int, round_: int) -> Path:
        """Where this session is cached, keyed on what DECIDES its contents.

        It used to be keyed on `gp_name` while the session is fetched by
        `(year, round_)`, so the file name and the data inside it came from
        different things and the name could lie. It did: `data.py` line 86
        documents the fallback where `get_gp_names` misses a year and returns
        the 2024 table, and 2025 round 3 comes back "Australia" when it is
        Suzuka. The pickle was then written as Melbourne with Suzuka inside it,
        and the cache hit checked only the version, so every later load of that
        name returned the wrong race and said nothing (#1119).

        A year and a round cannot disagree with the session they fetch.
        """
        return self.cache_dir / f"{year}_r{round_:02d}_race.pkl"

    def _process_all_drivers(
        self, session: Any, driver_nums: list, driver_codes: dict
    ) -> list[dict | None]:
        # Serial by default: pickling a fully-loaded FastF1 session across N
        # Windows spawn workers is heavy and has hung in prior sessions. Set
        # `pool_size > 1` explicitly to opt into parallel extraction once the
        # FastF1 cache is warm.
        args = [(n, session, driver_codes[n]) for n in driver_nums]
        if self.pool_size <= 1:
            return self._process_serial(args)
        try:
            with Pool(processes=min(self.pool_size, len(args))) as pool:
                return pool.map(_process_driver_data, args)
        except Exception as exc:
            logger.warning("Pool failed (%s) - falling back to serial", exc)
            return self._process_serial(args)

    def _process_serial(self, args: list[tuple]) -> list[dict | None]:
        results: list[dict | None] = []
        for i, a in enumerate(args, 1):
            logger.info("  driver %d/%d: %s", i, len(args), a[2])
            results.append(_process_driver_data(a))
        return results

    def _build_timeline(self, results: list[dict]) -> tuple[np.ndarray, float]:
        global_t_min = min(r["t_min"] for r in results)
        global_t_max = max(r["t_max"] for r in results)
        timeline = np.arange(0.0, global_t_max - global_t_min, DT)
        return timeline, global_t_min

    def _resample_all(
        self,
        results: list[dict],
        timeline: np.ndarray,
        global_t_min: float,
        circuit_length_m: float,
    ) -> dict[str, list[FrameData]]:
        # Pedal scales are decided ONCE for the session, not per frame. See
        # `_pedal_multiplier`.
        multipliers = {name: _pedal_multiplier(results, name) for name in ("throttle", "brake")}
        out: dict[str, list[FrameData]] = {}
        for r in results:
            t = r["data"]["t"] - global_t_min
            t_max_local = r["t_max"] - global_t_min
            out[r["code"]] = self._resample_driver(
                r["data"], t, timeline, t_max_local, multipliers, circuit_length_m
            )
        return out

    def _resample_driver(
        self,
        data: dict[str, np.ndarray],
        t: np.ndarray,
        timeline: np.ndarray,
        t_max_local: float,
        pedal_multipliers: dict[str, float],
        circuit_length_m: float,
    ) -> list[FrameData]:
        cont = {k: np.interp(timeline, t, data[k]) for k in ("x", "y", "speed", "throttle", "dist")}
        nearest = _nearest_sample(t, timeline)
        disc = {k: data[k][nearest] for k in ("gear", "drs", "lap", "tyre", "brake", "tyre_life")}
        # The two pedals are scaled the same way and no longer live in the same dict, so
        # they are applied one by one rather than by iterating `pedal_multipliers`. The
        # shape being avoided is not the loop, which would raise `KeyError: 'brake'` and
        # be loud: it is brake ADDED to the discrete set and left in `cont` as well, where
        # the loop would keep scaling a copy nothing reads while the wire served the
        # unscaled one. Naming each pedal beside the dict it lives in makes that
        # impossible to write.
        cont["throttle"] = np.clip(cont["throttle"] * pedal_multipliers["throttle"], 0.0, 100.0)
        disc["brake"] = np.clip(disc["brake"] * pedal_multipliers["brake"], 0.0, 100.0)
        # Race distance cannot decrease; the per-lap accumulator leaves float
        # seams at lap boundaries (measured worst 0.11 m on Melbourne 2025).
        cont["dist"] = np.maximum.accumulate(cont["dist"])
        lap_numbers = np.maximum(1, disc["lap"].astype(int))
        rel_dist = _lap_fraction_from_distance(cont["dist"], lap_numbers, circuit_length_m)
        frames: list[FrameData] = []
        for i, ti in enumerate(timeline):
            active = ti <= t_max_local
            frames.append(
                FrameData(
                    t=float(ti),
                    x=float(cont["x"][i]),
                    y=float(cont["y"][i]),
                    speed=float(cont["speed"][i]),
                    gear=int(disc["gear"][i]),
                    drs=int(disc["drs"][i]),
                    throttle=float(cont["throttle"][i]),
                    brake=float(disc["brake"][i]),
                    lap=int(lap_numbers[i]),
                    dist=float(cont["dist"][i]),
                    rel_dist=float(rel_dist[i]),
                    tyre=int(disc["tyre"][i]),
                    tyre_life=float(disc["tyre_life"][i]),
                    active=active,
                )
            )
        return frames

    def _resolve_driver_colors(
        self, session: Any, driver_codes: dict
    ) -> dict[str, tuple[int, int, int]]:
        try:
            mapping = fastf1.plotting.get_driver_color_mapping(session)
        except Exception as exc:
            logger.warning("Driver color mapping failed (%s)", exc)
            return {code: (200, 200, 200) for code in driver_codes.values()}
        out: dict[str, tuple[int, int, int]] = {}
        for code in driver_codes.values():
            hex_color = mapping.get(code)
            out[code] = _hex_to_rgb(hex_color) if hex_color else (200, 200, 200)
        return out

    def _extract_reference_lap(
        self, session: Any, year: int, round_: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Use the fastest qualifying lap for geometry + DRS zones.

        Rationale (cf. f1_replay/main.py:43-68): in qualifying, drivers open
        their DRS wing in every activation zone because they are on a push
        lap, so a single quali telemetry has the full DRS picture. A race
        fastest lap only has DRS open where the driver had a car to catch,
        producing fragmented zones in practice. Falls back to race
        fastest if qualifying data cannot be loaded."""
        quali_result = self._try_quali_reference(year, round_)
        if quali_result is not None:
            logger.info("DRS: using fastest qualifying lap for track reference")
            return quali_result
        logger.info("DRS: qualifying unavailable, falling back to race fastest lap")
        try:
            ref_lap = session.laps.pick_fastest()
            tel = ref_lap.get_telemetry()
            x = tel["X"].to_numpy().astype(float)
            y = tel["Y"].to_numpy().astype(float)
            drs = tel["DRS"].to_numpy().astype(float) if "DRS" in tel.columns else np.zeros(len(x))
            return x, y, drs
        except (KeyError, ValueError, AttributeError) as exc:
            # Same enumeration as _process_driver_data's get_telemetry() catch
            # above: pick_fastest() returning None -> AttributeError,
            # get_car_data/get_pos_data doing DriverNumber/column lookups ->
            # KeyError, numeric casts on malformed telemetry -> ValueError.
            logger.warning("Reference lap extraction failed (%s) - using empty geometry", exc)
            return np.zeros(0), np.zeros(0), np.zeros(0)

    def _try_quali_reference(
        self, year: int, round_: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        try:
            quali = fastf1.get_session(year, round_, "Q")
            quali.load(telemetry=True, laps=True, weather=False)
            if quali.laps.empty:
                return None
            fastest = quali.laps.pick_fastest()
            if fastest is None:
                return None
            tel = fastest.get_telemetry()
            if tel is None or tel.empty or "DRS" not in tel.columns:
                return None
            x = tel["X"].to_numpy().astype(float)
            y = tel["Y"].to_numpy().astype(float)
            drs = tel["DRS"].to_numpy().astype(float)
            return x, y, drs
        except Exception as exc:
            logger.info("Quali load failed (%s)", exc)
            return None

    def _extract_track_status_by_lap(self, session: Any) -> dict[int, str]:
        """Build ``{lap_number: TrackStatus}`` from the loaded laps DataFrame.

        FastF1 stores the status as a multi-digit string (``"1"`` clear,
        ``"2"`` yellow, ``"4"`` Safety Car, ``"5"`` red flag, ``"6"`` /
        ``"7"`` VSC).  Because every driver in the same lap window sees
        the same status, only the first row per lap is kept.  Missing /
        malformed columns return an empty dict so the panel just stays
        hidden instead of raising at load time.
        """
        try:
            laps = session.laps
            if laps is None or "TrackStatus" not in getattr(laps, "columns", []):
                return {}
            subset = laps[["LapNumber", "TrackStatus"]].dropna()
            if subset.empty:
                return {}
            grouped = subset.groupby("LapNumber")["TrackStatus"].first()
            return {int(lap): str(code) for lap, code in grouped.items()}
        except (KeyError, ValueError, TypeError) as exc:
            # KeyError: "LapNumber" missing despite the "TrackStatus" presence
            # check above. ValueError/TypeError: a lap value int() cannot
            # convert. Pure pandas indexing beyond this is not expected to
            # raise anything else.
            logger.debug("Track-status-by-lap extraction failed (%s) - panel stays hidden", exc)
            return {}

    def _extract_weather_by_lap(self, session: Any) -> dict[int, dict[str, Any]]:
        """Build ``{lap_number: {air_temp, track_temp, humidity, wind_speed,
        wind_direction, rain_state}}`` from FastF1's ``session.weather_data``.

        ``session.load(weather=True)`` already pulls this DataFrame (one row
        roughly every 60 s, columns ``Time``, ``AirTemp``, ``TrackTemp``,
        ``Humidity``, ``WindSpeed``, ``WindDirection``, ``Rainfall``); before
        #616 the arcade UI paid that cost and then discarded the result,
        showing a fixed 45 C / 18 C / DRY on every lap instead. ``Time`` in
        both ``weather_data`` and ``session.laps`` is the same session-elapsed
        timedelta, so each lap's completion time can be matched against the
        closest weather sample with ``pd.merge_asof(..., direction="nearest")``:
        weather changes slowly enough over a race that the nearest reading
        (rather than an interpolation between two samples) is close enough for
        a replay. Missing/malformed data returns an empty dict so the panel
        falls back to its own built-in constants instead of raising at load
        time, exactly as it did before this field existed.
        """
        try:
            weather = session.weather_data
            laps = session.laps
            if weather is None or weather.empty or laps is None or laps.empty:
                return {}
            if "Time" not in weather.columns or "Time" not in laps.columns:
                return {}
            lap_times = laps[["LapNumber", "Time"]].dropna()
            if lap_times.empty:
                return {}
            # One row per lap number: the last driver to cross the line that
            # lap, close enough to "lap N is done" for a slowly-changing
            # weather reading. Sorted ascending because merge_asof requires
            # the "on" column to be sorted on both sides.
            per_lap = lap_times.groupby("LapNumber")["Time"].max().reset_index()
            per_lap = per_lap.sort_values("Time")
            wx_columns = [
                "Time",
                "AirTemp",
                "TrackTemp",
                "Humidity",
                "WindSpeed",
                "WindDirection",
                "Rainfall",
            ]
            wx = weather[wx_columns].sort_values("Time")
            merged = pd.merge_asof(per_lap, wx, on="Time", direction="nearest")

            out: dict[int, dict[str, Any]] = {}
            for _, row in merged.iterrows():
                out[int(row["LapNumber"])] = self._weather_row_to_dict(row)
            return out
        except Exception as exc:  # noqa: BLE001 - see the enumeration below
            # KeyError: an expected weather/laps column absent. ValueError and
            # TypeError: int()/float() casts on a malformed row, or a dtype
            # mismatch merge_asof rejects.
            #
            # And the one that made this catch broad on purpose: reading
            # `session.weather_data` raises fastf1's DataNotLoadedError when the
            # weather channel is unavailable, and that subclasses Exception
            # DIRECTLY, not any of the three above. It shows up only on execution:
            # session.load() returns normally and the property
            # raises afterwards, so a narrow tuple let it escape and killed the
            # entire arcade load, after the full cold path, for a session that
            # used to degrade silently to the panel's own constants.
            #
            # The docstring above promises exactly that degradation. Catching
            # narrowly here would have made the docstring a lie and turned a
            # missing optional channel into a crash, which is the failure this
            # repo has a written lesson about.
            logger.debug("Weather-by-lap extraction failed (%s) - panel keeps defaults", exc)
            return {}

    def _weather_row_to_dict(self, row: "pd.Series") -> dict[str, Any]:
        """Map one merged weather+lap row to the ``WeatherPanel``-facing shape.

        Key names match the panel's lookups exactly, so a row missing a single
        reading (rare, but possible on an incomplete weather sample) loses only
        that one field instead of dropping the whole lap.

        **A missing reading is an explicit ``None`` under the key, never an
        absent key**, which is why the panel cannot lean on a ``dict.get``
        default: the default fires on a missing KEY and this stores a missing
        VALUE. ``overlays._reading`` is the consumer that coalesces it, and
        before #1087 it did not exist, so a single NaN sample raised inside
        ``on_draw`` and took the render loop with it.

        **``Rainfall`` needs the same ``pd.notna`` guard as the five numbers,
        and for a worse reason.** It is the one field where a dropped sample
        did not raise: ``bool(float("nan"))`` is ``True``, so a NaN read as
        "WET" and the panel announced rain on a dry race. A crash is loud and a
        wrong affirmative is not, which is why this line survived the fix that
        was written to close exactly this class."""
        return {
            "air_temp": float(row["AirTemp"]) if pd.notna(row.get("AirTemp")) else None,
            "track_temp": float(row["TrackTemp"]) if pd.notna(row.get("TrackTemp")) else None,
            "humidity": float(row["Humidity"]) if pd.notna(row.get("Humidity")) else None,
            "wind_speed": float(row["WindSpeed"]) if pd.notna(row.get("WindSpeed")) else None,
            "wind_direction": (
                float(row["WindDirection"]) if pd.notna(row.get("WindDirection")) else None
            ),
            "rain_state": (
                ("WET" if bool(row["Rainfall"]) else "DRY")
                if pd.notna(row.get("Rainfall"))
                else None
            ),
        }

    def _extract_official_status(self, session: Any, driver_codes: dict) -> dict[str, str]:
        """Map each driver abbreviation to FastF1's official ``Status`` string.

        ``session.results`` is indexed by driver number - the same
        ``session.drivers`` enumeration ``load()`` built ``driver_codes``
        from, so the keys here match ``frames_by_driver`` by construction.
        That matters: a table keyed one way and queried another misses on
        every lookup while looking perfectly healthy (#448).

        Interpreting the vocabulary is ``gaps._took_flag_officially``'s job,
        not this one's - this stores the fact and one place decides what it
        means. A missing row or an empty status is left OUT rather than
        stored as a guess; the gap calculator then falls back to the derived
        rule for that driver and logs the disagreement.
        """
        try:
            results = session.results
            if results is None or results.empty:
                return {}
            out: dict[str, str] = {}
            for number, code in driver_codes.items():
                if number not in results.index:
                    continue
                status = results.loc[number, "Status"]
                if isinstance(status, str) and status:
                    out[code] = status
            return out
        except Exception as exc:  # noqa: BLE001 - the enumeration is the point
            # Reading ``session.results`` can raise fastf1's
            # DataNotLoadedError, which subclasses Exception DIRECTLY - the
            # same trap ``_extract_weather_by_lap`` documents: a narrow tuple
            # lets it escape and kills the whole arcade load for a session
            # that should have degraded to the derived flag rule.
            logger.warning(
                "Official classification unavailable (%s) - the flag logic "
                "falls back to the derived rule",
                exc,
            )
            return {}

    def _safe_rotation(self, session: Any) -> float:
        try:
            info = session.get_circuit_info()
            if info is None or not hasattr(info, "rotation"):
                return 0.0
            return float(info.rotation)
        except Exception as exc:
            # session.get_circuit_info() fetches from the MultiViewer API
            # (fastf1.mvapi.get_circuit_info), genuine network I/O with an
            # undocumented exception surface (HTTP/connection/JSON errors),
            # so this stays broad. Logged so a real regression still leaves
            # a trace instead of silently flattening every rotation to 0.
            logger.debug("Circuit rotation lookup failed (%s) - defaulting to 0.0", exc)
            return 0.0

    def _session_circuit_length(self, session, ref_x: np.ndarray, ref_y: np.ndarray) -> float:
        """Pick the most trustworthy circuit length available.

        Preferred path: the fastest lap's FastF1 ``add_distance()``
        telemetry: that column is cumulative metres within the lap, so
        its last value IS the track length (Suzuka ≈ 5807 m, Monaco ≈
        3337 m, Las Vegas ≈ 6201 m). Falls back to the reference-lap
        polyline estimator when the fastest-lap query fails (qualifying
        accidents, sessions without a clean flying lap). The ±range
        sanity check rejects absurd values so a single bad estimate
        cannot blow up the downstream X axes to 50 km."""
        try:
            fastest = session.laps.pick_fastest()
            tel = fastest.get_car_data().add_distance()
            length = float(tel["Distance"].iloc[-1])
            if 1500.0 < length < 12000.0:
                return length
        except (KeyError, ValueError, IndexError, AttributeError) as exc:
            # pick_fastest() may return None (AttributeError on
            # .get_car_data()); get_car_data/add_distance do DriverNumber
            # and channel lookups (KeyError), numeric casts (ValueError),
            # and .iloc[-1] on a possibly-empty Distance column (IndexError).
            logger.debug("Fastest-lap circuit length failed (%s) - using polyline estimate", exc)
        return self._estimate_circuit_length(ref_x, ref_y)

    def _estimate_circuit_length(self, ref_x: np.ndarray, ref_y: np.ndarray) -> float:
        if ref_x.size < 2:
            return 5300.0
        dx = np.diff(ref_x)
        dy = np.diff(ref_y)
        length_raw = float(np.sum(np.hypot(dx, dy)))
        # ref_x/ref_y are FastF1 raw units (1/10 mm, see the module docstring), so a
        # real circuit polyline sums into the tens of millions. The threshold exists
        # to leave alone a caller that already handed us metres. Naming both numbers
        # answers the question the ternary hid: what unit does this return, and can
        # it silently return raw units instead of metres?
        raw_units_per_metre = 10_000.0
        looks_like_raw_units = length_raw > 1e6
        if looks_like_raw_units:
            return length_raw / raw_units_per_metre
        return length_raw
