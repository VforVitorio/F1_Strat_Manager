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
    # FastF1 ``session.event['Location']`` — matches the per-race folder name
    # under ``data/raw/<year>/`` (``Suzuka``, ``Melbourne``, …). Normally
    # identical to ``gp_name``, because both resolve from the same canonical
    # calendar. They diverge on one path: when
    # ``data/tire_compounds_by_race.json`` is missing or lacks the year,
    # ``get_gp_names`` falls back to a hardcoded 2024 table and 2025 round 3
    # comes back "Australia" when it is Suzuka. ``gp_name`` also names the
    # session pickle (``_cache_path``), so on that path the cache is
    # mislabelled too — which is why this field exists and why every path
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
    # this back recovers FastF1 `SessionTime` seconds — the clock that
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
    # extract weather (older cache, or the session genuinely has none); the
    # panel's own ``.get(key, default)`` calls keep the old constants as the
    # last-resort display instead of raising.
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
    channel is 0-100 (max 104): **72,104 frames, 2.34 % of the race**, were
    genuine sub-1 % openings published as 80-odd per cent.

    The session maximum has no such ambiguity: a 0-100 channel exceeds 1.0
    somewhere in a race and a 0-1 channel never does. One look at the whole
    array replaces three million guesses.
    """
    # `max()` keeps a NaN when the NaN comes FIRST, because every later
    # `x > nan` is False. One driver whose whole channel is NaN and who
    # happens to sort first would then flip the multiplier for the entire
    # session - every throttle above 1 % published as 100.0, for all twenty
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


def _enable_fastf1_cache() -> None:
    """Point FastF1 at our repo-local cache. Idempotent, safe across spawn."""
    FASTF1_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(str(FASTF1_CACHE_DIR))


def _compound_to_int(compound: Any) -> int:
    """Map a FastF1 compound string to our int code, defaulting to MEDIUM on unknowns."""
    if compound is None or (isinstance(compound, float) and np.isnan(compound)):
        return 1
    return _COMPOUND_TO_INT.get(str(compound).upper(), 1)


def _hex_to_rgb(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


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

    concat = {k: np.concatenate(v) for k, v in arrays.items()}
    order = np.argsort(concat["t"])
    for k in concat:
        concat[k] = concat[k][order]

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
        cache_path = self._cache_path(gp_name, year)

        if cache_path.exists():
            try:
                with cache_path.open("rb") as f:
                    sd: SessionData = pickle.load(f)
                if sd.version == CACHE_VERSION:
                    logger.info("Loaded session from cache: %s", cache_path)
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

    def _cache_path(self, gp_name: str, year: int) -> Path:
        safe = gp_name.replace(" ", "_")
        return self.cache_dir / f"{safe}_{year}_race.pkl"

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
        cont = {
            k: np.interp(timeline, t, data[k])
            for k in ("x", "y", "speed", "throttle", "brake", "dist", "tyre_life")
        }
        disc = {k: np.interp(timeline, t, data[k]) for k in ("gear", "drs", "lap", "tyre")}
        for name, multiplier in pedal_multipliers.items():
            cont[name] = np.clip(cont[name] * multiplier, 0.0, 100.0)
        # Race distance cannot decrease; the per-lap accumulator leaves float
        # seams at lap boundaries (measured worst 0.11 m on Melbourne 2025).
        cont["dist"] = np.maximum.accumulate(cont["dist"])
        lap_numbers = np.maximum(1, np.rint(disc["lap"]).astype(int))
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
                    gear=int(round(disc["gear"][i])),
                    drs=int(round(disc["drs"][i])),
                    throttle=float(cont["throttle"][i]),
                    brake=float(cont["brake"][i]),
                    lap=int(lap_numbers[i]),
                    dist=float(cont["dist"][i]),
                    rel_dist=float(rel_dist[i]),
                    tyre=int(round(disc["tyre"][i])),
                    tyre_life=float(cont["tyre_life"][i]),
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
        producing the fragmented zones we saw earlier. Falls back to race
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
        the same status we keep the first row per lap.  Missing /
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
            # DIRECTLY, not any of the three above. An adversarial gate caught it
            # by executing it: session.load() returns normally and the property
            # raises afterwards, so a narrow tuple let it escape and killed the
            # entire arcade load, after the full cold path, for a session that
            # used to degrade quietly to the panel's own constants.
            #
            # The docstring above promises exactly that degradation. Catching
            # narrowly here would have made the docstring a lie and turned a
            # missing optional channel into a crash, which is the failure this
            # repo has a written lesson about.
            logger.debug("Weather-by-lap extraction failed (%s) - panel keeps defaults", exc)
            return {}

    def _weather_row_to_dict(self, row: "pd.Series") -> dict[str, Any]:
        """Map one merged weather+lap row to the ``WeatherPanel``-facing shape.

        Key names match ``WeatherPanel.draw``'s ``weather.get(key, default)``
        calls exactly, so a row missing a single reading (rare, but possible
        on an incomplete weather sample) only loses that one field to the
        panel's own default instead of dropping the whole lap."""
        return {
            "air_temp": float(row["AirTemp"]) if pd.notna(row.get("AirTemp")) else None,
            "track_temp": float(row["TrackTemp"]) if pd.notna(row.get("TrackTemp")) else None,
            "humidity": float(row["Humidity"]) if pd.notna(row.get("Humidity")) else None,
            "wind_speed": float(row["WindSpeed"]) if pd.notna(row.get("WindSpeed")) else None,
            "wind_direction": (
                float(row["WindDirection"]) if pd.notna(row.get("WindDirection")) else None
            ),
            "rain_state": "WET" if bool(row.get("Rainfall")) else "DRY",
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
            # (fastf1.mvapi.get_circuit_info) — genuine network I/O with an
            # undocumented exception surface (HTTP/connection/JSON errors),
            # so this stays broad. Logged so a real regression still leaves
            # a trace instead of silently flattening every rotation to 0.
            logger.debug("Circuit rotation lookup failed (%s) - defaulting to 0.0", exc)
            return 0.0

    def _session_circuit_length(self, session, ref_x: np.ndarray, ref_y: np.ndarray) -> float:
        """Pick the most trustworthy circuit length we can derive.

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
