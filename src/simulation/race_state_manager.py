"""
Race state manager — enforces the single-driver data boundary.

Our driver  → full telemetry: LapTime, Sector1/2/3, TyreLife, Compound,
              CompoundID, Stint, Position, SpeedI1/I2/FL/ST, FuelLoad
              (estimated), gap_to_leader, is_in_lap.

Rivals      → timing-screen only: Position, LapTime, Compound, TyreLife,
              SpeedST, gap_to_leader_s, interval_to_driver_s, is_pitting.

This mirrors what a real team strategy wall sees during a race and is the
critical architectural constraint — agents must never be given data that
would not be available from live timing in a real scenario.

Gap computation: uses the FastF1 ``Time`` column (session elapsed time at
end of each lap), identical to how N27 computes on-track gaps in the notebook
via ``(session.laps.Time_X - session.laps.Time_Y).total_seconds()``. This is
more accurate than cumulative LapTime sums under safety car bunching.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.simulation.data_validation import validate_laps_df

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _to_seconds(td: Any) -> float | None:
    """Convert a timedelta-like value (or NaT/NaN) to float seconds."""
    if td is None or (hasattr(pd, "isna") and pd.isna(td)):
        return None
    if hasattr(td, "total_seconds"):
        return round(td.total_seconds(), 3)
    try:
        return round(float(td), 3)
    except (TypeError, ValueError):
        return None


def _compute_session_times(laps_df: pd.DataFrame) -> pd.DataFrame:
    """Add ``session_time_s`` = session elapsed time in seconds per driver per lap.

    Uses the ``Time`` column (timedelta from session start to end of lap),
    which is the same value FastF1 uses internally for gap computation in
    ``session.laps``. This is more accurate than cumulative LapTime sums
    because it accounts for timing corrections and safety car bunching where
    cumulative sums diverge from actual on-track gaps.

    Also adds ``lap_time_s`` (float seconds for each individual lap) which is
    used by agents for pace analysis.

    Returns a copy of ``laps_df`` with both additional columns.
    """
    df = laps_df.copy()
    df["lap_time_s"] = df["LapTime"].apply(_to_seconds)
    df["session_time_s"] = df["Time"].apply(_to_seconds)
    return df


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


class RaceStateManager:
    """Enforces the single-driver perspective architectural constraint.

    Pre-processes the full laps DataFrame into per-driver lookup tables and
    provides per-lap snapshots that respect the data boundary: rich telemetry
    for our driver, timing-screen-only data for every rival.

    The returned ``lap_state`` dict (from ``get_lap_state``) is the canonical
    format consumed by all seven agents (pace, tire, race_situation, radio,
    pit_strategy, rag, and the strategy orchestrator).

    Attributes:
        driver_code: Three-letter FIA driver code (e.g. ``"NOR"``).
        team:        Full team name as stored in the laps parquet.
        gp_name:     Grand Prix name for circuit-specific thresholds.
        year:        Season year.
        total_laps:  Total number of completed laps in the race.
    """

    def __init__(
        self,
        laps_df: pd.DataFrame,
        driver_code: str,
        team: str,
        gp_name: str = "",
        year: int = 2025,
    ) -> None:
        """Pre-process laps into per-driver, per-lap lookup tables.

        Cumulative race times are computed once at construction so that
        ``get_lap_state`` calls are O(1) lookups rather than repeated scans.
        For a 57-lap race with 20 drivers the preprocessing takes < 50 ms.

        Args:
            laps_df:     Raw laps parquet loaded into a DataFrame. Must have
                         at minimum: Driver, LapNumber, LapTime, Position,
                         Compound, TyreLife columns.
            driver_code: Our driver's FIA three-letter code (e.g. ``"NOR"``).
            team:        Our driver's team name as it appears in the parquet.
            gp_name:     Grand Prix name used by circuit-aware agent thresholds.
            year:        Season year (used for session_meta only).
        """
        self.driver_code = driver_code
        self.team = team
        self.gp_name = gp_name
        self.year = year

        # Validate the schema contract here (not only in RaceReplayEngine) so a
        # missing required column fails loudly at construction from ANY caller -
        # the backend simulator and tests build RaceStateManager directly. (F-02)
        validate_laps_df(laps_df, source=f"{gp_name or 'race'} {year} laps parquet")

        enriched = _compute_session_times(laps_df)

        self._all: pd.DataFrame = enriched
        self._driver: pd.DataFrame = enriched[enriched["Driver"] == driver_code].copy()
        self._rivals: pd.DataFrame = enriched[enriched["Driver"] != driver_code].copy()

        self.total_laps: int = int(enriched["LapNumber"].max())

        # Leader's cumulative time per lap — used as the reference for all gap
        # calculations. Computed once to avoid repeated group operations.
        self._leader_cum: dict[int, float] = self._precompute_leader_times()

        # TyreLife at the start of each of our driver's stints. Precomputed for the
        # same reason as the leader times: get_lap_state must stay an O(1) lookup.
        self._stint_baselines: dict[int, int] = self._precompute_stint_baselines()

    def _precompute_stint_baselines(self) -> dict[int, int]:
        """Return our driver's TyreLife at the first lap of each stint.

        N06 trains ``FuelEffect`` as ``(TyreLife - min(TyreLife of the stint)) * 0.055``
        — the seconds of lap time recovered since the stint began. The pace agent has no
        laps frame of its own (``run_pace_agent_from_state`` takes only the lap_state),
        so the baseline has to travel in the lap_state, and this is the producer that
        actually has the stint history. Without it the agent fell back to
        ``fuel_load * 0.03``, ~100x too small, and the model read every lap as a
        fresh-fuel stint start (#446).

        ``min`` rather than the first row's value so the result does not depend on row
        order. Stints whose TyreLife is entirely NaN are omitted, and the consumer then
        degrades to NaN rather than inventing a number.

        ``Stint`` and ``TyreLife`` are not in ``REQUIRED_LAPS_COLUMNS``, so a frame may
        legitimately arrive without them. That yields an empty map, which the consumer
        reads as "no baseline" and turns into the same honest NaN.
        """
        has_columns = {"Stint", "TyreLife"} <= set(self._driver.columns)
        if not has_columns or self._driver.empty:
            return {}

        baselines: dict[int, int] = {}
        for stint, group in self._driver.groupby("Stint"):
            if pd.isna(stint):
                continue
            tyre_life = group["TyreLife"].dropna()
            if tyre_life.empty:
                continue
            baselines[int(stint)] = int(tyre_life.min())
        return baselines

    def _precompute_leader_times(self) -> dict[int, float]:
        """Return the race leader's session elapsed time at the end of each lap.

        Uses the driver in Position=1 for each lap. Falls back to the minimum
        session_time_s across all drivers when Position data is missing (e.g.
        laps 1-2 during a standing start when timing feeds lag). The leader
        always has the smallest session elapsed time at any given lap because
        they crossed the finish line earliest.
        """
        leader_times: dict[int, float] = {}
        for lap in range(1, self.total_laps + 1):
            lap_rows = self._all[self._all["LapNumber"] == lap]
            pos1 = lap_rows[lap_rows["Position"] == 1.0]
            if not pos1.empty and pos1["session_time_s"].notna().any():
                leader_times[lap] = float(pos1["session_time_s"].iloc[0])
            else:
                # Fallback: minimum session time = car that crossed the line first
                valid = lap_rows["session_time_s"].dropna()
                if not valid.empty:
                    leader_times[lap] = float(valid.min())
        return leader_times

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def get_driver_state(self, lap_number: int) -> dict[str, Any]:
        """Full telemetry snapshot for our driver at the end of ``lap_number``.

        Returns all data that would be available from the car's data link and
        the pit wall timing screen: all sector times, speed trap readings, tyre
        state, fuel estimate, track status, and gap to the leader.

        Returns an empty dict when the driver did not complete that lap
        (retirement, DNF, or lap_number out of range). Callers must handle
        this gracefully — agents should treat an empty driver state as a
        race-ended signal.

        Args:
            lap_number: 1-indexed lap number to retrieve.
        """
        row = self._driver[self._driver["LapNumber"] == lap_number]
        if row.empty:
            return {}

        r = row.iloc[0]

        leader_cum = self._leader_cum.get(lap_number)
        driver_cum = r.get("session_time_s")

        if (
            driver_cum is not None
            and leader_cum is not None
            and pd.notna(driver_cum)
            and pd.notna(leader_cum)
        ):
            gap_to_leader: float | None = round(float(driver_cum) - float(leader_cum), 3)
        else:
            gap_to_leader = None

        return {
            "driver": self.driver_code,
            "team": self.team,
            "lap_number": int(lap_number),
            # --- Timing ---
            "lap_time_s": _to_seconds(r.get("LapTime")),
            # The featured parquet's own previous-lap time (N04's Prev_LapTime
            # column) — NOT this lap's lap_time_s reused as a stand-in. The pace
            # agent used to default prev_lap_time to the CURRENT lap's time
            # (`d.get('lap_time_s') or 90.0`), which fed its own most recent
            # prediction back in as the "previous" lap and made pace
            # self-fulfilling (#435). ``.get`` (not indexing) so a laps_df built
            # without this optional column returns None here rather than raising,
            # same as every other optional column read via ``r.get(...)`` below.
            # NaN (no earlier lap, e.g. the first lap of a stint) -> None, same
            # handling as every other timing field in this method.
            "prev_lap_time": _to_seconds(r.get("Prev_LapTime")),
            "sector1_s": _to_seconds(r.get("Sector1Time")),
            "sector2_s": _to_seconds(r.get("Sector2Time")),
            "sector3_s": _to_seconds(r.get("Sector3Time")),
            # --- Position & gap ---
            "position": int(r["Position"]) if pd.notna(r.get("Position")) else None,
            "gap_to_leader_s": gap_to_leader,
            # --- Tyre ---
            "compound": str(r.get("Compound", "")),
            "compound_id": int(r["CompoundID"]) if pd.notna(r.get("CompoundID")) else None,
            "tyre_life": int(r["TyreLife"]) if pd.notna(r.get("TyreLife")) else None,
            "stint": int(r["Stint"]) if pd.notna(r.get("Stint")) else None,
            # TyreLife when this stint began — N06's FuelEffect is measured from it.
            # Driver-only on purpose: the pace model runs on our car, and rivals stay
            # timing-screen-only per the single-driver boundary (#446).
            "stint_baseline_tyre_life": (
                self._stint_baselines.get(int(r["Stint"])) if pd.notna(r.get("Stint")) else None
            ),
            "fresh_tyre": bool(r.get("FreshTyre", False)),
            # --- Speed traps (all four sensor points) ---
            "speed_i1": float(r["SpeedI1"]) if pd.notna(r.get("SpeedI1")) else None,
            "speed_i2": float(r["SpeedI2"]) if pd.notna(r.get("SpeedI2")) else None,
            "speed_fl": float(r["SpeedFL"]) if pd.notna(r.get("SpeedFL")) else None,
            "speed_st": float(r["SpeedST"]) if pd.notna(r.get("SpeedST")) else None,
            # --- Fuel (linear depletion estimate from FuelLoad feature) ---
            "fuel_load": float(r["FuelLoad"]) if pd.notna(r.get("FuelLoad")) else None,
            # --- Track & pit state ---
            "track_status": str(r.get("TrackStatus", "")),
            "is_in_lap": bool(pd.notna(r.get("PitInTime"))),
            "is_out_lap": bool(pd.notna(r.get("PitOutTime"))),
        }

    def get_rival_states(self, lap_number: int) -> list[dict[str, Any]]:
        """Timing-screen-only view of all rivals at end of ``lap_number``.

        Mirrors exactly what a strategy engineer sees on the live timing
        monitor: position, gap to leader, interval to our car, lap time, tyre
        compound and age, and the final speed trap reading. No sector times,
        no fuel data, no detailed speed readings beyond SpeedST.

        ``interval_to_driver_s``: session elapsed time of the rival minus our
        driver's. **Negative → the rival is ahead of us** (they reached the end
        of this lap earlier, so less race time has elapsed for them). Positive
        → we are ahead of them. Verified on Lusail 2025 lap 20: PIA leading,
        every car behind reports a positive interval (NOR +3.578 s).

        Returns a list sorted by Position (ascending). A rival with no position
        that lap keeps ``position=None`` and sorts to the back through a local
        ``99`` placeholder. The placeholder is confined to the sort key on
        purpose: it never enters the emitted dict, so no consumer can look up a
        car "at position 99" and find one. Defaulting an unknown position to a
        number the code also searches by is the #428 bug.

        Args:
            lap_number: 1-indexed lap number to retrieve.
        """
        driver_row = self._driver[self._driver["LapNumber"] == lap_number]
        driver_cum: float | None = (
            driver_row.iloc[0].get("session_time_s") if not driver_row.empty else None
        )
        if driver_cum is not None and pd.isna(driver_cum):
            driver_cum = None

        leader_cum = self._leader_cum.get(lap_number)

        rival_rows = self._rivals[self._rivals["LapNumber"] == lap_number]
        states: list[dict[str, Any]] = []

        for _, r in rival_rows.iterrows():
            rival_cum = r.get("session_time_s")
            if rival_cum is not None and pd.isna(rival_cum):
                rival_cum = None

            gap_to_leader: float | None = (
                round(float(rival_cum) - float(leader_cum), 3)
                if rival_cum is not None and leader_cum is not None
                else None
            )
            interval_to_driver: float | None = (
                round(float(rival_cum) - float(driver_cum), 3)
                if rival_cum is not None and driver_cum is not None
                else None
            )

            states.append(
                {
                    "driver": str(r.get("Driver", "")),
                    "team": str(r.get("Team", "")),
                    "position": int(r["Position"]) if pd.notna(r.get("Position")) else None,
                    "lap_time_s": _to_seconds(r.get("LapTime")),
                    "compound": str(r.get("Compound", "")),
                    "tyre_life": int(r["TyreLife"]) if pd.notna(r.get("TyreLife")) else None,
                    "stint": int(r["Stint"]) if pd.notna(r.get("Stint")) else None,
                    "speed_st": float(r["SpeedST"]) if pd.notna(r.get("SpeedST")) else None,
                    "gap_to_leader_s": gap_to_leader,
                    "interval_to_driver_s": interval_to_driver,
                    "is_pitting": bool(pd.notna(r.get("PitInTime"))),
                }
            )

        return sorted(states, key=lambda s: s["position"] or 99)

    def get_weather_state(
        self,
        lap_number: int,
        weather_df: pd.DataFrame | None = None,
    ) -> dict[str, Any]:
        """Return weather snapshot nearest to ``lap_number``.

        ``TrackStatus`` is always present (sourced from laps). Optional
        ``weather_df`` (from weather.parquet) adds temperature, humidity,
        wind, and rainfall when available. The row is selected by linear
        interpolation of the lap fraction — weather changes slowly enough
        that a simple fractional index is sufficient for a replay demo.

        Args:
            lap_number:  1-indexed lap number.
            weather_df:  DataFrame from weather.parquet. When None, only
                         TrackStatus is returned.
        """
        row = self._driver[self._driver["LapNumber"] == lap_number]
        track_status = str(row.iloc[0]["TrackStatus"]) if not row.empty else ""

        weather: dict[str, Any] = {"track_status": track_status}

        if weather_df is not None and not weather_df.empty:
            lap_frac = (lap_number - 1) / max(self.total_laps - 1, 1)
            idx = int(lap_frac * (len(weather_df) - 1))
            w = weather_df.iloc[idx]
            weather.update(
                {
                    "air_temp": float(w["AirTemp"]) if pd.notna(w.get("AirTemp")) else None,
                    "track_temp": float(w["TrackTemp"]) if pd.notna(w.get("TrackTemp")) else None,
                    "humidity": float(w["Humidity"]) if pd.notna(w.get("Humidity")) else None,
                    "wind_speed": float(w["WindSpeed"]) if pd.notna(w.get("WindSpeed")) else None,
                    "rainfall": bool(w["Rainfall"]) if pd.notna(w.get("Rainfall")) else False,
                    # The session's FIRST track temperature, not this lap's. N14 trains
                    # `track_temp_delta = track_temp - track_temp_start` and it is its 5th
                    # most important feature (6.0% gain), above humidity and the circuit
                    # SC rate. The consumer defaults `track_temp_start` to the CURRENT
                    # temp when it is absent, so the delta came out 0.0 on every lap of
                    # every race: a live feature reading a constant. Real 2024 values run
                    # to -9.1 C (Monaco), -8.6 (Monza), -8.1 (Yas Island).
                    #
                    # It is a session constant, not a per-lap reading, but it ships in the
                    # weather dict because that is the channel the consumer looks in, and
                    # a second channel is how the two paths drifted apart in the first
                    # place (the FastF1 path reads `_wx['TrackTemp'].iloc[0]` correctly).
                    "track_temp_start": (
                        float(weather_df.iloc[0]["TrackTemp"])
                        if pd.notna(weather_df.iloc[0].get("TrackTemp"))
                        else None
                    ),
                }
            )

        return weather

    def get_session_meta(self) -> dict[str, Any]:
        """Static metadata about the race session.

        Does not change lap by lap. Agents use ``gp_name`` to look up
        circuit-specific thresholds (SC probability, cliff laps, undercut
        window). ``total_laps`` is used by the orchestrator to compute how
        deep into the race we are.
        """
        return {
            "gp_name": self.gp_name,
            "year": self.year,
            "driver": self.driver_code,
            "team": self.team,
            "total_laps": self.total_laps,
        }

    def get_lap_state(
        self,
        lap_number: int,
        weather_df: pd.DataFrame | None = None,
    ) -> dict[str, Any]:
        """Merged dict consumed by all agents for ``lap_number``.

        This is the canonical format passed into every agent's
        ``run_*_agent()`` call. The schema is always stable: the four
        top-level keys (``driver``, ``rivals``, ``weather``,
        ``session_meta``) are always present, even if some contain empty
        dicts or empty lists (e.g. after a DNF).

        Args:
            lap_number:  1-indexed lap number to emit (1 … total_laps).
            weather_df:  Optional weather DataFrame. When None, only
                         ``TrackStatus`` from the laps parquet is included
                         in the weather dict.

        Returns:
            ::

                {
                    "lap_number":   int,
                    "driver":       dict,   # full telemetry — see get_driver_state
                    "rivals":       list,   # timing-only — see get_rival_states
                    "weather":      dict,   # see get_weather_state
                    "session_meta": dict,   # see get_session_meta
                }
        """
        return {
            "lap_number": lap_number,
            "driver": self.get_driver_state(lap_number),
            "rivals": self.get_rival_states(lap_number),
            "weather": self.get_weather_state(lap_number, weather_df),
            "session_meta": self.get_session_meta(),
        }
