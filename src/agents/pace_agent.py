"""Pace Agent — src/agents/pace_agent.py

Extracted from N25_pace_agent.ipynb. Wraps the N06 XGBoost delta-lap-time
model into a clean OOP agent interface that returns lap time predictions,
delta signals, and bootstrap confidence intervals.

Public API
----------
run_pace_agent(**kwargs)               → PaceOutput
run_pace_agent_from_state(lap_state)   → PaceOutput

Internal structure
------------------
PaceAgent encapsulates all model state (XGBoost, encoding maps, reference
laps) as instance attributes. A module-level lazy singleton
(_default_pace_agent) is used by the module-level entry points so the
existing public API is preserved without globals.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import xgboost as xgb

from src.agents._shared_defaults import reading_or_default

# ── Repo root (with root-stop guard for uv tool install) ─────────────────────
_REPO_ROOT = Path(__file__).resolve().parent
while not (_REPO_ROOT / '.git').exists():
    if _REPO_ROOT.parent == _REPO_ROOT:
        break
    _REPO_ROOT = _REPO_ROOT.parent

# Route model + processed-data paths through the user cache helper so that
# `uv tool install` users land on ``~/.f1-strat/data/`` automatically; dev
# checkouts with a repo-relative ``data/`` short-circuit the helper.
try:
    from src.f1_strat_manager.data_cache import get_data_root as _get_data_root
    _DATA_ROOT = _get_data_root()
except (ImportError, OSError, RuntimeError):
    # Every way get_data_root() can fail, enumerated against its body in
    # src/f1_strat_manager/data_cache.py: ImportError from the import itself on a
    # bare dev checkout; OSError from the three mkdir() calls (read-only mount,
    # permissions); RuntimeError from Path.home() and Path.expanduser(), which
    # raise when no home directory resolves. That last one is not hypothetical:
    # the Path.home() branch IS the `uv tool install` path this block exists to
    # serve, and a container with no HOME would take it. Falling back to the
    # repo-relative data/ is right for all three.
    _DATA_ROOT = _REPO_ROOT / 'data'

# What stands in for the previous lap when there genuinely is not one: the first lap
# of a race, or of a stint, where N04's Prev_LapTime is NaN by construction.
#
# It is a module constant rather than a literal because it had already been restated:
# strategy_orchestrator's dict path carried 92.0 for the same quantity, so the two
# entry points of the same model disagreed about the same missing value. Exported so
# a second copy cannot appear again (#766).
#
# `_predict` reads this straight into `prev + delta` with no NaN branch, so it cannot
# be None. That is the whole reason a placeholder exists at all, and it is why every
# caller must use `or` rather than the two-arg `dict.get`: the key is PRESENT with a
# None value, which the two-arg form does not substitute for.
MISSING_PREV_LAP_TIME_S = 90.0

_MODELS_DIR = _DATA_ROOT / 'models' / 'lap_time'
_PROCESSED  = _DATA_ROOT / 'processed'

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
N_BOOTSTRAP: int   = 200
_NOISE_PCT: float  = 0.02   # 2 % Gaussian noise on continuous features

# Seconds of lap time recovered per lap as fuel burns off. N04 builds the training
# feature as (TyreLife - min(TyreLife of the stint)) * 0.055 — verified exactly against
# laps_featured_2025 (100% of laps reproduce, range 0..3.685 s). The same coefficient
# lives in tire_agent._add_fuel_cols; deliberately duplicated rather than shared, since
# unifying constants across agent modules is a wider refactor than this fix (#446).
FUEL_GAIN_PER_LAP_S: float = 0.055


# ─────────────────────────────────────────────────────────────────────────────
# PaceOutput dataclass (public API — untouched)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PaceOutput:
    """Structured output of the Pace Agent for one lap.

    lap_time_pred is the N06 XGBoost prediction in absolute seconds — the model
    outputs a delta vs Prev_LapTime internally; this field adds Prev_LapTime back
    so all downstream agents work in absolute lap time.

    delta_vs_prev is the raw model delta (predicted lap_time minus Prev_LapTime).
    Negative means the driver is faster than the previous lap.

    delta_vs_median is the difference between lap_time_pred and the historical
    session median for this GP/year/compound combination. NaN when no median
    reference is available (new circuits or sparse data).

    ci_p10 and ci_p90 are the P10/P90 bootstrap confidence bounds on
    lap_time_pred, computed over N_BOOTSTRAP perturbations of the continuous
    input features. N31 Monte Carlo simulation samples from this interval to
    model pace uncertainty across strategy candidates.

    reasoning is a human-readable summary forwarded verbatim to the N31
    Orchestrator for LLM synthesis.
    """

    lap_time_pred:   float
    delta_vs_prev:   float
    delta_vs_median: float
    ci_p10:          float
    ci_p90:          float
    reasoning:       str = ""


# ─────────────────────────────────────────────────────────────────────────────
# PaceAgent class
# ─────────────────────────────────────────────────────────────────────────────

class PaceAgent:
    """Encapsulates the N06 XGBoost lap-time prediction pipeline.

    All model artifacts (XGBoost weights, encoding maps, reference laps) are
    loaded once in __init__ and stored as instance attributes — no module-level
    globals are used.

    Deliberately deterministic, unlike its tire/pit/race_situation siblings:
    pace has no qualitative judgment to make (no warning_level/action/threat_level
    category alongside its numbers), so there is no LLM step to wire — see #778/#780
    for the archaeology and decision record.

    Instantiate via the module-level _get_default_pace_agent() factory to avoid
    redundant disk I/O; do not instantiate PaceAgent directly in hot paths.

    Args:
        models_dir: Directory containing xgb_laptime_delta_final.json and
            the feature name JSON. Defaults to the repo-root–relative path.
        processed_dir: Directory containing circuit clusters, laps_featured,
            and feature manifest. Defaults to the repo-root–relative path.
    """

    def __init__(
        self,
        models_dir: Path = _MODELS_DIR,
        processed_dir: Path = _PROCESSED,
    ) -> None:
        self.model, self.features     = self._load_model(models_dir)
        self.compound_id: dict        = {}
        self.circuit_cluster: dict    = {}
        self.team_id: dict            = {}
        self.compound_id, self.circuit_cluster, self.team_id = self._load_encoding_maps(processed_dir)
        self.laps_ref: pd.DataFrame   = self._load_reference_laps(processed_dir)

    # ── Loaders ───────────────────────────────────────────────────────────────

    def _load_model(
        self, models_dir: Path
    ) -> tuple[xgb.XGBRegressor, list[str]]:
        """Load N06 XGBoost model and ordered feature name list from disk.

        Both artifacts are returned together to guarantee the feature order is
        always consistent with the model version — callers must not reorder or
        drop features between load and predict.

        Args:
            models_dir: Directory containing the two JSON export files.

        Returns:
            Tuple (model, features) where model is a fitted XGBRegressor and
            features is a list of column name strings in predict order.
        """
        features = json.loads(
            (models_dir / 'xgb_laptime_delta_feature_names.json').read_text()
        )
        model = xgb.XGBRegressor()
        model.load_model(models_dir / 'xgb_laptime_delta_final.json')
        return model, features

    def _load_encoding_maps(
        self, processed_dir: Path
    ) -> tuple[dict, dict, dict]:
        """Load compound, circuit-cluster, and team label-encoding maps.

        Reads the compound encoding from the N06 feature manifest, the circuit
        cluster assignments from the k=4 clustering parquet (N05), and the
        team-to-integer map derived from the laps_featured parquet. All three
        are static training artifacts — they must not be recomputed at inference
        time to avoid encoding drift between train and serve.

        Args:
            processed_dir: Root of the processed data directory.

        Returns:
            Tuple (compound_id, circuit_cluster, team_id) dicts.
        """
        manifest    = json.loads((processed_dir / 'feature_manifest_laptime.json').read_text())
        compound_id = manifest['categorical_encoding']['Compound']

        clusters_df = pd.read_parquet(
            processed_dir / 'circuit_clustering' / 'circuit_clusters_k4_2025.parquet',
            columns=['GP_Name', 'Cluster'],
        )
        circuit_cluster = dict(
            zip(clusters_df['GP_Name'], clusters_df['Cluster'].astype(int))
        )

        laps = pd.read_parquet(
            processed_dir / 'laps_featured_2025.parquet',
            columns=['Team', 'TeamID'],
        ).dropna()
        team_id = (
            laps.drop_duplicates('Team')
                .set_index('Team')['TeamID']
                .astype(int)
                .to_dict()
        )
        return compound_id, circuit_cluster, team_id

    def _load_reference_laps(self, processed_dir: Path) -> pd.DataFrame:
        """Load the reference laps parquet used for session median computation.

        Four columns are loaded to keep the in-memory footprint small. The median
        baseline is used by N31 to contextualise absolute predictions.

        Args:
            processed_dir: Root of the processed data directory.

        Returns:
            DataFrame with columns GP_Name, Year, Compound, LapTime_s.
        """
        return pd.read_parquet(
            processed_dir / 'laps_featured_2025.parquet',
            columns=['GP_Name', 'Year', 'Compound', 'LapTime_s'],
        )

    # ── Encoding helpers ──────────────────────────────────────────────────────

    def _encode_categorical(
        self, compound: str, team: str, gp_name: str
    ) -> tuple[int, int, int]:
        """Map compound, team, and circuit to their integer label encodings.

        Unknown categories degrade gracefully to the most common training
        value rather than raising an error — compound→1, team→0, cluster→1.

        Args:
            compound: Pirelli compound string ('SOFT', 'MEDIUM', 'HARD', etc.).
            team: Team name matching self.team_id keys (e.g. 'McLaren').
            gp_name: GP name matching self.circuit_cluster keys.

        Returns:
            Tuple (compound_id_int, team_id_int, cluster_int).
        """
        c_id    = self.compound_id.get(compound, 1)
        t_id    = self.team_id.get(team, 0)
        cluster = self.circuit_cluster.get(gp_name, 1)
        return c_id, t_id, cluster

    def _compute_derived(
        self,
        tyre_life: int,
        fuel_load: float,
        lap_number: int,
        total_laps: int,
        prev_speed_st: float,
        mean_sector_speed: Optional[float],
        stint_baseline_tyre_life: Optional[int] = None,
    ) -> dict:
        """Compute features derived from raw inputs that are not in the source data.

        FreshTyre: binary flag for the first lap on a new tyre set — captures
        the outlap pace loss caused by tyre heating and rubber laydown.
        FuelEffect: cumulative fuel burn pace gain (lighter car = faster lap).
        laps_remaining: inverted lap count used as a proxy for race phase.
        mean_sector_speed: falls back to prev_speed_st when circuit_features
        are unavailable for the current GP.

        Args:
            tyre_life: Current laps on this tyre set.
            fuel_load: Estimated fuel fraction in [0, 1].
            lap_number: Current race lap.
            total_laps: Total scheduled race laps.
            prev_speed_st: Speed trap reading in km/h from the previous lap.
            mean_sector_speed: Average sector speed; None → use prev_speed_st.
            stint_baseline_tyre_life: TyreLife recorded at the start of the
                current stint. None means the caller has not been updated to
                supply it, in which case FuelEffect is forced to NaN instead
                of falling back to the old (wrong) formula — see the comment
                in the body for why NaN is the safe default here (#446).

        Returns:
            Dict with keys FreshTyre, FuelEffect, laps_remaining,
            mean_sector_speed.
        """
        # FuelEffect is the cumulative time recovered since the stint started, in
        # SECONDS (training range 0..3.685 s). It was being computed as
        # `fuel_load * 0.03` — a [0, 0.03] value, ~100x too small and semantically
        # different — so the model always read "fresh-fuel stint start" and the learned
        # fuel-burn pace gain was suppressed on every lap (#446).
        #
        # Absent baseline -> NaN, never a fabricated number. NaN is IN-DISTRIBUTION here:
        # the training parquet itself carries 2.0% null FuelEffect, so XGBoost has a
        # learned default direction for it, and `_build_feature_row`'s to_numeric(coerce)
        # tail already routes None -> NaN -> the model's native missing handling. The old
        # formula must NOT survive as the fallback: a producer we missed would silently
        # reproduce the exact bug this kills, whereas a NaN plus a warning is impossible
        # to mistake for a reading.
        if stint_baseline_tyre_life is None:
            logger.warning(
                'stint_baseline_tyre_life absent from lap_state: FuelEffect=NaN for this '
                'lap; the producer must supply it (#446)'
            )
            fuel_effect = float('nan')
        else:
            fuel_effect = (tyre_life - stint_baseline_tyre_life) * FUEL_GAIN_PER_LAP_S

        return {
            'FreshTyre':        int(tyre_life <= 1),
            'FuelEffect':       fuel_effect,
            'laps_remaining':   max(0, total_laps - lap_number),
            'mean_sector_speed': mean_sector_speed if mean_sector_speed is not None else prev_speed_st,
        }

    def _build_feature_row(
        self,
        driver_number: int,
        lap_number: int,
        stint: int,
        tyre_life: int,
        compound: str,
        position: Optional[int],
        team: str,
        laps_since_pit: int,
        fuel_load: float,
        year: int,
        prev_lap_time: float,
        prev_tyre_life: int,
        prev_speed_st: float,
        air_temp: float,
        track_temp: float,
        humidity: float,
        rainfall: float,
        total_laps: int,
        gp_name: str,
        mean_sector_speed: Optional[float] = None,
        prev_deg_rate: float = 0.0,
        prev_cum_deg: float = 0.0,
        prev_deg_accel: float = 0.0,
        stint_baseline_tyre_life: Optional[int] = None,
    ) -> pd.DataFrame:
        """Pack raw race state into a single-row DataFrame ready for predict().

        Encodes categorical inputs using self.*_id maps, appends derived
        features from _compute_derived(), and selects columns in the exact
        order self.features expects.

        Returns:
            Single-row pd.DataFrame with columns in self.features order.
        """
        c_id, t_id, cluster = self._encode_categorical(compound, team, gp_name)
        derived = self._compute_derived(
            tyre_life, fuel_load, lap_number, total_laps,
            prev_speed_st, mean_sector_speed, stint_baseline_tyre_life,
        )

        row = {
            'DriverNumber':         driver_number,
            'LapNumber':            lap_number,
            'Stint':                stint,
            'TyreLife':             tyre_life,
            'FreshTyre':            derived['FreshTyre'],
            'Position':             position,
            'CompoundID':           c_id,
            'TeamID':               t_id,
            'LapsSincePitStop':     laps_since_pit,
            'FuelLoad':             fuel_load,
            'Year':                 year,
            'FuelEffect':           derived['FuelEffect'],
            'Prev_LapTime':         prev_lap_time,
            'Prev_TyreLife':        prev_tyre_life,
            'Prev_SpeedST':         prev_speed_st,
            'AirTemp':              air_temp,
            'TrackTemp':            track_temp,
            'Humidity':             humidity,
            'Rainfall':             rainfall,
            'laps_remaining':       derived['laps_remaining'],
            'Cluster':              cluster,
            'mean_sector_speed':    derived['mean_sector_speed'],
            'Prev_DegradationRate': prev_deg_rate,
            'Prev_CumulativeDeg':   prev_cum_deg,
            'Prev_DegAcceleration': prev_deg_accel,
        }
        df = pd.DataFrame([row])[self.features]
        # Belt-and-braces: any caller that slips a None through lands here.
        # XGBoost refuses object-dtype columns; ``to_numeric(errors='coerce')``
        # converts None→NaN and the model handles NaN natively via its
        # sparse-aware split logic (default_left). Cheap and defensive —
        # no-op on already-numeric frames.
        return df.apply(pd.to_numeric, errors='coerce')

    # ── Inference helpers ─────────────────────────────────────────────────────

    def _predict(self, feature_df: pd.DataFrame) -> float:
        """Predict absolute lap time by adding the XGBoost delta to Prev_LapTime.

        The N06 model predicts a signed delta vs the previous lap, not an
        absolute time. This method adds the delta back so callers always receive
        an absolute lap time in seconds.

        Args:
            feature_df: Single-row DataFrame from _build_feature_row().

        Returns:
            Absolute predicted lap time in seconds.
        """
        delta = float(self.model.predict(feature_df)[0])
        prev  = float(feature_df['Prev_LapTime'].iloc[0])
        return prev + delta

    def _bootstrap_ci(
        self,
        feature_df: pd.DataFrame,
        n: int = N_BOOTSTRAP,
        seed: int = 42,
    ) -> tuple[float, float]:
        """Estimate a P10/P90 confidence interval via Gaussian feature perturbation.

        Runs n forward passes, each time adding independent Gaussian noise
        (sigma = NOISE_PCT × feature_value) to the continuous features most
        subject to real-world variability. The noise scale approximates sensor
        noise and lap-to-lap variation; it is not formal Bayesian uncertainty.

        N31 uses this interval to sample pace scenarios in Monte Carlo strategy
        evaluation — a wider interval increases the variance of the strategy
        score distribution and makes the agent more conservative.

        Args:
            feature_df: Single-row DataFrame from _build_feature_row().
            n: Number of bootstrap samples (default N_BOOTSTRAP = 200).
            seed: Integer seed for reproducibility.

        Returns:
            Tuple (p10, p90) of absolute lap times in seconds.
        """
        noise_cols = ['Prev_LapTime', 'Prev_SpeedST', 'mean_sector_speed',
                      'AirTemp', 'TrackTemp', 'TyreLife']

        rng     = np.random.default_rng(seed)
        base    = feature_df.values.copy().astype(float)
        col_idx = {c: feature_df.columns.get_loc(c) for c in noise_cols}

        preds = []
        for _ in range(n):
            row = base.copy()
            for col, idx in col_idx.items():
                sigma = abs(base[0, idx]) * _NOISE_PCT
                row[0, idx] += rng.normal(0, sigma)
            df_row = pd.DataFrame(row, columns=feature_df.columns)
            delta  = float(self.model.predict(df_row)[0])
            preds.append(float(df_row['Prev_LapTime'].iloc[0]) + delta)

        return float(np.percentile(preds, 10)), float(np.percentile(preds, 90))

    def _session_median(
        self, gp_name: str, year: int, compound: str
    ) -> Optional[float]:
        """Return the historical median lap time for a GP / year / compound.

        Filters self.laps_ref to the matching GP, year, and compound, then
        returns the median of LapTime_s. N31 uses this value to contextualise
        the absolute predicted lap time — a large positive delta_vs_median
        signals a degrading tyre or a slower compound choice.

        Args:
            gp_name: GP name matching the GP_Name column.
            year: Race year integer.
            compound: Pirelli compound string.

        Returns:
            Median lap time in seconds, or None when no matching laps exist.
        """
        mask = (
            (self.laps_ref['GP_Name']  == gp_name) &
            (self.laps_ref['Year']     == year)     &
            (self.laps_ref['Compound'] == compound)
        )
        subset = self.laps_ref.loc[mask, 'LapTime_s'].dropna()
        return float(subset.median()) if len(subset) > 0 else None

    # ── Main inference entrypoint ─────────────────────────────────────────────

    def run(
        self,
        driver_number: int,
        lap_number: int,
        stint: int,
        tyre_life: int,
        compound: str,
        position: Optional[int],
        team: str,
        laps_since_pit: int,
        fuel_load: float,
        year: int,
        prev_lap_time: float,
        prev_tyre_life: int,
        prev_speed_st: float,
        air_temp: float,
        track_temp: float,
        humidity: float,
        rainfall: float,
        total_laps: int,
        gp_name: str,
        mean_sector_speed: Optional[float] = None,
        prev_deg_rate: float = 0.0,
        prev_cum_deg: float = 0.0,
        prev_deg_accel: float = 0.0,
        stint_baseline_tyre_life: Optional[int] = None,
    ) -> PaceOutput:
        """Run pace prediction for a single lap and return a PaceOutput.

        Builds the N06 feature vector, calls the XGBoost model, computes a
        bootstrap P10/P90 uncertainty interval, and looks up the historical
        session median for the current GP/year/compound.

        Args:
            driver_number: Car number used to look up TeamID encoding.
            lap_number: Current race lap; used for FuelLoad estimation.
            stint: Stint number (1-indexed), forwarded as a raw feature.
            tyre_life: Laps on current tyre set; drives FreshTyre flag.
            compound: Pirelli compound name.
            position: Current race position (1-based). None when the source
                telemetry has no reading for this lap; propagates as a missing
                'Position' feature (NaN after _build_feature_row's numeric
                coercion) rather than a fabricated grid slot, since XGBoost
                splits natively on missing values (see the belt-and-braces
                comment in _build_feature_row).
            team: Team name matching self.team_id encoding map.
            laps_since_pit: Laps since most recent pit stop.
            fuel_load: Estimated fuel fraction in [0, 1].
            year: Race year (2023/2024/2025).
            prev_lap_time: Previous lap time in seconds.
            prev_tyre_life: TyreLife on the previous lap.
            prev_speed_st: Speed trap reading in km/h from the previous lap.
            air_temp: Air temperature in °C.
            track_temp: Track surface temperature in °C.
            humidity: Relative humidity in %.
            rainfall: True if rain was recorded during this lap.
            total_laps: Total scheduled race laps.
            gp_name: GP name matching self.circuit_cluster keys.
            mean_sector_speed: Average sector speed; defaults to prev_speed_st.
            prev_deg_rate: Degradation rate from the previous lap (s/lap).
            prev_cum_deg: Cumulative degradation at the previous lap.
            prev_deg_accel: Second derivative of degradation (s/lap²).
            stint_baseline_tyre_life: TyreLife at the start of the current
                stint. None forces FuelEffect to NaN rather than a fabricated
                estimate — see _compute_derived.

        Returns:
            PaceOutput with all fields populated and a reasoning string.
        """
        feature_df = self._build_feature_row(
            driver_number=driver_number, lap_number=lap_number, stint=stint,
            tyre_life=tyre_life, compound=compound, position=position, team=team,
            laps_since_pit=laps_since_pit, fuel_load=fuel_load, year=year,
            prev_lap_time=prev_lap_time, prev_tyre_life=prev_tyre_life,
            prev_speed_st=prev_speed_st, air_temp=air_temp, track_temp=track_temp,
            humidity=humidity, rainfall=rainfall, total_laps=total_laps,
            gp_name=gp_name, mean_sector_speed=mean_sector_speed,
            prev_deg_rate=prev_deg_rate, prev_cum_deg=prev_cum_deg,
            prev_deg_accel=prev_deg_accel,
            stint_baseline_tyre_life=stint_baseline_tyre_life,
        )

        lap_time_pred   = self._predict(feature_df)
        delta_vs_prev   = lap_time_pred - prev_lap_time
        p10, p90        = self._bootstrap_ci(feature_df)
        median          = self._session_median(gp_name, year, compound)
        delta_vs_median = (lap_time_pred - median) if median is not None else float('nan')

        trend  = "faster" if delta_vs_prev < 0 else "slower"
        vs_med = (
            f"{delta_vs_median:+.3f}s vs median"
            if median is not None else "no median reference"
        )
        reasoning = (
            f"Lap {lap_number}: predicted {round(lap_time_pred, 3):.3f}s "
            f"({round(delta_vs_prev, 3):+.3f}s, {trend} than prev). "
            f"CI [{round(p10, 1):.1f}-{round(p90, 1):.1f}s]. {vs_med}."
        )

        return PaceOutput(
            lap_time_pred   = round(lap_time_pred, 3),
            delta_vs_prev   = round(delta_vs_prev, 3),
            delta_vs_median = round(delta_vs_median, 3),
            ci_p10          = round(p10, 3),
            ci_p90          = round(p90, 3),
            reasoning       = reasoning,
        )

    def run_from_state(self, lap_state: dict) -> PaceOutput:
        """RSM adapter: run pace prediction from a RaceStateManager lap_state dict.

        Translates the nested lap_state produced by RaceStateManager.get_lap_state()
        into the flat kwargs expected by self.run(). Fields absent from the RSM
        schema (prev_deg_rate, prev_cum_deg, prev_deg_accel, mean_sector_speed)
        default to 0.0/None since the replay engine does not compute degradation
        history.

        Args:
            lap_state: Dict produced by RaceStateManager.get_lap_state(). Expected
                keys: lap_number, driver (full telemetry dict), weather (dict),
                session_meta (gp_name, year, driver, team, total_laps).

        Returns:
            PaceOutput with all fields populated.
        """
        d    = lap_state['driver']
        meta = lap_state['session_meta']
        wx   = lap_state.get('weather', {})

        lap_number     = lap_state['lap_number']
        total_laps     = meta['total_laps']
        laps_remaining = max(0, total_laps - lap_number)

        # ``dict.get(key, default)`` only applies the default when *key is
        # absent* — if the key is present with a None value (FastF1 logs
        # speed_st=None on laps where the trap beam is not crossed cleanly,
        # e.g. inlaps, traffic interruptions, sector anomalies), the default
        # is ignored and None flows into the DataFrame. The feature column
        # then becomes object dtype and XGBoost rejects it with:
        #     "DataFrame.dtypes for data must be int, float, bool or category"
        # That is exactly the crash observed on mid-stint laps. The ``or``
        # pattern handles both missing-key and None-value cases in one step.
        # This file's inline guard was the ONLY one of the three agents that handled
        # present-and-None, and it is now the shared helper the other two adopted rather
        # than a fourth copy of the pattern (#788).
        _speed_st  = d.get('speed_st')   or 300.0
        _air_temp  = reading_or_default(wx, 'air_temp',   25.0)
        _trk_temp  = reading_or_default(wx, 'track_temp', 35.0)
        _humidity  = reading_or_default(wx, 'humidity',   50.0)
        _rainfall  = reading_or_default(wx, 'rainfall', 0)

        return self.run(
            driver_number  = d.get('driver_number') or 0,
            lap_number     = lap_number,
            stint          = d.get('stint') or 1,
            tyre_life      = d.get('tyre_life') or 1,
            compound       = d.get('compound') or 'MEDIUM',
            # Plain .get, no `or` fallback: `d.get('position') or 1` collapsed a
            # missing telemetry reading AND the #428 sentinel (a stored `0`)
            # into P1, the race leader, straight into the live N06 XGBoost
            # feature (#628). Unlike race_situation_agent/pit_strategy_agent,
            # this value is never used in a `position - 1` rival lookup here,
            # so there is no lookup to fool — the honest fix is simply to let
            # an unknown position propagate as None. _build_feature_row's
            # existing pd.to_numeric(errors='coerce') turns that into NaN, and
            # XGBoost handles a missing 'Position' natively via its
            # default-left split direction, so no fabricated value is needed.
            position       = d.get('position'),
            team           = meta.get('team') or 'Unknown',
            laps_since_pit = d.get('tyre_life') or 1,
            fuel_load      = laps_remaining / max(total_laps, 1),
            year           = meta.get('year') or 2025,
            # ``d.get('prev_lap_time') or 90.0``, NOT ``d.get('lap_time_s')``: the
            # latter fed this LAP's own time back in as the PREVIOUS lap's time, so
            # the model chased its own most recent prediction instead of the real
            # preceding lap (#435). RaceStateManager.get_driver_state now emits the
            # real 'prev_lap_time' sourced from the parquet's Prev_LapTime column.
            # ``or``, not the two-arg get(key, default) form: Prev_LapTime is NaN
            # on the first lap of a stint/race (no earlier lap exists), which
            # get_driver_state turns into an explicit ``None`` — present key, None
            # value — and the two-arg form only substitutes when the KEY is
            # absent, never when the VALUE is (the same Series.get trap #428/#446/
            # #462 keep finding). Unlike stint_baseline_tyre_life below, None
            # cannot be allowed through here: _predict() reads Prev_LapTime
            # straight into ``prev + delta`` with no NaN branch, so a bare None
            # would turn lap_time_pred itself into NaN. 90.0 is the same
            # order-of-magnitude placeholder the old (wrong) code used, now only
            # reached on a genuinely missing previous lap.
            prev_lap_time  = d.get('prev_lap_time') or MISSING_PREV_LAP_TIME_S,
            prev_tyre_life = max(0, (d.get('tyre_life') or 1) - 1),
            prev_speed_st  = float(_speed_st),
            air_temp       = float(_air_temp),
            track_temp     = float(_trk_temp),
            humidity       = float(_humidity),
            rainfall       = float(_rainfall or 0),
            total_laps     = total_laps,
            gp_name        = meta.get('gp_name') or '',
            prev_deg_rate  = 0.0,
            prev_cum_deg   = 0.0,
            prev_deg_accel = 0.0,
            # Plain .get, deliberately NOT the `or` pattern used above: `or` would
            # collapse a legitimate baseline of 0 into None and re-introduce exactly
            # the sentinel-vs-real-value confusion this epic is about. Absent stays
            # absent, and _compute_derived turns that into a NaN plus a warning (#446).
            stint_baseline_tyre_life = d.get('stint_baseline_tyre_life'),
        )


# ─────────────────────────────────────────────────────────────────────────────
# Module-level lazy singleton
# ─────────────────────────────────────────────────────────────────────────────

_default_pace_agent: Optional[PaceAgent] = None


def _get_default_pace_agent() -> PaceAgent:
    """Return the shared PaceAgent instance, creating it on the first call.

    Uses a module-level variable so model weights are only loaded once per
    process, regardless of how many times the public entry points are called.
    Thread safety: acceptable for single-threaded inference; for multi-threaded
    servers wrap in a threading.Lock if needed.
    """
    global _default_pace_agent
    if _default_pace_agent is None:
        _default_pace_agent = PaceAgent()
    return _default_pace_agent


# ─────────────────────────────────────────────────────────────────────────────
# Public entry points (backward-compatible API — same signatures as before)
# ─────────────────────────────────────────────────────────────────────────────

def run_pace_agent(
    driver_number, lap_number, stint, tyre_life, compound,
    position, team, laps_since_pit, fuel_load, year,
    prev_lap_time, prev_tyre_life, prev_speed_st,
    air_temp, track_temp, humidity, rainfall,
    total_laps, gp_name,
    mean_sector_speed=None,
    prev_deg_rate=0.0, prev_cum_deg=0.0, prev_deg_accel=0.0,
) -> PaceOutput:
    """Run the Pace Agent for a single lap and return a structured PaceOutput.

    Thin entry point that delegates to the shared PaceAgent singleton. All
    inference logic lives in PaceAgent.run() — see its docstring for full
    parameter documentation.
    """
    return _get_default_pace_agent().run(
        driver_number=driver_number, lap_number=lap_number, stint=stint,
        tyre_life=tyre_life, compound=compound, position=position, team=team,
        laps_since_pit=laps_since_pit, fuel_load=fuel_load, year=year,
        prev_lap_time=prev_lap_time, prev_tyre_life=prev_tyre_life,
        prev_speed_st=prev_speed_st, air_temp=air_temp, track_temp=track_temp,
        humidity=humidity, rainfall=rainfall, total_laps=total_laps,
        gp_name=gp_name, mean_sector_speed=mean_sector_speed,
        prev_deg_rate=prev_deg_rate, prev_cum_deg=prev_cum_deg,
        prev_deg_accel=prev_deg_accel,
    )


def run_pace_agent_from_state(lap_state: dict) -> PaceOutput:
    """Adapter: run the Pace Agent from a RaceStateManager lap_state dict.

    Thin entry point that delegates to PaceAgent.run_from_state(). See that
    method's docstring for full documentation on the lap_state schema.

    Args:
        lap_state: Dict produced by RaceStateManager.get_lap_state().

    Returns:
        PaceOutput with all fields populated.
    """
    return _get_default_pace_agent().run_from_state(lap_state)
