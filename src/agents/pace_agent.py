"""Pace Agent: src/agents/pace_agent.py

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

from src.agents.race_state_builder import UNKNOWN_TYRE_LIFE
from src.agents._shared_defaults import (
    DEFAULT_AIR_TEMP_C,
    DEFAULT_TRACK_TEMP_C,
    reading_or_default,
)
from src.f1_strat_manager.gp_slugs import (
    normalise_gp_key,
    resolve_gp_key,
    slug_from_event_name,
)

# Safe in this direction: envelope.py is a leaf that imports nothing from src.agents.
from src.strategy.inference.envelope import OperatingEnvelope

# ── Repo root (with root-stop guard for uv tool install) ─────────────────────
_REPO_ROOT = Path(__file__).resolve().parent
while not (_REPO_ROOT / ".git").exists():
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
    _DATA_ROOT = _REPO_ROOT / "data"

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

_MODELS_DIR = _DATA_ROOT / "models" / "lap_time"
_PROCESSED = _DATA_ROOT / "processed"

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
N_BOOTSTRAP: int = 200
_NOISE_PCT: float = 0.02  # 2% Gaussian noise on continuous features

# Seconds of lap time recovered per lap as fuel burns off. N04 builds the training
# feature as (TyreLife - min(TyreLife of the stint)) * 0.055, verified exactly against
# laps_featured_2025 (100% of laps reproduce, range 0..3.685 s). The same coefficient
# lives in tire_agent._add_fuel_cols; deliberately duplicated rather than shared, since
# unifying constants across agent modules is a wider refactor than this fix (#446).
FUEL_GAIN_PER_LAP_S: float = 0.055


# ── N06's operating envelope (#710) ──────────────────────────────────────────
# The range each continuous feature actually took across N06's training seasons,
# so that a call outside it stops being silent. Until now this agent had no range
# check of any kind, which is the condition the envelope contract was written for:
# an XGBoost regressor answers an out-of-range call with exactly the confidence it
# answers an in-range one, and N26 spent two years doing precisely that.
#
# MEASURED, NOT CHOSEN, and reproducible: rebuild 2023 + 2024 through
# `augment_featured_laps`, apply the two N06 feature steps `pace_holdout.py` already
# owns (`_encode_categoricals` then `_add_lag_deg_features`), drop the rows N06 drops,
# and take the min/max of each column over the resulting 42,957 rows.
# `tests/agents/test_n06_envelope.py` re-runs that measurement and fails if any bound
# below has drifted from it, so a hand-typed number here fails that test.
#
# WHICH TEN OF THE TWENTY-FIVE, and why each of the other fifteen is out. A bound is a
# claim that the value at inference is the same quantity, in the same units, as the
# column the range was measured from. Where that is not true, a bound does not measure
# extrapolation, it measures a wiring defect, and it reports it under the wrong name.
#
#   Excluded, no range to be outside of (8): DriverNumber, TeamID, CompoundID, Cluster,
#   Year, Stint and Position are identifiers, codes or ranks; FreshTyre and Rainfall are
#   flags. A bound on a label is a category check wearing the wrong contract.
#
#   Excluded, the value at inference is NOT the quantity the range describes (4):
#     - The three Prev_Deg* features. `run_from_state` hardcodes all three to 0.0 on
#       every real call, which reads like the textbook out-of-range bug. Measured, it is
#       not one: 0.0 sits mid-distribution for each (42.6% / 41.9% / 46.7% of training
#       rows fall below it), so a bound could never fire and declaring one would
#       advertise a check that cannot work.
#     `mean_sector_speed` does NOT belong in this exclusion list. `_compute_derived`
#     used to substitute `prev_speed_st` whenever no mean sector speed was
#     supplied and `run_from_state` never supplied one, so the feature carried the speed
#     trap on every real call: a different physical quantity, training means 256.8 vs
#     303.0 km/h, and a bound over the first applied to the second fired on 83% of laps
#     at Monza while describing none of them. #797 fixed the FEED rather than deleting
#     the bound, so the value is once again the quantity the range was measured from and
#     the bound is once again meaningful: it now fires when a circuit falls outside the
#     set N06 was fitted on, which is the question it was always supposed to ask.
#
#   Excluded, a bound would report the same event twice (1): `LapsSincePitStop`.
#   `run_from_state` passes `d.get('tyre_life') or 1` for BOTH it and `TyreLife`, so at
#   inference they are the same number and a second bound is a second warning about one
#   underlying cause.
#
#   Excluded, training and inference encode it differently (1): `FuelLoad`. The featured
#   artefact stores it rounded to four decimals, giving a measured maximum of 0.9615,
#   while inference computes `laps_remaining / total_laps` live and unrounded. A 78-lap
#   race yields 0.96153..., above that maximum, so the bound would fire on a class of lap
#   the model was trained on. Comparing two encodings of a quantity is not a range check.
#
# Feeding a model a constant, or the wrong quantity, or a differently-rounded one, are
# all real defects. They are simply not THIS defect, and each needs its own instrument.
#
# The bounds are TRAINING-season ranges (2023-2024). This is not the same thing as the
# "range 0..3.685 s" recorded next to FUEL_GAIN_PER_LAP_S above, which was measured on
# laps_featured_2025: 2025 is the held-out TEST season, and an envelope sourced from it
# would be describing where the model is asked to work rather than where it was fitted.
_N06_TRAINED_BOUNDS: dict[str, tuple[float, float]] = {
    "LapNumber": (3.0, 78.0),
    "TyreLife": (3.0, 78.0),
    "FuelEffect": (0.055, 4.125),
    "Prev_LapTime": (67.719, 148.991),
    "Prev_TyreLife": (2.0, 77.0),
    "Prev_SpeedST": (156.0, 362.0),
    "AirTemp": (14.5, 33.7),
    "TrackTemp": (16.7, 50.7),
    "Humidity": (18.0, 92.0),
    "laps_remaining": (0.0, 75.0),
    "mean_sector_speed": (196.6292354740061, 314.9705586672411),
}

_N06_ENVELOPE = OperatingEnvelope(name="n06_laptime_delta", bounds=_N06_TRAINED_BOUNDS)


# The seasons that have a per-year featured artefact. Listed rather than globbed so a
# stray file cannot silently widen what the agent claims to know.
_FEATURED_SEASONS: tuple[int, ...] = (2023, 2024, 2025)

# N01's compound codes (`.nb_py/N01_data_download.py:114`), which is what the parquet's
# `CompoundID` column holds and therefore what N06 was trained on. Not the manifest's
# 0-based `categorical_encoding.Compound` block: that one encodes a column N06 drops.
#
# 0 is a TRAINED CLASS, not a spare slot: N01 does `.fillna(0)`, so every lap whose
# compound FastF1 did not report reached training as a 0. That is why an off-by-one here
# is worse than a shift: it makes a SOFT lap indistinguishable from an unreported one.
# `tests/agents/test_pace_compound_encoding.py` re-derives all of this from the artefact.
_N01_COMPOUND_ID: dict[str, int] = {
    "SOFT": 1,
    "MEDIUM": 2,
    "HARD": 3,
    "INTERMEDIATE": 4,
    "WET": 5,
}
_COMPOUND_ID_UNKNOWN: int = 0


# Moved to `gp_slugs` by the 2026-08-04 keyspace sweep: five consumers across three
# modules needed the same normalisation, and it was never pace-specific. Re-exported under
# the old private name so the #797 call sites and their tests do not move.
_normalise_gp_key = normalise_gp_key


# ─────────────────────────────────────────────────────────────────────────────
# PaceOutput dataclass (public API, untouched)
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class PaceOutput:
    """Structured output of the Pace Agent for one lap.

    lap_time_pred is the N06 XGBoost prediction in absolute seconds. The model
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

    lap_time_pred: float
    delta_vs_prev: float
    delta_vs_median: float
    ci_p10: float
    ci_p90: float
    reasoning: str = ""


# ─────────────────────────────────────────────────────────────────────────────
# PaceAgent class
# ─────────────────────────────────────────────────────────────────────────────


def _unknown_if_missing(tyre_life) -> int:
    """A tyre age, with an absent one encoded as the shared unknown.

    `or` cannot be used for this and that is the whole point: it is false for 0,
    which is the value `race_state_builder` publishes when it does not know the
    age, so `d.get("tyre_life") or 1` rewrote every unknown as a one-lap-old set.
    A missing key and a stored None both mean the same thing and both land on the
    same constant, which is what keeps this reader and the builder from drifting.
    """
    return UNKNOWN_TYRE_LIFE if tyre_life is None else int(tyre_life)


def _laps_since_pit(driver_state: dict) -> int:
    """Laps since the last stop, falling back to the tyre age under N01's rule.

    Written as branches rather than as an `or` chain, and the chain is why: it
    ended in `or 1`, so an unknown age fell straight through to a one-lap-old
    stint. That is the same fabrication `_unknown_if_missing` exists to stop, one
    keyword argument further down, and the first version of this fix kept it.

    The fallback to the age is N01's own definition on an unpitted stint, so it is
    a lookup rather than an approximation (#800). When neither is known the answer
    is the shared unknown: N06 predicts identically at 0, 1 and 2 on a real row
    (86.712000 against 86.654000 at 12), so nothing served moves, and the value
    that does not collide is the one to publish.
    """
    laps = driver_state.get("laps_since_pit")
    if laps is not None:
        return int(laps)
    return _unknown_if_missing(driver_state.get("tyre_life"))


def _previous_tyre_life(tyre_life: Optional[int]) -> Optional[int]:
    """The previous lap's tyre age, or None where training had no previous lap.

    N04 builds ``Prev_TyreLife`` as a stint-grouped shift, so a stint opener is
    NaN. Inference used to send ``max(0, tyre_life - 1)`` unconditionally, which
    made an opener 0: below the trained minimum of 2.0, and a NUMBER where
    training had an ABSENCE. XGBoost has a learned default direction for missing;
    it has none for a tyre younger than any it ever saw.
    """
    if tyre_life is None or tyre_life <= 1:
        return None
    return tyre_life - 1


class PaceAgent:
    """Encapsulates the N06 XGBoost lap-time prediction pipeline.

    All model artifacts (XGBoost weights, encoding maps, reference laps) are
    loaded once in __init__ and stored as instance attributes: no module-level
    globals are used.

    Deliberately deterministic, unlike its tire/pit/race_situation siblings:
    pace has no qualitative judgment to make (no warning_level/action/threat_level
    category alongside its numbers), so there is no LLM step to wire. See #778/#780
    for the archaeology and decision record.

    Instantiate via the module-level _get_default_pace_agent() factory to avoid
    redundant disk I/O; do not instantiate PaceAgent directly in hot paths.

    Args:
        models_dir: Directory containing xgb_laptime_delta_final.json and
            the feature name JSON. Defaults to the repo-root-relative path.
        processed_dir: Directory containing circuit clusters, laps_featured,
            and feature manifest. Defaults to the repo-root-relative path.
    """

    def __init__(
        self,
        models_dir: Path = _MODELS_DIR,
        processed_dir: Path = _PROCESSED,
    ) -> None:
        self.model, self.features = self._load_model(models_dir)
        self.compound_id: dict = {}
        self.circuit_cluster: dict = {}
        self.team_id: dict = {}
        self.compound_id, self.circuit_cluster, self.team_id = self._load_encoding_maps(
            processed_dir
        )
        self.circuit_mean_sector_speed: dict[tuple[int, str], float] = (
            self._load_circuit_mean_sector_speed(processed_dir)
        )
        self.laps_ref: pd.DataFrame = self._load_reference_laps(processed_dir)
        # Cached because `_session_median` resolves the caller's spelling against it on
        # every lap, and a `.unique()` over the whole reference frame is not a per-lap cost.
        self._reference_gp_names: frozenset[str] = frozenset(
            self.laps_ref["GP_Name"].dropna().astype(str).unique()
        )

    # ── Loaders ───────────────────────────────────────────────────────────────

    def _load_model(self, models_dir: Path) -> tuple[xgb.XGBRegressor, list[str]]:
        """Load N06 XGBoost model and ordered feature name list from disk.

        Both artifacts are returned together to guarantee the feature order is
        always consistent with the model version. Callers must not reorder or
        drop features between load and predict.

        Args:
            models_dir: Directory containing the two JSON export files.

        Returns:
            Tuple (model, features) where model is a fitted XGBRegressor and
            features is a list of column name strings in predict order.
        """
        features = json.loads((models_dir / "xgb_laptime_delta_feature_names.json").read_text())
        model = xgb.XGBRegressor()
        model.load_model(models_dir / "xgb_laptime_delta_final.json")
        return model, features

    def _load_encoding_maps(self, processed_dir: Path) -> tuple[dict, dict, dict]:
        """Load compound, circuit-cluster, and team label-encoding maps.

        The compound codes are N01's, the circuit clusters come from the k=4 clustering
        parquet (N05), and the team map is derived from the laps_featured parquet. All
        three are static training artifacts: they must not be recomputed at inference
        time to avoid encoding drift between train and serve.

        NOT the manifest's `categorical_encoding.Compound` block, which is what this used
        to read. That block is 0-based and describes N06's `encode_features` step, whose
        output column (`Compound`) is NOT among the 39 in `features_in`: the model eats
        `CompoundID`, N01's 1-based column, straight from the parquet. The eval harness
        says so in its own docstring ("the model consumes the pre-numeric CompoundID"),
        and inference contradicted it, so every lap arrived one class low.

        Args:
            processed_dir: Root of the processed data directory.

        Returns:
            Tuple (compound_id, circuit_cluster, team_id) dicts.
        """
        compound_id = dict(_N01_COMPOUND_ID)

        clusters_df = pd.read_parquet(
            processed_dir / "circuit_clustering" / "circuit_clusters_k4_2025.parquet",
            columns=["GP_Name", "Cluster"],
        )
        circuit_cluster = dict(zip(clusters_df["GP_Name"], clusters_df["Cluster"].astype(int)))

        laps = pd.read_parquet(
            processed_dir / "laps_featured_2025.parquet",
            columns=["Team", "TeamID"],
        ).dropna()
        team_id = laps.drop_duplicates("Team").set_index("Team")["TeamID"].astype(int).to_dict()
        return compound_id, circuit_cluster, team_id

    @staticmethod
    def _load_circuit_mean_sector_speed(processed_dir: Path) -> dict[tuple[int, str], float]:
        """``mean_sector_speed`` per ``(Year, GP_Name)``, from the combined featured parquet.

        This feature is a property of the CIRCUIT, not of the lap: the parquet holds
        exactly one distinct value per (year, GP), the mean of the three speed traps.
        N06 was trained on it and inference never had it, because `_compute_derived`
        substituted `prev_speed_st` and nothing ever supplied the real thing (#797).

        KEYED BY YEAR, because the value is not constant across seasons and the replay
        engine can replay any of the three. Across the GPs present in both eras the
        training-era and 2025 measurements differ on every one: mean absolute gap 4.8 km/h,
        largest Silverstone at 18.3. An earlier draft read only the 2025 artefact and
        served that value for every replay, feeding a 2023 Silverstone lap a measurement
        taken two years after it.

        Beware the mechanism, which is NOT what an earlier draft of this docstring said:
        the value is recomputed per ARTEFACT GENERATION, not per season. 2023 and 2024 are
        identical for every GP they share (max difference exactly 0.0), because one build
        pooled both training seasons and a later build produced 2025. Anyone "completing"
        a per-season resolution between 2023 and 2024 would find nothing to resolve.

        THE PER-YEAR FILES, and NOT the combined `laps_featured.parquet`, which is the
        trap that cost this lookup two rounds. The combined artefact BROADCASTS the
        training-era value across all three seasons: its Silverstone row reads 249.71 for
        2023 and for 2025 alike, and across every GP present in both eras the 2023-vs-2025
        difference is exactly 0.0. Reading it therefore looks year-aware and is not.

        The raw laps settle which artefact is telling the truth. Silverstone 2025's own
        speed traps average 232.32 km/h: the per-year file says 231.36 and the combined
        file says 249.71, the 2023 number. Melbourne 2025 is the same story, 252.93 raw
        against 256.84 per-year and 272.44 combined. So the per-year artefacts carry the
        season's own measurement and the combined one does not.

        The cost of that correctness is a hole rather than a wrong number:
        `laps_featured_2025.parquet` has NaN on all 760 Las Vegas rows, so that race
        resolves to NaN. That is the right failure. The alternative is serving the
        2023-2024 measurement for a 2025 lap, which is the same class of defect as the
        speed-trap substitution this whole fix exists to remove, only quieter.

        It follows that the ENVELOPE bound stays the 2023-2024 range while a 2025 lap is
        served a 2025 measurement, and that is the right way round rather than an oversight:
        the bound asks whether N06 was FITTED on inputs like this one, so a 2025 circuit
        outside the fitted range is genuine extrapolation and should be said out loud. Monza
        2025 at 317.24 against a fitted maximum of 314.97 is exactly that, and the only one.
        """
        by_race: dict[tuple[int, str], float] = {}
        for year in _FEATURED_SEASONS:
            parquet = processed_dir / f"laps_featured_{year}.parquet"
            if not parquet.exists():
                continue
            speeds = pd.read_parquet(parquet, columns=["GP_Name", "mean_sector_speed"]).dropna()
            for row in speeds.drop_duplicates("GP_Name").itertuples():
                by_race[(year, _normalise_gp_key(str(row.GP_Name)))] = float(row.mean_sector_speed)
        return by_race

    def _resolve_mean_sector_speed(self, gp_name: str, year: int) -> float:
        """This race's trained mean sector speed, or NaN when it does not resolve.

        NaN, never `prev_speed_st`, and that substitution is the whole bug: the speed
        trap is a different physical quantity (training means 256.8 vs 303.0 km/h), so
        feeding it does not degrade the prediction gracefully, it silently answers a
        different question. XGBoost handles a genuinely missing feature natively through
        its sparse-aware split direction, which is what "we do not know this circuit"
        should look like. Same rule as `FuelEffect` (#446) and `Position` (#628): unknown
        data stays unknown and never becomes a number the model can mistake for a reading.

        FOUR KEYSPACES, because a GP is named four different ways in this project and an
        earlier draft resolved only two of them. The parquet slug ('Miami'), the raw
        folder ('Miami_Gardens'), the metadata.json name the replay engine actually puts
        into `session_meta` ('Miami Gardens', with a SPACE, which neither of the other two
        forms matches), and the FastF1 event name ('Qatar Grand Prix'). Resolving only the
        first and last sent every lap of the 2025 Miami and 2023 Spanish races to NaN while
        their value sat in the map. That is the #448/#450 dual-keyspace trap for the third
        time, and this time the enumeration is checked rather than assumed: all 70 races
        under `data/raw/` resolve through the chain below, asserted in
        `tests/agents/test_pace_circuit_speed.py`.
        """
        if gp_name:
            for candidate in (gp_name, slug_from_event_name(gp_name) or ""):
                key = (year, _normalise_gp_key(candidate))
                if key in self.circuit_mean_sector_speed:
                    return self.circuit_mean_sector_speed[key]

        logger.warning(
            "no trained mean sector speed for %r in %s; N06 reads the feature as missing "
            "rather than being fed the speed trap in its place (#797)",
            gp_name,
            year,
        )
        return float("nan")

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
            processed_dir / "laps_featured_2025.parquet",
            columns=["GP_Name", "Year", "Compound", "LapTime_s"],
        )

    # ── Encoding helpers ──────────────────────────────────────────────────────

    def _encode_categorical(self, compound: str, team: str, gp_name: str) -> tuple[int, int, int]:
        """Map compound, team, and circuit to their integer label encodings.

        Unknown categories degrade gracefully to the most common training
        value rather than raising an error: compound→1, team→0, cluster→1.

        Args:
            compound: Pirelli compound string ('SOFT', 'MEDIUM', 'HARD', etc.).
            team: Team name matching self.team_id keys (e.g. 'McLaren').
            gp_name: GP name matching self.circuit_cluster keys.

        Returns:
            Tuple (compound_id_int, team_id_int, cluster_int).
        """
        # The unknown code, not 1: 1 is SOFT on the trained scale, so an unrecognised
        # compound string must resolve to the absent reading N01 encoded it as, not to
        # a specific tyre.
        c_id = self.compound_id.get(compound, _COMPOUND_ID_UNKNOWN)
        t_id = self.team_id.get(team, 0)
        # `circuit_cluster` is keyed by the parquet slug; the replay path queries with the
        # metadata name. Miami 2025 missed and took the default, which happens to be its
        # real cluster - a coincidence, not correctness (PR3_GP_KEYSPACE_SWEEP.md).
        cluster = self.circuit_cluster.get(resolve_gp_key(self.circuit_cluster, gp_name), 1)
        return c_id, t_id, cluster

    def _compute_derived(
        self,
        tyre_life: int,
        fuel_load: float,
        lap_number: int,
        total_laps: int,
        mean_sector_speed: float,
        stint_baseline_tyre_life: Optional[int] = None,
        fresh_tyre: Optional[bool] = None,
    ) -> dict:
        """Compute features derived from raw inputs that are not in the source data.

        FreshTyre: FastF1's "the fitted set was NEW" flag, TRUE for every lap of
        that stint. It is NOT a first-lap flag, and this docstring said it was:
        naming the wrong mechanism is how a proxy survives review, because the
        code then matches its own description (W-F3).
        FuelEffect: cumulative fuel burn pace gain (lighter car = faster lap).
        laps_remaining: inverted lap count used as a proxy for race phase.
        mean_sector_speed: passed through untouched, already resolved by the caller.
        It used to fall back to prev_speed_st here, which fed the speed trap in place
        of a circuit mean on every RaceStateManager call (#797). Resolution and its
        NaN-on-unknown rule now live in `_resolve_mean_sector_speed`, so this function
        has no opinion left about a value it cannot look up.

        Args:
            tyre_life: Current laps on this tyre set.
            fuel_load: Estimated fuel fraction in [0, 1].
            lap_number: Current race lap.
            total_laps: Total scheduled race laps.
            mean_sector_speed: This circuit's trained mean sector speed, already
                resolved; NaN when the GP does not resolve.
            stint_baseline_tyre_life: TyreLife recorded at the start of the
                current stint. None means the caller has not been updated to
                supply it, in which case FuelEffect is forced to NaN instead
                of falling back to the old (wrong) formula. See the comment
                in the body for why NaN is the safe default here (#446).

        Returns:
            Dict with keys FreshTyre, FuelEffect, laps_remaining,
            mean_sector_speed.
        """
        # FuelEffect is the cumulative time recovered since the stint started, in
        # SECONDS (training range 0..3.685 s). It was being computed as
        # `fuel_load * 0.03` (a [0, 0.03] value, ~100x too small and semantically
        # different), so the model always read "fresh-fuel stint start" and the learned
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
                "stint_baseline_tyre_life absent from lap_state: FuelEffect=NaN for this "
                "lap; the producer must supply it (#446)"
            )
            fuel_effect = float("nan")
        else:
            fuel_effect = (tyre_life - stint_baseline_tyre_life) * FUEL_GAIN_PER_LAP_S

        return {
            # The FITTED-SET flag, which is what N06 trained on, and not a first-lap
            # flag. FastF1's `FreshTyre` says the set was NEW when it went on and stays
            # True for EVERY lap of that stint; `int(tyre_life <= 1)` is an outlap flag,
            # so the two agreed only on outlaps and on used-set stints and disagreed on
            # every lap 2 or later of a fresh-set stint, which is most racing laps
            # (W-F3). The state manager has emitted the real flag all along
            # (`race_state_manager.py:438`) and this function computed a proxy beside
            # it. `fresh_tyre` falls back to the old expression only for callers that do
            # not supply it, which is the direct `run()` path and its tests.
            "FreshTyre": int(fresh_tyre) if fresh_tyre is not None else int(tyre_life <= 1),
            "FuelEffect": fuel_effect,
            "laps_remaining": max(0, total_laps - lap_number),
            "mean_sector_speed": mean_sector_speed,
        }

    def _build_feature_row(
        self,
        driver_number: Optional[int],
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
        prev_tyre_life: Optional[int],
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
        fresh_tyre: Optional[bool] = None,
    ) -> pd.DataFrame:
        """Pack raw race state into a single-row DataFrame ready for predict().

        Encodes categorical inputs using self.*_id maps, appends derived
        features from _compute_derived(), and selects columns in the exact
        order self.features expects.

        Returns:
            Single-row pd.DataFrame with columns in self.features order.
        """
        c_id, t_id, cluster = self._encode_categorical(compound, team, gp_name)
        # An explicit argument still wins, so a caller that genuinely measured this lap's
        # mean sector speed is not overridden by the circuit constant. Nothing in the
        # repo passes one today; the RaceStateManager path is exactly the caller that
        # left it None and got the speed trap for it (#797).
        resolved_mean_sector_speed = (
            mean_sector_speed
            if mean_sector_speed is not None
            else self._resolve_mean_sector_speed(gp_name, year)
        )
        derived = self._compute_derived(
            tyre_life,
            fuel_load,
            lap_number,
            total_laps,
            resolved_mean_sector_speed,
            stint_baseline_tyre_life,
            fresh_tyre,
        )

        row = {
            "DriverNumber": driver_number,
            "LapNumber": lap_number,
            "Stint": stint,
            "TyreLife": tyre_life,
            "FreshTyre": derived["FreshTyre"],
            "Position": position,
            "CompoundID": c_id,
            "TeamID": t_id,
            "LapsSincePitStop": laps_since_pit,
            "FuelLoad": fuel_load,
            "Year": year,
            "FuelEffect": derived["FuelEffect"],
            "Prev_LapTime": prev_lap_time,
            "Prev_TyreLife": prev_tyre_life,
            "Prev_SpeedST": prev_speed_st,
            "AirTemp": air_temp,
            "TrackTemp": track_temp,
            "Humidity": humidity,
            "Rainfall": rainfall,
            "laps_remaining": derived["laps_remaining"],
            "Cluster": cluster,
            "mean_sector_speed": derived["mean_sector_speed"],
            "Prev_DegradationRate": prev_deg_rate,
            "Prev_CumulativeDeg": prev_cum_deg,
            "Prev_DegAcceleration": prev_deg_accel,
        }
        df = pd.DataFrame([row])[self.features]
        # Belt-and-braces: any caller that slips a None through lands here.
        # XGBoost refuses object-dtype columns; ``to_numeric(errors='coerce')``
        # converts None→NaN and the model handles NaN natively via its
        # sparse-aware split logic (default_left). Cheap and defensive:
        # no-op on already-numeric frames.
        numeric = df.apply(pd.to_numeric, errors="coerce")
        self._label_against_envelope(numeric)
        return numeric

    @staticmethod
    def _label_against_envelope(feature_df: pd.DataFrame) -> None:
        """Say out loud when N06 is being asked to predict outside its trained range.

        LABELS ONLY, and that is the contract, not a shortcut: nothing here clips,
        refuses or alters a single value the model is fed, so the frame returned by
        `_build_feature_row` is byte-identical to the one returned before this
        existed and the strategy goldens cannot move (#710).

        Only ``violations`` are reported, never ``unknown``. A feature that arrived
        as NaN was not given a bad value, it was given no value, and the two must
        stay distinguishable: the places that can produce a NaN here already warn
        for themselves (`_compute_derived` on an absent stint baseline, and the
        deliberate None-propagation of an unknown Position), so folding them in
        would double-report the known cases and drown the one this exists to catch.
        """
        verdict = _N06_ENVELOPE.check(feature_df.iloc[0].to_dict())
        if verdict.violations:
            logger.warning(
                "N06 called outside its trained range on %d feature(s): %s; the "
                "prediction is an extrapolation, not a fit",
                len(verdict.violations),
                dict(verdict.violations),
            )

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
        prev = float(feature_df["Prev_LapTime"].iloc[0])
        return prev + delta

    def _bootstrap_ci(
        self,
        feature_df: pd.DataFrame,
        n: int = N_BOOTSTRAP,
        seed: int = 42,
    ) -> tuple[float, float]:
        """Estimate a P10/P90 confidence interval via Gaussian feature perturbation.

        Perturbs the row n times with independent Gaussian noise
        (sigma = NOISE_PCT × feature_value) on the continuous features most
        subject to real-world variability, then scores all n rows in one forward
        pass. The noise scale approximates sensor noise and lap-to-lap
        variation; it is not formal Bayesian uncertainty.

        N31 uses this interval to sample pace scenarios in Monte Carlo strategy
        evaluation. A wider interval increases the variance of the strategy
        score distribution and makes the agent more conservative.

        Args:
            feature_df: Single-row DataFrame from _build_feature_row().
            n: Number of bootstrap samples (default N_BOOTSTRAP = 200).
            seed: Integer seed for reproducibility.

        Returns:
            Tuple (p10, p90) of absolute lap times in seconds.
        """
        noise_cols = [
            "Prev_LapTime",
            "Prev_SpeedST",
            "mean_sector_speed",
            "AirTemp",
            "TrackTemp",
            "TyreLife",
        ]

        rng = np.random.default_rng(seed)
        base = feature_df.values.copy().astype(float)
        col_idx = [feature_df.columns.get_loc(c) for c in noise_cols]

        # One perturbed block and one predict() call, where this used to run n
        # single-row calls. The samples are the same numbers, not merely close
        # ones: numpy computes normal(0, sigma) as sigma * standard_normal(), and
        # a C-order (n, len(noise_cols)) block draws sample-major then column,
        # which is the order the per-sample loop drew in. Batching matters here
        # because XGBoost pays a fixed pandas-to-native conversion per call, and
        # at n=200 that conversion was 61% of an offline lap.
        sigmas = np.abs(base[0, col_idx]) * _NOISE_PCT
        block = np.repeat(base, n, axis=0)
        block[:, col_idx] += rng.normal(0.0, 1.0, size=(n, len(col_idx))) * sigmas

        perturbed = pd.DataFrame(block, columns=feature_df.columns)
        preds = perturbed["Prev_LapTime"].to_numpy() + self.model.predict(perturbed)

        return float(np.percentile(preds, 10)), float(np.percentile(preds, 90))

    def _session_median(self, gp_name: str, year: int, compound: str) -> Optional[float]:
        """Return the historical median lap time for a GP / year / compound.

        Filters self.laps_ref to the matching GP, year, and compound, then
        returns the median of LapTime_s. N31 uses this value to contextualise
        the absolute predicted lap time. A large positive delta_vs_median
        signals a degrading tyre or a slower compound choice.

        Args:
            gp_name: GP name matching the GP_Name column.
            year: Race year integer.
            compound: Pirelli compound string.

        Returns:
            Median lap time in seconds, or None when no matching laps exist.
        """
        # The parquet spells this race 'Miami'; the replay path asks for 'Miami Gardens'.
        # Unresolved, the mask was empty for the whole race and N31 lost delta_vs_median
        # on every lap (PR3_GP_KEYSPACE_SWEEP.md).
        stored_name = resolve_gp_key(self._reference_gp_names, gp_name)
        mask = (
            (self.laps_ref["GP_Name"] == stored_name)
            & (self.laps_ref["Year"] == year)
            & (self.laps_ref["Compound"] == compound)
        )
        subset = self.laps_ref.loc[mask, "LapTime_s"].dropna()
        return float(subset.median()) if len(subset) > 0 else None

    # ── Main inference entrypoint ─────────────────────────────────────────────

    def run(
        self,
        driver_number: Optional[int],
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
        prev_tyre_life: Optional[int],
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
        fresh_tyre: Optional[bool] = None,
    ) -> PaceOutput:
        """Run pace prediction for a single lap and return a PaceOutput.

        Builds the N06 feature vector, calls the XGBoost model, computes a
        bootstrap P10/P90 uncertainty interval, and looks up the historical
        session median for the current GP/year/compound.

        Args:
            driver_number: Car number. A RAW feature the model splits on, not a
                lookup key for anything - the TeamID encoding this line used to
                claim comes from `team`. None when the source has no reading, and
                it must stay None: 0 is not absent, it is a value the model finds,
                sorting below every real car number (#831).
            lap_number: Current race lap; used for FuelLoad estimation.
            stint: Stint number (1-indexed), forwarded as a raw feature.
            tyre_life: Laps on current tyre set. It no longer drives FreshTyre on
                the `from_state` path - the state manager emits FastF1's real
                set-was-new flag and `tyre_life <= 1` was only ever a proxy for it,
                agreeing on outlaps and disagreeing on every lap 2+ of a fresh-set
                stint (#831). The fallback still uses it when the flag is absent.
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
            prev_tyre_life: TyreLife on the previous lap, or None on a stint
                opener, where the trained column is NaN by construction.
            prev_speed_st: Speed trap reading in km/h from the previous lap.
            air_temp: Air temperature in °C.
            track_temp: Track surface temperature in °C.
            humidity: Relative humidity in %.
            rainfall: True if rain was recorded during this lap.
            total_laps: Total scheduled race laps.
            gp_name: GP name matching self.circuit_cluster keys.
            mean_sector_speed: This lap's measured mean sector speed if the caller
                has one; when None it is resolved per (year, GP) from the featured
                artefact, NaN when unknown (#797). Never defaulted to prev_speed_st.
            prev_deg_rate: Degradation rate from the previous lap (s/lap).
            prev_cum_deg: Cumulative degradation at the previous lap.
            prev_deg_accel: Second derivative of degradation (s/lap²).
            stint_baseline_tyre_life: TyreLife at the start of the current
                stint. None forces FuelEffect to NaN rather than a fabricated
                estimate. See _compute_derived.
            fresh_tyre: FastF1's set-was-new flag; None falls back to the
                `tyre_life <= 1` proxy.

        Returns:
            PaceOutput with all fields populated and a reasoning string.
        """
        feature_df = self._build_feature_row(
            driver_number=driver_number,
            lap_number=lap_number,
            stint=stint,
            tyre_life=tyre_life,
            compound=compound,
            position=position,
            team=team,
            laps_since_pit=laps_since_pit,
            fuel_load=fuel_load,
            year=year,
            prev_lap_time=prev_lap_time,
            prev_tyre_life=prev_tyre_life,
            prev_speed_st=prev_speed_st,
            air_temp=air_temp,
            track_temp=track_temp,
            humidity=humidity,
            rainfall=rainfall,
            total_laps=total_laps,
            gp_name=gp_name,
            mean_sector_speed=mean_sector_speed,
            prev_deg_rate=prev_deg_rate,
            prev_cum_deg=prev_cum_deg,
            prev_deg_accel=prev_deg_accel,
            stint_baseline_tyre_life=stint_baseline_tyre_life,
        )

        lap_time_pred = self._predict(feature_df)
        delta_vs_prev = lap_time_pred - prev_lap_time
        p10, p90 = self._bootstrap_ci(feature_df)
        median = self._session_median(gp_name, year, compound)
        delta_vs_median = (lap_time_pred - median) if median is not None else float("nan")

        trend = "faster" if delta_vs_prev < 0 else "slower"
        vs_med = (
            f"{delta_vs_median:+.3f}s vs median" if median is not None else "no median reference"
        )
        reasoning = (
            f"Lap {lap_number}: predicted {round(lap_time_pred, 3):.3f}s "
            f"({round(delta_vs_prev, 3):+.3f}s, {trend} than prev). "
            f"CI [{round(p10, 1):.1f}-{round(p90, 1):.1f}s]. {vs_med}."
        )

        return PaceOutput(
            lap_time_pred=round(lap_time_pred, 3),
            delta_vs_prev=round(delta_vs_prev, 3),
            delta_vs_median=round(delta_vs_median, 3),
            ci_p10=round(p10, 3),
            ci_p90=round(p90, 3),
            reasoning=reasoning,
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
        d = lap_state["driver"]
        meta = lap_state["session_meta"]
        wx = lap_state.get("weather", {})

        lap_number = lap_state["lap_number"]
        total_laps = meta["total_laps"]
        laps_remaining = max(0, total_laps - lap_number)

        # The PREVIOUS lap's trap, which is what N04 built and N06 ate: every `Prev_*`
        # column is one grouped shift within the stint. This used to read `speed_st`,
        # THIS lap's trap, so the feature was a lap ahead of itself on every call: the
        # exact defect #435 fixed for `Prev_LapTime` and left in place for its sibling.
        #
        # No `or 300.0`. That default was not a neutral filler: 300 km/h sits inside the
        # trained range (156-362), so an invented reading was indistinguishable from a
        # measured one, and it fired on the first lap of every stint, where the answer is
        # genuinely unknown. NaN says unknown, and XGBoost reads a missing feature
        # natively through its sparse-aware split direction.
        #
        # `.get` alone would return a stored None; `_build_feature_row`'s
        # `to_numeric(errors='coerce')` turns that into the NaN we want, but converting
        # here keeps the value numeric all the way down, which is what the envelope
        # labelling and the bootstrap both assume.
        _prev_speed_st = d.get("prev_speed_st")
        _prev_speed_st = float("nan") if _prev_speed_st is None else float(_prev_speed_st)
        _air_temp = reading_or_default(wx, "air_temp", DEFAULT_AIR_TEMP_C)
        _trk_temp = reading_or_default(wx, "track_temp", DEFAULT_TRACK_TEMP_C)
        _humidity = reading_or_default(wx, "humidity", 50.0)
        _rainfall = reading_or_default(wx, "rainfall", 0)

        return self.run(
            # The real car number, now that the state manager emits it (W-F2). It used
            # to be `or 0` against a key nobody produced, so every replay lap served 0:
            # a value outside the trained vocabulary (1-81) AND a findable one, since 0
            # sorts below every real number and sends each DriverNumber split down its
            # left branch. `.get` without `or`, because a genuinely missing number must
            # stay None and reach the model as NaN, which is a direction XGBoost
            # learned, rather than as a car that does not exist.
            driver_number=d.get("driver_number"),
            # The real fitted-set flag, which the state manager has emitted all along
            # while this call computed a first-lap proxy from tyre_life instead (W-F3).
            fresh_tyre=d.get("fresh_tyre"),
            lap_number=lap_number,
            stint=d.get("stint") or 1,
            # `.get`, not `or 1`. The state manager already publishes
            # UNKNOWN_TYRE_LIFE for an age it does not have
            # (`race_state_builder.py:379`), and `or` is false for 0, so the `or 1`
            # here turned that unknown back into a fresh set one hop after it was
            # chosen - undoing the sentinel at the only consumer that matters.
            # Same shape as the `position` fix three lines below, which this file
            # already carries the reasoning for (#628), and the same defect #832
            # removed from N15's own reader.
            #
            # Measured on the shipped N06 booster: tyre_life 0, 1 and 2 predict
            # 86.712000 identically on a real Lusail row (12 predicts 86.654000),
            # so passing the unknown through changes no served number today. What
            # it changes is `_previous_tyre_life`, which returns None at <= 1 and
            # so sends Prev_TyreLife to the model as NaN rather than as a
            # fabricated age.
            tyre_life=_unknown_if_missing(d.get("tyre_life")),
            compound=d.get("compound") or "MEDIUM",
            # Plain .get, no `or` fallback: `d.get('position') or 1` collapsed a
            # missing telemetry reading AND the #428 sentinel (a stored `0`)
            # into P1, the race leader, straight into the live N06 XGBoost
            # feature (#628). Unlike race_situation_agent/pit_strategy_agent,
            # this value is never used in a `position - 1` rival lookup here,
            # so there is no lookup to fool. The honest fix is simply to let
            # an unknown position propagate as None. _build_feature_row's
            # existing pd.to_numeric(errors='coerce') turns that into NaN, and
            # XGBoost handles a missing 'Position' natively via its
            # default-left split direction, so no fabricated value is needed.
            position=d.get("position"),
            team=meta.get("team") or "Unknown",
            # The REAL laps-since-pit, not the tyre's age. These are different
            # quantities and coincide only when the set was fitted at the last stop:
            # measured against N06's trained column they agree on 97.7% of laps at
            # Lusail and 34.6% at Melbourne, where two thirds of laps were fed the
            # wrong number. `RaceStateManager.laps_since_pit` reproduces N01's own
            # definition exactly, so this is a lookup rather than an approximation
            # (#800). `or` rather than the two-arg get: a producer that has not been
            # updated reports the key absent, and lap 1 of an unpitted race is 1
            # under N01's rule anyway.
            laps_since_pit=_laps_since_pit(d),
            fuel_load=laps_remaining / max(total_laps, 1),
            year=meta.get("year") or 2025,
            # ``d.get('prev_lap_time') or 90.0``, NOT ``d.get('lap_time_s')``: the
            # latter fed this LAP's own time back in as the PREVIOUS lap's time, so
            # the model chased its own most recent prediction instead of the real
            # preceding lap (#435). RaceStateManager.get_driver_state now emits the
            # real 'prev_lap_time' sourced from the parquet's Prev_LapTime column.
            # ``or``, not the two-arg get(key, default) form: Prev_LapTime is NaN
            # on the first lap of a stint/race (no earlier lap exists), which
            # get_driver_state turns into an explicit ``None`` (present key, None
            # value), and the two-arg form only substitutes when the KEY is
            # absent, never when the VALUE is (the same Series.get trap #428/#446/
            # #462 keep finding). Unlike stint_baseline_tyre_life below, None
            # cannot be allowed through here: _predict() reads Prev_LapTime
            # straight into ``prev + delta`` with no NaN branch, so a bare None
            # would turn lap_time_pred itself into NaN. 90.0 is the same
            # order-of-magnitude placeholder the old (wrong) code used, now only
            # reached on a genuinely missing previous lap.
            prev_lap_time=d.get("prev_lap_time") or MISSING_PREV_LAP_TIME_S,
            # None on a stint opener, where N04's grouped shift leaves NaN, instead of
            # the 0 this used to send (W-F5). 0 is below the trained minimum of 2.0, so
            # the model read a tyre younger than any it was fitted on rather than the
            # absence training taught it to handle; NaN is a direction XGBoost learned.
            # Off a stint opener the value stays current-1, which is the right answer
            # whenever consecutive laps survived N04's filter, and an approximation
            # where they did not.
            prev_tyre_life=_previous_tyre_life(d.get("tyre_life")),
            prev_speed_st=_prev_speed_st,
            air_temp=float(_air_temp),
            track_temp=float(_trk_temp),
            humidity=float(_humidity),
            rainfall=float(_rainfall or 0),
            total_laps=total_laps,
            gp_name=meta.get("gp_name") or "",
            prev_deg_rate=0.0,
            prev_cum_deg=0.0,
            prev_deg_accel=0.0,
            # Plain .get, deliberately NOT the `or` pattern used above: `or` would
            # collapse a legitimate baseline of 0 into None and re-introduce exactly
            # the sentinel-vs-real-value confusion this epic is about. Absent stays
            # absent, and _compute_derived turns that into a NaN plus a warning (#446).
            stint_baseline_tyre_life=d.get("stint_baseline_tyre_life"),
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
# Public entry points (backward-compatible API, same signatures as before)
# ─────────────────────────────────────────────────────────────────────────────


def run_pace_agent(
    driver_number,
    lap_number,
    stint,
    tyre_life,
    compound,
    position,
    team,
    laps_since_pit,
    fuel_load,
    year,
    prev_lap_time,
    prev_tyre_life,
    prev_speed_st,
    air_temp,
    track_temp,
    humidity,
    rainfall,
    total_laps,
    gp_name,
    mean_sector_speed=None,
    prev_deg_rate=0.0,
    prev_cum_deg=0.0,
    prev_deg_accel=0.0,
) -> PaceOutput:
    """Run the Pace Agent for a single lap and return a structured PaceOutput.

    Thin entry point that delegates to the shared PaceAgent singleton. All
    inference logic lives in PaceAgent.run(). See its docstring for full
    parameter documentation.
    """
    return _get_default_pace_agent().run(
        driver_number=driver_number,
        lap_number=lap_number,
        stint=stint,
        tyre_life=tyre_life,
        compound=compound,
        position=position,
        team=team,
        laps_since_pit=laps_since_pit,
        fuel_load=fuel_load,
        year=year,
        prev_lap_time=prev_lap_time,
        prev_tyre_life=prev_tyre_life,
        prev_speed_st=prev_speed_st,
        air_temp=air_temp,
        track_temp=track_temp,
        humidity=humidity,
        rainfall=rainfall,
        total_laps=total_laps,
        gp_name=gp_name,
        mean_sector_speed=mean_sector_speed,
        prev_deg_rate=prev_deg_rate,
        prev_cum_deg=prev_cum_deg,
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
