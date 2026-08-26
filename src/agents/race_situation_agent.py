"""Race Situation Agent: src/agents/race_situation_agent.py

Extracted from N27_race_situation_agent.ipynb. Combines N12 (overtake LightGBM)
and N14 (safety car LightGBM) into a single threat assessment per lap.

Public API
----------
run_race_situation_agent(lap_state)                       -> RaceSituationOutput  (FastF1 session)
run_race_situation_agent_from_state(lap_state, laps_df)   -> RaceSituationOutput  (RSM adapter)

Module-level singletons
-----------------------
CFG           : RaceSituationConfig: both model pairs + calibrators + feature lists.
                Kept at module level so RaceSituationOutput.__post_init__ can read thresholds.
TIRE_COMPOUNDS : authoritative compound allocation from data/tire_compounds_by_race.json.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd

from src.agents._shared_defaults import (
    DEFAULT_AIR_TEMP_C,
    DEFAULT_TOTAL_LAPS,
    DEFAULT_TRACK_TEMP_C,
    reading_or_default,
)

# ── Repo root (with root-stop guard for uv tool install) ─────────────────────
_REPO_ROOT = Path(__file__).resolve().parent
while not (_REPO_ROOT / ".git").exists():
    if _REPO_ROOT.parent == _REPO_ROOT:
        break
    _REPO_ROOT = _REPO_ROOT.parent

# Route every artefact path through the cache helper so the agent works
# transparently in both editable-dev mode (repo ``data/`` folder) and the
# ``uv tool install`` flow (``~/.f1-strat/data/``).
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

from src.f1_strat_manager.gp_slugs import (  # noqa: E402
    rekey_by_slug,
    resolve_gp_key,
    slug_from_event_name,
)

_MODELS = _DATA_ROOT / "models"
_PROCESSED = _DATA_ROOT / "processed"
_AGENTS = _DATA_ROOT / "models" / "agents"


# ── Authoritative compound allocation ─────────────────────────────────────────
_compounds_path = _DATA_ROOT / "tire_compounds_by_race.json"
TIRE_COMPOUNDS: dict = (
    json.loads(_compounds_path.read_text(encoding="utf-8")) if _compounds_path.exists() else {}
)

# ── Feature engineering constants (matching N13/N14 training definitions) ─────
CLIFF_THRESHOLDS = {"SOFT": 20, "MEDIUM": 35, "HARD": 50}
STATUS_ENC = {"1": 0, "2": 1, "5": 2, "7": 3, "6": 4, "4": 5}
STATUS_SEVERITY = {"1": 1, "2": 2, "5": 3, "7": 4, "6": 5, "4": 6}
_INCIDENT_RE = r"INCIDENT|COLLISION|CONTACT|SPIN|OFF TRACK|STOPPED CAR|DEBRIS|MARSHAL"
_EXCLUDE_RE = r"TRACK LIMITS|LAP TIME|PENALTY|PIT LANE|FORMATION|GRID|DRS|SAFETY CAR|VIRTUAL"


# ── RCM context override (post-hoc fix for "SC active but model says low") ────
# A full Safety Car (FIA Sporting Regs Art. 55) and a Virtual Safety Car (Art. 56) are
# different race situations, not one flag (#471): under a full SC the field queues behind
# the car and gaps compress (55.7 / 55.10); under a VSC there is no queue, gaps are
# broadly preserved and the delta binds throughout (56.5). Both ban overtaking
# (55.8 / 56.6), so both force overtake_prob = 0, but the pit-time saving is much smaller
# under a VSC, so the two are tracked apart from here on. Strings match exactly what
# radio_agent._classify_rcm_event() emits. (src/nlp/rcm_state.py mirrors these sets for
# its cross-lap tracker; keep them in sync.)
_SC_DEPLOY_EVENT_TYPES: frozenset[str] = frozenset({"SAFETY_CAR_DEPLOYED"})
_VSC_DEPLOY_EVENT_TYPES: frozenset[str] = frozenset({"VIRTUAL_SAFETY_CAR_DEPLOYED"})

# ── N11's trained domain ──────────────────────────────────────────────────────
# N11's pair builder DROPS every pair more than 2.5 s apart before labelling: "not an
# active battle" (`.nb_py/N11_overtake_eda.py:233-235`). The model therefore has no
# labelled example beyond it, and a probability returned out there is whatever leaf the
# extrapolation lands in, not an estimate.
#
# Measured over all 24 races of 2025 with N11's own pairing rule: 8,816 of 20,449
# position-adjacent pairs (43.1%) sit outside it, median gap 2.06 s, p90 9.11 s. Four in
# ten "can I pass the car ahead?" questions were answered from outside the training set.
_TRAINED_MAX_GAP_S: float = 2.5

# What the tool says instead of a number when it is asked outside that domain. Parsed back
# by `_parse_tool_outputs`, which must map it to None rather than to its 0.0 default: an
# unknown probability and a zero one are different claims, and the second is what the
# regulation asserts under a Safety Car (Art. 55.8).
_OUT_OF_DOMAIN_MARKER: str = "UNKNOWN"

# N11's unknown-circuit code (`.nb_py/N11_overtake_eda.py:210-212`). Distinct from cluster
# 0, which is a real kind of track.
_UNKNOWN_CIRCUIT_CLUSTER: int = -1

# SC release. Art. 55.15's "SAFETY CAR IN THIS LAP" (the real message, which classifies
# to SAFETY_CAR_ENDING, verified on Qatar 2025 L10 and Spain 2025 L60) means the car
# enters the pits at the END of that lap, and Art. 55.8 forbids overtaking until the
# driver passes the Line AFTER it has returned. So the announcement lap is still
# neutralised: _neutralization_from_rcm keeps the flag active for it and it clears the
# lap after. SAFETY_CAR_IN_PIT_LANE is the same case for the rarer "IN THE PIT LANE"
# wording.
_SC_RELEASE_EVENT_TYPES: frozenset[str] = frozenset(
    {
        "SAFETY_CAR_ENDING",
        "SAFETY_CAR_IN_PIT_LANE",
    }
)

# VSC release. Art. 56.7's restart is near-instant (green ~10-15 s after "VSC ENDING"),
# so at lap granularity the neutralisation is over on the announcement lap: this releases
# immediately, unlike the SC.
_VSC_RELEASE_EVENT_TYPES: frozenset[str] = frozenset({"VIRTUAL_SAFETY_CAR_ENDING"})


class Neutralization(str, Enum):
    """Which on-track neutralisation the RCM feed confirms is in force this lap.

    NONE : green-flag racing.
    SC   : full Safety Car (FIA Sporting Regs Art. 55): field queued, large pit saving.
    VSC  : Virtual Safety Car (Art. 56): no queue, gaps preserved, small pit saving.
    """

    NONE = "NONE"
    SC = "SC"
    VSC = "VSC"


# ─────────────────────────────────────────────────────────────────────────────
# RaceSituationConfig
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class RaceSituationConfig:
    """Runtime configuration for the Race Situation Agent.

    Loads both LightGBM model pairs (overtake from N12, SC from N14) plus their
    Platt calibrators and feature lists. Models are loaded with joblib because
    the repo path contains non-ASCII characters that break LightGBM's native
    save_model on Windows.

    Threat-level boundaries map the CALIBRATED probabilities to LOW/MEDIUM/HIGH
    categorical signals for N31. They are pit-wall alert levels (a product
    decision about how alarmed to be), and deliberately NOT the models' classifier
    operating points, which live on the raw scale (see __post_init__ and #665).

    Every band below was set on the served calibrated distribution, measured over
    real 2025 laps with the models loaded, so each one is reachable and its firing
    rate is known. The two models are banded on different terms because they are
    not comparable: N12 is a decent ranker (AUC-PR 0.549) whose calibrated output
    reads as a genuine probability, while N14 is weak (AUC-PR 0.072, lift 1.67x)
    and only means something relative to its own base rate.

    Attributes:
        model_name: LM Studio model identifier for the ReAct agent LLM.
        high_overtake: Calibrated P(overtake) above which threat_level is HIGH.
            0.65 = a strong opportunity in plain probability terms; fires on 1.36%
            of real chaser/ahead pairs (n=8171). The calibrated output tops out at
            0.751, so anything at or above 0.7659 is unreachable by construction.
        medium_overtake: Calibrated P(overtake) above which threat_level is MEDIUM.
            0.40 = a realistic shot; fires on 3.99% of pairs.
        high_sc: Calibrated P(SC within 3 laps) above which threat_level is HIGH.
            0.0864 = TWICE N14's 0.0432 base rate (feature_list_v1.json,
            target_comparison['3-lap']['baseline']); fires on 1.27% of laps
            (n=1420). The base rate is a class prevalence, not a value selected on
            the test set, so anchoring here does not reintroduce the contamination
            hygiene.py found in best_threshold. Retraining N14 moves the baseline:
            move this with it, which test_sc_bands_track_the_models_base_rate
            enforces.
        medium_sc: Calibrated P(SC within 3 laps) above which threat_level is MEDIUM.
            0.0432 = the base rate itself, i.e. "likelier than an average lap";
            fires on 13.59% of laps.
    """

    model_name: str = "gpt-4.1-mini"

    high_overtake: float = 0.65
    medium_overtake: float = 0.40
    high_sc: float = 0.0864
    medium_sc: float = 0.0432

    def __post_init__(self) -> None:
        self.export_dir = _AGENTS
        try:
            self.export_dir.mkdir(parents=True, exist_ok=True)
        except OSError:
            pass  # read-only mount in Docker

        # Overtake model (N12)
        _ov = _MODELS / "overtake_probability"
        self.overtake_model = joblib.load(_ov / "lgbm_overtake_v1.pkl")
        self.overtake_calibrator = joblib.load(_ov / "calibrator.pkl")
        with open(_ov / "model_config.json") as f:
            ov_cfg = json.load(f)
        self.overtake_features: list[str] = ov_cfg["features"]
        self.overtake_cat_features: list[str] = ov_cfg["categorical_features"]
        self.overtake_threshold: float = ov_cfg["optimal_threshold"]

        # SC model (N14)
        _sc = _MODELS / "safety_car_probability"
        self.sc_model = joblib.load(_sc / "lgbm_sc_v1.pkl")
        self.sc_calibrator = joblib.load(_sc / "calibrator_sc_v1.pkl")
        with open(_sc / "feature_list_v1.json") as f:
            sc_cfg = json.load(f)
        self.sc_features: list[str] = sc_cfg["features"]
        self.sc_threshold: float = sc_cfg["best_threshold"]

        # DO NOT assign overtake_threshold/sc_threshold onto high_overtake/high_sc.
        # #450 did exactly that and it is a unit error: both tuned thresholds are
        # operating points on the RAW model output, while threat_level compares the
        # CALIBRATED probability. N14 tunes best_threshold on `proba_test` (cell 20,
        # `m.predict_proba(X_test)[:,1]`) and only calibrates afterwards in cell 32;
        # N12 does the same in cells 22/25/26 vs cell 36. Pushed through each Platt
        # calibrator, raw 0.2335 is calibrated 0.0158 and raw 0.7976 is calibrated
        # 0.4183, and the overtake calibrator's ceiling is 0.7659 at raw 1.0, so
        # comparing a calibrated probability against 0.7976 could never be True.
        # Measured on real 2025 laps: SC HIGH fired 0/1420, overtake HIGH 0/8171.
        #
        # There is a second, independent reason not to wire them: src/strategy/eval/
        # hygiene.py already ruled BOTH thresholds test-contaminated (selected on the
        # 2025 test set), and concluded N14 has no honest validation split for an
        # operating threshold at all: the paper reports SC threshold-free. Promoting
        # a leaked operating point into a runtime signal would put the contamination
        # back into the decisions the paper's numbers describe.
        #
        # The bands below are deliberately NOT the classifier operating points; they
        # are pit-wall alert levels, set on the served calibrated scale (#665).

        # Circuit cluster map (k=4 parquet from N05)
        _cl = pd.read_parquet(_PROCESSED / "circuit_clustering" / "circuit_clusters_k4.parquet")
        self.circuit_cluster_map: dict = dict(zip(_cl["GP_Name"], _cl["Cluster"].astype(int)))

        # Circuit SC base rates (from N13 labeled parquet)
        _sc_df = pd.read_parquet(
            _PROCESSED / "sc_labeled" / "sc_labeled_2023_2025.parquet",
            columns=["event_name", "circuit_sc_rate"],
        )
        # Re-key to the SLUG keyspace the replay path queries with. This table ships
        # keyed by FastF1 event names, which the FastF1 session path supplies but the
        # replay path (CLI / arcade / webapp, all fed by RaceStateManager) does not:
        # it passes session_meta.gp_name, a slug. The keyspaces do not overlap, so on
        # every replay lap this lookup missed and returned the 0.10 default: a trained
        # per-circuit feature frozen to a constant for every race (#448). Lookups go
        # through `sc_rate_for`, which resolves EITHER keyspace, so the FastF1 path
        # keeps working too.
        self.circuit_sc_rate_map: dict = rekey_by_slug(
            _sc_df.drop_duplicates("event_name")
            .set_index("event_name")["circuit_sc_rate"]
            .to_dict(),
            "circuit_sc_rate_map",
        )

    def sc_rate_for(self, event_name: str) -> float:
        """Circuit SC base rate for a GP given in EITHER keyspace.

        The two callers disagree: the FastF1 session path passes a full event name
        ('Qatar Grand Prix'), the replay path passes the parquet slug ('Lusail').
        `slug_from_event_name` is reentrant, so it normalises both. Falls back to the
        0.10 training-set mean when a GP is genuinely unknown, the same default as
        before, but now only for real misses instead of on every replay lap.
        """
        slug = slug_from_event_name(event_name) or event_name
        return self.circuit_sc_rate_map.get(slug, 0.10)

    def cluster_for(self, gp_name: str, default: Optional[int] = None) -> Optional[int]:
        """This circuit's cluster, whichever of its four spellings the caller holds.

        `sc_rate_for` above already resolves ONE keyspace for this same config; the cluster
        map next to it did not, which is the one-copy-fixed-its-twin-not pattern this repo
        keeps producing. Four spellings need both resolvers, not just the slug one
        (PR3_GP_KEYSPACE_SWEEP.md).
        """
        return self.circuit_cluster_map.get(
            resolve_gp_key(self.circuit_cluster_map, gp_name), default
        )


# ── Module-level config singleton ─────────────────────────────────────────────
# Kept at module level because RaceSituationOutput.__post_init__ reads
# CFG.high_overtake, CFG.high_sc, CFG.medium_overtake, CFG.medium_sc.
CFG = RaceSituationConfig()


# ─────────────────────────────────────────────────────────────────────────────
# RaceSituationOutput
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class RaceSituationOutput:
    """Structured output of the Race Situation Agent for one lap snapshot.

    Combines overtaking opportunity assessment (N12) with safety car risk
    prediction (N14) into a single threat_level classification that N31 uses
    to condition pit timing and stint extension decisions.

    threat_level is derived automatically in __post_init__ so downstream agents
    get a categorical signal (LOW/MEDIUM/HIGH) without re-implementing thresholds.

    Attributes:
        overtake_prob: Calibrated P(overtake in next few laps) from N12 LightGBM
            + Platt calibration. Above CFG.high_overtake (0.65) = strong
            opportunity. Compare only against the bands in RaceSituationConfig,
            never against N12's optimal_threshold: that one lives on the raw
            scale and is unreachable here (#665).
        sc_prob_3lap: Calibrated P(SC within 3 laps) from N14 LightGBM + Platt
            calibration. Above CFG.high_sc (0.0864, twice the base rate) =
            elevated SC risk. Same caveat: N14's best_threshold is raw-scale.
        threat_level: LOW / MEDIUM / HIGH derived from both probabilities in __post_init__.
        gap_ahead_s: Gap to the car directly ahead (seconds). < 1.0s = DRS range.
            None when no overtake was scored - leading, no classified car ahead, or
            the tool was never called. There was no pair, so there is no gap: the
            same absence semantics ``overtake_prob`` already carries, and for the
            same reason, because 0.0 is also what a genuinely side-by-side pair
            reports (#878).
        pace_delta_s: SINGLE-lap pace delta vs the car ahead (s/lap), this driver's
            lap time minus theirs, so negative means the driver is faster. The rolling version is
            the separate ``pace_delta_rolling3`` field; this docstring used to describe
            that one, and four surfaces fed a self-delta into this field partly because
            the contract they were reading named the wrong quantity (#750).
        reasoning: LLM synthesis forwarded verbatim to N31 Orchestrator.
        sc_currently_active: Any neutralisation (SC OR VSC) confirmed deployed this lap
            by the RCM feed. Kept as the single back-compat flag every consumer already
            reads; True under both a full SC and a VSC.
        vsc_active: The active neutralisation is specifically a Virtual Safety Car
            (Art. 56). Only meaningful when sc_currently_active is True. Lets consumers
            that care about the pit-time saving (the Monte Carlo, the N28 prompt) tell a
            VSC apart from a full SC, which the single flag could not (#471).
    """

    # Optional because N11 has a domain and this says so. None means the pair sat farther
    # apart than any labelled example the model ever saw (43.1% of real adjacent pairs in
    # 2025), so there is no probability to report, as opposed to 0.0, which under a
    # Safety Car is a REGULATION FACT (Art. 55.8) rather than an absence. Every consumer
    # has to tell those two apart, which is the point.
    overtake_prob: Optional[float]
    sc_prob_3lap: float
    threat_level: str = field(init=False)
    gap_ahead_s: Optional[float] = None
    pace_delta_s: float = 0.0
    reasoning: str = ""
    sc_currently_active: bool = False  # any neutralisation (SC OR VSC) confirmed by RCM this lap
    vsc_active: bool = False  # the active neutralisation is specifically a VSC (Art. 56)

    def __post_init__(self) -> None:
        """Band the threat, treating an unknown overtake probability as no evidence.

        An unknown cannot RAISE the band: the alternative is to keep classifying on the
        extrapolated number, which is how a model's uninformed guess ends up presented to
        N31 as a threat. It cannot lower it either: the SC terms and the live
        neutralisation flag are evaluated exactly as before.
        """
        overtake_known = self.overtake_prob is not None

        if (
            self.sc_currently_active
            or (overtake_known and self.overtake_prob >= CFG.high_overtake)
            or self.sc_prob_3lap >= CFG.high_sc
        ):
            self.threat_level = "HIGH"
        elif (
            overtake_known and self.overtake_prob >= CFG.medium_overtake
        ) or self.sc_prob_3lap >= CFG.medium_sc:
            self.threat_level = "MEDIUM"
        else:
            self.threat_level = "LOW"

    @property
    def sc_active(self) -> bool:
        """True only under a full Safety Car (Art. 55): a neutralisation that is not a VSC.

        The complement of vsc_active within sc_currently_active, so the three read as a
        clean split (green / SC / VSC) without storing a redundant third field.
        """
        return self.sc_currently_active and not self.vsc_active


# ─────────────────────────────────────────────────────────────────────────────
# Pure feature sub-helpers: accept all state as arguments, read no globals
# ─────────────────────────────────────────────────────────────────────────────


def _abs_compound(relative: str, gp_name: str, year: int) -> str:
    """Map SOFT/MEDIUM/HARD → Cx string using TIRE_COMPOUNDS; fallback to input.

    The fifth consumer of this JSON, and the one the earlier keyspace sweeps missed. Its
    failure mode is the loudest of the three: unresolved it returns the RELATIVE name
    ('HARD') where the caller expects a Cx string (PR3_GP_KEYSPACE_SWEEP.md).
    """
    year_data = TIRE_COMPOUNDS.get(str(year), {})
    gp_data = year_data.get(resolve_gp_key(year_data, gp_name), {})
    return gp_data.get(relative.upper(), relative)


def _fmt_prob(value: Optional[float]) -> str:
    """A probability for human text, or the out-of-domain marker when there is none.

    Exists because `f"{value:.2f}"` raises TypeError on None, and every place that renders
    this probability into prose (the RCM override note, the N31 prompt, the dashboards)
    would otherwise have to remember that. One helper beats four remembering.
    """
    return _OUT_OF_DOMAIN_MARKER if value is None else f"{value:.2f}"


def _pair_rolling_features(
    laps_recent: pd.DataFrame,
    driver_x: str,
    driver_y: str,
    lap_number: float,
    gap_ahead_s: float,
    pace_delta_s: float,
) -> tuple[float, float]:
    """N12's ``pace_delta_rolling3`` and ``gap_trend``, over the pair's BATTLE series.

    Both are properties of the battle, not of either car. N12 groups by the pair key,
    sorts by LapNumber, and takes `rolling(3, min_periods=1).mean()` of that pair's
    per-lap `pace_delta_s`, with `gap_trend` as `.diff()` of the pair's gap series
    (`.nb_py/N12_overtake_model.py:141-146`).

    THE SERIES IS THE HARD PART, AND IT TOOK TWO GOES
    ------------------------------------------------
    Inference originally took each DRIVER's last three lap times as two arrays and
    subtracted them by array POSITION, which is a different quantity whenever the two
    windows hold different laps (10.78% of 2025 adjacent pairs). Replacing that with a
    LapNumber pairing fixed the arithmetic and left a second, subtler skew: it rolled over
    every lap where both cars merely have a row.

    N12's rows are the LABELLED pairs, and N11's builder only emits a row when the two cars
    are position-adjacent AND within 2.5 s (`.nb_py/N11_overtake_eda.py:228-235`). So a lap
    where the pair existed but was 4 s apart, or where they had swapped order, is simply
    NOT in the series N12 rolled over. Measured: 29.44% of in-domain 2025 pairs got a
    different window that way, moving the calibrated probability by up to 0.480 and pushing
    81 pairs across the MEDIUM band. It bit hardest on battles that had just closed up:
    exactly the ones the domain gate makes interesting.

    This reconstructs N11's membership rule, not merely a shared-lap window.

    Without a ``Position`` column the membership cannot be reconstructed at all, and the
    honest answer is the one N12 gives for the first row of a series (`min_periods=1`): the
    current lap alone. Guessing a longer history from shared laps is what this docstring
    exists to warn against.
    """
    series = _battle_series(laps_recent, driver_x, driver_y, lap_number)
    if not series:
        return pace_delta_s, 0.0

    laps = sorted(series)
    window = laps[-3:]
    rolling = float(sum(series[lap]["pace_delta"] for lap in window) / len(window))

    # `.diff()` against the previous BATTLE lap, which is not necessarily lap-1: if the
    # pair fell out of the domain for two laps and closed up again, N12's series skips
    # those rows and the diff spans them.
    trend = 0.0
    if len(laps) >= 2:
        current, previous = series[laps[-1]]["gap"], series[laps[-2]]["gap"]
        if pd.notna(current) and pd.notna(previous):
            trend = current - previous
    return rolling, trend


def _battle_series(
    laps_recent: pd.DataFrame, driver_x: str, driver_y: str, lap_number: float
) -> dict[float, dict[str, float]]:
    """The laps N11 would have emitted a labelled row for, keyed by LapNumber.

    N11's membership test, term for term: both cars present, x directly behind y on
    position, gap within the trained bound, and the timing columns actually populated (its
    builder drops the NaN rows before pairing).
    """
    if "Position" not in laps_recent.columns:
        return {}

    both = laps_recent[laps_recent["Driver"].isin([driver_x, driver_y])]
    window = both[both["LapNumber"] <= lap_number]

    series: dict[float, dict[str, float]] = {}
    for lap, rows in window.groupby("LapNumber"):
        x_rows = rows[rows["Driver"] == driver_x]
        y_rows = rows[rows["Driver"] == driver_y]
        if x_rows.empty or y_rows.empty:
            continue
        x, y = x_rows.iloc[0], y_rows.iloc[0]

        if not (pd.notna(x.get("Position")) and pd.notna(y.get("Position"))):
            continue
        if float(x["Position"]) != float(y["Position"]) + 1:
            continue  # not adjacent, or they had swapped order on this lap
        if not (pd.notna(x.get("LapTime")) and pd.notna(y.get("LapTime"))):
            continue

        gap = _pair_gap_seconds(x, y)
        if not pd.notna(gap) or gap > _TRAINED_MAX_GAP_S:
            continue  # N11 dropped it: "not an active battle"

        series[float(lap)] = {
            "pace_delta": float((x["LapTime"] - y["LapTime"]).total_seconds()),
            "gap": gap,
        }
    return series


def _pair_gap_seconds(row_x: pd.Series, row_y: pd.Series) -> float:
    """One battle lap's gap, by N11's rule.

    `abs`, matching `.nb_py/N11_overtake_eda.py:233`, NOT the `max(0.0, ...)` the caller
    applies to the current lap: a clamp turns a negative gap into a fabricated zero, and
    the caller's version is only safe because end-of-lap order guarantees the sign for an
    adjacent pair. Here the adjacency is asserted separately, so the honest absolute is
    both correct and equal to it.
    """
    t_x, t_y = row_x.get("Time"), row_y.get("Time")
    if pd.notna(t_x) and pd.notna(t_y):
        return abs(float((t_x - t_y).total_seconds()))
    return float("nan")


def _lap_count(lap: pd.Series, column: str) -> float:
    """An integer lap count from a lap row, or NaN when the artefact has none.

    `int()` on a NaN raises, and these columns are NOT reliably populated: 44% of the 2025
    Miami rows in `laps_featured_2025.parquet` carry no `TyreLife` at all (492 of 857 after
    augmentation, against 0 at Lusail). That crash never surfaced because the frame handed
    to the agents was the whole season and the (Driver, LapNumber) lookup landed on some
    other race's row; scoping the frame correctly is what exposed it.

    NaN rather than a substitute: LightGBM reads a missing feature natively, and inventing
    a tyre age here is how a sentinel ends up indistinguishable from a real value. The
    absent data itself is an artefact defect, not something inference can repair.
    """
    value = lap.get(column)
    if value is None or pd.isna(value):
        return float("nan")
    return int(value)


def _agg(grp: pd.DataFrame) -> pd.Series:
    """Aggregate lap times for one lap group into mean, std, min scalars."""
    lt = grp["LapTime"].dt.total_seconds().dropna()
    return pd.Series(
        {
            "lt_mean": lt.mean() if not lt.empty else np.nan,
            "lt_std": lt.std(ddof=1) if len(lt) > 1 else 0.0,
            "lt_min": lt.min() if not lt.empty else np.nan,
        }
    )


def _zscore(series: pd.DataFrame, col: str, lap_number: int) -> float:
    """Standardise the value at lap_number against the full causal history."""
    mu = series[col].mean()
    sig = max(float(series[col].std(ddof=1)), 0.01)
    val = series.loc[series.index == lap_number, col]
    return float((val.iloc[0] - mu) / sig) if not val.empty else 0.0


def _is_neutralised(track_status: object) -> bool:
    """True when the lap ran under a Safety Car or a Virtual Safety Car.

    FastF1 packs several codes into one string per lap, so '41' means the lap saw both
    green and SC. Code 4 is the Safety Car and 6 the VSC; a substring test is the right
    read because any appearance means the lap was neutralised for part of its length.

    Used to shut features the regulation shuts. It deliberately does NOT cover the lap
    after the restart, where DRS is still disabled (one lap, two under the 2023 wording)
    but the track status is already green: catching that needs restart-lap state this
    function does not have. Tracked separately.
    """
    if track_status is None or pd.isna(track_status):
        return False
    codes = str(track_status)
    return "4" in codes or "6" in codes


def _dominant_status(grp: pd.DataFrame) -> str:
    """Return the single worst TrackStatus CODE (one character) seen in a lap group.

    FastF1 packs several codes into one string per driver per lap, so a lap that crosses a
    status boundary carries both (e.g. '12' is yellow then clear, '41' Safety Car then
    green). The worst status of the lap therefore lives in an individual character, not in
    a whole row string. Ranking the row strings and returning one intact yields values like
    '12', which is a key neither STATUS_ENC nor STATUS_SEVERITY has, so downstream the lap
    falls to the green default (encoding 0) and a real yellow reads as clear. Joining every
    driver's codes and ranking the characters keeps the result inside the encoding domain.

    Mirrors N13's most_severe_status (notebook N13_sc_eda.ipynb, Step 2 loader cell).
    """
    codes = grp["TrackStatus"].dropna().astype(str).tolist()
    chars = [c for c in "".join(codes) if c in STATUS_SEVERITY]
    if not chars:
        return "1"
    return max(chars, key=lambda c: STATUS_SEVERITY[c])


def _compute_laptime_features(all_laps: pd.DataFrame, lap_number: int) -> dict:
    """Compute lap-time aggregate and z-score features for the current lap window.

    Replicates the N13 aggregate_laps logic, but NOT the N14 training pipeline's
    normalisation: N14 was trained on z-scores computed non-causally over the WHOLE
    race (mean/std of lap_time drawn from every lap, past and future relative to the
    labeled row). This function is deliberately causal instead: it z-scores only
    against laps up to lap_number, because a live agent has no future laps to read.
    That is a genuine train/serve skew (#450); retraining N14 causally is out of
    scope here, so this docstring names the mismatch rather than the previous
    (incorrect) claim that the two "match".

    Args:
        all_laps: Accurate FastF1 laps from race start to current lap.
        lap_number: Current lap number.

    Returns:
        Dict with: lap_time_mean_z, lap_time_std_z, lap_time_min_z,
        lap_time_cv, lap_time_trend_5.
    """
    causal = all_laps[all_laps["LapNumber"] <= lap_number]
    if causal.empty:
        # No prior lap data: return neutral defaults matching the N14 schema.
        # Happens on lap 1 of every replay (no lap has finished yet) and also
        # when the race has been neutralised (all LapTimes NaN under red flag).
        return {
            "lap_time_mean_z": 0.0,
            "lap_time_std_z": 0.0,
            "lap_time_min_z": 0.0,
            "lap_time_cv": 0.0,
            "lap_time_trend_5": 1.0,
        }

    per_lap = causal.groupby("LapNumber").apply(_agg)
    # apply() on an empty-after-filter group can return a DataFrame with no
    # columns at all: guard before dropna to avoid a KeyError on 'lt_mean'.
    if "lt_mean" not in per_lap.columns:
        return {
            "lap_time_mean_z": 0.0,
            "lap_time_std_z": 0.0,
            "lap_time_min_z": 0.0,
            "lap_time_cv": 0.0,
            "lap_time_trend_5": 1.0,
        }
    per_lap = per_lap.dropna(subset=["lt_mean"])

    lt_mean_z = _zscore(per_lap, "lt_mean", lap_number)
    lt_std_z = _zscore(per_lap, "lt_std", lap_number)
    lt_min_z = _zscore(per_lap, "lt_min", lap_number)
    lt_cv = (
        float(per_lap.loc[lap_number, "lt_std"] / max(per_lap.loc[lap_number, "lt_mean"], 1.0))
        if lap_number in per_lap.index
        else 0.0
    )

    lt_means = per_lap["lt_mean"].values
    n = len(lt_means)
    if n >= 5:
        last5 = lt_means[-5:].mean()
        prev5 = lt_means[-10:-5].mean() if n >= 10 else last5
        lt_trend5 = float(last5 / prev5) if prev5 > 0 else 1.0
    else:
        lt_trend5 = 1.0

    return {
        "lap_time_mean_z": lt_mean_z,
        "lap_time_std_z": lt_std_z,
        "lap_time_min_z": lt_min_z,
        "lap_time_cv": lt_cv,
        "lap_time_trend_5": lt_trend5,
    }


def _compute_driver_tyre_features(cur: pd.DataFrame, prev: pd.DataFrame) -> dict:
    """Compute driver count, tyre life, and pit-stop features for the current lap."""
    n_drv = int(cur["Driver"].nunique()) if not cur.empty else 0
    n_drv_prev = int(prev["Driver"].nunique()) if not prev.empty else n_drv
    n_drv_delta = n_drv - n_drv_prev

    tl = cur["TyreLife"].dropna()
    tl_mean = float(tl.mean()) if not tl.empty else np.nan
    tl_max = float(tl.max()) if not tl.empty else np.nan

    high_risk = 0
    for _, r in cur.iterrows():
        cmp = str(r.get("Compound", "")).upper()
        thr = CLIFF_THRESHOLDS.get(cmp, 999)
        try:
            if float(r["TyreLife"]) > thr:
                high_risk += 1
        except (TypeError, ValueError):
            pass

    pit_count = int(cur["PitInTime"].notna().sum()) if "PitInTime" in cur.columns else 0
    outlap = int((cur["TyreLife"] <= 2).sum()) if not cur.empty else 0

    return {
        "n_drivers": n_drv,
        "n_drivers_delta": n_drv_delta,
        "tyre_life_mean": tl_mean,
        "tyre_life_max": tl_max,
        "tyre_age_high_risk_count": high_risk,
        "active_pitstop_count": pit_count,
        "outlap_drivers": outlap,
    }


def _compute_track_status_features(all_laps: pd.DataFrame, lap_number: int) -> dict:
    """Compute track status encoding and yellow-flag escalation features.

    Returns sentinel keys _cur_code, _prev_code (popped by _build_sc_features)
    alongside the actual model features.

    Args:
        all_laps: Full race laps up to lap_number.
        lap_number: Current lap number.

    Returns:
        Dict with model features plus '_cur_code', '_prev_code', '_yel_esc' sentinels.
    """
    causal_laps = all_laps[all_laps["LapNumber"] <= lap_number]
    if causal_laps.empty:
        # No lap data yet: return green-flag defaults. Matches N14's behaviour
        # when the model receives a pre-race or post-red-flag blank state.
        return {
            "_cur_code": "1",
            "_prev_code": "1",
            "_yel_esc": 0,
            "track_status_enc": STATUS_ENC.get("1", 0),
            "status_changed": 0,
            "status_change_direction": 0,
            "yellow_escalation_count": 0,
            "laps_since_last_yellow": 10,
        }

    # Grouped by lap NUMBER across every car, so a driver a lap down inherits the
    # leader's flag. #647 prices that union at 0.84% of driver-laps over 2023-2025
    # and cites Heilmeier et al. 2020, who argue neutralisations belong on race
    # TIME. DO NOT "fix" it here: N13 builds the training features with the same
    # `groupby("LapNumber")` union, so keying this on the clock would feed N14 a
    # distribution it never saw. Inference has to reproduce its notebook, and this
    # repo has already paid for three bugs that were exactly that divergence.
    lap_status = causal_laps.groupby("LapNumber").apply(_dominant_status).sort_index()
    # Pandas quirk: when the grouped object is empty or apply() returns an
    # empty result, groupby().apply() can yield an empty DataFrame (with the
    # full column schema) instead of an empty Series. The early-return above
    # prevents that, but cheap belt-and-braces check in case of edge cases.
    if not isinstance(lap_status, pd.Series) or lap_status.empty:
        return {
            "_cur_code": "1",
            "_prev_code": "1",
            "_yel_esc": 0,
            "track_status_enc": STATUS_ENC.get("1", 0),
            "status_changed": 0,
            "status_change_direction": 0,
            "yellow_escalation_count": 0,
            "laps_since_last_yellow": 10,
        }

    cur_code = str(lap_status.iloc[-1])
    prev_code = str(lap_status.iloc[-2]) if len(lap_status) > 1 else cur_code

    cur_sev = STATUS_SEVERITY.get(cur_code, 1)
    prev_sev = STATUS_SEVERITY.get(prev_code, 1)

    # Force plain int dtype: FastF1 stores TrackStatus as a Categorical, and
    # .map() can preserve that dtype, which then blows up on .fillna(1) because
    # 1 is not in the original category set. Converting via pd.Series(..., dtype=int)
    # strips the Categorical wrapper so shift/fillna behave like plain numerics.
    sev_series = pd.Series(
        [STATUS_SEVERITY.get(str(c), 1) for c in lap_status],
        index=lap_status.index,
        dtype=int,
    )
    escalated = (sev_series > sev_series.shift(1).fillna(1)).astype(int)
    yel_esc = int(escalated.iloc[:-1].tail(3).sum())

    lsl, since = [], 10
    for code in lap_status:
        since = 0 if str(code) != "1" else min(since + 1, 10)
        lsl.append(since)
    laps_since_yellow = int(lsl[-2]) if len(lsl) > 1 else 10

    return {
        "_cur_code": cur_code,
        "_prev_code": prev_code,
        "_yel_esc": yel_esc,
        "track_status_enc": STATUS_ENC.get(cur_code, 0),
        "status_changed": int(cur_code != prev_code),
        "status_change_direction": int(cur_sev > prev_sev) - int(cur_sev < prev_sev),
        "yellow_escalation_count": yel_esc,
        "laps_since_last_yellow": laps_since_yellow,
    }


def _compute_rcm_features(
    all_laps: pd.DataFrame,
    lap_number: int,
    session_meta: dict,
    cur_code: str,
    prev_code: str,
) -> dict:
    """Compute Race Control Message incident and yellow-sector features.

    Mirrors the N13 build_clean_incident_mask logic. Returns zero values when
    no FastF1 session is available in session_meta (replay engine context).

    Args:
        all_laps: Full race laps up to lap_number.
        lap_number: Current lap number.
        session_meta: Must contain 'session' (FastF1 Session) for RCM access.
            When absent (RSM adapter context) all incident features default to 0.
        cur_code: Current lap track-status code.
        prev_code: Previous lap track-status code.

    Returns:
        Dict with: had_incident_msg, incident_escalation, yellow_sectors_this_lap,
        yellow_sectors_prev3, rcm_incident_count_prev3.
    """
    had_inc = inc_esc = ys_cur = ys_prev3 = rcm_prev3 = 0
    _sess = session_meta.get("session")
    if _sess is not None and hasattr(_sess, "race_control_messages"):
        rcm = _sess.race_control_messages.copy()
        if "Lap" not in rcm.columns:
            rcm["Lap"] = np.nan

        _caution = rcm.get("Flag", pd.Series(dtype=str)).isin(["YELLOW", "DOUBLE YELLOW", "RED"])
        _keyword = rcm.get("Message", pd.Series(dtype=str)).str.upper().str.contains(
            _INCIDENT_RE, na=False, regex=True
        ) & ~rcm.get("Message", pd.Series(dtype=str)).str.upper().str.contains(
            _EXCLUDE_RE, na=False, regex=True
        )
        _scope = (
            (rcm["Scope"].str.upper().isin(["TRACK", "SECTOR"]) | rcm["Scope"].isna())
            if "Scope" in rcm.columns
            else pd.Series(True, index=rcm.index)
        )
        clean = (_caution | _keyword) & _scope

        valid = set(all_laps["LapNumber"].dropna().astype(int))
        inc_raw = set(rcm.loc[clean, "Lap"].dropna().astype(int))
        inc_laps = {l for r in inc_raw for l in (r - 1, r, r + 1)} & valid

        had_inc = int(lap_number in inc_laps)
        inc_prev = int((lap_number - 1) in inc_laps)
        inc_esc = inc_prev * int(cur_code != prev_code)

        if "Scope" in rcm.columns and "Flag" in rcm.columns:
            sect_y = rcm[
                rcm["Scope"].str.upper().str.contains("SECTOR", na=False)
                & rcm["Flag"].str.upper().str.contains("YELLOW", na=False)
            ]
            sy_per_lap = sect_y.groupby("Lap").size()
        else:
            sy_per_lap = pd.Series(dtype=int)

        ys_cur = int(sy_per_lap.get(lap_number, 0))
        ys_prev3 = sum(int(sy_per_lap.get(l, 0)) for l in range(max(1, lap_number - 3), lap_number))
        inc_per = rcm.loc[clean].groupby("Lap").size() if clean.any() else pd.Series(dtype=int)
        rcm_prev3 = sum(int(inc_per.get(l, 0)) for l in range(max(1, lap_number - 3), lap_number))

    return {
        "had_incident_msg": had_inc,
        "incident_escalation": inc_esc,
        "yellow_sectors_this_lap": ys_cur,
        "yellow_sectors_prev3": ys_prev3,
        "rcm_incident_count_prev3": rcm_prev3,
    }


def _compute_weather_features(session_meta: dict) -> dict:
    """Extract weather scalars from session_meta for the SC feature vector."""
    # Uses the shared DEFAULT_TRACK_TEMP_C fallback (grep this file for 'TrackTemp' to
    # find the other reads), so a hand-built session_meta missing the key resolves to the
    # same temperature the entry points use, instead of silently disagreeing with itself.
    track_temp = float(session_meta.get("TrackTemp", DEFAULT_TRACK_TEMP_C))
    air_temp = float(session_meta.get("AirTemp", DEFAULT_AIR_TEMP_C))
    humidity = float(session_meta.get("Humidity", 50.0))
    track_temp_start = float(session_meta.get("track_temp_start", track_temp))
    return {
        "track_temp": track_temp,
        "air_temp": air_temp,
        "humidity": humidity,
        "track_temp_delta": track_temp - track_temp_start,
    }


def _ensure_timedelta_laps(laps_df: pd.DataFrame) -> pd.DataFrame:
    """Ensure LapTime column exists as pandas Timedelta, converting from float seconds if needed.

    Feature builders call .dt.total_seconds() on LapTime: this helper normalises
    the column so both FastF1-native DataFrames and replay-engine parquets work.

    Args:
        laps_df: Raw laps DataFrame from any source.

    Returns:
        Copy with LapTime as Timedelta and Sector*Time columns present (NaT if absent).
    """
    df = laps_df.copy()
    if "LapTime" not in df.columns:
        if "LapTime_s" in df.columns:
            df["LapTime"] = pd.to_timedelta(df["LapTime_s"], unit="s")
        else:
            df["LapTime"] = pd.to_timedelta(90.0, unit="s")
    elif not hasattr(df["LapTime"].iloc[0], "total_seconds"):
        df["LapTime"] = pd.to_timedelta(pd.to_numeric(df["LapTime"], errors="coerce"), unit="s")

    # Same normalisation for the session elapsed time, and it is load-bearing:
    # _build_overtake_features reads `Time` to compute the gap the way N11 was
    # trained (N11 cell 13: gap = abs(row_x["Time_s"] - row_y["Time_s"])), and
    # falls back to a single lap's LapTime delta when it is absent. The featured
    # parquet carries `Time_s`, never `Time`, so without this the fallback fired on
    # 100% of calls: at Lusail lap 20 the model was told OCO sat 1.645 s from the
    # leader when the real gap was 33.950 s, which is the difference between "in the
    # DRS window" and "half a minute away". #447 restored the column; this is what
    # lets the agent actually see it.
    if "Time" not in df.columns and "Time_s" in df.columns:
        df["Time"] = pd.to_timedelta(df["Time_s"], unit="s")

    for col in ("Sector1Time", "Sector2Time", "Sector3Time"):
        if col not in df.columns:
            df[col] = pd.NaT

    # Normalise TrackStatus: featured parquet has track_status_clean (int 0/1/2),
    # but feature builders call _dominant_status() which accesses TrackStatus (string).
    #
    # track_status_clean is documented as a 3-class signal (0=green, 1=yellow/VSC,
    # 2=SC/red), but it is measured uniformly 0 across all 204366 rows of every
    # published featured parquet (#615). The cause is structural, not a data bug:
    # N04's IsAccurate gate drops neutralised and pit laps before a lap reaches the
    # featured frame, so any lap that survives into it genuinely is green. A lap
    # carrying this column cannot hold a 1 or a 2, not merely happens not to. A
    # reverse map keyed on those two values would therefore never fire, and keeping
    # it would advertise a 3-class reconstruction this frame cannot deliver. If a
    # lap's real track status is needed (yellow/VSC/SC), it has to come from
    # data/raw/, where the neutralised laps are still present.
    if "TrackStatus" not in df.columns:
        df["TrackStatus"] = "1"

    return df


# ─────────────────────────────────────────────────────────────────────────────
# Stateless output parser
# ─────────────────────────────────────────────────────────────────────────────


def _parse_tool_outputs(messages: list) -> dict:
    """Extract numeric probabilities from ToolMessage strings in the agent history.

    Parses the exact output format of predict_overtake_tool and predict_sc_tool
    rather than the LLM's free-text answer, guaranteeing deterministic values.

    Args:
        messages: LangChain message objects from the agent's invoke result.

    Returns:
        Dict with: overtake_prob, sc_prob_3lap, gap_ahead_s, pace_delta_s.
        ``overtake_prob`` AND ``gap_ahead_s`` are None when the tool declined or was
        never called: no pair was scored, so neither a probability nor a gap exists.
        0.0 is also the gap a genuinely side-by-side pair reports, which is the same
        collision ``overtake_prob`` escaped one paragraph below (#878).
        ``sc_prob_3lap`` and ``pace_delta_s`` keep their 0.0 defaults, unchanged.

    WHY overtake_prob IS THE ONE THAT CAN BE None
    ---------------------------------------------
    0.0 is not a neutral placeholder for it: it is the value the REGULATION asserts under
    a Safety Car (Art. 55.8 bans overtaking), and `_run_core` sets exactly that. Leaving
    the no-answer case on 0.0 made "the rules forbid it" and "the model has no idea"
    the same number to every consumer downstream, which is the sentinel collision this
    repo keeps re-finding. They are different claims and they now look different.
    """
    result: dict[str, Optional[float]] = {
        "overtake_prob": None,
        "sc_prob_3lap": 0.0,
        "gap_ahead_s": None,
        "pace_delta_s": 0.0,
    }
    overtake_taken = False

    for msg in messages:
        content = getattr(msg, "content", "")
        if not isinstance(content, str):
            continue

        # The three overtake fields are read TOGETHER, from the first message that carries
        # an overtake verdict, and then locked. Field-by-field first-match-wins was safe
        # only while every call produced a number: now that a call can DECLINE, a declined
        # first call would leave the probability open while its gap and pace_delta had
        # already been taken, and a second call about a DIFFERENT pair of cars would fill
        # the hole. The reported gap would then describe one battle and the probability
        # another. Only reachable in LLM mode, where the agent may call the tool twice.
        if not overtake_taken and "P(overtake)" in content:
            overtake_taken = True
            # No digits in the out-of-domain string, so this stays None by construction.
            probability = re.search(r"P\(overtake\)\s*=\s*(\d+(?:\.\d+)?)", content)
            if probability:
                result["overtake_prob"] = float(probability.group(1))
            for pattern, key in (
                (r"gap=([\d.]+)s", "gap_ahead_s"),
                (r"pace_delta=([-\d.]+)s/lap", "pace_delta_s"),
            ):
                match = re.search(pattern, content)
                if match:
                    result[key] = float(match.group(1))

        sc = re.search(r"P\(SC 3-lap\)\s*=\s*(\d+(?:\.\d+)?)", content)
        if sc and result["sc_prob_3lap"] == 0.0:
            result["sc_prob_3lap"] = float(sc.group(1))
    return result


# ─────────────────────────────────────────────────────────────────────────────
# LangGraph / LangChain optional imports
# ─────────────────────────────────────────────────────────────────────────────

try:
    from langchain_core.tools import tool as lc_tool
    from langchain_core.messages import HumanMessage
    from langchain_openai import ChatOpenAI
    from langchain.agents import create_agent

    _LANGGRAPH_AVAILABLE = True
except ImportError:
    _LANGGRAPH_AVAILABLE = False


# ─────────────────────────────────────────────────────────────────────────────
# System prompt (module-level constant, interpolates CFG's loaded thresholds)
# ─────────────────────────────────────────────────────────────────────────────

# f-string, not a plain triple-quoted constant: the prompt must quote the SAME bands
# RaceSituationOutput.__post_init__ actually classifies against, or the LLM is told a
# rule the code no longer applies. CFG is fully built by the time this module-level
# constant is evaluated (CFG is constructed above, at import time). #450/#665.
_RACE_SITUATION_SYSTEM_PROMPT = f"""You are a Formula 1 race situation analyst embedded in a multi-agent strategy system.

Your job is to assess two dimensions of strategic threat per lap:

1. **Overtaking opportunity** — Is there a realistic window for the driver to pass the car directly ahead within the next few laps?
2. **Safety Car risk** — Is a Safety Car deployment likely within the next 3 laps based on current race chaos indicators?

## Workflow

1. If the gap to the car ahead is less than 2.5 seconds, call `predict_overtake_tool` with the chasing driver (driver_x) and the car ahead (driver_y) at the current lap number.
2. Always call `predict_sc_tool` with the current lap number to assess SC deployment risk.
3. Synthesize a **threat level** based on the two probabilities:
   - **HIGH**: Either P(overtake) >= {CFG.high_overtake:.3f} OR P(SC 3-lap) >= {CFG.high_sc:.3f}
   - **MEDIUM**: Either P(overtake) >= {CFG.medium_overtake:.3f} OR P(SC 3-lap) >= {CFG.medium_sc:.3f}
   - **LOW**: Both probabilities below medium thresholds

## Rules

- Always call BOTH tools before drawing conclusions.
- If gap ahead > 2.5s, skip overtake tool and assume P(overtake) = 0.0.
- Base your threat assessment ONLY on the numeric probabilities returned by the tools.
- Keep your final answer concise: state the threat level, both probabilities, and one sentence explaining why.

## Strategic guard-rails
- OPENING LAPS (laps 1-3): Race starts naturally inflate both overtake probability
  and SC risk due to first-lap chaos, bunched-up grid, and cold tyres. These are
  normal start dynamics, not genuine strategic threats. When reporting for laps 1-3:
  * Append "opening-lap inflation — discount for strategy decisions" to your reasoning.
  * Consider the effective threat ONE LEVEL LOWER than raw numbers suggest
    (HIGH → treat as MEDIUM, MEDIUM → treat as LOW for strategic purposes).
  * Note that DRS is typically not activated until lap 3, so overtake probability
    in laps 1-2 is inflated by models trained on DRS-enabled data.
- SAFETY CAR vs SC PROBABILITY: your sc_prob_3lap output represents a prediction,
  not a confirmed deployment. Make this distinction explicit in your reasoning:
  "SC probability 0.35 (elevated, but SC not confirmed)" — so downstream agents
  don't treat a prediction as a fact."""


# ─────────────────────────────────────────────────────────────────────────────
# RaceSituationAgent: encapsulated agent class
# ─────────────────────────────────────────────────────────────────────────────


class RaceSituationAgent:
    """Encapsulated Race Situation Agent combining N12 overtake and N14 SC models.

    Owns all mutable state previously held in module-level globals:
    - laps_df / session_meta: set per call by run() / run_from_state()
    - _react_agent: lazily created LangGraph CompiledGraph

    LangChain tools are built as closures inside _build_tools() so they read
    instance attributes (self.laps_df, self.session_meta, self.cfg) without
    depending on any module-level globals.

    Args:
        cfg: RaceSituationConfig instance. Defaults to the module-level CFG
            singleton so RaceSituationOutput.__post_init__ remains consistent.
    """

    def __init__(self, cfg: RaceSituationConfig = CFG) -> None:
        self.cfg: RaceSituationConfig = cfg
        self.laps_df: pd.DataFrame = pd.DataFrame()
        self.session_meta: dict = {}
        self._react_agent = None
        self._tools: list = self._build_tools()

    # ── Feature builders (instance methods: use self.cfg) ─────────────────────

    def _build_overtake_features(
        self,
        driver_x_lap: pd.Series,
        driver_y_lap: pd.Series,
        laps_recent: pd.DataFrame,
        circuit_cluster: int,
        gp_name: str = "",
        year: int = 2025,
    ) -> pd.DataFrame:
        """Build the 15 N12 overtake model features from a driver pair at one lap.

        Replicates the N12 training feature pipeline exactly. driver_x is the chasing
        car (attempting overtake), driver_y is the car directly ahead. laps_recent
        must contain at least 3 laps for both drivers to compute rolling trends.

        Gap is computed via session elapsed Time column when available (same method
        as N27 / N12 training), falling back to raw lap-time difference.

        Args:
            driver_x_lap: FastF1 lap Series for the chasing driver. Required:
                LapTime (Timedelta), TyreLife, Compound, SpeedST, LapNumber, Driver.
                Optional: Time (session elapsed Timedelta for accurate gap).
            driver_y_lap: FastF1 lap Series for the car directly ahead.
            laps_recent: DataFrame of laps for both drivers over the last 3+ laps.
                Columns: Driver, LapNumber, LapTime, Time (optional).
            circuit_cluster: Integer cluster ID (0-3) from cfg.circuit_cluster_map.
            gp_name: GP short name for absolute compound lookup (e.g. 'Sakhir').
            year: Race year for compound lookup.

        Returns:
            Single-row DataFrame with 15 columns in cfg.overtake_features order.
            compound_x/y cast to pandas category for LightGBM categorical encoding.
        """
        t_x = driver_x_lap.get("Time")
        t_y = driver_y_lap.get("Time")
        if pd.notna(t_x) and pd.notna(t_y):
            gap_ahead_s = float((t_x - t_y).total_seconds())
        else:
            gap_ahead_s = float((driver_x_lap["LapTime"] - driver_y_lap["LapTime"]).total_seconds())
        # NaN-preserving. `max(0.0, nan)` returns 0.0 in Python: the comparison is False,
        # so the first argument wins, which turned an unmeasurable gap into the single
        # most aggressive reading available: zero seconds, inside the trained domain, DRS
        # window open. The clamp is for a negative gap, which end-of-lap order makes
        # unreachable for an adjacent pair anyway; it was never meant to absorb an absence.
        gap_ahead_s = float("nan") if pd.isna(gap_ahead_s) else max(0.0, gap_ahead_s)

        pace_delta_s = float((driver_x_lap["LapTime"] - driver_y_lap["LapTime"]).total_seconds())
        tyre_life_x = _lap_count(driver_x_lap, "TyreLife")
        tyre_life_y = _lap_count(driver_y_lap, "TyreLife")
        tyre_life_diff = tyre_life_x - tyre_life_y
        speed_trap_delta = float(driver_x_lap.get("SpeedST", 300.0)) - float(
            driver_y_lap.get("SpeedST", 300.0)
        )
        lap_number = _lap_count(driver_x_lap, "LapNumber")
        # DRS is unavailable under SC/VSC (Art. 22.1(c): activation only resumes one lap
        # after a safety car period, two in the 2023 regulation). The gap-based rule
        # below cannot know that, so a neutralised lap used to report an open DRS window
        # purely because the field had bunched to under a second: the feature was live
        # and lying, on exactly the laps where it is regulated shut.
        drs_allowed = not _is_neutralised(driver_x_lap.get("TrackStatus"))
        drs_window = int(gap_ahead_s < 1.0) if drs_allowed else 0
        drs_ready_gap = gap_ahead_s * drs_window

        compound_x = _abs_compound(str(driver_x_lap.get("Compound", "MEDIUM")), gp_name, year)
        compound_y = _abs_compound(str(driver_y_lap.get("Compound", "MEDIUM")), gp_name, year)
        gap_pace_product = gap_ahead_s * pace_delta_s

        pace_delta_rolling3, gap_trend = _pair_rolling_features(
            laps_recent,
            driver_x=str(driver_x_lap["Driver"]),
            driver_y=str(driver_y_lap["Driver"]),
            lap_number=lap_number,
            gap_ahead_s=gap_ahead_s,
            pace_delta_s=pace_delta_s,
        )

        return pd.DataFrame(
            [
                {
                    "gap_ahead_s": gap_ahead_s,
                    "pace_delta_s": pace_delta_s,
                    "tyre_life_x": tyre_life_x,
                    "tyre_life_y": tyre_life_y,
                    "tyre_life_diff": tyre_life_diff,
                    "speed_trap_delta": speed_trap_delta,
                    "LapNumber": lap_number,
                    "drs_window": drs_window,
                    "compound_x": compound_x,
                    "compound_y": compound_y,
                    "circuit_cluster": circuit_cluster,
                    "gap_pace_product": gap_pace_product,
                    "drs_ready_gap": drs_ready_gap,
                    "gap_trend": gap_trend,
                    "pace_delta_rolling3": pace_delta_rolling3,
                }
            ]
        )[self.cfg.overtake_features]

    def _build_sc_features(
        self,
        all_laps: pd.DataFrame,
        lap_number: int,
        session_meta: dict,
    ) -> pd.DataFrame:
        """Build all 32 N14 SC model features for lap_number from the full race history.

        Replicates the N14 training feature pipeline (N13 aggregate_laps +
        track_status + RCM features). all_laps must contain all accurate laps from
        lap 1 to the current lap: passing only recent laps breaks the causal
        z-score normalisation that N14 was trained with.

        Args:
            all_laps: Accurate FastF1 laps from race start to current lap (inclusive).
                Required: Driver, LapNumber, LapTime, TyreLife, Compound, TrackStatus.
            lap_number: Current lap number (strictly causal, no future data used).
            session_meta: Dict with: circuit_cluster, circuit_sc_rate, total_laps,
                AirTemp, TrackTemp, Humidity, track_temp_start, and optionally 'session'
                (FastF1 Session for RCM access; omit in replay engine context).

        Returns:
            Single-row DataFrame with 32 columns in cfg.sc_features order.
        """
        cur = all_laps[all_laps["LapNumber"] == lap_number]
        prev = all_laps[all_laps["LapNumber"] == lap_number - 1]

        feat: dict = {}
        feat.update(_compute_laptime_features(all_laps, lap_number))
        feat.update(_compute_driver_tyre_features(cur, prev))

        ts_feat = _compute_track_status_features(all_laps, lap_number)
        cur_code = ts_feat.pop("_cur_code")
        prev_code = ts_feat.pop("_prev_code")
        ts_feat.pop("_yel_esc")
        feat.update(ts_feat)

        feat.update(_compute_rcm_features(all_laps, lap_number, session_meta, cur_code, prev_code))
        feat.update(_compute_weather_features(session_meta))

        # DEFAULT_TOTAL_LAPS: see src/agents/_shared_defaults.py for why this fallback
        # exists and why it is single-sourced across the strategy agents.
        total_laps = int(session_meta.get("total_laps", DEFAULT_TOTAL_LAPS))
        is_lap1 = int(lap_number == 1)
        lap_pct = float(lap_number) / max(total_laps, 1)

        anom_hard = 0
        hist = all_laps[all_laps["LapNumber"] < lap_number]
        if not cur.empty and not hist.empty:
            for drv in cur["Driver"].unique():
                h = hist[hist["Driver"] == drv]["LapTime"].dt.total_seconds().tail(5)
                if len(h) >= 2:
                    med = h.median()
                    lt_cur = cur.loc[cur["Driver"] == drv, "LapTime"].dt.total_seconds()
                    if not lt_cur.empty and med > 0 and float(lt_cur.iloc[0]) / med > 1.30:
                        anom_hard += 1

        yel_esc = feat.get("yellow_escalation_count", 0)
        feat["anomaly_and_yellow"] = int(anom_hard > 0 and yel_esc > 0)
        feat["lap1_chaos"] = is_lap1 * abs(feat.get("n_drivers_delta", 0))
        feat["circuit_cluster"] = int(session_meta.get("circuit_cluster", 0))
        feat["circuit_sc_rate"] = float(session_meta.get("circuit_sc_rate", 0.10))
        feat["lap_pct"] = lap_pct
        feat["is_lap1"] = is_lap1

        return pd.DataFrame([feat])[self.cfg.sc_features]

    # ── LLM-input validation (shared by both LangChain tools) ────────────────

    def _lap_range_error(self, lap_number: int) -> Optional[str]:
        """Reject a lap_number outside the causal range the agent was loaded with.

        run() loads the WHOLE FastF1 session into self.laps_df (session.laps.pick_
        accurate()), so nothing on its own stops a hallucinated future lap number
        from reaching the feature builders and returning a confident-looking
        prediction built from data the race hasn't reached yet (#476). self.
        _current_lap is set once per call in run()/run_from_state() to the lap
        actually being assessed; anything beyond it is a future lookahead.

        Args:
            lap_number: The lap_number argument as supplied by the LLM tool call.

        Returns:
            An error string naming the problem, or None when lap_number is valid.
        """
        if lap_number < 1:
            return f"invalid lap_number {lap_number} — laps start at 1"
        current = getattr(self, "_current_lap", None)
        if current is not None and lap_number > current:
            return (
                f"invalid lap_number {lap_number} — the race is currently at lap "
                f"{current}; cannot query a future lap"
            )
        return None

    def _unknown_driver_error(self, drivers: tuple[str, ...], lap_number: int) -> Optional[str]:
        """Reject driver codes that are not racing at lap_number.

        Both LangChain tools take driver abbreviations as free LLM text, so a
        hallucinated or retired driver code would otherwise reach the feature
        builders unchecked (#476), mirrors pit_strategy_agent's score_undercut_tool
        guard, same error shape (names the valid options instead of just failing).

        Args:
            drivers: One or more driver abbreviations to check.
            lap_number: Lap number, only used to phrase the error message.

        Returns:
            An error string naming the offending driver(s) and the valid roster,
            or None when every driver is live (or presence is unknown, in which
            case the guard disables rather than rejecting every target).
        """
        live = getattr(self, "_live_drivers", None)
        if live is None:
            return None
        unknown = [d for d in drivers if d not in live]
        if not unknown:
            return None
        return (
            f"{' and '.join(unknown)} not on track at lap {lap_number}. "
            f"Drivers racing this lap: {', '.join(sorted(live))}. Pick from that list."
        )

    # ── LangChain tool factory ────────────────────────────────────────────────

    def _build_tools(self) -> list:
        """Build LangChain tools as closures over this RaceSituationAgent instance.

        Each tool reads self.laps_df, self.session_meta, and self.cfg at call time.
        No module-level globals are accessed. Returns an empty list when LangGraph
        is not installed so the agent degrades gracefully.

        Returns:
            List of decorated LangChain tool functions.
        """
        if not _LANGGRAPH_AVAILABLE:
            return []

        agent = self  # capture instance for closures

        @lc_tool
        def predict_overtake_tool(driver_x: str, driver_y: str, lap_number: int) -> str:
            """Predict overtaking probability for driver_x chasing driver_y at lap_number.

            Constructs the 15 N12 overtake features from the session loaded into the
            agent instance, runs LightGBM + Platt calibration, returns calibrated P(overtake).

            Args:
                driver_x: FastF1 abbreviation of the chasing car (e.g. 'NOR').
                driver_y: FastF1 abbreviation of the car directly ahead (e.g. 'PIA').
                lap_number: Current lap number.

            Returns:
                "P(overtake) = 0.XXX | gap=X.XXs | pace_delta=X.XXXs/lap | DRS: active/inactive"
            """
            lap_err = agent._lap_range_error(lap_number)
            if lap_err:
                return f"Overtake scoring REFUSED — {lap_err}"
            drv_err = agent._unknown_driver_error((driver_x, driver_y), lap_number)
            if drv_err:
                return f"Overtake scoring REFUSED — {drv_err}"

            x_rows = agent.laps_df[
                (agent.laps_df["Driver"] == driver_x) & (agent.laps_df["LapNumber"] == lap_number)
            ]
            y_rows = agent.laps_df[
                (agent.laps_df["Driver"] == driver_y) & (agent.laps_df["LapNumber"] == lap_number)
            ]

            if x_rows.empty or y_rows.empty:
                return f"No lap data for {driver_x} or {driver_y} at lap {lap_number}"

            laps_recent = agent.laps_df[
                agent.laps_df["Driver"].isin([driver_x, driver_y])
                & (agent.laps_df["LapNumber"] >= lap_number - 3)
                & (agent.laps_df["LapNumber"] <= lap_number)
            ]

            feat_df = agent._build_overtake_features(
                x_rows.iloc[0],
                y_rows.iloc[0],
                laps_recent,
                circuit_cluster=agent.session_meta.get("circuit_cluster", _UNKNOWN_CIRCUIT_CLUSTER),
                gp_name=agent.session_meta.get("gp_name", ""),
                year=agent.session_meta.get("year", 2025),
            )

            for i, col in enumerate(["compound_x", "compound_y", "circuit_cluster"]):
                training_cats = agent.cfg.overtake_model._Booster.pandas_categorical[i]
                feat_df[col] = pd.Categorical(feat_df[col], categories=training_cats)

            gap = feat_df["gap_ahead_s"].iloc[0]
            pace = feat_df["pace_delta_s"].iloc[0]
            drs = "active" if feat_df["drs_window"].iloc[0] else "inactive"

            # Outside N11's trained domain the model is not wrong, it is uninformed: it
            # never saw a labelled pair this far apart. Say so instead of publishing the
            # extrapolation. The other two facts are still measurements, so they stay.
            #
            # An UNMEASURABLE gap declines too. Whether the pair is inside the domain is
            # exactly what cannot be established without it, and answering anyway would be
            # the same fabrication one branch further along.
            if pd.isna(gap):
                return (
                    f"P(overtake) = {_OUT_OF_DOMAIN_MARKER} (the gap could not be measured "
                    f"on this lap, so whether the pair is inside N11's trained "
                    f"{_TRAINED_MAX_GAP_S}s domain is unknown) | "
                    f"gap=unknown | pace_delta={pace:.3f}s/lap | DRS: {drs}"
                )
            if float(gap) > _TRAINED_MAX_GAP_S:
                return (
                    f"P(overtake) = {_OUT_OF_DOMAIN_MARKER} "
                    f"(gap {float(gap):.2f}s is beyond N11's trained {_TRAINED_MAX_GAP_S}s "
                    f"domain; no labelled example exists out here) | "
                    f"gap={gap:.2f}s | "
                    f"pace_delta={pace:.3f}s/lap | "
                    f"DRS: {drs}"
                )

            raw_proba = agent.cfg.overtake_model.predict_proba(feat_df)[:, 1]
            calib_proba = agent.cfg.overtake_calibrator.predict_proba(raw_proba.reshape(-1, 1))[
                :, 1
            ][0]

            return (
                f"P(overtake) = {calib_proba:.3f} | "
                f"gap={gap:.2f}s | "
                f"pace_delta={pace:.3f}s/lap | "
                f"DRS: {drs}"
            )

        @lc_tool
        def predict_sc_tool(lap_number: int) -> str:
            """Predict Safety Car deployment probability within the next 3 laps.

            Constructs the 32 N14 SC features from the session loaded into the agent
            instance, runs LightGBM + Platt calibration, returns calibrated P(SC within 3 laps).

            Args:
                lap_number: Current lap number.

            Returns:
                "P(SC 3-lap) = 0.XXX | lap_time_std_z=X.XX | circuit_sc_rate=X.XX | status: {status} | {incident}"
            """
            lap_err = agent._lap_range_error(lap_number)
            if lap_err:
                return f"SC scoring REFUSED — {lap_err}"
            if len(agent.laps_df) < 10:
                return f"Insufficient lap data at lap {lap_number}"

            feat_df = agent._build_sc_features(agent.laps_df, lap_number, agent.session_meta)

            raw_proba = agent.cfg.sc_model.predict_proba(feat_df)[:, 1]
            calib_proba = agent.cfg.sc_calibrator.predict_proba(raw_proba.reshape(-1, 1))[:, 1][0]

            lt_std_z = feat_df["lap_time_std_z"].iloc[0]
            sc_rate = feat_df["circuit_sc_rate"].iloc[0]
            status_enc = int(feat_df["track_status_enc"].iloc[0])
            had_incident = int(feat_df["had_incident_msg"].iloc[0])

            _status_desc = {
                0: "green",
                1: "yellow",
                2: "red flag",
                3: "VSC ending",
                4: "VSC",
                5: "SC",
            }
            return (
                f"P(SC 3-lap) = {calib_proba:.3f} | "
                f"lap_time_std_z={lt_std_z:.2f} | "
                f"circuit_sc_rate={sc_rate:.2f} | "
                f"status: {_status_desc.get(status_enc, 'unknown')} | "
                f"{'incident flagged' if had_incident else 'no incidents'}"
            )

        return [predict_overtake_tool, predict_sc_tool]

    # ── LangGraph agent (lazy) ────────────────────────────────────────────────

    def get_react_agent(
        self,
        provider: str = None,
        model_name: str = "gpt-4.1-mini",
        base_url: str = "http://localhost:1234/v1",
        api_key: str = "lm-studio",
    ):
        """Return the LangGraph ReAct agent, creating it on the first call (lazy).

        Avoids connecting to the LLM at import time: compiled only when N31 or
        a test actually invokes the agent.

        Args:
            provider: 'lmstudio' (default) or 'openai'.
            model_name: Model identifier for ChatOpenAI.
            base_url: Base URL for LM Studio (ignored when provider='openai').
            api_key: API key; use 'lm-studio' for local server.

        Returns:
            LangGraph CompiledGraph: invoke with {"messages": [HumanMessage(...)]}.

        Raises:
            ImportError: When LangGraph / LangChain are not installed.
        """
        if not _LANGGRAPH_AVAILABLE:
            raise ImportError("LangGraph / LangChain not installed.")

        if self._react_agent is not None:
            return self._react_agent

        import os

        if provider is None:
            provider = os.environ.get("F1_LLM_PROVIDER", "lmstudio")

        if provider == "lmstudio":
            llm = ChatOpenAI(
                model=model_name,
                base_url=base_url,
                api_key=api_key,
                temperature=0,
                timeout=120,
                max_retries=1,
            )
        else:
            llm = ChatOpenAI(model=model_name, temperature=0, timeout=120, max_retries=1)

        self._react_agent = create_agent(
            model=llm,
            tools=self._tools,
            system_prompt=_RACE_SITUATION_SYSTEM_PROMPT,
        )
        return self._react_agent

    # ── Entry point methods ───────────────────────────────────────────────────

    def run(self, lap_state: dict) -> RaceSituationOutput:
        """Run the Race Situation Agent from a FastF1 session-based lap_state.

        Populates self.laps_df and self.session_meta from the FastF1 Session in
        lap_state, then invokes the ReAct agent. Probabilities are extracted
        from tool call results (not LLM free text) for deterministic output.

        Args:
            lap_state: Dict with keys:
                session      : Loaded FastF1 Session (laps + weather cached).
                driver       : FastF1 driver abbreviation (e.g. 'NOR').
                rival_ahead  : Abbreviation of the car directly ahead. None = skip overtake.
                lap_number   : Current lap number (int).
                gp_name      : GP name matching circuit_cluster_map keys (e.g. 'Sakhir').
                event_name   : Event name matching circuit_sc_rate_map keys.
                year         : Race year (int).

        Returns:
            RaceSituationOutput with overtake_prob, sc_prob_3lap, threat_level,
            gap_ahead_s, pace_delta_s, and LLM reasoning string.
        """
        session = lap_state["session"]
        driver = lap_state["driver"]
        rival_ahead = lap_state.get("rival_ahead")
        lap_number = lap_state["lap_number"]
        gp_name = lap_state["gp_name"]
        event_name = lap_state["event_name"]

        self.laps_df = session.laps.pick_accurate().copy()
        _clean = self.laps_df[self.laps_df["TrackStatus"] == "1"]
        _wx = session.weather_data

        # Who is actually on track at this lap. session.laps carries the WHOLE race
        # (not just laps up to lap_number), so without this a free-text driver_x/
        # driver_y from the LLM can name a car that already retired, or a lap the
        # driver never reached, and still get a confident-looking prediction back
        # (#476), mirrors pit_strategy_agent's `_live_drivers` (#462). Empty means
        # that could not be told (e.g. lap_number outside the session), so the guard
        # disables (None) rather than rejecting every target.
        _at_lap = self.laps_df.loc[self.laps_df["LapNumber"] == lap_number, "Driver"].dropna()
        self._live_drivers = set(_at_lap) | {driver} if len(_at_lap) else None
        self._current_lap = lap_number

        self.session_meta = {
            "session": session,
            "gp_name": gp_name,
            "event_name": event_name,
            "year": lap_state.get("year", 2025),
            # N11's unknown-circuit code, not 0: 0 is a REAL cluster, so an unresolved
            # unresolved circuit to 0 would score it as a specific kind of track. -1 is
            # what N11's get_cluster returns (`.nb_py/N11_overtake_eda.py:210-212`), and
            # since the booster's trained levels are [0,1,2,3] the Categorical cast in
            # predict_overtake_tool turns it into the missing value LightGBM handles
            # natively. Latent today, since every race resolves, but the collision was real.
            "circuit_cluster": self.cfg.cluster_for(gp_name, _UNKNOWN_CIRCUIT_CLUSTER),
            "circuit_sc_rate": self.cfg.sc_rate_for(event_name),
            "total_laps": int(session.total_laps),
            "fastest_lap_s": _clean["LapTime"].min().total_seconds(),
            "AirTemp": float(_wx["AirTemp"].mean()) if "AirTemp" in _wx else 28.0,
            "TrackTemp": float(_wx["TrackTemp"].mean())
            if "TrackTemp" in _wx
            else DEFAULT_TRACK_TEMP_C,
            "Humidity": float(_wx["Humidity"].mean()) if "Humidity" in _wx else 50.0,
            "track_temp_start": float(_wx["TrackTemp"].iloc[0])
            if "TrackTemp" in _wx
            else DEFAULT_TRACK_TEMP_C,
        }

        # Carry the RCM events into _run_core so the SC override can read them
        # without changing the function's positional signature.
        self._pending_rcm_events = lap_state.get("rcm_events", []) or []
        return self._run_core(driver, rival_ahead, lap_number)

    def run_from_state(self, lap_state: dict, laps_df: pd.DataFrame) -> RaceSituationOutput:
        """RSM adapter: run the Race Situation Agent from a RaceStateManager lap_state.

        Translates the nested RSM lap_state dict into self.laps_df + self.session_meta.
        Unlike run(), this does NOT require a FastF1 session object: it builds state
        from laps_df and lap_state directly.

        RCM-based features (had_incident_msg, yellow_sectors_*) default to 0 when no
        FastF1 session is available: the agent still produces valid SC probability
        estimates using track status and lap-time variance signals.

        The rival_ahead is derived from lap_state['rivals'] by looking for the car
        with position = driver_position - 1. When the driver's own position is
        unknown (missing from the telemetry dict), rival_ahead is None rather than
        guessing a position: a guessed default is a searchable grid slot, so it
        would otherwise silently pair the driver with whoever is one place ahead of
        the DEFAULT, not the truth (#465).

        Args:
            lap_state: Dict from RaceStateManager.get_lap_state(). Expected keys:
                lap_number, driver (telemetry), rivals (list), weather, session_meta.
            laps_df: Full race laps DataFrame. Must include LapTime (Timedelta or
                float seconds in LapTime_s), Driver, LapNumber, TyreLife, Compound,
                TrackStatus. Time (session elapsed Timedelta) improves gap accuracy.

        Returns:
            RaceSituationOutput with all fields populated.
        """
        d = lap_state["driver"]
        meta = lap_state["session_meta"]
        wx = lap_state.get("weather", {})
        rivals = lap_state.get("rivals", [])

        lap_number = lap_state["lap_number"]
        driver = meta["driver"]
        gp_name = meta.get("gp_name", "")
        # DEFAULT_TOTAL_LAPS: see src/agents/_shared_defaults.py.
        total_laps = meta.get("total_laps", DEFAULT_TOTAL_LAPS)
        year = meta.get("year", 2025)

        # #465 (F6): a defaulted position (previously `.get('position', 20)`) is a
        # SEARCHABLE value, not a safe placeholder: if the real grid has a car at
        # P19, "unknown position" and "genuinely P20" silently produce the same
        # rival_ahead lookup. An unknown position must propagate as "no rival
        # computable", not guess P20.
        driver_pos = d.get("position")
        rival_ahead = (
            next((r["driver"] for r in rivals if r.get("position") == driver_pos - 1), None)
            if driver_pos is not None
            else None
        )

        self.laps_df = _ensure_timedelta_laps(laps_df)
        event_name = meta.get("event_name", gp_name)

        # Who is actually on track this lap, from the RSM's `rivals` list (a car that
        # retired simply is not in it, the same answer a timing screen gives). Used
        # to refuse predict_overtake_tool/predict_sc_tool calls naming a driver who
        # isn't racing this lap (#476), mirroring pit_strategy_agent's `_live_drivers`
        # (#462). An empty rivals list means the roster is unknown, not "only the driver's car
        # is racing", so it disables the guard (None) rather than rejecting every
        # target, same convention as pit_strategy_agent.run_from_state.
        on_track = {r["driver"] for r in rivals if r.get("driver")}
        self._live_drivers = on_track | {driver} if on_track else None
        self._current_lap = lap_number

        self.session_meta = {
            "session": None,
            "gp_name": gp_name,
            "event_name": event_name,
            "year": year,
            # N11's unknown-circuit code, not 0: 0 is a REAL cluster, so an unresolved
            # unresolved circuit to 0 would score it as a specific kind of track. -1 is
            # what N11's get_cluster returns (`.nb_py/N11_overtake_eda.py:210-212`), and
            # since the booster's trained levels are [0,1,2,3] the Categorical cast in
            # predict_overtake_tool turns it into the missing value LightGBM handles
            # natively. Latent today, since every race resolves, but the collision was real.
            "circuit_cluster": self.cfg.cluster_for(gp_name, _UNKNOWN_CIRCUIT_CLUSTER),
            "circuit_sc_rate": self.cfg.sc_rate_for(event_name),
            "total_laps": total_laps,
            "fastest_lap_s": float(self.laps_df["LapTime"].dt.total_seconds().min())
            if len(self.laps_df) > 0
            else 90.0,
            # reading_or_default, not .get(key, default): the producers report an
            # unmeasured reading as the key PRESENT holding None, which .get's default
            # never catches, and _compute_weather_features' float() then raises. That
            # 422'd /recommend on every 2025 lap (#788), see the helper's docstring.
            "AirTemp": reading_or_default(wx, "air_temp", DEFAULT_AIR_TEMP_C),
            "TrackTemp": reading_or_default(wx, "track_temp", DEFAULT_TRACK_TEMP_C),
            "Humidity": reading_or_default(wx, "humidity", 50.0),
            # The session's FIRST track temp, which the RSM now supplies. This used to
            # read `track_temp` (the CURRENT one), so `track_temp_delta` came out 0.0 on
            # every lap of every race, on every shipping path (CLI, arcade, backend,
            # /recommend, no-llm). The FastF1 path never had the bug: it reads
            # `_wx['TrackTemp'].iloc[0]`, which is what this now mirrors.
            #
            # It is N14's 5th most important feature (6.0% gain). Real 2024 deltas reach
            # -9.1 C (Monaco); the sensitivity is small mid-race and ~1.8x late, which is
            # exactly where a late SC decides a result. Falling back to the current temp
            # would reinstate the bug silently, so an absent value degrades to the
            # training default instead.
            "track_temp_start": wx.get("track_temp_start") or DEFAULT_TRACK_TEMP_C,
        }

        # Carry the RCM events into _run_core so the SC override can read them.
        self._pending_rcm_events = lap_state.get("rcm_events", []) or []
        return self._run_core(driver, rival_ahead, lap_number)

    def _run_core(
        self,
        driver: str,
        rival_ahead: Optional[str],
        lap_number: int,
    ) -> RaceSituationOutput:
        """Invoke the ReAct agent with session state already set; parse and return output.

        self.laps_df and self.session_meta must be populated before calling this method.

        Args:
            driver: FastF1 driver abbreviation.
            rival_ahead: Abbreviation of the car directly ahead, or None.
            lap_number: Current lap number.

        Returns:
            Fully populated RaceSituationOutput.
        """
        if not _LANGGRAPH_AVAILABLE:
            raise ImportError("LangGraph / LangChain not installed.")

        if rival_ahead:
            message = (
                f"Assess the race situation for driver {driver} at lap {lap_number}. "
                f"The car ahead is {rival_ahead}. "
                f"Determine the overtaking probability and Safety Car risk, then provide a threat level."
            )
        else:
            # NOT "no car is within overtaking range (gap > 2.5s)", which is what this
            # said. `rival_ahead` comes from a POSITION lookup with no gap filter at all
            # (see its derivation above), so it is None when the driver is leading, when
            # the car ahead is missing from the timing feed, or when the driver's own
            # position is unknown, never because of a gap. The prompt was handing the LLM a reason
            # that was not the reason, and one it could reason from. The gap-domain case
            # is the tool's own to report, and it now does.
            message = (
                f"Assess the race situation for driver {driver} at lap {lap_number}. "
                f"No car is classified directly ahead (leading, or the car ahead is not "
                f"in the timing feed), so there is no overtake to score. "
                f"Determine the Safety Car risk and provide a threat level."
            )

        react_agent = self.get_react_agent()
        response = react_agent.invoke({"messages": [HumanMessage(content=message)]})
        parsed = _parse_tool_outputs(response["messages"])
        reasoning = response["messages"][-1].content

        # Post-hoc override: when the lap's RCM events confirm a neutralisation is
        # currently deployed, force sc_prob_3lap to 1.0 and flag the output so downstream
        # agents (N28 pit, N31 orchestrator) can react. The legacy LightGBM model was
        # trained to predict a FUTURE SC, not to recognise one already in progress, hence
        # the patch. The SC/VSC split (#471) rides on the same signal: both are
        # neutralisations (overtake_prob = 0, sc_prob_3lap = 1.0), but only the KIND
        # changes the pit-time saving, carried downstream on vsc_active.
        neutralization = _neutralization_from_rcm(getattr(self, "_pending_rcm_events", None) or [])
        is_neutralized = neutralization is not Neutralization.NONE
        is_vsc = neutralization is Neutralization.VSC
        raw_sc_prob = round(parsed["sc_prob_3lap"], 3)
        effective_sc_prob = 1.0 if is_neutralized else raw_sc_prob

        # Art. 55.8 (SC) / 56.6 (VSC): no overtaking on track. N12 predicts a RACING
        # overtake, and under a neutralisation the only exception that yields a real
        # position gain is 55.8(h) ("a car slows with an obvious problem"), a mechanism
        # N12 has no feature for. Meanwhile every input it does use is regulation-
        # corrupted: DRS is disabled (Art. 22.1(c)), the gap compresses toward ten car
        # lengths (55.7/55.10) and pace_delta collapses to the FIA ECU delta. So the model
        # is not merely imprecise here, it is inapplicable: 0.0 is the correct value and
        # the honest one, and it holds identically under a VSC (56.6).
        #
        # The override applies whether or not the model answered, and that ordering is the
        # point: under a neutralisation 0.0 is asserted by the REGULATION, so it holds even
        # for a pair the model declined to score. An unscored pair on a green lap keeps its
        # None: "the rules forbid it" and "nobody knows" must not collapse into one number.
        raw_overtake_prob = (
            None if parsed["overtake_prob"] is None else round(parsed["overtake_prob"], 3)
        )
        effective_overtake_prob = 0.0 if is_neutralized else raw_overtake_prob

        _kind_label = "VIRTUAL_SAFETY_CAR_DEPLOYED" if is_vsc else "SAFETY_CAR_DEPLOYED"
        _overtake_article = "56.6" if is_vsc else "55.8"
        effective_reasoning = (
            reasoning
            if not is_neutralized
            else (
                f"[RCM OVERRIDE: {_kind_label} active — model output "
                f"sc_prob_3lap={raw_sc_prob:.2f} overridden to 1.00, "
                f"overtake_prob={_fmt_prob(raw_overtake_prob)} overridden to 0.00 "
                f"(no overtaking under a neutralisation, Art. {_overtake_article}).] {reasoning}"
            )
        )

        return RaceSituationOutput(
            overtake_prob=effective_overtake_prob,
            sc_prob_3lap=effective_sc_prob,
            gap_ahead_s=(
                round(parsed["gap_ahead_s"], 2) if parsed["gap_ahead_s"] is not None else None
            ),
            pace_delta_s=round(parsed["pace_delta_s"], 3),
            reasoning=effective_reasoning,
            sc_currently_active=is_neutralized,
            vsc_active=is_vsc,
        )


# ─────────────────────────────────────────────────────────────────────────────
# RCM context override helper
# ─────────────────────────────────────────────────────────────────────────────


def _classify_rcm_events(rcm_events: list | None) -> list[str]:
    """Classify a lap's RCM events into canonical event-type strings.

    Accepts whatever shape the lap_state ships:
      - dicts already pre-classified with an ``event_type`` key (the cheap path used by
        RaceStateManager and the CLI's synthetic events once classified upstream),
      - ``RCMEvent`` dataclass instances from radio_agent,
      - raw FastF1-shaped dicts (``message``, ``flag``, ``category``, ``lap``...), which
        are promoted to RCMEvent and classified.

    The radio_agent import is lazy AND only taken when a raw event actually needs
    classifying: it avoids the agents -> orchestrator -> radio_agent -> agents loop at
    module import time, and keeps a caller that passes only pre-classified dicts (the
    CLI's synthetic events, the hermetic tests) from pulling in the heavy model stack.
    """
    if not rcm_events:
        return []

    classified: list[str] = []
    _radio = None
    for ev in rcm_events:
        if isinstance(ev, dict) and "event_type" in ev:
            classified.append(str(ev["event_type"]))
            continue
        if _radio is None:
            from src.agents import radio_agent as _radio
        if isinstance(ev, _radio.RCMEvent):
            classified.append(_radio._classify_rcm_event(ev))
            continue
        if isinstance(ev, dict):
            classified.append(
                _radio._classify_rcm_event(
                    _radio.RCMEvent(
                        message=str(ev.get("message", "")),
                        flag=str(ev.get("flag", "") or ""),
                        category=str(ev.get("category", "")),
                        lap=int(ev.get("lap", 0) or 0),
                        racing_number=ev.get("racing_number") or ev.get("RacingNumber"),
                        scope=str(ev.get("scope", "") or ""),
                    )
                )
            )
    return classified


def _neutralization_from_rcm(rcm_events: list | None) -> Neutralization:
    """Which neutralisation the lap's RCM events confirm is in force RIGHT NOW.

    Resolution order encodes two regulations that differ (#471):

      1. VSC release (Art. 56.7)  -> NONE. The virtual SC restarts near-instantly, so at
         lap granularity it is already over on the "VSC ENDING" lap.
      2. SC release (Art. 55.15)  -> SC. "SAFETY CAR IN THIS LAP" means the car comes in
         at the END of this lap and Art. 55.8 keeps overtaking banned until the driver
         passes the Line after it has returned, so THIS lap is still neutralised. The
         flag clears the next lap, when the cross-lap tracker stops re-asserting the
         deploy and no release event remains in the window.
      3. SC deploy                -> SC.
      4. VSC deploy               -> VSC.
      5. anything else            -> NONE.

    Release beats deploy within one window (steps 1-2 before 3-4), matching the prior
    "release wins" ordering. The only change from the old single-flag helper is that an
    SC release no longer clears the override on its own announcement lap: that lap is
    still neutralised by Art. 55.8, which is exactly the one-lap-early bug (#471).
    """
    classified = _classify_rcm_events(rcm_events)
    if not classified:
        return Neutralization.NONE

    if any(t in _VSC_RELEASE_EVENT_TYPES for t in classified):
        return Neutralization.NONE
    if any(t in _SC_RELEASE_EVENT_TYPES for t in classified):
        return Neutralization.SC
    if any(t in _SC_DEPLOY_EVENT_TYPES for t in classified):
        return Neutralization.SC
    if any(t in _VSC_DEPLOY_EVENT_TYPES for t in classified):
        return Neutralization.VSC
    return Neutralization.NONE


def _sc_active_from_rcm(rcm_events: list | None) -> bool:
    """True if the lap's RCM events confirm any neutralisation (SC or VSC) right now.

    Back-compat bool wrapper over :func:`_neutralization_from_rcm`; kept because the
    override began life as a single flag. New code should read the finer distinction from
    the enum, since SC and VSC differ on the pit-time saving (#471).
    """
    return _neutralization_from_rcm(rcm_events) is not Neutralization.NONE


# ─────────────────────────────────────────────────────────────────────────────
# Lazy singleton
# ─────────────────────────────────────────────────────────────────────────────

_default_situation_agent: Optional[RaceSituationAgent] = None


def _get_default_situation_agent() -> RaceSituationAgent:
    """Return the process-level RaceSituationAgent singleton, creating it on first call.

    Returns:
        RaceSituationAgent with N12/N14 models loaded and tools built.
    """
    global _default_situation_agent
    if _default_situation_agent is None:
        _default_situation_agent = RaceSituationAgent()
    return _default_situation_agent


# ─────────────────────────────────────────────────────────────────────────────
# Public entry points: backward-compatible signatures (unchanged)
# ─────────────────────────────────────────────────────────────────────────────


def run_race_situation_agent(lap_state: dict) -> RaceSituationOutput:
    """Run the Race Situation Agent for one lap and return structured output.

    Delegates to the process-level RaceSituationAgent singleton. Populates session
    state from the FastF1 Session in lap_state, then invokes the LangGraph ReAct
    agent. Probabilities are extracted from tool call results for deterministic output.

    Args:
        lap_state: Dict with keys: session, driver, rival_ahead, lap_number,
            gp_name, event_name, year. See RaceSituationAgent.run for full spec.

    Returns:
        RaceSituationOutput with overtake_prob, sc_prob_3lap, threat_level,
        gap_ahead_s, pace_delta_s, and reasoning.
    """
    return _get_default_situation_agent().run(lap_state)


def run_race_situation_agent_from_state(
    lap_state: dict,
    laps_df: pd.DataFrame,
) -> RaceSituationOutput:
    """RSM adapter: run the Race Situation Agent from a RaceStateManager lap_state.

    Delegates to the process-level RaceSituationAgent singleton. No FastF1 session
    required: all context is derived from laps_df and the lap_state dict.

    Args:
        lap_state: Dict from RaceStateManager.get_lap_state(). Expected keys:
            lap_number, driver, rivals, weather, session_meta.
        laps_df: Full race laps DataFrame with required telemetry columns.

    Returns:
        RaceSituationOutput with all fields populated.
    """
    return _get_default_situation_agent().run_from_state(lap_state, laps_df)
