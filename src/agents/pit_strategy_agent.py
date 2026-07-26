"""Pit Strategy Agent — src/agents/pit_strategy_agent.py

Extracted from N28_pit_strategy_agent.ipynb. Wraps N15 (physical pit stop duration
P05/P50/P95) and N16 (undercut success probability) in a LangGraph ReAct agent
that recommends when to pit, what compound to fit, and whether to undercut.

Public API
----------
run_pit_strategy_agent(lap_state)                     → PitStrategyOutput  (FastF1 session)
run_pit_strategy_agent_from_state(lap_state, laps_df) → PitStrategyOutput  (RSM adapter)
get_pit_strategy_react_agent(**kwargs)                 → CompiledGraph
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# ── Repo root (with root-stop guard for uv tool install) ─────────────────────
_REPO_ROOT = Path(__file__).resolve().parent
while not (_REPO_ROOT / '.git').exists():
    if _REPO_ROOT.parent == _REPO_ROOT:
        break
    _REPO_ROOT = _REPO_ROOT.parent

# Route every artefact path through the cache helper so the agent works
# transparently in both editable-dev mode and the ``uv tool install`` flow.
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

from src.f1_strat_manager.gp_slugs import (  # noqa: E402
    canonical_gp_name,
    rekey_by_slug,
    slug_from_event_name,
)

logger = logging.getLogger(__name__)

_MODELS    = _DATA_ROOT / 'models'
_PROCESSED = _DATA_ROOT / 'processed'
_AGENTS    = _DATA_ROOT / 'models' / 'agents'

# ── Compound allocation (SOFT/MEDIUM/HARD → Cx per GP/year) ───────────────────
_compounds_path = _DATA_ROOT / 'tire_compounds_by_race.json'
TIRE_COMPOUNDS: dict = (
    json.loads(_compounds_path.read_text(encoding='utf-8'))
    if _compounds_path.exists() else {}
)

# ── Module-level constants ─────────────────────────────────────────────────────
# FastF1 team name variants → N15 training names
_TEAM_ALIASES: dict[str, str] = {'Racing Bulls': 'RB'}

# Fallback color→compound_id when TIRE_COMPOUNDS has no entry for the circuit/year
_COMPOUND_FALLBACK: dict[str, int] = {'HARD': 1, 'MEDIUM': 3, 'SOFT': 5}

# Pirelli-typical stint capacities, used as a modelling assumption and NOT a
# sourced measurement. This carried a citation to "Heilmeier et al. (2020, SAE
# 2020-01-1413)" that nothing corroborates: the identifier appears only here and
# in the notebook this module was extracted from, and the repo has already
# corrected one wrong blanket attribution to the same author, whose real paper
# holds none of the constants once cited to it. Treat these as assumptions with
# the same standing as VSC_PIT_BONUS and WINDOW_LAPS in strategy_orchestrator.
_STINT_CAPACITY_LAPS: dict[str, int] = {'SOFT': 18, 'MEDIUM': 30, 'HARD': 38}

# N16 trained Lap_gap as the offset between the two stops (lap_y - lap_x), never the
# race lap. At inference we always score the canonical undercut: we box on this lap and
# the rival responds on the next one, so the offset is 1 (#444).
_ASSUMED_UNDERCUT_LAP_GAP: int = 1

# N15 (pit duration) encoded compound_id with an ORDINAL rank, NOT the Pirelli number:
# verbatim from N15_pit_duration.ipynb cell 11, where it is applied as
# `.map(COMPOUND_ORDER).fillna(-1)`. Feeding `_compound_to_id` here (the absolute C1-C6)
# made the model read a SOFT as WET and a MEDIUM as INTERMEDIATE on essentially every
# call. N16 is different and DOES want the absolute Cx, so the two must not share a
# helper (#445).
# ponytail: duplicated from the notebook; load it from model_config.json once N15 exports it.
_N15_COMPOUND_ORDER: dict[str, int] = {
    'SOFT': 0, 'MEDIUM': 1, 'HARD': 2, 'INTERMEDIATE': 3, 'WET': 4,
}
_N15_COMPOUND_UNKNOWN: int = -1  # matches the notebook's .fillna(-1)

# N15 clipped tyre_life_in at 50 laps in training (cell 11); mirror that ceiling here so
# an absurd stint cannot extrapolate outside the fitted range (#450).
_MAX_TRAINED_TYRE_LIFE: int = 50

# N15's tight_pit_box feature (notebook cell 4): {"Monaco Grand Prix", "Singapore Grand
# Prix", "Hungarian Grand Prix"} — circuits with narrow/short pit boxes that cause minor
# stop delays. Keyed here by the CANONICAL SLUG ('Monaco', 'Marina Bay', 'Budapest'), not
# the full event name, because gp_name at inference can arrive in EITHER keyspace exactly
# like circuit_traversal/circuit_undercut_rate above; resolve through slug_from_event_name
# before checking membership (#450, the same dual-keyspace fix as #448).
_TIGHT_PIT_BOX_SLUGS: frozenset[str] = frozenset({'Monaco', 'Marina Bay', 'Budapest'})

# N15 only trained on "normal" stops (N15_pit_duration.ipynb Step 2): physical_stop_est
# in [2.0, 4.5] s. Penalties, jack failures and unsafe releases fall outside this and are
# handled by race-control messages elsewhere, not this model. Apply the same scope when
# aggregating team_year_median from raw data (#450) so the medians reflect the same
# stops N15 was actually trained on.
_NORMAL_STOP_MIN_S: float = 2.0
_NORMAL_STOP_MAX_S: float = 4.5

CLIFF_IMMINENT_LAPS = 3    # laps_to_cliff_p10 below this → PIT_NOW
CLIFF_SOON_LAPS     = 10   # laps_to_cliff_p10 below this → prefer harder compound


# ─────────────────────────────────────────────────────────────────────────────
# PitAgentCFG
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PitAgentCFG:
    """Runtime configuration for the Pit Strategy Agent.

    Loads three N15 quantile regressors (physical stop P05/P50/P95), the N16
    undercut LightGBM + Platt calibrator, and both JSON configs. All loaded with
    joblib — required on Windows paths with non-ASCII characters.

    circuit_traversal maps GP event name → pit lane traversal time (s). Subtracting
    it from total pit time yields physical_stop_est, the N15 target variable.

    team_encoder is a sklearn LabelEncoder reconstructed from the class list in
    model_config.json — no separate .pkl needed.

    team_year_median_fallback (2.8 s): N15 never exported its per-team×year medians, so
    this constant used to stand in for EVERY call regardless of team or year (#450).
    __post_init__ now aggregates the real medians from raw pit data
    (_load_team_year_medians); this constant only backstops a (team, year) combo with
    no raw data (e.g. a team's rookie season). It sits inside N15's [2.0, 4.5 s]
    training range so an unseen combo still lands in-distribution. It is NOT the
    centre of that range, which would be 3.25 s: real stop times are skewed toward
    the fast end, so 2.8 is nearer the typical stop than the midpoint is.

    circuit_undercut_rate and team_x_undercut_rate are pre-aggregated at startup
    from undercut_clean.parquet so tool calls are stateless.

    Attributes:
        model_name: LM Studio model identifier for the ReAct agent LLM.
        team_year_median_fallback: Global fallback for team_year_median feature (s).
    """

    model_name: str = 'gpt-4.1-mini'
    team_year_median_fallback: float = 2.8

    def __post_init__(self) -> None:
        self.export_dir = _AGENTS
        try:
            self.export_dir.mkdir(parents=True, exist_ok=True)
        except OSError:
            pass  # read-only mount in Docker

        _pit = _MODELS / 'pit_prediction'

        # N15: physical stop duration models
        self.pit_p05_model = joblib.load(_pit / 'hist_pit_p05_v1.pkl')
        self.pit_p50_model = joblib.load(_pit / 'hist_pit_p50_v1.pkl')
        self.pit_p95_model = joblib.load(_pit / 'hist_pit_p95_v1.pkl')

        with open(_pit / 'model_config.json') as f:
            pit_cfg = json.load(f)

        self.pit_features: list[str]   = pit_cfg['features']
        # Re-key the circuit tables into the SLUG keyspace the agents actually query
        # with (session_meta.gp_name / the featured parquet's GP_Name). They ship keyed
        # by FastF1 full event names, whose overlap with the slugs is exactly zero — so
        # every lookup missed and silently took its default, freezing traversal at
        # 20.0 s and circuit_undercut_rate at 0.38 for every circuit (#448).
        self.circuit_traversal: dict   = rekey_by_slug(
            pit_cfg['circuit_traversal_lookup'], 'circuit_traversal_lookup',
        )

        # Reconstruct LabelEncoder from saved class list
        le = LabelEncoder()
        le.classes_ = np.array(pit_cfg['label_encoder_classes']['team'])
        self.team_encoder: LabelEncoder = le

        # Real per-(team, year) physical-stop medians, aggregated from raw pit data now
        # that circuit_traversal is available (needed to isolate the physical-stop
        # component from total pit-lane time, same decomposition N15 was trained on).
        # See team_year_median_fallback's docstring above and #450.
        self.team_year_median: dict[tuple[str, int], float] = self._load_team_year_medians()

        # N16: undercut classifier + calibrator
        self.undercut_model      = joblib.load(_pit / 'lgbm_undercut_v1.pkl')
        self.undercut_calibrator = joblib.load(_pit / 'calibrator_undercut_v1.pkl')

        with open(_pit / 'model_config_undercut_v1.json') as f:
            uc_cfg = json.load(f)

        self.undercut_features: list[str] = uc_cfg['features']
        self.undercut_threshold: float    = uc_cfg['best_threshold']
        self.dry_compounds: list[str]     = uc_cfg['dry_compounds']

        # Pre-aggregate circuit and team undercut rates from N16 training parquet.
        # This parquet is keyed by full event name; re-key to slugs for the same
        # reason as circuit_traversal above (#448).
        _uc = pd.read_parquet(_PROCESSED / 'undercut_labeled' / 'undercut_clean.parquet')
        self.circuit_undercut_rate: dict = rekey_by_slug(
            _uc.groupby('GP_Name')['undercut_success'].mean().to_dict(),
            'circuit_undercut_rate',
        )
        self.team_x_undercut_rate: dict = (
            _uc.groupby('Team_X')['undercut_success'].mean().to_dict()
        )

    def _load_team_year_medians(self) -> dict[tuple[str, int], float]:
        """Aggregate real per-(team, year) physical-stop medians from raw pit data.

        Mirrors N15's own team_year_median feature (N15_pit_duration.ipynb Step 2,
        add_team_year_median): physical_stop_est = pit_duration_s - circuit_traversal,
        grouped by (team, year) and reduced to the median. That table was never
        exported from the notebook, so inference fed every call the SAME 2.8s
        constant no matter which team or year was asked about (#450) — this rebuilds
        it from data/raw/<year>/<GP>/pitstops.parquet instead of guessing.

        Requires self.circuit_traversal to already be populated (called after that
        assignment in __post_init__): each GP's traversal is needed to isolate the
        physical-stop component from the raw pit-lane time, the same decomposition
        N15 was trained on.

        Returns:
            Dict keyed by (team name normalised through _TEAM_ALIASES, year) → median
            physical_stop_est in seconds. A combo absent here falls back to
            team_year_median_fallback at read time (team_year_median_for).
        """
        records: list[dict] = []
        for gp_dir in sorted((_DATA_ROOT / 'raw').glob('*/*')):
            pit_path = gp_dir / 'pitstops.parquet'
            if not pit_path.exists():
                continue
            try:
                year = int(gp_dir.parent.name)
            except ValueError:
                continue

            gp_slug   = canonical_gp_name(gp_dir.name)
            traversal = self.circuit_traversal.get(gp_slug, 20.0)
            try:
                stops = _extract_physical_stops(pit_path, traversal)
            except Exception:
                logger.warning('team_year_median: failed to parse %s; skipping this GP', pit_path)
                continue

            for team_raw, physical_stop_est in stops:
                team = _TEAM_ALIASES.get(team_raw, team_raw)
                records.append({'team': team, 'year': year, 'physical_stop_est': physical_stop_est})

        if not records:
            return {}
        medians = (
            pd.DataFrame.from_records(records)
            .groupby(['team', 'year'])['physical_stop_est']
            .median()
        )
        return {key: float(value) for key, value in medians.items()}

    def traversal_for(self, gp_name: str) -> float:
        """Pit-lane traversal (s) for a GP given in EITHER keyspace.

        The two callers disagree, exactly as they do in race_situation_agent.sc_rate_for:
        the FastF1 session path passes a full event name ('Qatar Grand Prix'), the replay
        path passes the parquet slug ('Lusail'). circuit_traversal is keyed by slug (#448
        rekeyed it), so the full name missed and every FastF1-path circuit fell back to
        the 20.0 default. slug_from_event_name is reentrant and normalises both.
        """
        slug = slug_from_event_name(gp_name) or gp_name
        return self.circuit_traversal.get(slug, 20.0)

    def team_year_median_for(self, team: str, year: int) -> float:
        """Per-(team, year) physical-stop median (s), or the global fallback if unseen.

        Args:
            team: Team name already normalised through _TEAM_ALIASES — callers must
                apply that alias BEFORE this lookup so a 2025 'Racing Bulls' row
                lands on the same key as the '2023/2024 'RB' rows it aggregated with.
            year: Race year.

        Returns:
            Median physical_stop_est (s) for that team/year, or
            team_year_median_fallback (2.8 s) when the combo has no raw pit data.
        """
        return self.team_year_median.get((team, year), self.team_year_median_fallback)

    def undercut_rate_for(self, gp_name: str) -> float:
        """Circuit undercut base rate for a GP in either keyspace (an N16 feature).

        Same dual-keyspace resolution as traversal_for. Without it the FastF1 path fed
        N16 a constant 0.38 on every circuit, which is the #448 dead-feature bug moved to
        the sibling path rather than fixed.
        """
        slug = slug_from_event_name(gp_name) or gp_name
        return self.circuit_undercut_rate.get(slug, 0.38)


# ─────────────────────────────────────────────────────────────────────────────
# PitStrategyOutput
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PitStrategyOutput:
    """Structured output of the Pit Strategy Agent for one driver at one lap.

    Attributes:
        action: Strategy recommendation — PIT_NOW, STAY_OUT, UNDERCUT, OVERCUT,
            or REACTIVE_SC (box under an active/imminent Safety Car).
        recommended_lap: Lap on which the stop is recommended. None when STAY_OUT.
        compound_recommendation: SOFT, MEDIUM, or HARD for the next stint.
        stop_duration_p05: 5th-percentile physical stop time (s) from N15.
        stop_duration_p50: Median physical stop time (s). Used for timing decisions.
        stop_duration_p95: 95th-percentile physical stop time (s). Pessimistic bound.
        undercut_prob: Calibrated N16 P(undercut success). None when no dry rival
            is within 5 positions or conditions are wet.
        undercut_target: Driver abbreviation of the undercut target. None when
            undercut_prob is None.
        sc_reactive: True when recommendation was triggered by SC probability from
            N27, not by tyre degradation or position logic alone.
        reasoning: LLM synthesis explaining the recommendation.
    """

    action: str
    recommended_lap: Optional[int]
    compound_recommendation: str
    stop_duration_p05: float
    stop_duration_p50: float
    stop_duration_p95: float
    undercut_prob: Optional[float]
    undercut_target: Optional[str]
    sc_reactive: bool
    reasoning: str


# ─────────────────────────────────────────────────────────────────────────────
# Pure stateless helpers — accept all state as arguments, read no globals
# ─────────────────────────────────────────────────────────────────────────────

def _compound_to_id(compound: str, gp_name: str, year: int) -> int:
    """Convert SOFT/MEDIUM/HARD to Pirelli compound number (C1–C5/C6).

    Uses the nested TIRE_COMPOUNDS structure {year_str: {gp_name: {compound: Cx}}}.
    The Cx string (e.g. 'C3') is stripped to its integer (3). Falls back to
    _COMPOUND_FALLBACK when the circuit/year is not in the map.

    Args:
        compound: Compound name string (e.g. 'SOFT', 'MEDIUM', 'HARD').
        gp_name: GP name matching the TIRE_COMPOUNDS keys.
        year: Race year as int.

    Returns:
        Integer compound number (1–6).
    """
    cx_str = TIRE_COMPOUNDS.get(str(year), {}).get(gp_name, {}).get(compound.upper(), '')
    if cx_str and cx_str.startswith('C') and cx_str[1:].isdigit():
        return int(cx_str[1:])
    return _COMPOUND_FALLBACK.get(compound.upper(), 3)


def _extract_physical_stops(pit_path: Path, traversal_s: float) -> list[tuple[str, float]]:
    """Reconstruct (team, physical_stop_est) pairs from one raw pitstops.parquet.

    Mirrors N15_pit_duration.ipynb's collect_pit_data: the raw file holds one row per
    stop with PitInTime set (the in-lap) and a second row on the NEXT lap with
    PitOutTime set (the out-lap) for the same driver. Matching them by
    (Driver, LapNumber + 1) and subtracting timestamps gives total pit-lane time;
    subtracting traversal_s (this circuit's fixed traversal component) isolates the
    physical-stop component the same way N15's target was built. The TyreLife_out <= 5
    filter drops drive-through penalties and jack failures where no tyre was actually
    changed — those carry no team-quality signal and were excluded from training too.

    Args:
        pit_path: Path to one GP's raw pitstops.parquet.
        traversal_s: This circuit's pit-lane traversal time (s), already resolved from
            PitAgentCFG.circuit_traversal.

    Returns:
        List of (team_name, physical_stop_est_seconds) tuples, one per genuine,
        in-training-scope stop ([_NORMAL_STOP_MIN_S, _NORMAL_STOP_MAX_S]).
    """
    cols = ['Driver', 'Team', 'LapNumber', 'PitInTime', 'PitOutTime', 'TyreLife']
    laps = pd.read_parquet(pit_path, columns=cols)

    pit_in  = laps.loc[laps['PitInTime'].notna(), ['Driver', 'Team', 'LapNumber', 'PitInTime']].copy()
    pit_out = laps.loc[laps['PitOutTime'].notna(), ['Driver', 'LapNumber', 'PitOutTime', 'TyreLife']].copy()
    pit_in['LapNumber_out'] = pit_in['LapNumber'] + 1

    merged = pit_in.merge(
        pit_out.rename(columns={'LapNumber': 'LapNumber_out', 'TyreLife': 'TyreLife_out'}),
        on=['Driver', 'LapNumber_out'], how='inner',
    )
    if merged.empty:
        return []

    merged = merged[merged['TyreLife_out'] <= 5]
    if merged.empty:
        return []

    pit_duration_s    = (merged['PitOutTime'] - merged['PitInTime']).dt.total_seconds()
    physical_stop_est = (pit_duration_s - traversal_s).clip(lower=0.5)
    in_scope          = physical_stop_est.between(_NORMAL_STOP_MIN_S, _NORMAL_STOP_MAX_S)

    return list(zip(merged.loc[in_scope, 'Team'], physical_stop_est[in_scope]))


def _parse_tool_outputs(messages: list) -> dict:
    """Extract numeric values from tool call results in the agent message history.

    Scans ToolMessage content for structured strings from each tool. Values
    default to None when the corresponding tool was not called.

    The ReAct agent calls score_undercut_tool once PER candidate rival (system prompt
    rule 2: "for each rival within 5 positions ahead"), so several undercut results can
    appear across the message history. Only latching onto the FIRST P(undercut_success)
    match — and never onto WHICH driver it was scored against — meant undercut_target
    was assembled from the positional rival_ahead candidate instead of the driver N16
    actually scored highest, silently mismatching prob and target on every multi-rival
    call (#432). Tracking the argmax (driver, prob) pair across ALL score_undercut_tool
    calls fixes this; score_undercut_tool embeds the driver_y it scored (`candidate=...`)
    precisely so this function has something to key the argmax on.

    Args:
        messages: LangChain message objects from the agent's invoke result.

    Returns:
        Dict with: stop_duration_p05/p50/p95, undercut_prob, undercut_target,
        compound_recommendation. Missing keys default to None. undercut_target is
        the driver whose call produced the highest undercut_prob seen, NOT yet
        validated against the live-drivers roster — callers must validate before
        using it (see PitStrategyAgent._validate_undercut_target).
    """
    result: dict = {
        'stop_duration_p05':       None,
        'stop_duration_p50':       None,
        'stop_duration_p95':       None,
        'undercut_prob':           None,
        'undercut_target':         None,
        'compound_recommendation': None,
    }
    for msg in messages:
        content = getattr(msg, 'content', '')
        if not isinstance(content, str):
            continue
        m = re.search(r'P05=(\d+(?:\.\d+)?)s.*?P50=(\d+(?:\.\d+)?)s.*?P95=(\d+(?:\.\d+)?)s', content)
        if m:
            result['stop_duration_p05'] = float(m.group(1))
            result['stop_duration_p50'] = float(m.group(2))
            result['stop_duration_p95'] = float(m.group(3))
        m = re.search(r'candidate=(\w+)\s*\|\s*P\(undercut_success\)=(\d+(?:\.\d+)?)', content)
        if m:
            candidate_driver = m.group(1)
            candidate_prob   = float(m.group(2))
            if result['undercut_prob'] is None or candidate_prob > result['undercut_prob']:
                result['undercut_prob']   = candidate_prob
                result['undercut_target'] = candidate_driver
        m = re.search(r'Recommended:\s*(SOFT|MEDIUM|HARD)', content)
        if m:
            result['compound_recommendation'] = m.group(1)
    return result


def _parse_agent_summary(final_content: str) -> tuple[str, str, str]:
    """Extract ACTION, COMPOUND, and REASONING from the agent's structured summary.

    Falls back to ('STAY_OUT', 'MEDIUM', first 200 chars) when the structured
    block is absent — prevents crashes when the LLM deviates from format.

    Args:
        final_content: Last message content string from the agent.

    Returns:
        Tuple of (action, compound, reasoning) strings.
    """
    action   = re.search(r'ACTION:\s*(PIT_NOW|STAY_OUT|UNDERCUT|OVERCUT|REACTIVE_SC)', final_content)
    compound = re.search(r'COMPOUND:\s*(SOFT|MEDIUM|HARD)', final_content)
    reason   = re.search(r'REASONING:\s*(.+)', final_content)
    return (
        action.group(1)         if action   else 'STAY_OUT',
        compound.group(1)       if compound else 'MEDIUM',
        reason.group(1).strip() if reason   else final_content[:200],
    )


def _build_pit_prompt(
    driver: str,
    lap_number: int,
    tyre_life: int,
    compound: str,
    team: str,
    position: int,
    rival_str: str,
    sc_prob: float,
    laps_to_cliff_p10: float,
    sc_currently_active: bool = False,
    vsc_active: bool = False,
) -> str:
    """Build the natural-language prompt for the Pit Strategy ReAct agent.

    Args:
        driver: FastF1 driver abbreviation.
        lap_number: Current lap number.
        tyre_life: Laps completed on current tyre set.
        compound: Compound currently fitted.
        team: Team name string.
        position: Current race position.
        rival_str: Description of the nearest rival ahead.
        sc_prob: Safety Car probability from N27.
        laps_to_cliff_p10: P10 laps-to-cliff from N26 TireOutput.
        sc_currently_active: True when an RCM confirms a neutralisation (SC OR VSC)
            is deployed right now.  Replaces the legacy "SC probability" line with an
            unambiguous "SC STATUS: ... DEPLOYED RIGHT NOW" banner so the model cannot
            mistake an active neutralisation for a low future-SC probability (the
            historical N28 failure mode at Catar 2025 V7).
        vsc_active: True when that neutralisation is specifically a Virtual Safety Car
            (Art. 56). The field is not queued and the restart is instant, so a stop
            saves materially less than under a full SC; the banner says so, so the model
            does not treat a VSC stop as the near-free stop a full SC offers (#471).

    Returns:
        Formatted prompt string for the ReAct agent.
    """
    if sc_currently_active and vsc_active:
        sc_line = (
            "SC STATUS: VIRTUAL SAFETY CAR DEPLOYED RIGHT NOW (confirmed by RCM). "
            "Gaps are preserved and the restart is instant (Art. 56), so a stop saves "
            "materially less time than under a full Safety Car.\n"
        )
    elif sc_currently_active:
        sc_line = "SC STATUS: SAFETY CAR DEPLOYED RIGHT NOW (confirmed by RCM).\n"
    else:
        sc_line = f"SC probability (next 3 laps): {sc_prob:.2f}\n"
    return (
        f'Driver {driver} | Lap {lap_number} | P{position} | '
        f'Tyre: {compound} (life {tyre_life} laps)\n'
        f'Rival ahead: {rival_str}\n'
        f'{sc_line}'
        f'Laps to tyre cliff P10: {laps_to_cliff_p10:.1f}\n\n'
        f'Team: {team}\n\n'
        'Analyse the pit stop window. Call predict_pit_duration_tool, '
        'score_undercut_tool, and recommend_compound_tool. '
        'Then give a final recommendation.'
    )


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
# System prompt (module-level constant — unchanged from N28)
# ─────────────────────────────────────────────────────────────────────────────

_PIT_STRATEGY_SYSTEM_PROMPT = """You are the Pit Strategy Agent for an F1 race.
Your job is to decide whether a driver should pit now, and if so, what compound to fit.

You have three tools:
- predict_pit_duration_tool: estimates physical stop duration (P05/P50/P95) and total pit lane time.
- score_undercut_tool: scores the probability that pitting before a specific rival earns track position.
- recommend_compound_tool: recommends the optimal next compound based on remaining laps and FIA rules.

Decision rules:
1. Always call predict_pit_duration_tool first to know the stop cost.
2. Call score_undercut_tool for each rival within 5 positions ahead.
3. Call recommend_compound_tool to determine the optimal compound.
4. If P(undercut_success) >= 0.522 for any rival → recommend UNDERCUT.
5. If sc_prob context is provided and >= 0.30 → recommend REACTIVE_SC pit.
6. If tyre_cliff context is provided and laps_to_cliff <= 3 → recommend PIT_NOW.
7. Otherwise → STAY_OUT unless a clear strategic window exists.

## Strategic guard-rails (HARD constraints — override any decision rule above)

PIT WINDOW — early race:
  NEVER recommend PIT_NOW, UNDERCUT, or OVERCUT before lap 5 of the race.
  Exception: Safety Car IS currently deployed (prompt shows "SC STATUS: SAFETY
  CAR DEPLOYED RIGHT NOW"), or radio confirms damage/puncture/mechanical failure.
  Rationale: no tyre degrades enough in 1-4 laps to justify a stop; the pit lane time cost
  (~22-25s) is unrecoverable this early. Force STAY_OUT and note "pit window not open yet"
  in your reasoning.

PIT WINDOW — end of race:
  NEVER recommend PIT_NOW, UNDERCUT, or OVERCUT when remaining laps <= 3.
  Exception: tyre failure is imminent (laps_to_cliff P10 < 2) or Safety Car deployed.
  Rationale: a pit stop costs ~22-25s; with ≤3 laps, fresh tyres recover at most ~1.5s
  total. Net loss ≈ 20s = ~13 positions.

MINIMUM STINT LENGTH before a pit makes sense:
  SOFT: current tyre_life must be >= 8 laps before recommending a stop.
  MEDIUM: >= 12 laps.  HARD: >= 15 laps.
  If the driver has NOT completed the minimum stint, recommend STAY_OUT (the current
  set still has useful life; pitting now wastes a tyre allocation).

  EXCEPTION — SC ACTIVE: when the prompt states "SC STATUS: SAFETY CAR DEPLOYED
  RIGHT NOW", the minimum stint constraint DOES NOT APPLY: a stop under a deployed
  SC is far cheaper because the field is delta-limited and queued behind the SC, so
  your RELATIVE loss shrinks.
  This makes pitting cheaper. It does NOT make it correct: weigh it against what a
  stop surrenders. Stay out when you have already stopped (a second set buys you
  nothing), when you lead a pack that still has to stop, when you would rejoin into
  traffic, or when the race is ending — if the SC is still out on the last lap the
  race finishes behind it with no overtaking (Art. 55.17), so track position given
  away now cannot be won back. Box when you have yet to stop and must anyway, when
  the tyres are near the cliff, or when the two-compound rule is still unsatisfied
  and time is running out. Decide on the race state, not on the SC alone.

COMPOUND vs REMAINING LAPS:
  SOFT: recommend only if remaining laps <= 15 (it won't last longer).
  MEDIUM: suitable for 12-30 remaining laps.
  HARD: suitable for 20+ remaining laps.
  Picking SOFT with 25 laps to go forces an extra pit stop — factor that cost in.

REACTIVE_SC usage:
  REACTIVE_SC is for the rare in-between case where sc_prob is elevated but the
  Safety Car is NOT yet deployed.  When the prompt states "SC STATUS: SAFETY CAR
  DEPLOYED RIGHT NOW", prefer PIT_NOW directly.  Use REACTIVE_SC only when
  sc_prob >= 0.30 AND the prompt shows the legacy "SC probability" line.  A high
  sc_prob without confirmation is still a contingency — mention it in reasoning
  and set ACTION to STAY_OUT unless the SC is actually out.

Always end your response with a structured summary:
ACTION: <PIT_NOW|STAY_OUT|UNDERCUT|OVERCUT|REACTIVE_SC>
COMPOUND: <SOFT|MEDIUM|HARD>
REASONING: <one sentence>
"""


# ─────────────────────────────────────────────────────────────────────────────
# PitStrategyAgent — encapsulated agent class
# ─────────────────────────────────────────────────────────────────────────────

class PitStrategyAgent:
    """Encapsulated Pit Strategy Agent backed by N15/N16 models and LangGraph ReAct.

    Owns all mutable state previously held in module-level globals:
    - laps_df / session_meta: set per call by run() / run_from_state()
    - cfg: PitAgentCFG with all models and lookups (lazily loaded on first instantiation)
    - _react_agent: lazily created LangGraph CompiledGraph

    LangChain tools are built as closures inside _build_tools() so they read
    instance attributes without depending on any module-level globals.
    """

    def __init__(self) -> None:
        self.cfg: PitAgentCFG      = PitAgentCFG()
        self.laps_df: pd.DataFrame = pd.DataFrame()
        self.session_meta: dict    = {}
        # The cars on track this lap, or None when we do not know. This agent is a module
        # singleton (`_get_default_pit_agent`), so it MUST be initialised here and reset
        # on every entry point: left over from a previous call it would validate targets
        # against another race's roster, refusing live cars or admitting dead ones. None
        # means "unknown", which disables the guard rather than rejecting everything.
        self._live_drivers: set | None = None
        self._react_agent          = None
        self._tools: list          = self._build_tools()

    # ── Encoding / lookup helpers ─────────────────────────────────────────────

    def _encode_team(self, team_raw: str) -> int:
        """Encode team name to integer using the N15 LabelEncoder in self.cfg.

        Applies _TEAM_ALIASES before encoding so FastF1 2025 variants resolve to
        training-time names. Unknown teams encode to 0 to avoid inference crashes.

        Args:
            team_raw: Raw team name string from FastF1 (e.g. 'Racing Bulls').

        Returns:
            Integer team encoding used by N15 HistGBT models.
        """
        team_str = _TEAM_ALIASES.get(team_raw, team_raw)
        try:
            return int(self.cfg.team_encoder.transform([team_str])[0])
        except ValueError:
            return 0

    def _compounds_are_dry(self, comp_x: str, comp_y: str) -> bool:
        """Return True only if both compounds are in cfg.dry_compounds.

        Args:
            comp_x: Compound string for the chasing driver (e.g. 'SOFT').
            comp_y: Compound string for the car ahead.

        Returns:
            True when both compounds are in N16's training scope.
        """
        return comp_x in self.cfg.dry_compounds and comp_y in self.cfg.dry_compounds

    # ── Lap data helpers ──────────────────────────────────────────────────────

    def _get_lap_row(self, driver: str, lap_number: int) -> Optional[pd.Series]:
        """Return the single self.laps_df row for driver at lap_number, or None if missing.

        When no exact lap match exists (common at lap 1 because FastF1 sometimes
        omits the opening lap row, or at the very end of a stint when the LLM
        ReAct loop reasons about a lap that has not been recorded yet), fall back
        to the most recent prior lap for the same driver. This keeps the
        feature-builder usable instead of forcing the caller to raise.

        Args:
            driver: FastF1 driver abbreviation.
            lap_number: Lap number as int.

        Returns:
            Pandas Series for that lap, the closest prior lap, or None.
        """
        # A car that is not on track this lap has no valid row, however many stale ones
        # it left behind. Without this, a driver who retired on lap 5 answers a lap-50
        # query with his lap-5 row (the prior-lap fallback below has no staleness bound,
        # and no bound works: a finisher's last featured lap can lag 20 while a retiree
        # shows at 9, #462). `_live_drivers` is the presence signal RaceStateManager
        # already supplies; None means we cannot tell, so the guard stays off then.
        live = getattr(self, '_live_drivers', None)
        if live is not None and driver not in live:
            return None

        driver_rows = self.laps_df[self.laps_df['Driver'] == driver]
        if driver_rows.empty:
            return None
        exact = driver_rows[driver_rows['LapNumber'] == lap_number]
        if not exact.empty:
            return exact.iloc[0]
        prior = driver_rows[driver_rows['LapNumber'] < lap_number]
        if not prior.empty:
            return prior.sort_values('LapNumber').iloc[-1]
        # No prior lap: the query is for a lap before this driver's first recorded one
        # (FastF1 sometimes omits lap 1). Return the earliest lap as a start-of-race
        # stand-in. This is not the retired-car case, which the presence guard above and
        # the prior-lap branch already handle.
        return driver_rows.sort_values('LapNumber').iloc[0]

    def _get_position_map(self, lap_number: int) -> dict[str, float]:
        """Return {driver: position} for all drivers at lap_number with valid position.

        Args:
            lap_number: Current lap number.

        Returns:
            Dict mapping driver abbreviation to race position float.
        """
        lap_data = self.laps_df[self.laps_df['LapNumber'] == lap_number][['Driver', 'Position']].dropna()
        return dict(zip(lap_data['Driver'], lap_data['Position']))

    # ── Feature builders ──────────────────────────────────────────────────────

    @staticmethod
    def _tyre_life_in(row) -> int:
        """TyreLife for N15, clipped to the trained ceiling, NaN-safe.

        `row` is a pandas Series, so `row.get('TyreLife', 1)` returns the STORED value
        including NaN — the default only fires when the COLUMN is absent, never when the
        VALUE is. `int(nan)` then raises ValueError inside the ReAct loop, and 451 rows
        (1.98%) of the 2025 featured parquet have a NaN TyreLife. That is the same
        Series.get trap as #428/#462, on a third column.

        A missing tyre age means "fresh" is the only defensible read: a car with no
        recorded TyreLife has just been given a set, and 1 is what the dead default was
        reaching for anyway. It is also the conservative direction for N15, which is
        predicting how long the stop takes, not whether to make it.
        """
        value = row.get('TyreLife')
        if value is None or pd.isna(value):
            logger.warning('TyreLife missing on the lap row; N15 reads it as a fresh set')
            return 1
        return min(int(value), _MAX_TRAINED_TYRE_LIFE)

    def _build_pit_duration_features(
        self,
        driver: str,
        lap_number: int,
        compound: str,
        compound_change: bool,
        under_sc: bool,
    ) -> pd.DataFrame:
        """Build the 9-feature input row for the N15 quantile regressors.

        Assembles features from self.laps_df, self.session_meta, and self.cfg lookups.
        tight_pit_box checks the circuit against N15's trained tight-pit-box set
        (Monaco/Singapore/Hungary); team_year_median looks up the real aggregated
        per-(team, year) median, falling back to cfg.team_year_median_fallback (2.8s)
        only for a combo with no raw pit data (#450 — both used to be frozen constants).

        Args:
            driver: FastF1 driver abbreviation.
            lap_number: Lap on which the stop would occur.
            compound: Compound being fitted ('SOFT', 'MEDIUM', 'HARD').
            compound_change: True if switching compound vs current.
            under_sc: True if Safety Car is active during the stop.

        Returns:
            Single-row DataFrame with 9 columns in cfg.pit_features order.

        Raises:
            ValueError: When no lap row exists for the driver at lap_number.
        """
        row = self._get_lap_row(driver, lap_number)
        if row is None:
            raise ValueError(f'No lap data for {driver} at lap {lap_number}')

        # gp_name IS needed here now, unlike #445's note about compound_id: tight_pit_box
        # is a circuit-membership check, not a Pirelli compound lookup, so it does not
        # share that concern.
        # (helper `_tyre_life_in` below reads TyreLife safely; see its docstring)
        year     = self.session_meta.get('year', 2025)
        team_raw = self.session_meta.get('team_lookup', {}).get(driver, 'Unknown')
        gp_name  = self.session_meta.get('gp_name', '')

        # gp_name can arrive as either a FastF1 full event name or a replay-parquet
        # slug (same dual-keyspace situation as circuit_traversal/circuit_undercut_rate);
        # resolve to the slug before checking membership (#450).
        gp_slug       = slug_from_event_name(gp_name) or gp_name
        tight_pit_box = gp_slug in _TIGHT_PIT_BOX_SLUGS

        # team_year_median_for needs the SAME normalised team string _encode_team uses
        # internally, so the lookup and the encoding never disagree on which team a
        # 2025 'Racing Bulls' row aggregated with (#450).
        team_str = _TEAM_ALIASES.get(team_raw, team_raw)

        feat = {
            'team':             self._encode_team(team_raw),
            'year':             year,
            # N15 clipped tyre_life at 50 in training (cell 11); pass the raw value and
            # an absurd stint extrapolates outside the fitted range (#450).
            #
            # `_tyre_life_in` rather than `min(int(row.get('TyreLife', 1)), 50)`: `row` is
            # a Series, so `.get(k, 1)` returns the STORED NaN and the `1` is dead code —
            # the default only fires when the COLUMN is missing, never the VALUE. `int(nan)`
            # then raises ValueError inside the ReAct loop, on 451 real rows (1.98%) of the
            # 2025 featured parquet.
            'tyre_life_in':     self._tyre_life_in(row),
            'lap_number':       lap_number,
            'compound_id':      _N15_COMPOUND_ORDER.get(compound.upper(), _N15_COMPOUND_UNKNOWN),
            'compound_change':  int(compound_change),
            'under_sc':         int(under_sc),
            'tight_pit_box':    int(tight_pit_box),
            'team_year_median': self.cfg.team_year_median_for(team_str, year),
        }
        return pd.DataFrame([feat])[self.cfg.pit_features]

    def _build_undercut_features(
        self,
        driver_x: str,
        driver_y: str,
        lap_number: int,
    ) -> Optional[pd.DataFrame]:
        """Build the 13-feature input row for the N16 undercut classifier.

        Returns None if either driver has no lap row at lap_number or if either
        compound is not dry (wet/intermediate conditions are out of N16's scope).

        pit_delta_X estimates total stop cost as circuit_traversal + 4.5 s,
        representing inlap + outlap minus two representative race laps.

        The 4.5 s is `_NORMAL_STOP_MAX_S`, the UPPER bound of N15's normal-stop
        window, not a median: a median over [2.0, 4.5] on a right-skewed stop-time
        distribution sits well below it. It is deliberately pessimistic, which
        biases every pit_delta_X-driven undercut probability toward "the undercut
        costs more than it does". Calling it a median hid that bias behind a word
        that implies the estimate is centred when it is not.

        Args:
            driver_x: FastF1 abbreviation of the driver considering pitting first.
            driver_y: FastF1 abbreviation of the rival to undercut.
            lap_number: Current lap number.

        Returns:
            Single-row DataFrame with 13 columns in cfg.undercut_features order,
            or None when preconditions are not met.
        """
        x_row = self._get_lap_row(driver_x, lap_number)
        y_row = self._get_lap_row(driver_y, lap_number)
        if x_row is None or y_row is None:
            return None

        comp_x = str(x_row.get('Compound', 'MEDIUM')).upper()
        comp_y = str(y_row.get('Compound', 'MEDIUM')).upper()
        if not self._compounds_are_dry(comp_x, comp_y):
            return None

        # A row with no Position cannot be scored. `pos_gap` is N16's single strongest
        # feature (gain 0.690) and is built from both positions, so a NaN there is not a
        # degraded prediction: LightGBM routes it down its missing branch and returns a
        # confident-looking number computed from nothing.
        #
        # The defaults below cannot catch this: `Series.get(k, default)` returns the
        # STORED value including NaN, and only falls back when the COLUMN is absent. So
        # a crashed car whose Position is NaN sails straight through. That is the #428
        # sentinel bug wearing a different column, and the rule is the same: refuse,
        # never default to a number the model will read as a real reading (#462).
        # Position AND TyreLife, on both cars. `TyreLife` feeds three of N16's features
        # (`tyre_life_diff`, `TyreLife_X`, `TyreLife_Y`) and its `.get(k, 10)` defaults are
        # the same dead code as the position ones: a Series returns the stored NaN, so the
        # 10 never fires. Measured: 561 of 79,032 raw laps carry a real Position and a NaN
        # TyreLife, so this is reachable for a car that IS racing, not just a retiree.
        missing = [
            name for name, value in (
                (f'{driver_x} position',  x_row.get('Position')),
                (f'{driver_y} position',  y_row.get('Position')),
                (f'{driver_x} tyre life', x_row.get('TyreLife')),
                (f'{driver_y} tyre life', y_row.get('TyreLife')),
            )
            if value is None or pd.isna(value)
        ]
        if missing:
            logger.warning(
                'Undercut features refused at lap %s: %s missing. N16 reads these as real '
                'values and LightGBM answers from its missing branch, so a default here is '
                'a fabricated reading',
                lap_number, ', '.join(missing),
            )
            return None

        # The position check above only catches a car whose last row happens to carry a
        # NaN. It misses the worse case: BEA retired on lap 41 at Lusail, and at lap 50
        # `_get_lap_row`'s unbounded prior-lap fallback returns his lap-41 row, complete,
        # with Position 19.0. The features then build with no NaN and no warning: a
        # confident undercut probability against a car in the garage, from 9-lap-stale
        # telemetry.
        #
        # No staleness threshold can separate the two. Measured across 2025: a car that
        # FINISHED can have its last known lap lag by 20 (RUS at Sakhir, because the
        # featured frame drops SC/pit/out laps), while this bug fires at 9. The ranges
        # overlap, so any bound that catches BEA also declares RUS retired.
        #
        # The only sound signal is presence: RaceStateManager builds `rivals` from the
        # per-lap rows, so a car that is gone is simply absent. `_live_drivers` carries
        # that set when we were run from a lap_state. When it is absent (direct calls,
        # tests) we cannot know, and fall through rather than guess.
        # BOTH drivers, not just the target. This guarded `driver_y` alone, which is the
        # incomplete-fix pattern in its purest form: the same argument, one call away,
        # left open. Both are LLM free text.
        #
        # And on the FEATURED frame it bites harder than the NaN check above can catch.
        # HUL crashed on lap 7 at Lusail; the featured frame drops that lap as inaccurate,
        # so his LAST row is lap 6 — a normal racing lap carrying Position 10.0. Ask about
        # him at lap 20 and `_get_lap_row`'s unbounded fallback returns it complete: no
        # NaN, no warning, a confident undercut built from 14-lap-stale telemetry. The
        # position check cannot see it; only presence can.
        live = getattr(self, '_live_drivers', None)
        if live is not None:
            gone = [d for d in (driver_x, driver_y) if d not in live]
            if gone:
                logger.warning(
                    'Undercut features refused: %s not on track at lap %s '
                    '(absent from the lap_state rivals)',
                    ' and '.join(gone), lap_number,
                )
                return None

        gp_name    = self.session_meta.get('gp_name', '')
        year       = self.session_meta.get('year', 2025)
        total_laps = self.session_meta.get('total_laps', 57)
        team_x_raw = self.session_meta.get('team_lookup', {}).get(driver_x, 'Unknown')
        team_x     = _TEAM_ALIASES.get(team_x_raw, team_x_raw)

        comp_x_id = _compound_to_id(comp_x, gp_name, year)
        comp_y_id = _compound_to_id(comp_y, gp_name, year)

        feat = {
            # N16 trained pos_gap as pos_X_before - pos_Y_before, so X (us, the
            # undercutter) being BEHIND Y is positive — that is the model's single
            # strongest feature (gain 0.690). The operands were reversed here, which
            # handed every X-behind-Y pair a negative value the model never saw for
            # that case. Keep this order aligned with N16 cell 41 (#444).
            'pos_gap':               float(x_row.get('Position', 10)) - float(y_row.get('Position', 9)),
            # Trained as the OFFSET between the two stops (lap_y - lap_x, 1..max_lap_gap),
            # not the race lap. Feeding lap_number put it 10-70x out of range, so LightGBM
            # clamped to the deepest bin on every call. We score the canonical undercut:
            # we box now, the rival responds on the next lap (#444).
            'Lap_gap':               _ASSUMED_UNDERCUT_LAP_GAP,
            'tyre_life_diff':        float(x_row.get('TyreLife', 10)) - float(y_row.get('TyreLife', 10)),
            'TyreLife_X':            float(x_row.get('TyreLife', 10)),
            'TyreLife_Y':            float(y_row.get('TyreLife', 10)),
            'compound_x_id':         comp_x_id,
            'compound_y_id':         comp_y_id,
            'compound_delta':        comp_x_id - comp_y_id,
            'pit_delta_X':           self.cfg.traversal_for(gp_name) + 4.5,
            'lap_race_pct':          lap_number / total_laps,
            'pos_X_before':          float(x_row.get('Position', 10)),
            'circuit_undercut_rate': self.cfg.undercut_rate_for(gp_name),
            'team_x_undercut_rate':  self.cfg.team_x_undercut_rate.get(team_x, 0.38),
        }
        return pd.DataFrame([feat])[self.cfg.undercut_features]

    def _get_undercut_candidates(
        self,
        driver: str,
        lap_number: int,
        max_pos_gap: int = 5,
    ) -> list[str]:
        """Return drivers strictly ahead of driver within max_pos_gap positions.

        Result is sorted ascending by position so the immediate car ahead is first —
        the most likely undercut target.

        Args:
            driver: FastF1 driver abbreviation.
            lap_number: Current lap number.
            max_pos_gap: Maximum position gap to include as a candidate.

        Returns:
            List of driver abbreviations sorted by position (closest first).
        """
        pos_map = self._get_position_map(lap_number)
        my_pos  = pos_map.get(driver)
        if my_pos is None:
            return []
        candidates = [
            d for d, p in pos_map.items()
            if d != driver and (my_pos - max_pos_gap) <= p < my_pos
        ]
        return sorted(candidates, key=lambda d: pos_map[d])

    # ── LangChain tool factory ────────────────────────────────────────────────

    def _build_tools(self) -> list:
        """Build LangChain tools as closures over this PitStrategyAgent instance.

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
        def predict_pit_duration_tool(
            driver: str,
            lap_number: int,
            compound: str,
            compound_change: bool,
            under_sc: bool,
        ) -> str:
            """Predict physical pit stop duration (P05/P50/P95) and total pit lane time.

            Builds the 9 N15 features, runs the three HistGBT quantile regressors,
            adds circuit traversal time from cfg.

            Args:
                driver: FastF1 driver abbreviation (e.g. 'NOR').
                lap_number: Lap on which the stop would occur.
                compound: Compound being fitted ('SOFT', 'MEDIUM', 'HARD').
                compound_change: True if switching from current compound.
                under_sc: True if Safety Car is active during the stop. Cross-checked
                    against the RCM-confirmed ground truth when a real run set it
                    (see PitStrategyAgent._run_core); an LLM guess never overrides a
                    known fact (#476).

            Returns:
                "physical_stop: P05={p05:.2f}s | P50={p50:.2f}s | P95={p95:.2f}s | total_pit_P50={total:.2f}s (traversal={traversal:.2f}s)"
            """
            # driver is LLM free text, exactly like score_undercut_tool's driver_y —
            # replicate that tool's REFUSED shape instead of computing on a car that
            # is not (or no longer) on track (#476).
            live = getattr(agent, '_live_drivers', None)
            if live is not None and driver not in live:
                return (
                    f'Pit duration prediction REFUSED — {driver} is not on track at lap '
                    f'{lap_number}. Cars in the race this lap: {", ".join(sorted(live))}. '
                    f'Pick a driver from that list.'
                )

            # under_sc is also LLM free text. self.sc_currently_active (set only inside
            # _run_core, for the duration of one real invocation) is the RCM-confirmed
            # ground truth for THIS lap; when it disagrees with what the LLM supplied,
            # trust the confirmed state rather than the guess. getattr(..., None) keeps
            # direct/test calls of this tool (outside _run_core) unaffected — the
            # cross-check only fires when the ground truth is actually reachable.
            known_sc = getattr(agent, 'sc_currently_active', None)
            if known_sc is not None and bool(under_sc) != known_sc:
                logger.warning(
                    'predict_pit_duration_tool: LLM-supplied under_sc=%s contradicts the '
                    'RCM-confirmed sc_currently_active=%s at lap %s; using the RCM value (#476)',
                    under_sc, known_sc, lap_number,
                )
                under_sc = known_sc

            feat_df  = agent._build_pit_duration_features(driver, lap_number, compound, compound_change, under_sc)
            p05      = float(agent.cfg.pit_p05_model.predict(feat_df)[0])
            p50      = float(agent.cfg.pit_p50_model.predict(feat_df)[0])
            p95      = float(agent.cfg.pit_p95_model.predict(feat_df)[0])
            gp_name  = agent.session_meta.get('gp_name', '')
            traversal = agent.cfg.traversal_for(gp_name)
            return (
                f'physical_stop: P05={p05:.2f}s | P50={p50:.2f}s | P95={p95:.2f}s | '
                f'total_pit_P50={p50 + traversal:.2f}s (traversal={traversal:.2f}s)'
            )

        @lc_tool
        def score_undercut_tool(driver_x: str, driver_y: str, lap_number: int) -> str:
            """Score the probability that driver_x successfully undercuts driver_y.

            Builds the 13 N16 features and applies LightGBM + Platt calibration.
            Returns early when conditions fall outside N16's training scope.

            Args:
                driver_x: FastF1 abbreviation of the driver considering pitting first.
                driver_y: FastF1 abbreviation of the rival to undercut.
                lap_number: Current lap number.

            Returns:
                "candidate={driver_y} | P(undercut_success)={prob:.3f} | threshold={threshold} | pos_gap={gap:.0f} | tyre_life_diff={diff:+.0f} laps | verdict={YES/NO}"
                The leading candidate=<driver_y> is what lets _parse_tool_outputs track
                the argmax across multiple rivals scored in one run (#432) — every
                other tool output format is unaffected.
            """
            # The LLM picks driver_y as free text, so it can name any car on the grid,
            # including one that is no longer in the race. Answer with an error rather
            # than a probability: a number here reads as a finding, and the model has
            # no way to tell an invented target from a real one.
            live = getattr(agent, '_live_drivers', None)
            if live is not None and driver_y not in live:
                return (
                    f'Undercut scoring REFUSED — {driver_y} is not on track at lap '
                    f'{lap_number}. Cars in the race this lap: {", ".join(sorted(live))}. '
                    f'Pick a target from that list.'
                )

            feat_df = agent._build_undercut_features(driver_x, driver_y, lap_number)
            if feat_df is None:
                return (
                    f'Undercut scoring N/A — wet compound or missing lap data '
                    f'for {driver_x}/{driver_y} at lap {lap_number}'
                )

            raw_proba   = agent.cfg.undercut_model.predict_proba(feat_df)[:, 1]
            calib_proba = agent.cfg.undercut_calibrator.predict_proba(raw_proba.reshape(-1, 1))[:, 1][0]

            verdict   = 'YES' if calib_proba >= agent.cfg.undercut_threshold else 'NO'
            pos_gap   = feat_df['pos_gap'].iloc[0]
            tyre_diff = feat_df['tyre_life_diff'].iloc[0]

            return (
                f'candidate={driver_y} | P(undercut_success)={calib_proba:.3f} | '
                f'threshold={agent.cfg.undercut_threshold} | pos_gap={pos_gap:.0f} | '
                f'tyre_life_diff={tyre_diff:+.0f} laps | verdict={verdict}'
            )

        @lc_tool
        def recommend_compound_tool(
            driver: str,
            lap_number: int,
            current_compound: str,
            laps_to_cliff: Optional[float] = None,
        ) -> str:
            """Recommend the optimal next compound using N26 cliff signal or Pirelli stint windows.

            Priority 1 — laps_to_cliff from N26 TireOutput: drives compound choice
            directly when available. Priority 2 — Pirelli average stint capacities
            (SOFT ~18 laps, MEDIUM ~30 laps, HARD ~38 laps) as fallback.
            FIA mandatory two-compound rule is always applied.

            Args:
                driver: FastF1 driver abbreviation.
                lap_number: Current lap number.
                current_compound: Compound currently fitted.
                laps_to_cliff: P10 laps-to-cliff from N26 TireOutput (optional).

            Returns:
                "Recommended: {compound} | {strategy} | {urgency} | laps_remaining={n} | current={current} | source={source}"
            """
            # driver is LLM free text here too, exactly like the other two tools; guard
            # it the same way even though the computation below is driver-agnostic
            # today (only lap_number/current_compound/laps_to_cliff feed it) — a caller
            # that later wires in per-driver state must not inherit an unguarded arg (#476).
            live = getattr(agent, '_live_drivers', None)
            if live is not None and driver not in live:
                return (
                    f'Compound recommendation REFUSED — {driver} is not on track at lap '
                    f'{lap_number}. Cars in the race this lap: {", ".join(sorted(live))}. '
                    f'Pick a driver from that list.'
                )

            total_laps       = agent.session_meta.get('total_laps', 57)
            laps_remaining   = total_laps - lap_number
            current_compound = current_compound.upper()

            must_differ = current_compound in agent.cfg.dry_compounds
            valid       = (
                [c for c in agent.cfg.dry_compounds if c != current_compound]
                if must_differ else list(agent.cfg.dry_compounds)
            )
            candidates  = sorted(valid, key=lambda c: _STINT_CAPACITY_LAPS[c])

            if laps_to_cliff is not None:
                source  = 'N26_laps_to_cliff'
                urgency = (
                    'CLIFF_IMMINENT' if laps_to_cliff <= CLIFF_IMMINENT_LAPS
                    else 'CLIFF_SOON' if laps_to_cliff <= CLIFF_SOON_LAPS
                    else 'PLANNED'
                )
            else:
                source  = 'Pirelli_stint_windows'
                urgency = 'PLANNED'

            recommendation = next(
                (c for c in candidates if _STINT_CAPACITY_LAPS[c] >= laps_remaining),
                candidates[-1],
            )

            one_stop_viable = _STINT_CAPACITY_LAPS.get(recommendation, 30) >= laps_remaining
            strategy        = '1-stop viable' if one_stop_viable else '2-stop likely'
            cliff_str       = f' | laps_to_cliff={laps_to_cliff:.1f}' if laps_to_cliff is not None else ''

            return (
                f'Recommended: {recommendation} | {strategy} | {urgency} | '
                f'laps_remaining={laps_remaining} | current={current_compound} | '
                f'source={source}{cliff_str}'
            )

        return [predict_pit_duration_tool, score_undercut_tool, recommend_compound_tool]

    # ── LangGraph agent (lazy) ────────────────────────────────────────────────

    def get_react_agent(
        self,
        provider: str = None,
        model_name: str = 'gpt-4.1-mini',
        base_url: str = 'http://localhost:1234/v1',
        api_key: str = 'lm-studio',
    ):
        """Return the LangGraph ReAct agent, creating it on the first call (lazy).

        parallel_tool_calls is disabled via model_kwargs to avoid a Jinja NullValue
        rendering error in LM Studio when the agent sends tool results back.
        disable_streaming avoids partial-chunk issues with local models.

        Args:
            provider: 'lmstudio' (default) or 'openai'.
            model_name: Model identifier for ChatOpenAI.
            base_url: Base URL for LM Studio (ignored when provider='openai').
            api_key: API key; 'lm-studio' for local server.

        Returns:
            LangGraph CompiledGraph.

        Raises:
            ImportError: When LangGraph / LangChain are not installed.
        """
        if not _LANGGRAPH_AVAILABLE:
            raise ImportError('LangGraph / LangChain not installed.')

        if self._react_agent is not None:
            return self._react_agent

        import os
        if provider is None:
            provider = os.environ.get('F1_LLM_PROVIDER', 'lmstudio')

        if provider == 'lmstudio':
            llm = ChatOpenAI(
                base_url=base_url,
                api_key=api_key,
                model=model_name,
                temperature=0,
                model_kwargs={'parallel_tool_calls': False},
                disable_streaming=True,
                timeout=120,
                max_retries=1,
            )
        else:
            llm = ChatOpenAI(model=model_name, temperature=0, timeout=120, max_retries=1)

        self._react_agent = create_agent(
            llm, self._tools, system_prompt=_PIT_STRATEGY_SYSTEM_PROMPT
        )
        return self._react_agent

    # ── Entry point methods ───────────────────────────────────────────────────

    def run(self, lap_state: dict) -> PitStrategyOutput:
        """Run the Pit Strategy Agent from a FastF1 session-based lap_state.

        Populates self.laps_df and self.session_meta from the FastF1 Session,
        then invokes the ReAct agent and parses the response.

        Args:
            lap_state: Dict with keys:
                session     — Loaded FastF1 Session.
                driver      — FastF1 driver abbreviation (e.g. 'NOR').
                lap_number  — Current lap number.
                compound    — Compound currently fitted (default 'MEDIUM').
                rival       — Abbreviation of the nearest rival ahead (optional).
                sc_prob     — SC probability from N27 (float, default 0.0).
                laps_to_cliff — P10 laps-to-cliff from N26 (float, optional).

        Returns:
            PitStrategyOutput with action, compound, stop durations, undercut signal.
        """
        session    = lap_state['session']
        driver     = lap_state['driver']
        lap_number = lap_state['lap_number']
        compound   = lap_state.get('compound', 'MEDIUM')
        rival      = lap_state.get('rival')
        sc_prob    = lap_state.get('sc_prob', 0.0)
        laps_cliff = lap_state.get('laps_to_cliff')

        laps               = session.laps.pick_accurate().copy()
        laps['LapNumber']  = laps['LapNumber'].astype(int)
        self.laps_df       = laps

        # Who is on track this lap. This agent is a module singleton, so leaving the
        # previous call's roster in place would validate targets against another race.
        # A car that is gone has no row for the lap, which is the same presence signal
        # RaceStateManager's `rivals` gives the other entry point (#462). Empty means we
        # could not tell, so it stays None and the guard disables rather than rejecting
        # every target.
        _at_lap = laps.loc[laps['LapNumber'] == lap_number, 'Driver'].dropna()
        self._live_drivers = set(_at_lap) | {driver} if len(_at_lap) else None

        team_lookup = (
            laps[['Driver', 'Team']].drop_duplicates()
                .set_index('Driver')['Team'].to_dict()
        )
        self.session_meta = {
            'gp_name':     session.event['EventName'],
            'year':        session.event['EventDate'].year,
            'total_laps':  int(laps['LapNumber'].max()),
            'team_lookup': team_lookup,
        }

        sc_currently_active = bool(lap_state.get('sc_currently_active', False))
        vsc_active          = bool(lap_state.get('vsc_active', False))
        return self._run_core(
            driver, lap_number, compound, rival, sc_prob, laps_cliff,
            sc_currently_active=sc_currently_active,
            vsc_active=vsc_active,
        )

    def run_from_state(self, lap_state: dict, laps_df: pd.DataFrame) -> PitStrategyOutput:
        """RSM adapter: run the Pit Strategy Agent from a RaceStateManager lap_state.

        Populates self.laps_df and self.session_meta from laps_df and lap_state
        without requiring a FastF1 session object. The rival ahead is derived from
        lap_state['rivals'] by finding the car with position = driver_position - 1.

        team_lookup is built from the rivals list (each rival has a 'team' field)
        plus the driver's team from session_meta.

        Args:
            lap_state: Dict from RaceStateManager.get_lap_state(). Keys used:
                lap_number, driver (telemetry), rivals (list), session_meta, weather.
            laps_df: Full race laps DataFrame. LapNumber must be int-castable.

        Returns:
            PitStrategyOutput with all fields populated.
        """
        d      = lap_state['driver']
        meta   = lap_state['session_meta']
        rivals = lap_state.get('rivals', [])

        lap_number = lap_state['lap_number']
        driver     = meta['driver']
        gp_name    = meta.get('gp_name', '')
        # 57 = median/mode race length across the 2022-2025 dataset; the same fallback the
        # other strategy surfaces use, so a missing key resolves to one value everywhere.
        total_laps = meta.get('total_laps', 57)
        year       = meta.get('year', 2025)
        team       = meta.get('team', 'Unknown')
        compound   = d.get('compound', 'MEDIUM')

        # A missing position must not resolve to a searchable value: defaulting to 20
        # lets `driver_pos - 1 == 19` accidentally match a REAL P19 rival, inventing a
        # rival-ahead relationship for a car whose position we don't actually know. This
        # is the same Series.get(k, default)-only-fires-on-missing-COLUMN sentinel
        # pattern documented elsewhere in this file (#428/#444/#462) — here the default
        # is dict.get, but the failure mode is identical: an explicit unknown must
        # propagate as None, not as a plausible-looking P20 (#465/F6).
        driver_pos  = d.get('position')
        rival_ahead = (
            next((r['driver'] for r in rivals if r.get('position') == driver_pos - 1), None)
            if driver_pos is not None else None
        )

        team_lookup = {r['driver']: r.get('team', 'Unknown') for r in rivals}
        team_lookup[driver] = team

        # Who is actually on track this lap. RaceStateManager builds `rivals` from the
        # per-lap rows, so a car that has retired simply is not in it; this is the same
        # answer a timing screen gives, and the only one available without guessing.
        #
        # It matters because score_undercut_tool takes a free-text driver code from the
        # LLM, which can name anyone on the grid. Without this, undercutting HUL at
        # Lusail lap 40 (he crashed on lap 7) scores happily, and worse: BEA retired on
        # lap 41, so at lap 50 the features come out complete, with no NaN and no
        # warning, describing a car sitting in the garage (#462).
        #
        # An empty rivals list means the roster is unknown, not "only our car is racing":
        # keeping {driver} alone would reject every undercut target silently. None disables
        # the guard instead, matching run() (`... if len(_at_lap) else None`) and
        # _live_drivers_from (`return live or None`) in strategy_orchestrator.
        on_track = {r['driver'] for r in rivals if r.get('driver')}
        self._live_drivers = on_track | {driver} if on_track else None

        self.laps_df = laps_df.copy()
        if 'LapNumber' in self.laps_df.columns:
            self.laps_df['LapNumber'] = self.laps_df['LapNumber'].astype(int)

        self.session_meta = {
            'gp_name':     gp_name,
            'year':        year,
            'total_laps':  total_laps,
            'team_lookup': team_lookup,
        }

        sc_prob             = lap_state.get('sc_prob', 0.0)
        laps_cliff          = lap_state.get('laps_to_cliff')
        sc_currently_active = bool(lap_state.get('sc_currently_active', False))
        vsc_active          = bool(lap_state.get('vsc_active', False))

        return self._run_core(
            driver, lap_number, compound, rival_ahead, sc_prob, laps_cliff,
            sc_currently_active=sc_currently_active,
            vsc_active=vsc_active,
        )

    def _validate_undercut_target(self, candidate: Optional[str]) -> Optional[str]:
        """Return candidate only when it is a currently-live driver (or liveness unknown).

        score_undercut_tool already refuses to score a driver_y absent from
        _live_drivers, so its P(undercut_success) never reaches _parse_tool_outputs'
        regex for an invalid target — but that guard and this one must not depend on
        each other to be correct: assembly re-checks independently before the parsed
        argmax driver becomes PitStrategyOutput.undercut_target (#432).

        Args:
            candidate: Driver abbreviation parsed as the argmax undercut target from
                _parse_tool_outputs, or None when no score_undercut_tool call matched.

        Returns:
            candidate unchanged when liveness can't be determined (_live_drivers is
            None) or candidate is live; otherwise None.
        """
        if candidate is None:
            return None
        if self._live_drivers is not None and candidate not in self._live_drivers:
            logger.warning(
                'undercut_target %s parsed from tool output but not on track; dropping',
                candidate,
            )
            return None
        return candidate

    def _run_core(
        self,
        driver: str,
        lap_number: int,
        compound: str,
        rival: Optional[str],
        sc_prob: float,
        laps_cliff: Optional[float],
        sc_currently_active: bool = False,
        vsc_active: bool = False,
    ) -> PitStrategyOutput:
        """Core invocation: self.laps_df / self.session_meta already set.

        Builds the prompt, invokes the ReAct agent, parses and returns output.

        Args:
            driver: FastF1 driver abbreviation.
            lap_number: Current lap number.
            compound: Compound currently fitted.
            rival: Abbreviation of the nearest rival ahead (or None).
            sc_prob: Safety Car probability from N27.
            laps_cliff: P10 laps-to-cliff from N26 (or None).
            sc_currently_active: True when an RCM confirms a neutralisation now.
            vsc_active: True when that neutralisation is a VSC (Art. 56); shapes the
                prompt banner so the model does not treat a VSC stop as a near-free
                full-SC stop (#471).

        Returns:
            Fully populated PitStrategyOutput.
        """
        if not _LANGGRAPH_AVAILABLE:
            raise ImportError('LangGraph / LangChain not installed.')

        # RCM-confirmed ground truth for this lap, stored so predict_pit_duration_tool
        # can cross-check the LLM-supplied under_sc argument against it instead of
        # trusting free text alone (#476). Set only here, not in __init__: a direct
        # tool call outside a real run leaves it absent (getattr returns None), which
        # skips the cross-check rather than asserting a False ground truth against an
        # intentional test input.
        self.sc_currently_active = sc_currently_active

        candidates = self._get_undercut_candidates(driver, lap_number)
        rival_str  = rival if rival else (candidates[0] if candidates else 'no rival in range')

        driver_row = self._get_lap_row(driver, lap_number)
        # Series.get returns a stored NaN, so int() would crash on the ~486 featured
        # rows with NaN TyreLife/Position — guard with pd.notna, not a bare .get
        # default (which only fires when the KEY is missing, never the VALUE) (#499).
        _raw_tl    = driver_row.get('TyreLife') if driver_row is not None else None
        tyre_life  = int(_raw_tl) if pd.notna(_raw_tl) else 1
        _raw_pos   = driver_row.get('Position') if driver_row is not None else None
        position   = int(_raw_pos) if pd.notna(_raw_pos) else 0
        team       = self.session_meta.get('team_lookup', {}).get(driver, 'Unknown')

        message = _build_pit_prompt(
            driver=driver, lap_number=lap_number, tyre_life=tyre_life,
            compound=compound, team=team, position=position, rival_str=rival_str,
            sc_prob=sc_prob,
            laps_to_cliff_p10=laps_cliff if laps_cliff is not None else 0.0,
            sc_currently_active=sc_currently_active,
            vsc_active=vsc_active,
        )

        react_agent = self.get_react_agent()
        response    = react_agent.invoke({'messages': [HumanMessage(content=message)]})
        messages    = response['messages']

        parsed                          = _parse_tool_outputs(messages)
        action, compound_rec, reasoning = _parse_agent_summary(messages[-1].content)

        # A deployed SC does NOT force a stop. There used to be a guard here flipping
        # STAY_OUT to PIT_NOW, and it was a strategy opinion, not a rule: staying out
        # is right whenever you have already stopped, you lead a pack that must stop
        # anyway, or the race is ending (Art. 55.17 finishes it behind the SC, making
        # the surrendered position unrecoverable). The regulatory facts an SC does
        # force live in N27; see tests/mc/test_sc_regulatory_rails.py.
        sc_reactive = sc_currently_active or (action == 'REACTIVE_SC') or (
            sc_prob >= 0.30 and action in ('PIT_NOW', 'UNDERCUT')
        )

        return PitStrategyOutput(
            action                  = action,
            recommended_lap         = lap_number if action != 'STAY_OUT' else None,
            compound_recommendation = compound_rec or parsed.get('compound_recommendation') or 'MEDIUM',
            stop_duration_p05       = parsed['stop_duration_p05'] or 0.0,
            stop_duration_p50       = parsed['stop_duration_p50'] or 0.0,
            stop_duration_p95       = parsed['stop_duration_p95'] or 0.0,
            undercut_prob           = parsed['undercut_prob'],
            # The driver score_undercut_tool actually scored highest, not the
            # POSITIONAL rival_str candidate used to build the prompt — those two
            # silently disagreed whenever the agent scored more than one rival (#432).
            undercut_target         = self._validate_undercut_target(parsed['undercut_target']),
            sc_reactive             = sc_reactive,
            reasoning               = reasoning,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Lazy singleton
# ─────────────────────────────────────────────────────────────────────────────

_default_pit_agent: Optional[PitStrategyAgent] = None


def _get_default_pit_agent() -> PitStrategyAgent:
    """Return the process-level PitStrategyAgent singleton, creating it on first call.

    N15/N16 models and lookup tables are loaded only once per process.

    Returns:
        PitStrategyAgent with all models loaded and tools built.
    """
    global _default_pit_agent
    if _default_pit_agent is None:
        _default_pit_agent = PitStrategyAgent()
    return _default_pit_agent


# ─────────────────────────────────────────────────────────────────────────────
# Public entry points — backward-compatible signatures (unchanged)
# ─────────────────────────────────────────────────────────────────────────────

def run_pit_strategy_agent(lap_state: dict) -> PitStrategyOutput:
    """Run the Pit Strategy Agent for a driver at a given lap.

    Delegates to the process-level PitStrategyAgent singleton. Populates session
    state from the FastF1 Session in lap_state, then invokes the LangGraph ReAct
    agent and parses the response.

    Args:
        lap_state: Dict with keys: session, driver, lap_number, compound, rival,
            sc_prob, laps_to_cliff. See PitStrategyAgent.run for full specification.

    Returns:
        PitStrategyOutput with action, compound, stop durations, undercut signal.
    """
    return _get_default_pit_agent().run(lap_state)


def run_pit_strategy_agent_from_state(
    lap_state: dict,
    laps_df: pd.DataFrame,
) -> PitStrategyOutput:
    """RSM adapter: run the Pit Strategy Agent from a RaceStateManager lap_state.

    Delegates to the process-level PitStrategyAgent singleton. No FastF1 session
    required — all context is derived from laps_df and the lap_state dict.

    Args:
        lap_state: Dict from RaceStateManager.get_lap_state(). Keys used:
            lap_number, driver, rivals, session_meta, sc_prob, laps_to_cliff.
        laps_df: Full race laps DataFrame.

    Returns:
        PitStrategyOutput with all fields populated.
    """
    return _get_default_pit_agent().run_from_state(lap_state, laps_df)


def get_pit_strategy_react_agent(
    provider: str = 'lmstudio',
    model_name: str = 'gpt-4.1-mini',
    base_url: str = 'http://localhost:1234/v1',
    api_key: str = 'lm-studio',
):
    """Return the LangGraph ReAct agent backed by the singleton PitStrategyAgent.

    Avoids connecting to the LLM at import time — created only when N31 or tests
    actually invoke the agent.

    Args:
        provider: 'lmstudio' or 'openai'.
        model_name: Model identifier for ChatOpenAI.
        base_url: Base URL for LM Studio (ignored when provider='openai').
        api_key: API key; 'lm-studio' for local server.

    Returns:
        LangGraph CompiledGraph.
    """
    return _get_default_pit_agent().get_react_agent(
        provider=provider,
        model_name=model_name,
        base_url=base_url,
        api_key=api_key,
    )
