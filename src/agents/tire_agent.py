"""Tire Agent — src/agents/tire_agent.py

Extracted from N26_tire_agent.ipynb. Wraps the per-compound TireDegTCN models
(N09/N10) in a LangGraph ReAct agent that answers: how many laps remain before
the degradation cliff?

Public API
----------
run_tire_agent(stint_state)                   → TireOutput  (FastF1 session in stint_state)
run_tire_agent_from_state(lap_state, laps_df) → TireOutput  (RSM adapter, no FastF1 session)
get_tire_react_agent(**kwargs)                → CompiledGraph

Module-level singletons
-----------------------
CFG — TireAgentConfig: loads routing, calibration, encoding maps, cliff thresholds.
      Kept at module level so TireOutput.__post_init__ can call CFG.get_cliff_thresholds.
      Model bundles (BUNDLES) are loaded lazily inside TireAgent.__init__ to avoid
      expensive I/O at import time.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Repo root (module-relative) ───────────────────────────────────────────────
# Walker with a root-stop guard so we don't spin forever when the module is
# imported from outside a git checkout (e.g. uv tool install).
_REPO_ROOT = Path(__file__).resolve().parent
while not (_REPO_ROOT / '.git').exists():
    if _REPO_ROOT.parent == _REPO_ROOT:
        break
    _REPO_ROOT = _REPO_ROOT.parent

# Prefer f1_strat_manager.data_cache.get_data_root() so the uv tool install
# flow (where data lives under ~/.f1-strat/) works transparently; fall back
# to the repo-relative layout when the helper is not importable.
try:
    from src.f1_strat_manager.data_cache import get_data_root as _get_data_root
    _DATA_ROOT = _get_data_root()
except Exception:
    _DATA_ROOT = _REPO_ROOT / 'data'

logger = logging.getLogger(__name__)

_MODEL_DIR  = _DATA_ROOT / 'models' / 'tire_degradation'
_PROCESSED  = _DATA_ROOT / 'processed'
_AGENTS_DIR = _DATA_ROOT / 'models' / 'agents'


# ─────────────────────────────────────────────────────────────────────────────
# TireDegTCN — reproduced from N10 (different state dict layout from legacy N09)
# ─────────────────────────────────────────────────────────────────────────────

class CausalConv1dBlock(nn.Module):
    """Single causal dilated convolution layer with left-side padding.

    Uses manual left-side padding instead of PyTorch's built-in padding to
    guarantee strict causality — no future timestep information leaks into
    the current prediction. This is critical for tire degradation modelling
    because the model is used at inference time with partial stint sequences
    where future laps are not yet observed.

    Args:
        in_ch: Number of input channels (feature dimension after projection).
        out_ch: Number of output channels.
        kernel_size: Convolutional kernel width; combined with dilation controls
            the effective receptive field.
        dilation: Dilation factor. Doubling dilation across layers (1, 2, 4, 8)
            gives exponential receptive field growth with linear parameter count.
        dropout: Dropout probability applied after GELU activation. Kept active
            at inference time for MC Dropout uncertainty estimation.
    """

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, dilation: int, dropout: float = 0.1):
        super().__init__()
        self.pad  = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, dilation=dilation, padding=0)
        self.norm = nn.LayerNorm(out_ch)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.pad(x, (self.pad, 0))
        x = self.conv(x)
        return self.drop(F.gelu(self.norm(x.transpose(1, 2)).transpose(1, 2)))


class TCNResidualBlock(nn.Module):
    """Two stacked CausalConv1dBlocks with an additive residual connection.

    The residual shortcut allows gradients to flow unobstructed through deep
    stacks of dilated convolutions, preventing vanishing gradients and enabling
    the network to learn incremental refinements on top of the identity mapping.

    Args:
        ch: Number of channels (equal for input and output — no projection).
        kernel_size: Kernel size passed to both inner CausalConv1dBlocks.
        dilation: Dilation factor passed to both inner CausalConv1dBlocks.
        dropout: Dropout probability.
    """

    def __init__(self, ch: int, kernel_size: int, dilation: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            CausalConv1dBlock(ch, ch, kernel_size, dilation, dropout),
            CausalConv1dBlock(ch, ch, kernel_size, dilation, dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.net(x) + x)


class TireDegTCN(nn.Module):
    """Temporal Convolutional Network for tire degradation prediction.

    Architecture: linear input projection → stack of TCNResidualBlocks with
    exponentially increasing dilation (2^0, 2^1, …, 2^(n_layers-1)) → linear
    output head predicting a single scalar (FuelAdjustedDegAbsolute).

    Redefined here (not imported from src/) because the N10 fine-tuning exports
    use a different state dict layout than the legacy EnhancedTCN in
    src/strategy/models/tire_degradation_model.py.

    MC Dropout is enabled by calling model.train() before inference and running
    N_MC forward passes — see TireAgent._build_tools.

    Args:
        n_features: Number of input features per timestep (42 in N10 exports).
        d_model: Hidden channel dimension after input projection (64 in N10).
        n_layers: Number of TCNResidualBlocks. Receptive field = kernel_size × (2^n_layers - 1).
        kernel_size: Convolutional kernel width (3 in N10).
        dropout: Dropout probability (0.1 in N10; must match training for MC calibration).
    """

    def __init__(
        self,
        n_features: int,
        d_model: int = 64,
        n_layers: int = 4,
        kernel_size: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_proj  = nn.Linear(n_features, d_model)
        self.blocks      = nn.ModuleList([
            TCNResidualBlock(d_model, kernel_size, 2**i, dropout)
            for i in range(n_layers)
        ])
        self.output_head = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.input_proj(x).transpose(1, 2)
        for block in self.blocks:
            x = block(x)
        return self.output_head(x.transpose(1, 2)[:, -1, :]).squeeze(-1)


# ─────────────────────────────────────────────────────────────────────────────
# TireAgentConfig
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TireAgentConfig:
    """Runtime configuration for the Tire Agent.

    Resolves all model paths relative to the repo root and loads — once — the
    three JSON artefacts produced by N10: routing config (compound → bundle file
    + window), MC Dropout calibration (per-compound uncertainty sigma), and
    encoding maps (label encodings for Team, Compound, AbsoluteCompound). Also
    loads the circuit cluster map from the k=4 parquet (N05) and the cluster-aware
    cliff thresholds from tire_agent_config_v1.json (written at N26 Step 6).

    Attributes:
        n_mc: Number of Monte Carlo Dropout forward passes per inference call.
            50 passes give a stable P10/P50/P90 interval without excessive latency.
        model_name: LM Studio local model identifier for the ReAct agent LLM.
        cliff_pit_soon_laps: Global fallback threshold below which warning_level
            is PIT_SOON. Per-cluster values take precedence when available.
        cliff_monitor_laps: Global fallback threshold below which warning_level
            is MONITOR. Per-cluster values take precedence when available.
    """

    n_mc: int = 50
    model_name: str = 'gpt-4.1-mini'
    cliff_pit_soon_laps: int = 3
    cliff_monitor_laps: int  = 7

    def __post_init__(self) -> None:
        self._model_dir = _MODEL_DIR
        self.export_dir = _AGENTS_DIR
        try:
            self.export_dir.mkdir(parents=True, exist_ok=True)
        except OSError:
            pass  # read-only mount in Docker

        self.routing_cfg                              = self._load_routing_cfg()
        self.mc_calibration, self.mc_sigma_fallback   = self._load_mc_calibration()
        self._load_encoding_maps()
        self.circuit_cluster_map                      = self._load_circuit_clusters()
        self._load_cliff_thresholds()

    def _load_routing_cfg(self) -> dict:
        """Load routing_config.json: compound ID → bundle filename + window size."""
        with open(self._model_dir / 'routing_config.json') as f:
            return json.load(f)

    def _load_mc_calibration(self) -> tuple[dict, float]:
        """Load MC Dropout calibration JSON and compute cross-compound sigma fallback.

        mc_dropout_calibration.json stores per-compound mean_sigma_s values fitted
        in N10. The fallback sigma is the mean across all compounds — used when
        compound_id is absent from the dict (e.g. C6 with sparse data).

        Returns:
            Tuple (calibration_dict, sigma_fallback).
        """
        with open(self._model_dir / 'mc_dropout_calibration.json') as f:
            mc_cal = json.load(f)
        fallback = float(np.mean([v['mean_sigma_s'] for v in mc_cal.values()]))
        return mc_cal, fallback

    def _load_encoding_maps(self) -> None:
        """Load encoding_maps.json and set four label-encoding dicts as instance attrs."""
        with open(self._model_dir / 'encoding_maps.json') as f:
            enc = json.load(f)
        self.team_id_map: dict           = enc['team_id']
        self.compound_id_map: dict       = enc['compound_id']
        self.abs_compound_id_map: dict   = enc['absolute_compound_id']
        self.compound_hardness_map: dict = enc['compound_hardness']

    def _load_circuit_clusters(self) -> dict:
        """Load k=4 circuit cluster parquet and return GP_Name → Cluster int dict."""
        cluster_df = pd.read_parquet(
            _PROCESSED / 'circuit_clustering' / 'circuit_clusters_k4.parquet'
        )
        return dict(zip(cluster_df['GP_Name'], cluster_df['Cluster'].astype(int)))

    def _load_cliff_thresholds(self) -> None:
        """Load cluster-aware and GP-level cliff thresholds from tire_agent_config_v1.json.

        Falls back to empty dicts (global thresholds only) when the file does not
        exist yet — this covers the case where N26 Step 6 has not been run.
        """
        cfg_path = self.export_dir / 'tire_agent_config_v1.json'
        if cfg_path.exists():
            with open(cfg_path) as f:
                agent_cfg = json.load(f)
            cat = agent_cfg.get('cluster_aware_thresholds', {})
            self.cliff_pit_soon_by_cluster: dict = {
                int(k): v for k, v in cat.get('pit_soon_by_cluster', {}).items()
            }
            self.cliff_monitor_by_cluster: dict = {
                int(k): v for k, v in cat.get('monitor_by_cluster', {}).items()
            }
            self.cliff_overrides_by_gp: dict = cat.get('overrides_by_gp', {})
        else:
            self.cliff_pit_soon_by_cluster = {}
            self.cliff_monitor_by_cluster  = {}
            self.cliff_overrides_by_gp     = {}

    def get_cliff_thresholds(self, gp_name: str) -> tuple[int, int]:
        """Return (pit_soon_laps, monitor_laps) for the given GP.

        GP-level overrides take highest priority, then cluster-specific thresholds,
        then global defaults. This hierarchy lets circuits whose tyre behaviour is
        poorly captured by their cluster label (e.g. Mexico City at altitude) be
        tuned individually without touching global values.

        Args:
            gp_name: GP name as in circuit_cluster_map (e.g. 'Sakhir').

        Returns:
            Tuple (pit_soon_laps, monitor_laps) as integers.
        """
        if gp_name in self.cliff_overrides_by_gp:
            ov = self.cliff_overrides_by_gp[gp_name]
            return ov['pit_soon'], ov['monitor']
        cluster_id = self.circuit_cluster_map.get(gp_name)
        if cluster_id is not None:
            pit_soon = self.cliff_pit_soon_by_cluster.get(cluster_id, self.cliff_pit_soon_laps)
            monitor  = self.cliff_monitor_by_cluster.get(cluster_id, self.cliff_monitor_laps)
            return pit_soon, monitor
        return self.cliff_pit_soon_laps, self.cliff_monitor_laps

    def load_bundle(self, compound_id: str) -> dict:
        """Load a compound .pt bundle and attach an instantiated TireDegTCN in eval mode.

        Each .pt file is a self-contained dict from N10: state dict, fitted
        StandardScaler, feature name list, window size, and architecture hparams.
        """
        cfg    = self.routing_cfg[compound_id]
        bundle = torch.load(
            self._model_dir / cfg['bundle'],
            map_location='cpu',
            weights_only=False,
        )
        model = TireDegTCN(bundle['n_features'], **bundle['model_hparams'])
        model.load_state_dict(bundle['state_dict'])
        model.eval()
        bundle['model'] = model
        return bundle

    def load_all_bundles(self) -> dict:
        """Load every compound defined in routing_config; return {compound_id: bundle}."""
        return {cid: self.load_bundle(cid) for cid in self.routing_cfg}


# ── Module-level config singleton ─────────────────────────────────────────────
# Kept at module level because TireOutput.__post_init__ calls
# CFG.get_cliff_thresholds(self.gp_name). Model bundles are NOT loaded here;
# they are loaded lazily inside TireAgent.__init__ to avoid expensive I/O at import time.
CFG = TireAgentConfig()

# ── Per-compound cumulative degradation cliff thresholds (seconds) ────────────
# p75 of last-stint-lap FuelAdjustedDegAbsolute in N10 training data (2023-2024).
# 75% of stints had already pitted by this level — a practical proxy for the cliff.
CLIFF_THRESHOLD: dict[str, int] = {
    'C1': 3,  # p75 = 2.20 → ceil = 3
    'C2': 2,  # p75 = 1.74 → ceil = 2
    'C3': 2,  # p75 = 1.96 → ceil = 2
    'C4': 2,  # p75 = 1.75 → ceil = 2
    'C5': 2,  # p75 = 1.43 → ceil = 2
    'C6': 2,  # p75 = 1.82 → ceil = 2
}

# Longest race on the current calendar (Monaco, 78 laps; max observed across
# 2023-2025 data). Fallback ceiling for laps-to-cliff when session_meta carries no
# total_laps: a cliff beyond the longest possible race is "not this race" regardless
# of circuit, so clamping there caps absurd values without touching any real one. The
# earlier default of 100 sat above every race, so the clamp it belonged to never fired.
# session_meta.total_laps is present on every shipping path, so this only guards
# hand-built states.
MAX_RACE_LAPS: int = 78


# ─────────────────────────────────────────────────────────────────────────────
# TireOutput
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TireOutput:
    """Structured output of the Tire Agent for one driver at one point in the race.

    The TCN produces a single scalar (predicted cumulative degradation) per forward
    pass. From N_MC MC passes we derive deg_rate and the P10/P50/P90 interval for
    laps remaining before the cliff. warning_level is derived in __post_init__ so
    downstream agents (N28, N31) get a categorical signal without re-implementing thresholds.

    Attributes:
        compound: Pirelli compound ID string (e.g. 'C2', 'C4'). Passed through
            from stint_state and used for MC Dropout calibration lookup.
        current_tyre_life: Laps completed on this tyre set at inference time.
            Used by N28 Pit Strategy as baseline for undercut feature construction.
        deg_rate: Predicted degradation rate in seconds per lap (median of MC passes).
        laps_to_cliff_p10: Pessimistic estimate (P10) of laps before the cliff.
            Drives PIT_SOON warning — conservative to avoid running too long.
        laps_to_cliff_p50: Median estimate of laps before the cliff.
            Primary planning value used in strategy timelines.
        laps_to_cliff_p90: Optimistic estimate (P90) of laps before the cliff.
            Bounds the stay-out scenario in the Pit Strategy Agent.
        gp_name: GP name forwarded from stint_state; used in __post_init__ to
            resolve cluster-aware cliff thresholds via CFG.get_cliff_thresholds.
        warning_level: Categorical urgency derived from laps_to_cliff_p10 in
            __post_init__: PIT_SOON (< pit_soon threshold), MONITOR (< monitor
            threshold), or OK. Thresholds are circuit-cluster aware.
        reasoning: LLM synthesis from the ReAct agent, forwarded verbatim to N31.
    """

    compound: str
    current_tyre_life: int
    deg_rate: float
    laps_to_cliff_p10: float
    laps_to_cliff_p50: float
    laps_to_cliff_p90: float
    gp_name: str   = ''
    warning_level: str = field(init=False)
    reasoning: str = ''

    def __post_init__(self) -> None:
        pit_soon, monitor = CFG.get_cliff_thresholds(self.gp_name)
        if self.laps_to_cliff_p10 < pit_soon:
            self.warning_level = 'PIT_SOON'
        elif self.laps_to_cliff_p10 < monitor:
            self.warning_level = 'MONITOR'
        else:
            self.warning_level = 'OK'


# ─────────────────────────────────────────────────────────────────────────────
# Feature pipeline helpers (must match N10 training order exactly)
# Pure functions — receive all required state as arguments, read no globals.
# ─────────────────────────────────────────────────────────────────────────────

def _add_timing_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Convert FastF1 Timedelta columns to float seconds.

    Handles two source formats:
    - FastF1 raw: LapTime/Sector*Time are pandas Timedelta objects
    - Featured parquet: LapTime_s/Sector*_s are already plain floats

    LapsSincePitStop is aliased from TyreLife.
    """
    def _to_seconds(df, td_col, s_col):
        if s_col in df.columns:
            df[s_col] = pd.to_numeric(df[s_col], errors='coerce')
        elif td_col in df.columns:
            val = df[td_col]
            if hasattr(val.iloc[0] if len(val) > 0 else None, 'total_seconds'):
                df[s_col] = val.dt.total_seconds()
            else:
                df[s_col] = pd.to_numeric(val, errors='coerce')
        else:
            df[s_col] = float('nan')
        return df

    df = _to_seconds(df, 'LapTime',    'LapTime_s')
    df = _to_seconds(df, 'Sector1Time', 'Sector1_s')
    df = _to_seconds(df, 'Sector2Time', 'Sector2_s')
    df = _to_seconds(df, 'Sector3Time', 'Sector3_s')
    df['LapsSincePitStop'] = df['TyreLife']
    return df


def _add_weather_cols(df: pd.DataFrame, session_meta: dict) -> pd.DataFrame:
    """Ensure weather columns exist; fill from session_meta race averages if absent."""
    for col in ('AirTemp', 'TrackTemp', 'Humidity', 'Rainfall'):
        if col not in df.columns:
            df[col] = session_meta.get(col, 0.0)
    return df


def _add_prev_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Shift timing and speed measurements back one lap to create prev-lap context.

    First lap of a stint has no predecessor — filled with the current lap's value
    to avoid NaN in the scaler input. Must run before _add_delta_cols.
    """
    for new_col, src_col in [
        ('Prev_LapTime',  'LapTime_s'),
        ('Prev_SpeedFL',  'SpeedFL'),
        ('Prev_SpeedI1',  'SpeedI1'),
        ('Prev_SpeedI2',  'SpeedI2'),
        ('Prev_SpeedST',  'SpeedST'),
        ('Prev_TyreLife', 'TyreLife'),
    ]:
        df[new_col] = df[src_col].shift(1).fillna(df[src_col])
    return df


def _add_laptime_delta(df: pd.DataFrame) -> pd.DataFrame:
    """Compute lap time first and second derivative features.

    LapTime_Delta: LapTime_s[i] - LapTime_s[i-1]. Requires Prev_LapTime.
    LapTime_Trend: LapTime_Delta[i] - LapTime_Delta[i-1] (second derivative).
    """
    df['LapTime_Delta'] = (df['LapTime_s'] - df['Prev_LapTime']).fillna(0)
    df['LapTime_Trend'] = (df['LapTime_Delta'] - df['LapTime_Delta'].shift(1)).fillna(0)
    return df


def _add_degradation_rate(df: pd.DataFrame) -> pd.DataFrame:
    """Compute degradation rate and acceleration from a rolling 3-lap polyfit.

    DegradationRate: slope of FuelAdjustedLapTime vs TyreLife over a 3-lap window.
    Captures per-lap pace loss due to tyre wear, corrected for fuel mass.
    Requires FuelAdjustedLapTime (from _add_fuel_cols).

    DegAcceleration: change in DegradationRate between consecutive laps.

    Both are shifted by 1 lap (leakage fix matching N10 training) so at position i
    the model sees the rate from lap i-1.
    """
    tyre_lives = df['TyreLife'].values
    adj_times  = df['FuelAdjustedLapTime'].values
    n = len(df)

    raw_deg   = np.zeros(n)
    raw_accel = np.zeros(n)

    for i in range(1, n):
        start = max(0, i - 2)
        x = tyre_lives[start: i + 1]
        y = adj_times[start: i + 1]
        if len(x) >= 2 and not np.isnan(y).any():
            raw_deg[i] = np.polyfit(x, y, 1)[0]

    for i in range(1, n):
        raw_accel[i] = raw_deg[i] - raw_deg[i - 1]

    df['DegradationRate'] = pd.Series(raw_deg, index=df.index).shift(1).fillna(0)
    df['DegAcceleration'] = pd.Series(raw_accel, index=df.index).shift(1).fillna(0)
    return df


def _add_delta_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Orchestrate laptime delta and degradation rate computation."""
    df = _add_laptime_delta(df)
    df = _add_degradation_rate(df)
    return df


def _add_speed_delta_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Compute trap-speed deltas (current minus previous lap) for all four sensors."""
    for sensor in ('FL', 'I1', 'I2', 'ST'):
        df[f'Speed{sensor}_Delta'] = df[f'Speed{sensor}'] - df[f'Prev_Speed{sensor}']
    return df


def _add_compound_cols(df: pd.DataFrame, compound_id: str) -> pd.DataFrame:
    """Set compound identity features from the label-encoding maps in CFG.

    All three encodings are constant within a stint. CompoundHardness is the
    inverse of AbsoluteCompoundID: C1=6 (hardest), C6=1 (softest), as encoded
    in the N10 training data.
    """
    df['AbsoluteCompoundID'] = CFG.abs_compound_id_map.get(compound_id, 3)
    df['CompoundHardness']   = CFG.compound_hardness_map.get(compound_id, 4)
    df['CompoundID']         = CFG.compound_id_map.get(df['Compound'].iloc[0], 1)
    return df


def _add_fuel_cols(df: pd.DataFrame, session_meta: dict) -> pd.DataFrame:
    """Estimate fuel load and cumulative fuel-burn pace gain, matching N04 training.

    FuelLoad: fraction remaining = (total_laps - LapNumber) / total_laps.
    FuelEffect: cumulative gain from fuel burn = (TyreLife - baseline_tyrelife) * 0.055 s/lap.
    FuelAdjustedLapTime: intermediate column needed by _add_degradation_rate.
    """
    total_laps = session_meta['total_laps']

    if 'FuelLoad' not in df.columns:
        df['FuelLoad'] = ((total_laps - df['LapNumber']) / total_laps).clip(lower=0.0)

    baseline_tyrelife         = df['TyreLife'].iloc[0]
    df['FuelEffect']          = (df['TyreLife'] - baseline_tyrelife) * 0.055
    df['FuelAdjustedLapTime'] = df['LapTime_s'] + df['FuelEffect']
    return df


def _add_session_cols(df: pd.DataFrame, session_meta: dict) -> pd.DataFrame:
    """Normalise lap times against session fastest lap and circuit cluster mean.

    lap_time_pct_of_race_fastest: ratio to the race's fastest lap (~1.04 mean).
    lap_time_vs_cluster_mean: delta vs the cluster's typical lap time (seconds).
    mean_sector_speed: circuit-level mean of the three speed traps (km/h).
    track_status_clean: 3-class int — 0=green, 1=yellow/VSC, 2=SC/red flag.

    Two of these columns are trained per-circuit constants, not per-lap values.
    N04 built lap_time_vs_cluster_mean as LapTime_s minus a global per-cluster
    mean lap time (81.36-100.92 s depending on cluster, std 0.0 within a cluster),
    and mean_sector_speed as a per-GP circuit feature (one value per race, std 0.0
    within a GP). Both are TCN inputs listed in tiredeg_feature_manifest.json, and
    both are already shipped by the featured parquet. Recomputing them here from
    the handed-in frame overwrites the trained constant with a per-frame quantity:
    the cluster-mean delta was off by up to 14.9 s per lap (Lusail) and the sector
    speed by up to 17.5 s. They are therefore guarded like FuelLoad and
    track_status_clean — recomputed only when the frame does not already carry them
    (the raw FastF1 path, which has no better source). The sibling
    lap_time_pct_of_race_fastest and laps_remaining are recomputed unconditionally
    because their per-frame recompute reproduces the shipped value exactly
    (0.0000 s delta across all 24 GPs).
    """
    df['lap_time_pct_of_race_fastest'] = (
        df['LapTime_s'] / session_meta['fastest_lap_s']
    )
    if 'lap_time_vs_cluster_mean' not in df.columns:
        df['lap_time_vs_cluster_mean'] = (
            df['LapTime_s'] - session_meta['cluster_mean_lap_s']
        )
    df['laps_remaining'] = session_meta['total_laps'] - df['LapNumber']
    if 'mean_sector_speed' not in df.columns:
        df['mean_sector_speed'] = (df['SpeedI1'] + df['SpeedI2'] + df['SpeedFL']) / 3

    if 'track_status_clean' not in df.columns:
        status_map = {'1': 0, '2': 1, '3': 2, '4': 2, '5': 2, '6': 1, '7': 1}
        if 'TrackStatus' in df.columns:
            df['track_status_clean'] = (
                df['TrackStatus'].astype(str).map(status_map).fillna(0).astype(int)
            )
        else:
            df['track_status_clean'] = 0
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Stateless helpers
# ─────────────────────────────────────────────────────────────────────────────

def _parse_tool_outputs(messages: list) -> dict:
    """Extract numeric fields from ToolMessage strings in the agent message history.

    Parses the structured output lines produced by predict_tire_deg_tool and
    estimate_laps_to_cliff_tool rather than the LLM's free-text final answer,
    guaranteeing the returned values are the exact numbers computed by inference.

    Args:
        messages: LangChain message objects from the agent's invoke result.

    Returns:
        Dict with keys deg_rate, p10, p50, p90 (all floats, defaulting to 0.0).
    """
    result: dict[str, float] = {}
    for msg in messages:
        content = getattr(msg, 'content', '')
        if not isinstance(content, str):
            continue
        for pattern, key in [
            (r'Degradation rate:\s*([\d.]+)', 'deg_rate'),
            (r'P10:\s*([\d.]+)',              'p10'),
            (r'P50:\s*([\d.]+)',              'p50'),
            (r'P90:\s*([\d.]+)',              'p90'),
        ]:
            m = re.search(pattern, content)
            if m and key not in result:
                result[key] = float(m.group(1))
    return result


def _compound_name_to_id(compound_name: str, gp_name: str, year: int) -> str:
    """Map a Pirelli compound name (SOFT/MEDIUM/HARD) to its Cx ID for this GP.

    Loads data/tire_compounds_by_race.json (authoritative source) to resolve
    the Cx allocation for this GP/year. Falls back to C3/C2/C1 if the GP is
    not found — these are the most common mid-season assignments.

    Args:
        compound_name: Pirelli compound name string (e.g. 'SOFT', 'MEDIUM').
        gp_name: GP name matching the tire_compounds_by_race.json keys.
        year: Race year as int.

    Returns:
        Compound ID string such as 'C3'.
    """
    compounds_path = _REPO_ROOT / 'data' / 'tire_compounds_by_race.json'
    fallback = {'SOFT': 'C3', 'MEDIUM': 'C2', 'HARD': 'C1',
                'INTERMEDIATE': 'INT', 'WET': 'WET'}
    if not compounds_path.exists():
        return fallback.get(compound_name.upper(), 'C3')

    with open(compounds_path) as f:
        alloc = json.load(f)

    year_data = alloc.get(str(year), {})
    gp_data   = year_data.get(gp_name, {})
    return gp_data.get(compound_name.upper(), fallback.get(compound_name.upper(), 'C3'))


# ─────────────────────────────────────────────────────────────────────────────
# LangGraph / LangChain optional imports
# ─────────────────────────────────────────────────────────────────────────────

try:
    from langchain_core.tools import tool as lc_tool
    from langchain_openai import ChatOpenAI
    from langchain.agents import create_agent
    _LANGGRAPH_AVAILABLE = True
except ImportError:
    _LANGGRAPH_AVAILABLE = False


# ─────────────────────────────────────────────────────────────────────────────
# System prompt (module-level constant — unchanged from N26)
# ─────────────────────────────────────────────────────────────────────────────

_TIRE_SYSTEM_PROMPT = """You are a Formula 1 tyre degradation analyst embedded in a race strategy system.

Your job is to assess the current state of a tyre stint and determine how many laps remain
before the degradation cliff — the point at which pace loss accelerates sharply and a pit
stop becomes unavoidable.

## Workflow
1. Call `predict_tire_deg_tool` with the driver, compound_id and tyre_life to get the
   current cumulative degradation and instantaneous rate.
2. Call `estimate_laps_to_cliff_tool` with the same inputs to get P10/P50/P90 laps
   remaining before the cliff threshold.
3. Based on the P10 estimate and warning level, reason about whether to:
   - STAY OUT — P10 > 7 laps remaining, no urgent action.
   - MONITOR   — P10 between 3 and 7 laps, prepare pit window.
   - PIT SOON  — P10 < 3 laps, cliff imminent.

## Rules
- Always call both tools before drawing conclusions.
- Base your recommendation on P10 (conservative / worst-case estimate).
- A negative degradation rate means the driver is improving pace on this stint
  (track evolution or fuel load reduction) — this is real, not an error.
- Keep your final answer concise: state the warning level, laps to cliff (P50),
  and one sentence of reasoning.

## Strategic guard-rails
- FRESH TYRES (tyre_life <= 3 laps): the TCN model extrapolates from minimal data
  and cliff predictions are unreliable. Always report STAY OUT regardless of raw
  model output — no tyre degrades to its cliff in the first 3 laps of a stint under
  normal dry conditions. Note "fresh tyres — cliff prediction low confidence" in
  your reasoning.
- EXTENDED STINT: if tyre_life exceeds the compound's typical race life
  (SOFT ~18 laps, MEDIUM ~28 laps, HARD ~38 laps), the driver is extending
  beyond normal limits. Flag "tyre past expected compound life — cliff risk
  elevated" in your reasoning, even if the model has not detected a cliff yet.
  Consider bumping your warning level up by one tier (STAY OUT → MONITOR,
  MONITOR → PIT SOON)."""


# ─────────────────────────────────────────────────────────────────────────────
# TireAgent — encapsulated agent class
# ─────────────────────────────────────────────────────────────────────────────

class TireAgent:
    """Encapsulated Tire Degradation Agent backed by TireDegTCN and LangGraph ReAct.

    Owns all mutable state that was previously held in module-level globals:
    - laps_df / session_meta: set per call by run() / run_from_state()
    - bundles: {compound_id: bundle_dict} with loaded TireDegTCN models
    - _react_agent: lazily created LangGraph CompiledGraph

    LangChain tools are built as closures inside _build_tools() so they read
    instance attributes (self.laps_df, self.session_meta, self.bundles) without
    depending on any module-level globals.

    Args:
        cfg: TireAgentConfig instance. Defaults to the module-level CFG singleton
            so TireOutput.__post_init__ remains consistent.
    """

    def __init__(self, cfg: TireAgentConfig = CFG) -> None:
        self.cfg: TireAgentConfig = cfg
        self.bundles: dict        = self.cfg.load_all_bundles()
        self.laps_df: pd.DataFrame = pd.DataFrame()
        self.session_meta: dict    = {}
        self._react_agent          = None
        self._tools: list          = self._build_tools()

    # ── Feature pipeline (instance methods: use self.bundles / self.cfg) ──────

    def _build_stint_features(
        self,
        stint_laps: pd.DataFrame,
        compound_id: str,
        session_meta: dict,
    ) -> pd.DataFrame:
        """Compute all 42 TCN input features from a FastF1 stint slice.

        Orchestrates the feature helpers in the same order applied during N04/N10
        training. Critical ordering constraints:
        - _add_prev_cols must run before _add_delta_cols (LapTime_Delta needs Prev_LapTime)
        - _add_fuel_cols must run before _add_delta_cols (DegradationRate needs FuelAdjustedLapTime)
        - _add_speed_delta_cols must run after _add_prev_cols

        Args:
            stint_laps: FastF1 laps for one driver and one stint, sorted by LapNumber.
                Required columns: LapTime, Sector1/2/3Time, SpeedFL/I1/I2/ST,
                TyreLife, Position, Compound, LapNumber, TrackStatus, Team.
                Weather columns are filled from session_meta if absent.
            compound_id: Pirelli compound ID (e.g. 'C2').
            session_meta: Dict with keys: fastest_lap_s, cluster_mean_lap_s,
                total_laps, cluster_id, team_id, year, and optionally weather averages.

        Returns:
            DataFrame with 42 float columns in bundle['feature_names'] order.
        """
        df = stint_laps.copy().reset_index(drop=True)

        df = _add_timing_cols(df)
        df = _add_weather_cols(df, session_meta)
        df = _add_compound_cols(df, compound_id)
        df = _add_prev_cols(df)
        df = _add_fuel_cols(df, session_meta)
        df = _add_delta_cols(df)
        df = _add_speed_delta_cols(df)
        df = _add_session_cols(df, session_meta)

        df['Cluster'] = session_meta['cluster_id']
        df['TeamID']  = session_meta['team_id']
        df['Year']    = session_meta['year']

        return df[self.bundles[compound_id]['feature_names']].astype(float)

    def _build_stint_tensor(
        self,
        stint_laps: pd.DataFrame,
        compound_id: str,
        session_meta: dict,
    ) -> torch.Tensor:
        """Scale and tensorise a stint feature DataFrame for TCN inference.

        Applies the StandardScaler stored inside the compound bundle (fitted on
        2023-2024 training data), then pads or trims the sequence to the compound's
        window length. Short stints are left-padded by repeating the first row.

        NaN values (first-lap shifted features and missing speed-trap readings) are
        replaced with 0.0 in RAW space before scaling, matching N09's apply_scaler
        (`scaler.transform(df[features].fillna(0))`). A zero in raw space maps to a
        negative z-score after scaling, which is the "no reading" signal the model
        learned in training; zeroing AFTER scaling instead injects each feature's
        mean, which the model never learned to read as missing. The two are
        equivalent only for a zero-mean feature, and a missing SpeedI1 turned a
        -1.8 s cumulative-degradation reading into +0.4 s (a sign flip across the
        2 s C4 cliff threshold) on the ~20% of mid-stint laps with a NaN speed trap.

        Args:
            stint_laps: Raw FastF1 laps for one driver + stint, sorted ascending.
            compound_id: Pirelli compound ID (e.g. 'C2').
            session_meta: Same dict passed to _build_stint_features.

        Returns:
            Float32 tensor of shape (1, window, 42) on CPU.
        """
        bundle = self.bundles[compound_id]
        window = bundle['window']

        feat_df = self._build_stint_features(stint_laps, compound_id, session_meta)
        scaled  = bundle['scaler'].transform(feat_df.fillna(0))

        if len(scaled) >= window:
            seq = scaled[-window:]
        else:
            # Left-ZERO-pad, verbatim from N09's `_pad_or_truncate`:
            #   pad = np.zeros((window - L, F)); seq = np.concatenate([pad, arr])
            # This used to tile the stint's first lap instead. Windows are 28-31 laps and
            # the median stint is 21-23, so the pad branch is the COMMON path, not the
            # fallback: measured at Barcelona 2024, 97% of calls take it. The TCN's
            # receptive field (61) exceeds the window, so the padding reaches the output.
            #
            # Repeating a real lap tells the model the car ran 20 identical laps before
            # the stint began, which is a fiction it never saw in training; zeros are the
            # scaled-space "no data" it did see. Measured: the tile-pad put 87% of
            # predictions outside the training target's 5-95% band (mean -29.97 s of
            # cumulative degradation against a band of [-5.80, +2.46]), and flipped
            # `warning_level` on 10.5% of laps.
            pad = np.zeros((window - len(scaled), scaled.shape[1]), dtype=scaled.dtype)
            seq = np.vstack([pad, scaled])

        return torch.tensor(seq, dtype=torch.float32).unsqueeze(0)  # (1, window, 42)

    # ── Stint helper ──────────────────────────────────────────────────────────

    def _get_driver_stint(self, driver: str, tyre_life: int) -> Optional[pd.DataFrame]:
        """Filter self.laps_df to the current driver's stint up to the given tyre life.

        Resolves compound from self.session_meta['{driver}_compound'] if available,
        falling back to the most recent compound in laps_df for that driver.
        Returns None when no matching rows exist.

        Args:
            driver: FastF1 driver abbreviation (e.g. 'NOR').
            tyre_life: Current laps completed on this tyre set.

        Returns:
            Filtered and sorted DataFrame, or None if no laps found.
        """
        driver_laps = self.laps_df.loc[self.laps_df['Driver'] == driver]
        compound = self.session_meta.get(
            f'{driver}_compound',
            driver_laps['Compound'].iloc[-1] if len(driver_laps) > 0 else 'MEDIUM',
        )
        mask = (
            (self.laps_df['Driver']   == driver) &
            (self.laps_df['Compound'] == compound) &
            (self.laps_df['TyreLife'] <= tyre_life)
        )

        # Scope to THIS stint and to laps that have already happened. N10 built its
        # training windows grouped by ['Year','GP_Name','DriverNumber','Stint'] (cell 10),
        # and matching on Compound alone does not reproduce that: a driver runs the same
        # compound in more than one stint 26% of the time, so a later stint joined the
        # window, and with no lap bound those laps had not happened yet.
        #
        # Measured on 2024 (26,606 decision points): 37.5% of windows mixed stints and
        # 17.2% contained FUTURE laps. HAM at Barcelona lap 16 got [1..16, 44..59], and
        # `deg_rate` is read as the window's last row, so the agent reported lap 59's
        # degradation while the car was on lap 16. That flips `warning_level`, which is
        # the orchestrator's routing key, so it also decides whether N28 runs at all.
        #
        # Both keys are optional: the FastF1 path does not supply them and keeps its old
        # behaviour rather than breaking.
        current_stint = self.session_meta.get(f'{driver}_stint')
        if current_stint is not None and 'Stint' in self.laps_df.columns:
            mask &= (self.laps_df['Stint'] == current_stint)

        current_lap = self.session_meta.get('current_lap')
        if current_lap is not None and 'LapNumber' in self.laps_df.columns:
            mask &= (self.laps_df['LapNumber'] <= current_lap)

        stint = self.laps_df[mask].sort_values('LapNumber')
        return stint if len(stint) > 0 else None

    # ── LangChain tool factory ────────────────────────────────────────────────

    def _build_tools(self) -> list:
        """Build LangChain tools as closures over this TireAgent instance.

        Each tool reads self.laps_df, self.session_meta, self.bundles, and
        self.cfg at call time — no module-level globals are accessed. Returns
        an empty list when LangGraph is not installed so the agent degrades
        gracefully.

        Returns:
            List of decorated LangChain tool functions.
        """
        if not _LANGGRAPH_AVAILABLE:
            return []

        agent = self  # capture instance for closures

        @lc_tool
        def predict_tire_deg_tool(driver: str, compound_id: str, tyre_life: int) -> str:
            """Predict cumulative tyre degradation and instantaneous rate for the current stint.

            Runs a single deterministic forward pass through the per-compound TireDegTCN
            using the recent laps of the requested driver from the session loaded into
            the agent instance.

            Args:
                driver: FastF1 driver abbreviation (e.g. 'NOR').
                compound_id: Pirelli compound ID (e.g. 'C2'). Must be a key in bundles.
                tyre_life: Current laps on this set of tyres.

            Returns:
                Multi-line string: cumulative degradation (s) and degradation rate (s/lap).
                Returns an error string if no laps are found.
            """
            stint = agent._get_driver_stint(driver, tyre_life)
            if stint is None:
                return f'No laps found for driver {driver} with tyre_life <= {tyre_life}.'

            tensor = agent._build_stint_tensor(stint, compound_id, agent.session_meta)
            model  = agent.bundles[compound_id]['model']

            with torch.no_grad():
                model.eval()
                pred = model(tensor).item()

            feat_df  = agent._build_stint_features(stint, compound_id, agent.session_meta)
            deg_rate = float(feat_df['DegradationRate'].iloc[-1])

            return (
                f'Driver {driver} | Compound {compound_id} | TyreLife {tyre_life}\n'
                f'Cumulative degradation: {pred:.3f} s | Degradation rate: {deg_rate:.4f} s/lap'
            )

        @lc_tool
        def estimate_laps_to_cliff_tool(driver: str, compound_id: str, tyre_life: int) -> str:
            """Estimate laps remaining before tyre cliff using MC Dropout uncertainty.

            Switches the model to train mode so dropout stays active, then runs
            cfg.n_mc forward passes to sample the predictive distribution. P10/P50/P90
            laps remaining are computed from the remaining degradation budget.

            Cliff is defined as cumulative FuelAdjustedDegAbsolute >= CLIFF_THRESHOLD[compound_id].

            Args:
                driver: FastF1 driver abbreviation (e.g. 'NOR').
                compound_id: Pirelli compound ID (e.g. 'C2'). Must be a key in bundles.
                tyre_life: Current laps on this set of tyres.

            Returns:
                Multi-line string: P10/P50/P90 laps to cliff, deg rate, MC std, warning level.
            """
            stint = agent._get_driver_stint(driver, tyre_life)
            if stint is None:
                return f'No laps found for driver {driver} with tyre_life <= {tyre_life}.'

            tensor = agent._build_stint_tensor(stint, compound_id, agent.session_meta)
            model  = agent.bundles[compound_id]['model']
            model.train()  # keep dropout active for MC

            preds = []
            try:
                with torch.no_grad():
                    for _ in range(agent.cfg.n_mc):
                        preds.append(model(tensor).item())
            finally:
                # Never leave the shared bundle in train mode: any later consumer of
                # bundles[cid]['model'] would silently get stochastic "deterministic"
                # predictions. Today only predict_tire_deg_tool saves us, by defensively
                # calling eval() itself (#449).
                model.eval()

            mean_pred = float(np.mean(preds))
            mc_std    = float(np.std(preds))
            sigma     = (
                float(agent.cfg.mc_calibration[compound_id]['mean_sigma_s'])
                if compound_id in agent.cfg.mc_calibration
                else agent.cfg.mc_sigma_fallback
            )
            total_std = np.sqrt(mc_std**2 + sigma**2)

            feat_df  = agent._build_stint_features(stint, compound_id, agent.session_meta)
            deg_rate = max(float(feat_df['DegradationRate'].abs().iloc[-1]), 0.001)

            threshold        = CLIFF_THRESHOLD.get(compound_id, 2.5)
            remaining_budget = max(0.0, threshold - mean_pred)

            # Flooring the divisor without clamping the quotient produced cliffs beyond
            # any possible race: HAM at Austin reported "P50: 27375.2 laps, OK". The floor
            # fires by construction on a stint's first two laps (`_add_degradation_rate`
            # shifts and fills 0), and measured over six 2024 GPs it fires on 10.5% of
            # decision points, 11.8% of which yield >200 laps. The failure mode is "never
            # pit", precisely when the model has no signal.
            #
            # A cliff past the chequered flag is operationally "not this race", so the
            # race distance is the honest ceiling: it says the same thing without looking
            # like a reading. Nothing is invented — the bound is the race itself.
            #
            # The fallback must stay at or below the shortest real race, otherwise a
            # missing total_laps lifts the ceiling above the race and the clamp stops
            # clamping. MAX_RACE_LAPS (78) is the longest race on the calendar; the
            # earlier default of 100 exceeded every race, so an absent key silently
            # disabled the clamp for any race up to 100 laps.
            cliff_ceiling = float(
                agent.session_meta.get('total_laps', MAX_RACE_LAPS) or MAX_RACE_LAPS
            )

            p50 = min(remaining_budget / deg_rate, cliff_ceiling)
            p10 = min(max(0.0, (remaining_budget - total_std) / deg_rate), cliff_ceiling)
            p90 = min((remaining_budget + total_std) / deg_rate, cliff_ceiling)

            to = TireOutput(
                compound=compound_id,
                current_tyre_life=tyre_life,
                deg_rate=round(deg_rate, 4),
                laps_to_cliff_p10=round(p10, 1),
                laps_to_cliff_p50=round(p50, 1),
                laps_to_cliff_p90=round(p90, 1),
            )

            return (
                f'Driver {driver} | Compound {compound_id} | TyreLife {tyre_life}\n'
                f'Laps to cliff — P10: {to.laps_to_cliff_p10} | P50: {to.laps_to_cliff_p50} | P90: {to.laps_to_cliff_p90}\n'
                f'Degradation rate: {deg_rate:.4f} s/lap | MC std: {mc_std:.4f} s | Calibrated sigma: {sigma:.4f} s\n'
                f'Warning level: {to.warning_level}'
            )

        return [predict_tire_deg_tool, estimate_laps_to_cliff_tool]

    # ── LangGraph agent (lazy) ────────────────────────────────────────────────

    def get_react_agent(
        self,
        provider: str = None,
        model_name: str = 'gpt-4.1-mini',
        base_url: str = 'http://localhost:1234/v1',
        api_key: str = 'lm-studio',
    ):
        """Return the LangGraph ReAct agent, creating it on the first call (lazy).

        Avoids connecting to the LLM at import time — the graph is compiled only
        when N31 or a test actually invokes the agent.

        Args:
            provider: 'lmstudio' (default) or 'openai'.
            model_name: Model identifier for ChatOpenAI.
            base_url: Base URL for LM Studio (ignored when provider='openai').
            api_key: API key; use 'lm-studio' for local server.

        Returns:
            LangGraph CompiledGraph — invoke with {"messages": [{"role": "user", "content": ...}]}.

        Raises:
            ImportError: When LangGraph / LangChain are not installed.
        """
        if not _LANGGRAPH_AVAILABLE:
            raise ImportError(
                'LangGraph / LangChain not installed. '
                'Install with: pip install langgraph langchain-openai'
            )

        if self._react_agent is not None:
            return self._react_agent

        import os
        if provider is None:
            provider = os.environ.get('F1_LLM_PROVIDER', 'lmstudio')

        if provider == 'lmstudio':
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
            system_prompt=_TIRE_SYSTEM_PROMPT,
        )
        return self._react_agent

    # ── Entry point methods ───────────────────────────────────────────────────

    def run(self, stint_state: dict) -> TireOutput:
        """Run the Tire Agent from a FastF1 session-based stint_state.

        Populates self.laps_df and self.session_meta from the FastF1 Session in
        stint_state, then invokes the ReAct agent. Numeric values are extracted
        from tool call results in the message history — not from the LLM's
        free-text answer — so the output is deterministic.

        Args:
            stint_state: Dict with keys:
                session     — loaded FastF1 Session (laps + weather already cached).
                driver      — FastF1 driver abbreviation (e.g. 'NOR').
                compound_id — Pirelli compound ID (e.g. 'C2').
                tyre_life   — Current laps on this tyre set.
                gp_name     — GP name matching circuit_cluster_map keys (e.g. 'Sakhir').
                team        — Team name matching team_id_map keys (e.g. 'McLaren').
                year        — Race year (int).

        Returns:
            TireOutput with deg_rate, laps_to_cliff P10/P50/P90, gp_name,
            warning_level, and reasoning.
        """
        session     = stint_state['session']
        driver      = stint_state['driver']
        compound_id = stint_state['compound_id']
        tyre_life   = stint_state['tyre_life']
        gp_name     = stint_state.get('gp_name', '')

        self.laps_df = session.laps.pick_accurate().copy()
        _clean       = self.laps_df[self.laps_df['TrackStatus'] == '1']
        _weather     = session.weather_data.mean(numeric_only=True)

        self.session_meta = {
            'fastest_lap_s':      _clean['LapTime'].min().total_seconds(),
            'cluster_mean_lap_s': _clean['LapTime'].dt.total_seconds().mean(),
            'total_laps':         int(session.total_laps),
            'cluster_id':         self.cfg.circuit_cluster_map.get(gp_name, 0),
            'team_id':            self.cfg.team_id_map.get(stint_state.get('team', 'Unknown'), 4),
            'year':               stint_state.get('year', 2025),
            'AirTemp':   float(_weather.get('AirTemp',   28.0)),
            'TrackTemp': float(_weather.get('TrackTemp', 38.0)),
            'Humidity':  float(_weather.get('Humidity',  50.0)),
            'Rainfall':  0.0,
        }

        return self._run_core(driver, compound_id, tyre_life, gp_name)

    def run_from_state(self, lap_state: dict, laps_df: pd.DataFrame) -> TireOutput:
        """RSM adapter: run the Tire Agent from a RaceStateManager lap_state dict.

        Translates the nested RSM lap_state into self.laps_df / self.session_meta.
        No FastF1 session is required — all context is derived directly from laps_df
        and the lap_state dict produced by RaceStateManager.

        Args:
            lap_state: Dict from RaceStateManager.get_lap_state(). Expected keys:
                lap_number, driver (full telemetry), weather, session_meta.
            laps_df: Full race laps DataFrame (columns must include LapTime, Driver,
                Compound, TyreLife, TrackStatus, LapNumber, SpeedFL/I1/I2/ST,
                Sector1/2/3Time, Team).

        Returns:
            TireOutput with all fields populated.
        """
        d    = lap_state['driver']
        meta = lap_state['session_meta']
        wx   = lap_state.get('weather', {})

        driver      = meta['driver']
        compound    = d.get('compound', 'MEDIUM')
        tyre_life   = d.get('tyre_life', 1)
        gp_name     = meta.get('gp_name', '')
        total_laps  = meta.get('total_laps', 57)
        year        = meta.get('year', 2025)
        team        = meta.get('team', 'Unknown')

        compound_id = (
            compound if compound.startswith('C')
            else _compound_name_to_id(compound, gp_name, year)
        )

        self.laps_df = laps_df.copy()

        # Build session_meta from laps_df (FastF1 Timedelta → float if needed)
        lt_col = 'LapTime_s' if 'LapTime_s' in self.laps_df.columns else 'LapTime'
        if lt_col == 'LapTime' and hasattr(self.laps_df[lt_col].iloc[0], 'total_seconds'):
            lap_times = self.laps_df[lt_col].dropna().apply(lambda t: t.total_seconds())
        else:
            lap_times = pd.to_numeric(self.laps_df[lt_col], errors='coerce').dropna()

        if 'TrackStatus' in self.laps_df.columns:
            clean_mask  = self.laps_df['TrackStatus'].astype(str) == '1'
            clean_times = lap_times[clean_mask] if clean_mask.sum() > 0 else lap_times
        else:
            clean_times = lap_times

        self.session_meta = {
            'fastest_lap_s':      float(clean_times.min()) if len(clean_times) > 0 else 90.0,
            'cluster_mean_lap_s': float(clean_times.mean()) if len(clean_times) > 0 else 90.0,
            'total_laps':         total_laps,
            'cluster_id':         self.cfg.circuit_cluster_map.get(gp_name, 0),
            'team_id':            self.cfg.team_id_map.get(team, 4),
            'year':               year,
            'AirTemp':   wx.get('air_temp',   28.0),
            'TrackTemp': wx.get('track_temp', 38.0),
            'Humidity':  wx.get('humidity',   50.0),
            'Rainfall':  float(wx.get('rainfall', 0)),
            f'{driver}_compound': compound,
            # The stint we are actually in, and the lap we are actually on. N10 trained
            # on windows grouped by ['Year','GP_Name','DriverNumber','Stint'], and the
            # slice below had neither: it matched on Compound alone, so a later stint on
            # the same compound joined the window, and nothing bounded it to the past.
            # At Barcelona 2024 lap 16, HAM's window ran [1..16, 44..59] and the TCN read
            # LAP 59's degradation as current. Both keys travel here rather than through
            # the signature so the FastF1 path, which has neither, degrades to the old
            # behaviour instead of breaking (#449).
            f'{driver}_stint': d.get('stint'),
            'current_lap': lap_state.get('lap_number'),
        }

        return self._run_core(driver, compound_id, tyre_life, gp_name)

    def _run_core(
        self,
        driver: str,
        compound_id: str,
        tyre_life: int,
        gp_name: str,
    ) -> TireOutput:
        """Invoke the ReAct agent with session state already set; parse and return TireOutput.

        self.laps_df and self.session_meta must be populated before calling this method.
        Numeric values are extracted from ToolMessage contents in the message history
        to guarantee determinism regardless of LLM phrasing.

        Args:
            driver: FastF1 driver abbreviation.
            compound_id: Pirelli compound ID string.
            tyre_life: Current laps on this tyre set.
            gp_name: GP name for cliff threshold lookup.

        Returns:
            Fully populated TireOutput.
        """
        # TCN bundles only exist for dry compounds (C1–C6). For wet/intermediate
        # compounds return a stub with conservative defaults — no TCN inference.
        if compound_id not in self.bundles:
            return TireOutput(
                compound          = compound_id,
                current_tyre_life = tyre_life,
                gp_name           = gp_name,
                deg_rate          = 0.03,
                laps_to_cliff_p10 = 20.0,
                laps_to_cliff_p50 = 30.0,
                laps_to_cliff_p90 = 40.0,
                reasoning         = (
                    f"[{compound_id} — TCN model not available for wet/intermediate compounds; "
                    f"conservative defaults used]"
                ),
            )

        react_agent = self.get_react_agent()
        msg = (
            f'Analyse the tyre state for driver {driver}, compound {compound_id}, '
            f'tyre life {tyre_life} laps. Use both tools and give your recommendation.'
        )
        response = react_agent.invoke({'messages': [{'role': 'user', 'content': msg}]})
        parsed   = _parse_tool_outputs(response['messages'])

        reasoning = ''
        for m in reversed(response['messages']):
            if hasattr(m, 'content') and isinstance(m.content, str) and m.content.strip():
                if not getattr(m, 'tool_calls', None):
                    reasoning = m.content.strip()
                    break

        # A parse miss must NOT become 0.0 laps-to-cliff. `_parse_tool_outputs` only
        # writes a key when its regex matched, so an absent 'p10' means "the tool was
        # skipped or its output did not parse" — whereas a PRESENT 0.0 legitimately
        # means "the cliff is now". Defaulting the miss to 0.0 collapsed those two into
        # the alarming one: the MC read "cliff NOW" (penalising STAY_OUT by ~4 s) and
        # the warning level flipped to PIT_SOON, so a silent regex miss became a
        # confident call to box. Fall back to the same conservative stub the
        # wet/intermediate branch above already uses, and say so in the reasoning so the
        # degradation is visible instead of masquerading as a reading (#436).
        if 'p10' not in parsed:
            logger.warning(
                'Tire tool output did not parse for %s (tyre_life=%s) — using conservative '
                'defaults instead of a 0.0 cliff', compound_id, tyre_life,
            )
            return TireOutput(
                compound          = compound_id,
                current_tyre_life = tyre_life,
                gp_name           = gp_name,
                deg_rate          = 0.03,
                laps_to_cliff_p10 = 20.0,
                laps_to_cliff_p50 = 30.0,
                laps_to_cliff_p90 = 40.0,
                reasoning         = (
                    f"[{compound_id} — tire tool output could not be parsed; conservative "
                    f"defaults used] {reasoning}"
                ),
            )

        return TireOutput(
            compound          = compound_id,
            current_tyre_life = tyre_life,
            gp_name           = gp_name,
            deg_rate          = round(parsed.get('deg_rate', 0.0), 4),
            laps_to_cliff_p10 = round(parsed['p10'], 1),
            laps_to_cliff_p50 = round(parsed.get('p50', 0.0), 1),
            laps_to_cliff_p90 = round(parsed.get('p90', 0.0), 1),
            reasoning         = reasoning,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Lazy singleton
# ─────────────────────────────────────────────────────────────────────────────

_default_tire_agent: Optional[TireAgent] = None


def _get_default_tire_agent() -> TireAgent:
    """Return the process-level TireAgent singleton, creating it on first call.

    Model bundles are loaded only once per process. Subsequent calls return the
    cached instance immediately.

    Returns:
        TireAgent with all compound bundles loaded and tools built.
    """
    global _default_tire_agent
    if _default_tire_agent is None:
        _default_tire_agent = TireAgent()
    return _default_tire_agent


# ─────────────────────────────────────────────────────────────────────────────
# Public entry points — backward-compatible signatures (unchanged)
# ─────────────────────────────────────────────────────────────────────────────

def run_tire_agent(stint_state: dict) -> TireOutput:
    """Run the Tire Agent for a given stint and return a structured TireOutput.

    Delegates to the process-level TireAgent singleton. Populates session state
    from the FastF1 Session object inside stint_state, then invokes the LangGraph
    ReAct agent. Numeric outputs are extracted from tool call results in the
    message history — not from the LLM's free-text answer.

    Args:
        stint_state: Dict with keys: session, driver, compound_id, tyre_life,
            gp_name, team, year. See TireAgent.run for full specification.

    Returns:
        TireOutput with deg_rate, laps_to_cliff P10/P50/P90, warning_level, reasoning.
    """
    return _get_default_tire_agent().run(stint_state)


def run_tire_agent_from_state(lap_state: dict, laps_df: pd.DataFrame) -> TireOutput:
    """RSM adapter: run the Tire Agent from a RaceStateManager lap_state dict.

    Delegates to the process-level TireAgent singleton. No FastF1 session required —
    all context is derived from laps_df and the lap_state produced by RaceStateManager.

    Args:
        lap_state: Dict from RaceStateManager.get_lap_state(). Expected keys:
            lap_number, driver (full telemetry), weather, session_meta.
        laps_df: Full race laps DataFrame with required telemetry columns.

    Returns:
        TireOutput with all fields populated.
    """
    return _get_default_tire_agent().run_from_state(lap_state, laps_df)


def get_tire_react_agent(
    provider: str = 'lmstudio',
    model_name: str = 'gpt-4.1-mini',
    base_url: str = 'http://localhost:1234/v1',
    api_key: str = 'lm-studio',
):
    """Return the LangGraph ReAct agent backed by the singleton TireAgent instance.

    Avoids connecting to the LLM at import time — created only when N31 or tests
    actually invoke the agent.

    Args:
        provider: 'lmstudio' or 'openai'.
        model_name: Model identifier for ChatOpenAI.
        base_url: Base URL for LM Studio (ignored when provider='openai').
        api_key: API key; use 'lm-studio' for local server.

    Returns:
        LangGraph CompiledGraph — invoke with {"messages": [{"role": "user", "content": ...}]}.
    """
    return _get_default_tire_agent().get_react_agent(
        provider=provider,
        model_name=model_name,
        base_url=base_url,
        api_key=api_key,
    )
