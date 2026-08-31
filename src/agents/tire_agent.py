"""Tire Agent: src/agents/tire_agent.py

Extracted from N26_tire_agent.ipynb. Wraps the per-compound TireDegTCN models
(N09/N10) in a LangGraph ReAct agent that answers: how many laps remain before
the degradation cliff?

Public API
----------
run_tire_agent(stint_state)                   -> TireOutput  (FastF1 session in stint_state)
run_tire_agent_from_state(lap_state, laps_df) -> TireOutput  (RSM adapter, no FastF1 session)

Module-level singletons
-----------------------
CFG : TireAgentConfig: loads routing, calibration, encoding maps, cliff thresholds.
      Kept at module level so TireOutput.__post_init__ can call CFG.get_cliff_thresholds.
      Model bundles (BUNDLES) are loaded lazily inside TireAgent.__init__ to avoid
      expensive I/O at import time.
"""

from __future__ import annotations

import importlib.util
import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.agents._shared_defaults import (
    DEFAULT_AIR_TEMP_C,
    DEFAULT_TOTAL_LAPS,
    DEFAULT_TRACK_TEMP_C,
    LLM_MAX_RETRIES,
    reading_or_default,
)
from src.agents.race_state_builder import UNKNOWN_TYRE_LIFE, normalise_compound
from src.agents.tire_parsing import parse_tool_outputs

# ── Repo root (module-relative) ───────────────────────────────────────────────
# Walker with a root-stop guard to avoid spinning forever when the module is
# imported from outside a git checkout (e.g. uv tool install).
_REPO_ROOT = Path(__file__).resolve().parent
while not (_REPO_ROOT / ".git").exists():
    if _REPO_ROOT.parent == _REPO_ROOT:
        break
    _REPO_ROOT = _REPO_ROOT.parent

# Prefer f1_strat_manager.data_cache.get_data_root() so the uv tool install
# flow (where data lives under ~/.f1-strat/) works transparently; fall back
# to the repo-relative layout when the helper is not importable.
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

from src.f1_strat_manager.gp_slugs import resolve_gp_key  # noqa: E402

logger = logging.getLogger(__name__)

_MODEL_DIR = _DATA_ROOT / "models" / "tire_degradation"
_PROCESSED = _DATA_ROOT / "processed"
_AGENTS_DIR = _DATA_ROOT / "models" / "agents"


# ─────────────────────────────────────────────────────────────────────────────
# TireDegTCN: reproduced from N10 (different state dict layout from legacy N09)
# ─────────────────────────────────────────────────────────────────────────────


class CausalConv1dBlock(nn.Module):
    """Single causal dilated convolution layer with left-side padding.

    Uses manual left-side padding instead of PyTorch's built-in padding to
    guarantee strict causality: no future timestep information leaks into
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

    def __init__(
        self, in_ch: int, out_ch: int, kernel_size: int, dilation: int, dropout: float = 0.1
    ):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
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
        ch: Number of channels (equal for input and output, no projection).
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
    N_MC forward passes, see TireAgent._build_tools.

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
        self.input_proj = nn.Linear(n_features, d_model)
        self.blocks = nn.ModuleList(
            [TCNResidualBlock(d_model, kernel_size, 2**i, dropout) for i in range(n_layers)]
        )
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

    Resolves all model paths relative to the repo root and loads (once) the
    three JSON artefacts produced by N10: routing config (compound → bundle file
    + window), MC Dropout calibration (per-compound uncertainty sigma), and
    encoding maps (label encodings for Team, Compound, AbsoluteCompound). Also
    loads the circuit cluster map from the k=4 parquet (N05) and the cluster-aware
    cliff thresholds from tire_agent_config_v1.json (written at N26 Step 6).

    Attributes:
        n_mc: Number of Monte Carlo Dropout forward passes per inference call.
            50 passes give a stable P10/P50/P90 interval without excessive latency.
        mc_seed: Torch seed applied before the MC Dropout passes, so two identical
            race states yield an identical TireOutput (#735).

            The choice is deliberate rather than incidental, because the opposite
            is also arguable: MC Dropout exists to express epistemic uncertainty,
            and a fixed seed makes that uncertainty identical on every call. What
            settles it is what the interval is FOR here. Downstream wants a
            P10/P50/P90 band, which is a property of the model and the input; the
            run-to-run wobble is sampling noise in the ESTIMATE of that band, not
            a second source of uncertainty. Seeding removes the noise and leaves
            the band untouched.

            The rest of the layer already assumes this: the Monte Carlo seeds its
            own RNG at 42, shares draws across candidates, and is pinned by
            byte-identical goldens. Leaving one stochastic step upstream made the
            whole chain non-reproducible: measured at 53/57 and 52/57 on two
            identical captures of the same commit.

            ``src/strategy/eval/tire_holdout.py`` made the same choice for the
            holdout sweep and takes its seed as an explicit argument. That is the
            twin; if this decision is ever revisited, revisit both.
        model_name: LM Studio local model identifier for the ReAct agent LLM.
        cliff_pit_soon_laps: Global fallback threshold below which warning_level
            is PIT_SOON. Per-cluster values take precedence when available.
        cliff_monitor_laps: Global fallback threshold below which warning_level
            is MONITOR. Per-cluster values take precedence when available.
        fresh_reference_tyre_life: Tyre life that stands in for "fresh" when asking
            the model what this set was worth before it wore. N04 defines the target
            against the stint's own lowest-tyre-life lap, which is an out-lap or a
            standing start, so the level is biased slow and cannot be charged as a
            cost directly. Asking the same model on the same stint's early laps
            cancels that baseline algebraically instead of approximately.
        fresh_reference_max_pct_of_fastest: Reject a fresh-reference candidate lap
            whose ``lap_time_pct_of_race_fastest`` exceeds this ratio. ``track_status_
            clean`` (see ``_add_session_cols``) is supposed to be the signal for a
            Safety-Car- or red-flag-affected lap, but it is dead on the shipping
            path: a constant 0 across every featured parquet, because N04's
            ``IsAccurate`` gate does not catch every neutralised lap (measured
            counter-example: Mexico City 2023 car 4, lap 36, 137.8 s on a circuit
            whose green-flag pace is ~83 s, ``track_status_clean == 0`` regardless).
            When such a lap lands as the fresh reference, the model correctly
            predicts it as an outlier and every later lap in the stint reads as
            tens of seconds "faster than fresh", not tyre wear, an artefact of a
            contaminated zero point. ``lap_time_pct_of_race_fastest`` is already a
            TCN input feature, computed unconditionally and cheaply from
            ``session_meta['fastest_lap_s']``, so it is available at the same point
            ``track_status_clean`` should have been. Threshold measured over 31,624
            training-season laps: gating at 1.10 cuts the deg_cost_s error bound's
            mean absolute error from 0.650 to 0.434 s/lap and its signed bias from
            +0.351 to +0.139, at the cost of 49 of 1714 stints (2.9%) losing their
            reference entirely and falling back to ``None`` (``scripts/
            measure_deg_error_bound.py``, ``documents/audits/MEASURE_fresh_reference_
            quality_gate.md``).
        deg_cost_floor_s / deg_cost_ceiling_s: Bounds on the referenced wear, in
            seconds per lap. MEASURED, not chosen: they are the 1st and 99th
            percentiles over 31,624 training-season laps
            (``scripts/measure_tyre_reference.py``). The raw quantity reaches
            +-15 s/lap because a handful of stints have a Safety Car or an out-lap
            as their N04 baseline, and one of those reaching the scorer prices a
            single lap like ten positions.

            Do NOT tighten these to ``CLIFF_LOSS = 0.80``. The measured median at
            20-25 laps of tyre life is 1.03 s/lap, so that bound would delete the
            signal rather than the outliers. These are a guard against nonsense,
            not an opinion about how much a worn tyre costs.
    """

    n_mc: int = 50
    mc_seed: int = 42
    model_name: str = "gpt-4.1-mini"
    cliff_pit_soon_laps: int = 3
    cliff_monitor_laps: int = 7
    fresh_reference_tyre_life: int = 3
    fresh_reference_max_pct_of_fastest: float = 1.10
    deg_cost_floor_s: float = -2.33
    deg_cost_ceiling_s: float = 3.67

    def __post_init__(self) -> None:
        self._model_dir = _MODEL_DIR
        self.export_dir = _AGENTS_DIR
        try:
            self.export_dir.mkdir(parents=True, exist_ok=True)
        except OSError:
            pass  # read-only mount in Docker

        self.routing_cfg = self._load_routing_cfg()
        self.mc_calibration, self.mc_sigma_fallback = self._load_mc_calibration()
        self._load_encoding_maps()
        self.circuit_cluster_map = self._load_circuit_clusters()
        self._load_cliff_thresholds()

    def _load_routing_cfg(self) -> dict:
        """Load routing_config.json: compound ID → bundle filename + window size."""
        with open(self._model_dir / "routing_config.json") as f:
            return json.load(f)

    def _load_mc_calibration(self) -> tuple[dict, float]:
        """Load MC Dropout calibration JSON and compute cross-compound sigma fallback.

        mc_dropout_calibration.json stores per-compound mean_sigma_s values fitted
        in N10. The fallback sigma is the mean across all compounds: used when
        compound_id is absent from the dict (currently C1 and C3, whose sigmas are
        not yet fitted; C2/C4/C5/C6 are present). Regenerating them needs N09's MC
        Dropout calibration cell, so the cross-compound mean stands in until then.

        Returns:
            Tuple (calibration_dict, sigma_fallback).
        """
        with open(self._model_dir / "mc_dropout_calibration.json") as f:
            mc_cal = json.load(f)
        fallback = float(np.mean([v["mean_sigma_s"] for v in mc_cal.values()]))
        return mc_cal, fallback

    def _load_encoding_maps(self) -> None:
        """Load encoding_maps.json and set four label-encoding dicts as instance attrs."""
        with open(self._model_dir / "encoding_maps.json") as f:
            enc = json.load(f)
        self.team_id_map: dict = enc["team_id"]
        self.compound_id_map: dict = enc["compound_id"]
        self.abs_compound_id_map: dict = enc["absolute_compound_id"]
        self.compound_hardness_map: dict = enc["compound_hardness"]

    # The per-cluster mean lap time N04 subtracted to build `lap_time_vs_cluster_mean`,
    # which is a TCN input. MEASURED, not chosen: recovered from `laps_tiredeg.parquet`
    # as `LapTime_s - lap_time_vs_cluster_mean`, which comes out to ONE value per
    # cluster with standard deviation exactly 0.0. `tests/agents/test_tire_cluster_mean.py`
    # re-derives them and fails if they drift.
    #
    # Hardcoded rather than read at runtime because neither source ships on a clean
    # install: `_build_allow_patterns` pulls only `laps_featured_2025.parquet`, and
    # reading an artefact that is not downloaded is the #798 failure class.
    #
    # THE FAMILY MATTERS. N07 builds `laps_tiredeg` by reading the COMBINED
    # `laps_featured.parquet` (`.nb_py/N07_tiredeg_eda.py:92`) and inheriting its
    # cluster columns, so the TCN trained on the POOLED clustering. That is the one
    # `circuit_clusters_k4.parquet` carries, and it agrees with `laps_tiredeg` on 24 of
    # 24 GPs, where `circuit_clusters_k4_2025.parquet` agrees on only 17.
    _TRAINED_CLUSTER_MEAN_LAP_S: ClassVar[dict[int, float]] = {
        0: 100.92462574340107,
        1: 95.43860940701592,
        2: 84.6488860957042,
        3: 81.36461922886834,
    }

    def _load_circuit_clusters(self) -> dict:
        """Load k=4 circuit cluster parquet and return GP_Name → Cluster int dict."""
        cluster_df = pd.read_parquet(
            _PROCESSED / "circuit_clustering" / "circuit_clusters_k4.parquet"
        )
        return dict(zip(cluster_df["GP_Name"], cluster_df["Cluster"].astype(int)))

    def _load_cliff_thresholds(self) -> None:
        """Load cluster-aware and GP-level cliff thresholds from tire_agent_config_v1.json.

        Falls back to empty dicts (global thresholds only) when the file does not
        exist yet: this covers the case where N26 Step 6 has not been run.
        """
        cfg_path = self.export_dir / "tire_agent_config_v1.json"
        if cfg_path.exists():
            with open(cfg_path) as f:
                agent_cfg = json.load(f)
            cat = agent_cfg.get("cluster_aware_thresholds", {})
            self.cliff_pit_soon_by_cluster: dict = {
                int(k): v for k, v in cat.get("pit_soon_by_cluster", {}).items()
            }
            self.cliff_monitor_by_cluster: dict = {
                int(k): v for k, v in cat.get("monitor_by_cluster", {}).items()
            }
            self.cliff_overrides_by_gp: dict = cat.get("overrides_by_gp", {})
        else:
            self.cliff_pit_soon_by_cluster = {}
            self.cliff_monitor_by_cluster = {}
            self.cliff_overrides_by_gp = {}

    def cluster_for(self, gp_name: str, default: Optional[int] = None) -> Optional[int]:
        """This circuit's cluster, whichever of its four spellings the caller holds.

        The map is keyed by the parquet slug ('Miami'); the replay path queries with the
        metadata name ('Miami Gardens'). Today that particular query still lands, but only
        because the pooled clustering artefact carries the race TWICE under both spellings
        (25 rows for 24 circuits) - a duplicate PR 6 removes, at which point an unresolved
        query would start defaulting silently. See PR3_GP_KEYSPACE_SWEEP.md.
        """
        return self.circuit_cluster_map.get(
            resolve_gp_key(self.circuit_cluster_map, gp_name), default
        )

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
        override_key = resolve_gp_key(self.cliff_overrides_by_gp, gp_name)
        if override_key in self.cliff_overrides_by_gp:
            ov = self.cliff_overrides_by_gp[override_key]
            return ov["pit_soon"], ov["monitor"]
        cluster_id = self.cluster_for(gp_name)
        if cluster_id is not None:
            pit_soon = self.cliff_pit_soon_by_cluster.get(cluster_id, self.cliff_pit_soon_laps)
            monitor = self.cliff_monitor_by_cluster.get(cluster_id, self.cliff_monitor_laps)
            return pit_soon, monitor
        return self.cliff_pit_soon_laps, self.cliff_monitor_laps

    def load_bundle(self, compound_id: str) -> dict:
        """Load a compound .pt bundle and attach an instantiated TireDegTCN in eval mode.

        Each .pt file is a self-contained dict from N10: state dict, fitted
        StandardScaler, feature name list, window size, and architecture hparams.
        """
        cfg = self.routing_cfg[compound_id]
        bundle = torch.load(
            self._model_dir / cfg["bundle"],
            map_location="cpu",
            weights_only=False,
        )
        model = TireDegTCN(bundle["n_features"], **bundle["model_hparams"])
        model.load_state_dict(bundle["state_dict"])
        model.eval()
        bundle["model"] = model
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
# 75% of stints had already pitted by this level: a practical proxy for the cliff.
CLIFF_THRESHOLD: dict[str, int] = {
    "C1": 3,  # p75 = 2.20 -> ceil = 3
    "C2": 2,  # p75 = 1.74 -> ceil = 2
    "C3": 2,  # p75 = 1.96 -> ceil = 2
    "C4": 2,  # p75 = 1.75 -> ceil = 2
    "C5": 2,  # p75 = 1.43 -> ceil = 2
    "C6": 2,  # p75 = 1.82 -> ceil = 2
}

# Longest race on the current calendar (Monaco, 78 laps; max observed across
# 2023-2025 data). Fallback ceiling for laps-to-cliff when session_meta carries no
# total_laps: a cliff beyond the longest possible race is "not this race" regardless
# of circuit, so clamping there caps absurd values without touching any real one. The
# earlier default of 100 sat above every race, so the clamp it belonged to never fired.
# session_meta.total_laps is present on every shipping path, so this only guards
# hand-built states.
MAX_RACE_LAPS: int = 78


def _reject_contaminated_laps(
    laps: pd.DataFrame,
    fastest_lap_s: float,
    max_pct: float,
) -> pd.DataFrame:
    """Drop candidate fresh-reference laps slower than ``max_pct`` times the
    race's fastest lap -- a Safety-Car or red-flag-affected lap that
    ``track_status_clean`` should flag but does not (see
    ``TireAgentConfig.fresh_reference_max_pct_of_fastest``).

    Pure and leaf-level so the threshold is testable without model weights --
    the same split ``tire_parsing.py`` made, and for the same reason.
    """
    return laps[laps["LapTime_s"] <= max_pct * fastest_lap_s]


def _referenced_wear(parsed: dict) -> Optional[float]:
    """Seconds per lap this set costs versus fresh, bounded, or ``None``.

    Both halves must have parsed. ``.get`` with a numeric default would turn a
    missing reference into "this tyre is exactly at its fresh pace", which is a
    reading the scorer can also legitimately receive.
    """
    if "cum_deg" not in parsed or "fresh_ref" not in parsed:
        return None

    wear = parsed["cum_deg"] - parsed["fresh_ref"]
    bounded = min(max(wear, CFG.deg_cost_floor_s), CFG.deg_cost_ceiling_s)
    return round(bounded, 4)


# ─────────────────────────────────────────────────────────────────────────────
# TireOutput
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class TireOutput:
    """Structured output of the Tire Agent for one driver at one point in the race.

    The TCN produces a single scalar (predicted cumulative degradation) per forward
    pass. From N_MC MC passes, deg_rate and the P10/P50/P90 interval are derived for
    laps remaining before the cliff. warning_level is derived in __post_init__ so
    downstream agents (N28, N31) get a categorical signal without re-implementing thresholds.

    Attributes:
        compound: Pirelli compound ID string (e.g. 'C2', 'C4'). Passed through
            from stint_state and used for MC Dropout calibration lookup.
        current_tyre_life: Laps completed on this tyre set at inference time.
            Used by N28 Pit Strategy as baseline for undercut feature construction.
        deg_rate: Predicted degradation rate in seconds per lap (median of MC passes).

            Read this before using it as a tyre-wear signal: it is the last row of
            the RAW lap-to-lap derivative, not a fuel-corrected one, and fuel
            burn-off pushes lap times down at roughly the rate wear pushes them
            up. Measured over 110 real laps it has median +0.006 s/lap, is
            negative on 43 of them, and correlates +0.115 with tyre life: its
            median by tyre-life band is not even monotonic. It does not separate
            a worn tyre from a fresh one. ``cumulative_deg_s`` is the field that
            does (#727).
        cumulative_deg_s: The TCN's own prediction, seconds per lap this set is
            slower than it was when fresh, fuel-corrected (N04's
            ``FuelAdjustedDegAbsolute``). ``None`` when no TCN ran or the tool
            output did not parse.

            ``None`` and not ``0.0``, deliberately: 0.0 is a legitimate reading
            here (a tyre at its baseline pace), so a sentinel of 0.0 would be a
            value the code can also genuinely find. ``deg_rate`` already
            demonstrates the collision: 12 of those 110 laps carry a parse miss
            indistinguishable from a real zero.

            This is the scalar the whole tyre-degradation model family exists to
            produce, and until #727 it was computed on every call, printed into
            the tool string, and dropped at the parser, so it reached neither
            the Monte Carlo, nor the orchestrator prompt, nor any UI. Measured
            over the same 110 laps it correlates +0.369 with tyre life and swings
            0.411 s/lap across a stint.

            #727 restored the first two. The UI half stayed true until #1041:
            PITWALL's TIRE console now prints this and ``deg_cost_s`` in its
            model detail (``src/pitwall/reasoning_lines.py``), which is the only
            surface that renders either.
        deg_cost_s: Seconds per lap staying out costs versus a fresh set, bounded.
            This is ``cumulative_deg_s`` minus the model's own prediction on this
            stint's early laps, which cancels N04's per-stint baseline instead of
            approximating it (see ``TireAgent._fresh_reference`` for why a pooled
            per-compound table was measured and refused).

            **This is the field the scorers consume**, not ``cumulative_deg_s``:
            the level carries a per-stint offset whose standard deviation across
            stints is 5.48 s, so it is not comparable between cars or laps.

            ``None`` when either side is missing, never 0.0, for the same reason
            given above for ``cumulative_deg_s``: a genuinely fresh set reads 0.0
            here, so the sentinel would collide with a real value. A ``None``
            leaves both scorers on the ``FRESH_GAIN`` fallback.
        laps_to_cliff_p10: Pessimistic estimate (P10) of laps before the cliff.
            Drives PIT_SOON warning: conservative to avoid running too long.
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
    gp_name: str = ""
    cumulative_deg_s: float | None = None
    deg_cost_s: float | None = None
    warning_level: str = field(init=False)
    reasoning: str = ""

    def __post_init__(self) -> None:
        pit_soon, monitor = CFG.get_cliff_thresholds(self.gp_name)
        if self.laps_to_cliff_p10 < pit_soon:
            self.warning_level = "PIT_SOON"
        elif self.laps_to_cliff_p10 < monitor:
            self.warning_level = "MONITOR"
        else:
            self.warning_level = "OK"


# ─────────────────────────────────────────────────────────────────────────────
# Feature pipeline helpers (must match N10 training order exactly)
# Pure functions: receive all required state as arguments, read no globals.
# ─────────────────────────────────────────────────────────────────────────────


def _add_timing_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Convert FastF1 Timedelta columns to float seconds.

    Handles two source formats:
    - FastF1 raw: LapTime/Sector*Time are pandas Timedelta objects
    - Featured parquet: LapTime_s/Sector*_s are already plain floats

    LapsSincePitStop is left alone when the frame already carries it, which both
    featured artefacts do. Overwriting it unconditionally with TyreLife is wrong:
    the two are different quantities. TyreLife is the age of the tyre SET and
    counts laps run before this race, so they coincide only when the set was
    fitted at the last stop, and an unconditional alias disagreed with the
    trained column on 15.8% of rows. Same defect as #800 on the pace path; this
    is its twin on the tire path.

    A frame that lacks the column still gets the alias, because a missing
    feature would break the scaler outright, and the alias is right four rows in five.
    """

    def _to_seconds(df, td_col, s_col):
        if s_col in df.columns:
            df[s_col] = pd.to_numeric(df[s_col], errors="coerce")
        elif td_col in df.columns:
            val = df[td_col]
            if hasattr(val.iloc[0] if len(val) > 0 else None, "total_seconds"):
                df[s_col] = val.dt.total_seconds()
            else:
                df[s_col] = pd.to_numeric(val, errors="coerce")
        else:
            df[s_col] = float("nan")
        return df

    df = _to_seconds(df, "LapTime", "LapTime_s")
    df = _to_seconds(df, "Sector1Time", "Sector1_s")
    df = _to_seconds(df, "Sector2Time", "Sector2_s")
    df = _to_seconds(df, "Sector3Time", "Sector3_s")
    if "LapsSincePitStop" not in df.columns:
        df["LapsSincePitStop"] = df["TyreLife"]
    return df


def _add_weather_cols(df: pd.DataFrame, session_meta: dict) -> pd.DataFrame:
    """Ensure weather columns exist; fill from session_meta race averages if absent."""
    for col in ("AirTemp", "TrackTemp", "Humidity", "Rainfall"):
        if col not in df.columns:
            df[col] = session_meta.get(col, 0.0)
    return df


def _add_prev_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Shift timing and speed measurements back one lap to create prev-lap context.

    A row with no predecessor keeps NaN, which is what the TCN trained on: N10's
    scaler does `fillna(0)` (`.nb_py/N10_tiredeg_compound_finetuning.py:176,181`),
    so the model learned a raw zero there, not a repeat of the current lap.

    This used to fill the gap with the CURRENT lap's own value, described as "first
    lap of a stint has no predecessor". Two things were wrong with that. The frame is
    not grouped by stint here, so the fill actually hit only the first row of the
    whole frame plus any row whose source was itself missing; and substituting the
    current lap makes `Prev_X - X` read exactly zero where training read whatever the
    scaled zero mapped to. It moved the TCN's output by a mean of 0.198 s.

    Must run before _add_delta_cols.
    """
    for new_col, src_col in [
        ("Prev_LapTime", "LapTime_s"),
        ("Prev_SpeedFL", "SpeedFL"),
        ("Prev_SpeedI1", "SpeedI1"),
        ("Prev_SpeedI2", "SpeedI2"),
        ("Prev_SpeedST", "SpeedST"),
        ("Prev_TyreLife", "TyreLife"),
    ]:
        df[new_col] = df[src_col].shift(1)
    return df


def _add_laptime_delta(df: pd.DataFrame) -> pd.DataFrame:
    """Compute lap time first and second derivative features.

    LapTime_Delta: LapTime_s[i] - LapTime_s[i-1]. Requires Prev_LapTime.
    LapTime_Trend: LapTime_Delta[i] - LapTime_Delta[i-1] (second derivative).
    """
    df["LapTime_Delta"] = (df["LapTime_s"] - df["Prev_LapTime"]).fillna(0)
    df["LapTime_Trend"] = (df["LapTime_Delta"] - df["LapTime_Delta"].shift(1)).fillna(0)
    return df


def _add_degradation_rate(df: pd.DataFrame) -> pd.DataFrame:
    """Compute degradation rate and acceleration from a rolling 3-lap polyfit.

    DegradationRate: slope of FuelAdjustedLapTime vs TyreLife over a 3-lap window.
    Captures per-lap pace loss due to tyre wear, corrected for fuel mass.
    Requires FuelAdjustedLapTime (from _add_fuel_cols).

    DegAcceleration: change in DegradationRate between consecutive laps.

    NEITHER IS SHIFTED, because training never shifted them. Both used to carry a
    `.shift(1)` described as a "leakage fix matching N10 training", and that comment
    named a mechanism training never had: N04 computes both unshifted
    (`.nb_py/N04_feature_engineering.py:481-504`) and N09 consumed them as stored
    (`.nb_py/N09_tiredeg_tcn.py:219-220`). Serving a lagged value moved the TCN's
    output by a mean of 0.185 s, concentrated at cliff onset, which is the one place
    the number is read against a threshold.

    The lag was not leakage protection either way: `raw_deg[i]` is fitted over laps
    i-2..i, all of which have already happened at the moment the model is asked about
    lap i. There was nothing from the future to exclude.

    THREE THINGS N04 DOES THAT THIS HAS TO DO TOO, all of them missed when the shift
    came out (`.nb_py/N04_feature_engineering.py:478-556`):

    1. Both arrays start as NaN, not zeros. N04 says so in its own words: "No fillna -
       NaN on first lap of each stint is meaningful signal." A zero is a reading; a NaN
       is the absence of one, and the tensor builder's single `fillna(0)` is what turns
       it into the value the scaler saw.
    2. The acceleration at index 1 is NaN, because N04 writes it only when BOTH
       neighbours are non-NaN and `deg_rates[0]` is always NaN. Computing it as
       `deg[1] - deg[0]` with a zero-filled `deg[0]` served `deg[1]` itself on 326 of
       327 real stints, where training served nothing. The old lagged code was
       accidentally right on that one row, which is why the shift removal made this
       row worse before this fix put it back.
    3. Both are clipped to [-2.0, 2.0] s/lap. Without it the TCN is handed values
       outside the range it was fitted on precisely at cliff and chaos laps, which are
       the laps the number exists to describe.
    """
    tyre_lives = df["TyreLife"].values
    adj_times = df["FuelAdjustedLapTime"].values
    n = len(df)

    raw_deg = np.full(n, np.nan)
    raw_accel = np.full(n, np.nan)

    for i in range(1, n):
        start = max(0, i - 2)
        x = tyre_lives[start : i + 1]
        y = adj_times[start : i + 1]
        if len(x) >= 2 and not np.isnan(y).any():
            raw_deg[i] = np.polyfit(x, y, 1)[0]

    for i in range(1, n):
        if not np.isnan(raw_deg[i]) and not np.isnan(raw_deg[i - 1]):
            raw_accel[i] = raw_deg[i] - raw_deg[i - 1]

    df["DegradationRate"] = pd.Series(raw_deg, index=df.index).clip(-2.0, 2.0)
    df["DegAcceleration"] = pd.Series(raw_accel, index=df.index).clip(-2.0, 2.0)
    return df


def _add_delta_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Orchestrate laptime delta and degradation rate computation."""
    df = _add_laptime_delta(df)
    df = _add_degradation_rate(df)
    return df


def _add_speed_delta_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Compute trap-speed deltas (current minus previous lap) for all four sensors."""
    for sensor in ("FL", "I1", "I2", "ST"):
        df[f"Speed{sensor}_Delta"] = df[f"Speed{sensor}"] - df[f"Prev_Speed{sensor}"]
    return df


def _add_compound_cols(df: pd.DataFrame, compound_id: str) -> pd.DataFrame:
    """Set compound identity features from the label-encoding maps in CFG.

    All three encodings are constant within a stint. CompoundHardness is the
    inverse of AbsoluteCompoundID: C1=6 (hardest), C6=1 (softest), as encoded
    in the N10 training data.

    The defaults (C3 / mid-hardness / SOFT) are real, in-range codes, so an unknown
    compound is silently scored as a real one rather than flagged. That is only reached
    with corrupt or out-of-scope compound data, but it is a sentinel by construction, so
    it is logged loudly rather than left invisible.
    """
    name = str(df["Compound"].iloc[0])
    if compound_id not in CFG.abs_compound_id_map:
        logger.warning("Unknown compound_id %r: encoding as C3 (AbsoluteCompoundID=3)", compound_id)
    if name not in CFG.compound_id_map:
        logger.warning("Unknown compound name %r: encoding as SOFT (CompoundID=1)", name)
    df["AbsoluteCompoundID"] = CFG.abs_compound_id_map.get(compound_id, 3)
    df["CompoundHardness"] = CFG.compound_hardness_map.get(compound_id, 4)
    df["CompoundID"] = CFG.compound_id_map.get(name, 1)
    return df


def _encode_team_id(team_id_map: dict, team: str) -> int:
    """Team label encoding, with the McLaren default made loud.

    `team_id_map.get(team, 4)` defaults to 4, which is a real team (McLaren), so an
    unrecognised team is silently scored as one. This is reachable in normal operation:
    engine.py sets team='Unknown' whenever a lap row is empty, and 'Unknown' is not in
    the map. Log it rather than let a mislabelled team ride as McLaren unnoticed.
    """
    if team not in team_id_map:
        logger.warning("Unknown team %r: encoding as team_id 4 (McLaren default)", team)
    return team_id_map.get(team, 4)


def _add_fuel_cols(df: pd.DataFrame, session_meta: dict) -> pd.DataFrame:
    """Estimate fuel load and cumulative fuel-burn pace gain, matching N04 training.

    FuelLoad: fraction remaining = (total_laps - LapNumber) / total_laps.
    FuelEffect: cumulative gain from fuel burn = (TyreLife - baseline_tyrelife) * 0.055 s/lap.
    FuelAdjustedLapTime: intermediate column needed by _add_degradation_rate.
    """
    total_laps = session_meta["total_laps"]

    if "FuelLoad" not in df.columns:
        df["FuelLoad"] = ((total_laps - df["LapNumber"]) / total_laps).clip(lower=0.0)

    baseline_tyrelife = df["TyreLife"].iloc[0]
    df["FuelEffect"] = (df["TyreLife"] - baseline_tyrelife) * 0.055
    df["FuelAdjustedLapTime"] = df["LapTime_s"] + df["FuelEffect"]
    return df


def _add_session_cols(df: pd.DataFrame, session_meta: dict) -> pd.DataFrame:
    """Normalise lap times against session fastest lap and circuit cluster mean.

    lap_time_pct_of_race_fastest: ratio to the race's fastest lap (~1.04 mean).
    lap_time_vs_cluster_mean: delta vs the cluster's typical lap time (seconds).
    mean_sector_speed: circuit-level mean of the three speed traps (km/h).
    track_status_clean: 3-class int - 0=green, 1=yellow/VSC, 2=SC/red flag.
        DEAD ON THE SHIPPING PATH: this column is uniformly 0 across all
        204366 rows of every published featured parquet, laps_tiredeg.parquet
        included, so the TCN's input for it is a constant. The cause is not a
        broken recode: N04's IsAccurate gate drops neutralised laps, so every
        lap that survives into the featured parquet genuinely IS green. The
        three-class encoding is only reachable from the raw parquet, where the
        neutralised laps still exist. Do not spend time debugging degradation
        behaviour under a Safety Car through this feature; it cannot cause it.

    Two of these columns are trained per-circuit constants, not per-lap values.
    N04 built lap_time_vs_cluster_mean as LapTime_s minus a global per-cluster
    mean lap time (81.36-100.92 s depending on cluster, std 0.0 within a cluster),
    and mean_sector_speed as a per-GP circuit feature (one value per race, std 0.0
    within a GP). Both are TCN inputs listed in tiredeg_feature_manifest.json, and
    both are already shipped by the featured parquet. Recomputing them here from
    the handed-in frame overwrites the trained constant with a per-frame quantity:
    the cluster-mean delta was off by up to 14.9 s per lap (Lusail) and the sector
    speed by up to 17.5 s. They are therefore guarded like FuelLoad and
    track_status_clean: recomputed only when the frame does not already carry them
    (the raw FastF1 path, which has no better source). The sibling
    lap_time_pct_of_race_fastest and laps_remaining are recomputed unconditionally
    because their per-frame recompute reproduces the shipped value exactly
    (0.0000 s delta across all 24 GPs).
    """
    df["lap_time_pct_of_race_fastest"] = df["LapTime_s"] / session_meta["fastest_lap_s"]
    # UNCONDITIONALLY, not guarded on the column's absence. Guarding it kept whatever
    # the handed-in frame carried, and the featured artefacts carry the 2025 clustering
    # while `Cluster` beside it comes from the POOLED map the TCN trained on. Serving one
    # family's delta next to the other family's cluster id is a mix neither model saw:
    # the two disagree on 100% of rows, mean 5.75 s.
    df["lap_time_vs_cluster_mean"] = df["LapTime_s"] - session_meta["cluster_mean_lap_s"]
    df["laps_remaining"] = session_meta["total_laps"] - df["LapNumber"]
    if "mean_sector_speed" not in df.columns:
        df["mean_sector_speed"] = (df["SpeedI1"] + df["SpeedI2"] + df["SpeedFL"]) / 3

    if "track_status_clean" not in df.columns:
        status_map = {"1": 0, "2": 1, "3": 2, "4": 2, "5": 2, "6": 1, "7": 1}
        if "TrackStatus" in df.columns:
            df["track_status_clean"] = (
                df["TrackStatus"].astype(str).map(status_map).fillna(0).astype(int)
            )
        else:
            df["track_status_clean"] = 0
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Stateless helpers
# ─────────────────────────────────────────────────────────────────────────────

# Kept importable under its original private name so existing callers and the
# hardening tests do not move. The body lives in the leaf module because importing
# THIS module builds TireAgentConfig() and therefore needs data/models/ on disk,
# which is what left a pure string parser with no CI coverage (#727).
_parse_tool_outputs = parse_tool_outputs


def _compound_name_to_id(compound_name: str, gp_name: str, year: int) -> str:
    """Map a Pirelli compound name (SOFT/MEDIUM/HARD) to its Cx ID for this GP.

    Loads data/tire_compounds_by_race.json (authoritative source) to resolve
    the Cx allocation for this GP/year. Falls back to C3/C2/C1 if the GP is
    not found: these are the most common mid-season assignments.

    Args:
        compound_name: Pirelli compound name string (e.g. 'SOFT', 'MEDIUM').
        gp_name: GP name matching the tire_compounds_by_race.json keys.
        year: Race year as int.

    Returns:
        Compound ID string such as 'C3'.
    """
    compounds_path = _REPO_ROOT / "data" / "tire_compounds_by_race.json"
    fallback = {"SOFT": "C3", "MEDIUM": "C2", "HARD": "C1", "INTERMEDIATE": "INT", "WET": "WET"}
    if not compounds_path.exists():
        return fallback.get(compound_name.upper(), "C3")

    with open(compounds_path) as f:
        alloc = json.load(f)

    year_data = alloc.get(str(year), {})
    # The JSON is keyed by the parquet slug. Queried with the metadata name, 2025 Miami
    # missed and took the fallback, routing MEDIUM/HARD stints to the C2/C1 TCN bundle
    # instead of C4/C3 for the whole race (PR3_GP_KEYSPACE_SWEEP.md).
    gp_data = year_data.get(resolve_gp_key(year_data, gp_name), {})
    return gp_data.get(compound_name.upper(), fallback.get(compound_name.upper(), "C3"))


# ─────────────────────────────────────────────────────────────────────────────
# LangGraph / LangChain optional imports
# ─────────────────────────────────────────────────────────────────────────────

# lc_tool stays eager: the @lc_tool decorators further down run at import time.
# ChatOpenAI and create_agent do not, and importing them here cost 7.2 s measured,
# because langchain_openai drags transformers and the langgraph stack in behind
# it. This module is imported by the orchestrator, so every CLI invocation paid
# that, `--help` included, whether or not the ReAct agent was ever built. The two
# names are now imported in get_react_agent(), which already creates its client
# on first call, and find_spec answers the availability question without
# executing anything.
try:
    from langchain_core.tools import tool as lc_tool

    _LC_CORE_AVAILABLE = True
except ImportError:
    _LC_CORE_AVAILABLE = False

_LANGGRAPH_AVAILABLE = (
    _LC_CORE_AVAILABLE
    and importlib.util.find_spec("langchain_openai") is not None
    and importlib.util.find_spec("langchain.agents") is not None
)


# ─────────────────────────────────────────────────────────────────────────────
# System prompt (module-level constant, unchanged from N26)
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
# TireAgent: encapsulated agent class
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
        self.bundles: dict = self.cfg.load_all_bundles()
        self.laps_df: pd.DataFrame = pd.DataFrame()
        self.session_meta: dict = {}
        # Driver codes actually on track for the CURRENT loaded state (#476). Reset
        # on every run()/run_from_state() call: this instance is a process-level
        # singleton reused across many laps/drivers, so a stale set from a previous
        # call must never leak into the next one's tool-arg validation.
        self._live_drivers: Optional[set] = None
        self._react_agent = None
        self._tools: list = self._build_tools()

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

        df["Cluster"] = session_meta["cluster_id"]
        df["TeamID"] = session_meta["team_id"]
        df["Year"] = session_meta["year"]

        return df[self.bundles[compound_id]["feature_names"]].astype(float)

    def _fresh_reference(self, driver: str, compound_id: str) -> Optional[float]:
        """What the model said about this same set before it wore, in seconds.

        A second deterministic pass over the stint's own early laps. Subtracting it
        from the current prediction gives *seconds per lap this set costs versus
        fresh*, which is what a scorer needs and what the raw level is not: N04
        measures against the stint's slowest lap, so the level is negative on most
        laps and charging it directly would pay a car for staying out.

        WHY THIS AND NOT A COMMITTED PER-COMPOUND TABLE
        -----------------------------------------------
        Because the baseline is per stint, not per compound. Measured over 2,343
        training-season stints, the per-stint median target has a standard deviation
        of 5.48 s and a 1st percentile of -32.87 s: 13x the 0.411 s/lap signal being
        chased. A pooled constant cannot normalise 2,343 different zero points, and
        measurement put it at Spearman +0.188 against +0.191 for having no reference
        at all. This one measures +0.308 and 73.9% non-negative
        (``scripts/measure_tyre_reference.py``, ``documents/audits/MEASURE_744a_*``).

        Returns ``None``, never 0.0, when the stint has no lap at the reference tyre
        life: a replay that starts mid-stint. Zero is a legitimate reading of the
        wear it feeds, so collapsing "unknown" into it would be the same sentinel
        collision that #436 fixed for the cliff.

        Also drops any candidate lap slower than ``cfg.fresh_reference_max_pct_of_
        fastest`` times the race's fastest lap, a Safety-Car or red-flag-affected
        lap that ``track_status_clean`` should have flagged but does not (see the
        config docstring). Returns ``None`` rather than falling back to a
        contaminated lap when every candidate is rejected, for the same reason: a
        wrong reference is worse than none.
        """
        early_laps = self._get_driver_stint(driver, self.cfg.fresh_reference_tyre_life)
        if early_laps is None or not len(early_laps):
            return None

        # `_get_driver_stint` returns the raw slice of `self.laps_df` as-is, so
        # `LapTime_s` does not exist yet on the FastF1 `run()` path (raw `LapTime`
        # Timedelta) -- it is only created by `_add_timing_cols`, the first step
        # inside `_build_stint_tensor`. Reuse it here rather than assume a column
        # that `_build_stint_features` itself only guarantees after this point.
        early_laps = _reject_contaminated_laps(
            _add_timing_cols(early_laps),
            self.session_meta["fastest_lap_s"],
            self.cfg.fresh_reference_max_pct_of_fastest,
        )
        if not len(early_laps):
            return None

        tensor = self._build_stint_tensor(early_laps, compound_id, self.session_meta)
        model = self.bundles[compound_id]["model"]
        with torch.no_grad():
            model.eval()
            return float(model(tensor).item())

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
        window = bundle["window"]

        feat_df = self._build_stint_features(stint_laps, compound_id, session_meta)
        scaled = bundle["scaler"].transform(feat_df.fillna(0))

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
        driver_laps = self.laps_df.loc[self.laps_df["Driver"] == driver]
        compound = self.session_meta.get(
            f"{driver}_compound",
            driver_laps["Compound"].iloc[-1] if len(driver_laps) > 0 else "MEDIUM",
        )
        mask = (
            (self.laps_df["Driver"] == driver)
            & (self.laps_df["Compound"] == compound)
            & (self.laps_df["TyreLife"] <= tyre_life)
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
        current_stint = self.session_meta.get(f"{driver}_stint")
        if current_stint is not None and "Stint" in self.laps_df.columns:
            mask &= self.laps_df["Stint"] == current_stint

        current_lap = self.session_meta.get("current_lap")
        if current_lap is not None and "LapNumber" in self.laps_df.columns:
            mask &= self.laps_df["LapNumber"] <= current_lap

        stint = self.laps_df[mask].sort_values("LapNumber")
        return stint if len(stint) > 0 else None

    # ── Live-driver guard (#476) ───────────────────────────────────────────────

    def _live_drivers_at_current_lap(self) -> Optional[set]:
        """Return driver codes on track for the currently loaded state, or None.

        run_from_state() sets self._live_drivers from the RSM's rivals list plus
        the driver's own car: both are PRESENCE-based (a row exists for this lap), the
        same signal race_state_manager.py and pit_strategy_agent.py rely on for
        their own retired-car guards (#470/#462): an age/lap-count cutoff cannot
        separate a retiree from a finisher because the ranges overlap (a finisher
        can go 20 laps without a row, a retirement can show up in 9). The FastF1
        run() path has no per-lap rivals list, so it falls back to every driver
        present anywhere in laps_df.

        Returns:
            Set of driver codes, or None when there is nothing to validate against
            (callers treat None as "cannot tell" and skip the guard rather than
            block every driver on missing data, the same convention
            pit_strategy_agent.py uses for its own `_live_drivers`, #470/#462).
        """
        if self._live_drivers is not None:
            return self._live_drivers
        if len(self.laps_df) > 0 and "Driver" in self.laps_df.columns:
            drivers = set(self.laps_df["Driver"].dropna().unique())
            return drivers or None
        return None

    def _validate_driver_on_track(self, driver: str) -> Optional[str]:
        """Return an error string if `driver` is not on track right now, else None.

        Guards the two LLM-facing tools against a hallucinated or long-retired
        driver code. Reachable in production: laps_df carries the WHOLE race's
        history, so the stint filter in _get_driver_stint happily builds a
        'stint' out of laps recorded before a car crashed: it never checks
        whether the driver is still racing at the lap the agent is currently
        analysing. Example (#476): Austin 2024, HAM crashed on lap 2 of 56;
        asked about at a later lap, the tool used to return a confident
        'P50: 21867.1' instead of erroring.

        Args:
            driver: FastF1 driver abbreviation as passed by the LLM tool call.

        Returns:
            An error string (do not compute) when the driver is not in the live
            set, or when the loaded current_lap falls outside [1, total_laps].
            None when the driver checks out and the tool should proceed.
        """
        live = self._live_drivers_at_current_lap()
        lap = self.session_meta.get("current_lap")
        total_laps = self.session_meta.get("total_laps")
        lap_out_of_range = (
            lap is not None and total_laps is not None and not (1 <= lap <= total_laps)
        )
        if (live is not None and driver not in live) or lap_out_of_range:
            lap_display = lap if lap is not None else "unknown"
            valid = sorted(live) if live is not None else []
            return f"error: '{driver}' is not on track at lap {lap_display}; valid: {valid}"
        return None

    # ── LangChain tool factory ────────────────────────────────────────────────

    def _build_tools(self) -> list:
        """Build LangChain tools as closures over this TireAgent instance.

        Each tool reads self.laps_df, self.session_meta, self.bundles, and
        self.cfg at call time: no module-level globals are accessed. Returns
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
                Returns an error string if no laps are found, or if the driver is not
                on track at the currently loaded lap (#476).
            """
            guard_error = agent._validate_driver_on_track(driver)
            if guard_error:
                return guard_error

            stint = agent._get_driver_stint(driver, tyre_life)
            if stint is None:
                return f"No laps found for driver {driver} with tyre_life <= {tyre_life}."

            tensor = agent._build_stint_tensor(stint, compound_id, agent.session_meta)
            model = agent.bundles[compound_id]["model"]

            with torch.no_grad():
                model.eval()
                pred = model(tensor).item()

            feat_df = agent._build_stint_features(stint, compound_id, agent.session_meta)
            deg_rate = float(feat_df["DegradationRate"].iloc[-1])

            reference = agent._fresh_reference(driver, compound_id)
            reference_line = "" if reference is None else f" | Fresh reference: {reference:.3f} s"

            return (
                f"Driver {driver} | Compound {compound_id} | TyreLife {tyre_life}\n"
                f"Cumulative degradation: {pred:.3f} s | Degradation rate: {deg_rate:.4f} s/lap"
                f"{reference_line}"
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
                Returns an error string if no laps are found, or if the driver is not
                on track at the currently loaded lap (#476).
            """
            guard_error = agent._validate_driver_on_track(driver)
            if guard_error:
                return guard_error

            stint = agent._get_driver_stint(driver, tyre_life)
            if stint is None:
                return f"No laps found for driver {driver} with tyre_life <= {tyre_life}."

            tensor = agent._build_stint_tensor(stint, compound_id, agent.session_meta)
            model = agent.bundles[compound_id]["model"]
            model.train()  # keep dropout active for MC
            # Seed the dropout masks so identical inputs give an identical band
            # (#735). Order mirrors the holdout twin in src/strategy/eval/
            # tire_holdout.py::mc_dropout_global_sigma; the reasoning for seeding
            # at all is on TireAgentConfig.mc_seed.
            torch.manual_seed(agent.cfg.mc_seed)

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
            mc_std = float(np.std(preds))
            sigma = (
                float(agent.cfg.mc_calibration[compound_id]["mean_sigma_s"])
                if compound_id in agent.cfg.mc_calibration
                else agent.cfg.mc_sigma_fallback
            )
            total_std = np.sqrt(mc_std**2 + sigma**2)

            feat_df = agent._build_stint_features(stint, compound_id, agent.session_meta)
            deg_rate = max(float(feat_df["DegradationRate"].abs().iloc[-1]), 0.001)

            threshold = CLIFF_THRESHOLD.get(compound_id, 2.5)
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
            # like a reading. Nothing is invented: the bound is the race itself.
            #
            # The fallback must stay at or below the shortest real race, otherwise a
            # missing total_laps lifts the ceiling above the race and the clamp stops
            # clamping. MAX_RACE_LAPS (78) is the longest race on the calendar; the
            # earlier default of 100 exceeded every race, so an absent key silently
            # disabled the clamp for any race up to 100 laps.
            cliff_ceiling = float(
                agent.session_meta.get("total_laps", MAX_RACE_LAPS) or MAX_RACE_LAPS
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
                f"Driver {driver} | Compound {compound_id} | TyreLife {tyre_life}\n"
                f"Laps to cliff — P10: {to.laps_to_cliff_p10} | P50: {to.laps_to_cliff_p50} | P90: {to.laps_to_cliff_p90}\n"
                f"Degradation rate: {deg_rate:.4f} s/lap | MC std: {mc_std:.4f} s | Calibrated sigma: {sigma:.4f} s\n"
                f"Warning level: {to.warning_level}"
            )

        return [predict_tire_deg_tool, estimate_laps_to_cliff_tool]

    # ── LangGraph agent (lazy) ────────────────────────────────────────────────

    def get_react_agent(
        self,
        provider: str = None,
        model_name: str = "gpt-4.1-mini",
        base_url: str = "http://localhost:1234/v1",
        api_key: str = "lm-studio",
    ):
        """Return the LangGraph ReAct agent, creating it on the first call (lazy).

        Avoids connecting to the LLM at import time: the graph is compiled only
        when N31 or a test actually invokes the agent.

        Args:
            provider: 'lmstudio' (default) or 'openai'.
            model_name: Model identifier for ChatOpenAI.
            base_url: Base URL for LM Studio (ignored when provider='openai').
            api_key: API key; use 'lm-studio' for local server.

        Returns:
            LangGraph CompiledGraph: invoke with {"messages": [{"role": "user", "content": ...}]}.

        Raises:
            ImportError: When LangGraph / LangChain are not installed.
        """
        if not _LANGGRAPH_AVAILABLE:
            raise ImportError(
                "LangGraph / LangChain not installed. "
                "Install with: pip install langgraph langchain-openai"
            )

        if self._react_agent is not None:
            return self._react_agent

        import os

        from langchain.agents import create_agent
        from langchain_openai import ChatOpenAI

        if provider is None:
            provider = os.environ.get("F1_LLM_PROVIDER", "lmstudio")

        if provider == "lmstudio":
            llm = ChatOpenAI(
                model=model_name,
                base_url=base_url,
                api_key=api_key,
                temperature=0,
                timeout=120,
                max_retries=LLM_MAX_RETRIES,
            )
        else:
            llm = ChatOpenAI(model=model_name, temperature=0, timeout=120, max_retries=LLM_MAX_RETRIES)

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
        from tool call results in the message history, not from the LLM's
        free-text answer, so the output is deterministic.

        Args:
            stint_state: Dict with keys:
                session     : loaded FastF1 Session (laps + weather already cached).
                driver      : FastF1 driver abbreviation (e.g. 'NOR').
                compound_id : Pirelli compound ID (e.g. 'C2').
                tyre_life   : Current laps on this tyre set.
                gp_name     : GP name matching circuit_cluster_map keys (e.g. 'Sakhir').
                team        : Team name matching team_id_map keys (e.g. 'McLaren').
                year        : Race year (int).

        Returns:
            TireOutput with deg_rate, laps_to_cliff P10/P50/P90, gp_name,
            warning_level, and reasoning.
        """
        session = stint_state["session"]
        driver = stint_state["driver"]
        compound_id = stint_state["compound_id"]
        tyre_life = stint_state["tyre_life"]
        gp_name = stint_state.get("gp_name", "")

        self.laps_df = session.laps.pick_accurate().copy()
        _clean = self.laps_df[self.laps_df["TrackStatus"] == "1"]
        _weather = session.weather_data.mean(numeric_only=True)

        self.session_meta = {
            "fastest_lap_s": _clean["LapTime"].min().total_seconds(),
            # The TRAINED per-cluster constant, not this race's own mean lap time
            # (`_clean["LapTime"].dt.total_seconds().mean()`), which is a
            # different quantity: N04 subtracted one constant per CLUSTER (std 0.0
            # within a cluster), and a race's own mean is a per-race number that
            # happens to have the same units.
            "cluster_mean_lap_s": TireAgentConfig._TRAINED_CLUSTER_MEAN_LAP_S.get(
                self.cfg.cluster_for(gp_name, 0), 0.0
            ),
            "total_laps": int(session.total_laps),
            "cluster_id": self.cfg.cluster_for(gp_name, 0),
            "team_id": _encode_team_id(self.cfg.team_id_map, stint_state.get("team", "Unknown")),
            "year": stint_state.get("year", 2025),
            "AirTemp": float(_weather.get("AirTemp", DEFAULT_AIR_TEMP_C)),
            "TrackTemp": float(_weather.get("TrackTemp", DEFAULT_TRACK_TEMP_C)),
            "Humidity": float(_weather.get("Humidity", 50.0)),
            # Was hardcoded 0.0 (#477) while run_from_state() correctly reads
            # wx.get('rainfall', 0) from the RSM weather dict: mirror that here
            # from the session's own weather data instead of silently telling
            # every dry-model feature the race was rain-free regardless of what
            # actually happened.
            "Rainfall": float(_weather.get("Rainfall", 0.0)),
        }
        # No per-lap rivals list on this path (single-shot FastF1 query, not a live
        # simulation lap): _live_drivers_at_current_lap() falls back to laps_df.
        self._live_drivers = None

        return self._run_core(driver, compound_id, tyre_life, gp_name)

    def run_from_state(self, lap_state: dict, laps_df: pd.DataFrame) -> TireOutput:
        """RSM adapter: run the Tire Agent from a RaceStateManager lap_state dict.

        Translates the nested RSM lap_state into self.laps_df / self.session_meta.
        No FastF1 session is required: all context is derived directly from laps_df
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
        d = lap_state["driver"]
        meta = lap_state["session_meta"]
        wx = lap_state.get("weather", {})

        driver = meta["driver"]
        # This adapter reads the RAW lap_state, not the RaceState, so the canonical
        # builder's normalisation does not reach it: it has to apply the same rules
        # itself or the unification stops at the object boundary (#784). Both of the
        # old two-arg defaults were also dead on the RSM path and wrong when they did
        # fire: the key is always present, so a stored NaN arrives as the STRING "nan"
        # and a None arrives as None, neither of which .get's default catches.
        compound = normalise_compound(d.get("compound"))
        raw_tyre_life = d.get("tyre_life")
        tyre_life = UNKNOWN_TYRE_LIFE if raw_tyre_life is None else raw_tyre_life
        gp_name = meta.get("gp_name", "")
        total_laps = meta.get("total_laps", DEFAULT_TOTAL_LAPS)
        year = meta.get("year", 2025)
        team = meta.get("team", "Unknown")

        compound_id = (
            compound if compound.startswith("C") else _compound_name_to_id(compound, gp_name, year)
        )

        self.laps_df = laps_df.copy()

        # Build session_meta from laps_df (FastF1 Timedelta -> float if needed)
        lt_col = "LapTime_s" if "LapTime_s" in self.laps_df.columns else "LapTime"
        if lt_col == "LapTime" and hasattr(self.laps_df[lt_col].iloc[0], "total_seconds"):
            lap_times = self.laps_df[lt_col].dropna().apply(lambda t: t.total_seconds())
        else:
            lap_times = pd.to_numeric(self.laps_df[lt_col], errors="coerce").dropna()

        if "TrackStatus" in self.laps_df.columns:
            clean_mask = self.laps_df["TrackStatus"].astype(str) == "1"
            clean_times = lap_times[clean_mask] if clean_mask.sum() > 0 else lap_times
        else:
            clean_times = lap_times

        self.session_meta = {
            "fastest_lap_s": float(clean_times.min()) if len(clean_times) > 0 else 90.0,
            # The TRAINED per-cluster constant, for the same reason as the FastF1 path
            # above: this race's own mean lap time is a different quantity wearing the
            # same units.
            "cluster_mean_lap_s": TireAgentConfig._TRAINED_CLUSTER_MEAN_LAP_S.get(
                self.cfg.cluster_for(gp_name, 0), 0.0
            ),
            "total_laps": total_laps,
            "cluster_id": self.cfg.cluster_for(gp_name, 0),
            "team_id": _encode_team_id(self.cfg.team_id_map, team),
            "year": year,
            # reading_or_default, not .get(key, default): the producers report an
            # unmeasured reading as the key PRESENT holding None, which .get's default
            # never catches. These Nones do not crash here: they flow through
            # _add_weather_cols into the TCN's feature frame and moved the cliff estimate
            # 2.3 laps OPTIMISTIC on the 2025 reference lap, silently. Optimistic is the
            # dangerous direction: it delays the pit call. See the helper's docstring.
            "AirTemp": reading_or_default(wx, "air_temp", DEFAULT_AIR_TEMP_C),
            "TrackTemp": reading_or_default(wx, "track_temp", DEFAULT_TRACK_TEMP_C),
            "Humidity": reading_or_default(wx, "humidity", 50.0),
            "Rainfall": float(reading_or_default(wx, "rainfall", 0.0)),
            f"{driver}_compound": compound,
            # The stint actually being run, and the lap actually being raced. N10 trained
            # on windows grouped by ['Year','GP_Name','DriverNumber','Stint'], and the
            # slice below had neither: it matched on Compound alone, so a later stint on
            # the same compound joined the window, and nothing bounded it to the past.
            # At Barcelona 2024 lap 16, HAM's window ran [1..16, 44..59] and the TCN read
            # LAP 59's degradation as current. Both keys travel here rather than through
            # the signature so the FastF1 path, which has neither, degrades to the old
            # behaviour instead of breaking (#449).
            f"{driver}_stint": d.get("stint"),
            "current_lap": lap_state.get("lap_number"),
        }

        # Presence-based on-track set for this lap (#476): the driver's own car plus
        # every rival the RSM actually emitted a row for. A driver who crashed
        # earlier in the race simply stops appearing in rivals from that lap on
        # (the same signal race_state_manager.py itself relies on, #470), so
        # this catches an LLM tool call for a long-retired driver code without
        # reimplementing any retirement/lap-count heuristic.
        self._live_drivers = {driver} | {
            str(r["driver"]) for r in lap_state.get("rivals", []) or [] if r.get("driver")
        }

        return self._run_core(driver, compound_id, tyre_life, gp_name)

    @staticmethod
    def _conservative_stub(
        compound_id: str,
        tyre_life: int,
        gp_name: str,
        reason: str,
    ) -> TireOutput:
        """TireOutput with the fixed conservative defaults used whenever a real TCN
        reading is unavailable (no bundle for this compound, or a tool-output parse
        miss). ``reason`` is folded into ``reasoning`` so the degradation is visible
        instead of masquerading as a genuine reading.
        """
        return TireOutput(
            compound=compound_id,
            current_tyre_life=tyre_life,
            gp_name=gp_name,
            deg_rate=0.03,
            laps_to_cliff_p10=20.0,
            laps_to_cliff_p50=30.0,
            laps_to_cliff_p90=40.0,
            reasoning=reason,
        )

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
        # TCN bundles only exist for dry compounds (C1-C6). For wet/intermediate
        # compounds return a stub with conservative defaults: no TCN inference.
        if compound_id not in self.bundles:
            return self._conservative_stub(
                compound_id,
                tyre_life,
                gp_name,
                reason=(
                    f"[{compound_id} — TCN model not available for wet/intermediate compounds; "
                    f"conservative defaults used]"
                ),
            )

        react_agent = self.get_react_agent()
        msg = (
            f"Analyse the tyre state for driver {driver}, compound {compound_id}, "
            f"tyre life {tyre_life} laps. Use both tools and give your recommendation."
        )
        response = react_agent.invoke({"messages": [{"role": "user", "content": msg}]})
        parsed = _parse_tool_outputs(response["messages"])

        reasoning = ""
        for m in reversed(response["messages"]):
            if hasattr(m, "content") and isinstance(m.content, str) and m.content.strip():
                if not getattr(m, "tool_calls", None):
                    reasoning = m.content.strip()
                    break

        # A parse miss must NOT become 0.0 laps-to-cliff. `_parse_tool_outputs` only
        # writes a key when its regex matched, so an absent 'p10' means "the tool was
        # skipped or its output did not parse", whereas a PRESENT 0.0 legitimately
        # means "the cliff is now". Defaulting the miss to 0.0 collapsed those two into
        # the alarming one: the MC read "cliff NOW" (penalising STAY_OUT by ~4 s) and
        # the warning level flipped to PIT_SOON, so a silent regex miss became a
        # confident call to box. Fall back to the same conservative stub the
        # wet/intermediate branch above already uses, and say so in the reasoning so the
        # degradation is visible instead of masquerading as a reading (#436).
        if "p10" not in parsed:
            logger.warning(
                "Tire tool output did not parse for %s (tyre_life=%s) — using conservative "
                "defaults instead of a 0.0 cliff",
                compound_id,
                tyre_life,
            )
            return self._conservative_stub(
                compound_id,
                tyre_life,
                gp_name,
                reason=(
                    f"[{compound_id} — tire tool output could not be parsed; conservative "
                    f"defaults used] {reasoning}"
                ),
            )

        return TireOutput(
            compound=compound_id,
            current_tyre_life=tyre_life,
            gp_name=gp_name,
            deg_rate=round(parsed.get("deg_rate", 0.0), 4),
            # `.get(...)` then an explicit None, rather than a numeric default:
            # predict_tire_deg_tool can legitimately be skipped while the cliff
            # tool ran, and 0.0 is a real reading for this quantity.
            cumulative_deg_s=(round(parsed["cum_deg"], 4) if "cum_deg" in parsed else None),
            deg_cost_s=_referenced_wear(parsed),
            laps_to_cliff_p10=round(parsed["p10"], 1),
            laps_to_cliff_p50=round(parsed.get("p50", 0.0), 1),
            laps_to_cliff_p90=round(parsed.get("p90", 0.0), 1),
            reasoning=reasoning,
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
# Public entry points: backward-compatible signatures (unchanged)
# ─────────────────────────────────────────────────────────────────────────────


def run_tire_agent(stint_state: dict) -> TireOutput:
    """Run the Tire Agent for a given stint and return a structured TireOutput.

    Delegates to the process-level TireAgent singleton. Populates session state
    from the FastF1 Session object inside stint_state, then invokes the LangGraph
    ReAct agent. Numeric outputs are extracted from tool call results in the
    message history, not from the LLM's free-text answer.

    Args:
        stint_state: Dict with keys: session, driver, compound_id, tyre_life,
            gp_name, team, year. See TireAgent.run for full specification.

    Returns:
        TireOutput with deg_rate, laps_to_cliff P10/P50/P90, warning_level, reasoning.
    """
    return _get_default_tire_agent().run(stint_state)


def run_tire_agent_from_state(lap_state: dict, laps_df: pd.DataFrame) -> TireOutput:
    """RSM adapter: run the Tire Agent from a RaceStateManager lap_state dict.

    Delegates to the process-level TireAgent singleton. No FastF1 session required:
    all context is derived from laps_df and the lap_state produced by RaceStateManager.

    Args:
        lap_state: Dict from RaceStateManager.get_lap_state(). Expected keys:
            lap_number, driver (full telemetry), weather, session_meta.
        laps_df: Full race laps DataFrame with required telemetry columns.

    Returns:
        TireOutput with all fields populated.
    """
    return _get_default_tire_agent().run_from_state(lap_state, laps_df)
