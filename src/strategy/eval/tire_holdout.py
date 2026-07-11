"""Tire-degradation (TireDegTCN) holdout reconstruction from the featured laps (#372).

The headline tire MAE 0.7078 s is the N09 global Model A evaluated on the 2025
test set (per-compound best is C2 fine-tuned at 0.5501). This module rebuilds
that exact test set in-memory from ``laps_tiredeg.parquet`` and the deployed
``tiredeg_modelA_v4.pt`` bundle, then scores it.

The bundle is self-describing: it carries the fitted StandardScaler, the 42
feature names, the window (28), the target (FuelAdjustedDegAbsolute), the model
hparams and the state dict, so no training-time state has to be reconstructed -
only the N10 sequence-building (per-stint left-pad/truncate to the window,
cumulative target) is ported.

The ``TireDegTCN`` architecture is redefined here (not imported from
``src/agents/tire_agent.py``, which is untouchable and runs heavy config I/O at
import): the N10 export is a plain state dict for this module, and redefining it
is the same convention the agent and the notebook already follow.

Two quantities are exposed:
- ``load_tire_predictions`` -> the deterministic ``model.eval()`` forward pass
  MAE (the 0.7078 headline).
- ``mc_dropout_global_sigma`` -> a seeded MC-Dropout epistemic sigma over the
  whole 2025 test set, validating the order of magnitude of the stored
  per-compound (single-stint, stochastic) calibration.

--- WHERE TO CHANGE IF THE TIRE PIPELINE CHANGES ---
notebooks/strategy/tire_degradation/N10_tiredeg_compound_finetuning.ipynb
(cells 5 sequence build, 7/8 dataset + scaler, 30 global baseline, 51 MC dropout)
is the source of truth; mirror any edit here. Validated: MAE 0.7078 on 2025,
exact to the thesis-final global headline.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.f1_strat_manager.data_cache import get_data_root, get_models_root

_TIRE_DIR = "tire_degradation"
_GLOBAL_BUNDLE = "tiredeg_modelA_v4.pt"
_PARQUET = "laps_tiredeg.parquet"
# one stint of one driver in one race (N10 STINT_KEYS)
_STINT_KEYS = ["Year", "GP_Name", "DriverNumber", "Stint"]
_DRY_COMPOUNDS = ["SOFT", "MEDIUM", "HARD"]
_TEST_YEAR = 2025
_MC_BATCH = 256


# ── TireDegTCN (redefined from N10; same as src/agents/tire_agent.py) ──────────
class _CausalConv1dBlock(nn.Module):
    """Causal dilated conv with left-side padding, LayerNorm, GELU, dropout."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, dilation: int, dropout: float):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, dilation=dilation, padding=0)
        self.norm = nn.LayerNorm(out_ch)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.pad(x, (self.pad, 0))
        x = self.conv(x)
        return self.drop(F.gelu(self.norm(x.transpose(1, 2)).transpose(1, 2)))


class _TCNResidualBlock(nn.Module):
    """Two stacked causal conv blocks with an additive residual connection."""

    def __init__(self, ch: int, kernel_size: int, dilation: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            _CausalConv1dBlock(ch, ch, kernel_size, dilation, dropout),
            _CausalConv1dBlock(ch, ch, kernel_size, dilation, dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.net(x) + x)


class TireDegTCN(nn.Module):
    """Temporal Convolutional Network for tire degradation (N10 export).

    Input projection -> exponentially dilated residual blocks -> linear head
    predicting a single scalar (cumulative FuelAdjustedDegAbsolute at the last
    timestep). ``mask`` is accepted for signature parity with the notebook and
    ignored (the head reads only the last position).
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
            [_TCNResidualBlock(d_model, kernel_size, 2**i, dropout) for i in range(n_layers)]
        )
        self.output_head = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.input_proj(x).transpose(1, 2)
        for block in self.blocks:
            x = block(x)
        return self.output_head(x.transpose(1, 2)[:, -1, :]).squeeze(-1)


def _pad_or_truncate(arr: np.ndarray, window: int) -> np.ndarray:
    """Left-zero-pad or truncate-from-start a (laps, features) array to ``window`` rows (N10)."""
    length = arr.shape[0]
    if length >= window:
        return arr[-window:].astype(np.float32)
    pad = np.zeros((window - length, arr.shape[1]), dtype=np.float32)
    return np.concatenate([pad, arr], axis=0).astype(np.float32)


def _build_sequences(
    df: Any, features: list, window: int, target: str
) -> tuple[np.ndarray, np.ndarray]:
    """Port N10 _build_sequences (cumulative target): one sample per stint lap t>=1.

    For each stint (sorted by TyreLife), sample t predicts the cumulative
    degradation at lap t from the padded/truncated window of laps up to t.
    """
    seqs, tgts = [], []
    for _, grp in df.groupby(_STINT_KEYS, sort=False):
        grp = grp.sort_values("TyreLife").reset_index(drop=True)
        if len(grp) < 2:
            continue
        cum = grp[target].to_numpy()
        feat = grp[features].to_numpy()
        for t in range(1, len(grp)):
            if np.isnan(cum[t]):
                continue
            seqs.append(_pad_or_truncate(feat[:t], window))
            tgts.append(float(cum[t]))
    return np.stack(seqs), np.array(tgts, dtype=np.float32)


def _load_model_and_test(
    year: int = _TEST_YEAR,
) -> tuple[TireDegTCN, np.ndarray, np.ndarray] | None:
    """Rebuild the tire test set and return ``(model, X, y_true)`` or ``None`` if absent.

    Shared seam: both the MAE reproduction and the MC-Dropout sigma need the same
    global-model bundle + the same 2025 test sequences, so the load lives once
    here. Scales the features with the bundle's fitted scaler (N10 apply_scaler:
    fillna(0) then transform), leaving the target unscaled.
    """
    import pandas as pd

    bundle_path = get_models_root() / _TIRE_DIR / _GLOBAL_BUNDLE
    parquet = get_data_root() / "processed" / _PARQUET
    if not (bundle_path.exists() and parquet.exists()):
        return None

    bundle = torch.load(bundle_path, map_location="cpu", weights_only=False)
    features = bundle["feature_names"]
    window = bundle["window"]
    target = bundle["target"]
    scaler = bundle["scaler"]

    df = pd.read_parquet(parquet)
    test = df[
        (df["Year"] == year)
        & df["Compound"].isin(_DRY_COMPOUNDS)
        & df["AbsoluteCompound"].notna()
        & df[target].notna()
    ].copy()
    if test.empty:
        return None
    test[features] = scaler.transform(test[features].fillna(0))

    x, y_true = _build_sequences(test, features, window, target)

    model = TireDegTCN(bundle["n_features"], **bundle["model_hparams"])
    model.load_state_dict(bundle["state_dict"])
    model.eval()
    return model, x, y_true


def load_tire_predictions(year: int = _TEST_YEAR) -> tuple[np.ndarray, np.ndarray] | None:
    """Rebuild the tire holdout and return ``(y_true, y_pred)`` cumulative degradation (s).

    Deterministic ``model.eval()`` forward pass over the 2025 test sequences - the
    quantity the 0.7078 global headline is on. Returns ``None`` when the bundle or
    holdout parquet is absent, so the caller degrades to a ``pending`` result.
    """
    loaded = _load_model_and_test(year)
    if loaded is None:
        return None
    model, x, y_true = loaded

    preds = []
    with torch.no_grad():
        for i in range(0, len(x), _MC_BATCH):
            xb = torch.tensor(x[i : i + _MC_BATCH], dtype=torch.float32)
            preds.append(model(xb).numpy())
    y_pred = np.concatenate(preds)
    return y_true, y_pred


def mc_dropout_global_sigma(n_mc: int = 50, seed: int = 42, year: int = _TEST_YEAR) -> float | None:
    """Seeded MC-Dropout epistemic sigma averaged over the whole 2025 test set (s).

    Keeps dropout active (``model.train()``) and runs ``n_mc`` forward passes per
    sequence, then averages the per-sequence prediction std. Unlike the stored
    per-compound ``mc_dropout_calibration.json`` (each fitted on a single
    hand-picked 2025 stint, unseeded), this is a single reproducible global
    figure that validates the order of magnitude of that calibration. Returns
    ``None`` when the bundle or holdout is absent.
    """
    loaded = _load_model_and_test(year)
    if loaded is None:
        return None
    model, x, _ = loaded
    model.train()  # keep dropout active for MC sampling
    torch.manual_seed(seed)

    samples = np.empty((n_mc, len(x)), dtype=np.float64)
    with torch.no_grad():
        for s in range(n_mc):
            preds = []
            for i in range(0, len(x), _MC_BATCH):
                xb = torch.tensor(x[i : i + _MC_BATCH], dtype=torch.float32)
                preds.append(model(xb).numpy())
            samples[s] = np.concatenate(preds)
    per_sequence_sigma = samples.std(axis=0)
    return float(per_sequence_sigma.mean())
