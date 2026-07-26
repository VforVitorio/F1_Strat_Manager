"""Pace baselines benchmark — persistence vs team x circuit median vs XGBoost delta.

Runs three pure-evaluation models against the 2025 holdout laps parquet
and writes the resulting MAE / RMSE / R^2 metrics to ``data/eval/``.

The XGBoost row reuses the production model and feature pipeline from
:mod:`src.agents.pace_agent` so the reported MAE matches what the
multi-agent system actually sees at inference time. The other two rows
are deliberately trivial baselines whose only purpose is to give the
thesis a numeric reference point against which the XGBoost gain can be
quantified.

Usage::

    uv run scripts/bench_pace_baselines.py [--year 2025] [--device cpu|cuda|auto]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Repo-root path injection: must happen before any src.* import
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = next(
    (p for p in [_SCRIPT_DIR, *_SCRIPT_DIR.parents] if (p / ".git").exists()),
    _SCRIPT_DIR.parent,
)
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Silence transformer / xgboost log noise so the Rich console stays clean.
logging.getLogger("xgboost").setLevel(logging.ERROR)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import xgboost as xgb  # noqa: E402
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score  # noqa: E402

from scripts.bench._common import (  # noqa: E402
    BenchResult,
    export_csv,
    export_markdown,
    make_start_panel,
    render_results_table,
)
from scripts.cli.theme import console  # noqa: E402

_DATA_ROOT = _REPO_ROOT / "data"
_MODELS_DIR = _DATA_ROOT / "models" / "lap_time"
_PROCESSED_DIR = _DATA_ROOT / "processed"
_EVAL_DIR = _DATA_ROOT / "eval"


_PACE_FEATURES_PATH = _MODELS_DIR / "xgb_laptime_delta_feature_names.json"
_PACE_MODEL_PATH = _MODELS_DIR / "xgb_laptime_delta_final.json"


class PaceBaselineRunner:
    """Encapsulate the pace-baseline evaluation against the 2025 holdout.

    Loads the holdout parquet once, then computes three independent
    rows: a persistence baseline (predict ``Prev_LapTime``), a team x
    circuit median pulled from the 2023+2024 training parquets, and the
    production XGBoost delta model loaded straight from the JSON
    artefact under ``data/models/lap_time/``. Each row is wrapped in a
    :class:`BenchResult` ready for export.
    """

    REQUIRED_COLUMNS = (
        "Year",
        "GP_Name",
        "Team",
        "LapTime_s",
        "Prev_LapTime",
        "DriverNumber",
        "LapNumber",
        "Stint",
        "TyreLife",
        "FreshTyre",
        "Position",
        "CompoundID",
        "TeamID",
        "LapsSincePitStop",
        "FuelLoad",
        "FuelEffect",
        "Prev_TyreLife",
        "Prev_SpeedST",
        "AirTemp",
        "TrackTemp",
        "Humidity",
        "Rainfall",
        "laps_remaining",
        "Cluster",
        "mean_sector_speed",
    )

    def __init__(self, year: int = 2025, device: str = "cpu") -> None:
        """Load model + parquet artefacts; do not run any prediction yet.

        Args:
            year: Holdout season — defaults to 2025, which matches the
                value declared in MEMORY.md as the canonical evaluation
                year for the N06 production model.
            device: Accepted for forward-compat with future GPU XGBoost
                builds but currently ignored — the JSON artefact is
                loaded with the default CPU predictor.
        """
        self.year = int(year)
        self.device = device
        self.holdout_df = self._load_holdout(self.year)
        self.train_df = self._load_train(self.year)
        self.model, self.features = self._load_xgb_model()

    # ── Loaders ──────────────────────────────────────────────────────────────

    def _load_holdout(self, year: int) -> pd.DataFrame:
        """Read ``laps_featured_<year>.parquet`` and drop unusable rows.

        Filters out rows whose target ``LapTime_s`` or pace baseline
        ``Prev_LapTime`` is missing — those rows cannot be scored by
        any of the three models. The returned frame still carries
        every feature column needed by the XGB model so callers can
        slice it without re-reading the parquet, plus the three
        ``Prev_*`` degradation columns reconstructed via the same
        per ``(Driver, Stint)`` lag-1 shift used in N06.
        """
        path = _PROCESSED_DIR / f"laps_featured_{year}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"Holdout parquet missing for year {year}: {path}")
        df = pd.read_parquet(path)
        df = self._add_prev_deg_features(df)
        # Match N06's df25_d filter exactly: drop rows missing the delta target
        # or the Prev_LapTime baseline so the holdout used at bench time is
        # bit-identical to the one quoted in MEMORY.md.
        drop_cols = [c for c in ("LapTime_Delta", "Prev_LapTime", "LapTime_s") if c in df.columns]
        df = df.dropna(subset=drop_cols).copy()
        return df

    @staticmethod
    def _add_prev_deg_features(df: pd.DataFrame) -> pd.DataFrame:
        """Recover ``Prev_DegradationRate`` / ``Prev_CumulativeDeg`` / ``Prev_DegAcceleration``.

        The N06 notebook engineers these three features as a lag-1
        shift of the live ``DegradationRate`` / ``CumulativeDeg`` /
        ``DegAcceleration`` columns within each ``(Driver, Stint)``
        group (the model is trained on data with the shift applied,
        not on the raw current-lap values). Replicating the shift here
        means the bench feeds the XGBoost model exactly the feature
        distribution it was trained on, which is the only way to
        reproduce the headline MAE quoted in MEMORY.md.
        """
        df = df.sort_values(["GP_Name", "Year", "DriverNumber", "Stint", "LapNumber"]).reset_index(
            drop=True
        )
        grp = df.groupby(["GP_Name", "Year", "DriverNumber", "Stint"], sort=False, group_keys=False)
        for src, dst in (
            ("DegradationRate", "Prev_DegradationRate"),
            ("CumulativeDeg", "Prev_CumulativeDeg"),
            ("DegAcceleration", "Prev_DegAcceleration"),
        ):
            if src in df.columns:
                df[dst] = grp[src].shift(1)
            else:
                df[dst] = float("nan")
        return df

    def _load_train(self, year: int) -> pd.DataFrame:
        """Concatenate the parquets for every season strictly before ``year``.

        Used only by the team x circuit median baseline, so only three
        columns are loaded to keep the memory footprint tiny. Missing
        season files are skipped with a warning printed via the
        shared Rich console.
        """
        train_frames: list[pd.DataFrame] = []
        for season in (2023, 2024):
            if season >= year:
                continue
            path = _PROCESSED_DIR / f"laps_featured_{season}.parquet"
            if not path.exists():
                console.print(
                    f"[yellow]Warning:[/yellow] training parquet missing for {season} ({path}) — skipping"
                )
                continue
            train_frames.append(pd.read_parquet(path, columns=["GP_Name", "Team", "LapTime_s"]))
        if not train_frames:
            return pd.DataFrame(columns=["GP_Name", "Team", "LapTime_s"])
        return pd.concat(train_frames, ignore_index=True)

    def _load_xgb_model(self) -> tuple[xgb.XGBRegressor, list[str]]:
        """Load the production XGBoost regressor and its feature-name list.

        Mirrors :class:`src.agents.pace_agent.PaceAgent._load_model` so
        the benchmark is guaranteed to use the same artefacts (and in
        the same order) that the inference pipeline does at runtime.
        """
        if not _PACE_MODEL_PATH.exists():
            raise FileNotFoundError(f"XGB model artefact missing: {_PACE_MODEL_PATH}")
        if not _PACE_FEATURES_PATH.exists():
            raise FileNotFoundError(f"XGB feature list missing: {_PACE_FEATURES_PATH}")
        import json

        features = json.loads(_PACE_FEATURES_PATH.read_text())
        model = xgb.XGBRegressor()
        model.load_model(_PACE_MODEL_PATH)
        return model, features

    # ── Baselines ────────────────────────────────────────────────────────────

    def run_persistence(self) -> BenchResult:
        """Score ``y_pred = Prev_LapTime`` against the actual lap time.

        Captures the trivial "next lap = previous lap" baseline so the
        thesis can quote the absolute error gap closed by both the
        team x circuit median and the XGBoost model.
        """
        y_true = self.holdout_df["LapTime_s"].to_numpy(dtype=float)
        y_pred = self.holdout_df["Prev_LapTime"].to_numpy(dtype=float)
        return self._score(
            "persistence",
            y_true,
            y_pred,
            notes="y_pred = Prev_LapTime",
        )

    def run_team_circuit_median(self) -> BenchResult:
        """Score the per ``(Team, GP_Name)`` median lap time from 2023+2024.

        The median is computed once on the training parquets and looked
        up by ``(Team, GP_Name)`` for every 2025 holdout row. Pairs not
        seen in training fall back to the global training median so
        every row receives a prediction.
        """
        if self.train_df.empty:
            return self._score(
                "team_circuit_median",
                np.array([]),
                np.array([]),
                notes="no training parquets available",
            )

        median_lookup = (
            self.train_df.dropna(subset=["LapTime_s"])
            .groupby(["Team", "GP_Name"], dropna=True)["LapTime_s"]
            .median()
        )
        global_median = float(self.train_df["LapTime_s"].median())

        keys = list(zip(self.holdout_df["Team"], self.holdout_df["GP_Name"]))
        y_pred = np.array(
            [float(median_lookup.get(k, global_median)) for k in keys],
            dtype=float,
        )
        y_true = self.holdout_df["LapTime_s"].to_numpy(dtype=float)
        return self._score(
            "team_circuit_median",
            y_true,
            y_pred,
            notes="median LapTime_s from 2023+2024 by (Team, GP_Name)",
        )

    def run_xgb_delta(self) -> BenchResult:
        """Score the production XGBoost delta + Prev_LapTime reconstruction.

        Slices the holdout frame by ``self.features`` (the same 25
        columns the live agent uses), fills the three optional
        degradation columns with 0.0 when they are missing, and adds
        the predicted delta back onto ``Prev_LapTime`` to recover an
        absolute lap time. The resulting MAE must match the value
        recorded in MEMORY.md within +/- 0.001 s; the caller asserts
        this in :meth:`run_all`.
        """
        feature_df = self._build_feature_frame(self.holdout_df, self.features)
        delta = self.model.predict(feature_df)
        y_pred = self.holdout_df["Prev_LapTime"].to_numpy(dtype=float) + np.asarray(
            delta, dtype=float
        )
        y_true = self.holdout_df["LapTime_s"].to_numpy(dtype=float)
        return self._score(
            "xgb_delta_prod",
            y_true,
            y_pred,
            notes="XGBoost delta + Prev_LapTime, 25 production features",
        )

    @staticmethod
    def _build_feature_frame(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
        """Return a numeric DataFrame with exactly ``features`` as columns.

        Missing columns are added with NaN so XGBoost can route them
        through its sparse-aware split logic; ``Prev_DegradationRate``,
        ``Prev_CumulativeDeg`` and ``Prev_DegAcceleration`` are not
        materialised in the featured-laps parquet (they are computed
        live by the agent from a degradation history we do not have at
        bench time) so they default to 0.0 to match the agent's own
        signature defaults.
        """
        feature_df = pd.DataFrame(index=df.index)
        for col in features:
            if col in df.columns:
                feature_df[col] = df[col]
            else:
                feature_df[col] = np.nan
        return feature_df.apply(pd.to_numeric, errors="coerce")

    @staticmethod
    def _score(
        model_name: str,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        notes: str,
    ) -> BenchResult:
        """Compute MAE / RMSE / R^2 / row count and wrap them in a BenchResult.

        Drops any rows where ``y_true`` or ``y_pred`` are non-finite so
        the metrics reflect rows that every baseline can score; this
        keeps the three rows directly comparable. Returns NaN metrics
        when the input is empty (no training data, etc.) so the
        artefact still emits a row instead of crashing.
        """
        if len(y_true) == 0:
            return BenchResult(
                name=model_name,
                metrics={
                    "mae_s": float("nan"),
                    "rmse_s": float("nan"),
                    "r2": float("nan"),
                    "n_laps_evaluated": 0,
                    "notes": notes,
                },
            )
        mask = np.isfinite(y_true) & np.isfinite(y_pred)
        y_true_clean = y_true[mask]
        y_pred_clean = y_pred[mask]
        mae = float(mean_absolute_error(y_true_clean, y_pred_clean))
        rmse = float(np.sqrt(mean_squared_error(y_true_clean, y_pred_clean)))
        r2 = float(r2_score(y_true_clean, y_pred_clean))
        return BenchResult(
            name=model_name,
            metrics={
                "mae_s": mae,
                "rmse_s": rmse,
                "r2": r2,
                "n_laps_evaluated": int(mask.sum()),
                "notes": notes,
            },
        )

    # ── Orchestration ────────────────────────────────────────────────────────

    def run_all(self) -> list[BenchResult]:
        """Execute the three baselines in declaration order.

        Order is fixed so the markdown / CSV output reads
        persistence -> team x circuit median -> XGBoost; the thesis
        chapter renders the table top-to-bottom in that progression.
        """
        return [
            self.run_persistence(),
            self.run_team_circuit_median(),
            self.run_xgb_delta(),
        ]


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

_COLUMNS = ["model", "mae_s", "rmse_s", "r2", "n_laps_evaluated", "notes"]
_TITLE = "Pace baselines (2025 holdout)"


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the pace baselines benchmark.")
    parser.add_argument("--year", type=int, default=2025, help="Holdout season (default 2025).")
    parser.add_argument(
        "--device", type=str, default="cpu", help="Reserved for forward-compat (CPU XGBoost only)."
    )
    return parser.parse_args(argv)


def _check_xgb_anchor(results: list[BenchResult]) -> None:
    """Abort when the XGB MAE drifts more than 0.001 s from the memory anchor.

    The anchor (MAE = 0.4104 s) is recorded in MEMORY.md as the value
    reported by the production N06 notebook; if the bench drifts past
    the +/- 0.001 s tolerance the feature pipeline almost certainly
    diverged from the agent loader, so the script aborts loudly rather
    than silently shipping a wrong number into the thesis.
    """
    expected_mae = 0.4104
    tolerance = 0.001
    for row in results:
        if row.name != "xgb_delta_prod":
            continue
        actual_mae = row.metrics.get("mae_s")
        if actual_mae is None or not np.isfinite(actual_mae):
            raise RuntimeError("xgb_delta_prod produced a non-finite MAE")
        if abs(actual_mae - expected_mae) > tolerance:
            raise RuntimeError(
                f"xgb_delta_prod MAE drift: got {actual_mae:.4f}s, "
                f"expected {expected_mae:.4f} +/- {tolerance:.3f} s"
            )


def main(argv: Optional[list[str]] = None) -> int:
    args = _parse_args(argv)
    console.print(
        make_start_panel(
            "bench_pace_baselines.py",
            f"Pace baselines vs 2025 holdout (year={args.year}).",
        )
    )

    runner = PaceBaselineRunner(year=args.year, device=args.device)
    results = runner.run_all()

    _check_xgb_anchor(results)

    md_path = _EVAL_DIR / "pace_baselines.md"
    csv_path = _EVAL_DIR / "pace_baselines.csv"
    export_markdown(results, md_path, _TITLE, _COLUMNS)
    export_csv(results, csv_path, _COLUMNS)

    console.print(render_results_table(results, _TITLE, _COLUMNS))
    console.print(f"[green]Markdown:[/green] {md_path.resolve()}")
    console.print(f"[green]CSV:     [/green] {csv_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
