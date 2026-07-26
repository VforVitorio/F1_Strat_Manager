"""Pit-duration holdout regeneration from raw laps (#364).

``pit_labeled/`` is empty on disk, so unlike SC/undercut the pit holdout must be
rebuilt from the raw FastF1 laps (``data/raw/<year>/<circuit>/laps.parquet``).
This ports the N15 pipeline verbatim (pair pit-in lap N with pit-out lap N+1 ->
pit_duration_s; filter drive-throughs + non-normal stops; per-circuit traversal
from train; physical_stop_est; per-team-year median) and returns the deployed
HistGBT P05/P50/P95 quantile models to score it.

The raw circuit directory name is used as the ``circuit`` key: it is bijective
with N15's ``GP_Name`` (one directory = one race = one GP), and the traversal is
recomputed per key from train, so the physical_stop_est is identical to N15's.
The only GP-name dependency (the 3 tight-pit-box circuits) is mapped explicitly.

--- WHERE TO CHANGE IF THE PIT PIPELINE CHANGES ---
notebooks/strategy/pit_prediction/N15_pit_duration.ipynb (cells 4/11/12/17) is
the source of truth; mirror any edit here. Validated: P50 MAE 0.4893 vs the
published 0.487 (delta 0.0023, within tolerance).
"""

from __future__ import annotations

import logging

import json
from typing import Any

from src.f1_strat_manager.data_cache import get_data_root, get_models_root
from src.f1_strat_manager.team_aliases import canonical_team

_PIT_DIR = "pit_prediction"
_COMPOUND_ORDER = {"SOFT": 0, "MEDIUM": 1, "HARD": 2, "INTERMEDIATE": 3, "WET": 4}
_SC_STATUSES = set("4567")  # TrackStatus digits meaning SC / VSC / red
_TIGHT_DIRS = {"Monaco", "Marina_Bay", "Budapest"}  # Monaco / Singapore / Hungarian GP
_STOP_MIN_S, _STOP_MAX_S = 2.0, 4.5  # normal physical stop window (N15)
_RAW_MIN_S, _RAW_MAX_S = 15.0, 60.0  # raw pit_duration_s filter (N15 filter_pit_stops)
_PHYSICAL_STOP_FLOOR = 1.5
_TRAIN_YEARS = (2023, 2024)
_TARGET = "physical_stop_est"


def _collect_pit_stops(pd: Any) -> Any:
    """Pair pit-in lap N with pit-out lap N+1 from raw laps -> one row per stop.

    Reproduces N15 collect_pit_data over the on-disk laps instead of a FastF1
    fetch. Filters drive-throughs (TyreLife_out <= 5) and keeps the raw
    pit_duration_s in [15, 60] s.
    """
    raw_root = get_data_root() / "raw"
    records = []
    for laps_path in sorted(raw_root.glob("*/*/laps.parquet")):
        circuit = laps_path.parent.name
        year = int(laps_path.parent.parent.name)
        laps = pd.read_parquet(laps_path)
        needed = {
            "Driver",
            "Team",
            "LapNumber",
            "Compound",
            "TyreLife",
            "PitInTime",
            "PitOutTime",
            "TrackStatus",
        }
        if not needed <= set(laps.columns):
            continue
        pit_in = laps[laps["PitInTime"].notna()][
            ["Driver", "Team", "LapNumber", "Compound", "TyreLife", "PitInTime", "TrackStatus"]
        ].copy()
        pit_out = laps[laps["PitOutTime"].notna()][
            ["Driver", "LapNumber", "PitOutTime", "TyreLife"]
        ].copy()
        pit_in["LapNumber_out"] = pit_in["LapNumber"] + 1
        merged = pit_in.merge(
            pit_out.rename(columns={"LapNumber": "LapNumber_out", "TyreLife": "TyreLife_out"}),
            on=["Driver", "LapNumber_out"],
            how="inner",
        )
        merged["pit_duration_s"] = (merged["PitOutTime"] - merged["PitInTime"]).dt.total_seconds()
        merged = merged[merged["TyreLife_out"] <= 5].copy()
        merged["year"] = year
        merged["circuit"] = circuit
        records.append(merged)
    if not records:
        return pd.DataFrame()
    stops = pd.concat(records, ignore_index=True)
    return stops[(stops["pit_duration_s"] >= _RAW_MIN_S) & (stops["pit_duration_s"] <= _RAW_MAX_S)]


def _engineer(stops: Any) -> Any:
    """Add the N15 model features (compound_id, tyre_life_in, under_sc, compound_change, ...)."""
    out = stops.copy()
    out["Compound"] = out["Compound"].str.upper().str.strip()
    out["compound_id"] = out["Compound"].map(_COMPOUND_ORDER).fillna(-1).astype(int)
    out["tyre_life_in"] = out["TyreLife"].clip(upper=50).fillna(0).astype(float)
    out["team"] = out["Team"].astype(str).str.strip()
    out["lap_number"] = out["LapNumber"].fillna(0).astype(int)
    out["under_sc"] = (
        out["TrackStatus"].astype(str).apply(lambda s: int(any(c in _SC_STATUSES for c in s)))
    ).astype(int)
    out["tight_pit_box"] = out["circuit"].isin(_TIGHT_DIRS).astype(int)
    out = out.sort_values(["year", "circuit", "Driver", "LapNumber"]).copy()
    prev_compound = out.groupby(["year", "circuit", "Driver"])["compound_id"].shift(1)
    out["compound_change"] = (
        (prev_compound.notna()) & (out["compound_id"] != prev_compound)
    ).astype(int)
    return out


def _add_team_year_median(train: Any, target_slice: Any) -> Any:
    """Per-team prior: median physical_stop_est from the most recent training year (N15)."""
    lookup = train.groupby(["team", "year"])[_TARGET].median()
    global_median = train[_TARGET].median()
    teams_seen = set(lookup.index.get_level_values("team"))

    def most_recent(team: str, year: int) -> float:
        if (team, year) in lookup.index:
            return float(lookup[(team, year)])
        if team in teams_seen:
            return float(lookup.xs(team, level="team").iloc[-1])
        return float(global_median)

    target_slice = target_slice.copy()
    target_slice["team_year_median"] = [
        most_recent(t, y) for t, y in zip(target_slice["team"], target_slice["year"])
    ]
    return target_slice



logger = logging.getLogger(__name__)


def _team_class_index(raw_team: str, team_classes: list[str]) -> int:
    """Encoder index for a team name, resolving a rebrand first and warning if it fails.

    The fallback is index 0, and index 0 is NOT "unknown": it is a real class the
    frozen model trained on, so an unrecognised name is silently scored as another
    team. That is what happened to the 2025 holdout, where FastF1 says `Racing Bulls`
    and the 2024-fitted encoder knows `RB`: 20 of 252 rows were evaluated as
    `Alfa Romeo` (#629). Measured impact on the published P50 MAE was -0.0045 s, so
    the number stands, but a reproduction harness that quietly reinterprets its own
    input is not reproducing anything, which is why the fallback now shouts.
    """
    resolved = canonical_team(raw_team)
    if resolved in team_classes:
        return team_classes.index(resolved)

    logger.warning(
        "Team %r is not in this artefact's label-encoder classes (%s) and has no alias; "
        "scoring it as %r, which is a DIFFERENT team. Add it to TEAM_ALIASES in "
        "src/f1_strat_manager/team_aliases.py or the metric is measuring the wrong car.",
        raw_team,
        ", ".join(team_classes),
        team_classes[0],
    )
    return 0

def load_pit_holdout(year: int = 2025) -> tuple[Any, dict[str, Any], list[str]] | None:
    """Rebuild the pit holdout and return ``(test_slice, quantile_models, features)``.

    ``test_slice`` carries the engineered features + ``physical_stop_est`` target
    for the requested year; ``quantile_models`` maps ``p05/p50/p95`` to the loaded
    HistGBT models. Returns ``None`` when the raw laps or models are absent.
    """
    import joblib
    import pandas as pd

    model_dir = get_models_root() / _PIT_DIR
    cfg_path = model_dir / "model_config.json"
    if not cfg_path.exists():
        return None

    stops = _collect_pit_stops(pd)
    if stops.empty:
        return None
    df = _engineer(stops)
    train = df[df["year"].isin(_TRAIN_YEARS)].copy()
    target_slice = df[df["year"] == year].copy()

    traversal = train.groupby("circuit")["pit_duration_s"].quantile(0.05) - _PHYSICAL_STOP_FLOOR
    for part in (train, target_slice):
        part["circuit_traversal"] = part["circuit"].map(traversal).fillna(traversal.mean())
        part[_TARGET] = (part["pit_duration_s"] - part["circuit_traversal"]).clip(lower=0.5)
    train = train[(train[_TARGET] >= _STOP_MIN_S) & (train[_TARGET] <= _STOP_MAX_S)]
    target_slice = target_slice[
        (target_slice[_TARGET] >= _STOP_MIN_S) & (target_slice[_TARGET] <= _STOP_MAX_S)
    ].copy()
    if target_slice.empty:
        return None
    target_slice = _add_team_year_median(train, target_slice)

    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    features = cfg["features"]
    team_classes = list(cfg["label_encoder_classes"]["team"])
    target_slice["team"] = target_slice["team"].map(
        lambda raw: _team_class_index(raw, team_classes)
    )

    models = {}
    for quantile in ("p05", "p50", "p95"):
        path = model_dir / f"hist_pit_{quantile}_v1.pkl"
        if not path.exists():
            return None
        models[quantile] = joblib.load(path)
    return target_slice, models, features
