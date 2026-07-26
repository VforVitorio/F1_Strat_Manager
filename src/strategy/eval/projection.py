"""Eval report for the Monte Carlo projection layer: its tables and its accuracy.

Two things live here, and they answer two different questions.

**Is the layer measured?** ``data/mc_measured_v1.json`` carries six tables the
Monte Carlo scorer reads at runtime, each one counted off real laps rather than
assumed. This report lists them with their sample sizes, so "measured" is a
number a reader can check and not a word in a commit message.

**Is the layer right?** The projection claims that a stop lands a car in a given
place. Every real pit stop in the dataset is a labelled example of exactly that
claim, which makes the accuracy checkable without anyone hand-labelling
anything: project the stop from the lap before, compare with where the car
actually came out.

The measurement itself is here rather than in the test that gates it. Reversing
that would mean the harness reimplements the test, and this repo has already
paid for one parallel implementation drifting away from the reference it copied.
The test imports ``measure_projection_ground_truth`` and asserts a floor on it.

WHERE TO CHANGE IF THE PROJECTION CHANGES:
- ``src/agents/position_projection.py`` holds the geometry under test. A change
  there moves the number here, and ``MIN_WITHIN_ONE`` in
  ``tests/mc/test_position_projection.py`` is the floor that notices.
- ``scripts/measure_mc_tables.py`` regenerates ``data/mc_measured_v1.json``;
  this report reads that file and never recomputes the tables.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.agents.position_projection import (
    DriverPlan,
    ProjectionConfig,
    RivalState,
    project_positions,
)
from src.f1_strat_manager.data_cache import _find_repo_root
from src.strategy.eval.report import build_header, write_report

MEASURED_TABLES_PATH = "data/mc_measured_v1.json"

# A stop is a stop whether or not the tyre falls off a cliff two laps later, so
# the ground truth uses a flat degradation profile and a cliff far outside the
# window. What is under test is the geometry of the rejoin, not the tyre model.
_GROUND_TRUTH_CLIFF_LAPS = 99.0

# The horizon has to be the one the comparison uses. Each stop is projected from
# the lap before and checked against the lap after, which is two laps, so the
# config projects two laps. Leaving the runtime default of five measures a
# five-lap outcome against a two-lap observation and scores several points too
# high for the wrong reason: the extra laps let rivals separate, and a projection
# graded on a horizon it was not asked about is not validated at all. The tyre
# terms are off for the same reason the cliff is far away.
_GROUND_TRUTH_CONFIG = ProjectionConfig(
    window_laps=2,
    racing_laps=2.0,
    fresh_gain_s=0.0,
    cliff_loss_s=0.0,
    neutralisation_saving_s=0.0,
)

_STOP_NOW = DriverPlan("PIT_NOW", stops_in_window=True, stop_offset_laps=0)

# What each measured table exists to answer, in one line. Kept beside the report
# rather than in the JSON because it is editorial, and the JSON is machine-read.
_TABLE_PURPOSE: dict[str, str] = {
    "clean_air": "seconds a lap a follower gains once the car directly ahead pits, per circuit",
    "gap_density": "seconds between consecutive cars, so a projected gap maps to a place",
    "neutralisation_rate": "chance a Safety Car arrives while a stop is being deferred",
    "sc_window": "green laps left inside the 5-lap decision window once neutralised",
    "status_mix": "share of laps that are actually racing, the denominator for the rest",
    "stop_hazard": "chance a rival stops in the window, by tyre life",
    "undercut_band": "undercut success against the gap to the target",
}


@dataclass(frozen=True)
class GroundTruth:
    """Accuracy of the position projection against real pit stops.

    ``errors`` is signed (projected minus actual) so a systematic bias is
    visible rather than absorbed into a magnitude. ``within_one`` is the headline
    because a strategy call that lands one place either side of the projection is
    still the same call; being three places out is a different race.
    """

    errors: np.ndarray
    races: int

    @property
    def sample_size(self) -> int:
        return int(self.errors.size)

    @property
    def within_one(self) -> float:
        return float((np.abs(self.errors) <= 1).mean())

    @property
    def exact(self) -> float:
        return float((self.errors == 0).mean())

    @property
    def mean_signed_error(self) -> float:
        return float(self.errors.mean())

    @property
    def mean_absolute_error(self) -> float:
        return float(np.abs(self.errors).mean())


def _raw_data_root() -> Path | None:
    """``data/raw/`` if this is a checkout with the dataset pulled, else None."""
    repo = _find_repo_root()
    if repo is None:
        return None
    raw = repo / "data" / "raw"
    return raw if raw.is_dir() else None


def _elapsed_pivot(laps):
    """LapNumber x Driver table of elapsed session seconds."""
    frame = laps[["LapNumber", "Driver", "Time"]].dropna()
    frame = frame.assign(elapsed=frame["Time"].dt.total_seconds())
    return frame.pivot_table(index="LapNumber", columns="Driver", values="elapsed", aggfunc="first")


def _neutralised_laps(laps) -> set[int]:
    """Lap numbers showing a Safety Car, VSC or red flag on any car's status."""
    neutralised = set()
    for lap, group in laps.groupby("LapNumber"):
        joined = "".join(group["TrackStatus"].dropna().astype(str))
        if any(flag in joined for flag in ("4", "5", "6")):
            neutralised.add(int(lap))
    return neutralised


def _rivals_around(pivot, medians, pitters, driver, lap, row_before, row_after, ours_before):
    """Every car with a lap either side of the stop, with its own stop loss if it pitted."""
    rivals: list[RivalState] = []
    for other in pivot.columns:
        if other == driver:
            continue
        their_before, their_after = row_before.get(other), row_after.get(other)
        if their_before is None or np.isnan(their_before):
            continue
        if their_after is None or np.isnan(their_after):
            continue

        also_pits = other in pitters.get(lap, ()) or other in pitters.get(lap + 1, ())
        their_normal = medians.get(other)
        their_loss = 0.0
        if also_pits and their_normal and not np.isnan(their_normal):
            their_loss = max(0.0, (their_after - their_before) - 2 * their_normal)

        rivals.append(
            RivalState(
                driver=str(other),
                gap_s=float(their_before - ours_before),
                is_pitting=also_pits,
                stop_loss_s=their_loss,
            )
        )
    return rivals


def _actual_position(pivot, row_after, driver, ours_after) -> int:
    """Where the car actually came out: one plus the cars ahead of it on the lap after."""
    ahead = sum(
        1
        for other in pivot.columns
        if other != driver
        and not np.isnan(row_after.get(other, np.nan))
        and row_after[other] < ours_after
    )
    return 1 + int(ahead)


def project_one_stop(pivot, medians, pitters, driver, lap) -> int | None:
    """Projected minus actual rejoin position for one real stop, or None to skip.

    Returns None rather than a sentinel when the stop cannot be reconstructed:
    a missing lap either side, or a realised pit loss that comes out negative,
    which means the "two normal laps" baseline does not describe what happened.
    Feeding those through would measure the reconstruction, not the projection.
    """
    before, after = lap - 1, lap + 1
    if before < 1 or before not in pivot.index or after not in pivot.index:
        return None

    row_before, row_after = pivot.loc[before], pivot.loc[after]
    ours_before, ours_after = row_before.get(driver), row_after.get(driver)
    normal = medians.get(driver)
    if any(value is None or np.isnan(value) for value in (ours_before, ours_after, normal)):
        return None

    realised_loss = (ours_after - ours_before) - 2 * normal
    if realised_loss <= 0:
        return None

    rivals = _rivals_around(
        pivot, medians, pitters, driver, lap, row_before, row_after, ours_before
    )
    if not rivals:
        return None

    result = project_positions(
        rivals,
        _STOP_NOW,
        _GROUND_TRUTH_CONFIG,
        np.array([realised_loss]),
        np.array([_GROUND_TRUTH_CLIFF_LAPS]),
    )
    return int(result.positions[0]) - _actual_position(pivot, row_after, driver, ours_after)


def measure_projection_ground_truth(years: tuple[int, ...] = (2023, 2024, 2025)) -> GroundTruth:
    """Project every real green-flag pit stop and compare with what actually happened.

    Neutralised stops are excluded, and not to flatter the number. Under a Safety
    Car every lap is slow, so the "two normal laps" baseline used to reconstruct
    the realised pit loss is wrong there: that corrupts the INPUT rather than the
    projection. Measuring the two separately showed exactly that signature, a
    mean error of +1.54 positions under neutralisation against +0.57 under green.

    Raises FileNotFoundError when ``data/raw/`` is absent, since a silently empty
    sample would report perfect accuracy over zero stops.
    """
    import pandas as pd

    raw = _raw_data_root()
    if raw is None:
        raise FileNotFoundError(
            "data/raw/ is not present; the ground truth needs the raw laps from the "
            "Hugging Face dataset (the featured parquet drops the pit and neutralised laps)"
        )

    errors: list[int] = []
    races = 0
    for year in years:
        year_dir = raw / str(year)
        if not year_dir.is_dir():
            continue

        for race_dir in sorted(path for path in year_dir.iterdir() if path.is_dir()):
            laps_path = race_dir / "laps.parquet"
            if not laps_path.exists():
                continue

            laps = pd.read_parquet(laps_path)
            if "PitInTime" not in laps.columns:
                continue

            races += 1
            pivot = _elapsed_pivot(laps)
            medians = laps.groupby("Driver")["LapTime"].median().dt.total_seconds().to_dict()
            neutralised = _neutralised_laps(laps)

            stops = laps[laps["PitInTime"].notna()][["Driver", "LapNumber"]]
            pitters: dict[int, set[str]] = {}
            for _, row in stops.iterrows():
                pitters.setdefault(int(row["LapNumber"]), set()).add(str(row["Driver"]))

            for _, stop in stops.iterrows():
                lap = int(stop["LapNumber"])
                if lap in neutralised or lap + 1 in neutralised:
                    continue
                error = project_one_stop(pivot, medians, pitters, str(stop["Driver"]), lap)
                if error is not None:
                    errors.append(error)

    return GroundTruth(errors=np.array(errors, dtype=int), races=races)


def _measured_tables() -> dict[str, Any]:
    """The six runtime tables, or an empty dict when the file has not been generated."""
    repo = _find_repo_root()
    path = (repo / MEASURED_TABLES_PATH) if repo else Path(MEASURED_TABLES_PATH)
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _table_rows(tables: dict[str, Any]) -> list[dict[str, Any]]:
    """One row per measured table: what it answers and how many cells back it."""
    rows = []
    for name, purpose in sorted(_TABLE_PURPOSE.items()):
        body = tables.get(name)
        cells = "-"
        if isinstance(body, dict):
            keyed = {k: v for k, v in body.items() if isinstance(v, dict)}
            if keyed:
                cells = str(sum(len(v) for v in keyed.values()))
        rows.append(
            {
                "table": name,
                "answers": purpose,
                "cells": cells,
                "present": body is not None,
            }
        )
    return rows


def _render_table(rows: list[dict[str, Any]], truth: GroundTruth | None, tables: dict) -> str:
    """Two sections: the accuracy headline, then the tables that feed the scorer."""
    lines = ["## Position projection against real pit stops", ""]
    if truth is None:
        lines += ["Not measured: `data/raw/` is absent from this checkout.", ""]
    else:
        lines += [
            "| quantity | value |",
            "|---|---|",
            f"| green-flag stops projected | {truth.sample_size} |",
            f"| races covered | {truth.races} |",
            f"| within one position | {truth.within_one:.1%} |",
            f"| exactly right | {truth.exact:.1%} |",
            f"| mean signed error (positions) | {truth.mean_signed_error:+.3f} |",
            f"| mean absolute error (positions) | {truth.mean_absolute_error:.3f} |",
            "",
            "Every real stop is a labelled example of the claim the projection makes, so",
            "this is measured accuracy and not a proxy. Neutralised stops are excluded",
            "because the pit-loss reconstruction, not the projection, is wrong under a",
            "Safety Car.",
            "",
        ]

    lines += ["## Measured tables the scorer reads", ""]
    if not tables:
        lines += [f"`{MEASURED_TABLES_PATH}` not present; run `scripts/measure_mc_tables.py`.", ""]
        return "\n".join(lines)

    lines += ["| table | answers | cells |", "|---|---|---|"]
    for row in rows:
        cells = row["cells"] if row["present"] else "MISSING"
        lines.append(f"| {row['table']} | {row['answers']} | {cells} |")

    races = tables.get("races_measured", "-")
    years = ", ".join(str(y) for y in tables.get("years", []))
    lines += [
        "",
        f"Counted off {races} races ({years}) of raw laps. The raw parquet is the source",
        "and not the featured one, which drops the neutralised and pit laps these tables",
        "are about.",
        "",
    ]
    return "\n".join(lines)


def build_projection_report() -> dict[str, Any]:
    """Write ``documents/eval_reports/projection.{md,json}`` and return the payload."""
    tables = _measured_tables()
    rows = _table_rows(tables)

    try:
        truth: GroundTruth | None = measure_projection_ground_truth()
    except FileNotFoundError:
        truth = None

    header = build_header(dataset="data/raw laps 2023-2025 (RAW, not featured)")
    md_path, json_path = write_report(
        "projection",
        header,
        _render_table(rows, truth, tables),
        {
            "ground_truth": None
            if truth is None
            else {
                "sample_size": truth.sample_size,
                "races": truth.races,
                "within_one": truth.within_one,
                "exact": truth.exact,
                "mean_signed_error": truth.mean_signed_error,
                "mean_absolute_error": truth.mean_absolute_error,
            },
            "tables": rows,
            "measured_tables_path": MEASURED_TABLES_PATH,
        },
    )
    return {
        "md_path": str(md_path),
        "json_path": str(json_path),
        "ground_truth": truth,
        "tables": rows,
    }
