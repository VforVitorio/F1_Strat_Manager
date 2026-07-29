"""Eval report for the guard rail's own bound: how long do real stints actually run?

``guard_rails.py`` refuses a pit call when ``tyre_life < _MIN_STINT_LAPS[compound]``
(SOFT 8, MEDIUM 12, HARD 15 laps). Those numbers have never been checked against a
single real race. This report answers the question that decides whether they are
doing their job: over every green-flag pit stop in 2023-2025, what share of real
stints were shorter than the bound that would have overridden them?

If the answer is close to zero, the bound sits where professional strategy never
goes and is a cheap safety net. If it is not close to zero, the bound is quietly
vetoing calls a real pit wall has made, which is a very different finding from
"the rail never fires" — this report exists because nobody had counted it either
way.

WHAT COUNTS AS A STINT LENGTH
------------------------------
The length recorded for a completed stint is the ``TyreLife`` reading on the lap
of the stop, not the number of race laps run on that tyre set within this race.
The two usually agree, but ``TyreLife`` also counts laps a set ran before the
race (most commonly tyres carried over from qualifying), and it is exactly the
field ``apply_guard_rails`` reads at decision time. Measuring anything else would
answer a question the guard rail does not ask.

A stint that ended because the race finished, or because the driver retired, is
not a decision to stop, and both are excluded by construction rather than by an
extra filter: neither ever produces a lap with ``PitInTime`` set, so neither ever
appears in ``green_flag_stops``.

WHERE TO CHANGE IF THINGS MOVE:
- ``src/strategy/inference/guard_rails.py`` owns ``_MIN_STINT_LAPS``. It is
  imported here rather than restated, for the reason its own module docstring
  gives: a retyped boundary has shipped wrong in this codebase before.
- ``src/strategy/eval/projection.py`` owns ``green_flag_stops``, the definition
  of a real, gradeable pit stop that every eval tier in this package shares.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from src.strategy.eval.projection import _neutralised_laps, _raw_data_root, green_flag_stops
from src.strategy.eval.report import build_header, write_report
from src.strategy.inference.guard_rails import _MIN_STINT_LAPS

# Dry compounds the guard rail actually gates. Order also fixes the report's
# row order, so a reader sees the softest, shortest-lived compound first.
_COMPOUNDS: tuple[str, ...] = ("SOFT", "MEDIUM", "HARD")

# Wet compounds run no minimum-stint rule at all (`_MIN_STINT_LAPS.get(compound,
# _DEFAULT_MIN_STINT)` never fires the SOFT/MEDIUM/HARD boundaries for them), so a
# wet stint answers a different question than the one this report asks.
_WET_COMPOUNDS: frozenset[str] = frozenset({"INTERMEDIATE", "WET"})

# The eight points the task and the report both read: the extremes, the tails
# that matter for a minimum-bound question, and the median. Kept as one ordered
# table so `CompoundStints.summary()` and the markdown header can share it
# instead of each hand-listing the same eight numbers.
_PERCENTILE_POINTS: tuple[tuple[str, float], ...] = (
    ("min", 0),
    ("p1", 1),
    ("p5", 5),
    ("p10", 10),
    ("p25", 25),
    ("median", 50),
    ("p75", 75),
    ("max", 100),
)


@dataclass(frozen=True)
class CompoundStints:
    """Completed green-flag stint lengths for one dry compound, in tyre-age laps.

    ``threshold`` travels with the sample rather than being looked up again at
    render time, so a report and a test reading the same ``CompoundStints`` can
    never disagree about which bound they are comparing against.
    """

    compound: str
    lengths: np.ndarray
    threshold: int

    @property
    def sample_size(self) -> int:
        return int(self.lengths.size)

    def percentile(self, q: float) -> float:
        """The q-th percentile of stint length in laps, or 0.0 over an empty sample.

        0.0 rather than NaN so a report or a JSON payload never has to special-case
        the empty compound: it reads as "no data", not as a stint of zero laps,
        because ``sample_size`` is the field that actually says whether there was
        a sample at all.
        """
        if self.sample_size == 0:
            return 0.0
        return float(np.percentile(self.lengths, q))

    @property
    def share_below_threshold(self) -> float:
        """Share of real stints strictly SHORTER than the guard rail's own minimum.

        Strict, matching ``apply_guard_rails``'s own ``tyre_life < min_life``: a
        stint that ended exactly on the boundary is a stint the rail would have
        allowed, not one it would have blocked, and folding it into "below" would
        overstate how often the rail actually binds.
        """
        if self.sample_size == 0:
            return 0.0
        return float((self.lengths < self.threshold).mean())

    def summary(self) -> dict[str, float]:
        """The eight percentile points in `_PERCENTILE_POINTS`, by label."""
        return {label: self.percentile(q) for label, q in _PERCENTILE_POINTS}


@dataclass(frozen=True)
class StintLengthSample:
    """Completed green-flag stint lengths across the sampled seasons, by compound."""

    by_compound: dict[str, CompoundStints]
    dropped_wet: int
    dropped_missing: int
    races: int

    @property
    def total_counted(self) -> int:
        """Real green-flag stints that fed one of the three dry-compound samples."""
        return sum(stats.sample_size for stats in self.by_compound.values())


def _compound_bucket(compound: str) -> str:
    """Which counting bucket a raw ``Compound`` reading belongs in.

    Returns the compound name itself for a dry compound, ``"wet"`` for
    INTERMEDIATE/WET, or ``"unknown"`` for anything else (there is no other label
    on a real green-flag pit lap; this is a defensive catch-all, not a case this
    dataset is expected to hit). Dataframe-free on purpose: this is the whole
    rule behind "ignore wet compounds but report how many you dropped", and it
    needs to be checkable without a parquet file.
    """
    if compound in _COMPOUNDS:
        return compound
    if compound in _WET_COMPOUNDS:
        return "wet"
    return "unknown"


def _stop_tyre(laps, driver: str, lap: int) -> tuple[str | None, float | None]:
    """Compound and TyreLife on the lap a driver actually pitted, or (None, None).

    The ``PitInTime`` row IS the last lap of the ending stint, so its own
    ``Compound`` and ``TyreLife`` are exactly the reading ``apply_guard_rails``
    would have seen at that moment; no ``Stint``-group aggregation is needed.
    Either field missing marks the stop unusable rather than guessing at a bucket
    for it, the same choice ``decision_modes._stop_context`` makes for the same
    two columns.
    """
    import pandas as pd

    row = laps[(laps["Driver"] == driver) & (laps["LapNumber"] == lap)]
    if not len(row):
        return None, None

    compound = row["Compound"].iloc[0]
    tyre_life = row["TyreLife"].iloc[0]
    if pd.isna(compound) or pd.isna(tyre_life):
        return None, None
    return str(compound), float(tyre_life)


def measure_stint_lengths(years: tuple[int, ...] = (2023, 2024, 2025)) -> StintLengthSample:
    """Completed green-flag stint lengths for every driver in every sampled race.

    Raises FileNotFoundError when ``data/raw/`` is absent, since a silently empty
    sample would report a 0% below-threshold share that means nothing rather than
    the honest "not measured".
    """
    import pandas as pd

    raw = _raw_data_root()
    if raw is None:
        raise FileNotFoundError(
            "data/raw/ is not present; the stint-length distribution needs the raw "
            "laps from the Hugging Face dataset (Stint/Compound/TyreLife are dropped "
            "from the featured parquet)"
        )

    lengths_by_compound: dict[str, list[float]] = {compound: [] for compound in _COMPOUNDS}
    dropped_wet = 0
    dropped_missing = 0
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
            neutralised = _neutralised_laps(laps)

            for driver, stop_laps in green_flag_stops(laps, neutralised).items():
                for lap in stop_laps:
                    compound, tyre_life = _stop_tyre(laps, driver, lap)
                    if compound is None:
                        dropped_missing += 1
                        continue

                    bucket = _compound_bucket(compound)
                    if bucket == "wet":
                        dropped_wet += 1
                    elif bucket == "unknown":
                        dropped_missing += 1
                    else:
                        lengths_by_compound[bucket].append(tyre_life)

    by_compound = {
        compound: CompoundStints(
            compound=compound,
            lengths=np.array(lengths_by_compound[compound], dtype=float),
            threshold=_MIN_STINT_LAPS[compound],
        )
        for compound in _COMPOUNDS
    }
    return StintLengthSample(
        by_compound=by_compound,
        dropped_wet=dropped_wet,
        dropped_missing=dropped_missing,
        races=races,
    )


def _compound_row(stats: CompoundStints) -> str:
    """One markdown table row: sample size, threshold, the headline share, then the percentiles."""
    summary = stats.summary()
    percentile_cells = " | ".join(f"{summary[label]:.1f}" for label, _ in _PERCENTILE_POINTS)
    return (
        f"| {stats.compound} | {stats.sample_size} | {stats.threshold} | "
        f"{stats.share_below_threshold:.1%} | {percentile_cells} |"
    )


def _render_table(sample: StintLengthSample | None) -> str:
    """Per-compound percentile table plus the drop counts, or a not-measured note."""
    if sample is None:
        return (
            "Not measured: `data/raw/` is absent from this checkout. Pull the "
            "Hugging Face dataset and re-run.\n"
        )

    header_cells = " | ".join(label for label, _ in _PERCENTILE_POINTS)
    divider_cells = "---|" * len(_PERCENTILE_POINTS)
    lines = [
        "## Real green-flag stint lengths by compound (2023-2025 raw laps)",
        "",
        "Every completed stint that ended in a real green-flag pit stop, counted in",
        "tyre-age laps: the `TyreLife` reading at the moment of the stop, the exact",
        "field `apply_guard_rails` compares against `_MIN_STINT_LAPS`. A stint that",
        "ended because the race finished, or because the driver retired, is not a",
        "decision to stop and is excluded by construction: neither ever produces a",
        "lap with `PitInTime` set.",
        "",
        f"| compound | n | threshold (laps) | shorter than threshold | {header_cells} |",
        f"|---|---|---|---|{divider_cells}",
    ]
    lines += [_compound_row(sample.by_compound[compound]) for compound in _COMPOUNDS]
    lines += [
        "",
        '"shorter than threshold" is the share of real stints the current guard rail',
        "would have overridden to STAY_OUT had a strategist tried to make that exact",
        "call: `TyreLife < _MIN_STINT_LAPS[compound]`, the same strict inequality",
        "`apply_guard_rails` itself uses. Close to zero means the bound sits where",
        "real strategy essentially never goes; anywhere else means it is vetoing",
        "calls a real pit wall has actually made.",
        "",
        f"- real green-flag stints counted: {sample.total_counted} across {sample.races} races",
        "- wet-compound stops dropped (INTERMEDIATE/WET is not a dry-tyre-life "
        f"question): {sample.dropped_wet}",
        f"- stops dropped for missing compound/tyre-life data: {sample.dropped_missing}",
        "",
    ]
    return "\n".join(lines)


def build_stint_lengths_report() -> dict[str, Any]:
    """Write ``documents/eval_reports/stint_lengths.{md,json}`` and return the payload."""
    try:
        sample: StintLengthSample | None = measure_stint_lengths()
    except FileNotFoundError:
        sample = None

    header = build_header(dataset="data/raw laps 2023-2025 (RAW, not featured)")
    md_path, json_path = write_report(
        "stint_lengths",
        header,
        _render_table(sample),
        {
            "sample": None
            if sample is None
            else {
                "races": sample.races,
                "total_counted": sample.total_counted,
                "dropped_wet": sample.dropped_wet,
                "dropped_missing": sample.dropped_missing,
                "compounds": {
                    compound: {
                        "sample_size": stats.sample_size,
                        "threshold": stats.threshold,
                        "share_below_threshold": stats.share_below_threshold,
                        **stats.summary(),
                    }
                    for compound, stats in sample.by_compound.items()
                },
            }
        },
    )
    return {
        "md_path": str(md_path),
        "json_path": str(json_path),
        "sample": sample,
    }
