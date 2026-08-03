"""Eval report for the guard rail's own bound: how long do real stints actually run?

``guard_rails.py`` refuses a pit call when ``tyre_life`` falls below the bound its
``_MIN_STINT_LAPS.get(compound, _DEFAULT_MIN_STINT)`` lookup resolves. This report
answers the question that decides whether those bounds are doing their job: over
every green-flag pit stop in 2023-2025, what share of real stints were shorter than
the bound that would have overridden them?

If the answer is close to zero, the bound sits where professional strategy never
goes and is a cheap safety net. If it is not close to zero, the bound is quietly
vetoing calls a real pit wall has made, which is a very different finding from
"the rail never fires". This report exists because nobody had counted it either way.

It is now also the standing calibration check, not just a one-off measurement.
`_MIN_STINT_LAPS` and `_DEFAULT_MIN_STINT` are IMPORTED, never restated, so every
run re-measures the bounds actually shipping rather than the ones that were shipping
when the report was written. #716 set them from this table against a 5% ceiling, and
`tests/eval/test_stint_lengths.py` asserts that ceiling holds.

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
from src.strategy.inference.guard_rails import _DEFAULT_MIN_STINT, _MIN_STINT_LAPS

# Dry compounds the guard rail actually gates. Order also fixes the report's
# row order, so a reader sees the softest, shortest-lived compound first.
_COMPOUNDS: tuple[str, ...] = ("SOFT", "MEDIUM", "HARD")

# INTERMEDIATE and WET, counted under the label of the bound they actually hit.
#
# This block used to claim wet compounds "run no minimum-stint rule at all", and
# the report dropped their stints on that basis. The claim is false: the rail
# resolves its bound with `_MIN_STINT_LAPS.get(compound, _DEFAULT_MIN_STINT)`, so
# a wet compound misses the three named entries and lands on the FALLBACK, which
# is a minimum-stint rule like any other. The parenthetical that followed was true
# in isolation ("never fires the SOFT/MEDIUM/HARD boundaries"), and that is how the
# wrong headline survived review. It cost this file its purpose on that one bound:
# when every bound was finally calibrated (#716), the fallback was the worst of
# them, and it was the only one nothing here had ever measured.
_WET_COMPOUNDS: frozenset[str] = frozenset({"INTERMEDIATE", "WET"})
_WET_BUCKET: str = "WET"

# Every bucket the report measures, in row order: the three compounds the rail
# names, then the fallback every other compound resolves to. Thresholds are read
# back out of the rail's own lookup rather than restated here, so this report
# cannot describe a bound the rail does not enforce.
_REPORTED_BUCKETS: tuple[str, ...] = (*_COMPOUNDS, _WET_BUCKET)


def _bound_for(bucket: str) -> int:
    """The minimum-stint bound the rail resolves for this bucket.

    The same expression `apply_guard_rails` evaluates, called here rather than
    mirrored, because a retyped boundary has shipped wrong in this codebase before.
    """
    return _MIN_STINT_LAPS.get(bucket, _DEFAULT_MIN_STINT)

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
    """Completed green-flag stint lengths across the sampled seasons, by bucket."""

    by_compound: dict[str, CompoundStints]
    dropped_missing: int
    races: int

    @property
    def total_counted(self) -> int:
        """Real green-flag stints that fed one of the four measured samples."""
        return sum(stats.sample_size for stats in self.by_compound.values())


def _compound_bucket(compound: str) -> str:
    """Which counting bucket a raw ``Compound`` reading belongs in.

    Returns the compound name itself for a dry compound, ``"WET"`` for
    INTERMEDIATE/WET, or ``"unknown"`` for anything else (there is no other label
    on a real green-flag pit lap; this is a defensive catch-all, not a case this
    dataset is expected to hit). Dataframe-free on purpose: this is the whole rule
    behind which bound a stop is graded against, and it needs to be checkable
    without a parquet file.
    """
    if compound in _COMPOUNDS:
        return compound
    if compound in _WET_COMPOUNDS:
        return _WET_BUCKET
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

    lengths_by_compound: dict[str, list[float]] = {bucket: [] for bucket in _REPORTED_BUCKETS}
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
                    if bucket == "unknown":
                        dropped_missing += 1
                    else:
                        lengths_by_compound[bucket].append(tyre_life)

    by_compound = {
        bucket: CompoundStints(
            compound=bucket,
            lengths=np.array(lengths_by_compound[bucket], dtype=float),
            threshold=_bound_for(bucket),
        )
        for bucket in _REPORTED_BUCKETS
    }
    return StintLengthSample(
        by_compound=by_compound,
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
        "field `apply_guard_rails` compares against its minimum-stint bound. A stint",
        "that ended because the race finished, or because the driver retired, is not a",
        "decision to stop and is excluded by construction: neither ever produces a",
        "lap with `PitInTime` set.",
        "",
        "The last row is INTERMEDIATE and WET together. They carry no entry of their",
        "own in `_MIN_STINT_LAPS`, so the rail's `.get(compound, _DEFAULT_MIN_STINT)`",
        "resolves them to the fallback -- a minimum-stint bound like any other, and",
        "one this report used to drop rather than measure.",
        "",
        f"| compound | n | threshold (laps) | shorter than threshold | {header_cells} |",
        f"|---|---|---|---|{divider_cells}",
    ]
    lines += [_compound_row(sample.by_compound[bucket]) for bucket in _REPORTED_BUCKETS]
    lines += [
        "",
        '"shorter than threshold" is the share of real stints the current guard rail',
        "would have overridden to STAY_OUT had a strategist tried to make that exact",
        "call: `TyreLife < the bound`, the same strict inequality `apply_guard_rails`",
        "itself uses.",
        "",
        "This share IS the calibration of a proscriptive bound, and the number the",
        "bounds are set from (#716). A bound exists so a generative model cannot emit",
        "nonsense, so it has to sit where real strategy essentially never goes; once it",
        "is vetoing a meaningful share of what professional pit walls actually did, it",
        "is separating unusual from usual rather than absurd from sane. The ceiling the",
        "bounds are held to is **5%**, and every row above is expected to clear it.",
        "",
        f"- real green-flag stints counted: {sample.total_counted} across {sample.races} races",
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
