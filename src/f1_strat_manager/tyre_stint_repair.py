"""Repair tyre-stint metadata that the F1 live-timing feed published wrong (#790).

The F1 `TimingAppData` feed sometimes drops the stint records for a stint. FastF1 copies
that faithfully and so do we -- verified: `fastf1.get_session(2025, "Miami", "R")` and
`data/raw/2025/Miami_Gardens/laps.parquet` carry the identical 446/457 NaN block, so
nothing in this repo's extraction loses it. The gap is upstream of us.

What makes it worth repairing rather than merely guarding is the SECOND defect, which is
invisible. When the feed recovers it does not resume the stint that was actually running:
it starts a NEW one at the recovery lap. At Miami 2025 the metadata reappears at lap 24
for the whole field, so every car reads `TyreLife 1` on a set that has done 24 racing
laps, and then:

  * cars that had not yet pitted carry a fabricated age until they do,
  * the count continues ACROSS the pit stop, which is physically impossible, and
  * the following stint's age is understated by the same 4-8 laps.

`TyreLife` feeds N26's TCN and N15/N16, and Miami 2025 is in the TEST season. A
degradation model reading "5 laps old" for a 29-lap-old tyre predicts no wear.

--- WHERE TO CHANGE IF THE ARTEFACT CHANGES ---
This runs at LOAD time, called from `laps_augment.augment_featured_laps`, for the same
reasons that module documents: the featured parquet is published on Hugging Face and
re-downloaded by `scripts/download_data.py`, so a locally-patched file would be silently
reverted, and its only producer is a read-only notebook (N04).

## The two repairs, and the one thing this never does

1. **Null the fabricated ages.** Where the feed invented a stint mid-race, the affected
   laps belong to the set the car started on, whose age is NOT recoverable. They become
   NaN -- a visible unknown replacing an invisible wrong integer.
2. **Move the misplaced boundary.** Where the car pitted before the feed recovered, the
   new stint really began on the out-lap, and from there the age IS recoverable.

It never invents a tyre age. A car may start a stint on a used set, and the offset is
real: measured across every healthy stint transition in 2024 and 2025, a FRESH set starts
at `TyreLife` 1.0 in 1156/1156 cases, while USED sets start anywhere from 2 to 16. The
rebuild therefore ANCHORS ON THE AGE THE FEED PUBLISHED at the boundary lap rather than
hardcoding 1 -- the feed mislabels WHERE a stint began but still reports the set's own
starting age, so subtracting one recovers the prior usage. (`FreshTyre` is deliberately
NOT the mechanism: on the affected laps it is a fabricated uniform `True`, so it carries
no information exactly where it would be needed.) When the boundary lap has no published
age, there is no anchor and nothing is rebuilt.

## Why a pit entry alone does not mean a new set

A stop-and-go or drive-through penalty sends the car down the pit lane with no work done,
so the stint correctly does NOT advance. Monaco 2025, RUS, lap 62 is exactly that -- a
served stop-and-go -- and its record is CORRECT. `find_misplaced_boundaries` therefore
requires the compound to change later AND no other pit entry to sit in between (a later
real stop explains the change by itself; that is the Monaco case).

## Known limits, stated rather than discovered later

* **Same compound refitted is invisible to the boundary rule.** 743 of 2594 stint
  transitions across the 71 shipped races fit the same compound again, and a misplaced
  boundary there cannot be seen by a compound-change test. No shipped race is affected
  today; the gap is structural. It errs toward missing a defect, never toward rewriting a
  correct record.
* **Raw-fed surfaces see feed values.** Consumers that read a raw parquet directly rather
  than through `augment_featured_laps` -- the replay engine and the backend's own race
  loader -- are not repaired. Unifying that is a wider change than this repair.
* **A stop made while the feed was dark is nulled, not rebuilt, even past the stop.**
  When the pre-stop compound was lost there is nothing left to confirm a tyre change with
  (a pit entry alone does not prove one -- see Monaco above), so `find_misplaced_boundaries`
  declines and the nulling pass takes the whole block to race end. Miami 2025 STR, HAD and
  BOR are this shape: 63 of their featured laps are post-stop and would have been
  recoverable to within 1-4 laps under the anchor assumption, and they become NaN instead.
  That is a deliberate trade -- a visible unknown over a probably-right integer on laps
  whose tyre change is inferred rather than evidenced -- and it is recorded here because
  the alternative reading, that those laps are simply unrecoverable, would be false.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

# Columns this module reads. A frame missing any of them is returned untouched rather
# than half-repaired, which is what the raw-parquet-absent path already does upstream.
REQUIRED_COLUMNS = ("Driver", "LapNumber", "Stint", "TyreLife", "Compound", "PitInTime")

# The extractor stringifies a missing compound, so absence arrives as one of these rather
# than as NaN. They must never be treated as "a different compound" (which would let
# data loss masquerade as evidence of a tyre change) nor written over a real value.
_COMPOUND_SENTINELS = frozenset({"", "nan", "none", "unknown"})


@dataclass
class RepairReport:
    """What a repair pass actually changed, so a caller can log or assert on it.

    Kept separate from the frame because the interesting property of this repair is how
    LITTLE it touches: a healthy race must come back byte-identical, and that is only
    checkable if the count of touched rows is reported rather than inferred.
    """

    boundaries_corrected: int = 0
    tyre_life_rebuilt: int = 0
    fabricated_ages_nulled: int = 0
    unknown_after: int = 0
    drivers_touched: list[str] = field(default_factory=list)

    @property
    def changed_anything(self) -> bool:
        return bool(self.boundaries_corrected or self.tyre_life_rebuilt or self.fabricated_ages_nulled)


def _is_real_compound(value: object) -> bool:
    """True when the value names an actual compound rather than absent data."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return False
    return str(value).strip().lower() not in _COMPOUND_SENTINELS


def _pit_in_laps(driver_laps: pd.DataFrame) -> list[int]:
    """Laps on which this driver entered the pit lane, ascending.

    `PitInTime` is the column the broken feed usually leaves intact, which is what makes
    the boundary repair possible: it is the independent ground truth the stint metadata
    failed to agree with. It is not guaranteed -- Miami 2025 BEA has no pit record at all
    -- which is why the age-nulling repair below does not depend on it.
    """
    entered = driver_laps.loc[driver_laps["PitInTime"].notna(), "LapNumber"]
    return sorted(int(lap) for lap in entered.dropna())


def _first_compound_change_after(driver_laps: pd.DataFrame, lap: int) -> Optional[int]:
    """First lap after ``lap`` whose compound is a DIFFERENT REAL compound.

    Sentinels are skipped rather than counted as a change: `'HARD' -> 'None'` is the feed
    dying, not a pit stop, and treating it as evidence of a new set is how a repair ends
    up writing a data-loss marker over a real compound.
    """
    current = driver_laps.loc[driver_laps["LapNumber"] == lap, "Compound"]
    if current.empty or not _is_real_compound(current.iloc[0]):
        return None
    later = driver_laps[driver_laps["LapNumber"] > lap]
    for _, row in later.iterrows():
        if _is_real_compound(row["Compound"]) and row["Compound"] != current.iloc[0]:
            return int(row["LapNumber"])
    return None


def find_misplaced_boundaries(driver_laps: pd.DataFrame) -> list[tuple[int, int]]:
    """Pit stops whose stint boundary landed on the wrong lap, for one driver.

    Returns ``(pit_lap, boundary_lap)`` pairs: the car pitted on ``pit_lap`` but the
    metadata only starts the new stint at ``boundary_lap``.

    Both conditions must hold, and the second one is not optional -- see the module
    docstring's Monaco case, where dropping it turns a correct record into a wrong one.
    """
    driver_laps = driver_laps.sort_values("LapNumber")
    pit_laps = _pit_in_laps(driver_laps)
    misplaced: list[tuple[int, int]] = []

    for pit_lap in pit_laps:
        boundary = _first_compound_change_after(driver_laps, pit_lap)
        if boundary is None:
            continue  # no real tyre change: a served penalty, or the feed went dark
        if boundary - pit_lap <= 1:
            continue  # the new stint starts on the out-lap, which is correct
        if any(pit_lap < lap < boundary for lap in pit_laps):
            continue  # a later real stop explains the change
        misplaced.append((pit_lap, boundary))

    return misplaced


def _fabricated_age_mask(driver_laps: pd.DataFrame) -> pd.Series:
    """Laps whose ``TyreLife`` the feed invented after going dark mid-race.

    The signature is a stint that appears out of nowhere: the metadata is absent at the
    driver's first lap and later resumes on a lap that is NOT a pit out-lap. Everything
    from that recovery up to the driver's first real pit stop is still the race-start set,
    whose age was never published, so the published numbers there count from the wrong
    zero and are provably too small.

    This is what covers the cars a boundary correction cannot reach: a driver who pits
    only after the feed recovered (Miami GAS, HUL) has no misplaced boundary at all, yet
    carries the same fabricated block.
    """
    ordered = driver_laps.sort_values("LapNumber")
    blank = pd.Series(False, index=driver_laps.index)
    if ordered.empty or ordered["TyreLife"].notna().iloc[0]:
        return blank  # the feed was healthy at the start: nothing was invented

    known = ordered[ordered["TyreLife"].notna()]
    if known.empty:
        return blank  # never recovered; the laps are already an honest NaN

    recovery_lap = int(known["LapNumber"].iloc[0])
    out_laps = {lap + 1 for lap in _pit_in_laps(ordered)}
    if recovery_lap in out_laps:
        return blank  # the stint really did start here, via a pit stop

    later_stops = [lap for lap in _pit_in_laps(ordered) if lap >= recovery_lap]
    end_lap = later_stops[0] if later_stops else int(ordered["LapNumber"].max())

    laps = driver_laps["LapNumber"]
    return (laps >= recovery_lap) & (laps <= end_lap) & driver_laps["TyreLife"].notna()


def _rebuild_driver(driver_laps: pd.DataFrame, report: RepairReport) -> pd.DataFrame:
    """Null one driver's fabricated ages, then correct any misplaced stint boundary."""
    repaired = driver_laps.sort_values("LapNumber").copy()
    touched = False

    fabricated = _fabricated_age_mask(repaired)
    if fabricated.any():
        repaired.loc[fabricated, "TyreLife"] = float("nan")
        report.fabricated_ages_nulled += int(fabricated.sum())
        touched = True

    lap_numbers = repaired["LapNumber"]
    for pit_lap, boundary in find_misplaced_boundaries(repaired):
        at_boundary = repaired[lap_numbers == boundary]
        if at_boundary.empty:
            continue

        # Everything the rebuild needs is resolved BEFORE anything is written. A partial
        # apply -- rewriting `Compound` and then bailing on a missing `Stint` -- would
        # return a mutated frame while reporting no change, which the caller reads as
        # "nothing to patch" and so would leave the raw and featured views disagreeing.
        stint_id = at_boundary["Stint"].iloc[0]
        published_age = at_boundary["TyreLife"].iloc[0]
        if pd.isna(stint_id) or pd.isna(published_age):
            # Without a published age there is no anchor, and counting from 1 would
            # invent a fresh set -- the one thing this module refuses to do. The laps
            # keep whatever they had; the nulling pass above already handled the ones
            # that were provably fabricated.
            continue

        # The new set goes on during the stop, so the out-lap is the new stint's first
        # lap. Its age is the set's prior usage plus one, not a hardcoded 1.
        out_lap = pit_lap + 1
        prior_usage = max(0.0, float(published_age) - 1.0)
        in_new_stint = lap_numbers >= out_lap
        mislabelled = in_new_stint & (lap_numbers < boundary)

        new_compound = at_boundary["Compound"].iloc[0]
        if _is_real_compound(new_compound):
            repaired.loc[mislabelled, "Compound"] = new_compound
        repaired.loc[mislabelled, "Stint"] = stint_id

        of_this_stint = in_new_stint & (repaired["Stint"] == stint_id)
        repaired.loc[of_this_stint, "TyreLife"] = (
            lap_numbers[of_this_stint] - out_lap + 1 + prior_usage
        ).astype(float)
        report.tyre_life_rebuilt += int(of_this_stint.sum())
        report.boundaries_corrected += 1
        touched = True

    if touched and len(repaired):
        report.drivers_touched.append(str(repaired["Driver"].iloc[0]))
    return repaired


def repair_tyre_stints(laps: pd.DataFrame) -> tuple[pd.DataFrame, RepairReport]:
    """Null fabricated tyre ages and correct misplaced stint boundaries.

    Only rows that are demonstrably wrong are touched, so a healthy race comes back
    unchanged by construction rather than by a comparison after the fact.

    Args:
        laps: A per-race laps frame carrying at least ``REQUIRED_COLUMNS``.

    Returns:
        The repaired frame and a ``RepairReport`` describing exactly what moved. The
        frame is returned unchanged, with an empty report, when the required columns are
        absent -- the same degrade-quietly contract the raw-parquet merge upstream uses.
    """
    report = RepairReport()
    if not set(REQUIRED_COLUMNS).issubset(laps.columns):
        return laps, report

    # dropna=False: a NaN driver key would otherwise be dropped by the groupby and come
    # back all-NaN through the reindex, destroying columns this repair has no business
    # touching. No shipped race carries one, but this runs on every future download.
    groups = [_rebuild_driver(group, report) for _, group in laps.groupby("Driver", sort=False, dropna=False)]
    repaired = pd.concat(groups).reindex(laps.index)

    # Counted AFTER the repair, so the number means what the log says it means.
    unknown = repaired["TyreLife"].isna()
    if "Position" in repaired.columns:
        unknown &= repaired["Position"].notna()
    report.unknown_after = int(unknown.sum())

    if report.changed_anything:
        logger.info(
            "Tyre-stint repair on %d driver(s): %d fabricated age(s) nulled, %d boundary/ies "
            "corrected, %d age(s) rebuilt; %d racing lap(s) now carry an honest unknown age (#790)",
            len(report.drivers_touched),
            report.fabricated_ages_nulled,
            report.boundaries_corrected,
            report.tyre_life_rebuilt,
            report.unknown_after,
        )
    return repaired, report
