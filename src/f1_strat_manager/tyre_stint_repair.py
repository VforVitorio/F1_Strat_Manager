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

## The three repairs, and the one thing this never does

1. **Null the fabricated ages.** Where the feed invented a stint mid-race, the affected
   laps belong to the set the car started on, whose age is NOT recoverable. They become
   NaN -- a visible unknown replacing an invisible wrong integer.
2. **Null the republished ages (#988).** Where the age RESTARTS AT 1 with the compound
   unchanged AND the pit passage sits in a run of consecutive entries, the feed reset the
   count on a set that never came off the car. Melbourne 2025 is the case: five cars reset
   to 1 on a safety-car pit-lane transit and then count on from that wrong zero for the
   rest of the run. Nulled to the next real compound change. Both conditions are load
   bearing and neither is sufficient alone: see `_age_restarted_on_an_unchanged_set` and
   `_entered_the_pits_twice_running`.
3. **Move the misplaced boundary.** Where the car pitted before the feed recovered, the
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
* **And it is what bounds repair 2 in the other direction.** A car that fits another set
  of the SAME compound, is published as FRESH, and does it on a lap adjacent to another
  pit entry produces exactly what a republished age produces, in every column this module
  reads, and those laps would be nulled. Two conditions have to coincide for that, and the
  second means the car was in the pit lane on two consecutive laps -- which cars DO do, about
  one driver per two races, so the residual is reachable rather than hypothetical. It has not
  been observed: the repair nulls nothing on any race checked beyond Melbourne. On Melbourne
  2025 it nulls 162 of 927 rows (17.5%), the whole opening stint of five cars, and leaves all
  24 of that race's genuine stops alone.
* **A nulled age is not neutral downstream, and that bounds how far this should ever go.**
  `pit_strategy_agent._tyre_life_in` reads a NaN `TyreLife` as a FRESH SET (1), so a null
  reaches N15 as a number, not as an unknown, and a wronger one than the artefact it
  replaced. That is why repair 2 is scoped to a shape that is provably not a stop rather
  than to every drop this module cannot explain.
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
    republished_ages_nulled: int = 0
    unknown_after: int = 0
    drivers_touched: list[str] = field(default_factory=list)

    @property
    def changed_anything(self) -> bool:
        return bool(
            self.boundaries_corrected
            or self.tyre_life_rebuilt
            or self.fabricated_ages_nulled
            or self.republished_ages_nulled
        )


def is_real_compound(value: object) -> bool:
    """True when the value names an actual compound rather than absent data.

    Public because it carries the sentinel rule above, and PITWALL's stop count
    needs the same rule: a stringified missing compound must never read as
    evidence of a tyre change. A second copy of `_COMPOUND_SENTINELS` in another
    module is the twin this repo pays for more often than any other defect.
    """
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
    if current.empty or not is_real_compound(current.iloc[0]):
        return None
    later = driver_laps[driver_laps["LapNumber"] > lap]
    for _, row in later.iterrows():
        if is_real_compound(row["Compound"]) and row["Compound"] != current.iloc[0]:
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


def _republished_age_mask(driver_laps: pd.DataFrame) -> pd.Series:
    """Laps whose ``TyreLife`` the feed reset on a set that never came off the car.

    The signature is an age that RESTARTS AT 1 while the compound stays the same real
    compound. A set cannot get younger, so one of the two readings is wrong, and the
    restart is the one with no corroboration: nothing else in the record changed. Why the
    landing value carries the rule, rather than the drop, is measured in
    `_age_restarted_on_an_unchanged_set` - a plain drop test fires on 435 genuine pit
    stops.

    Melbourne 2025 is the shape this exists for. The safety car led the field through
    the pit lane on laps 2, 3 and 4, and FastF1 opens a new stint on every one of those
    transits for all seventeen runners. Twelve of them carry the age forward correctly
    (NOR reads 2, 3, 4, 5). Five do not: ALB and STR reset to 1 on lap 3, LAW on lap 4,
    BEA and OCO on lap 5, all with INTERMEDIATE on the car throughout.

    **The mask runs to the next real compound change, not to the end of the drop.** The
    feed does not resume the true count after republishing; it counts on from the wrong
    zero for the rest of the run. STR reads 29 at lap 33 on a set that has done 33 laps,
    OCO 35 at lap 39 on a set that has done 39. Nulling only the transit lap would leave
    thirty laps of an age that is short by up to four.

    --- WHY THIS NULLS RATHER THAN REBUILDS ---
    The count IS recoverable by anchoring on the last honest age and continuing it, which
    is the move `_rebuild_driver` already makes at a real boundary. It is not made here,
    because the anchor there is backed by evidence a set changed (a pit entry AND a later
    compound change) while the anchor here would rest on the absence of evidence that one
    did. A set refitted with the SAME compound is invisible to every rule in this module,
    as the module docstring states, so continuing the count would publish a confident
    number on exactly the case the module cannot see. A visible unknown is the trade this
    module already makes everywhere else.

    --- WHY PIT-LANE DURATION IS NOT PART OF THE RULE ---
    It cannot separate the two populations. Measured across all 82 pit-lane passages of
    Melbourne 2025: transits that carry no age drop run 12.8 to 18.8 s, real stops run
    17.9 to 26.1 s, and the ranges OVERLAP. Two of the five flagged transits are longer
    than the median real stop (OCO 19.7 s, BEA 22.3 s), so a duration test would clear
    them and leave the artefact in place on the two cars whose age is short by four laps.
    """
    ordered = driver_laps.sort_values("LapNumber")
    marked = pd.Series(False, index=driver_laps.index)
    lap_numbers = driver_laps["LapNumber"]
    last_lap = int(ordered["LapNumber"].iloc[-1]) if len(ordered) else 0

    entries = _pit_in_laps(ordered)
    for position in range(1, len(ordered)):
        if not _age_restarted_on_an_unchanged_set(ordered, position):
            continue
        drop_lap = int(ordered["LapNumber"].iloc[position])
        if not _entered_the_pits_twice_running(entries, drop_lap):
            continue
        change_lap = _first_compound_change_after(ordered, drop_lap)
        end_lap = change_lap - 1 if change_lap is not None else last_lap
        in_run = (lap_numbers >= drop_lap) & (lap_numbers <= end_lap)
        marked |= in_run & driver_laps["TyreLife"].notna()

    return marked


def _entered_the_pits_twice_running(entries: list[int], out_lap: int) -> bool:
    """Was the pit passage this out-lap belongs to part of a RUN of consecutive ones?

    **This is a JOINT condition and it does not stand alone. Cars really do pit on two
    consecutive laps.** An earlier version of this docstring asserted they do not, which is
    false and was measured false: across eight races, five drivers made two genuine stops on
    adjacent laps (2025 Las Vegas ALB twice, at entries 13/14 and 34/35, and BOR at 1/2;
    2025 Qatar BEA at 40/41 and OCO at 8/9; 2024 Imola ALB at 8/9), roughly one driver per
    two races. Qatar's OCO goes MEDIUM (8 laps) to a fresh HARD to a fresh MEDIUM on three
    consecutive laps. What blocks the null on every one of them is the COMPOUND changing or
    the previous age already reading 1, not this test.

    What this condition is actually for is the case those two miss: a fresh set of the SAME
    compound, which reads exactly like a republished age. That case is not rare, it is the
    modal raw-frame shape. Measured by neutralising this condition and re-running the repair
    on real sessions:

    * **2025 Qatar: 428 rows across 17 drivers** would be nulled without it, and 0 with it;
    * 2025 Las Vegas: 21 rows on 1 driver without it, 0 with it;
    * Melbourne 2025 is unchanged at 162 rows on 5 drivers either way.

    So the pairing is what works: an age restarting at 1 on an unchanged compound is common
    and usually a real stop; an age restarting at 1 on an unchanged compound **while the car
    is in the pit lane on adjacent laps** is the artefact. At Melbourne all seventeen runners
    have `PitInTime` on laps 2, 3 AND 4, because the safety car led the field through.

    Over that race's 29 passages ending in an age of 1: the five same-compound artefacts sit
    in a run, 5 of 5; the twenty-four genuine stops do not, 0 of 24.

    --- WHY NOT THE FEATURED FRAMES ---
    They drop the out-lap, which is the lap a fresh set reads 1 on, so a genuine stop's first
    VISIBLE row there reads 2 by construction: 0 age-1 rows on Melbourne's featured slice
    against 52 in the raw frame, and 14 in the whole of 2025. Any "no real stop lands on 1"
    count taken there measures the filtering. This condition is measured on raw frames only.

    The failure mode is missing an artefact, never nulling a real age, which is the direction
    that matters: `pit_strategy_agent._tyre_life_in` reads a NaN age as a fresh set, so a
    false null publishes a number rather than an unknown.
    """
    entry_lap = out_lap - 1
    if entry_lap not in entries:
        return False
    return (entry_lap - 1) in entries or (entry_lap + 1) in entries


def _age_restarted_on_an_unchanged_set(ordered: pd.DataFrame, position: int) -> bool:
    """True when the age RESTARTS at 1 on the compound the car is already running.

    This is only HALF the rule. `_republished_age_mask` also requires the pit passage to
    sit in a run of consecutive entries, and that second condition is the one carrying
    the weight. Read both before changing either.

    **A plain "the age dropped" test fires on every genuine same-compound pit stop**,
    which is the modal dry strategy of this regulation: across the shipped 2023-2025
    featured laps there are 435 such drops in 53 grands prix, and the `Stint` column
    advances on all 435 of them. Nulling those would take 9.7 to 15.2 % of a season's
    ages, and N15 reads a NaN `TyreLife` as a fresh set, so the loss would come back as
    a number further from the truth than the one the feed published. Requiring a landing
    on exactly 1 removes all 435, and Melbourne 2025's five republished ages all land on
    exactly 1, so the pair of conditions keeps the artefact and drops the false positives.

    --- WHAT THAT 435 DOES AND DOES NOT SHOW ---
    It shows the plain drop test is unusable. It does **not** show that a genuine
    same-compound refit lands on 2 or more, and an earlier version of this docstring
    claimed exactly that. The featured parquet DROPS THE OUT-LAP, which is the lap a
    fresh set reads 1 on, so a genuine stop's first visible row there reads 2 by
    construction: Melbourne's featured slice holds **0** rows with `TyreLife == 1.0`
    against **52** in the raw frame it is built from, and the whole 2025 season holds 14
    of 22,760. A "nothing real lands on 1" measurement taken there is a property of the
    filtering. The repair runs on the RAW frames, where landing on 1 is ordinary, and
    that is why the pit-run condition exists.

    Both compounds must NAME a compound: a `HARD -> 'None'` pair is the feed dying, and
    reading it as "the same set" would let data loss mark a run for nulling.
    """
    previous_age = ordered["TyreLife"].iloc[position - 1]
    current_age = ordered["TyreLife"].iloc[position]
    if pd.isna(previous_age) or pd.isna(current_age):
        return False
    restarted = current_age == 1.0 and previous_age > 1.0
    if not restarted:
        return False
    previous_compound = ordered["Compound"].iloc[position - 1]
    current_compound = ordered["Compound"].iloc[position]
    both_named = is_real_compound(previous_compound) and is_real_compound(current_compound)
    return both_named and previous_compound == current_compound


def _rebuild_driver(driver_laps: pd.DataFrame, report: RepairReport) -> pd.DataFrame:
    """Null one driver's fabricated ages, then correct any misplaced stint boundary."""
    repaired = driver_laps.sort_values("LapNumber").copy()
    touched = False

    fabricated = _fabricated_age_mask(repaired)
    if fabricated.any():
        repaired.loc[fabricated, "TyreLife"] = float("nan")
        report.fabricated_ages_nulled += int(fabricated.sum())
        touched = True

    # After the nulling pass above, so a block the feed never published cannot also be
    # read as a set getting younger. The two artefacts are disjoint on every race
    # measured, and the order makes that a property rather than a coincidence.
    # `republished_ages_nulled` counts what this pass NULLED, which is not always what the
    # frame returns: the boundary rebuild below rewrites `TyreLife` on the stint it
    # corrects, and that stint can overlap a run this pass just nulled. `unknown_after` is
    # the count of unknowns that survive; this one is the count of ages this pass refused
    # to trust. No shipped race carries both artefacts on one driver.
    republished = _republished_age_mask(repaired)
    if republished.any():
        repaired.loc[republished, "TyreLife"] = float("nan")
        report.republished_ages_nulled += int(republished.sum())
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
        if is_real_compound(new_compound):
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
    groups = [
        _rebuild_driver(group, report)
        for _, group in laps.groupby("Driver", sort=False, dropna=False)
    ]
    repaired = pd.concat(groups).reindex(laps.index)

    # Counted AFTER the repair, so the number means what the log says it means.
    unknown = repaired["TyreLife"].isna()
    if "Position" in repaired.columns:
        unknown &= repaired["Position"].notna()
    report.unknown_after = int(unknown.sum())

    if report.changed_anything:
        logger.info(
            "Tyre-stint repair on %d driver(s): %d fabricated age(s) nulled, %d republished "
            "age(s) nulled, %d boundary/ies corrected, %d age(s) rebuilt; %d racing lap(s) now "
            "carry an honest unknown age (#790, #988)",
            len(report.drivers_touched),
            report.fabricated_ages_nulled,
            report.republished_ages_nulled,
            report.boundaries_corrected,
            report.tyre_life_rebuilt,
            report.unknown_after,
        )
    return repaired, report
