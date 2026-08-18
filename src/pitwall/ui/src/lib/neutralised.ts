/**
 * Which laps the field was NOT racing freely on, from the label the wire carries.
 *
 * **One module because TWO panels need it**, and a second reduction over
 * `bulk.drivers` is the twin this repo pays for more than any other defect: the
 * race-pace grid marks the lap in its own column and the race trace shades the
 * same lap range on its x axis, and two panels disagreeing about which laps were
 * neutralised would be worse than neither marking them.
 *
 * The rule itself is not here. `LapRow.neutralised` arrives decoded by
 * `src/arcade/track_status.py`, so nothing in TypeScript knows that a `4` means
 * a safety car.
 */

import type { Bulk } from "./bridge";

/**
 * `lap -> label`, for every revealed lap any car was neutralised on.
 *
 * **ANY row, not a majority of them**, and the conservative direction is the
 * point: the label is a warning that this lap's ranking is not about pace, and a
 * lap where the safety car was out for part of the field is exactly as unsafe to
 * read as one where it was out for all of it.
 *
 * Measured through this code's own decode path on the real race, exactly two laps
 * are mixed: **lap 33, marked on 13 of 16 rows, and lap 46, marked on 3 of 15**.
 * Lap 46 is the case that makes ANY-row the RIGHT rule rather than an incidental
 * one - a majority vote would drop its rail while three cars queue through the pit
 * lane.
 *
 * (This paragraph used to say the mixed laps were 33, 34 and 47 and that the SC
 * digit was "on the majority of rows anyway". Both were measured on the RAW
 * `TrackStatus` STRINGS rather than on the decoded label, and `'124'`, `'24'` and
 * `'4'` are three different strings that all decode to SAFETY CAR - so the figure
 * described a population the sentence was not about, and the false clause was the
 * one justifying the rule.)
 *
 * Generated rows are skipped: FastF1 synthesised them for a car that never
 * finished the lap, and they count towards nothing anywhere else either.
 */
export function neutralisedLaps(bulk: Bulk | null): Map<number, string> {
  const laps = new Map<number, string>();
  if (!bulk?.available) return laps;
  for (const driver of Object.values(bulk.drivers)) {
    for (const row of driver.laps) {
      if (row.generated || row.neutralised === null) continue;
      if (!laps.has(row.lap)) laps.set(row.lap, row.neutralised);
    }
  }
  return laps;
}

/**
 * The neutralised laps as CONTIGUOUS ranges, which is what a chart shades.
 *
 * Twenty-two separate one-lap bands over a 57-lap axis is visual noise; the same
 * laps as three bands is a caption. Melbourne 2025 collapses to exactly three:
 * 1-7, 33-41 and 46-51.
 */
export function neutralisedRanges(laps: Map<number, string>): { from: number; to: number; label: string }[] {
  const sorted = [...laps.keys()].sort((left, right) => left - right);
  const ranges: { from: number; to: number; label: string }[] = [];
  for (const lap of sorted) {
    const last = ranges[ranges.length - 1];
    if (last && lap === last.to + 1) last.to = lap;
    else ranges.push({ from: lap, to: lap, label: laps.get(lap) as string });
  }
  return ranges;
}
