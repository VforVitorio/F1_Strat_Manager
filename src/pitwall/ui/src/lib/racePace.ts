/**
 * The race-pace grid: one cell per driver per lap, coloured by how that lap
 * ranked AGAINST THE SAME LAP.
 *
 * This is the RaceX client's Run Timeline, the panel the research found
 * occupies more screen area than anything else on a real wall.
 *
 * **The colour is a RANK within the lap, not a percentage off a session best,
 * and the difference is the whole design.** Measured on the real Melbourne
 * 2025 payload: against the session's fastest lap the median lap is +13.79 %
 * and 82.4 % of the race falls past +10 %, because the race was wet and ran
 * safety cars - so any fixed percentage band paints four fifths of the grid in
 * one colour and says nothing. Ranked inside its own lap instead, the field
 * splits into thirds by construction on every lap of every race, wet or dry,
 * green or neutralised, with no threshold tuned on the one race that happens
 * to be on this disk. It is also what "race pace" means to a strategist: who
 * is quick RIGHT NOW, not who set the best lap two hours ago.
 *
 * The absolute time is in the cell, so the colour never has to carry a value
 * the reader cannot check.
 *
 * One cell is purple: the session's outright fastest revealed lap. That is the
 * same rule the tower's sector cells use, and it comes from the same module
 * (`sessionBests`) rather than a second reduction over `bulk.drivers` - two
 * panels disagreeing about which lap was quickest is the twin this repo pays
 * for most often.
 */

import type { Bulk, LapRow } from "./bridge";
import { sessionBests } from "./sessionBests";

/**
 * What a cell paints. `best` is the session's fastest; `t1`/`t2`/`t3` are the
 * thirds of the lap's own ranking; `pit` and `out` are the in-lap and out-lap;
 * `none` is a lap the driver has no time for, revealed or not.
 */
export type PaceTone = "best" | "t1" | "t2" | "t3" | "pit" | "out" | "none";

export interface PaceCell {
  text: string;
  tone: PaceTone;
}

export interface PaceGrid {
  /** Lap numbers, ascending. Empty until something is revealed. */
  laps: number[];
  /** Driver codes in WIRE order, which is the only order that holds mid-lap. */
  columns: string[];
  /** `rows[i][j]` is lap `laps[i]` for driver `columns[j]`. */
  rows: PaceCell[][];
}

const EMPTY: PaceCell = { text: "", tone: "none" };

/**
 * The column order: every driver the wire names, sorted by CAR NUMBER.
 *
 * **Which drivers comes from the wire, the order does not, and the two halves
 * have different reasons.** The wire's `race_order` is the only list that
 * carries all twenty - three of Melbourne's crashed on lap 1 and have nothing
 * but generated rows, so a grid keyed on the bulk renders seventeen columns
 * and loses them silently. But `race_order` is ranked by POSITION, and a
 * position-ranked grid re-sorts its own columns every time two cars swap: the
 * reader looks back at lap 12 and it is in a different place than it was a
 * moment ago. A car number never changes for the whole race, which is what a
 * history panel needs and what a timing sheet has always used.
 *
 * A driver with no number keeps his wire position, so an unknown never
 * collapses several columns onto one sort key.
 */
function stableColumns(bulk: Bulk, order: string[]): string[] {
  const keyed = order.map((code, index) => {
    const raw = bulk.drivers[code]?.number;
    const number = raw === null || raw === undefined ? null : Number.parseInt(raw, 10);
    return { code, index, number: Number.isNaN(number as number) ? null : number };
  });
  keyed.sort((left, right) => {
    if (left.number === null || right.number === null) return left.index - right.index;
    return left.number - right.number;
  });
  return keyed.map((entry) => entry.code);
}

/**
 * `m:ss.d` - the broadcast form, truncated to tenths.
 *
 * **The truncation is what makes the grid fit at all.** Measured against the
 * real built stylesheet at the right column's real width: twenty columns of
 * 38.75 px hold this form with room to spare, while the same cell in seconds
 * (`149.413`) clips 793 of 1140 cells the moment a cell carries one pixel of
 * padding. Tenths is also the resolution the grid needs, because the ranking
 * colour carries the ordering and the number carries the magnitude.
 */
export function paceLabel(seconds: number): string {
  const minutes = Math.floor(seconds / 60);
  const rest = seconds - minutes * 60;
  return `${minutes}:${rest.toFixed(1).padStart(4, "0")}`;
}

/**
 * The lap a driver has for that number, or null.
 *
 * A `generated` row is FastF1's synthetic stand-in for a car that did not
 * finish the lap. It is rendered as nothing and counts towards nothing - its
 * `Time` stamp sorts before the entire field, which is how a naive ranking
 * puts the lap-1 crashers on the podium.
 */
function timedRow(row: LapRow | undefined): LapRow | null {
  if (row === undefined || row.generated) return null;
  return row;
}

/** In and out laps are not racing laps and never enter a ranking. */
function isRacingLap(row: LapRow): boolean {
  return !row.pit_in && !row.pit_out && !row.deleted && row.lap_time !== null;
}

/**
 * Each lap's ranked racing times, so a cell can ask where it placed.
 *
 * Built once for the whole grid rather than per cell: a per-cell scan is
 * O(drivers^2) per lap and this runs on every reveal.
 */
function rankedByLap(byDriver: Map<string, Map<number, LapRow>>, laps: number[]) {
  const ranked = new Map<number, number[]>();
  for (const lap of laps) {
    const times: number[] = [];
    for (const rows of byDriver.values()) {
      const row = timedRow(rows.get(lap));
      if (row && isRacingLap(row)) times.push(row.lap_time as number);
    }
    times.sort((left, right) => left - right);
    ranked.set(lap, times);
  }
  return ranked;
}

/**
 * Which third of the lap this time placed in.
 *
 * Ceil rather than floor on the boundaries, so a field of four splits 2/1/1
 * instead of leaving the last third empty. A lap with fewer than three timed
 * cars has no meaningful thirds and everything in it reads as the top one -
 * which is honest: with two cars running, both ARE the field.
 */
function tone(times: number[], value: number): PaceTone {
  const index = times.indexOf(value);
  if (index < 0 || times.length < 3) return "t1";
  const third = times.length / 3;
  if (index < third) return "t1";
  if (index < third * 2) return "t2";
  return "t3";
}

/**
 * The grid for what the clock has revealed.
 *
 * **Columns come from the wire's `order`, never from the bulk's keys.** Three
 * of Melbourne's twenty drivers crashed on lap 1 and have only `generated`
 * rows, so a grid keyed on the bulk renders seventeen columns and silently
 * loses them. The wire always carries twenty.
 *
 * The bottom edge is RAGGED on purpose: the reveal is per driver and strict,
 * and at most instants the field spans two or three different laps.
 */
export function racePaceGrid(bulk: Bulk | null, order: string[]): PaceGrid {
  if (!bulk?.available) return { laps: [], columns: [...order], rows: [] };
  const columns = stableColumns(bulk, order);

  const byDriver = new Map<string, Map<number, LapRow>>();
  let lastLap = 0;
  for (const code of columns) {
    const rows = new Map<number, LapRow>();
    for (const row of bulk.drivers[code]?.laps ?? []) {
      rows.set(row.lap, row);
      if (!row.generated && row.lap > lastLap) lastLap = row.lap;
    }
    byDriver.set(code, rows);
  }

  const laps = Array.from({ length: lastLap }, (_, index) => index + 1);
  const ranked = rankedByLap(byDriver, laps);
  const fastest = sessionBests(bulk).lap_time[0] ?? null;

  const rows = laps.map((lap) =>
    columns.map((code) => {
      const row = timedRow(byDriver.get(code)?.get(lap));
      if (row === null) return EMPTY;
      if (row.pit_in) return { text: "IN PIT", tone: "pit" as PaceTone };
      if (row.pit_out) return { text: "OUT", tone: "out" as PaceTone };
      if (row.lap_time === null) return EMPTY;
      const text = paceLabel(row.lap_time);
      // The session's fastest lap is purple wherever it appears, matching the
      // tower's sector code. Compared on the VALUE and the driver together:
      // two drivers can set the identical time to the millisecond, and only
      // one of them set the session's best.
      if (fastest && fastest.code === code && fastest.value === row.lap_time) {
        return { text, tone: "best" as PaceTone };
      }
      return { text, tone: tone(ranked.get(lap) ?? [], row.lap_time) };
    }),
  );
  return { laps, columns, rows };
}
