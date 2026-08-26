/**
 * The race trace: accumulated time against a reference, one line per driver.
 *
 * The panel a strategist reads with a RULER held vertically. At any lap the
 * vertical distance between two lines IS the gap between those cars at that
 * moment, so one cut across the chart answers the whole grid's worth of
 * intervals at once - which is the pit-wall question the race-pace grid
 * cannot answer, because a grid of lap times says who was quick and never
 * says who is where.
 *
 * A pit stop is a STEP down of about the stop's own length; an overtake is a
 * crossing; a stint going off is a line bending away from its neighbours.
 * None of those are drawn - they fall out of the arithmetic.
 *
 * **The seconds come from `crossings`, the same clock `gapCell` subtracts to
 * print the tower's GAP and INT columns.** The alternative - summing
 * `lap_time` - drifts, because a lap time is rounded to the millisecond and a
 * race is 57 of them.
 *
 * ⚠️ **That does NOT mean the two panels always print the same number, and the
 * claim that they cannot disagree was FALSE.** What holds, and what was
 * measured over all 7,018 at-line pairs of Melbourne 2025 under all three
 * references, is that the VERTICAL DISTANCE between two lines equals the
 * difference of their crossings - the thing the panel is read with. What does
 * not hold is agreement with the tower's GAP column, because the two answer to
 * different leaders: the tower measures against `race_order[0]`, the wire's
 * classified P1, and this measures against the car that crossed THAT LAP's
 * line first. Melbourne has three at-line lead changes (laps 44, 46, 47), and
 * in the window after VER first crosses line 44 every driver's tower GAP sits
 * a uniform 3.737 s from the trace's end value - VER's lap-43 advantage over
 * NOR, applied to the whole field. Both are right about their own question.
 *
 * ⚠️ **The reveal is per driver and strict, so the newest laps are ragged**,
 * and a reference averaged over a ragged edge moves under the reader: at the
 * instant the leader has revealed lap 27 and the field lap 25, a field average
 * at lap 27 is an average of the two cars in front, which drags every line
 * towards zero and then springs back on the next reveal. The race-pace grid
 * can be ragged because each of its cells is its own claim; a trace cannot,
 * because every point is measured against a shared line. So the trace stops at
 * the last lap the whole classified field has completed - see `lastCommonLap`.
 */

import type { ArcadeState, Bulk } from "./bridge";
import { driverStatus } from "./driverStatus";
import { stableColumns } from "./racePace";

/**
 * What y = 0 means.
 *
 * `leader` is the classic form and the one that shows the race's shape: the
 * car in front at that lap sits on zero and everyone else hangs below it.
 * `field` is the mean of the classified field, which spreads the leaders above
 * the axis and makes a midfield battle legible instead of squashing it against
 * the top. `own` puts OUR car flat on zero, which is the pit wall's own
 * question - everything above is a car that is ahead, everything below is a
 * car that is behind, and a line climbing towards zero is a car coming for us.
 */
export type TraceReference = "leader" | "field" | "own";

export interface TraceLine {
  code: string;
  /** `[lap, seconds ahead of the reference]`, ascending by lap. */
  points: [number, number][];
}

export interface RaceTrace {
  /** The laps plotted: 1..`lastCommonLap`, or empty when there is nothing to plot. */
  laps: number[];
  lines: TraceLine[];
  /** What the zero line is, for the header to say out loud. */
  zero: string;
  /**
   * The race's own length, so the header can say how far behind the bound sits.
   *
   * **This is what stops a frozen cap being silent.** A car that took the
   * chequered flag but lost telemetry mid-race keeps `laps_completed` stuck at
   * the dropout - the plan carries it as OBS-4 - and being classified, he stays
   * in the cap population and pins the whole trace there. Measured shape: a
   * lap-20 dropout freezes the panel at 20 of 57 and nothing on screen says so,
   * because a trace that ends is indistinguishable from a race that ended.
   * Printing the total turns a frozen panel into a visible one.
   */
  total: number;
}

const EMPTY: RaceTrace = { laps: [], lines: [], zero: "", total: 0 };

/**
 * The cars that BOUND the lap axis. Not the cars the reference averages over -
 * see `referenceTimes`, and see the defect below for why they are two lists.
 *
 * ⚠️ **A single list would move the drawn history under the
 * reader.** This filter reads CURRENT status, so the moment a car retires it
 * leaves the set - and if the reference were averaged over the same set, every
 * point of every line, all the way back to lap 1, would be recomputed without
 * him. Executed on the real payload: at LAW's retirement all 45 of NOR's
 * historical points shift, by up to 7.612 s, in FIELD mode. Melbourne alone
 * does that three times.
 *
 * It is the twin of the bound this very module introduced. `lastCommonLap`
 * exists because a reference computed over a population that changes with the
 * REVEAL swings every line; the identical failure on the STATUS axis went
 * unfixed in the same file. The smoke could not see it: its retirees carry no
 * crossings at all, so removing them from an average changes nothing.
 *
 * Two filters, and each one deletes the panel if it is missing.
 *
 * **Still classified - running or finished, never the retired.** The
 * population exists to bound the plot at a lap EVERYONE has, and a car that
 * crashed on lap 1 has completed zero, so including the retired would pin the
 * whole trace at lap 0 on any race with a first-lap incident. Melbourne 2025
 * has three. `driverStatus` rather than `active` alone, for the reason its own
 * module records: at the end of a race nineteen of twenty cars read as
 * inactive, the winner included, because taking the flag stops the broadcast
 * exactly as crashing does.
 *
 * **And named by the BULK, which is the half that is easy to leave out.** The
 * race-pace grid deliberately takes its columns from the wire and renders a
 * blank column for a car the bulk does not name - so a car in `race_order`
 * with no bulk entry is a state this window already has, and one the fixture
 * carries. Bounding a MINIMUM by such a car reads his lap count as zero and
 * caps the whole trace at lap zero: the panel disappears, blaming a reveal
 * that is fine. We have no lap data for him, so he cannot enter an average and
 * cannot bound one either.
 *
 * The cost of that second filter, stated rather than buried: a car the bulk
 * has no entry for is silently absent from the field average. On the real
 * payload the set is empty - `race_order` and `Object.keys(bulk.drivers)` were
 * measured identical at every reveal - so this is a guard against a shape the
 * types allow, not a rate.
 */
function capPopulation(bulk: Bulk, arcade: ArcadeState, codes: string[]): string[] {
  return codes.filter((code) => {
    const car = arcade.drivers[code];
    if (car === undefined || driverStatus(car) === "out") return false;
    return bulk.drivers[code] !== undefined;
  });
}

/** The highest lap this driver has a crossing for, or 0 when he has none. */
function lastCrossing(crossings: Record<number, number>): number {
  let last = 0;
  for (const key of Object.keys(crossings)) {
    const lap = Number(key);
    if (lap > last) last = lap;
  }
  return last;
}

/**
 * The last lap every classified car has completed.
 *
 * This is the trace's whole answer to the ragged reveal edge, and it is a
 * MINIMUM rather than a threshold on purpose: no fraction of the field to tune
 * on the one race that happens to be on this disk, and no lap where the
 * reference is computed over a different population than the lap below it.
 *
 * The cost, stated so nobody files it as a bug with both tabs open: the trace
 * ends one to two laps behind the race-pace grid, because the grid shows every
 * lap ANY car has and this shows every lap ALL of them have. They are
 * answering different questions - the grid "how quick was that lap", the trace
 * "where is everyone" - and the second is only answerable on a lap the whole
 * field has driven.
 *
 * A lapped car bounds it too, and correctly: while he has not completed lap
 * 27, no reference at lap 27 can include him.
 */
function lastCommonLap(bulk: Bulk, codes: string[]): number {
  let common = Infinity;
  for (const code of codes) {
    // Every code here is named by the bulk - `population` guarantees it - so
    // an empty crossing map means "revealed nothing yet", which
    // bounds the trace at zero and renders the empty state.
    //
    // The `?.` is nevertheless not decoration, and the cost asymmetry is why:
    // a bare dereference throws inside a React render, and a throw here takes
    // the WHOLE DATA window down - tower, bests, status strip and all - for an
    // invariant that lives in a different function. Measured: removing the
    // bulk filter from `population` did exactly that, and the harness stopped
    // reporting anything at all rather than reporting the trace. Degrading to
    // "revealed nothing" costs one lap of the plot and keeps the window up.
    common = Math.min(common, lastCrossing(bulk.drivers[code]?.crossings ?? {}));
  }
  return Number.isFinite(common) ? common : 0;
}

/**
 * The zero line, lap by lap.
 *
 * Null at a lap the reference does not exist for, which only `own` can produce
 * - our car may itself be the one that retired - and which drops the lap from
 * every line rather than substituting a plausible number for it. A reference
 * nobody can check is worse than a shorter chart.
 */
function referenceTimes(
  bulk: Bulk,
  laps: number[],
  codes: string[],
  reference: TraceReference,
  own: string,
): Map<number, number> {
  const times = new Map<number, number>();
  for (const lap of laps) {
    if (reference === "own") {
      const value = bulk.drivers[own]?.crossings[lap];
      if (value !== undefined) times.set(lap, value);
      continue;
    }
    const atLap: number[] = [];
    for (const code of codes) {
      const value = bulk.drivers[code]?.crossings[lap];
      if (value !== undefined) atLap.push(value);
    }
    if (atLap.length === 0) continue;
    if (reference === "leader") {
      times.set(lap, Math.min(...atLap));
    } else {
      times.set(lap, atLap.reduce((sum, value) => sum + value, 0) / atLap.length);
    }
  }
  return times;
}

/** The header's description of the zero line, in the reader's words. */
function zeroLabel(reference: TraceReference, own: string): string {
  if (reference === "leader") return "zero is the car leading that lap";
  if (reference === "field") return "zero is the classified field's average";
  return `zero is ${own}`;
}

/**
 * Every driver's accumulated time against the reference.
 *
 * Positive is AHEAD. A car that has taken twelve seconds less than the
 * reference to reach the same lap is twelve seconds up the road, so it plots
 * twelve above the axis - which is the direction the eye already reads a
 * position from, and the direction the gap columns of the tower above it grow.
 *
 * A retired car keeps every lap he actually drove; his line simply ends. That
 * is not a special case in the code and should not become one: his crossings
 * stop, so his points stop.
 *
 * Series order is `stableColumns`', imported rather than re-derived, with our
 * own car moved LAST so it draws on top of the nineteen it has to be picked
 * out from.
 */
export function raceTrace(
  bulk: Bulk | null,
  arcade: ArcadeState,
  reference: TraceReference,
): RaceTrace {
  if (!bulk?.available) return EMPTY;

  const own = arcade.driver_main;
  const ordered = stableColumns(bulk, arcade.race_order);
  const last = lastCommonLap(bulk, capPopulation(bulk, arcade, ordered));
  const total = bulk.race.total_laps;
  if (last < 1) return { ...EMPTY, total, zero: zeroLabel(reference, own) };

  const laps = Array.from({ length: last }, (_, index) => index + 1);
  // EVERY driver the wire names, not the ones still running: a car that
  // retired on lap 30 was part of the race on laps 1-30 and belongs in those
  // laps' reference. `referenceTimes` skips whoever has no crossing at a lap,
  // so the set is exactly "who completed this lap" - which only ever GROWS as
  // the reveal advances, and therefore cannot move a point already drawn.
  const times = referenceTimes(bulk, laps, ordered, reference, own);
  const plotted = laps.filter((lap) => times.has(lap));

  const drawLast = ordered.filter((code) => code !== own);
  if (ordered.includes(own)) drawLast.push(own);

  const lines = drawLast.map((code) => {
    const crossings = bulk.drivers[code]?.crossings ?? {};
    const points: [number, number][] = [];
    for (const lap of plotted) {
      const value = crossings[lap];
      if (value === undefined) continue;
      points.push([lap, (times.get(lap) as number) - value]);
    }
    return { code, points };
  });

  return { laps: plotted, lines, total, zero: zeroLabel(reference, own) };
}
