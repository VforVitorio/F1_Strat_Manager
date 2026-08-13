/**
 * The interval between two cars, as a timing tower renders it.
 *
 * A port of `DriverInfoPanel._gap_label` (`src/arcade/overlays.py`), and it
 * is ONE exported function for the reason `driverStatus.ts` is: the tower's
 * GAP column and its INT column are the second and third callers of a rule
 * whose four branches were written once, and two inline copies is this
 * repo's dominant defect - one gets fixed and its twin does not. The Qt
 * panel's own docstring records what the missing branches cost: without the
 * OUT branch it rendered "an interval, up to 22 minutes old, naming a car
 * that stopped".
 *
 * **Two clocks, and this one deliberately picks the other one.** The arcade
 * measures intervals from crossings it detects by interpolating a step
 * function; the parquet's `Time` is FastF1's timing table. Measured over all
 * 7,018 at-line pairs of Melbourne 2025 they differ by a median 17 ms
 * (p95 105 ms, worst 568 ms), which is enough to change 83 % of the strings
 * a tower prints at two decimals, and one pair is sign-inverted outright.
 * `gaps.py` concedes the gap: its crossings sit a median 22 ms from the
 * parquet's and 9.8 % of them land more than a frame away. So the SECONDS
 * come from the bulk (the official clock) while the ORDER, the status and
 * the laps-down all keep coming from the wire, which is the only side that
 * can answer them mid-lap.
 *
 * The accepted cost, so nobody files it as a bug while both windows are on
 * screen: PITWALL's hundredths differ from the arcade's on about 83 % of
 * pairs by at most a tenth, and on 0.1 % of labels PITWALL shows a dash
 * where the arcade shows a number. PITWALL is the one that matches official
 * timing.
 */

import type { ArcadeState, Bulk } from "./bridge";
import { driverStatus } from "./driverStatus";

/**
 * One frame of the replay, and the tolerance on an inverted interval.
 *
 * `interval_at_line` reports a difference within one frame period of zero as
 * 0.0 rather than as an inversion, because the crossing itself is only
 * resolved to a frame and a sub-frame negative is noise. Anything more
 * negative than this means the pair was passed in the wrong order, and the
 * answer is a dash - **never a clamp to zero**, because zero is a legitimate
 * interval (two cars level) and clamping turns a detectable inversion into a
 * plausible reading. Measured in the arcade: under such a clamp an inverted
 * call returned `0.000` on five of six sampled laps while the truth was 1 to
 * 25 seconds the other way.
 */
const FRAME_SECONDS = 1 / 25;

export type GapCell =
  /** This car leads; there is nothing in front to measure against. */
  | { kind: "leader" }
  /** One of the two cars has stopped. Any interval to it is frozen and stale. */
  | { kind: "out" }
  /** More than a full lap of track apart. */
  | { kind: "laps"; laps: number }
  /** Seconds, measured at the line ending `lap`. */
  | { kind: "secs"; seconds: number; lap: number }
  /** An input is unknown. Never a plausible-looking substitute. */
  | { kind: "na" };

/**
 * Whole laps of track between two cars, or null when the answer does not exist.
 *
 * Positional, from the wire's `progress`, exactly as `RaceGapCalculator.
 * laps_down` is - and deliberately NOT a difference of lap numbers, which
 * differs by one for the whole window between the two cars crossing the
 * line, which is not being lapped. Null rather than 0 when either progress
 * is unknown or the pair is inverted, because 0 legitimately means "same
 * lap".
 */
function lapsDown(frontProgress: number | null, backProgress: number | null): number | null {
  if (frontProgress === null || backProgress === null) return null;
  const difference = frontProgress - backProgress;
  if (difference < 0) return null;
  return Math.trunc(difference);
}

/**
 * The interval between the car in front and the car behind it.
 *
 * Four outcomes, decided in this order - the order is the contract, not the
 * arithmetic:
 *
 * 1. **OUT** when either car has stopped. `np.interp` clamps past a driver's
 *    last sample, so a parked car keeps reporting its final state forever
 *    and every later branch would happily measure against it.
 * 2. **+N LAP(S)** when the car in front is more than a full lap of track
 *    ahead, the way a timing screen shows it. Without this a lapped pair
 *    reads as tens of seconds, as if they were racing.
 * 3. **seconds at the line** ending the last lap BOTH cars have completed.
 * 4. **N/A** when any input is unknown.
 *
 * The lap is taken from the bulk's own `laps_revealed` rather than from the
 * tick's `laps_completed`, and that is not a shortcut. The two agree, but
 * the bulk's crossings only exist for the laps the bulk has revealed, so
 * asking for a lap the tick knows about and the bulk has not served yet
 * would blink the whole column to dashes once per lap. Taking both from the
 * same payload makes the pair self-consistent by construction.
 */
export function gapCell(
  frontCode: string,
  backCode: string,
  arcade: ArcadeState,
  bulk: Bulk | null,
): GapCell {
  const front = arcade.drivers[frontCode];
  const back = arcade.drivers[backCode];
  if (!front || !back) return { kind: "na" };

  if (driverStatus(front) === "out" || driverStatus(back) === "out") return { kind: "out" };

  const laps = lapsDown(front.progress, back.progress);
  if (laps === null) return { kind: "na" };
  if (laps >= 1) return { kind: "laps", laps };

  const frontLaps = bulk?.drivers[frontCode];
  const backLaps = bulk?.drivers[backCode];
  if (!frontLaps || !backLaps) return { kind: "na" };

  const lap = Math.min(frontLaps.laps_revealed, backLaps.laps_revealed);
  // No crossing map holds lap 0, so the opening lap reads N/A rather than
  // inventing an interval - which is right: no classification exists there.
  const frontTime = frontLaps.crossings[lap];
  const backTime = backLaps.crossings[lap];
  if (frontTime === undefined || backTime === undefined) return { kind: "na" };

  const seconds = backTime - frontTime;
  if (seconds < -FRAME_SECONDS) return { kind: "na" };
  return { kind: "secs", seconds: Math.max(0, seconds), lap };
}

/**
 * The cell as the tower prints it.
 *
 * **The "(L)" suffix the arcade appends is NOT here, and the omission is
 * deliberate.** In the arcade the label is loose text with no header, so the
 * suffix is the only place it can say the number is quantised to the line;
 * in a table there are forty of these cells and a header, and the header
 * carries it once. The rule the delivery plan sets is that the column says
 * so on screen, and it does - "an unlabelled number on a fidelity surface
 * implies a liveness this one does not have" is satisfied by the column
 * heading, not by repeating three characters forty times.
 */
export function formatGapCell(cell: GapCell): string {
  switch (cell.kind) {
    case "leader":
      return "LEADER";
    case "out":
      return "OUT";
    case "laps":
      return `+${cell.laps} LAP${cell.laps > 1 ? "S" : ""}`;
    case "secs":
      return `+${cell.seconds.toFixed(2)}s`;
    case "na":
      // The same dash every other unknown cell in the tower uses. The arcade
      // prints "N/A" because it has no column of dashes to be consistent with.
      return "—";
  }
}
