/**
 * The per-lap sample store behind band 4, ported from `TelemetryPanel`.
 *
 * The wire sends a SPAN per tick - every frame the replay clock crossed
 * since the last broadcast, oldest first - so this panel is one of only two
 * in the DATA window that accumulate (`PITWALL_V2_ARCHITECTURE.md` 3.4). It
 * is a keyed map rather than an array for exactly the reason that section
 * gives: below 1x, and permanently while paused, the same samples arrive
 * more than once, and a map makes a duplicate tick a no-op instead of a
 * doubled trace.
 *
 * That idempotence is load-bearing beyond duplicate ticks: React StrictMode
 * invokes a `useMemo` factory twice in development, so `ingest` runs twice
 * on the same span. Keys are integer metres, the lap-change clear lands on
 * the same sample both times, and the second pass therefore reproduces the
 * first exactly.
 *
 * Ported 1:1, including the rule that looks like a bug and is not:
 *
 * - **Eviction happens BEFORE the append**, not after. The tick that reports
 *   `dropped` also carries a valid post-jump span, and clearing afterwards
 *   threw those samples away - up to 250 of them, ten seconds of trace the
 *   payload had already delivered.
 *
 * **The second inherited rule is GONE, and this says so because it was load
 *   bearing for two years.** Qt kept a rival sample only while that car was on
 *   the main driver's current lap, on the argument that mixing laps puts
 *   samples with unrelated `t` in one distance-keyed store. That argument is
 *   the BROADCAST convention - the battle graphic drawn only for two cars
 *   close together - and band 4 is the engineer's overlay, which is lap
 *   against lap everywhere in motorsport. Keeping it cost the whole panel:
 *   once the tower could pin any car (#1051), five of seven pinnable cars drew
 *   nothing at all, because a car not yet across the main driver's line was
 *   never stored. Each car now keeps its OWN lap and the delta subtracts the
 *   two anchors, so the store never holds two laps at once (#1066).
 */

import type { TelemetrySample } from "../../lib/bridge";
import { driverStatus, type StatusInputs } from "../../lib/driverStatus";

/** What a chart needs out of one sample; `dist` is the key, not a field. */
export interface TraceRow {
  t: number;
  speed: number;
  throttle: number;
  brake: number;
  /** 1-8 on the real session, all eight values present. A STEP channel, not a curve. */
  gear: number;
  /** Decoded producer-side, because the open code set is not this language's to own. */
  drsOpen: boolean;
}

/** A buffer flattened for plotting: x ascending, rows in the same order. */
export interface SortedTrace {
  xs: number[];
  rows: TraceRow[];
}

const EMPTY: SortedTrace = { xs: [], rows: [] };

/** One driver's current lap and the samples it has covered of it. */
interface LapBuffer {
  lap: number;
  rows: Map<number, TraceRow>;
}

/**
 * One buffer per DRIVER, with the comparison chosen at render (#1050).
 *
 * It used to key two buffers by ROLE - `main` and `rival` - and that was safe
 * only because the producer cannot change its mind: `_driver_rival` is assigned
 * once, at construction (`src/arcade/app.py:236`). The tower's pin introduces the
 * mid-lap switch, and re-pointing a role-keyed buffer leaves the old car's
 * samples in the same distance-keyed map: `deltaSeries` then interpolates across
 * the seam and draws a delta against a car that is half one driver and half
 * another.
 *
 * Accumulating every driver removes the question. The newly pinned car's trace is
 * already populated back to the start of ITS OWN lap, because it was being kept
 * all along, and one lap is the buffer's whole horizon anyway - the wire carries
 * only the span since the last tick and cannot backfill.
 *
 * Each buffer also owns its own LAP NUMBER (#1066), which is the half #1050 left
 * on the main driver: accumulating every car while letting one car's crossing
 * wipe every buffer meant a car not yet across that line was stored and then
 * thrown away, so nine of nineteen drew nothing on the tick measured.
 *
 * Cost: twenty spans of `FPS x speed / 10 Hz` samples, so about 400 `Map.set`
 * calls a tick at 8x against 40 before, and 5,000 on a capped forward jump. On
 * integer keys that is sub-millisecond beside the ECharts render it feeds.
 */
export class TraceAccumulator {
  private entries = new Map<string, LapBuffer>();

  /**
   * Fold one tick's spans in. Safe to call twice with the same tick.
   *
   * `drivers` is the tick's own state block, and it is here for ONE reason: a
   * car that has stopped must stop being stored. The producer republishes a
   * retired car's last frame every tick with an advancing `t`, so a buffer that
   * keeps accepting it rewrites its last distance key with the current session
   * clock forever - measured on Melbourne 2025, ALO's last point drifts +2,785 s
   * between its lap-33 retirement and the flag, and the readout that reads it
   * counts up one second per second. The old shared lap number wiped every
   * buffer at each of the main car's crossings and hid this; per-driver laps
   * remove that wipe, so the predicate has to be explicit.
   */
  ingest(
    spans: Record<string, TelemetrySample[]>,
    evict: boolean,
    drivers: Record<string, StatusInputs>,
  ): void {
    if (evict) this.clear();

    // The main driver's code was a parameter here until #1066 and is not one now,
    // which is the change in one line: its lap used to decide what every other car
    // was allowed to store, and that is what forced its span to run first and to
    // completion. Each car now answers only for itself, so there is no ordering
    // between drivers and nothing to privilege.
    for (const [code, span] of Object.entries(spans)) {
      const state = drivers[code];
      // Unknown rather than stopped: a code with a span and no state block is not
      // a case the wire produces (`bridge.ts` pins the key sets equal), and
      // dropping it silently would be the same mistake as trusting it silently.
      if (state !== undefined && driverStatus(state) !== "running") continue;
      for (const sample of span) this.absorb(code, sample);
    }
  }

  clear(): void {
    this.entries.clear();
  }

  /** One driver's accumulated lap, or the empty trace for a car with nothing. */
  trace(code: string | null): SortedTrace {
    if (code === null) return EMPTY;
    const entry = this.entries.get(code);
    return entry === undefined ? EMPTY : sorted(entry.rows);
  }

  /**
   * The lap THIS buffer holds, which is the only honest source for the header.
   *
   * `tick.arcade.drivers[code].lap` is the other candidate and it is a different
   * number: it reads the frame at the tick's own index while this reads the last
   * sample actually stored, and the two disagree across a lap change inside one
   * span and at all 70 of the lap-channel glitches a race carries. One number,
   * one source.
   */
  lapOf(code: string | null): number | null {
    if (code === null) return null;
    return this.entries.get(code)?.lap ?? null;
  }

  /** Store one sample under its driver's own lap, opening a new lap when it turns. */
  private absorb(code: string, sample: TelemetrySample): void {
    const lap = Math.trunc(sample.lap || 0);
    const entry = this.entries.get(code);
    if (entry === undefined) {
      this.entries.set(code, { lap, rows: new Map() });
    } else if (lap > entry.lap) {
      // A new lap replaces the buffer rather than adding to it: one lap is the
      // whole horizon, and the wire cannot backfill what came before.
      this.entries.set(code, { lap, rows: new Map() });
    } else if (lap < entry.lap) {
      // A lap number that goes BACKWARDS is a glitch, not a rewind - a rewind
      // arrives as `rewound` and evicts everything. Measured on Melbourne 2025:
      // 70 of these a race across 17 of the 20 drivers, one frame each, and the
      // glitch frame carries a stale lap with a mid-lap `dist` (HAM reads lap 23
      // at 2586.2 m one frame after crossing into 24). Storing it would put a
      // foreign lap in the map; clearing on it, which `lap !== currentLap` used
      // to do, would throw the lap away twice per crossing. It is a producer-side
      // defect and it reaches every consumer of `rel_dist`, not just this buffer.
      return;
    }
    store(this.entries.get(code)!.rows, sample);
  }
}

/**
 * Key a sample into its integer metre of lap distance.
 *
 * `dist` is null when the driver's telemetry never placed the car (#856).
 * Dropping the sample is the only honest option: 0 is a place a real car can
 * be, so bucketing it there would draw the blind car sitting on the start
 * line rather than nowhere.
 */
function store(buffer: Map<number, TraceRow>, sample: TelemetrySample): void {
  if (sample.dist === null || !Number.isFinite(sample.dist)) return;
  buffer.set(Math.trunc(sample.dist), {
    t: sample.t,
    speed: sample.speed,
    throttle: sample.throttle,
    brake: sample.brake,
    gear: sample.gear,
    drsOpen: sample.drs_open,
  });
}

function sorted(buffer: Map<number, TraceRow>): SortedTrace {
  if (buffer.size === 0) return EMPTY;
  const xs = [...buffer.keys()].sort((a, b) => a - b);
  return { xs, rows: xs.map((x) => buffer.get(x)!) };
}

/**
 * Linear interpolation of `ys` at `x`, given ascending `xs`.
 *
 * Returns null OUTSIDE the known range rather than clamping to the end, so
 * the delta chart plots nothing where the rival has not reached yet instead
 * of a flat extrapolated tail that looks like real data.
 */
export function lerpSorted(xs: number[], ys: number[], x: number): number | null {
  if (xs.length === 0 || x < xs[0] || x > xs[xs.length - 1]) return null;
  let lo = 0;
  let hi = xs.length;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (xs[mid] < x) lo = mid + 1;
    else hi = mid;
  }
  if (lo < xs.length && xs[lo] === x) return ys[lo];
  if (lo === 0) return ys[0];
  const [x0, x1] = [xs[lo - 1], xs[lo]];
  const [y0, y1] = [ys[lo - 1], ys[lo]];
  if (x1 === x0) return y0;
  return y0 + ((y1 - y0) * (x - x0)) / (x1 - x0);
}

/**
 * The delta series along the main driver's x, RE-BASED to lap-relative time.
 *
 * Main is the flat reference at y=0, so a POSITIVE trace means the rival is
 * slower to that point of the lap and a negative one means faster.
 *
 * **What the subtraction at the end buys, and why it is the whole of #1066.**
 * The raw difference `t_rival(x) - t_main(x)` is session time, so it carries the
 * gap between the two cars as a constant offset. That was harmless while the
 * only rival ever charted was one the producer picked to sit two seconds away;
 * the moment the tower could pin anyone, a car ten seconds ahead drew a trace
 * ten seconds off a lane locked to three. Subtracting the value at the first
 * common x removes the offset and leaves the SHAPE, which is the question an
 * engineer's overlay asks: not how far apart the two cars are, but where one is
 * quicker. The gap is answered elsewhere, by the tower's GAP and INT columns.
 *
 * With both buffers rooted at their own line crossings the anchor sits at x = 0
 * and this IS the lap-relative delta. It is exact to within one frame rather
 * than exactly: `store` keys by integer metre, so a second frame landing in the
 * same metre displaces the first, and the anchor moves with it. Measured over
 * the Melbourne capture the divergence is 0.080 s on the one car whose lap
 * channel glitched and 0.000 s on the other fifteen, and it is reachable on any
 * crossing taken below 90 km/h, where 40 ms covers less than a metre: the pit
 * lane, a safety-car queue, a spin.
 *
 * Below two common points there is no series and no anchor. Returning the empty
 * array rather than a one-point one is deliberate: a single point draws nothing
 * but would still feed the lane's readout, and by construction its value is
 * exactly 0.00 - a manufactured number indistinguishable from a genuinely level
 * pair. `delta.length < 2` is what the header tests to explain the blank lane.
 */
export function deltaSeries(main: SortedTrace, rival: SortedTrace): [number, number][] {
  if (main.xs.length < 2 || rival.xs.length < 2) return [];
  const rivalTimes = rival.rows.map((row) => row.t);
  const out: [number, number][] = [];
  main.xs.forEach((x, index) => {
    const interpolated = lerpSorted(rival.xs, rivalTimes, x);
    if (interpolated === null) return;
    out.push([x, interpolated - main.rows[index].t]);
  });
  // Two traces that are each long enough can still share NO track: two cars on
  // their own laps sit at different points of the circuit, and a buffer only
  // holds from its own root forward. Measured at 2,564 of 9,936 car-ticks on the
  // Melbourne capture, every running car among them.
  if (out.length < 2) return [];
  const anchor = out[0][1];
  return out.map(([x, value]) => [x, value - anchor]);
}
