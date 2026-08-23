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
 * Ported 1:1, including the two rules that look like bugs and are not:
 *
 * - **Eviction happens BEFORE the append**, not after. The tick that reports
 *   `dropped` also carries a valid post-jump span, and clearing afterwards
 *   threw those samples away - up to 250 of them, ten seconds of trace the
 *   payload had already delivered.
 * - **Rival samples are only kept while the rival is on the main driver's
 *   current lap.** Mixing laps puts samples with unrelated `t` next to each
 *   other in a distance-keyed store and the delta interpolation spikes 4-6 s.
 *   The visible cost is real and inherited: each lap opens with the rival
 *   trace empty for exactly the gap between the two cars (16.76 s on lap 30
 *   of the session measured), and a rival a full lap down never matches at
 *   all.
 */

import type { TelemetrySample } from "../../lib/bridge";

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
 * already populated back to the start of the main driver's current lap, because
 * it was being kept all along, and the lap is the buffer's whole horizon anyway -
 * the wire carries only the span since the last tick and cannot backfill.
 *
 * Cost: twenty spans of `FPS x speed / 10 Hz` samples, so about 400 `Map.set`
 * calls a tick at 8x against 40 before, and 5,000 on a capped forward jump. On
 * integer keys that is sub-millisecond beside the ECharts render it feeds.
 */
export class TraceAccumulator {
  private buffers = new Map<string, Map<number, TraceRow>>();
  private currentLap: number | null = null;

  /** Fold one tick's spans in. Safe to call twice with the same tick. */
  ingest(spans: Record<string, TelemetrySample[]>, mainCode: string, evict: boolean): void {
    if (evict) this.clear();

    // The MAIN driver's span runs first and to completion, then everyone else is
    // compared against the lap it left behind. That ordering is inherited rather
    // than incidental: reading it as "store a sample while its driver is on the
    // current lap" is a different algorithm, interleaved in time, and the two
    // agree only because the lap-change clear wipes the old lap anyway.
    for (const sample of spans[mainCode] ?? []) {
      const lap = Math.trunc(sample.lap || 0);
      if (lap !== this.currentLap) {
        this.currentLap = lap;
        this.buffers.clear();
      }
      store(this.bufferFor(mainCode), sample);
    }

    for (const [code, span] of Object.entries(spans)) {
      if (code === mainCode) continue;
      for (const sample of span) {
        // Same rule the single rival has always been held to: a car on another
        // lap carries an unrelated `t`, and mixing those into one distance-keyed
        // store spikes the delta interpolation by 4-6 s.
        if (Math.trunc(sample.lap || 0) === this.currentLap) store(this.bufferFor(code), sample);
      }
    }
  }

  clear(): void {
    this.buffers.clear();
    this.currentLap = null;
  }

  /** One driver's accumulated lap, or the empty trace for a car with nothing. */
  trace(code: string | null): SortedTrace {
    if (code === null) return EMPTY;
    const buffer = this.buffers.get(code);
    return buffer === undefined ? EMPTY : sorted(buffer);
  }

  get lap(): number | null {
    return this.currentLap;
  }

  private bufferFor(code: string): Map<number, TraceRow> {
    let buffer = this.buffers.get(code);
    if (buffer === undefined) {
      buffer = new Map<number, TraceRow>();
      this.buffers.set(code, buffer);
    }
    return buffer;
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
 * The delta series: `t_rival(x) - t_main(x)` along the main driver's x.
 *
 * F1-broadcast convention - main is the flat reference at y=0, so a POSITIVE
 * trace means the rival is slower at that point on the lap and a negative one
 * means faster. Verified against the producer's own `interval_at_line`: the
 * delta at each lap's first common x agreed to 0.000 s on laps 10, 30 and 50
 * of Melbourne 2025, which is what one shared clock looks like.
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
  return out;
}
