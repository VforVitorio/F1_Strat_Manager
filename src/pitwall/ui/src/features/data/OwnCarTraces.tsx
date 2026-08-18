/**
 * Band 4: the own car's lap, as four locked-axis traces against distance.
 *
 * A 1:1 port of the retired Qt dashboard's `telemetry_panel.py`. Every number in
 * this file - the four Y ranges, the fallback X range, the colour each
 * metric draws in - is read out of that module rather than chosen here, and
 * the acceptance reference is the capture at
 * `documents/dev_docs/migration/pitwall/legacy-qt-telemetry.png`.
 *
 *     +---------------------------+---------------------------+
 *     | Delta Time  (rival - main)| Speed                km/h |
 *     +---------------------------+---------------------------+
 *     | Brake Pressure          % | Throttle                % |
 *     +---------------------------+---------------------------+
 *
 * What band 4 adds over the Qt original, and nothing else: a shared vertical
 * cursor at the car's current point on the lap, and the BROADCAST tier label
 * on the rival's legend.
 *
 * **Gear and DRS are deliberately absent.** They ride on the wire and they
 * carry real values, but the window being ported charts neither, and `drs`
 * arrives as the raw FastF1 code whose open set `{10, 12, 14}` exists only
 * in `src/arcade/track.py:40`. Charting it here would fork that constant
 * across two languages, which is the defect `driver_colors` is on the wire
 * to prevent. If the charts are wanted the producer publishes a decoded
 * `drs_open` first.
 */

import { useMemo, useRef } from "react";
import type { Tick } from "../../lib/bridge";
import type { Discontinuity } from "../../lib/frameClock";
import { TraceChart } from "./TraceChart";
import { TraceAccumulator, deltaSeries, type SortedTrace } from "./traceBuffer";

// --- Everything below is `telemetry_panel.py`, transcribed --------------

/** `_SPEED_Y_RANGE`: Monza's straight tops out around 357 km/h. */
const SPEED_Y: [number, number] = [0, 360];
/**
 * `_BRAKE_Y_RANGE`: padded so a trace at 0 or 100 does not kiss the frame.
 *
 * Separate from the throttle range even though the two hold the same pair.
 * `telemetry_panel.py` declares them as two constants, and merging rules that
 * agree by coincidence is a defect class this repo has already paid for: a
 * compound-suitability floor and a minimum-stint bound shared a 12, so
 * recalibrating one silently rewrote the other.
 */
const BRAKE_Y: [number, number] = [-5, 105];
/** `_THROTTLE_Y_RANGE`. Equal to the brake range today, and its own rule. */
const THROTTLE_Y: [number, number] = [-5, 105];
/** `_DELTA_Y_RANGE`: generous for one lap, and it clips when the series wanders. */
const DELTA_Y: [number, number] = [-3, 3];
/** `_DEFAULT_X_RANGE`, used until a broadcast carries a real circuit length. */
const FALLBACK_X_MAX = 5500;
/** Below this a `circuit_length_m` is not a circuit; `update_from`'s own guard. */
const MIN_CREDIBLE_CIRCUIT_M = 100;

/**
 * The colour each metric's own-car trace draws in, and the one the rival
 * always draws in. SLOTS, not a palette membership: swapping brake from
 * DANGER to WARNING would keep every hex inside the palette and still be
 * wrong, so `test_pitwall_tokens.py` pins each name to its palette constant.
 */
const TRACE_COLOURS = {
  /** palette.INFO */
  delta_main: "#3b82f6",
  /** palette.INFO */
  speed_main: "#3b82f6",
  /** palette.DANGER */
  brake_main: "#ef4444",
  /** palette.SUCCESS */
  throttle_main: "#10b981",
  /** palette.WARNING - the rival, on all four charts */
  rival: "#f59e0b",
} as const;

interface OwnCarTracesProps {
  tick: Tick;
  discontinuity: Discontinuity;
  /** The producer is gone; these buffers will not fill. */
  frozen?: boolean;
}

export function OwnCarTraces({ tick, discontinuity, frozen = false }: OwnCarTracesProps) {
  const accumulator = useRef(new TraceAccumulator());
  const { arcade } = tick;
  const rivalCode = arcade.driver_rival;

  // Keyed on the sequence, so one tick is folded in once per tick even
  // though StrictMode runs this factory twice - `ingest` is idempotent for
  // exactly that reason. The X lock is derived here too: `circuit_length_m`
  // rides on every tick, so unlike Qt there is no first-time flag to hold.
  const frame = useMemo(() => {
    const { telemetry } = arcade;
    // The producer's own eviction signals, plus the clock's - `FrameClock`
    // sees a backwards jump smaller than one broadcast that the flags miss.
    const evict =
      telemetry.rewound || telemetry.dropped > 0 || discontinuity.kind !== "continuous";
    accumulator.current.ingest(telemetry.main, telemetry.rival, evict);
    const main = accumulator.current.mainTrace;
    const rival = accumulator.current.rivalTrace;
    return { main, rival, delta: deltaSeries(main, rival) };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [tick.seq]);

  const xMax =
    arcade.circuit_length_m > MIN_CREDIBLE_CIRCUIT_M ? arcade.circuit_length_m : FALLBACK_X_MAX;

  // The cursor comes from the DRIVERS block, never from the tail of the
  // span. The two agree to 0.310 m at worst over a whole race - they read
  // the same frame and differ only in rounding - but the span is EMPTY on
  // a paused tick and on the tick after a rewind, while the drivers block
  // is published unconditionally. Only one of the two can answer "where is
  // the car" at every instant.
  const ownCar = arcade.drivers[arcade.driver_main];
  const cursorX = ownCar?.rel_dist == null ? null : ownCar.rel_dist * xMax;

  /**
   * A frozen board whose traces never filled says so, instead of showing four
   * empty plots.
   *
   * A window OPENED onto a dead feed is the case: the tower, the bests, the ring
   * and the radio all populate from the host's last payload, because those are a
   * per-lap reveal, while these four accumulate PER TICK and only one tick was
   * ever served. Every other empty panel on this window explains itself -
   * `data-waiting`, `trace-band-empty`, the radio's `no corpus`, this chart's own
   * `single-driver mode` - so four silent axes were the odd one out.
   *
   * Two samples, not one: one point draws nothing a reader can see either.
   */
  const starved = (points: unknown[]) => (frozen && points.length < 2 ? "no telemetry since the feed stopped" : null);

  return (
    <section className="traces card">
      <TracesHeader tick={tick} />
      <div className="traces-grid">
        <TraceChart
          title="Δ Time (s)"
          subtitle="(rival − main)"
          mainColour={TRACE_COLOURS.delta_main}
          rivalColour={TRACE_COLOURS.rival}
          yRange={DELTA_Y}
          xMax={xMax}
          rivalCode={rivalCode}
          main={[]}
          rival={frame.delta}
          mainAsZeroLine
          cursorX={cursorX}
          // Two-driver mode is a property of the SESSION, not of whether
          // this tick happened to carry a rival sample. Keyed on the buffer
          // instead, the chart collapsed to its placeholder for the whole
          // of a rewind hold and every lap change.
          // **The session property wins here, and the starved caption only applies
          // when there IS a rival.** `deltaSeries` is empty BY CONSTRUCTION in
          // single-driver mode, so letting the frozen caption take precedence put
          // "no telemetry since the feed stopped" beside three charts showing full
          // traces of exactly the telemetry that sentence denies - a true state
          // with a false cause. Measured: rival code and rival telemetry stripped,
          // the caption flipped on the producer's death.
          placeholder={rivalCode ? starved(frame.delta) : "single-driver mode"}
        />
        <TraceChart
          title="Speed"
          subtitle="km/h"
          mainColour={TRACE_COLOURS.speed_main}
          rivalColour={TRACE_COLOURS.rival}
          yRange={SPEED_Y}
          xMax={xMax}
          rivalCode={rivalCode}
          main={channel(frame.main, "speed")}
          rival={channel(frame.rival, "speed")}
          cursorX={cursorX}
          placeholder={starved(frame.main.xs)}
        />
        <TraceChart
          title="Brake Pressure"
          subtitle="%"
          mainColour={TRACE_COLOURS.brake_main}
          rivalColour={TRACE_COLOURS.rival}
          yRange={BRAKE_Y}
          xMax={xMax}
          rivalCode={rivalCode}
          main={channel(frame.main, "brake")}
          rival={channel(frame.rival, "brake")}
          cursorX={cursorX}
          placeholder={starved(frame.main.xs)}
        />
        <TraceChart
          title="Throttle"
          subtitle="%"
          mainColour={TRACE_COLOURS.throttle_main}
          rivalColour={TRACE_COLOURS.rival}
          yRange={THROTTLE_Y}
          xMax={xMax}
          rivalCode={rivalCode}
          main={channel(frame.main, "throttle")}
          rival={channel(frame.rival, "throttle")}
          cursorX={cursorX}
          placeholder={starved(frame.main.xs)}
        />
      </div>
    </section>
  );
}

/** `LAP 24  NOR vs PIA`, plus the note for a car the telemetry never placed. */
function TracesHeader({ tick }: { tick: Tick }) {
  const { arcade } = tick;
  const rivalCode = arcade.driver_rival;
  // BOTH charted drivers, not just the main one. Reading the main alone left
  // a header saying nothing was wrong above an empty rival trace when the
  // blind car was the one being compared against (#856).
  const blind = [arcade.driver_main, rivalCode].filter(
    (code): code is string => !!code && arcade.drivers[code]?.has_position === false,
  );

  return (
    <header className="traces-header">
      <span className="traces-lap">
        {arcade.lap ? `LAP ${arcade.lap}` : "LAP —"}
        {blind.length ? `  ·  NO POSITION DATA (${blind.join(", ")})` : ""}
      </span>
      <span className="driver-chip driver-chip-main">{arcade.driver_main || "—"}</span>
      {rivalCode ? (
        <>
          <span className="traces-vs">vs</span>
          {/* The BROADCAST tag rides on the rival's chip now that the per-cell
           * legends are gone. Rival car data is real and public, but it is the
           * coarse low-rate channel every team sees rather than pit-wall-grade
           * telemetry, and the Qt window rendered it unlabelled. */}
          <span className="driver-chip driver-chip-rival" title="broadcast tier">
            {rivalCode} <span className="trace-tier">BROADCAST</span>
          </span>
        </>
      ) : null}
    </header>
  );
}

function channel(trace: SortedTrace, key: "speed" | "throttle" | "brake"): [number, number][] {
  return trace.xs.map((x, index) => [x, trace.rows[index][key]]);
}
