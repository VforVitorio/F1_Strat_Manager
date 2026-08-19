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
 * **Gear and DRS are here now, and the refusal that kept them out was honoured
 * rather than overruled.** This docstring used to say they were "deliberately
 * absent" because `drs` arrived as the raw FastF1 code whose open set lives in one
 * place, and charting it here would fork that constant across two languages. That
 * was right. So the producer decodes it: `config.DRS_OPEN_CODES` is the one home
 * and the wire carries `drs_open`, the same treatment `track_status_label` gets.
 * `gear` was already on the wire and read by nothing.
 *
 * **And the 2x2 is gone.** The agreed layout drawing assigned the stacked form to
 * this sprint and calls the 2x2 "the wrong shape"; `TraceStack` is that form. What
 * stays here is what the stack does not own: the accumulator, the header, and the
 * frozen/starved question.
 */

import { useMemo, useRef } from "react";
import type { Tick } from "../../lib/bridge";
import type { Discontinuity } from "../../lib/frameClock";
import { TraceStack } from "./TraceStack";
import { TraceAccumulator, deltaSeries } from "./traceBuffer";

// --- What the stack does not own -----------------------------------------
//
// The four locked Y ranges and the per-metric colour slots moved into
// `TraceStack`'s lane table, where they sit next to the two new lanes and are
// pinned there by `test_pitwall_tokens.py`. What stays here is the X lock, because
// it is a property of the SESSION rather than of a lane.

/** `_DEFAULT_X_RANGE`, used until a broadcast carries a real circuit length. */
const FALLBACK_X_MAX = 5500;
/** Below this a `circuit_length_m` is not a circuit; `update_from`'s own guard. */
const MIN_CREDIBLE_CIRCUIT_M = 100;

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


  return (
    <section className="traces card">
      <TracesHeader tick={tick} />
      <TraceStack
        main={frame.main}
        rival={frame.rival}
        delta={frame.delta}
        xMax={xMax}
        rivalCode={rivalCode}
        cursorX={cursorX}
        // ONE caption over the whole stack, where the 2x2 printed the same sentence
        // four times. It claims starvation only when the MAIN trace is starved: a
        // true state with a false cause is the defect this sprint already paid for
        // on the delta chart's single-driver mode.
        placeholder={
          frozen && frame.main.xs.length < 2 ? "no telemetry since the feed stopped" : null
        }
      />
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

