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

import type { Tick } from "../../lib/bridge";
import { driverStatus } from "../../lib/driverStatus";
import { TraceStack } from "./TraceStack";
import type { TraceFrame } from "./useTraceFrame";

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
  /**
   * The accumulated buffers, owned by `DataWindow`.
   *
   * **A prop rather than a `useRef` here, and that is the fix for #1056.** This
   * component is rendered conditionally by the tab strip, so it unmounts when the
   * reader leaves TRACES; owning the buffer meant losing the rest of the lap, with
   * nothing on the wire able to rebuild it.
   */
  frame: TraceFrame;
  /**
   * The car being compared against, resolved by `DataWindow`.
   *
   * A prop rather than `arcade.driver_rival` read here, and the same for the
   * header below. This file held TWO independent reads of that field, so
   * converting one would have left the header saying `vs PIA · BROADCAST`, and
   * computing its NO-POSITION warning against PIA, above a chart plotting the
   * pinned car.
   */
  rival: string | null;
  /**
   * The rival's own team colour, resolved by `DataWindow` alongside its code.
   *
   * Both halves travel together on purpose (#1070): this file holds the chip and
   * the stack, and handing it a code without a colour is how they came to
   * disagree in the first place.
   */
  rivalColour: string;
  /** The producer is gone; these buffers will not fill. */
  frozen?: boolean;
}

export function OwnCarTraces({
  tick,
  frame,
  rival,
  rivalColour,
  frozen = false,
}: OwnCarTracesProps) {
  const { arcade } = tick;

  // The X lock is derived here because `circuit_length_m` rides on every tick, so
  // unlike Qt there is no first-time flag to hold. The BUFFERS are not derived
  // here any more; see the `frame` prop.
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
      <TracesHeader tick={tick} rival={rival} rivalColour={rivalColour} frame={frame} />
      <TraceStack
        main={frame.main}
        rival={frame.rival}
        delta={frame.delta}
        xMax={xMax}
        rivalCode={rival}
        rivalColour={rivalColour}
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

/** `LAP 24  NOR vs PIA`, plus the notes for a rival the chart cannot draw. */
function TracesHeader({
  tick,
  rival: rivalCode,
  rivalColour,
  frame,
}: {
  tick: Tick;
  rival: string | null;
  /** The chip paints in the rival's own team colour; the border follows it. */
  rivalColour: string;
  /** The buffers themselves: every note below is about what the DELTA lane has. */
  frame: TraceFrame;
}) {
  const { arcade } = tick;
  // BOTH charted drivers, not just the main one. Reading the main alone left
  // a header saying nothing was wrong above an empty rival trace when the
  // blind car was the one being compared against (#856).
  const blind = [arcade.driver_main, rivalCode].filter(
    (code): code is string => !!code && arcade.drivers[code]?.has_position === false,
  );

  /**
   * The delta lane has nothing to draw, and the note has to say WHICH nothing.
   *
   * **The predicate is the DELTA, not the rival's sample count**, and the
   * difference is the whole of #1066. While every car was held to the main
   * driver's lap the two were equivalent: no samples meant no series. Now a
   * rival carries hundreds of samples of its OWN lap and can still share no
   * track with the main car, because each buffer only holds from its own
   * crossing forward and the two cars sit at different points of the circuit.
   * Measured on the Melbourne capture that state is 2,564 of 9,936 car-ticks,
   * and a note keyed on `rival.xs.length` fires on none of them: four silent
   * axes under a chip saying the car is being compared.
   *
   * Three causes, three sentences, because a true state with a false cause is
   * the defect this file already pays for twice above:
   *
   * - the rival has STOPPED, so its trace ended whenever it ended;
   * - the MAIN car is blind, so there is no reference to compare against;
   * - neither, so the two simply have no common track yet.
   */
  const rivalState = rivalCode === null ? undefined : arcade.drivers[rivalCode];
  const rivalStopped = rivalState !== undefined && driverStatus(rivalState) !== "running";
  const mainBlind = arcade.drivers[arcade.driver_main]?.has_position === false;
  const noDelta = rivalCode !== null && frame.delta.length < 2 && !blind.includes(rivalCode);

  /**
   * Where the delta lane's zero sits, when it is not the start/finish line.
   *
   * The series is anchored at the first x the two cars share, which IS the line
   * once both buffers have rooted at a crossing. Until then - a window just
   * opened, or just seeked, measured at up to 84.9 s of session time - the
   * anchor is wherever the main buffer happens to start, and the readout then
   * means "lost since 409 m" while reading `Δ TIME s +0.42`. Nothing else on
   * screen carries that 409 m: every other lane starts at the same x, so a
   * partial lap looks identical either way.
   */
  const anchorX = frame.delta.length >= 2 ? frame.delta[0][0] : null;
  const offLine = anchorX !== null && anchorX > 0;

  return (
    <header className="traces-header">
      <span className="traces-lap">
        {arcade.lap ? `LAP ${arcade.lap}` : "LAP —"}
        {blind.length ? `  ·  NO POSITION DATA (${blind.join(", ")})` : ""}
        {offLine ? `  ·  Δ FROM ${Math.round(anchorX)} m` : ""}
        {noDelta && rivalStopped ? `  ·  ${rivalCode} STOPPED, NO DELTA` : ""}
        {noDelta && !rivalStopped && mainBlind ? "  ·  NO DELTA WITHOUT THE OWN CAR" : ""}
        {noDelta && !rivalStopped && !mainBlind
          ? `  ·  NO TRACK IN COMMON WITH ${rivalCode} YET`
          : ""}
      </span>
      <span className="driver-chip driver-chip-main">{arcade.driver_main || "—"}</span>
      {rivalCode ? (
        <>
          <span className="traces-vs">vs</span>
          {/* The BROADCAST tag rides on the rival's chip now that the per-cell
           * legends are gone. Rival car data is real and public, but it is the
           * coarse low-rate channel every team sees rather than pit-wall-grade
           * telemetry, and the Qt window rendered it unlabelled. */}
          {/* The rival's OWN lap, whenever it differs from the main car's. The
           * overlay compares two laps rather than two instants (#1066), so which
           * two has to be on screen: `NOR vs GAS L23` under `LAP 24` is the whole
           * statement, and without it a reader has no way to know the two traces
           * are not simultaneous. Both numbers come from the accumulator, never
           * one from it and one from `arcade.drivers`. */}
          {/* `color` inline, because the value is per-driver and a stylesheet
           * cannot know it. `data.css` gives this chip a `currentColor` border,
           * so the border follows the team colour for free. */}
          <span
            className="driver-chip driver-chip-rival"
            style={{ color: rivalColour }}
            title="broadcast tier"
          >
            {rivalCode}
            {frame.rivalLap !== null && frame.rivalLap !== frame.mainLap ? (
              <span className="trace-rival-lap"> L{frame.rivalLap}</span>
            ) : null}{" "}
            <span className="trace-tier">BROADCAST</span>
          </span>
        </>
      ) : null}
    </header>
  );
}

