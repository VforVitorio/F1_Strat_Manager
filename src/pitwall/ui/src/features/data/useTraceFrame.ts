/**
 * The own-car trace buffers, owned ABOVE the tab that renders them.
 *
 * **This exists because of where it is called, not because of what it does.**
 * The accumulator used to be a `useRef` inside `OwnCarTraces`, and `DataWindow`
 * renders that component conditionally (`{tab === "traces" && ...}`), so leaving
 * the tab UNMOUNTED it and destroyed the buffer. Coming back inside the same lap
 * built a fresh empty one and the six panels restarted from whatever distance the
 * car happened to be at, with the rest of the lap gone.
 *
 * Nothing could rebuild it: the wire carries only the span since the last tick
 * (`_telemetry_span_bounds`), there is no history on it, and there is no
 * addressable telemetry on disk. The buffer is the only copy.
 *
 * So the hook is called from `DataWindow`, which is mounted for every tick
 * whichever tab is showing, and ingestion continues while the traces tab is away.
 *
 * Not solved by keeping the panel mounted and hiding it with CSS: the tab strip
 * exists precisely because the column cannot afford both worlds at once (with the
 * ring still mounted its cells clip 1,101 of 1,140), and an ECharts instance
 * re-rendering off-screen at 10 Hz is the cost the tabs were introduced to avoid.
 */

import { useMemo, useRef } from "react";
import type { Tick } from "../../lib/bridge";
import type { Discontinuity } from "../../lib/frameClock";
import { TraceAccumulator, deltaSeries, type SortedTrace } from "./traceBuffer";

export interface TraceFrame {
  main: SortedTrace;
  rival: SortedTrace;
  delta: ReturnType<typeof deltaSeries>;
}

export function useTraceFrame(
  tick: Tick | null,
  discontinuity: Discontinuity,
  /**
   * The car to compare against, ALREADY RESOLVED by the caller.
   *
   * Not `tick.arcade.driver_rival` read here. This hook is one of the four sites
   * that choose the rival, and the pin has to reach every one of them or the
   * window shows two: a version that kept reading the tick here would have left
   * the header, the chip and the ring following the pin while the CHART kept
   * plotting the broadcast rival (#1051).
   */
  rivalCode: string | null,
): TraceFrame | null {
  const accumulator = useRef(new TraceAccumulator());

  // Keyed on the sequence AND the pinned code. The sequence folds each tick in
  // once even though StrictMode runs this factory twice (`ingest` is idempotent
  // for exactly that reason, including the evict path: a clear plus a re-fold of
  // the same spans reproduces the state). The code is here for the SELECTION,
  // which is the half that changes without a new tick.
  return useMemo(() => {
    if (!tick) return null;
    const { telemetry, driver_main } = tick.arcade;
    // The producer's own eviction signals, plus the clock's - `FrameClock` sees a
    // backwards jump smaller than one broadcast that the flags miss.
    const evict =
      telemetry.rewound || telemetry.dropped > 0 || discontinuity.kind !== "continuous";
    // Every driver on the wire is accumulated; the comparison is a LOOKUP at
    // render (#1050). Selecting at ingest was safe only while the producer could
    // not switch rivals, and the tower's pin is exactly that switch.
    accumulator.current.ingest(telemetry.drivers, driver_main, evict);
    const main = accumulator.current.trace(driver_main);
    // Null in single-driver mode, which is a real state and not padding: the
    // rival trace is meant to be empty there, and `TraceStack` renders its own
    // placeholder for it.
    const rival = accumulator.current.trace(rivalCode);
    return { main, rival, delta: deltaSeries(main, rival) };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [tick?.seq, rivalCode]);
}
