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

export function useTraceFrame(tick: Tick | null, discontinuity: Discontinuity): TraceFrame | null {
  const accumulator = useRef(new TraceAccumulator());

  // Keyed on the sequence, so one tick is folded in once per tick even though
  // StrictMode runs this factory twice - `ingest` is idempotent for exactly that
  // reason.
  return useMemo(() => {
    if (!tick) return null;
    const { telemetry, driver_main, driver_rival } = tick.arcade;
    // The producer's own eviction signals, plus the clock's - `FrameClock` sees a
    // backwards jump smaller than one broadcast that the flags miss.
    const evict =
      telemetry.rewound || telemetry.dropped > 0 || discontinuity.kind !== "continuous";
    // Schema v2 puts a span under every driver code, so the pair this panel
    // charts is a LOOKUP rather than two fixed keys (#1048). `?? []` is not
    // defensive padding: single-driver mode has no rival at all, and the
    // accumulator's rival map is meant to stay empty then.
    //
    // Selecting here rather than accumulating per driver is deliberate for
    // now: switching the pinned car mid-lap would mix two cars' samples in one
    // distance-keyed map, and the fix for that is per-driver buffers with the
    // choice applied at render (#1050), which is R2's subject.
    accumulator.current.ingest(
      telemetry.drivers[driver_main] ?? [],
      driver_rival ? (telemetry.drivers[driver_rival] ?? []) : [],
      evict,
    );
    const main = accumulator.current.mainTrace;
    const rival = accumulator.current.rivalTrace;
    return { main, rival, delta: deltaSeries(main, rival) };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [tick?.seq]);
}
