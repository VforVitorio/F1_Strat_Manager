/**
 * The subscription hook: one poll loop per window, sequenced.
 *
 * Pull rather than push, deliberately. The UI never receives faster than it
 * renders, so the cadence is a UI concern rather than a wire concern, and
 * 10 Hz of RPC over an in-process bridge costs nothing.
 */

import { useEffect, useRef, useState } from "react";
import { getTick, whenBridgeReady, type Tick } from "./bridge";
import { FrameClock, type Discontinuity } from "./frameClock";

/** Matches the producer's own ~10 Hz. Polling faster only returns null more often. */
const POLL_INTERVAL_MS = 100;

export interface TickState {
  tick: Tick | null;
  /** What the last accepted tick did to the clock; panels evict on this. */
  discontinuity: Discontinuity;
  /** False until the first tick lands, so a window can say "waiting" honestly. */
  live: boolean;
}

export function useTick(): TickState {
  const [state, setState] = useState<TickState>({
    tick: null,
    discontinuity: { kind: "continuous" },
    live: false,
  });
  // Refs, not state: the poll loop must read the newest sequence without
  // re-subscribing, and a stale closure here would ask for the same tick
  // forever.
  const lastSeq = useRef(-1);
  const clock = useRef(new FrameClock());

  useEffect(() => {
    let cancelled = false;
    let timer: number | undefined;

    const poll = async () => {
      const tick = await getTick(lastSeq.current);
      if (cancelled) return;
      if (tick) {
        lastSeq.current = tick.seq;
        const discontinuity = clock.current.advance(
          { frameIndex: tick.playback.frame_index, lap: tick.arcade.lap },
          tick.arcade.telemetry.rewound,
          tick.arcade.telemetry.dropped,
        );
        setState({ tick, discontinuity, live: true });
      }
      timer = window.setTimeout(poll, POLL_INTERVAL_MS);
    };

    whenBridgeReady().then(() => {
      if (!cancelled) poll();
    });

    return () => {
      cancelled = true;
      if (timer !== undefined) window.clearTimeout(timer);
    };
  }, []);

  return state;
}
