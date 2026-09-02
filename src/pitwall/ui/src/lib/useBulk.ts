/**
 * The revealed lap table, polled on its own timer.
 *
 * The second data channel (design decision C7): the tick carries the instant,
 * this carries everything the timing table and the bests panel show. It is
 * static parquet, known before lap 1, so the panel is a progressive reveal
 * rather than a stream to accumulate - and the host applies the mask, so what
 * arrives here is already only what the clock has opened.
 *
 * **Compare the revision on inequality, never on "greater than".** A rewind
 * takes laps BACK and LOWERS the revision's content; treating a lower value
 * as stale would keep rows the clock has un-revealed, which is the leak
 * host-side masking exists to prevent, one level up. The host holds the same
 * rule; this hook simply hands back whatever it is given.
 *
 * **Nothing here accumulates.** Each result REPLACES the previous one. A
 * grow-only cache would survive a seek to the end and then show the final
 * classification on a screen whose clock says lap 10.
 *
 * Half a second, not the tick's 100 ms: the mask changes only when some
 * driver completes a lap, about once every four and a half seconds, and the
 * column it feeds is lap-quantised anyway.
 */

import { useEffect, useRef, useState } from "react";
import { getBulk, whenBridgeReady, type Bulk } from "./bridge";

const POLL_INTERVAL_MS = 500;

export function useBulk(): Bulk | null {
  const [bulk, setBulk] = useState<Bulk | null>(null);
  // A ref, not state: the loop must read the newest revision without
  // re-subscribing, and a stale closure would ask for the same one forever.
  const revision = useRef(-1);

  useEffect(() => {
    let cancelled = false;
    let timer: number | undefined;

    const poll = async () => {
      const next = await getBulk(revision.current);
      if (cancelled) return;
      if (next) {
        revision.current = next.rev;
        setBulk(next);
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

  return bulk;
}
