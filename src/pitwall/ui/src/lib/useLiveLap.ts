/**
 * The lap in progress, polled fast enough that a sector appears when it happens.
 *
 * The third channel, and it exists because the other two are masked by
 * different things. `useBulk` is masked by completed laps and changes about
 * once every four and a half seconds; this is masked by the CLOCK, and a
 * sector opens somewhere in the field every 2.22 seconds. Widening the bulk
 * to carry it would re-send up to 342 KB at that cadence; this payload is
 * 2 KB for the whole field.
 *
 * Two hundred milliseconds, so a sector lands within a fifth of a second of
 * the car crossing it. The host answers null on every poll in between, which
 * is almost all of them.
 *
 * Nothing accumulates, and here that is not an optimisation: a rewind CLOSES
 * sectors, and a cache that only ever filled cells would leave a time on
 * screen for track the car has yet to re-drive.
 */

import { useEffect, useRef, useState } from "react";
import { getLiveLap, whenBridgeReady, type LiveLap } from "./bridge";

const POLL_INTERVAL_MS = 200;

export function useLiveLap(): LiveLap | null {
  const [live, setLive] = useState<LiveLap | null>(null);
  const revision = useRef(-1);

  useEffect(() => {
    let cancelled = false;
    let timer: number | undefined;

    const poll = async () => {
      const next = await getLiveLap(revision.current);
      if (cancelled) return;
      if (next) {
        revision.current = next.rev;
        setLive(next);
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

  return live;
}
