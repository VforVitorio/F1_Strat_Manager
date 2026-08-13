/**
 * The socket's state, polled on its own slow timer.
 *
 * Separate from `useTick` on purpose: the tick loop learns nothing when the
 * producer dies, because what a dead producer sends is nothing. This asks the
 * host what its socket is doing, which is the only thing that still moves.
 *
 * One second rather than the tick's 100 ms. A connection label that lags a
 * beat costs nothing; ten polls a second for a word that changes twice a
 * session buys nothing.
 */

import { useEffect, useState } from "react";
import { getConnection, whenBridgeReady } from "./bridge";

const POLL_INTERVAL_MS = 1000;

export function useConnection(): string | null {
  const [connection, setConnection] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    let timer: number | undefined;

    const poll = async () => {
      const label = await getConnection();
      if (cancelled) return;
      setConnection(label);
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

  return connection;
}
