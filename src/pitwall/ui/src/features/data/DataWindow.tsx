/**
 * PITWALL · DATA - the four-band shell.
 *
 * Band 4 (the own-car traces, and the ring beside them) lands first: it is
 * the only band with an existing original to port, so its fidelity is
 * checkable field by field against `telemetry_panel.py` instead of being a
 * matter of taste. Bands 1 and 2 (status strip, timing table, bests) follow
 * in sprint 5 and band 3 (race pace) in sprint 6; the order changed on
 * 2026-08-09 and this docstring used to teach the old one.
 *
 * **The status bar knows two states, not Qt's four.** `TelemetryWindow` also
 * paints "Stream connected" and "Disconnected — retrying…", which it gets
 * from its own client's signals. PITWALL's client is shared and owned by the
 * host, which surfaces its connection label only through `get_agents_view`.
 * Rather than grow a host method for one string, connection state waits for
 * the band-1 STATUS STRIP in sprint 5, where it belongs next to the flag and
 * the track status. What is here is honest about what this window can know:
 * it is waiting, or the producer spoke within the last second and a half.
 */

import { OwnCarTraces } from "./OwnCarTraces";
import { useStatusText } from "../../lib/useStatusText";
import { useTick } from "../../lib/useTick";

const WAITING = { text: "Waiting for arcade stream…", transient: false } as const;

export function DataWindow() {
  const { tick, discontinuity, live } = useTick();
  // Qt re-arms `showMessage(f"lap {lap} · live", 1500)` on every broadcast,
  // so the line stays up while streaming and clears 1.5 s after the producer
  // stops. `useStatusText` is keyed on the sequence for that reason.
  const status = tick ? { text: `lap ${tick.arcade.lap} · live`, transient: true } : WAITING;
  const statusText = useStatusText(status, tick?.seq ?? null);

  return (
    <main className="data-window">
      <div className="data-body">
        {live && tick ? (
          <OwnCarTraces tick={tick} discontinuity={discontinuity} />
        ) : (
          <p className="data-waiting">
            Waiting for the arcade broadcast. Start a replay with <code>--strategy</code>.
          </p>
        )}
      </div>
      <footer className="status-bar">{statusText}</footer>
    </main>
  );
}
