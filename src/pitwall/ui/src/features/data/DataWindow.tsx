/**
 * PITWALL · DATA - band 1 over two columns.
 *
 * **The four bands are not four stacked rows, and the arithmetic is why.**
 * This window's real client area is 1485 x 833 logical px, not the 1500 x 950
 * `WindowSpec` asks for: `place()` clamps the height to the screen and the
 * title bar takes 37 more. Minus the status bar and the body padding, 790 px
 * are left. A full 20-row timing tower is 439 and band 4 stops being a chart
 * below about 420 - stacked with band 1 and the gaps that is 908, over budget
 * by 118 px with band 3 still at zero, on the largest screen in the fleet.
 * Rendered at the 303 px actually left, band 4's axis labels collide and its
 * traces are ribbons.
 *
 * So band 1 stays full width and the rest becomes two columns: the all-cars
 * world on the left (tower over bests, sprints 5) and the own-car world on
 * the right (band 4, and band 3 as a tab of it in sprint 6). That split is
 * not a compromise imposed by the pixels - it is the zoning a real wall
 * uses, where the two worlds sit on physically different surfaces.
 *
 * The measured budget is in `~/.claude/plans/pitwall-sprint5/
 * band-height-budget.md`; the drawn layout is in the project's memory.
 */

import { useState } from "react";

import { BestsPanel } from "./BestsPanel";
import { OwnCarTraces } from "./OwnCarTraces";
import { RacePaceGrid } from "./RacePaceGrid";
import { RaceTraceChart } from "./RaceTraceChart";
import { RadioFeed } from "./RadioFeed";
import { StatusStrip } from "./StatusStrip";
import { TimingTower } from "./TimingTower";
import { TrackRing } from "./TrackRing";
import { useBulk } from "../../lib/useBulk";
import { useConnection } from "../../lib/useConnection";
import { useLiveLap } from "../../lib/useLiveLap";
import { useStatusText } from "../../lib/useStatusText";
import { useTick } from "../../lib/useTick";
import { waitingBody, waitingStatus } from "../../lib/waitingCopy";

/**
 * The right column's tabs, in the order a strategist reaches for them.
 *
 * A list rather than three literals in the JSX, so the id a button carries and
 * the id the panel below switches on cannot drift apart - which is exactly how
 * a fourth tab would arrive rendering the third one's panel.
 */
const TABS = [
  ["traces", "TRACES"],
  ["pace", "RACE PACE"],
  ["trace", "RACE TRACE"],
] as const;

export function DataWindow() {
  const { tick, discontinuity, live } = useTick();
  const connection = useConnection();
  const bulk = useBulk();
  const liveLap = useLiveLap();
  // Qt re-arms `showMessage(f"lap {lap} · live", 1500)` on every broadcast,
  // so the line stays up while streaming and clears 1.5 s after the producer
  // stops. `useStatusText` is keyed on the sequence for that reason.
  /**
   * The producer is gone, and every panel below is holding numbers from before.
   *
   * The socket's own label is the only thing that still moves once the ticks
   * stop - `useConnection` polls it at 1 Hz for exactly this - so the state is
   * known client-side and needs no wire change. Before this, a dead producer left
   * a full board of confident values, the lap counter still saying `L 28/57`, the
   * track chip still asserting GREEN and `PLAYBACK 2x` still claiming the replay
   * was advancing, with a 77 x 18 chip and a blank status bar as the only tells.
   * A frozen tower that looks live is the one state a pit wall must not mistake,
   * and this window sits beside a moving arcade.
   */
  const frozen = connection?.label === "Disconnected" && tick !== null;
  // Not transient when frozen: the 1.5 s auto-clear is what a LIVE bar wants, and
  // it is why the only remaining signal used to be an EMPTY bar. It also has to
  // stop saying `live`, or the window contradicts its own banner one line down.
  const status = !tick
    ? { text: waitingStatus(connection?.label ?? null), transient: false }
    : frozen
      ? { text: `DATA FROZEN · last tick lap ${tick.arcade.lap}`, transient: false }
      : { text: `lap ${tick.arcade.lap} · live`, transient: true };
  const statusText = useStatusText(status, tick?.seq ?? null);
  const [tab, setTab] = useState<"traces" | "pace" | "trace">("traces");

  return (
    <main className="data-window">
      <div className="data-body">
        <StatusStrip tick={live ? tick : null} connection={connection} frozen={frozen} />
        {live && tick ? (
          // Dimmed, never desaturated, never blurred and never faded much:
          // the last known state is still operationally useful and has to stay
          // readable. What the treatment says is "this is history", not "this is
          // unavailable".
          <div className={frozen ? "data-main is-frozen" : "data-main"}>
            <div className="left-column">
              <TimingTower arcade={tick.arcade} bulk={bulk} live={liveLap} />
              <BestsPanel bulk={bulk} />
            </div>
            <div className="right-column">
              {/* The tab strip the delivery plan put here. Band 3 needs the
                  full 825 px of this column - measured: with the ring still
                  mounted its cells clip 1,101 of 1,140 - so the two worlds
                  take turns rather than share.

                  Band 3 is TWO panels, not one, and they answer questions the
                  other cannot: the grid says how quick each lap was, the trace
                  says where everyone is. Splitting them across two tabs rather
                  than stacking them is the same column arithmetic again - a
                  grid squeezed to half this height stops showing enough laps to
                  be a history, and a trace at 300 px stops resolving the gaps
                  it exists to show. */}
              <nav className="tab-strip" role="tablist">
                {TABS.map(([id, label]) => (
                  <button
                    key={id}
                    role="tab"
                    aria-selected={tab === id}
                    className={tab === id ? "tab is-active" : "tab"}
                    onClick={() => setTab(id)}
                  >
                    {label}
                  </button>
                ))}
              </nav>
              {tab === "traces" && (
                <div className="band4">
                  <OwnCarTraces tick={tick} discontinuity={discontinuity} frozen={frozen} />
                  <div className="side-column">
                    <TrackRing arcade={tick.arcade} />
                    <RadioFeed bulk={bulk} driverMain={tick.arcade.driver_main} />
                  </div>
                </div>
              )}
              {tab === "pace" && <RacePaceGrid bulk={bulk} order={tick.arcade.race_order} />}
              {tab === "trace" && <RaceTraceChart bulk={bulk} arcade={tick.arcade} />}
            </div>
          </div>
        ) : (
          <p className="data-waiting">{waitingBody(connection?.label ?? null)}</p>
        )}
      </div>
      <footer className="status-bar">{statusText}</footer>
    </main>
  );
}
