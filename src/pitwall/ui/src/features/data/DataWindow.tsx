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

import { useEffect, useState } from "react";

import { BestsPanel } from "./BestsPanel";
import { OwnCarTraces } from "./OwnCarTraces";
import { useTraceFrame } from "./useTraceFrame";
import { RacePaceGrid } from "./RacePaceGrid";
import { RaceTraceChart } from "./RaceTraceChart";
import { RadioFeed } from "./RadioFeed";
import { StatusStrip } from "./StatusStrip";
import { TimingTower } from "./TimingTower";
import { TrackRing } from "./TrackRing";
import { driverStatus } from "../../lib/driverStatus";
import { useBulk } from "../../lib/useBulk";
import { useConnection } from "../../lib/useConnection";
import { useLiveLap } from "../../lib/useLiveLap";
import { useStatusText } from "../../lib/useStatusText";
import { useTick } from "../../lib/useTick";
import { waitingBody, waitingStatus } from "../../lib/waitingCopy";
import { AXIS_TEXT } from "../../lib/chart";
import { driverColour } from "../../lib/driverColour";

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
  // **Owned here, not in the panel that draws it (#1056).** The tab strip renders
  // `OwnCarTraces` conditionally, so a reader leaving TRACES unmounts it; a buffer
  // living inside it was destroyed and the rest of the lap went with it, because
  // the wire carries only the span since the last tick and cannot backfill. This
  // component sees every tick whichever tab is showing, so ingestion continues
  // while the panel is away.
  // **Resolved ONCE, here, and passed down.** The rival has four consumers on
  // this window - this hook's selection, the header chip, the BROADCAST tag's
  // blind-note, and the ring's label placement - and each used to read
  // `tick.arcade.driver_rival` for itself. Four independent reads agree only
  // while there is one possible answer; the moment the tower can pin a car
  // (#1051) any consumer left reading the tick shows a different rival from the
  // rest, inside one window. One value passed down cannot disagree with itself.
  //
  // `null` means FOLLOW THE PRODUCER, not "no rival". It cannot collide with a
  // real choice because a driver code on the wire is a non-null non-empty
  // string, which is the sentinel rule this repo has paid for twice: a defaulted
  // position of 0 is a place a real car can be, and the leader then "found" the
  // car that had just crashed.
  const [pinned, setPinned] = useState<string | null>(null);
  const rival = pinned ?? tick?.arcade.driver_rival ?? null;
  const traceFrame = useTraceFrame(tick, discontinuity, rival);
  // The rival's COLOUR is resolved here for the same reason its code is: band 4
  // draws it on six series and on the header chip, and those were free to
  // disagree. The fallback is a literal rather than a `var(--qt-*)` because one
  // of the consumers is an ECharts canvas, which cannot resolve a custom
  // property and would silently fall back to its own palette (#1070).
  const rivalColour = driverColour(tick?.arcade.driver_colors ?? {}, rival, AXIS_TEXT);
  // **And the OWN car's, from the same map, for the same reason.** #1070 gave the
  // rival its team colour and left the main chip on a palette blue in the
  // stylesheet, five lines above its own explanatory comment - so this window
  // painted NOR blue on the traces chip while its tower, its ring and its race
  // trace all painted the same car papaya.
  //
  // The fallback is INFO rather than the axis grey the rival degrades to, and
  // the two are deliberately different: an unknown RIVAL is one of nineteen and
  // reads as "not identified", while the main car is always named right beside
  // the chip, so its degrade only has to be a colour no team owns. Measured, INFO
  // sits 70.8 from the nearest team colour on the wire.
  const mainColour = driverColour(
    tick?.arcade.driver_colors ?? {},
    tick?.arcade.driver_main ?? null,
    "#3b82f6",
  );

  // The pin releases when its car retires, or when the code stops being on the
  // wire at all - a relaunched arcade pointed at another race is the second
  // case, and it is why the car is checked for existence before its status.
  // Holding a pin on a car that is not racing would leave band 4 comparing
  // against a frozen trace with nothing to say.
  useEffect(() => {
    if (pinned === null || !tick) return;
    const car = tick.arcade.drivers[pinned];
    if (!car || driverStatus(car) === "out") setPinned(null);
  }, [pinned, tick]);
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
              <TimingTower
                arcade={tick.arcade}
                bulk={bulk}
                live={liveLap}
                pinned={pinned}
                onPin={setPinned}
              />
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
                  {traceFrame && (
                    <OwnCarTraces
              tick={tick}
              frame={traceFrame}
              rival={rival}
              rivalColour={rivalColour}
              mainColour={mainColour}
              frozen={frozen}
            />
                  )}
                  <div className="side-column">
                    <TrackRing arcade={tick.arcade} rival={rival} />
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
