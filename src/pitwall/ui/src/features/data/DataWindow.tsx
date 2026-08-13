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

import { BestsPanel } from "./BestsPanel";
import { OwnCarTraces } from "./OwnCarTraces";
import { StatusStrip } from "./StatusStrip";
import { TimingTower } from "./TimingTower";
import { TrackRing } from "./TrackRing";
import { useBulk } from "../../lib/useBulk";
import { useConnection } from "../../lib/useConnection";
import { useLiveLap } from "../../lib/useLiveLap";
import { useStatusText } from "../../lib/useStatusText";
import { useTick } from "../../lib/useTick";

const WAITING = { text: "Waiting for arcade stream…", transient: false } as const;

export function DataWindow() {
  const { tick, discontinuity, live } = useTick();
  const connection = useConnection();
  const bulk = useBulk();
  const liveLap = useLiveLap();
  // Qt re-arms `showMessage(f"lap {lap} · live", 1500)` on every broadcast,
  // so the line stays up while streaming and clears 1.5 s after the producer
  // stops. `useStatusText` is keyed on the sequence for that reason.
  const status = tick ? { text: `lap ${tick.arcade.lap} · live`, transient: true } : WAITING;
  const statusText = useStatusText(status, tick?.seq ?? null);

  return (
    <main className="data-window">
      <div className="data-body">
        <StatusStrip tick={live ? tick : null} connection={connection} />
        {live && tick ? (
          <div className="data-main">
            <div className="left-column">
              <TimingTower arcade={tick.arcade} bulk={bulk} live={liveLap} />
              <BestsPanel bulk={bulk} />
            </div>
            <div className="band4">
              <OwnCarTraces tick={tick} discontinuity={discontinuity} />
              <TrackRing arcade={tick.arcade} />
            </div>
          </div>
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
