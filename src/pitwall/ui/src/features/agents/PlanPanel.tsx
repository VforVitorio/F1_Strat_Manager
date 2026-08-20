/**
 * The decision band's fourth module: what happens next.
 *
 * A card around the timeline, and the reason it is a separate file from it: the
 * module is one of the band's four and wears their shared chrome, while the
 * timeline is a drawing that could be placed anywhere.
 */

import type { PlanTimelineView } from "../../lib/agents";
import { PlanTimeline } from "./PlanTimeline";

export function PlanPanel({ view }: { view: PlanTimelineView }) {
  return (
    <section className="card plan-panel">
      <h2 className="band-title">PLAN</h2>
      <PlanTimeline view={view} />
    </section>
  );
}
