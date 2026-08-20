/**
 * The band's fourth module: the race on one lap axis.
 *
 * Three lanes over lap 1 to the flag - the tyre cliff above, the stints on the
 * track, the marks below - plus the orchestrator's plan line as a caption.
 *
 * **Positioned spans, not a third chart.** This is four rectangles and two
 * rules; a third imperative ECharts instance re-rendering at 10 Hz for that is
 * cost with no return, and a span's geometry can be asserted from the DOM
 * without asking a canvas what it drew.
 *
 * Every position arrives as a per cent computed host side (`timeline.py`), so
 * the arithmetic that decides where lap 24 sits is in the same language as the
 * rest of the view and a Python test can read it.
 *
 * The distinction the lane carries is **filled versus hollow**, not solid
 * versus dashed: a dashed stroke already means broadcast-tier data in the
 * sibling DATA window, and one stroke cannot mean two things across two
 * windows a reader has open at once.
 */

import type { PlanTimelineView } from "../../lib/agents";

export function PlanTimeline({ view }: { view: PlanTimelineView }) {
  // Before the arcade has said how long the race is there is no axis to place
  // anything on. An empty track and the caption's own empty-state branch is
  // the honest rendering; a track drawn against a guessed length would put
  // every mark somewhere wrong.
  const drawable = view.total_laps >= 2;

  return (
    <div className="plan-timeline">
      <div className="plan-track" role="img" aria-label={planLabel(view)}>
        <div className="plan-lane plan-lane-hazard">
          {view.cliff ? (
            <span
              className="plan-cliff"
              style={{
                left: `${view.cliff.left_pct}%`,
                width: `${view.cliff.width_pct}%`,
                // The tyre chart's own band colour, through the view. A token
                // here would be a second place the amber lives.
                backgroundColor: view.cliff.colour,
              }}
            />
          ) : null}
        </div>

        <div className="plan-lane plan-lane-stints">
          {view.segments.map((segment) => (
            <span
              key={`${segment.lo}-${segment.hi}-${segment.planned ? "p" : "r"}`}
              className={segment.planned ? "plan-stint is-planned" : "plan-stint"}
              style={{
                left: `${segment.left_pct}%`,
                width: `${segment.width_pct}%`,
                // A run stint is a tint with a full-strength top edge; a
                // planned one is an outline with nothing inside it.
                borderColor: segment.colour,
                backgroundColor: segment.planned ? "transparent" : segment.colour,
              }}
            />
          ))}
        </div>

        {/* The marks. The cursor is the same 1 px `TEXT_TERTIARY` vertical the
            two charts use for the current lap, so "now" is one mark in three
            places rather than three marks that happen to line up. */}
        {view.current_pct !== null ? (
          <span className="plan-now" style={{ left: `${view.current_pct}%` }} />
        ) : null}
        {view.pit_pct !== null ? (
          <span className="plan-pit" style={{ left: `${view.pit_pct}%` }}>
            ▼
          </span>
        ) : null}

        <div className="plan-lane plan-lane-labels">
          {drawable ? <span className="plan-end is-start">L1</span> : null}
          {view.current_lap !== null ? (
            <span className="plan-cursor-label" style={{ left: `${view.current_pct}%` }}>
              L{view.current_lap}
            </span>
          ) : null}
          {drawable ? <span className="plan-end is-finish">L{view.total_laps}</span> : null}
        </div>
      </div>

      {/* The orchestrator's own plan line, verbatim, including the compound
          pill it can carry. That pill is an HTML span built and escaped in
          `src/arcade/palette.py` and it is the last markup sink on this
          window's decision surface. */}
      <p className="plan-caption" dangerouslySetInnerHTML={{ __html: view.caption }} />
    </div>
  );
}

/**
 * One sentence for a screen reader, since the lanes are geometry.
 *
 * Built from the same fields the spans are placed from, so it cannot describe
 * a race the picture is not showing.
 */
function planLabel(view: PlanTimelineView): string {
  if (view.total_laps < 2) return "No race loaded";
  const stints = view.segments
    .filter((segment) => !segment.planned)
    .map((segment) => `${segment.compound ?? "unknown"} laps ${segment.lo} to ${segment.hi}`);
  const planned = view.segments.find((segment) => segment.planned);
  const parts = [`Lap ${view.current_lap ?? "?"} of ${view.total_laps}`, ...stints];
  if (planned) parts.push(`planned ${planned.compound} from lap ${planned.lo}`);
  if (view.cliff) parts.push(`cliff between laps ${view.cliff.lo} and ${view.cliff.hi}`);
  return parts.join(", ");
}
