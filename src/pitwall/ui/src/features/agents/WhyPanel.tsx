/**
 * The decision band's second module: why this call, and what moved.
 *
 * **It replaces the reasoning tab panel, which is retired here.** That panel
 * was the largest bordered surface on the window and carried five short lines
 * at 1.9 % ink, with 318 px of it empty; its glance-value content was one line
 * of orchestrator prose. So the line is promoted onto the glass and the rest
 * goes one keypress away, into the same popup the agent consoles use for their
 * model detail.
 *
 * Nothing the panel showed is lost. The five agents' bodies moved to their own
 * consoles' tooltips (#1018, which landed first for exactly that reason), and
 * the orchestrator's whole narrative - including the DecisionMemory block that
 * appears only on a lap where the call moved - is this module's own tooltip.
 *
 * The narrative renders as a React TEXT NODE. It is an LLM's free text, and
 * free text becoming markup on the way to a webview is a class this window has
 * already closed once.
 */

import type { OrchestratorView } from "../../lib/agents";
import { Tooltip, useTooltipTarget } from "./Tooltip";

const TOOLTIP_ID = "tip-why";

export function WhyPanel({ view }: { view: OrchestratorView }) {
  const { anchor, props } = useTooltipTarget(TOOLTIP_ID, view.why_detail !== null);

  return (
    <section className="card why-panel" {...props}>
      <h2 className="band-title">WHY THIS CALL</h2>

      {view.why_detail && anchor ? (
        <Tooltip view={view.why_detail} anchor={anchor} id={TOOLTIP_ID} />
      ) : null}

      {/* An em dash, not an empty box. Before the first decision there is no
          narrative, and a blank module reads as one that failed to render. */}
      <p className="why-narrative">{view.why || "—"}</p>

      {/* What changed since the last lap, which this window had no
          first-class answer to: everything else overwrites in place ten times
          a second, and the only trace of a moved call was a heading inside a
          tab panel. Only rendered on the lap the ACTION moved, which is rare
          enough to stay a signal - and it sits here rather than beside the
          call, because "what changed" is part of why, not part of what. */}
      {view.changed ? <p className="orch-changed">{view.changed}</p> : null}
    </section>
  );
}
