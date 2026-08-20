/**
 * The decision band's first module: what the orchestrator called, and how
 * sure it is.
 *
 * The plan line moved to `PlanPanel` when the band replaced the left column:
 * "what we are doing" and "what happens next" are two of the four questions
 * the band answers in order, and they were sharing a card.
 *
 * **The action sheds its button costume.** It was a 200x70 px filled pill with
 * a 10 px radius, centred - the geometry of a primary call-to-action, on a
 * surface where nothing can be pressed. Filled, it also made the action the
 * largest painted colour on the window, so a red `PIT NOW` out-alarmed the
 * actual alarms: the same DANGER red that means a fault elsewhere was carrying
 * an identity here, at ten times the area.
 *
 * Now it is text in `action_colour` on the panel, with a 4 px rule of the same
 * colour down the card's left edge. The identity survives at a fraction of the
 * painted area, and every action colour clears AA as text on `--qt-panel`
 * (#181633) - DANGER 4.64:1, INFO 4.75, ACCENT 6.42, SUCCESS 6.89, WARNING
 * 8.14, and TEXT_SECONDARY for DNF far above. Which is why
 * `action_text_colour` is gone: it existed to pick a readable ink for the
 * FILL, and there is no fill.
 *
 * The banner is bounded (`nowrap` + ellipsis) because `classify_action` falls
 * back to echoing the raw string for anything outside its seven, and an
 * unknown producer word at 32 px/800 is unbounded.
 *
 * The `was STAY OUT (0.58) · L22` chip lives in the WHY module, not here: what
 * CHANGED is part of why, not part of what, and rendering it in both modules
 * put the same sentence on the glass twice.
 */

import type { OrchestratorView } from "../../lib/agents";

export function OrchestratorCard({ view }: { view: OrchestratorView }) {
  return (
    <section className="card orchestrator" style={{ borderLeftColor: view.action_colour }}>
      <p className="orch-action" style={{ color: view.action_colour }}>
        {view.action}
      </p>

      <div className="orch-conf">
        <span className="orch-conf-value" style={{ color: view.confidence_colour }}>
          {view.confidence_text}
        </span>
        {/* A div rather than <progress>: Qt paints a flat two-stop bar and
            the native element brings a platform look that is not it. */}
        <div className="orch-bar">
          <div
            className="orch-bar-fill"
            style={{ width: `${view.confidence_fill}%`, background: view.confidence_colour }}
          />
        </div>
        <span className="orch-conf-caption">CONFIDENCE</span>
      </div>

      {/* Two facts, not two chips. The outline-pill costume died with the
          filled one: both said "press me" on a surface with nothing to press,
          and the words carry the meaning on their own. */}
      <p className="orch-facts">
        <span style={{ color: view.pace_colour }}>{view.pace}</span>
        <span className="orch-facts-sep"> · </span>
        <span style={{ color: view.risk_colour }}>{view.risk}</span>
      </p>
    </section>
  );
}
