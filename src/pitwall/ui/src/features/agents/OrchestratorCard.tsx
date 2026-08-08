/**
 * The N31 decision panel: action badge, confidence, the two regime chips,
 * the plan line and the guardrail.
 *
 * 1:1 with `orchestrator_card.py`. Every string and colour is decided
 * host side, including the plan line's three branches, so the "stint
 * continues · no pit window yet" copy cannot quietly become three "--"
 * chips here.
 */

import type { OrchestratorView } from "../../lib/agents";

export function OrchestratorCard({ view }: { view: OrchestratorView }) {
  const filled = view.confidence === null ? 0 : Math.round(view.confidence * 100);
  return (
    <section className="card orchestrator">
      <div className="orch-top">
        <div className="orch-badge" style={{ background: view.action_colour }}>
          {view.action}
        </div>
        <div className="orch-conf">
          <span className="orch-conf-label">{view.confidence_label}</span>
          {/* A div rather than <progress>: Qt paints a flat two-stop bar and
              the native element brings a platform look that is not it. */}
          <div className="orch-bar">
            <div
              className="orch-bar-fill"
              style={{ width: `${filled}%`, background: view.confidence_colour }}
            />
          </div>
        </div>
      </div>

      <div className="orch-chips">
        <span className="orch-chip" style={{ color: view.pace_colour, borderColor: view.pace_colour }}>
          {view.pace}
        </span>
        <span className="orch-chip" style={{ color: view.risk_colour, borderColor: view.risk_colour }}>
          {view.risk}
        </span>
      </div>

      {/* The plan line can carry a compound pill, which is an HTML span
          built and escaped in src/arcade/palette.py. */}
      <p className="orch-plan" dangerouslySetInnerHTML={{ __html: view.plan }} />

      {view.guardrail ? <p className="orch-guardrail">{view.guardrail}</p> : null}
    </section>
  );
}
