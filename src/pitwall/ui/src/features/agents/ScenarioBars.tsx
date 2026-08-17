/**
 * Four horizontal bars: STAY / PIT / UCUT / OCUT.
 *
 * The normalisation, the winner, the enacted plan and the `--` for an
 * absent scenario are all decided host side — the Monte Carlo scores are
 * signed and often all negative, and getting the shift or the tie-break
 * wrong changes which row reads as the winner, which is not something a
 * renderer should be able to do.
 *
 * Two things this renders that it did not, both from the sprint-8 gate.
 * A row nobody scored draws **no track**, so it cannot be mistaken for
 * the one that came last (#963) — those were the same pixels before, an
 * executed diff found nought differing. And a Monte Carlo winner the
 * orchestrator overruled carries a `VETOED` mark (#962), because a "why"
 * panel crowning the plan that was not taken reads as the opposite of
 * the call.
 */

import type { ScenarioRow } from "../../lib/agents";

export function ScenarioBars({ rows }: { rows: ScenarioRow[] }) {
  return (
    <section className="card scenarios">
      <h2 className="scenarios-title">SCENARIO SCORES</h2>
      {rows.map((row) => (
        <div className="scenario-row" key={row.key}>
          <span className="scenario-label" style={{ color: row.label_colour }}>
            {row.label}
          </span>
          <span className={row.is_scored ? "scenario-bar" : "scenario-bar is-unscored"}>
            <span
              className="scenario-bar-fill"
              style={{ width: `${row.fill_pct}%`, background: row.bar_colour }}
            />
          </span>
          {row.note ? <span className="scenario-note">{row.note}</span> : null}
          <span className="scenario-score" style={{ color: row.score_colour }}>
            {row.score}
          </span>
        </div>
      ))}
    </section>
  );
}
