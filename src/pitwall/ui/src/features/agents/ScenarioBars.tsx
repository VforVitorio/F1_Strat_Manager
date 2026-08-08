/**
 * Four horizontal bars: STAY / PIT / UCUT / OCUT.
 *
 * 1:1 with `scenario_bars.py`. The normalisation, the winner and the
 * `--` for an absent scenario are all decided host side — the Monte
 * Carlo scores are signed and often all negative, and getting the shift
 * or the tie-break wrong changes which row reads as the winner, which is
 * not something a renderer should be able to do.
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
          <span className="scenario-bar">
            <span
              className="scenario-bar-fill"
              style={{ width: `${row.fill_pct}%`, background: row.bar_colour }}
            />
          </span>
          <span className="scenario-score" style={{ color: row.score_colour }}>
            {row.score}
          </span>
        </div>
      ))}
    </section>
  );
}
