/**
 * PIT EXIT: where a stop taken on THIS lap would put us, and next to whom.
 *
 * The card is a HYPOTHETICAL and says so in its own header. That is the whole
 * defence against #962's defect class, and it replaces the obvious rule that
 * was rejected during design: suppressing the card on STAY_OUT laps would idle
 * it on exactly the laps a pit wall is asking "should we box?", which is when
 * the readout earns its space.
 *
 * Nothing here decides anything. The builder picks the state, the sentence and
 * the words "ahead"/"behind"; this renders what it is handed, the same division
 * `ScenarioBars` keeps for the same reason.
 *
 * **No identity colours on the body.** The live call wears ACCENT one card up;
 * a branch that is not happening must not, which is the rule
 * `build_contingencies` states for itself.
 */

import type { PitExitView } from "../../lib/agents";

export function PitExitCard({ view }: { view?: PitExitView }) {
  // An absent field is an OLD host feeding a NEW bundle, and it lands in the
  // same place a host with nothing to say does.
  const state = view ?? { state: "idle" as const, note: "— no rejoin geometry —" };

  return (
    <section className="card agent-card slot-exit">
      <header className="agent-card-head">
        <span className="agent-title">PIT EXIT</span>
        {state.state === "ready" && <span className="exit-qualifier">{state.qualifier}</span>}
      </header>
      {state.state === "idle" ? (
        <div className="agent-card-body">
          <p className="agent-line is-idle">{state.note}</p>
        </div>
      ) : (
        <div className="agent-card-body">
          <p className="agent-headline exit-headline">
            {state.headline}
            {state.band && <span className="exit-band">{state.band}</span>}
          </p>
          {state.rows.map((row) => (
            <p className="agent-line exit-row" key={row.side}>
              <span className="exit-driver">{row.driver}</span>
              {row.gap} {row.side}
            </p>
          ))}
        </div>
      )}
    </section>
  );
}
