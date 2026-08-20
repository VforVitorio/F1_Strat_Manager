/**
 * The decision band's fourth module: what happens next.
 *
 * Today it is the orchestrator's own plan line, moved out of the decision card
 * and given a column of its own. The line is built host side and carries three
 * branches - a scheduled stop, "stint continues · no pit window yet" on
 * STAY_OUT with nothing tactical, and "Pit plan pending" otherwise - so the
 * copy cannot quietly become three "--" chips here.
 *
 * It keeps `dangerouslySetInnerHTML` because the line can carry a compound
 * pill, an HTML span built and escaped in `src/arcade/palette.py`. That is the
 * last markup sink on this window's decision surface and it is the reason the
 * pill has to become a typed segment before the card bodies do: a component
 * that re-imports the path the migration is retiring makes the migration's
 * claim false.
 */

export function PlanPanel({ plan }: { plan: string }) {
  return (
    <section className="card plan-panel">
      <h2 className="band-title">PLAN</h2>
      <p className="plan-caption" dangerouslySetInnerHTML={{ __html: plan }} />
    </section>
  );
}
