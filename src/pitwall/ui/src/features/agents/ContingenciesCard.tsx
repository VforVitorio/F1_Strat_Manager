/**
 * The orchestrator's branch plans, in the side column's empty region.
 *
 * The band this card closes runs decision, scores, plan: what we are doing and
 * why. It never said what we do INSTEAD. `contingencies` and `key_risks` have
 * ridden the wire since #1048's schema bump with nothing rendering them.
 *
 * **A bespoke card rather than a `Console`.** `AgentCard` has no footer slot,
 * and the routing strip has to sit BELOW the scrolling body rather than inside
 * it; its `children` land in a 140 px `.agent-chart` instead. `WhyPanel` is the
 * precedent for a card that is its own component and still reuses the tooltip.
 *
 * **It uses `overflow: hidden`, which `.agent-card` deliberately does not.**
 * That card keeps `overflow: visible` under a comment about a scrolling
 * ancestor clipping its popup. `Tooltip` portals to `document.body` now, so no
 * ancestor can clip it, and the comment is describing a constraint that no
 * longer applies. Do not "restore" `visible` here.
 *
 * The frozen-feed `filter: brightness(0.72)` reaches this card through
 * `.agents-body`, and that is correct: a contingency is exactly as stale as the
 * call above it.
 */

import { useCallback, useEffect, useState } from "react";

import type { ContingenciesView, ContingencyRow } from "../../lib/agents";
import { useScrollEdges } from "../../lib/useScrollEdges";
import { Tooltip, useTooltipTarget } from "./Tooltip";

/**
 * The room below which the card renders nothing at all.
 *
 * Measured by hand at the wide client, adding the parts: 13 px of title, a 6 px
 * gap, and one row at about 45 px (trigger 13, action 12, two clamped rationale
 * lines 22), inside 20 px of padding and 2 px of border. Below that the card can
 * only show a title over a cut-off row.
 *
 * The three clients the product actually opens at leave 302, 222 and **30** px.
 * The last one is a 1366x768 laptop, where a card that rendered anyway would be
 * a heading with nothing under it, which is the phantom-box defect this window
 * already removed once.
 */
const MIN_ROOM = 86;

/**
 * How much room the column left this card, and whether that is enough.
 *
 * **The measured quantity cannot be moved by the decision, and that is the
 * whole design.** The card is `flex: 1 1 0` with `min-height: 0`, so its
 * flex basis is zero and its contents never enter the sizing calculation: its
 * used height is exactly the column's free space and nothing it renders can
 * change it. The callback therefore depends on the node alone.
 *
 * `useFitsRanked` in the DATA window is the counter-example, and #1083 is open
 * because of it: its decision changes the height it then re-measures, and its
 * callback depends on its own fit state, so the observer is torn down and
 * rebuilt on every decision and a resize landing in that window is missed.
 *
 * The rect, not `clientHeight`: with `box-sizing: border-box` the border-box
 * height is the same whether the card is showing its border or not, so the
 * reading does not jump by 2 px across the decision.
 */
function useRoom(node: HTMLElement | null): number {
  const [room, setRoom] = useState(0);

  const measure = useCallback(() => {
    if (!node) return;
    setRoom(node.getBoundingClientRect().height);
  }, [node]);

  useEffect(() => {
    if (!node) return;
    measure();
    const observer = new ResizeObserver(measure);
    observer.observe(node, { box: "border-box" });
    return () => observer.disconnect();
  }, [node, measure]);

  return room;
}

export function ContingenciesCard({ view }: { view: ContingenciesView }) {
  const [host, setHost] = useState<HTMLElement | null>(null);
  const [body, setBody] = useState<HTMLElement | null>(null);
  const room = useRoom(host);
  const { edges, measure } = useScrollEdges(body);

  // Below the floor the element stays in the flow, holding the residual, and
  // drops everything that makes it look like a panel. Window background reads
  // as nothing; an empty bordered box reads as content that failed to arrive.
  const shown = room >= MIN_ROOM;

  return (
    <section
      ref={setHost}
      className={shown ? "card contingencies" : "contingencies is-collapsed"}
      aria-hidden={shown ? undefined : true}
    >
      {shown ? (
        <>
          <h2 className="cty-title">
            CONTINGENCIES
            {view.rows.length ? <span className="cty-count">{view.rows.length}</span> : null}
          </h2>
          <div
            ref={setBody}
            className={`cty-body${edges.above ? " has-above" : ""}${
              edges.below ? " has-below" : ""
            }`}
            onScroll={measure}
            tabIndex={0}
            role="region"
            aria-label="Contingency plans"
          >
            {view.empty ? (
              <p className="cty-empty">{view.empty}</p>
            ) : (
              view.rows.map((row, index) => (
                <ContingencyRowView key={`${row.trigger}-${index}`} row={row} index={index} />
              ))
            )}
            {/* Beside the branches, not behind a hover on the title. A risk the
                orchestrator flagged is content, and content a reader has to go
                hunting for may as well not be on the wire. */}
            {view.risks.length ? (
              <section className="cty-risks">
                <h3 className="cty-risks-title">RISKS</h3>
                {view.risks.map((risk) => (
                  <p key={risk} className="cty-risk">
                    {risk}
                  </p>
                ))}
              </section>
            ) : null}
          </div>
        </>
      ) : null}
    </section>
  );
}

/**
 * One branch.
 *
 * The hover affordance is gated on a class this component sets, NOT on
 * `:has(.agent-tooltip)`: the popup is portaled to `document.body`, so it is
 * not a descendant of the card whose hover it was meant to drive, and that
 * selector matched nothing. The six consoles carried it until #1089 and now
 * set their own class the same way.
 *
 * Hover changes the BACKGROUND only. This card sits in a flex column above
 * nothing and below PIT, so anything that changes its box moves a neighbour.
 */
function ContingencyRowView({ row, index }: { row: ContingencyRow; index: number }) {
  const id = `tip-cty-${index}`;
  const { anchor, props, hold } = useTooltipTarget(id, row.detail !== null);
  return (
    <article className={`cty-row${row.detail ? " has-detail" : ""}`} {...props}>
      {row.detail && anchor ? (
        <Tooltip view={row.detail} anchor={anchor} id={id} hold={hold} />
      ) : null}
      <p className="cty-trigger">
        <span className="cty-priority">{row.priority}</span>
        {row.trigger}
      </p>
      <p className="cty-switch">{row.switch_to}</p>
      <p className="cty-rationale">{row.rationale}</p>
    </article>
  );
}
