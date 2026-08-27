/**
 * One sub-agent card, shared by all six.
 *
 * A dumb view, like the Qt `AgentCard` it replaces: it does not know
 * which agent it shows. Title and content arrive from the host, which
 * produced them with the Qt window's own formatters.
 *
 * `dangerouslySetInnerHTML` survives on the headline and the body lines,
 * which can carry the compound pill and the flag chips - HTML spans built
 * in `src/arcade/palette.py`. Because those lines ARE markup, every other
 * field on them has to be escaped, and `agent_formatters._escaped` is what
 * does it: the messages are free text off the NLP pipeline, so an unescaped
 * `<` in a transcript becomes a tag.
 * **The tooltip no longer does**: the two tooltip
 * formatters into structured data, so the popup below is built from
 * fields rather than parsed out of a dead toolkit's dialect. The pills
 * are the same debt one layer down and are filed separately.
 */

import { useState } from "react";

import type { AgentCardView } from "../../lib/agents";
import { useScrollEdges } from "../../lib/useScrollEdges";
import { Tooltip, useTooltipTarget } from "./Tooltip";

export function AgentCard({
  title,
  card,
  slot,
  children,
}: {
  title: string;
  card: AgentCardView;
  /** Which console this is, as a `slot-*` class; the stylesheet owns the shape. */
  slot?: string;
  children?: React.ReactNode;
}) {
  const idle = card.status === "IDLE";
  // The whole card is the target, as in Qt. `null` suppresses the popup - not
  // the empty string Qt used, because a falsy value that is also a legitimate
  // rendering is the sentinel shape this repo keeps paying for.
  const tooltipId = `tip-${slot ?? title.toLowerCase()}`;
  const { anchor, props, hold } = useTooltipTarget(tooltipId, card.tooltip !== null);
  // State rather than a ref: the observer has to attach the first time the node
  // exists, and assigning a ref does not re-run an effect. `BestsPanel` learned
  // the same thing.
  const [body, setBody] = useState<HTMLElement | null>(null);
  const { edges, measure } = useScrollEdges(body);


  return (
    <section
      className={
        [
          "card",
          "agent-card",
          idle ? "is-idle" : "",
          slot ? `slot-${slot}` : "",
          // The hover affordance is gated on THIS, not on
          // `:has(.agent-tooltip)`. The popup is portaled to `document.body`,
          // so it is never a descendant of the card and that selector matched
          // nothing for as long as it existed (#1089). `ContingenciesCard`
          // already did it this way and the reason was written down in the
          // stylesheet beside it; it was never carried back to the six
          // consoles.
          card.tooltip ? "has-tooltip" : "",
        ]
          .filter(Boolean)
          .join(" ")
      }
      {...props}
    >
      <header className="agent-card-head">
        <span className="agent-glyph" style={{ color: card.glyph_colour }}>
          {card.glyph}
        </span>
        <span className="agent-title">{title}</span>
      </header>

      {card.tooltip && anchor ? (
        <Tooltip view={card.tooltip} anchor={anchor} id={tooltipId} hold={hold} />
      ) : null}

      {/* The card scrolls here, not on the card itself, so the card is not
          a scrolling ancestor of anything. Qt clips a card that overflows
          its cap - the migration README records the right column "clipped
          mid-card" - so the footprint matches and the content stays
          reachable. */}
      <div
        className={
          ["agent-card-body", edges.above ? "has-above" : "", edges.below ? "has-below" : ""]
            .filter(Boolean)
            .join(" ")
        }
        ref={setBody}
        onScroll={measure}
        // Keyboard-reachable for the same reason the pace grid's scroller is:
        // with the scrollbar hidden this is the only way a keyboard user reaches
        // the rest of the card at all.
        tabIndex={0}
        role="region"
        aria-label={`${title}, scrollable`}
      >
        <p
          className="agent-headline"
          style={{ color: card.headline_colour }}
          dangerouslySetInnerHTML={{ __html: card.headline }}
        />

        {card.lines.map((line, index) => (
          <p
            key={index}
            className="agent-line"
            style={{ color: line.colour }}
            dangerouslySetInnerHTML={{ __html: line.text }}
          />
        ))}

        {children ? <div className="agent-chart">{children}</div> : null}
      </div>
    </section>
  );
}
