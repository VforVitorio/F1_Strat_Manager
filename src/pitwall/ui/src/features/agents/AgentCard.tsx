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
 * does it: the messages are free text off the NLP pipeline, so a `<` in a
 * transcript was a tag until the exit gate said so.
 * **The tooltip no longer does**: sprint 8 turned the two tooltip
 * formatters into structured data, so the popup below is built from
 * fields rather than parsed out of a dead toolkit's dialect. The pills
 * are the same debt one layer down and are filed separately.
 */

import type { AgentCardView } from "../../lib/agents";
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


  return (
    <section
      className={
        [
          "card",
          "agent-card",
          idle ? "is-idle" : "",
          slot ? `slot-${slot}` : "",
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
      <div className="agent-card-body">
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
