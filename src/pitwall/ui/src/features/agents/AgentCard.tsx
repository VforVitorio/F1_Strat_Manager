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

import { useCallback, useEffect, useState } from "react";

import type { AgentCardView } from "../../lib/agents";
import { Tooltip, useTooltipTarget } from "./Tooltip";

/**
 * Which edges of a scrolling body have content beyond them.
 *
 * **The card body has always been `overflow: auto` and never said so.**
 * `qt-base.css` hides every scrollbar, deliberately, and its own comment ends
 * by admitting the cost: nothing hints that a card has more below. Measured at
 * the 720p client the product opens on, that costs the PACE and TIRE cards
 * 51 px each, which is their entire Lap axis, and cuts a line through the
 * middle of its glyphs on SITUATION and PIT. The content is wheel-reachable
 * the whole time; it is the absence of any signal that makes it lost.
 *
 * Measured rather than keyed on a breakpoint, so a card that happens to fit
 * shows nothing, and the same mask the pace grid uses so the two panels speak
 * one visual language.
 */
function useScrollEdges(body: HTMLElement | null) {
  const [edges, setEdges] = useState({ above: false, below: false });

  const measure = useCallback(() => {
    if (!body) return;
    const overflows = body.scrollHeight - body.clientHeight > 1;
    setEdges({
      above: overflows && body.scrollTop > 1,
      below: overflows && body.scrollTop < body.scrollHeight - body.clientHeight - 1,
    });
  }, [body]);

  useEffect(() => {
    if (!body) return;
    measure();
    // The card resizes with the window and its CONTENT changes every lap, so
    // both have to be watched: a card that grew a line is as much a change as
    // a window that shrank.
    const observer = new ResizeObserver(measure);
    observer.observe(body);
    for (const child of body.children) observer.observe(child);
    return () => observer.disconnect();
  }, [body, measure]);

  return { edges, measure };
}

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
