/**
 * One sub-agent card, shared by all six.
 *
 * A dumb view, like the Qt `AgentCard` it replaces: it does not know
 * which agent it shows. Title and content arrive from the host, which
 * produced them with the Qt window's own formatters.
 *
 * `dangerouslySetInnerHTML` is deliberate and bounded. The headline and
 * the body lines can carry the compound pill and the flag chips, which
 * are HTML spans built in `src/arcade/palette.py`, and every free-text
 * field inside them is escaped there. It is Qt's restricted rich-text
 * dialect (`<b>`, `<br>`, `&nbsp;` and those spans), which is debt sprint
 * 8 unpicks - not a licence to inject arbitrary markup here.
 */

import { useLayoutEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";

import type { AgentCardView } from "../../lib/agents";

/**
 * The popup, rendered OUTSIDE the window tree.
 *
 * **Two attempts got this wrong before, the same way.** An absolutely
 * positioned popup is clipped by any scrolling ancestor, and overflow to
 * the LEFT of a scroll box cannot even be scrolled to. The first fix put
 * the scroll on the card; the second moved the scroll to
 * `.agent-card-body` and the clip simply became the next ancestor up —
 * `.agents-right`, which scrolls because Qt clips that column too. At
 * 1320x900 it amputated 130 px off every transcript line, and the check
 * that was supposed to catch it asserted the card's `overflow` and the
 * popup's DOM placement — the mechanism, never the rendered box.
 *
 * A portal ends the class rather than moving it: Qt's `QToolTip` is a
 * top-level window clipped by nothing, and `position: fixed` in the body
 * is the same thing. The clamp keeps it on screen near either edge.
 */
/** Breathing room between the card and the popup, and from the viewport edge. */
const TOOLTIP_GAP = 10;

function Tooltip({ html, anchor }: { html: string; anchor: DOMRect }) {
  const ref = useRef<HTMLSpanElement>(null);
  const [box, setBox] = useState({ left: anchor.left, top: anchor.top + 32 });

  useLayoutEffect(() => {
    const node = ref.current;
    if (!node) return;
    const width = node.offsetWidth;
    const height = node.offsetHeight;

    // **BESIDE the card, never on top of it.** The popup used to open at the
    // card's own left edge, 32 px down, which covered the thing you hovered
    // to read: measured at 68,877 px² of overlap on a 216 px card, so the
    // card was blanketed and the popup spilled below it. A tooltip that
    // hides its own subject is worse than no tooltip.
    //
    // Right of the card by preference, left when there is no room, and only
    // then below - which is the old behaviour kept as the last resort for a
    // card that is wider than the space either side of it.
    const roomRight = window.innerWidth - anchor.right - TOOLTIP_GAP;
    const roomLeft = anchor.left - TOOLTIP_GAP;
    let left: number;
    if (roomRight >= width) left = anchor.right + TOOLTIP_GAP;
    else if (roomLeft >= width) left = anchor.left - TOOLTIP_GAP - width;
    else left = Math.max(TOOLTIP_GAP, Math.min(anchor.left, window.innerWidth - width - TOOLTIP_GAP));

    // Vertically aligned with the card's top, then pulled up only as far as
    // it must be to stay on screen.
    const top = Math.max(
      TOOLTIP_GAP,
      Math.min(anchor.top, window.innerHeight - height - TOOLTIP_GAP),
    );
    setBox({ left, top });
  }, [anchor, html]);

  return createPortal(
    <span
      ref={ref}
      className="agent-tooltip"
      style={box}
      dangerouslySetInnerHTML={{ __html: html }}
    />,
    document.body,
  );
}

export function AgentCard({
  title,
  card,
  children,
}: {
  title: string;
  card: AgentCardView;
  children?: React.ReactNode;
}) {
  const idle = card.status === "IDLE";
  // The whole card is the hover target, as in Qt, and an empty tooltip
  // string is Qt's own convention for suppressing the popup.
  const [anchor, setAnchor] = useState<DOMRect | null>(null);

  return (
    <section
      className={idle ? "card agent-card is-idle" : "card agent-card"}
      onMouseEnter={(event) => card.tooltip && setAnchor(event.currentTarget.getBoundingClientRect())}
      onMouseLeave={() => setAnchor(null)}
    >
      <header className="agent-card-head">
        <span className="agent-glyph" style={{ color: card.glyph_colour }}>
          {card.glyph}
        </span>
        <span className="agent-title">{title}</span>
      </header>

      {card.tooltip && anchor ? <Tooltip html={card.tooltip} anchor={anchor} /> : null}

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
