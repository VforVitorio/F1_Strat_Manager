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
function Tooltip({ html, anchor }: { html: string; anchor: DOMRect }) {
  const ref = useRef<HTMLSpanElement>(null);
  const [left, setLeft] = useState(anchor.left);

  useLayoutEffect(() => {
    const width = ref.current?.offsetWidth ?? 0;
    setLeft(Math.max(8, Math.min(anchor.left, window.innerWidth - width - 8)));
  }, [anchor]);

  return createPortal(
    <span
      ref={ref}
      className="agent-tooltip"
      style={{ top: anchor.top + 32, left }}
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
