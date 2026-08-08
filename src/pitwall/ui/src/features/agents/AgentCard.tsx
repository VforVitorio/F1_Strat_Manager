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

import type { AgentCardView } from "../../lib/agents";

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
  return (
    <section className={idle ? "card agent-card is-idle" : "card agent-card"}>
      <header className="agent-card-head">
        <span className="agent-glyph" style={{ color: card.glyph_colour }}>
          {card.glyph}
        </span>
        <span className="agent-title">{title}</span>
      </header>

      {/* The whole card is the hover target, as in Qt, and nothing extra
          is drawn. An empty string means no tooltip, which is Qt's own
          convention for suppressing the popup. */}
      {card.tooltip ? (
        <span className="agent-tooltip" dangerouslySetInnerHTML={{ __html: card.tooltip }} />
      ) : null}

      {/* The scroll lives on this box, not on the card. An absolutely
          positioned popup is clipped by any positioned ancestor that
          scrolls, and overflow to the LEFT of a scroll container cannot
          even be scrolled to: measured, a 502 px transcript inside a
          375 px card lost the first 140 px of every line, unreachably.
          Qt uses a native QToolTip, which floats above everything. */}
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
