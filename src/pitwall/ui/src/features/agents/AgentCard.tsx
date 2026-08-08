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
        {card.tooltip ? <Tooltip html={card.tooltip} /> : null}
      </header>

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
    </section>
  );
}

/**
 * The hover transcript Qt shows as a tooltip.
 *
 * A real element rather than the `title` attribute: the content is rich
 * text, and `title` would print the tags. HTML has no 300 ms delay and no
 * fixed width, so the one thing that improves for free here is that the
 * whole lap fits.
 */
function Tooltip({ html }: { html: string }) {
  return (
    <span className="agent-tooltip-host" aria-label="full transcript">
      i<span className="agent-tooltip" dangerouslySetInnerHTML={{ __html: html }} />
    </span>
  );
}
