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

import { useLayoutEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";

import type { AgentCardView, TooltipView } from "../../lib/agents";

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

function Tooltip({ view, anchor, id }: { view: TooltipView; anchor: DOMRect; id: string }) {
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
    // Right of the card by preference, left when there is no room, and
    // OUTSIDE IT VERTICALLY when neither side fits.
    //
    // **That third branch used to keep the popup at the card's own left edge,
    // which is on top of it.** It was survivable while every card was about a
    // third of the window; the decision band made RADIO span two columns, so
    // the room either side of a 984 px card is 492 px and a tooltip wider than
    // that lands on its own subject. It reached CI and not this machine,
    // because `width` is `max-content` and the runner has neither JetBrains
    // Mono nor the body font: 37,285 px2 of overlap there, zero here, from the
    // same code and the same fixture.
    const roomRight = window.innerWidth - anchor.right - TOOLTIP_GAP;
    const roomLeft = anchor.left - TOOLTIP_GAP;
    const onScreenTop = (wanted: number) =>
      Math.max(TOOLTIP_GAP, Math.min(wanted, window.innerHeight - height - TOOLTIP_GAP));

    let left: number;
    let top: number;
    if (roomRight >= width) {
      left = anchor.right + TOOLTIP_GAP;
      top = onScreenTop(anchor.top);
    } else if (roomLeft >= width) {
      left = anchor.left - TOOLTIP_GAP - width;
      top = onScreenTop(anchor.top);
    } else {
      left = Math.max(
        TOOLTIP_GAP,
        Math.min(anchor.left, window.innerWidth - width - TOOLTIP_GAP),
      );
      // Under the card if it fits there, over it if not. Preferring BELOW
      // rather than above keeps the reading order for the four cards that have
      // room under them; RADIO and RAG sit on the bottom row and take the
      // other branch.
      const roomBelow = window.innerHeight - anchor.bottom - TOOLTIP_GAP;
      const roomAbove = anchor.top - TOOLTIP_GAP;
      if (roomBelow >= height) top = anchor.bottom + TOOLTIP_GAP;
      else if (roomAbove >= height) top = anchor.top - TOOLTIP_GAP - height;
      else top = onScreenTop(anchor.top);
    }
    setBox({ left, top });
  }, [anchor, view]);

  // The markup is decided HERE, from data the host produced. It used to be a
  // string of Qt's restricted rich-text dialect pushed through
  // `dangerouslySetInnerHTML` - a dead toolkit's parser constraints shaping a
  // webview popup. The content still comes from one place, so only the
  // presentation below can drift from the Qt window's; the structure the host
  // returns is pinned by a test.
  return createPortal(
    <span ref={ref} id={id} role="tooltip" className="agent-tooltip" style={box}>
      {view.sections.map((section, index) => (
        <span className="tip-section" key={index}>
          <span className="tip-title">{section.title}</span>
          {section.rows.map((row, rowIndex) => (
            <span className="tip-row" key={rowIndex}>
              {row.lead ? <span className="tip-lead">{row.lead}</span> : null}
              <span className="tip-text">{row.text}</span>
            </span>
          ))}
        </span>
      ))}
      {view.footer ? <span className="tip-footer">{view.footer}</span> : null}
    </span>,
    document.body,
  );
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
  // The whole card is the hover target, as in Qt. `null` suppresses the
  // popup - not the empty string Qt used, because a falsy value that is also
  // a legitimate rendering is the sentinel shape this repo keeps paying for.
  const [anchor, setAnchor] = useState<DOMRect | null>(null);
  const tooltipId = `tip-${slot ?? title.toLowerCase()}`;

  /**
   * **The popup opens on FOCUS as well as on hover, and Escape closes it.**
   *
   * The reasoning tabs were `<button>`s: reachable with Tab, by anyone, and
   * that is the only reason the per-agent dumps they held were reachable at
   * all. Moving that content into a mouse-only popup would have made the
   * window strictly worse for a keyboard user than the panel it replaces -
   * zero of the six dumps against six - which is why the elevation spec's own
   * line about the keyboard pattern no longer being needed is wrong: the
   * WIDGET went away, the requirement did not.
   *
   * `onFocus`/`onBlur` rather than `onFocusCapture`: the card is the tab stop
   * and nothing inside it is focusable, so the events do not repeat.
   */
  const open = (element: HTMLElement) => card.tooltip && setAnchor(element.getBoundingClientRect());
  const close = () => setAnchor(null);

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
      // Only a card that HAS something to show is a tab stop; six empty stops
      // between the reader and the next panel is the cost of doing otherwise.
      tabIndex={card.tooltip ? 0 : undefined}
      aria-describedby={card.tooltip && anchor ? tooltipId : undefined}
      onMouseEnter={(event) => open(event.currentTarget)}
      onMouseLeave={close}
      onFocus={(event) => open(event.currentTarget)}
      onBlur={close}
      onKeyDown={(event) => {
        if (event.key === "Escape") close();
      }}
    >
      <header className="agent-card-head">
        <span className="agent-glyph" style={{ color: card.glyph_colour }}>
          {card.glyph}
        </span>
        <span className="agent-title">{title}</span>
      </header>

      {card.tooltip && anchor ? (
        <Tooltip view={card.tooltip} anchor={anchor} id={tooltipId} />
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
