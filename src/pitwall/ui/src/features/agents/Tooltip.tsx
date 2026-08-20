/**
 * The drill-down popup, and the one definition of how it is opened.
 *
 * Two surfaces show one: every agent console, and the band's WHY module. They
 * had better behave identically - the same placement rules, the same keyboard
 * contract - and this repo's dominant defect is two copies where one got a fix.
 *
 * **The popup renders OUTSIDE the window tree, and two attempts got that wrong
 * the same way.** An absolutely positioned popup is clipped by any scrolling
 * ancestor, and overflow to the LEFT of a scroll box cannot even be scrolled
 * to. The first fix put the scroll on the card; the second moved it to
 * `.agent-card-body` and the clip simply became the next ancestor up, which
 * amputated 130 px off every transcript line - and the check that was supposed
 * to catch it asserted the card's `overflow` and the popup's DOM placement,
 * the mechanism, never the rendered box.
 *
 * A portal ends the class rather than moving it: Qt's `QToolTip` is a
 * top-level window clipped by nothing, and `position: fixed` in the body is
 * the same thing.
 */

import { useLayoutEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";

import type { TooltipView } from "../../lib/agents";

/** Breathing room between the card and the popup, and from the viewport edge. */
const TOOLTIP_GAP = 10;

export function Tooltip({ view, anchor, id }: { view: TooltipView; anchor: DOMRect; id: string }) {
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

/**
 * Hover, focus, Escape - the whole opening contract, once.
 *
 * **Focus is not a nicety here.** What these popups carry is what the reasoning
 * tabs used to hold, and those were `<button>`s: reachable with Tab by anyone.
 * A mouse-only replacement would make the window strictly worse for a keyboard
 * user than the panel it retires.
 *
 * Returns the props to spread on whatever element is the target, plus the
 * anchor rect the popup positions itself against.
 */
export function useTooltipTarget(id: string, enabled: boolean) {
  const [anchor, setAnchor] = useState<DOMRect | null>(null);
  const open = (element: HTMLElement) => enabled && setAnchor(element.getBoundingClientRect());
  const close = () => setAnchor(null);

  return {
    anchor,
    props: {
      // Only a target that HAS something to show is a tab stop; empty stops
      // between the reader and the next panel are a cost, not a courtesy.
      tabIndex: enabled ? 0 : undefined,
      "aria-describedby": enabled && anchor ? id : undefined,
      onMouseEnter: (event: React.MouseEvent<HTMLElement>) => open(event.currentTarget),
      onMouseLeave: close,
      onFocus: (event: React.FocusEvent<HTMLElement>) => open(event.currentTarget),
      onBlur: close,
      onKeyDown: (event: React.KeyboardEvent<HTMLElement>) => {
        if (event.key === "Escape") close();
      },
    },
  };
}
