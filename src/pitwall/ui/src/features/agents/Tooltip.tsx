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

export function Tooltip({
  view,
  anchor,
  id,
  hold,
}: {
  view: TooltipView;
  anchor: DOMRect;
  id: string;
  /** Keeps the popup open while the pointer is inside it. */
  hold?: { onMouseEnter: () => void; onMouseLeave: () => void };
}) {
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
    <span ref={ref} id={id} role="tooltip" className="agent-tooltip" style={box} {...hold}>
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
  // The pointer's grace period between leaving the card and entering the popup,
  // which sits `TOOLTIP_GAP` away from it. Long enough to cross 10 px, short
  // enough that a popup left behind by a moving pointer does not linger.
  const closing = useRef<number | undefined>(undefined);

  const open = (element: HTMLElement) => {
    window.clearTimeout(closing.current);
    if (enabled) setAnchor(element.getBoundingClientRect());
  };
  const close = () => {
    window.clearTimeout(closing.current);
    setAnchor(null);
  };
  const closeSoon = () => {
    window.clearTimeout(closing.current);
    closing.current = window.setTimeout(() => setAnchor(null), 140);
  };

  /**
   * **The popup's own content had no input path to it.**
   *
   * It has `max-height: 22rem` and a visible scrollbar, and the stylesheet
   * argues the bar is the signal for a busy lap. Nothing could drive it. The
   * mouse could not: the popup is a portal positioned 10 px away, so crossing
   * the gap fired the card's `mouseleave` and unmounted it. The keyboard could
   * not either: focus sits on the card, the popup has no `tabIndex`, and Tab
   * moves focus, which closes it. So a RADIO tooltip on a busy lap - rows, then
   * the model-detail sections appended after them - kept its reasoning and its
   * nine `key = value` rows permanently below the fold.
   *
   * Two paths, one for each hand. The pointer gets the standard hover-intent
   * grace period, and the popup keeps itself open while the pointer is inside
   * it. The keyboard gets the arrows, forwarded from the card, so the reader
   * never has to reach an element the tab order does not visit.
   */
  const hold = {
    onMouseEnter: () => window.clearTimeout(closing.current),
    onMouseLeave: close,
  };

  const scrollPopup = (delta: number) => {
    const popup = document.getElementById(id);
    if (!popup) return false;
    const before = popup.scrollTop;
    popup.scrollTop = before + delta;
    return popup.scrollTop !== before;
  };

  return {
    anchor,
    /** Spread on the popup, so the pointer can enter it without closing it. */
    hold,
    props: {
      // Only a target that HAS something to show is a tab stop; empty stops
      // between the reader and the next panel are a cost, not a courtesy.
      tabIndex: enabled ? 0 : undefined,
      "aria-describedby": enabled && anchor ? id : undefined,
      onMouseEnter: (event: React.MouseEvent<HTMLElement>) => open(event.currentTarget),
      onMouseLeave: closeSoon,
      onFocus: (event: React.FocusEvent<HTMLElement>) => open(event.currentTarget),
      // Immediate, unlike the pointer's. Blur means the reader has moved on
      // deliberately, and a grace period here left the previous console's popup
      // on screen beside the next one's - two popups, and `aria-describedby`
      // pointing at whichever the DOM listed first.
      onBlur: close,
      onKeyDown: (event: React.KeyboardEvent<HTMLElement>) => {
        if (event.key === "Escape") {
          close();
          return;
        }
        // Only swallow the arrow when it actually moved the popup; otherwise
        // the reader loses the page scroll for nothing.
        const step = event.key === "ArrowDown" ? 48 : event.key === "ArrowUp" ? -48 : 0;
        if (step && scrollPopup(step)) event.preventDefault();
      },
    },
  };
}
