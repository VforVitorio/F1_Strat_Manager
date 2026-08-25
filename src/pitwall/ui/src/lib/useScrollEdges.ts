/**
 * The scroll-edge mask, shared by every card whose body can overflow.
 *
 * Lifted out of `AgentCard` when the CONTINGENCIES card needed the same
 * affordance. A second copy would have been the shape this repository produces
 * most: one gets a fix and its twin does not.
 */

import { useCallback, useEffect, useState } from "react";

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
export function useScrollEdges(body: HTMLElement | null) {
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
