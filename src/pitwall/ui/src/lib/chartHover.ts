/**
 * Where the pointer is on a chart's x axis, for the hover readouts (#999).
 *
 * One hook for all four ECharts surfaces on both windows. Every chart here is
 * read with a VERTICAL cut - six lanes at one distance, twenty cars at one lap,
 * a prediction against an actual at one lap - so the only quantity a readout
 * needs from the pointer is its position on the x axis, and the pixel to put a
 * cursor at.
 *
 * **Why this is not an ECharts `tooltip` with an `axisPointer`.** That is the
 * mechanism #999 proposed and it does not survive this window. `useEChart`
 * pushes every option with `notMerge: true`, which rebuilds the components, and
 * the tooltip goes with them. Measured on the real host with the pointer
 * PARKED: visible on 0 of 25 samples while the app pushed, 25 of 25 with
 * `setOption` frozen mid-session, 0 of 25 with it restored. It is re-created on
 * `mousemove` and destroyed by the next push, so it exists only while the mouse
 * is physically moving - and a reader parks the pointer to read a number.
 *
 * A moving-pointer probe sees it 14 times out of 14 and passes the broken
 * design, which is why the guards park.
 *
 * `alwaysShowContent` plus a re-dispatched `showTip` after every push does
 * reach 25 of 25, at the price of tracking the hovered index in our own state
 * and firing an action five times a second on the data path. That is the whole
 * of the hover state, owned here, with ECharts drawing the box - so the box may
 * as well be ours and land where each panel wants it.
 */

import { useCallback, useState, type PointerEvent, type RefObject } from "react";
import type * as echarts from "echarts";

/** The pointer, resolved onto one chart's x axis. */
export interface ChartHover {
  /** The x axis value under the pointer, in that axis's own units. */
  dataX: number;
  /** Where that sits on the host element, in px from its left edge. */
  pixelX: number;
  /**
   * The host's own width, so a readout can decide which side of the cursor to
   * open on.
   *
   * Reported rather than measured again by each caller: it is already in the
   * `getBoundingClientRect` this hook takes on every move, and a panel that
   * measured it separately would be reading a different box on the tick a
   * resize lands.
   */
  hostWidth: number;
}

/**
 * Track the pointer over a chart and report its x axis position.
 *
 * `domain` is the axis's locked `[min, max]`, and it is what keeps the readout
 * out of the axis gutter: `convertFromPixel` answers happily for a pixel
 * outside the grid, so the value it returns there is out of domain rather than
 * absent. Comparing against the range the caller locked is the guard, and it
 * costs nothing because the caller already owns both numbers. A null domain
 * means the chart has no range yet, so there is nothing to hover.
 *
 * Deliberately NOT `containPixel`: on the six-grid trace stack the label rows
 * and the gaps between lanes are outside every grid, so containment on the
 * pointer's own y would drop the cursor five times on the way down a panel that
 * exists to be read with ONE cut. The cut is horizontal; y is not part of it.
 *
 * The returned handlers go on the same element as the chart's ref. The state
 * lives here, in the chart's own component, so it dies with the unmount - a
 * DATA tab switch disposes the instance and creates a new one, and hover state
 * lifted any higher would outlive it and paint a stale readout onto a fresh
 * chart.
 */
export function useChartHover(
  chart: RefObject<echarts.ECharts | null>,
  domain: readonly [number, number] | null,
  gridIndex = 0,
): [ChartHover | null, {
  onPointerMove: (event: PointerEvent<HTMLDivElement>) => void;
  onPointerLeave: () => void;
}] {
  const [hover, setHover] = useState<ChartHover | null>(null);

  const onPointerMove = useCallback(
    (event: PointerEvent<HTMLDivElement>) => {
      const instance = chart.current;
      if (!instance || !domain) {
        setHover(null);
        return;
      }
      const box = event.currentTarget.getBoundingClientRect();
      const pixelX = event.clientX - box.left;
      const converted = instance.convertFromPixel({ gridIndex }, [
        pixelX,
        event.clientY - box.top,
      ]);
      const dataX = Array.isArray(converted) ? converted[0] : null;
      if (dataX === null || !Number.isFinite(dataX) || dataX < domain[0] || dataX > domain[1]) {
        setHover(null);
        return;
      }
      setHover({ dataX, pixelX, hostWidth: box.width });
    },
    // `domain` is a tuple rebuilt per render by most callers, so it is spread
    // into the dependency list by value. Depending on the array identity would
    // rebuild the handler on every tick.
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [chart, gridIndex, domain?.[0], domain?.[1]],
  );

  const onPointerLeave = useCallback(() => setHover(null), []);

  return [hover, { onPointerMove, onPointerLeave }];
}
