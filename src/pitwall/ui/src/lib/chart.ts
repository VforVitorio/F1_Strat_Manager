/**
 * The ECharts primitives both windows share: the mount hook and the axis look.
 *
 * **Chart data never goes through React state.** That is the web form of
 * P3 finding A6 - the Qt dashboard rebuilt six cards, six highlighted text
 * areas and two whole charts ten times a second for content that changes
 * once a lap - applied before it happens rather than after.
 *
 * `animationDurationUpdate: 0` is the contract `useFirstPaintAnimation.ts`
 * already encodes in the webapp: the entrance sweep plays once, on the first
 * `setOption` for a new series, and every later update lands instantly. On a
 * screen fed ten times a second, the difference between that and animating
 * updates is the difference between polish and nausea.
 *
 * **Why the three colours live here and not in each feature.** They are the
 * axis pens `pace_chart.py` and `telemetry_panel.py` both build out of
 * `palette.BORDER_COLOR` and `palette.TEXT_SECONDARY`, so a copy per window
 * would be the sixth copy of the arcade palette in this repo - and copy
 * number five (the agents chart theme) shipped with no detector at all until
 * #876 went looking. `test_pitwall_tokens.py` counts them HERE, per site.
 *
 * --- WHERE TO CHANGE IF THE PALETTE MOVES ---
 * `src/arcade/palette.py` is the source. Then: this file, `styles/qt-base.css`,
 * `features/agents/AgentsWindow.tsx`'s boot literals, and the counts in
 * `tests/surfaces/test_pitwall_tokens.py`.
 */

import { useCallback, useEffect, useRef, useState } from "react";
import * as echarts from "echarts";

/** `palette.TEXT_SECONDARY` - axis names and tick labels. */
export const AXIS_TEXT = "#d1d5db";
/** `palette.BORDER_COLOR` - the axis line itself. */
export const AXIS_LINE = "#2d2d3a";
/** pyqtgraph's `showGrid(alpha=0.12)`, matched by eye against the Qt capture. */
export const SPLIT_LINE = "rgba(255,255,255,0.06)";
/**
 * `palette.TEXT_TERTIARY` - band 4's shared vertical cursor.
 *
 * Deliberately the dimmest text colour rather than an accent: the cursor
 * marks where the car is NOW on four charts at once, and anything brighter
 * competes with the four traces it is supposed to be annotating.
 */
export const CURSOR_LINE = "#9ca3af";

/**
 * Mount an ECharts instance on a div and push options into it imperatively.
 *
 * **The host is tracked as STATE behind a callback ref, not as a plain ref
 * with an empty dependency list.** A ref does not re-render, so a mount
 * effect keyed on `[]` runs exactly once for the life of the component - and
 * band 4's delta chart swaps its plot for a "single-driver mode" placeholder
 * and back, which unmounts and remounts the div underneath. After one such
 * swap the effect never re-ran, the instance pointed at a detached node, and
 * the chart was dead for the rest of the session with no error anywhere.
 */
export function useEChart(option: echarts.EChartsOption | null) {
  const [host, setHost] = useState<HTMLDivElement | null>(null);
  const chart = useRef<echarts.ECharts | null>(null);
  const ref = useCallback((node: HTMLDivElement | null) => setHost(node), []);

  useEffect(() => {
    if (!host) return;
    const instance = echarts.init(host, undefined, { renderer: "canvas" });
    chart.current = instance;
    // The headless smoke reads the COMPUTED axis extent and converts data
    // coordinates to pixels through this handle. Without it the only thing
    // a test can inspect is the option object it already knows we passed -
    // the mechanism - and the sprint-3 exit gate's whole lesson was that a
    // check written against the mechanism passes over a broken effect. An
    // ECharts instance is not reachable from outside the module otherwise.
    (host as HTMLDivElement & { __pitwallChart?: echarts.ECharts }).__pitwallChart = instance;
    const observer = new ResizeObserver(() => instance.resize());
    observer.observe(host);
    return () => {
      observer.disconnect();
      instance.dispose();
      if (chart.current === instance) chart.current = null;
    };
  }, [host]);

  // `host` is a dependency so the option is re-applied to a chart that was
  // just re-created. Without it a remounted plot came back blank until the
  // next tick happened to produce a new option object.
  useEffect(() => {
    if (!chart.current || !option) return;
    chart.current.setOption({ ...option, animationDurationUpdate: 0 }, { notMerge: true });
  }, [option, host]);

  return ref;
}

export interface AxisSpec {
  name?: string;
  nameGap?: number;
  /**
   * Fit the axis to the data. Without it an ECharts value axis always
   * includes 0, so a window over laps 12-27 rendered 0-30 and half the plot
   * was dead space. pyqtgraph autoranges on the data; this is that.
   *
   * Leave it off and pass `min`/`max` for a LOCKED axis, which is what the
   * telemetry charts want: a viewport that moves on every broadcast is
   * visually noisy and hides where on the lap the car actually is.
   */
  scale?: boolean;
  min?: number;
  max?: number;
}

/**
 * A value axis in the arcade's chart look, locked or autoranged.
 *
 * The return type is left inferred on purpose. ECharts types `xAxis` and
 * `yAxis` as two incompatible unions - they disagree on a phantom `mainType`
 * - so annotating either one makes the helper unusable on the other axis,
 * while a plain object literal satisfies both.
 */
export function valueAxis(spec: AxisSpec = {}) {
  // A locked axis does not label its own bounds. pyqtgraph labels the ticks
  // it chooses and leaves the range ends bare, so the Qt capture reads
  // "500 … 5000" on an axis locked to [0, 5220] and "0 … 100" on one locked
  // to [-5, 105]. ECharts labels the bounds unless told not to, which put a
  // "5,220" hard against the "5,000" tick and printed the -5/105 padding as
  // if it were data. Only for a locked axis: an autoranged one's bounds ARE
  // its data, and the AGENTS charts want them.
  const locked = spec.min !== undefined || spec.max !== undefined;
  return {
    type: "value" as const,
    name: spec.name,
    nameLocation: "middle" as const,
    nameGap: spec.nameGap,
    nameTextStyle: { color: AXIS_TEXT, fontSize: 10 },
    scale: spec.scale,
    min: spec.min,
    max: spec.max,
    axisLine: { lineStyle: { color: AXIS_LINE } },
    axisLabel: {
      color: AXIS_TEXT,
      fontSize: 10,
      showMinLabel: locked ? false : undefined,
      showMaxLabel: locked ? false : undefined,
    },
    splitLine: { lineStyle: { color: SPLIT_LINE } },
  };
}
