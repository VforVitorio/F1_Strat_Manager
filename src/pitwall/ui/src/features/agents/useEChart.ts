/**
 * Mount one ECharts instance on a div and push options into it imperatively.
 *
 * **Chart data never goes through React state.** That is the web form of
 * P3 finding A6 — the Qt dashboard rebuilt six cards, six highlighted
 * text areas and two whole charts ten times a second for content that
 * changes once a lap — applied before it happens rather than after.
 *
 * `animationDurationUpdate: 0` is the contract
 * `useFirstPaintAnimation.ts` already encodes in the webapp: the entrance
 * sweep plays once, on the first `setOption` for a new series, and every
 * later update lands instantly. On a screen fed ten times a second, the
 * difference between that and animating updates is the difference between
 * polish and nausea.
 */

import { useEffect, useRef } from "react";
import * as echarts from "echarts";

export function useEChart(option: echarts.EChartsOption | null) {
  const host = useRef<HTMLDivElement | null>(null);
  const chart = useRef<echarts.ECharts | null>(null);

  useEffect(() => {
    if (!host.current) return;
    chart.current = echarts.init(host.current, undefined, { renderer: "canvas" });
    const resize = () => chart.current?.resize();
    const observer = new ResizeObserver(resize);
    observer.observe(host.current);
    return () => {
      observer.disconnect();
      chart.current?.dispose();
      chart.current = null;
    };
  }, []);

  useEffect(() => {
    if (!chart.current || !option) return;
    chart.current.setOption({ ...option, animationDurationUpdate: 0 }, { notMerge: true });
  }, [option]);

  return host;
}

/** Axis and grid styling shared by both cards, from the Qt charts' own look. */
export const CHART_BASE: echarts.EChartsOption = {
  backgroundColor: "transparent",
  animationDurationUpdate: 0,
  grid: { left: 44, right: 10, top: 10, bottom: 28, containLabel: false },
  xAxis: {
    type: "value",
    // Without this an ECharts value axis always includes 0, so a window
    // over laps 12-27 rendered 0-30 and half the plot was dead space.
    // pyqtgraph autoranges on the data; this is that.
    scale: true,
    name: "Lap",
    nameLocation: "middle",
    nameGap: 18,
    nameTextStyle: { color: "#d1d5db", fontSize: 10 },
    axisLine: { lineStyle: { color: "#2d2d3a" } },
    axisLabel: { color: "#d1d5db", fontSize: 10 },
    splitLine: { lineStyle: { color: "rgba(255,255,255,0.06)" } },
  },
  yAxis: {
    type: "value",
    name: "Lap time (s)",
    nameLocation: "middle",
    nameGap: 34,
    nameTextStyle: { color: "#d1d5db", fontSize: 10 },
    scale: true,
    axisLine: { lineStyle: { color: "#2d2d3a" } },
    axisLabel: { color: "#d1d5db", fontSize: 10 },
    splitLine: { lineStyle: { color: "rgba(255,255,255,0.06)" } },
  },
};
