/**
 * The axis frame the two AGENTS cards share: laps across, seconds up.
 *
 * The mount hook and the three axis colours moved to `lib/chart.ts` when the
 * DATA window needed the same look - one palette site rather than two, which
 * is the whole reason `driver_colors` rides on the wire. What stays here is
 * the part that is genuinely about THESE charts: the axis names, the gaps
 * their labels need, and autorange rather than a locked viewport.
 */

import type { EChartsOption } from "echarts";
import { valueAxis } from "../../lib/chart";

/** Axis and grid styling shared by both cards, from the Qt charts' own look. */
export const CHART_BASE: EChartsOption = {
  backgroundColor: "transparent",
  animationDurationUpdate: 0,
  grid: { left: 44, right: 10, top: 10, bottom: 28, containLabel: false },
  xAxis: valueAxis({ name: "Lap", nameGap: 18, scale: true }),
  yAxis: valueAxis({ name: "Lap time (s)", nameGap: 34, scale: true }),
};
