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

/**
 * No animation, for the same measured reason band 4 has none.
 *
 * `notMerge: true` makes every `setOption` look like a fresh series, so
 * ECharts uses the ENTRANCE duration rather than the update one, and
 * `animationDurationUpdate: 0` does not reach it. The host returns a new view
 * on every tick - **40 `setOption` calls in 4 seconds** - so the entrance is
 * restarted about ten times a second.
 *
 * Measured on the TIRE card, capturing the same element at increasing delays
 * after one view lands: it differs at 80, 150, 300 and 600 ms and only
 * settles at ~1200 ms. **At a 100 ms push cadence it therefore never finishes
 * once**, and the dashed cliff marker and compound boundaries redraw
 * permanently - which is what a viewer sees as a flicker.
 *
 * ⚠️ This block previously argued the opposite, that the two AGENTS charts
 * were "correct by accident" because their axis extent is constant across
 * 97.7 % of ticks. That reasoning is refuted: a constant extent decides where
 * the animation ENDS, not whether it gets there, and it never gets there.
 * Víctor reported the flicker; the measurement above is what settled it.
 * `animation: false` is also the 1:1 answer, since pyqtgraph does not animate.
 */
/** Axis and grid styling shared by both cards, from the Qt charts' own look. */
export const CHART_BASE: EChartsOption = {
  backgroundColor: "transparent",
  animation: false,
  animationDurationUpdate: 0,
  grid: { left: 44, right: 10, top: 10, bottom: 28, containLabel: false },
  xAxis: valueAxis({ name: "Lap", nameGap: 18, scale: true }),
  yAxis: valueAxis({ name: "Lap time (s)", nameGap: 34, scale: true }),
};
