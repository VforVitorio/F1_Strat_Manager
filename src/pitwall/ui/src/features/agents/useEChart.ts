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
 * These two charts still ANIMATE, unlike band 4's, and the reason recorded
 * for that was wrong.
 *
 * The claim was "they update once a lap, so the entrance completes". Measured:
 * **40 `setOption` calls in 4 seconds** - the host returns a fresh view on
 * every tick, so these redraw at the same ~10 Hz band 4 does, and their
 * entrance animation is restarted just as often. The effect is nonetheless
 * correct, by accident: their axis extent is constant across 97.7 % of ticks,
 * so a restarted grow-from-the-left animation lands pixel-identical and
 * nothing flickers. Band 4's is not - its X axis is a locked 0-5220 that the
 * data does not fill, so the same restart left the delta baseline a stub.
 *
 * Written down because the next person to touch this will reach for the same
 * false reason. If these charts ever gain a moving extent, they need
 * `animation: false` too.
 */
/** Axis and grid styling shared by both cards, from the Qt charts' own look. */
export const CHART_BASE: EChartsOption = {
  backgroundColor: "transparent",
  animationDurationUpdate: 0,
  grid: { left: 44, right: 10, top: 10, bottom: 28, containLabel: false },
  xAxis: valueAxis({ name: "Lap", nameGap: 18, scale: true }),
  yAxis: valueAxis({ name: "Lap time (s)", nameGap: 34, scale: true }),
};
