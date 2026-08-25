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
 * 97.7% of ticks. That reasoning is refuted: a constant extent decides where
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

/**
 * The lap axis, locked to a range the host computed.
 *
 * Built through `valueAxis` rather than spread onto `CHART_BASE.xAxis`,
 * because the bound-label suppression is decided INSIDE that helper from the
 * spec it is handed. Spreading `min`/`max` in afterwards locks the axis and
 * leaves the suppression off, which printed a computed bound of `12.5` as if
 * half a lap were a tick on an integer quantity.
 */
export function lapAxis(range: readonly [number, number] | null) {
  if (!range) return CHART_BASE.xAxis;
  return valueAxis({ name: "Lap", nameGap: 18, min: range[0], max: range[1] });
}

/** The lap-time axis, bounded to the values plotted rather than autoranged. */
export function secondsAxis(range: readonly [number, number]) {
  return valueAxis({ name: "Lap time (s)", nameGap: 34, min: range[0], max: range[1] });
}

/**
 * A thin vertical at the current lap, for both cards.
 *
 * SOLID and in the dimmest text colour. A dashed vertical already means a
 * compound boundary on the tyre chart, and the two charts previously shared
 * a quantity without sharing a landmark: "now" sat at 95% of one plot and
 * 47% of the other with nothing naming it on either.
 */
export function currentLapMark(lap: number | null | undefined, colour: string) {
  // `== null` on purpose, covering undefined too. A view built before this
  // field existed carries neither, and `undefined !== null` slipped a mark
  // with no axis position into ECharts, which throws where it renders.
  if (lap == null) return undefined;
  return {
    silent: true,
    symbol: "none" as const,
    label: { show: false },
    data: [{ xAxis: lap, lineStyle: { color: colour, width: 1, type: "solid" as const } }],
  };
}
