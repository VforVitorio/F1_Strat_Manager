/**
 * Predicted lap time against actual, with the P10-P90 credible band.
 *
 * 1:1 with `pace_chart.py`: solid blue actual, dashed purple predicted,
 * translucent purple band. What may be plotted was decided host side —
 * the 30-200 s sanity window, and three independent series so a lap that
 * has an actual but no prediction does not drag the dashed line to zero.
 *
 * The band is ECharts' stacked-area idiom: an invisible line at P10 and
 * a filled one carrying the height to P90. `stack` needs the height, not
 * the absolute value, which is the one place this file does arithmetic.
 */

import { useMemo } from "react";
import type { EChartsOption } from "echarts";
import type { PaceSeries } from "../../lib/agents";
import { useEChart } from "../../lib/chart";
import { CHART_BASE, currentLapMark, lapAxis } from "./useEChart";

export function PaceChart({ series }: { series: PaceSeries }) {
  /**
   * **The prediction stopped and the actual did not.**
   *
   * A tick with no `per_agent` block still carries the lap time, so the solid
   * line keeps advancing while the dashed one and its band stay where the last
   * prediction was. Nothing on the chart said so: the reader saw two lines,
   * one of which had quietly become history.
   *
   * Dimmed AND labelled, because dimming alone is ambiguous on a chart that
   * already draws a translucent band, and a label alone is easy to miss.
   */
  const stale =
    series.prediction_lap !== null &&
    series.current_lap !== null &&
    series.prediction_lap < series.current_lap;

  const option = useMemo<EChartsOption>(
    () => ({
      ...CHART_BASE,
      graphic: stale
        ? [
            {
              type: "text",
              right: 4,
              top: 2,
              silent: true,
              style: {
                text: `prediction to L${series.prediction_lap}`,
                fill: series.cursor_colour,
                font: "10px sans-serif",
              },
            },
          ]
        : [],
      // The tyre chart's lap axis, borrowed. Two charts of the same quantity
      // side by side used to autorange independently, so lap 23 sat at 95 %
      // of this plot and 47 % of its neighbour — a comparison a reader
      // cannot make, on the one screen where they are meant to make it.
      xAxis: lapAxis(series.x_range),
      series: [
        {
          type: "line",
          name: "p10",
          stack: "band",
          data: series.band.map(([lap, low]) => [lap, low]),
          lineStyle: { opacity: 0 },
          itemStyle: { opacity: 0 },
          symbol: "none",
          silent: true,
        },
        {
          type: "line",
          name: "band",
          stack: "band",
          data: series.band.map(([lap, low, high]) => [lap, high - low]),
          lineStyle: { opacity: 0 },
          areaStyle: { color: series.band_colour, opacity: stale ? 0.07 : 0.2 },
          symbol: "none",
          silent: true,
        },
        {
          type: "line",
          name: "predicted",
          data: series.pred,
          lineStyle: {
            color: series.pred_colour,
            width: 2,
            type: "dashed",
            opacity: stale ? 0.35 : 1,
          },
          itemStyle: { color: series.pred_colour, opacity: stale ? 0.35 : 1 },
          symbol: "none",
        },
        {
          type: "line",
          name: "actual",
          data: series.actual,
          lineStyle: { color: series.actual_colour, width: 2 },
          itemStyle: { color: series.actual_colour },
          symbol: "none",
          markLine: currentLapMark(series.current_lap, series.cursor_colour),
        },
      ],
    }),
    [series, stale],
  );

  return <div className="chart" ref={useEChart(option)} />;
}
