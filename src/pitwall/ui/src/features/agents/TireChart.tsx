/**
 * Lap time over laps, coloured per stint, with the cliff projection.
 *
 * 1:1 with `tire_chart.py`. Three things it gets from the host and does
 * not re-derive:
 *
 * - **one series per stint, not one line coloured by compound.** A single
 *   line would be drawn through the in-lap and the out-lap, which are
 *   neither the same length as each other nor as a racing lap. The break
 *   IS the compound change.
 * - the 3-lap centred rolling mean behind the dashed overlay;
 * - the cliff band at `[lap + p10, lap + p90]` and the marker at
 *   `lap + p50`, suppressed whole outside the sane horizon because the
 *   TCN emits tens of thousands of laps on the first laps of a stint.
 */

import { useMemo } from "react";
import type { EChartsOption } from "echarts";
import type { TireSeries } from "../../lib/agents";
import { useEChart } from "../../lib/chart";
import { CHART_BASE, lapAxis, secondsAxis } from "./useEChart";

export function TireChart({ series }: { series: TireSeries }) {
  const option = useMemo<EChartsOption>(() => {
    const stints = series.stints.map((stint) => ({
      type: "line" as const,
      name: stint.compound,
      data: stint.points,
      lineStyle: { color: stint.colour, width: 2 },
      itemStyle: { color: stint.colour },
      symbol: "circle" as const,
      symbolSize: 4,
    }));

    // The band and the median live on the FIRST series as marks, which is
    // how ECharts attaches geometry that is not itself data.
    const cliff = series.cliff;
    const markArea =
      cliff && cliff.lo !== null && cliff.hi !== null
        ? {
            silent: true,
            itemStyle: { color: series.cliff_colour, opacity: 0.16 },
            // A 2D mark area is a two-element tuple, and ECharts types it
            // as exactly that; an inferred array is one element short.
            data: [[{ xAxis: cliff.lo }, { xAxis: cliff.hi }] as [object, object]],
          }
        : undefined;
    // The median marker and one faint dashed vertical per compound change
    // share a markLine, because each carries its own lineStyle.
    const marks = [
      ...(cliff && cliff.p50 !== null
        ? [
            {
              xAxis: cliff.p50,
              lineStyle: { color: series.cliff_colour, width: 2, type: "dashed" as const },
            },
          ]
        : []),
      ...series.boundaries.map((lap) => ({
        xAxis: lap,
        lineStyle: {
          color: series.boundary_colour,
          opacity: series.boundary_opacity,
          width: 1,
          type: "dashed" as const,
        },
      })),
      // Where the car is now. Solid, because dashed is already taken twice
      // over on this chart - the cliff median and the compound boundaries.
      ...(series.current_lap == null
        ? []
        : [
            {
              xAxis: series.current_lap,
              lineStyle: { color: series.cursor_colour, width: 1, type: "solid" as const },
            },
          ]),
    ];
    const markLine = marks.length
      ? { silent: true, symbol: "none" as const, label: { show: false }, data: marks }
      : undefined;

    const trend = {
      type: "line" as const,
      name: "trend",
      data: series.trend,
      lineStyle: { color: series.trend_colour, width: 1, type: "dashed" as const, opacity: 0.6 },
      symbol: "none" as const,
      markArea,
      markLine,
    };

    return {
      ...CHART_BASE,
      // Through `lapAxis`, not spread onto the autoranged base. Spreading
      // `min`/`max` locks the axis but leaves the base's own bound-label
      // suppression switched off, which is how a lap axis running to a
      // computed 12.5 printed "12.5" as though half a lap were a tick.
      xAxis: lapAxis(series.x_range),
      // Bounded to the laps plotted. Autoranged against a cliff band that
      // reaches 140 s, the trace at 81.2 s had four pixels of a 150 px plot.
      yAxis: series.y_range ? secondsAxis(series.y_range) : CHART_BASE.yAxis,
      series: [trend, ...stints],
    };
  }, [series]);

  return <div className="chart" ref={useEChart(option)} />;
}
