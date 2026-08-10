/**
 * One cell of band 4's 2x2 grid: a title row over a locked-axis plot.
 *
 * Both axes are LOCKED, which is the whole design of the Qt panel this
 * ports and not an oversight: X is `[0, circuit_length]` and Y is fixed per
 * metric, so only the lines move between updates. A viewport that
 * autoranges on every broadcast is visually noisy and hides where on the
 * lap the car actually is - and this surface is fed ten times a second.
 *
 * The rival is dashed on every chart because it is BROADCAST-TIER data:
 * real and public, but the coarse low-rate channel every team sees, not the
 * pit-wall-grade telemetry the own car has
 * (`PITWALL_REALISM_AND_TELEMETRY_SURFACE.md` 2.2). The Qt window rendered
 * it unlabelled and that document names the omission; the legend chip says
 * so here.
 */

import { useMemo } from "react";
import type { EChartsOption } from "echarts";
import { CURSOR_LINE, useEChart, valueAxis } from "../../lib/chart";

export interface TraceChartProps {
  title: string;
  subtitle: string;
  /** The own car's colour on THIS chart - the Qt panel varies it per metric. */
  mainColour: string;
  rivalColour: string;
  yRange: [number, number];
  /** Locked X maximum: the circuit length, or the fallback until one arrives. */
  xMax: number;
  mainCode: string;
  rivalCode: string | null;
  main: [number, number][];
  rival: [number, number][];
  /**
   * Draw the own car as a flat reference at y=0 instead of a series.
   * The delta chart's convention: main IS the baseline, the rival trace is
   * the gap to it.
   */
  mainAsZeroLine?: boolean;
  /** Where the car is on the lap right now, in metres. Null before the first tick. */
  cursorX: number | null;
  /** Replaces the plot entirely when set - the Qt panel's "single-driver mode". */
  placeholder?: string | null;
}

export function TraceChart(props: TraceChartProps) {
  const {
    title,
    subtitle,
    mainColour,
    rivalColour,
    yRange,
    xMax,
    mainCode,
    rivalCode,
    main,
    rival,
    mainAsZeroLine = false,
    cursorX,
    placeholder = null,
  } = props;

  const option = useMemo<EChartsOption>(() => {
    // Both reference lines hang off the first series, which is why that
    // series exists even on the delta chart where it plots nothing.
    const marks: object[] = [];
    if (mainAsZeroLine) {
      marks.push({ yAxis: 0, lineStyle: { color: mainColour, width: 2, type: "solid" } });
    }
    if (cursorX !== null) {
      marks.push({ xAxis: cursorX, lineStyle: { color: CURSOR_LINE, width: 1, type: "dashed" } });
    }

    return {
      backgroundColor: "transparent",
      // **No animation at all, and this is the 1:1 answer as well as the
      // correct one.** `useEChart` sets `animationDurationUpdate: 0`, but
      // `notMerge: true` makes every `setOption` look like a fresh series, so
      // ECharts uses the ENTRANCE duration instead - ~1 s of growing from the
      // left, restarted ten times a second and never once completing.
      // Measured on a real payload: the delta chart's zero baseline reached
      // 1328 m of a 5220 m axis at +250 ms and only reached 5214 m after four
      // seconds of silence. On a live screen it is permanently a stub, and it
      // is the line every value on that chart is read against. pyqtgraph's
      // `setData` is instantaneous, and `pg.InfiniteLine` is infinite the
      // moment it is drawn. The architecture's "animate the entrance, never
      // the update" assumes a surface with a separable entrance; at 10 Hz
      // there is not one.
      animation: false,
      animationDurationUpdate: 0,
      grid: { left: 44, right: 12, top: 8, bottom: 36, containLabel: false },
      xAxis: valueAxis({ name: "Distance (m)", nameGap: 20, min: 0, max: xMax }),
      yAxis: valueAxis({ min: yRange[0], max: yRange[1] }),
      series: [
        {
          type: "line",
          name: "main",
          data: mainAsZeroLine ? [] : main,
          lineStyle: { color: mainColour, width: 2 },
          itemStyle: { color: mainColour },
          symbol: "none",
          markLine: marks.length
            ? { silent: true, symbol: "none", label: { show: false }, data: marks }
            : undefined,
        },
        {
          type: "line",
          name: "rival",
          data: rivalCode ? rival : [],
          lineStyle: { color: rivalColour, width: 2, type: "dashed" },
          itemStyle: { color: rivalColour },
          symbol: "none",
        },
      ],
    };
  }, [main, rival, mainColour, rivalColour, yRange, xMax, mainAsZeroLine, cursorX, rivalCode]);

  const host = useEChart(option);

  return (
    <div className="trace-cell">
      <div className="trace-title-row">
        <span className="trace-title">{title}</span>
        <span className="trace-subtitle">{subtitle}</span>
        <span className="trace-spacer" />
        <span className="trace-legend" style={{ color: mainColour }}>
          MAIN · {mainCode || "—"}
        </span>
        {rivalCode ? (
          <span className="trace-legend" style={{ color: rivalColour }} title="broadcast tier">
            RIVAL · {rivalCode} <span className="trace-tier">BROADCAST</span>
          </span>
        ) : null}
      </div>
      {placeholder ? (
        <div className="trace-placeholder">{placeholder}</div>
      ) : (
        <div className="trace-plot" ref={host} />
      )}
    </div>
  );
}
