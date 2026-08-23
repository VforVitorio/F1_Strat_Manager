/**
 * Band 4: the own car's lap as SIX lanes over ONE distance axis.
 *
 * **This is the shape the agreed layout drawing assigned to this sprint, and the
 * 2x2 it replaces is the one that drawing calls wrong.** The memory holding it
 * (`project_pitwall_data_layout`, agreed with Victor 2026-08-13) lists under "What
 * sprint 5 did NOT build": *"the traces stacked on ONE x axis with a shared cursor
 * (today they are the Qt 2x2, which the research calls the wrong shape)"* -> sprint
 * 9. It also answers the obvious objection in advance: *"The traces are not
 * oversized; the right column is doing one job with room for two."*
 *
 * Full spec, with the research behind the lane order and every height:
 * `documents/research/PITWALL_TRACES_SPACE_SPEC.md`.
 *
 * **What the stack buys, in the same box.** The 2x2 spent 142 px of its 666 on
 * chrome - two title rows, two axis bands, two grid tops, one grid gap - because
 * every cell repeated the whole apparatus. One shared axis spends 64. That
 * difference plus the four original lanes giving up height pays for two channels
 * the wire was already carrying unread, and the drawing area goes from 206 px wide
 * per channel to 477: the panel exists to be read with a vertical cut at a point of
 * track, and it now has 2.3x the resolution to make that cut in.
 *
 *     +-------------------------------------------------+
 *     | SPEED km/h                                287 | 145
 *     | ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~|~~~~~~~~~~~~~~ |
 *     | D TIME s                                +0.42 | 145
 *     | -----------------0--------------|~~~~~~~~~~~~~ |
 *     | THROTTLE %                                100 |  96
 *     | BRAKE %                                     0 |  96
 *     | GEAR                                        7 |  82
 *     | DRS                                    CLOSED |  38
 *     |  0      1k      2k     Distance (m)            |  34
 *     +-------------------------------------------------+
 *       the | is ONE cursor div, lane 1's top to lane 6's bottom
 *
 * The costs, stated rather than discovered: each original channel loses drawing
 * height (262 -> 145 for speed and delta, 262 -> 96 for the pedals). Lanes of
 * 90-150 px are the stacked convention, and the trade is what makes one cursor
 * legible across six channels instead of four charts a reader crosses by eye.
 */

import { useCallback, useMemo, useRef, useState } from "react";
import type { EChartsOption } from "echarts";

import { CURSOR_LINE, useEChart, valueAxis } from "../../lib/chart";
import type { SortedTrace } from "./traceBuffer";

/**
 * One lane: its label, its locked y range, its own-car colour, and its share of the
 * stack's height.
 *
 * The weights are the spec's, and they are RELATIVE on purpose: the same table has
 * to land on 666 px at the 1485 client and 430 at 1265, so a lane cannot carry a
 * pixel constant. Speed and delta are the channels a strategist reads a value off;
 * the pedals are read as shapes; gear is a staircase; DRS is a bit.
 */
interface Lane {
  key: "speed" | "delta" | "throttle" | "brake" | "gear" | "drs";
  label: string;
  unit: string;
  /** Locked, exactly as the four Qt ranges were. */
  range: [number, number];
  colour: string;
  weight: number;
  /** A staircase, not a curve: the value holds until the next sample. */
  step?: boolean;
  /**
   * Which y values print a label. Omitted means "whatever the scale picks".
   *
   * **A locked range picks its labels from the range, never from the room.** THROTTLE
   * and BRAKE printed `-5 0 20 40 60 80 100 105`, so their closest pair - the `-5`
   * padding value and `0` - landed **2.1 px apart in a 46 px lane under a 10 px
   * font** at the 1080p client, and GEAR's closest pair 4.1 px apart in 37 px. The
   * digits overlapped into a grey column.
   *
   * And it was never a narrow-client defect: the same pair is **3.8 px apart at
   * 1485x833**, on the screenshot this panel was signed off against. The padding
   * values are the worst offenders and the least worth a row - `-5` and `105` exist
   * only to keep the trace off the frame.
   *
   * An allow-list rather than a tick `interval`, because `interval` counts from the
   * scale's own first tick and would print `-5, 45, 95`: right arithmetic, and
   * numbers nobody means.
   */
  labels?: number[];
  /** How the live value reads in the label row. */
  readout: (value: number) => string;
}

/** palette.INFO */
const INFO = "#3b82f6";
/** palette.SUCCESS */
const SUCCESS = "#10b981";
/** palette.DANGER */
const DANGER = "#ef4444";
/** palette.ACCENT - gear, the one new colour site */
const ACCENT = "#a78bfa";

/**
 * Speed first, which is the convention every client the spec surveyed uses (MoTeC
 * i2, The Field, Pi Toolbox). The delta-first alternative is a one-array flip and
 * is recorded as an open question in the spec rather than half-taken here.
 */
const LANES: Lane[] = [
  {
    key: "speed",
    label: "SPEED",
    unit: "km/h",
    range: [0, 360],
    colour: INFO,
    weight: 3,
    readout: (value) => `${Math.round(value)}`,
  },
  {
    key: "delta",
    label: "Δ TIME",
    unit: "s",
    // The Qt range, kept deliberately: no autorange churn at 10 Hz.
    //
    // **Its old justification died with #1066 and this replaces it.** The comment
    // used to add "and the tower's GAP column carries the number while the trace
    // is off the top". GAP and INT report an ON-TRACK gap; this lane now reports a
    // lap-time difference, and no other panel on the window carries that. The
    // escape hatch is the lane's OWN readout, which prints the value whether or
    // not the line is inside the axis.
    //
    // It is off the top more often than "wanders" suggests, and the number is
    // measured rather than felt. Over Melbourne 2025, 6,451 same-lap driver pairs:
    // the end-of-lap difference exceeds 3 s on 27.7 % of them, a lap with a pit
    // stop costs a median 16.47 s, a neutralised lap puts 57.6 % outside, and the
    // worst pairing in the race is 50.97 s, seventeen lane-heights. Widening to
    // [-6, 6] would buy 83.9 % coverage for half the resolution near zero, which
    // is where two cars actually racing live; autoranging per lap would fit
    // everything and make a quiet lap's noise look like drama. Locked, clipped and
    // read off the number is the deliberate choice.
    range: [-3, 3],
    colour: INFO,
    weight: 3,
    readout: (value) => `${value > 0 ? "+" : ""}${value.toFixed(2)}`,
  },
  {
    key: "throttle",
    label: "THROTTLE",
    unit: "%",
    range: [-5, 105],
    colour: SUCCESS,
    weight: 2,
    labels: [0, 100],
    readout: (value) => `${Math.round(value)}`,
  },
  {
    key: "brake",
    label: "BRAKE",
    unit: "%",
    // Its OWN constant, never merged with throttle's. Two rules that agree by
    // coincidence are a defect class this repo has already paid for: a compound
    // floor and a minimum-stint bound shared a 12, and recalibrating one silently
    // rewrote the other.
    range: [-5, 105],
    colour: DANGER,
    weight: 2,
    // Its own list too, for the reason the range comment above gives.
    labels: [0, 100],
    readout: (value) => `${Math.round(value)}`,
  },
  {
    key: "gear",
    label: "GEAR",
    unit: "",
    // 1-8 on the real session with all eight present, so [0, 9] frames every step
    // with half a step of air.
    range: [0, 9],
    colour: ACCENT,
    weight: 1.7,
    step: true,
    // **Two marks, and the reason the list is not three.** Every-other-gear was still
    // 8 px of pitch in a 37 px lane at the 1080p client. `[0, 4, 8]` was the next try
    // and the comment here claimed "bottom, middle, top" - but 0 is this axis's own
    // MINIMUM and `valueAxis` sets `showMinLabel: false` on every locked axis
    // (`chart.ts`), so the 0 never printed at any client and the entry was dead weight
    // under a comment naming a mark that does not exist. Measured at all five clients
    // before and after: `4` and `8`, 16 px apart. The readout to the right carries the
    // exact gear.
    labels: [4, 8],
    readout: (value) => `${Math.round(value)}`,
  },
  {
    key: "drs",
    label: "DRS",
    unit: "",
    range: [-0.2, 1.2],
    colour: INFO,
    weight: 0.8,
    step: true,
    // **A word, not a 1 or a 0, and the reason is measured.** On the only race on
    // disk DRS is open in 0.4 % of frames, so this lane is flat for whole laps -
    // and a flat lane with no readout is indistinguishable from a broken one.
    readout: (value) => (value > 0.5 ? "OPEN" : "CLOSED"),
  },
];

/** The shared x-axis band: ticks plus "Distance (m)". Only lane 6 renders it. */
const AXIS_BAND = 34;
/** Between lanes. Five of them. */
const LANE_GAP = 6;
/** The label row lives INSIDE its lane, above the plot. */
const LABEL_ROW = 12;
/** Shared by all six grids so the lanes align and the cursor is one straight line. */
const PLOT_LEFT = 44;
const PLOT_RIGHT = 12;

interface LaneBox {
  top: number;
  height: number;
}

/**
 * Where each lane's PLOT sits, given the stack's measured height.
 *
 * Derived from the weights rather than from a table of pixels, because the same
 * six lanes have to fit 666 px at one client and 430 at another. The axis band and
 * the gaps come off the top; what is left is shared by weight; the last lane keeps
 * the rounding so the six always sum to exactly the height given.
 */
export function laneLayout(stackHeight: number): LaneBox[] {
  const usable = Math.max(
    0,
    stackHeight - AXIS_BAND - LANE_GAP * (LANES.length - 1),
  );
  const total = LANES.reduce((sum, lane) => sum + lane.weight, 0);
  const boxes: LaneBox[] = [];
  let spent = 0;
  let top = 0;
  LANES.forEach((lane, index) => {
    const last = index === LANES.length - 1;
    const height = last
      ? usable - spent
      : Math.round((usable * lane.weight) / total);
    boxes.push({ top, height });
    spent += height;
    top += height + LANE_GAP;
  });
  return boxes;
}

export interface TraceStackProps {
  main: SortedTrace;
  rival: SortedTrace;
  /** (rival - main) against distance, already interpolated onto one x axis. */
  delta: [number, number][];
  /** Locked X maximum: the circuit length, or the fallback until one arrives. */
  xMax: number;
  rivalCode: string | null;
  /**
   * The rival's own team colour, resolved by `DataWindow` (#1070).
   *
   * A prop rather than a constant here, and that IS the fix. Every rival series
   * on all six lanes used to draw in a fixed `palette.WARNING` amber, inherited
   * 1:1 from the Qt panel this is a port of, which fixes the rival to WARNING at
   * `telemetry_panel.py:101,131,167,181,195`. That was deliberate and it stopped
   * being right the moment the tower could pin any car: the amber sits an RGB
   * distance of 33.5 from McLaren papaya, the closest pair in the whole team
   * palette, so pinning a McLaren looked correct and pinning anyone else looked
   * like the colour had gone stale.
   *
   * Resolved once above and passed down for the same reason the CODE is (#1051):
   * the chip and the six series must not be able to disagree about one car.
   */
  rivalColour: string;
  /** Where the car is on the lap right now, in metres. Null before the first tick. */
  cursorX: number | null;
  /** Replaces the whole stack when set - a dead feed with nothing accumulated. */
  placeholder?: string | null;
}

function channel(trace: SortedTrace, lane: Lane): [number, number][] {
  if (lane.key === "drs")
    return trace.xs.map((x, i) => [x, trace.rows[i].drsOpen ? 1 : 0]);
  if (lane.key === "gear")
    return trace.xs.map((x, i) => [x, trace.rows[i].gear]);
  if (lane.key === "delta") return [];
  const key = lane.key as "speed" | "throttle" | "brake";
  return trace.xs.map((x, i) => [x, trace.rows[i][key]]);
}

/** The newest main-span value for a lane, for its label row. Null when starved. */
function latest(
  lane: Lane,
  main: SortedTrace,
  delta: [number, number][],
): number | null {
  if (lane.key === "delta")
    return delta.length ? delta[delta.length - 1][1] : null;
  if (!main.xs.length) return null;
  const row = main.rows[main.rows.length - 1];
  if (lane.key === "drs") return row.drsOpen ? 1 : 0;
  if (lane.key === "gear") return row.gear;
  return row[lane.key as "speed" | "throttle" | "brake"];
}

export function TraceStack(props: TraceStackProps) {
  const {
    main,
    rival,
    delta,
    xMax,
    rivalCode,
    rivalColour,
    cursorX,
    placeholder = null,
  } = props;
  const [box, attachHost] = useMeasuredHeight();

  const option = useMemo<EChartsOption>(() => {
    const boxes = laneLayout(box);
    return {
      backgroundColor: "transparent",
      // The same answer the 2x2 reached and for the same reason: `useEChart` passes
      // `notMerge: true`, so every tick looks like a fresh series and ECharts plays
      // the ENTRANCE sweep rather than the zeroed update. At 10 Hz there is no
      // separable entrance to animate.
      animation: false,
      animationDurationUpdate: 0,
      grid: boxes.map((lane) => ({
        left: PLOT_LEFT,
        right: PLOT_RIGHT,
        // The label row lives inside the lane, so the plot starts below it and data
        // can never collide with the value it is being read against.
        top: lane.top + LABEL_ROW,
        height: Math.max(0, lane.height - LABEL_ROW),
        containLabel: false,
      })),
      // **One labelled axis, five silent ones.** This is where the height comes
      // from: the 2x2 painted four 36 px axis bands, the stack paints one.
      xAxis: LANES.map((_lane, index) => {
        const last = index === LANES.length - 1;
        const axis = valueAxis({
          min: 0,
          max: xMax,
          name: last ? "Distance (m)" : undefined,
          nameGap: last ? 20 : undefined,
          label: last ? (value) => `${value / 1000}k` : undefined,
        });
        return {
          ...axis,
          gridIndex: index,
          axisLabel: last ? axis.axisLabel : { show: false },
          axisTick: { show: last },
        };
      }),
      yAxis: LANES.map((lane, index) => {
        const axis = valueAxis({ min: lane.range[0], max: lane.range[1] });
        return {
          ...axis,
          gridIndex: index,
          // A binary lane's label row and readout carry its meaning; two tick
          // labels reading 0 and 1 would be noise. Gear labels whole gears only.
          // **Six ticks and five split lines inside an 11 px lane are a grey smear,
          // not an axis.** Seen at 3x on the 1080p client: they merge into one band
          // hanging off the left of the frame and read as damage. The lane is a state
          // strip - its meaning is the step's height and the word to its right - so it
          // keeps none of the three. (The split lines went first and the smear stayed:
          // the marks were the TICKS, outside the axis line, which the fix had missed.)
          ...(lane.key === "drs"
            ? { splitLine: { show: false }, axisTick: { show: false } }
            : {}),
          axisLabel:
            lane.key === "drs"
              ? { show: false }
              : {
                  ...axis.axisLabel,
                  // Blank rather than absent: a hidden label keeps its tick's
                  // position, so the ones that DO print stay where the value is.
                  ...(lane.labels
                    ? {
                        formatter: (value: number) =>
                          lane.labels?.some(
                            (allowed) => Math.abs(allowed - value) < 0.01,
                          )
                            ? String(value)
                            : "",
                      }
                    : {}),
                },
        };
      }),
      series: LANES.flatMap((lane, index) => {
        const shared = {
          type: "line" as const,
          xAxisIndex: index,
          yAxisIndex: index,
          symbol: "none" as const,
          ...(lane.step ? { step: "end" as const } : {}),
        };
        const own =
          lane.key === "delta"
            ? {
                ...shared,
                name: `${lane.key}-main`,
                data: [] as [number, number][],
                // The main car IS the baseline on this lane; the trace is the gap
                // to it. The zero line hangs off this otherwise-empty series.
                markLine: {
                  silent: true,
                  symbol: "none" as const,
                  label: { show: false },
                  data: [
                    {
                      yAxis: 0,
                      lineStyle: {
                        color: lane.colour,
                        width: 2,
                        type: "solid" as const,
                      },
                    },
                  ],
                },
              }
            : {
                ...shared,
                name: `${lane.key}-main`,
                data: channel(main, lane),
                lineStyle: { color: lane.colour, width: 2 },
                itemStyle: { color: lane.colour },
              };
        const rivalSeries = {
          ...shared,
          name: `${lane.key}-rival`,
          // Dashed on every lane, because dashed means BROADCAST TIER on this
          // window and the rival's span is the coarse public channel. The tag
          // itself stays on the header chip: one card, one header, six lanes.
          data: rivalCode
            ? lane.key === "delta"
              ? delta
              : channel(rival, lane)
            : [],
          lineStyle: { color: rivalColour, width: 2, type: "dashed" as const },
          itemStyle: { color: rivalColour },
        };
        // **The rival is declared FIRST so the own car paints on top of it.** The
        // race trace states the rule ("our own car moved LAST so it draws on top")
        // and the 2x2 had it inverted until this sprint; the stack keeps the fixed
        // order rather than inheriting the bug.
        return [rivalSeries, own];
      }),
    };
  }, [box, main, rival, delta, xMax, rivalCode, rivalColour]);

  const chart = useEChart(box > 0 && !placeholder ? option : null);
  const boxes = laneLayout(box);
  const stackBottom = boxes.length
    ? boxes[boxes.length - 1].top + boxes[boxes.length - 1].height
    : 0;
  const cursorFraction =
    cursorX === null || xMax <= 0
      ? null
      : Math.min(Math.max(cursorX, 0), xMax) / xMax;

  return (
    <div className="trace-stack" ref={attachHost}>
      {placeholder ? (
        <div className="trace-placeholder">{placeholder}</div>
      ) : (
        <>
          <div className="trace-stack-plot" ref={chart} />
          {/* The label rows and the cursor are HTML over the canvas, not ECharts
              graphics: `notMerge: true` would restart them ten times a second. */}
          {/* **Single-driver mode captions the Δ LANE, not the whole panel.** In the
              2x2 it replaced a whole 262 x 328 cell, which was the only granularity
              there was; here the other five lanes are perfectly good with one car in
              them, and only the delta is meaningless without a rival. `deltaSeries`
              is empty by construction in that case, so the caption is the honest
              rendering of a lane that has nothing to draw rather than a fallback. */}
          {rivalCode === null && boxes.length > 1 ? (
            <div
              className="trace-lane-caption"
              style={{
                top: boxes[1].top + LABEL_ROW,
                height: Math.max(0, boxes[1].height - LABEL_ROW),
                left: PLOT_LEFT,
                right: PLOT_RIGHT,
              }}
            >
              single-driver mode
            </div>
          ) : null}
          {boxes.map((lane, index) => (
            <div
              key={LANES[index].key}
              className="trace-lane-label"
              style={{ top: lane.top, left: PLOT_LEFT, right: PLOT_RIGHT }}
            >
              <span
                className="trace-lane-name"
                style={{ color: LANES[index].colour }}
              >
                {LANES[index].label}
                {LANES[index].unit ? (
                  <span className="trace-lane-unit"> {LANES[index].unit}</span>
                ) : null}
              </span>
              <span className="trace-lane-value">
                {(() => {
                  const value = latest(LANES[index], main, delta);
                  return value === null ? "—" : LANES[index].readout(value);
                })()}
              </span>
            </div>
          ))}
          {cursorFraction === null ? null : (
            // ONE div, spanning lane 1's top to lane 6's bottom, so the cut a
            // reader makes is unbroken across the gaps. Solid, not dashed: the car
            // covers under a pixel per tick, and a dashed line at that speed
            // shifts its pattern by a fraction of a pixel and shimmers.
            <div
              className="trace-cursor"
              style={{
                left: `calc(${PLOT_LEFT}px + (100% - ${PLOT_LEFT + PLOT_RIGHT}px) * ${cursorFraction})`,
                top: 0,
                height: stackBottom,
                background: CURSOR_LINE,
              }}
            />
          )}
        </>
      )}
    </div>
  );
}

/**
 * The stack's own pixel height, and the callback ref that keeps it current.
 *
 * The lane arithmetic needs a NUMBER, and the box is a CSS flex remainder the two
 * clients answer differently - 666 px at 1485, 430 at 1265 - so it is measured
 * rather than tabulated, and re-measured when the window resizes.
 *
 * The node arrives through a callback ref rather than a `useRef`, for the reason
 * `useEChart` documents: a panel that unmounted and came back left a dead observer
 * for the rest of the session.
 */
function useMeasuredHeight(): [number, (node: HTMLDivElement | null) => void] {
  const [height, setHeight] = useState(0);
  const observer = useRef<ResizeObserver | null>(null);

  const attach = useCallback((node: HTMLDivElement | null) => {
    observer.current?.disconnect();
    observer.current = null;
    if (!node) return;
    setHeight(node.clientHeight);
    observer.current = new ResizeObserver(() => setHeight(node.clientHeight));
    observer.current.observe(node);
  }, []);

  return [height, attach];
}
