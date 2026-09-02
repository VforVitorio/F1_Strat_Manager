/**
 * Band 3's second half: the race trace, a tab of the right column.
 *
 * The panel is read with a ruler held VERTICALLY. One cut down the chart at a
 * lap gives every gap in the race at that lap at once, which is the question
 * the race-pace grid beside it cannot answer - a grid of lap times says who
 * was quick and never says who is where.
 *
 * **The three references are the panel's only control, and each one answers a
 * different question.** LEADER draws the race the way a broadcast graphic
 * does. FIELD spreads the leaders above the axis so a midfield battle is not
 * squashed against the top of the plot. OWN puts our car flat on zero, which
 * is the pit wall's own question: a line climbing towards zero is a car coming
 * for us, and the lap it reaches zero is the lap it arrives.
 *
 * Twenty lines need twenty identities, and a legend for twenty codes eats a
 * third of the plot. ECharts labels each line at its right-hand END instead,
 * which is where a race trace has always been read and where the lines are
 * furthest apart - by the end of a race the field is strung out, so the labels
 * separate exactly where the data does.
 *
 * The colours are the wire's `driver_colors`, as the tower and the ring
 * already take them. A driver the wire has no colour for draws in the axis
 * grey rather than in an invented hue - this repo has already found five
 * copies of the arcade palette and none of them is going to be the sixth.
 *
 * ⚠️ **Those colours are TEAM colours, so the end label is the only thing that
 * tells two team-mates apart.** Measured on the real Melbourne payload: ten
 * colours across twenty cars, every one of them shared by exactly two drivers
 * (VER/LAW both `rgb(6,0,239)`, LEC/HAM both `rgb(232,0,32)`, and so on). The
 * tower has the same property and answers it with a code per row; here the
 * answer is the code at the end of the line, which is why the de-collision
 * below is load-bearing rather than polish. Dashing the second car of each
 * team is the broadcast solution and is deliberately NOT taken: a dashed line
 * already means BROADCAST-TIER data on band 4, one tab away, and two meanings
 * for one stroke on one window is worse than two lines of one colour.
 */

import { useMemo, useState } from "react";
import type { EChartsOption } from "echarts";

import { AXIS_TEXT, CURSOR_LINE, NEUTRALISED_BAND, useEChart, valueAxis } from "../../lib/chart";
import { useChartHover } from "../../lib/chartHover";
import { driverStatus } from "../../lib/driverStatus";
import { neutralisedLaps, neutralisedRanges } from "../../lib/neutralised";
import { raceTrace } from "../../lib/raceTrace";
import type { RaceTrace, TraceReference } from "../../lib/raceTrace";
import type { ArcadeState, Bulk } from "../../lib/bridge";
import { driverColour } from "../../lib/driverColour";

/** The own car is drawn heavier than the nineteen it has to be picked out of. */
const OWN_WIDTH = 2;
const FIELD_WIDTH = 1;

/**
 * Never a CSS custom property here, unlike the tower's fallback: an ECharts
 * canvas cannot resolve `var(--qt-fg-1)` and would draw the series in its own
 * default palette, which is the one colour set on this window that answers to
 * nothing.
 */
const NO_COLOUR = AXIS_TEXT;

/** `+12.3` / `-4.5` - the sign is the whole reading, so it is always printed. */
function signedSeconds(value: number): string {
  return `${value > 0 ? "+" : ""}${value.toFixed(1)}`;
}

/** One line's value at the hovered lap. `null` means that car has none there. */
interface ReadoutRow {
  code: string;
  value: number | null;
}

/**
 * Every car's gap at one lap, which is the vertical cut this panel exists for.
 *
 * **Sorted DESCENDING, because higher on this chart is further up the road.**
 * That holds under all three references and is what makes the list read front
 * to back like a timing tower: in LEADER mode the car in front sits on zero and
 * the rest hang below it at negative values, so ascending would have put the
 * leader last and the tail-ender first. The first version did exactly that and
 * only the screenshot said so - the numbers were all correct and the order was
 * upside down.
 *
 * A car with no value at that lap - retired before it, or the reveal has not
 * reached it - goes to the BOTTOM with an em dash rather than being left out: a
 * car that vanishes from the list reads as a car that is not in the race, and
 * this panel is where a reader counts the field.
 *
 * Laps are integers on this axis, so the pointer's continuous x is rounded to
 * the nearest one and the row is looked up by exact lap. No interpolation: a
 * gap "at lap 23.4" is not a quantity the race has.
 */
function traceReadout(trace: RaceTrace, x: number): { lap: number; rows: ReadoutRow[] } | null {
  const lap = Math.round(x);
  if (!trace.laps.includes(lap)) return null;
  const rows = trace.lines.map((line) => ({
    code: line.code,
    value: line.points.find(([pointLap]) => pointLap === lap)?.[1] ?? null,
  }));
  const known = rows.filter((row) => row.value !== null);
  const missing = rows.filter((row) => row.value === null);
  known.sort((a, b) => (b.value as number) - (a.value as number));
  return { lap, rows: [...known, ...missing] };
}

export function RaceTraceChart({ bulk, arcade }: { bulk: Bulk | null; arcade: ArcadeState }) {
  const [reference, setReference] = useState<TraceReference>("leader");
  const own = arcade.driver_main;

  // **The signature, not the objects, and this is the plan's directive rather
  // than a micro-optimisation.** `arcade` is a fresh object ten times a
  // second, so memoising on it would rebuild twenty series and call
  // `setOption({notMerge: true})` at 10 Hz for content that changes once every
  // four and a half seconds - P3 finding A6, which is on this window's list of
  // things to prevent rather than repeat. What the trace actually depends on
  // is the reveal, the reference, who is racing and who has stopped.
  const stopped = arcade.race_order
    .filter((code) => {
      const car = arcade.drivers[code];
      return car === undefined || driverStatus(car) === "out";
    })
    .join(",");
  const signature = `${bulk?.rev ?? -1}|${reference}|${own}|${arcade.race_order.join(",")}|${stopped}`;

  const trace = useMemo(
    () => raceTrace(bulk, arcade, reference),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [signature],
  );

  // Same source as the pace grid's rail, one tab away, so the two panels cannot
  // disagree about which laps were neutralised. Memoised on the reveal, not on
  // `arcade`, for the reason the trace itself is.
  const neutral = useMemo(() => neutralisedRanges(neutralisedLaps(bulk)), [bulk?.rev]);

  const option = useMemo<EChartsOption>(() => {
    const first = trace.laps[0] ?? 1;
    const last = trace.laps[trace.laps.length - 1] ?? 1;
    return {
      backgroundColor: "transparent",
      // Same answer as band 4's traces and for a weaker version of the same
      // reason: `useEChart` passes `notMerge: true`, so every reveal looks to
      // ECharts like a brand-new series and it plays the ENTRANCE sweep, not
      // the zeroed update duration. Twenty lines growing in from the left
      // every four and a half seconds is not an entrance, it is a flicker.
      animation: false,
      animationDurationUpdate: 0,
      // The right margin is the end labels' room. Measured on the real payload:
      // the rightmost label's edge lands at 776 px on an 803 px canvas, so 52
      // clears it. Without the margin the codes are cut by the plot edge and the
      // panel loses the only identification it has.
      grid: { left: 52, right: 52, top: 10, bottom: 34, containLabel: false },
      xAxis: valueAxis({ name: "Lap", nameGap: 20, min: first, max: last }),
      yAxis: {
        ...valueAxis({ name: "Δ (s)", nameGap: 38, scale: true }),
        axisLabel: {
          color: AXIS_TEXT,
          fontSize: 10,
          formatter: (value: number) => signedSeconds(value),
        },
      },
      series: trace.lines.map((line) => {
        const colour = driverColour(arcade.driver_colors, line.code, NO_COLOUR);
        const isOwn = line.code === own;
        return {
          type: "line" as const,
          name: line.code,
          data: line.points,
          symbol: "none" as const,
          lineStyle: { color: colour, width: isOwn ? OWN_WIDTH : FIELD_WIDTH },
          itemStyle: { color: colour },
          // The zero line hangs off the first series, as band 4's cursor does.
          markLine: line.code === trace.lines[0]?.code
            ? {
                silent: true,
                symbol: "none" as const,
                label: { show: false },
                data: [{ yAxis: 0, lineStyle: { color: CURSOR_LINE, width: 1, type: "solid" } }],
              }
            : undefined,
          // **The neutralised lap ranges, shaded, and they explain the shape the
          // panel already draws.** Melbourne's trace has a V across laps 5-8 and
          // a long convergence from 33 that are the field bunching behind a
          // safety car, not anybody's pace; unlabelled they read as racing. Hung
          // off the same first series as the zero line, for the same reason.
          //
          // A translucent fill, never a stroke: a dashed line means
          // BROADCAST-TIER one tab away and a solid thin vertical is the current
          // lap. The band is a fifth channel rather than a second meaning for a
          // taken one.
          markArea: line.code === trace.lines[0]?.code && neutral.length
            ? {
                silent: true,
                itemStyle: { color: NEUTRALISED_BAND },
                label: {
                  show: true,
                  // At the band's LEFT edge, not its top centre: a band that ends
                  // at the plot's right - which is what a LIVE neutralisation looks
                  // like - put its label straight on top of the NOR/PIA end labels,
                  // and those labels are the chart's only identification.
                  position: "insideTopLeft" as const,
                  color: AXIS_TEXT,
                  fontSize: 8,
                  formatter: (params: { name?: string }) => params.name ?? "",
                },
                // **Padded to the lap's own cell, because a lap is a POINT on this
                // axis.** Unpadded, a one-lap range is `from == to` and ECharts
                // paints a zero-width area: a safety car that has just come out -
                // the moment the band exists for - rendered NOTHING while its label
                // floated over the driver codes. Measured live at lap 35 with only
                // lap 33 revealed as neutralised. The half-lap also puts a boundary
                // lap's data point INSIDE its band rather than on the edge.
                data: neutral.map((band) => [
                  { name: band.label, xAxis: band.from - 0.5 },
                  { xAxis: band.to + 0.5 },
                ]),
              }
            : undefined,
          endLabel: {
            show: line.points.length > 0,
            formatter: line.code,
            // **`AXIS_TEXT`, not the line's own colour, and this panel is where it
            // matters most.** The docstring above says the end label is the ONLY
            // thing that tells two team-mates apart - and for VER and LAW that sole
            // identification was 9 px of rgb(6,0,239) on the card, 1.88:1. Six of
            // the twenty codes failed AA and four failed even the 3.0 floor.
            //
            // The mapping survives because the label sits ON its own line's end:
            // adjacency carries it, which is why this panel can give the colour up
            // where the tower had to move it to a swatch instead. #d1d5db is
            // 11.9:1 on the card.
            color: AXIS_TEXT,
            fontSize: 9,
            fontWeight: isOwn ? ("bold" as const) : ("normal" as const),
            distance: 3,
          },
          // **Two labels land on the same pixels, and only the screenshot said
          // so.** Measured on the real Melbourne payload in the real bundle:
          // ALB's and HAM's codes overlapped by 4.5 px of their 9 px height,
          // because the midfield is where cars run a second apart and a second
          // is four pixels on a 676 px plot spanning 77 s. Nothing else could
          // see it - not the axis extents, not any of the smoke's checks -
          // because an ECharts label is canvas text and not a DOM node.
          //
          // `shiftY` nudges the colliding label down instead of dropping it.
          // `hideOverlap` is the other option ECharts offers and it is the
          // wrong one here: the label IS the identification, so hiding it
          // leaves an anonymous line, and it would hide exactly the labels a
          // strategist is most likely to be reading - two cars a second apart
          // are two cars racing each other.
          labelLayout: { moveOverlap: "shiftY" as const, hideOverlap: false },
        };
      }),
    };
    // `arcade` is read only for its colours and the own code, both of which
    // are already in the signature; see the note above.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [trace, own]);

  const [host, instance] = useEChart(trace.lines.length ? option : null);
  const lapDomain: [number, number] | null = trace.laps.length
    ? [trace.laps[0], trace.laps[trace.laps.length - 1]]
    : null;
  const [hover, hoverProps] = useChartHover(instance, lapDomain);
  const readout = hover === null ? null : traceReadout(trace, hover.dataX);

  return (
    <section className="card trace-band">
      <header className="pace-header">
        <span className="pace-title">RACE TRACE</span>
        <span className="pace-subtitle">{trace.zero}</span>
        {/* How far the bound is behind the race, said out loud. A trace that
            stops is indistinguishable from a race that stopped, and one car
            with a mid-race telemetry dropout pins it silently - see
            `RaceTrace.total`. */}
        {trace.laps.length ? (
          <span className="pace-range">
            LAPS {trace.laps[0]}-{trace.laps[trace.laps.length - 1]} of {trace.total}
          </span>
        ) : null}
        <nav className="ref-strip" role="tablist" aria-label="reference">
          {(["leader", "field", "own"] as const).map((id) => (
            <button
              key={id}
              role="tab"
              aria-selected={reference === id}
              className={reference === id ? "ref is-active" : "ref"}
              onClick={() => setReference(id)}
            >
              {id === "own" ? own : id.toUpperCase()}
            </button>
          ))}
        </nav>
      </header>
      {trace.lines.length ? (
        <div className="trace-band-hover">
          <div className="trace-band-plot" ref={host} {...hoverProps} />
          {/* The vertical cut, made with the mouse. Twenty lines is the one case
              on either window where a floating list is the right shape: the
              values are spread over the whole plot height and no per-line label
              could hold them.

              It opens on whichever side of the cursor has room, and it is
              absolutely positioned inside the card - never fixed, because
              `.data-main.is-frozen`'s filter would become its containing block
              and move it 50 px the moment the feed froze. */}
          {hover !== null && readout !== null ? (
            <>
              <div className="trace-band-cursor" style={{ left: hover.pixelX }} />
              <div
                className="trace-band-box"
                style={
                  hover.pixelX > hover.hostWidth / 2
                    ? { right: hover.hostWidth - hover.pixelX + 8 }
                    : { left: hover.pixelX + 8 }
                }
              >
                <span className="trace-band-box-lap">LAP {readout.lap}</span>
                {readout.rows.map((row) => (
                  <span key={row.code} className="trace-band-box-row">
                    {/* The CODE identifies the car, not the colour: ten team
                        colours cover twenty drivers, so every hue here is shared
                        by exactly two of them. The same reason the end labels
                        are drawn in the axis grey. */}
                    <span className="trace-band-box-code">{row.code}</span>
                    <span className="trace-band-box-value">
                      {row.value === null ? "—" : signedSeconds(row.value)}
                    </span>
                  </span>
                ))}
              </div>
            </>
          ) : null}
        </div>
      ) : (
        // A trace with nothing to draw SAYS so. An empty plot and a race whose
        // field has not yet completed a common lap are the same pixels
        // otherwise, which is the twin the radio feed's own empty state
        // already had to grow.
        <p className="trace-band-empty">
          No lap the whole classified field has completed yet.
        </p>
      )}
    </section>
  );
}
