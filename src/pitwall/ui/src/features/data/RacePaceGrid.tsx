/**
 * Band 3 - the race-pace grid, a tab of the right column.
 *
 * One COLUMN per driver and one ROW per lap, which is the orientation the real
 * client uses and the one a measurement chose rather than a preference. The
 * transposed form the sprint-5 height gate proposed was derived against a
 * stacked band model that has since died; measured against the column model at
 * the right column's real width, transposing gives cells 13.5 px wide against
 * 19-22 px of text and clips 1,121 of 1,140 - it can hold a colour and nothing
 * else, and the number in the cell is the panel's whole point.
 *
 * **The ring and the radio feed hide while this tab is open, and that is
 * measured too.** With the 260 px ring column still mounted the grid gets
 * 555 px, columns fall to 25.25 px against 25 px of text, and 1,101 of 1,140
 * cells clip. There is no arrangement that keeps both.
 *
 * Both of those counts come from the PROTOTYPE that chose the orientation
 * (`b3_measure.mjs`), not from this component: they describe layouts the tree
 * deliberately no longer contains, so they cannot be re-measured here and are
 * named as prototype figures rather than as measurements of what ships.
 *
 * **The sentence that stood here claimed the shipped grid clips nothing, and it
 * was measured at every client HEIGHT the fleet produces and at exactly one
 * WIDTH.** True as stated - 20 columns of 38.75 px at the 1485 px client, 0 of
 * 1,140 cells cut - and it read as a property of the panel. On a 1080p laptop at
 * 150 % scaling the client is 1265 x 593, the columns are 27.75 px, and 495 of
 * 514 populated cells lost their last glyph. The width axis had no floor and no
 * fallback because nobody had measured it. `paceLabel`'s `coarse` form is that
 * fallback, and `fineFormFits` below decides between them from a real cell.
 *
 * The lap axis scrolls, because it must: Monaco's 78 laps overflow the column
 * on every machine in the fleet at every legible font size. It is pinned to
 * the newest lap so the panel follows the race, and the header states the
 * range on screen - a hidden scrollbar is not an affordance, and this window
 * hides them globally.
 */

import { useCallback, useEffect, useRef, useState } from "react";

import { racePaceGrid } from "../../lib/racePace";
import type { Bulk } from "../../lib/bridge";

/**
 * What the ruler measures before the first lap is revealed and there is no data to
 * ask. Six glyphs, the ordinary case.
 *
 * **Once there IS data the ruler measures `grid.widestFine` instead**, because a
 * constant was wrong in both directions: `0:00.0` is blind to the seven-glyph
 * `10:00.0` that `paceLabel`'s own docstring says a red-flagged race produces, and
 * `10:00.0` never fits the 38.75 px column that holds Melbourne's `1:29.7` with
 * room to spare - so hardcoding the safe one would have coarsened the WIDE client
 * for every race.
 */
const FALLBACK_FINE_LABEL = "0:00.0";

/** One canvas for the page, made on first use. Null when there is no 2d context. */
let ruler: CanvasRenderingContext2D | null | undefined;

/**
 * How wide `text` renders in a cell's OWN computed font.
 *
 * Measured rather than compared against a pixel constant, and the stylesheet's
 * own comment says why: it counts on `1:29.4` measuring 33.2 px inside a
 * 38.75 px column, which is true of JetBrains Mono at 9 px and not of whatever
 * a machine without it falls back to. A constant tuned on one font is the
 * threshold-tuned-on-one-race defect wearing a stylesheet.
 *
 * It measures the FINE form whatever the cell currently renders, which is what
 * keeps this from oscillating: asking "does the text in the cell overflow"
 * would answer no as soon as the coarse form landed, and flip back.
 */
function fineFormFits(cell: HTMLElement, label: string): boolean {
  if (ruler === undefined) ruler = document.createElement("canvas").getContext("2d");
  const style = window.getComputedStyle(cell);
  const padding = parseFloat(style.paddingLeft) + parseFloat(style.paddingRight);
  const room = cell.clientWidth - padding;
  // No canvas: keep the tenths. Losing them is a real cost, so it is not the
  // thing to do when the measurement is unavailable rather than negative.
  if (ruler === null) return true;
  ruler.font = `${style.fontWeight} ${style.fontSize} ${style.fontFamily}`;
  return ruler.measureText(label).width <= room;
}

export function RacePaceGrid({ bulk, order }: { bulk: Bulk | null; order: string[] }) {
  const scroller = useRef<HTMLDivElement | null>(null);
  const [visible, setVisible] = useState<[number, number] | null>(null);
  const [coarse, setCoarse] = useState(false);
  const grid = racePaceGrid(bulk, order, coarse);
  // A ref, not a dependency: `fit` must not be rebuilt on every reveal, and the
  // value it needs is whatever the grid last knew. Independent of `coarse`, which
  // is what keeps the decision from oscillating.
  const widest = useRef(FALLBACK_FINE_LABEL);
  widest.current = grid.widestFine || FALLBACK_FINE_LABEL;

  /**
   * The laps ACTUALLY on screen, from the scroller's own geometry.
   *
   * **Not `grid.laps[0]`, which is the defect this replaces (#949).** That is
   * always 1, so the header read `LAPS 1-57` while the panel - pinned to the
   * bottom - was showing 8 to 57. This window hides scrollbars globally and
   * the stylesheet says in as many words that this range is the affordance
   * standing in for one, so it was wrong in exactly the case it exists for and
   * right only when the grid fits, which is when nobody needs it.
   *
   * Measured off each row's own box rather than derived from a row-height
   * constant: the height is a CSS token and a second copy of it here is the
   * twin this repo pays for most often.
   */
  const measure = useCallback(() => {
    const box = scroller.current;
    if (!box) return;
    const rows = [...box.querySelectorAll<HTMLElement>("tbody tr")];
    if (rows.length === 0) return setVisible(null);
    const top = box.scrollTop;
    const bottom = top + box.clientHeight;
    const shown = rows.filter((row) => row.offsetTop + row.offsetHeight > top && row.offsetTop < bottom);
    const edges = shown.length ? shown : rows;
    const lapOf = (row: HTMLElement) => Number(row.querySelector("th")?.textContent ?? 0);
    setVisible([lapOf(edges[0]), lapOf(edges[edges.length - 1])]);
  }, []);

  /**
   * Whether the tenths fit, from a real cell rather than from a breakpoint.
   *
   * The trigger is the COLUMN's width, which is the client width and the number
   * of cars together - a media query would have to guess at one of them. The
   * observer is on the scroller so a resized window re-answers it; nothing else
   * changes width under this panel.
   */
  const fit = useCallback(() => {
    // A BODY cell, because those are the cells that have to hold the label. The
    // header is the fallback for the render before the first reveal, and it is a
    // proxy rather than the thing: it is `font-weight: 700` where a body cell is
    // 400. Measured on this machine both give `0:00.0` a width of 33.23 px, so
    // there is no difference today; with a fallback font whose bold is wider the
    // header would coarsen slightly EARLY, which is the safe direction, but the
    // point of `fineFormFits` is to measure the rendered font and not a proxy.
    const box = scroller.current;
    const cell =
      box?.querySelector<HTMLElement>("tbody td") ??
      box?.querySelector<HTMLElement>("thead th + th");
    if (cell && cell.clientWidth > 0) setCoarse(!fineFormFits(cell, widest.current));
  }, []);

  useEffect(() => {
    const box = scroller.current;
    if (!box) return;
    fit();
    const observer = new ResizeObserver(fit);
    observer.observe(box);
    return () => observer.disconnect();
  }, [fit, grid.columns.length, grid.widestFine]);

  // Pinned to the newest lap. A race-pace grid that opens at lap 1 shows the
  // formation lap of a race that is on lap 40, and the laps a strategist is
  // deciding on are always the last few. Re-pinned on every reveal, which is
  // about once every four and a half seconds.
  //
  // The pin only has work to do once the table outgrows the scroller, which on a
  // 57-lap race is around lap 55. What holds the newest lap still for the other
  // 54 is the stylesheet: `.pace` is `align-self: end`, so the CARD grows upward
  // from the bottom of its column and the newest row sits at a fixed height from
  // lap 1. (Anchoring the TABLE inside a full-height card does the same thing and
  // leaves the empty space INSIDE the card; that is what this comment described
  // for one commit, and it is not what ships. See the `.pace` rule.)
  useEffect(() => {
    const box = scroller.current;
    if (box) box.scrollTop = box.scrollHeight;
    measure();
  }, [grid.laps.length, measure]);

  const total = bulk?.race.total_laps ?? grid.laps.length;
  const [first, last] = visible ?? [grid.laps[0], grid.laps[grid.laps.length - 1]];

  return (
    <section className="card pace">
      <header className="pace-header">
        <span className="pace-title">RACE PACE</span>
        <span className="pace-subtitle">
          colour ranks each lap against itself
          {/* Said out loud, because a coarser number that nobody was told about
              is the same silence the truncation was. */}
          {coarse ? " · times to the second at this width" : ""}
          {/* And the rail needs a key, or it is a decoration. Only shown when
              there is a marked lap on the panel at all. */}
          {grid.neutralised.some(Boolean) ? (
            <>
              {" · "}
              <span className="pace-legend-rail" /> neutralised
            </>
          ) : null}
        </span>
        <span className="pace-range">
          {grid.laps.length ? `LAPS ${first}-${last} of ${total}` : "no laps revealed"}
        </span>
      </header>
      <div className="pace-scroll" ref={scroller} onScroll={measure}>
        <table className="pace-table">
          <thead>
            <tr>
              <th className="pace-lapcol" />
              {grid.columns.map((code) => (
                <th key={code}>{code}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {grid.rows.map((row, index) => (
              <tr key={grid.laps[index]}>
                {/* The rail on the lap number is the whole marker: on a
                 * neutralised lap the colour thirds rank the safety car's queue,
                 * not pace, and 213 of the 776 cells this grid ranks on the real
                 * race sit on one. It is a BORDER rather than a text colour
                 * because amber text one column right already means "slowest
                 * third" - one hue, but a different channel, so the two cannot be
                 * confused. The label rides in the title for the reader who
                 * wonders which neutralisation it was. */}
                <th
                  className={`pace-lapcol${grid.neutralised[index] ? " is-neutralised" : ""}`}
                  title={grid.neutralised[index] ?? undefined}
                >
                  {grid.laps[index]}
                </th>
                {row.map((cell, column) => (
                  <td key={grid.columns[column]} className={`is-${cell.tone}`}>
                    {cell.text}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}
