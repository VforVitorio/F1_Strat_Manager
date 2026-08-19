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
  if (ruler === undefined)
    ruler = document.createElement("canvas").getContext("2d");
  const style = window.getComputedStyle(cell);
  const padding =
    parseFloat(style.paddingLeft) + parseFloat(style.paddingRight);
  const room = cell.clientWidth - padding;
  // No canvas: keep the tenths. Losing them is a real cost, so it is not the
  // thing to do when the measurement is unavailable rather than negative.
  if (ruler === null) return true;
  ruler.font = `${style.fontWeight} ${style.fontSize} ${style.fontFamily}`;
  return ruler.measureText(label).width <= room;
}

/**
 * Where the scroller has to sit for the newest REVEALED lap to touch its bottom edge.
 *
 * One definition, used by the pin that writes it and by the measure that asks whether the
 * reader has moved away from it. Two copies of this arithmetic would drift, and the drift
 * would read as "the panel keeps stealing my scroll".
 */
function pinTarget(box: HTMLElement, revealedTo: number): number {
  const rows = box.querySelectorAll<HTMLElement>("tbody tr");
  const newest = revealedTo > 0 ? rows[revealedTo - 1] : undefined;
  if (!newest) return box.scrollHeight;
  const target = newest.offsetTop + newest.offsetHeight - box.clientHeight;
  // The browser clamps a negative scrollTop to 0; clamping here too keeps the comparison
  // in `measure` honest, which is the whole reason this is a function.
  return Math.max(0, Math.min(target, box.scrollHeight - box.clientHeight));
}

export function RacePaceGrid({
  bulk,
  order,
}: {
  bulk: Bulk | null;
  order: string[];
}) {
  const scroller = useRef<HTMLDivElement | null>(null);
  const [visible, setVisible] = useState<[number, number] | null>(null);
  /** Which edges have more behind them, so the fade can say so. */
  const [edges, setEdges] = useState({ above: false, below: false });
  /** False while the reader has scrolled away from the newest revealed lap. */
  const following = useRef(true);
  /** The reveal the pin is currently tracking, so `measure` can compute its target. */
  const revealedTo = useRef(0);
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
    // **Does it overflow, and is there more in each direction?**
    //
    // Victor asked for "the skeleton, and if it overflows, a small scroll you can drive
    // with the mouse wheel". The wheel already worked - hiding a scrollbar does not
    // disable it - but nothing on screen said so, and the research on this is unanimous
    // and unkind: NN/g, Baymard (26 % of top e-commerce sites get inline scroll wrong)
    // and a GitLab production bug all document the same failure once the one universal
    // affordance is removed. So the panel says it itself, with a fade at whichever edge
    // has more behind it.
    const overflows = box.scrollHeight - box.clientHeight > 1;
    setEdges({
      above: overflows && box.scrollTop > 1,
      below:
        overflows && box.scrollTop < box.scrollHeight - box.clientHeight - 1,
    });
    // **And whether the reader has taken the wheel.** The pin below re-fires on every
    // reveal, about every four and a half seconds, so without this a scroll to look at
    // lap 5 is undone before it can be read - which would make the thing Victor asked
    // for useless. Standard follow-tail break: leave the tail when the reader moves away
    // from it, rejoin when they come back.
    following.current =
      Math.abs(box.scrollTop - pinTarget(box, revealedTo.current)) <= 4;
    const rows = [...box.querySelectorAll<HTMLElement>("tbody tr")];
    if (rows.length === 0) return setVisible(null);
    const top = box.scrollTop;
    const bottom = top + box.clientHeight;
    const shown = rows.filter(
      (row) => row.offsetTop + row.offsetHeight > top && row.offsetTop < bottom,
    );
    const edges = shown.length ? shown : rows;
    const lapOf = (row: HTMLElement) =>
      Number(row.querySelector("th")?.textContent ?? 0);
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
    if (cell && cell.clientWidth > 0)
      setCoarse(!fineFormFits(cell, widest.current));
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
  // **⚠️ What this comment claimed, and what is actually true.**
  //
  // It said the pin "only has work to do once the table outgrows the scroller" and that
  // `.pace`'s `align-self: end` held the newest lap at a fixed height for the rest. Both
  // halves are stale: that declaration is gone with the revealed-only table, and the
  // table is the whole race from lap 1.
  //
  // **A fixed eye-line and a whole-race axis are mutually exclusive, and it is
  // arithmetic rather than a defect.** 57 rows at 12 px is 684 px; the scroller is 674
  // at 1485x833, so there are 22 px of scroll in the entire race. Measured live at
  // revealed lap 23: `scrollTop` 0 (the target below clamps negative) and the newest
  // revealed row 386 px ABOVE the scroller's bottom edge - 158 px at 1265x593, where
  // the shorter box leaves more room and the row reaches the edge around lap 37. With
  // every lap of the race laid out in advance, the newest row's position IS the lap
  // number; nothing can pin it without scrolling the future off screen, which is the
  // skeleton's whole point.
  //
  // So the trade #990 made is reversed here, deliberately and with a different subject:
  // #990 bought a fixed eye-line with a table that showed only what had run, and paid
  // for it with a 426 px void at lap 24. The skeleton pays the eye-line and keeps the
  // room. The row is on screen throughout, and where it sits now reads as progress.
  useEffect(() => {
    const box = scroller.current;
    if (!box) return;
    // **Pinned to the newest REVEALED lap, not to the bottom of the table.** The table
    // is the whole race now, so its bottom is lap 57 - the future - and scrolling there
    // would show a wall of empty rows.
    //
    // What this buys is that the newest revealed row is never scrolled PAST: it is on
    // screen for the whole race, and it comes to rest against the bottom edge once the
    // revealed block is taller than the box (around lap 37 at 1265x593; at 1485x833 the
    // race barely outgrows the box at all, so the offset clamps to 0 nearly throughout).
    // It does NOT hold the row at a fixed height - see the paragraph above for why no
    // scroll offset can, and measured numbers for both clients.
    //
    // **And it yields to the reader.** `following` is false while they are somewhere else
    // in the race; the reveal then leaves the scroll alone and only re-measures.
    revealedTo.current = grid.revealedTo;
    if (following.current) box.scrollTop = pinTarget(box, grid.revealedTo);
    measure();
  }, [grid.revealedTo, grid.laps.length, measure]);

  const total = bulk?.race.total_laps ?? grid.laps.length;
  const [first, last] = visible ?? [
    grid.laps[0],
    grid.laps[grid.laps.length - 1],
  ];

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
          {grid.laps.length
            ? `LAPS ${first}-${last} of ${total}`
            : "no laps revealed"}
        </span>
      </header>
      <div
        className={`pace-scroll${edges.above ? " has-above" : ""}${edges.below ? " has-below" : ""}`}
        ref={scroller}
        onScroll={measure}
        // Keyboard-reachable, because with the scrollbar hidden this is the only way a
        // keyboard user reaches the rest of the race at all.
        tabIndex={0}
        role="region"
        aria-label="Race pace by lap, scrollable"
      >
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
              <tr
                key={grid.laps[index]}
                // A lap nobody has driven yet. It carries its number and nothing
                // else, so the panel shows how much race is left without pretending
                // to know anything about it.
                className={
                  grid.laps[index] > grid.revealedTo ? "is-future" : undefined
                }
              >
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
