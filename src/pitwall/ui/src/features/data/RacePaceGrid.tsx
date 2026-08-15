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
 * The lap axis scrolls, because it must: Monaco's 78 laps overflow the column
 * on every machine in the fleet at every legible font size. It is pinned to
 * the newest lap so the panel follows the race, and the header states the
 * range on screen - a hidden scrollbar is not an affordance, and this window
 * hides them globally.
 */

import { useEffect, useRef } from "react";

import { racePaceGrid } from "../../lib/racePace";
import type { Bulk } from "../../lib/bridge";

export function RacePaceGrid({ bulk, order }: { bulk: Bulk | null; order: string[] }) {
  const grid = racePaceGrid(bulk, order);
  const scroller = useRef<HTMLDivElement | null>(null);

  // Pinned to the newest lap. A race-pace grid that opens at lap 1 shows the
  // formation lap of a race that is on lap 40, and the laps a strategist is
  // deciding on are always the last few. Re-pinned on every reveal, which is
  // about once every four and a half seconds.
  useEffect(() => {
    const box = scroller.current;
    if (box) box.scrollTop = box.scrollHeight;
  }, [grid.laps.length]);

  const first = grid.laps[0];
  const last = grid.laps[grid.laps.length - 1];

  return (
    <section className="card pace">
      <header className="pace-header">
        <span className="pace-title">RACE PACE</span>
        <span className="pace-subtitle">colour ranks each lap against itself</span>
        <span className="pace-range">
          {grid.laps.length ? `LAPS ${first}-${last}` : "no laps revealed"}
        </span>
      </header>
      <div className="pace-scroll" ref={scroller}>
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
                <th className="pace-lapcol">{grid.laps[index]}</th>
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
