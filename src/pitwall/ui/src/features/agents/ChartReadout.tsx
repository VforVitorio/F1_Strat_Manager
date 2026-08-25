/**
 * The hover box both AGENTS cards share (#999).
 *
 * One component rather than one per card, because the two boxes would otherwise
 * be the same twenty lines twice - and a pair of copies where one gets a fix and
 * the other does not is this repository's most productive defect.
 *
 * It is deliberately small. These two charts plot three or four quantities at a
 * lap, not twenty, so the box is a lap number and a short list; the race trace's
 * list of the whole field is a different shape and lives with that panel.
 */

import type { ChartHover } from "../../lib/chartHover";

/** One quantity at the hovered lap. `null` prints an em dash. */
export interface ReadoutRow {
  label: string;
  value: string | null;
  /** The series' own colour, so a row and the line it describes agree. */
  colour: string;
}

/**
 * A series' y at an exact lap, formatted, or null where it has no point there.
 *
 * Laps are integers, so this is a lookup rather than an interpolation: a
 * predicted lap time "at lap 23.4" is not a quantity these models produce.
 */
export function at(points: [number, number][], lap: number): string | null {
  const found = points.find(([atLap]) => atLap === lap);
  return found === undefined ? null : found[1].toFixed(1);
}

/**
 * The box, opening on whichever side of the cursor has room.
 *
 * `position: absolute` inside the card, never `fixed`. The AGENTS body takes a
 * `filter` when the feed freezes, exactly as the DATA window's does, and a
 * filter becomes the containing block for fixed descendants - so a fixed box
 * would jump the moment the feed stopped. Measured on the DATA window at 50 px.
 */
export function ChartReadout({
  hover,
  lap,
  rows,
}: {
  hover: ChartHover;
  lap: number;
  rows: ReadoutRow[];
}) {
  const side =
    hover.pixelX > hover.hostWidth / 2
      ? { right: hover.hostWidth - hover.pixelX + 8 }
      : { left: hover.pixelX + 8 };
  return (
    <div className="chart-readout" style={side}>
      <span className="chart-readout-lap">LAP {lap}</span>
      {rows.map((row) => (
        <span key={row.label} className="chart-readout-row">
          <span className="chart-readout-label" style={{ color: row.colour }}>
            {row.label}
          </span>
          <span className="chart-readout-value">{row.value ?? "—"}</span>
        </span>
      ))}
    </div>
  );
}
