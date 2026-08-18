/**
 * The race-pace grid: one cell per driver per lap, coloured by how that lap
 * ranked AGAINST THE SAME LAP.
 *
 * This is the RaceX client's Run Timeline, the panel the research found
 * occupies more screen area than anything else on a real wall.
 *
 * **The colour is a RANK within the lap, not a percentage off a session best,
 * and the difference is the whole design.** Measured on the real Melbourne
 * 2025 payload, over the 776 rows this grid actually RANKS: against the
 * session's fastest lap the median lap is +13.03 % and 80.7 % of the race falls
 * past +10 %, because the race was wet and ran safety cars - so any fixed
 * percentage band paints four fifths of the grid in one colour and says
 * nothing. (This pair used to read +13.79 % / 82.4 %, which is the same
 * arithmetic over 858 rows including the 82 pit laps and 6 deleted times the
 * ranking EXCLUDES and the grid never colours - a figure describing a
 * population the sentence is not about. An earlier gate certified it as "exact"
 * by reproducing the same wrong population: replicating arithmetic is not
 * confirming a claim.) Ranked inside its own lap instead, the field
 * splits into thirds by construction on every lap of every race, wet or dry,
 * green or neutralised, with no threshold tuned on the one race that happens
 * to be on this disk. It is also what "race pace" means to a strategist: who
 * is quick RIGHT NOW, not who set the best lap two hours ago.
 *
 * The absolute time is in the cell, so the colour never has to carry a value
 * the reader cannot check.
 *
 * One cell is purple: the session's outright fastest revealed lap. That is the
 * same rule the tower's sector cells use, and it comes from the same module
 * (`sessionBests`) rather than a second reduction over `bulk.drivers` - two
 * panels disagreeing about which lap was quickest is the twin this repo pays
 * for most often.
 */

import type { Bulk, LapRow } from "./bridge";
import { sessionBests } from "./sessionBests";

/**
 * What a cell paints. `best` is the session's fastest; `t1`/`t2`/`t3` are the
 * thirds of the lap's own ranking; `pit` and `out` are the in-lap and out-lap;
 * `deleted` is a time the stewards took away; `none` is a lap the driver has
 * no time for, revealed or not.
 */
export type PaceTone = "best" | "t1" | "t2" | "t3" | "pit" | "out" | "deleted" | "none";

export interface PaceCell {
  text: string;
  tone: PaceTone;
}

export interface PaceGrid {
  /** Lap numbers, ascending. Empty until something is revealed. */
  laps: number[];
  /** Driver codes in WIRE order, which is the only order that holds mid-lap. */
  columns: string[];
  /** `rows[i][j]` is lap `laps[i]` for driver `columns[j]`. */
  rows: PaceCell[][];
}

const EMPTY: PaceCell = { text: "", tone: "none" };

/**
 * The column order: every driver the wire names, sorted by CAR NUMBER.
 *
 * **Which drivers comes from the wire, the order does not, and the two halves
 * have different reasons.** `race_order` is the wire's own list of who is in
 * this race, published by the producer that ranks them, so the grid never
 * re-derives it. (It is NOT true that the bulk would render seventeen columns:
 * `SessionLaps.masked_view` iterates every driver it loaded, so a car with
 * nothing revealed still gets an entry with an empty `laps` array - measured
 * on the live host, `bulk.drivers` has all twenty keys at every reveal. That
 * claim was wrong in this docstring, in the smoke's own message and in the
 * pull request; the design stands, its stated reason did not.) But
 * `race_order` is ranked by POSITION, and a
 * position-ranked grid re-sorts its own columns every time two cars swap: the
 * reader looks back at lap 12 and it is in a different place than it was a
 * moment ago. A car number never changes for the whole race, which is what a
 * history panel needs and what a timing sheet has always used.
 *
 * A driver with no number sorts to the END, in wire order among his own kind.
 * **Not "keeps his wire position", which is what this said and did.** Two
 * different orderings inside one comparator is not a consistent order: with
 * A(44), B(no number) and C(1) it answers A<B, B<C and C<A, and the result is
 * whatever the engine's sort happens to do. Executed on the real bundle with
 * one unknown at wire index 1, car 44 rendered FIRST, ahead of car 1. Sorting
 * on a pair - the number, then the wire index - is a total order by
 * construction and cannot do that.
 *
 * Exported because the race trace orders its series the same way and for the
 * same reason. A second copy of this comparator is the twin defect this repo
 * pays for more than any other - and it would be the third copy of the sort,
 * not the second, since the version that shipped here was already wrong once.
 */
export function stableColumns(bulk: Bulk, order: string[]): string[] {
  const keyed = order.map((code, index) => {
    const raw = bulk.drivers[code]?.number;
    const number = raw === null || raw === undefined ? null : Number.parseInt(raw, 10);
    return { code, index, number: Number.isNaN(number as number) ? null : number };
  });
  keyed.sort((left, right) => {
    const byNumber = (left.number ?? Infinity) - (right.number ?? Infinity);
    return byNumber !== 0 ? byNumber : left.index - right.index;
  });
  return keyed.map((entry) => entry.code);
}

/**
 * `m:ss.d` - the broadcast form, rounded to tenths.
 *
 * **Rounded, and the minutes are split off AFTERWARDS, which is not a detail.**
 * Splitting first and rounding the remainder renders a non-time in the 50 ms
 * under every minute boundary: 119.96 s came out as `1:60.0`, and the smoke's
 * `^\d:\d\d\.\d$` accepts that happily. No lap of the one race on disk
 * reaches such a value; a season will. (This docstring also used to say
 * "truncated", which it never did.)
 *
 * **The truncation is what makes the grid fit at all.** Measured against the
 * real built stylesheet at the right column's real width: twenty columns of
 * 38.75 px hold this form with room to spare, while the same cell in seconds
 * (`149.413`) clips **205 of 1,140 cells on the real Melbourne payload**, at
 * all three client heights the fleet produces. (An earlier version of this
 * sentence said 793, which was measured on a prototype whose synthetic times
 * were spread across the race's whole range rather than on the payload the
 * window serves - the wrong-distribution class, in a comment. The smoke's own
 * smoke's own fixture clips materially fewer than the real race, so an "it
 * fits" measured on the fixture is weaker evidence than one measured on the
 * payload. No count is quoted here on purpose: the last one was written four
 * minutes after a commit changed the fixture underneath it and was stale on
 * arrival.)
 *
 * Tenths is also the resolution the grid needs, because the ranking colour
 * carries the ordering and the number carries the magnitude.
 *
 * **It runs to seven characters from ten minutes upward** (`600 -> "10:00.0"`).
 * No lap on disk approaches it - Melbourne's slowest is 149.413 s and nothing
 * reaches 300 - but a red-flagged race would, and no race with a red flag is
 * downloadable here to test it.
 *
 * **`coarse` drops the tenths, and it exists because the column does not always
 * have room for them.** On a 1080p laptop at 150 % scaling - Windows' own
 * recommended scaling for a 13-14" screen - this window's client area is
 * 1265 x 593, the grid's twenty columns fall to 27.75 px, and the six-glyph form
 * measures 35: `1:59.4` rendered `1:59.`, `IN PIT` rendered `IN PI`, on 495 of
 * 514 populated cells, with `overflow: hidden` and globally hidden scrollbars so
 * nothing said so. A trailing dot was the only tell.
 *
 * Four glyphs fit, so the choice was which four. `ss.d` keeps the tenths and
 * drops the minute, which is more information for the panel's own question - the
 * field is normally inside one minute of itself, so the tenths is the whole
 * comparison. It is rejected anyway: on this race the safety-car laps run 2:19
 * against a green 1:29, so `19.7` beside `59.4` reads as forty seconds apart
 * when it is twenty, and a cell that can be MISREAD is worse than one that is
 * openly coarser. `m:ss` loses precision the header states out loud; it never
 * lies about magnitude. The tone still carries the ranking, which is where this
 * panel puts the ordering anyway.
 */
export function paceLabel(seconds: number, coarse = false): string {
  if (coarse) {
    const whole = Math.round(seconds);
    const minutes = Math.floor(whole / 60);
    return `${minutes}:${String(whole - minutes * 60).padStart(2, "0")}`;
  }
  const tenths = Math.round(seconds * 10);
  const minutes = Math.floor(tenths / 600);
  const rest = (tenths - minutes * 600) / 10;
  return `${minutes}:${rest.toFixed(1).padStart(4, "0")}`;
}

/**
 * The lap a driver has for that number, or null.
 *
 * A `generated` row is FastF1's synthetic stand-in for a car that did not
 * finish the lap. It is rendered as nothing and counts towards nothing - its
 * `Time` stamp sorts before the entire field, which is how a naive ranking
 * puts the lap-1 crashers on the podium.
 */
function timedRow(row: LapRow | undefined): LapRow | null {
  if (row === undefined || row.generated) return null;
  return row;
}

/** In and out laps are not racing laps and never enter a ranking. */
function isRacingLap(row: LapRow): boolean {
  return !row.pit_in && !row.pit_out && !row.deleted && row.lap_time !== null;
}

/**
 * Each lap's ranked racing times, so a cell can ask where it placed.
 *
 * Built once for the whole grid rather than per cell: a per-cell scan is
 * O(drivers^2) per lap and this runs on every reveal.
 */
function rankedByLap(byDriver: Map<string, Map<number, LapRow>>, laps: number[]) {
  const ranked = new Map<number, number[]>();
  for (const lap of laps) {
    const times: number[] = [];
    for (const rows of byDriver.values()) {
      const row = timedRow(rows.get(lap));
      if (row && isRacingLap(row)) times.push(row.lap_time as number);
    }
    times.sort((left, right) => left - right);
    ranked.set(lap, times);
  }
  return ranked;
}

/**
 * Which third of the lap this time placed in.
 *
 * The boundaries are compared against a FRACTIONAL third rather than a rounded
 * one, so a field of four splits 2/1/1 instead of leaving the last third empty.
 * A lap with fewer than three timed cars has no meaningful thirds and
 * everything in it reads as the top one - which is honest: with two cars
 * running, both ARE the field.
 *
 * **A value that is not in the ranking is not the fastest third.** It used to
 * be: `indexOf` answers -1 for a time the ranking excludes, and the guard read
 * `index < 0 || times.length < 3` and returned `t1` for both. A deleted lap
 * took that branch and was painted the green the legend means as "quickest
 * third" - measured on the real race, GAS's lap 54 ranked 14th of 14 and
 * rendered the same colour as the quickest. Callers now hand deleted rows
 * their own tone and never reach here; if anything else ever does, `none` is
 * the honest answer, because "we do not know where this placed" is not a claim
 * about pace.
 */
function tone(times: number[], value: number): PaceTone {
  if (times.length < 3) return "t1";
  const index = times.indexOf(value);
  if (index < 0) return "none";
  const third = times.length / 3;
  if (index < third) return "t1";
  if (index < third * 2) return "t2";
  return "t3";
}

/**
 * The grid for what the clock has revealed.
 *
 * Columns come from the wire's `order` and are sorted by car number; see
 * `stableColumns` for why each half is decided where it is.
 *
 * The bottom edge is RAGGED on purpose: the reveal is per driver and strict,
 * and at most instants the field spans two or three different laps.
 */
export function racePaceGrid(bulk: Bulk | null, order: string[], coarse = false): PaceGrid {
  if (!bulk?.available) return { laps: [], columns: [...order], rows: [] };
  const columns = stableColumns(bulk, order);

  const byDriver = new Map<string, Map<number, LapRow>>();
  let lastLap = 0;
  for (const code of columns) {
    const rows = new Map<number, LapRow>();
    for (const row of bulk.drivers[code]?.laps ?? []) {
      rows.set(row.lap, row);
      if (!row.generated && row.lap > lastLap) lastLap = row.lap;
    }
    byDriver.set(code, rows);
  }

  const laps = Array.from({ length: lastLap }, (_, index) => index + 1);
  const ranked = rankedByLap(byDriver, laps);
  const fastest = sessionBests(bulk).lap_time[0] ?? null;

  const rows = laps.map((lap) =>
    columns.map((code) => {
      const row = timedRow(byDriver.get(code)?.get(lap));
      if (row === null) return EMPTY;
      // The two word cells shorten with the numbers. `IN PIT` is six glyphs, the
      // same six the times could not fit, so leaving it alone would have kept
      // the P0 alive in the one place the reader most needs the whole word -
      // `IN PI` ran straight into its neighbour as `IN PIIN PI`.
      if (row.pit_in) return { text: coarse ? "PIT" : "IN PIT", tone: "pit" as PaceTone };
      if (row.pit_out) return { text: "OUT", tone: "out" as PaceTone };
      if (row.lap_time === null) return EMPTY;
      const text = paceLabel(row.lap_time, coarse);
      // A deleted time is shown and struck through, as the tower already shows
      // it, and it is excluded from the ranking - which is why it cannot be
      // given a rank here. Before this branch existed it fell through to
      // `tone`, whose `indexOf` answered -1, and -1 rendered as the quickest
      // third: on the real race the slowest car on the lap wore the same green
      // as the fastest.
      if (row.deleted) return { text, tone: "deleted" as PaceTone };
      // The session's fastest lap is purple wherever it appears, matching the
      // tower's sector code. Compared on the VALUE and the driver together:
      // two drivers can set the identical time to the millisecond, and only
      // one of them set the session's best.
      if (fastest && fastest.code === code && fastest.value === row.lap_time) {
        return { text, tone: "best" as PaceTone };
      }
      return { text, tone: tone(ranked.get(lap) ?? [], row.lap_time) };
    }),
  );
  return { laps, columns, rows };
}
