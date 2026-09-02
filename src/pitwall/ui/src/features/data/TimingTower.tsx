/**
 * Band 2, left: the timing tower.
 *
 * `P | # | DRV | GAP | INT | S1 t+v | S2 t+v | S3 t+v | LAST | ST | TYRE | STOPS`,
 * which is the SBG/Catapult RaceX client's row read off the pit-wall
 * photographs in this project's research - including the detail that each
 * sector cell carries two numbers, a time and a trap speed, and that `ST` is
 * the speed trap rather than the stint.
 *
 * **Rows and order come from the TICK; the numbers come from the BULK.** The
 * split is not stylistic. Order changes mid-lap on ticks that change no
 * `laps_completed`, so only the wire can answer it; the seconds are quantised
 * to the line and the parquet is the official clock, which the arcade's own
 * `gaps.py` concedes its interpolated crossings sit a median 22 ms away from.
 *
 * **Iterate `race_order`, never the bulk's keys.** SAI, DOO and HAD have only
 * `FastF1Generated` rows on Melbourne 2025, so they reveal nothing at all -
 * measured, `bulk.drivers.SAI` is present with zero rows for the whole race.
 * A tower keyed on what the bulk contains renders them as blanks it cannot
 * place; `race_order` always carries twenty codes and puts them where the
 * producer's own ranking does.
 *
 * The GAP and INT columns are quantised to the line and the HEADER says so,
 * once, rather than each of the forty cells repeating it.
 */

import { useEffect, useRef, useState } from "react";

import type { ArcadeState, Bulk, DriverLaps, LapRow, LiveLap } from "../../lib/bridge";
import { driverStatus, type DriverStatus } from "../../lib/driverStatus";
import { formatSeconds } from "../../lib/format";
import { formatGapCell, gapCell } from "../../lib/gapCell";
import { sessionBests, type BestField, type SessionBests } from "../../lib/sessionBests";
import { driverColour } from "../../lib/driverColour";

interface TimingTowerProps {
  arcade: ArcadeState;
  bulk: Bulk | null;
  live: LiveLap | null;
  /** The pinned rival, or null while the window follows the producer's choice. */
  pinned: string | null;
  onPin: (code: string | null) => void;
}

/**
 * The car's status, with the tower's own missing-car guard.
 *
 * `driverStatus` takes a car, and a code in `race_order` can have no entry in
 * `drivers` - a relaunched arcade pointed at another race with a pin still set is
 * the reachable case. The bare call reads `undefined.active` there.
 */
function statusOf(arcade: ArcadeState, code: string): DriverStatus {
  const car = arcade.drivers[code];
  return car ? driverStatus(car) : "out";
}

export function TimingTower({ arcade, bulk, live, pinned, onPin }: TimingTowerProps) {
  const order = arcade.race_order;
  const leader = order[0];
  // The same reduction the bests panel ranks, from the same module. Two
  // components each reducing over `bulk.drivers` is how the tower ends up
  // painting a purple the panel does not list.
  const bests = sessionBests(bulk);

  // **A retired car is not selectable, and that is one rule rather than two.**
  // The pin clears when its car retires, so allowing it to be pinned in the
  // first place would be a state the next tick undoes. Retired rows still
  // RENDER - a timing screen classifies retirements, it does not hide them -
  // they simply leave the keyboard order.
  const selectable = order.filter((code) => statusOf(arcade, code) !== "out");
  const selectableKey = selectable.join(",");
  const rows = useRef(new Map<string, HTMLTableRowElement>());
  const [anchor, setAnchor] = useState<string | null>(null);
  // Where the single tab stop sits: the reader's own candidate while it is still
  // selectable, otherwise the pinned row, otherwise the leader.
  const candidate =
    anchor !== null && selectable.includes(anchor)
      ? anchor
      : pinned !== null && selectable.includes(pinned)
        ? pinned
        : (selectable[0] ?? null);

  // **Any row leaving the order moves the anchor, not just the pinned one.**
  // A merely FOCUSED row can retire on the same tick by the same mechanism, and
  // an anchor left on it leaves every row at `tabIndex=-1`, so keyboard entry
  // lands nowhere.
  //
  // Focus itself moves ONLY if the vanished row was holding it. The tower is
  // mounted on every tick whichever tab is showing, so an unconditional move
  // would yank focus out of whatever the reader is actually using.
  useEffect(() => {
    if (anchor === null || selectable.includes(anchor)) return;
    const wasAt = order.indexOf(anchor);
    const replacement =
      selectable.find((code) => order.indexOf(code) >= wasAt) ??
      selectable[selectable.length - 1] ??
      null;
    const held = rows.current.get(anchor);
    const hadFocus = held !== undefined && document.activeElement === held;
    setAnchor(replacement);
    if (hadFocus && replacement !== null) rows.current.get(replacement)?.focus();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [anchor, selectableKey]);

  const moveCandidate = (delta: number) => {
    if (selectable.length === 0) return;
    const at = candidate === null ? 0 : Math.max(0, selectable.indexOf(candidate));
    const next = selectable[Math.min(selectable.length - 1, Math.max(0, at + delta))];
    setAnchor(next);
    rows.current.get(next)?.focus();
  };

  const onKeyDown = (event: React.KeyboardEvent<HTMLTableSectionElement>) => {
    if (event.key === "ArrowDown") {
      event.preventDefault();
      moveCandidate(1);
    } else if (event.key === "ArrowUp") {
      event.preventDefault();
      moveCandidate(-1);
    } else if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      // A second Enter on the pinned row releases it, so the keyboard can undo
      // itself without reaching for Escape.
      if (candidate !== null) onPin(candidate === pinned ? null : candidate);
    } else if (event.key === "Escape") {
      event.preventDefault();
      onPin(null);
    }
  };

  return (
    <section className="tower card">
      {/* **`role="grid"`, chosen rather than inherited.** `aria-selected` is not
       * meaningful on a row outside a grid, and the two roving idioms this repo
       * already ships (`RaceTraceChart`'s reference strip, the DATA tab strip)
       * are `<button>` strips that do not transfer to a `<table>` of `<tr>`.
       * Promoting the table changes its semantics for every screen-reader user,
       * which is why it is stated here rather than left to fall out of the
       * `tabIndex` on the rows. */}
      <table className="tower-table" role="grid" aria-multiselectable="false">
        <thead>
          <tr>
            <th className="col-pos">P</th>
            <th className="col-num">#</th>
            <th className="col-drv">DRV</th>
            {/* The (L) that the arcade suffixes onto every label lives here
             * instead: it is the same claim, made once, on a surface that has
             * a header to make it on. */}
            <th className="col-gap">
              GAP <span className="col-note">(L)</span>
            </th>
            <th className="col-gap">
              INT <span className="col-note">(L)</span>
            </th>
            <th className="col-sector">S1</th>
            <th className="col-sector">S2</th>
            <th className="col-sector">S3</th>
            <th className="col-last">LAST</th>
            <th className="col-st">ST</th>
            <th className="col-tyre">TYRE</th>
            <th className="col-stops">STOPS</th>
          </tr>
        </thead>
        <tbody onKeyDown={onKeyDown}>
          {order.map((code, index) => (
            <TowerRow
              key={code}
              code={code}
              position={index + 1}
              front={index === 0 ? null : order[index - 1]}
              leader={leader}
              arcade={arcade}
              bulk={bulk}
              live={live}
              bests={bests}
              pinned={pinned === code}
              selectable={selectable.includes(code)}
              isCandidate={candidate === code}
              onPin={onPin}
              register={(element) => {
                if (element === null) rows.current.delete(code);
                else rows.current.set(code, element);
              }}
            />
          ))}
        </tbody>
      </table>
    </section>
  );
}

interface TowerRowProps {
  code: string;
  position: number;
  /** The car directly ahead in the published order, or null for the leader. */
  front: string | null;
  leader: string;
  arcade: ArcadeState;
  bulk: Bulk | null;
  live: LiveLap | null;
  bests: SessionBests;
  /** This row's car is the one band 4 is comparing against. */
  pinned: boolean;
  /** A retired car cannot be pinned, so it is not a keyboard stop either. */
  selectable: boolean;
  /** The single tab stop, under the roving tabindex. */
  isCandidate: boolean;
  onPin: (code: string | null) => void;
  register: (element: HTMLTableRowElement | null) => void;
}

function TowerRow({
  code,
  position,
  front,
  leader,
  arcade,
  bulk,
  live,
  bests,
  pinned,
  selectable,
  isCandidate,
  onPin,
  register,
}: TowerRowProps) {
  const car = arcade.drivers[code];
  const status: DriverStatus = car ? driverStatus(car) : "out";
  const laps: DriverLaps | undefined = bulk?.drivers[code];
  // The last lap this driver has actually completed. It can be a generated
  // row - a car that never finished one - and then every number on it is
  // null, which is the honest rendering rather than an absent row.
  const last: LapRow | null = laps?.laps.length ? laps.laps[laps.laps.length - 1] : null;
  // Absent for a car that has retired or taken the flag: it has no lap in
  // progress, so its sector columns are dashes rather than the last lap it
  // happened to drive.
  const sectors = live?.drivers[code];

  const gap = front === null ? { kind: "leader" as const } : gapCell(leader, code, arcade, bulk);
  const interval = front === null ? null : gapCell(front, code, arcade, bulk);

  return (
    <tr
      ref={register}
      className={`tower-row is-${status}${pinned ? " is-pinned" : ""}`}
      // `aria-selected` only on rows that can carry the state at all. On a
      // retired row it would announce "not selected" about a car that can never
      // be selected, which is noise rather than information.
      aria-selected={selectable ? pinned : undefined}
      tabIndex={selectable && isCandidate ? 0 : -1}
      onClick={selectable ? () => onPin(pinned ? null : code) : undefined}
    >
      <td className="col-pos">{position}</td>
      <td className="col-num">{laps?.number ?? "—"}</td>
      {/* **The colour moved off the glyphs and onto a swatch beside them.**
       * `driver_colors` are the arcade's own and correct, and six of the twenty
       * fail AA as TEXT on the card they are drawn on: VER and LAW at 1.88:1,
       * ALO and STR at 2.55, HAM and LEC at 3.71, all at 11 px where 4.5 applies.
       * Four of them fail even the 3.0 large-text floor. The DRV column is the row
       * key of this window's primary panel, so it was the identification that
       * could not be read.
       *
       * The code is `--qt-fg-1` now, **17.48:1** - the tower has no row striping, so
       * it sits on `--qt-panel`; the 15.8 this comment first claimed is white on
       * `--qt-elevated`, which is the PACE GRID's banding and not this ground - and
       * the team colour is a filled bar
       * next to it. A bar is a shape rather than glyphs, so a dim one degrades to
       * "hard to see" instead of "unreadable" - and it is REDUNDANT here, because
       * the code beside it already names the car.
       *
       * The alternative was lifting each failing colour's luminance until it
       * passed. Rejected: it keeps the hue and the legibility but it publishes a
       * colour the arcade never sent, on a window whose whole colour discipline is
       * that `driver_colors` crosses the wire so no consumer invents one. */}
      <td className="col-drv">
        <span className="drv-swatch" style={{ background: driverColour(arcade.driver_colors, code, NO_COLOUR) }} />
        {code}
      </td>
      <td className="col-gap">{formatGapCell(gap)}</td>
      <td className="col-gap">{interval === null ? "—" : formatGapCell(interval)}</td>
      {/* The sector cells ROLL, they do not blank. Each shows the freshest
       * value it has - this lap's once the car has crossed that sector, the
       * previous lap's until then - and `stale` dims the carried-over ones.
       * Every other column on this row is about the last COMPLETED lap, which
       * is what LAST, ST, TYRE and STOPS mean. */}
      <SectorCell
        time={sectors?.s1 ?? null}
        speed={sectors?.v1 ?? null}
        stale={!sectors?.s1_fresh}
        tone={sectorTone(sectors?.s1 ?? null, laps?.best.s1 ?? null, bests, "s1")}
      />
      <SectorCell
        time={sectors?.s2 ?? null}
        speed={sectors?.v2 ?? null}
        stale={!sectors?.s2_fresh}
        tone={sectorTone(sectors?.s2 ?? null, laps?.best.s2 ?? null, bests, "s2")}
      />
      <SectorCell
        time={sectors?.s3 ?? null}
        speed={sectors?.vfl ?? null}
        stale={!sectors?.s3_fresh}
        tone={sectorTone(sectors?.s3 ?? null, laps?.best.s3 ?? null, bests, "s3")}
      />
      <td className={`col-last ${last?.deleted ? "is-deleted" : ""}`}>{lastCell(status, last)}</td>
      <td className="col-st">{last?.vst === null || last?.vst === undefined ? "—" : last.vst}</td>
      <td className="col-tyre">{tyreCell(last)}</td>
      <td className="col-stops">{laps?.stops ?? "—"}</td>
    </tr>
  );
}

/**
 * A sector's time with its trap speed beside it, dimmer and smaller.
 *
 * Inline rather than stacked. Stacking the pair costs six pixels a row, which
 * over twenty rows is 120 px of a window that has none to give, and buys back
 * 62 px of width the column does not need.
 */
function SectorCell({
  time,
  speed,
  tone,
  stale,
}: {
  time: number | null;
  speed: number | null;
  tone: SectorTone;
  /** Carried over from the previous lap rather than set on this one. */
  stale: boolean;
}) {
  return (
    <td className={`col-sector is-${tone}${stale ? " is-stale" : ""}`}>
      {time === null ? "—" : time.toFixed(3)}
      {/* A real space, not only the margin: the cell is read by a screen
       * reader and copied by a mouse, and "29.412301" is a different number. */}
      {speed === null ? null : <span className="cell-speed"> {speed}</span>}
    </td>
  );
}

/** Purple, green, yellow or nothing - the timing screen's four states. */
type SectorTone = "purple" | "green" | "yellow" | "plain";

/**
 * The sector colour code every timing screen uses.
 *
 * Purple is fastest of the session outright, green is the driver's own best,
 * yellow is slower than his own best, and plain is a time with nothing to
 * compare against yet. The comparisons are made on the values the reader
 * served, which already exclude deleted laps and generated rows, so a struck
 * time cannot paint the tower purple.
 *
 * Both bests are recomputed over the REVEALED subset, so a rewind takes the
 * colours back with the laps: the purple that only exists at lap 44 must not
 * survive onto a screen whose clock says lap 10.
 */
function sectorTone(
  value: number | null,
  personalBest: number | null,
  bests: SessionBests,
  field: BestField,
): SectorTone {
  if (value === null) return "plain";
  const sessionBest = bests[field][0];
  if (sessionBest !== undefined && value <= sessionBest.value) return "purple";
  if (personalBest === null) return "plain";
  return value <= personalBest ? "green" : "yellow";
}

/**
 * What the LAST column shows, in the order a timing screen decides it.
 *
 * A broadcast screen puts `IN PIT` or `OUT` **in place of the lap time**, and
 * that is what happens here - except that "OUT" is used for a car that has
 * stopped, matching the arcade leaderboard on screen beside this one, so an
 * out-lap says `PIT EXIT` rather than borrowing the same word for a car that
 * is very much still racing.
 */
function lastCell(status: DriverStatus, last: LapRow | null): string {
  if (status === "out") return "OUT";
  if (last === null) return "—";
  if (last.pit_in) return "IN PIT";
  if (last.pit_out) return "PIT EXIT";
  return last.lap_time === null ? "—" : formatSeconds(last.lap_time, 3);
}

/**
 * The compound's letter and the set's age.
 *
 * The letter is the compound's own first character, which is correct for all
 * five (SOFT, MEDIUM, HARD, INTERMEDIATE, WET) and is therefore not a second
 * copy of the arcade's letter table. The compound arrives from the BULK,
 * which is the repaired frame: the live-timing feed drops stint records and
 * restarts the stint at the recovery lap, so the unrepaired value reads
 * `TyreLife 1` on a set that has done twenty-four racing laps.
 */
function tyreCell(last: LapRow | null): string {
  if (!last?.compound) return "—";
  const age = last.tyre_life === null ? "" : ` ${Math.round(last.tyre_life)}`;
  return `${last.compound[0]}${age}`;
}



/**
 * `--qt-border` for a car the wire has no colour for, not `--qt-fg-1`: as a
 * swatch the old fallback painted a bright white bar, which reads as a team
 * colour rather than as the absence of one.
 */
const NO_COLOUR = "var(--qt-border)";
