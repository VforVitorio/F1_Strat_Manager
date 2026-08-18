/**
 * Band 2, right: the session's best sectors and laps.
 *
 * Four ranked sections - S1, S2, S3, Lap - plus the theoretical lap, which is
 * the shape the SBG/Catapult RaceX client uses: *four ranked lists plus
 * Theoretical, each row rank / driver / time / % delta / compound*.
 *
 * **Top three, and that is a structural limit rather than a preference.** A
 * fully ranked section is twenty rows, and four of those are 1,668 px of a
 * window whose whole body is 790. Three rows per section fit the space left
 * under the tower with room to spare, and the fourth-fastest S2 of the
 * afternoon is not a number anybody reads off a live wall.
 *
 * The delta is a PERCENTAGE, not seconds. Across the four sections the
 * absolute gaps differ by an order of magnitude - tenths in a sector,
 * seconds over a lap - and a percentage is the one form that reads the same
 * in all four.
 */

import { useCallback, useEffect, useRef, useState } from "react";

import type { Bulk } from "../../lib/bridge";
import { formatSeconds } from "../../lib/format";
import {
  sessionBests,
  theoreticalLap,
  type BestEntry,
  type BestField,
} from "../../lib/sessionBests";

/** How many rows a section shows. See the note above on why it is not twenty. */
const RANKED = 3;

const SECTIONS: { field: BestField; label: string }[] = [
  { field: "s1", label: "S1" },
  { field: "s2", label: "S2" },
  { field: "s3", label: "S3" },
  { field: "lap_time", label: "LAP" },
];

/**
 * Whether the ranked panel fits the room the left column actually leaves it.
 *
 * **The room is a property of the TOWER, not of this panel, which is what makes
 * this measurable without oscillating.** The tower's grid row is `auto` and its
 * twenty rows are fixed at 437 px, so this card's slot is whatever the column
 * has left - 303 px at the 1485 x 833 client and **63 px** at the 1265 x 593 one
 * a 1080p laptop at 150 % scaling produces. It does not move when this panel
 * renders less.
 *
 * The full panel's own natural height is latched the one time it renders, and
 * never re-read in the compact form. Comparing the room against `scrollHeight`
 * of whatever is currently rendered is the version that flips forever: 63 < 153
 * chooses compact, compact measures 54, 63 >= 54 chooses full again.
 */
function useFitsRanked(card: HTMLElement | null): boolean {
  const [compact, setCompact] = useState(false);
  const rankedHeight = useRef<number | null>(null);

  const fit = useCallback(() => {
    const column = card?.parentElement;
    if (!card || !column) return;
    if (!compact) rankedHeight.current = card.scrollHeight;
    const needed = rankedHeight.current;
    if (needed === null) return;
    const room = column.getBoundingClientRect().bottom - card.getBoundingClientRect().top;
    setCompact(room < needed);
  }, [card, compact]);

  useEffect(() => {
    const column = card?.parentElement;
    if (!column) return;
    fit();
    const observer = new ResizeObserver(fit);
    observer.observe(column);
    return () => observer.disconnect();
  }, [card, fit]);

  return !compact;
}

export function BestsPanel({ bulk }: { bulk: Bulk | null }) {
  const bests = sessionBests(bulk);
  const theoretical = theoreticalLap(bests);
  // State rather than a plain ref: the observer has to be attached to the node
  // the FIRST time it exists, and a ref's assignment does not re-run an effect.
  // The same lesson `useEChart` learned when a remounted panel left a dead chart.
  const [card, setCard] = useState<HTMLElement | null>(null);
  const ranked = useFitsRanked(card);

  return (
    <section className="bests card" ref={setCard}>
      {/* The compact form drops the header ROW too, not just the ranks: it puts
       * the title into the same flex run as the values. Keeping a separate 16 px
       * header plus its 8 px gap left the card 65 px tall in a 63 px slot -
       * fighting for two pixels, which a font fallback would take straight back.
       *
       * **It is not literally one line, and the version of this comment that said
       * "one line has 24 px of air" was wrong.** Measured at the 1265 x 593 client:
       * `.bests-leaders` is 40 px, two wrapped flex lines, with THEO on the second
       * one - where its `margin-left: auto` happens to read as a deliberate footer.
       * The card comes to 62 px in a 63 px slot, so it FITS, but the margin is one
       * wrap: a slightly narrower client or a wider fallback font puts a second
       * entry on line two and pushes THEO under an unannounced fold, which is the
       * failure this whole degradation exists to remove. The next step down the
       * ladder, if that ever bites, is to drop the VALUES and keep the codes. */}
      {ranked ? (
        <>
          <header className="bests-header">
            <span className="bests-title">BESTS</span>
            <span className="bests-subtitle">session, revealed laps only</span>
          </header>

          <div className="bests-sections">
            {SECTIONS.map(({ field, label }) => (
              <BestsSection key={field} label={label} entries={bests[field].slice(0, RANKED)} />
            ))}
          </div>

          <footer className="bests-theoretical">
            <span className="bests-field-label">THEORETICAL</span>
            <span className="bests-theoretical-value">
              {theoretical === null ? "—" : formatTime(theoretical)}
            </span>
            <span className="bests-theoretical-note">
              {theoretical === null
                ? "waiting for a sector nobody has set yet"
                : `${bests.s1[0].code} · ${bests.s2[0].code} · ${bests.s3[0].code}`}
            </span>
          </footer>
        </>
      ) : (
        <BestsLeaders bests={bests} theoretical={theoretical} />
      )}
    </section>
  );
}

/**
 * One line: who holds each purple, and the theoretical.
 *
 * **Not "one ranked row per section", which is what the design gate prescribed
 * and what does not fit.** Measured at the 1265 x 593 client: the card's slot is
 * 63 px and its chrome alone - 20 px of padding, a 16 px header, two 8 px gaps
 * and the 27 px theoretical footer - is 79 before a single row exists. One row
 * per section needs 112. The ranks are not shrinkable here; they are droppable.
 *
 * What survives is what a wall reads off this panel at a glance: the four purple
 * holders and the theoretical lap. Ranks 2 and 3 go, and the header says so
 * rather than leaving the reader to think the panel is one row tall - which is
 * exactly what the silent clip did.
 */
function BestsLeaders({
  bests,
  theoretical,
}: {
  bests: ReturnType<typeof sessionBests>;
  theoretical: number | null;
}) {
  return (
    <div className="bests-leaders">
      <span className="bests-leader">
        <span className="bests-title">BESTS</span>
        <span className="bests-subtitle">leaders</span>
      </span>
      {SECTIONS.map(({ field, label }) => {
        const leader = bests[field][0];
        return (
          <span className="bests-leader" key={field}>
            <span className="bests-field-label">{label}</span>
            {leader === undefined ? (
              <span className="bests-empty">—</span>
            ) : (
              <>
                <span className="bests-code">{leader.code}</span>
                <span className="bests-value">{formatTime(leader.value)}</span>
              </>
            )}
          </span>
        );
      })}
      <span className="bests-leader is-theoretical">
        <span className="bests-field-label">THEO</span>
        <span className="bests-theoretical-value">
          {theoretical === null ? "—" : formatTime(theoretical)}
        </span>
      </span>
    </div>
  );
}

function BestsSection({ label, entries }: { label: string; entries: BestEntry[] }) {
  const leader = entries[0];

  return (
    <div className="bests-section">
      <div className="bests-field-label">{label}</div>
      {entries.length === 0 ? (
        <div className="bests-empty">—</div>
      ) : (
        entries.map((entry, index) => (
          <div key={entry.code} className={`bests-row ${index === 0 ? "is-purple" : ""}`}>
            <span className="bests-rank">{index + 1}</span>
            <span className="bests-code">{entry.code}</span>
            <span className="bests-value">{formatTime(entry.value)}</span>
            <span className="bests-delta">
              {index === 0 ? "" : formatDelta(entry.value, leader.value)}
            </span>
            <span className="bests-compound">{entry.compound ? entry.compound[0] : ""}</span>
          </div>
        ))
      )}
    </div>
  );
}

/** `1:25.744` past the minute, `29.412` under it. One arithmetic, in `lib/format`. */
const formatTime = (seconds: number) => formatSeconds(seconds, 3);

/** How far off the section's leader, as the percentage a timing screen shows. */
function formatDelta(value: number, leader: number): string {
  if (leader <= 0) return "";
  return `+${(((value - leader) / leader) * 100).toFixed(2)}%`;
}
