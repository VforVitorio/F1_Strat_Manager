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

/**
 * The FLOOR for how many rows a section shows, and the cap. The room decides.
 *
 * **Three used to be a constant, and the sentence justifying it was true of a slot
 * that no longer exists.** The panel's own docstring argued that "the
 * fourth-fastest S2 of the afternoon is not a number anybody reads off a live wall"
 * - and it argued that against the 63 px this card gets at the narrow client, not
 * against the 303 px the agreed layout drawing gives it at the wide one, where
 * three ranks leave 150 px of nothing. Victor saw that hole and called it out.
 *
 * **The cap is the FIELD, and it used to be ten.** Ten was chosen because P10 is the
 * last point-scoring position - a real argument about relevance, and the wrong one for
 * a cap, because a cap only bites when there is room to spare. Measured at 1485 px
 * wide: the ramp works to 833 px tall (3 -> 6 -> 10 ranks, 17-24 px of air left) and
 * then the depth sticks at ten while the hole grows - **91 px at 900, 191 at 1000, 271
 * at 1600x1080**. Víctor saw exactly that on his own screen, one sprint after seeing
 * the 150 px version of it.
 *
 * Twenty is the whole grid, so the panel now runs out of DATA before it runs out of
 * room, and the leftover then means "there is no more", which explains itself. The
 * subtitle still names the depth, so nothing is claimed that is not shown.
 *
 * **The FLOOR is one, and three was the other half of the same mistake as the cap.**
 * Raising the cap fixed the wide client and left the narrow one degrading
 * all-or-nothing: the card's height is linear in depth at 99 + 18k px, so a
 * floor of three meant the panel jumped straight from a 153 px ranked card to
 * the 62 px compact one with nothing in between. Measured across the client
 * heights `WindowSpec.place` produces, the 16:10 laptops land at 143 px of room:
 * enough for a depth-2 card at 135 and eight pixels to spare, and the panel
 * discarded all of it, showing four leader values where it had room for eight
 * ranked ones. A rank-1 list is still the four best holders plus THEORETICAL,
 * which is the panel's whole job; nothing about it needed a floor of three.
 *
 * What this does NOT reach is the 1366x768 client, whose 111 px of room fits
 * neither the 117 px depth-1 card nor anything above the compact form. That band
 * is 49 px of waste on one screen class, and a third form to serve it would be
 * more form than defect.
 */
const RANKED_FLOOR = 1;
const RANKED_CAP = 20;
/**
 * Rounding tolerance, and no longer a stand-in for a font that has not landed.
 *
 * **This was 8 px, and 8 px of caution cost 102 px of hole.** It existed because
 * `--qt-mono` arrives after first layout and grows the card by about that much, so a
 * boundary fit could become a clip. But it is subtracted from `room` before the
 * ranked-or-compact decision too, and at 1350x673 the room is 153 px against a
 * 153 px floor-depth card: the guard pushed it under, the panel fell all the way to
 * its 41 px one-line form, and left **102 px of nothing** - a worse version of the
 * hole this whole mechanism exists to close.
 *
 * The honest signal for "the font has landed" is `document.fonts.ready`, which the
 * effect below now waits on, plus the ResizeObserver that was already watching the
 * card's own box. With a real signal in place this is just sub-pixel slack.
 */
const FONT_GUARD = 2;

/**
 * What the room allows: the ranked form or the compact one, and how deep.
 *
 * One value rather than two pieces of state, so a render can never see "ranked"
 * paired with the other form's depth.
 */
interface Fit {
  ranked: boolean;
  depth: number;
}

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
function useFitsRanked(card: HTMLElement | null, content: number): Fit {
  const [fitState, setFitState] = useState<Fit>({
    ranked: true,
    depth: RANKED_FLOOR,
  });
  /** The card's height at the FLOOR depth, latched while it is showing exactly that. */
  const floorHeight = useRef<number | null>(null);
  /** The last row height actually rendered, so the compact form can still decide. */
  const lastRowHeight = useRef<number | null>(null);

  const fit = useCallback(() => {
    const column = card?.parentElement;
    if (!card || !column) return;
    // **Latched from ANY ranked depth, normalised back to the floor.**
    //
    // The first version of this latched only while the card was showing exactly the
    // floor depth - and the arithmetic below MOVES it off the floor, so after the
    // first bump the condition never held again and the latch kept the height of the
    // empty panel. Measured: it clipped 72 px at 1265x650, which is the same defect
    // this hook was hardened against one commit earlier, re-entering through a door
    // the fix itself opened.
    //
    // Normalising is exact rather than approximate: `.bests-sections` is a
    // four-COLUMN grid, so one extra rank adds one row height to the card, not four.
    // And it is only ever read while the card is ranked, because the compact card's
    // height says nothing about what the ranked one would need - that is why the
    // latch exists at all.
    // **The row height is READ, never copied.** It was a `17` here against a stylesheet
    // that renders 18, and the error is per-row: harmless at a depth of three, ten px of
    // drift at ten, and with the cap now at twenty it would be seventeen - enough to turn
    // a fit into a clip. The pace grid's own row-height comment already demands this
    // ("nothing in TSX may copy this number - a consumer reads it with getComputedStyle");
    // this panel had the copy anyway.
    // **Latched as well as read, because the compact form has no row to read.**
    //
    // `.bests-row` only exists while the panel is ranked, so in the compact form
    // this measured 0 and the early return below fired before any decision was
    // made. That made the degradation ONE-WAY: measured on the real page, a
    // window opened at 833 ranked 11 deep, shrank to 593 and went compact, and
    // then grew back to 833 - room 303, the same room it had ranked in a moment
    // earlier - and stayed compact for the rest of the session. Every later
    // resize was decided by an early return rather than by the room.
    //
    // The row height is a stylesheet constant at a given client, so carrying the
    // last ranked measurement across is honest rather than a guess, and it is
    // still READ rather than copied from a number in this file.
    const measuredRow =
      card.querySelector(".bests-row")?.getBoundingClientRect().height ?? 0;
    if (measuredRow > 0) lastRowHeight.current = measuredRow;
    const rowHeight = measuredRow > 0 ? measuredRow : (lastRowHeight.current ?? 0);
    if (rowHeight <= 0) return;
    if (fitState.ranked) {
      const extraRows = Math.max(0, fitState.depth - RANKED_FLOOR);
      floorHeight.current = card.scrollHeight - extraRows * rowHeight;
    }
    const atFloor = floorHeight.current;
    if (atFloor === null) return;
    const room =
      column.getBoundingClientRect().bottom - card.getBoundingClientRect().top;
    if (room < atFloor) {
      setFitState({ ranked: false, depth: RANKED_FLOOR });
      return;
    }
    // **How many MORE rows the leftover room buys, per section.**
    //
    // A function of `room` alone, which is what keeps it from oscillating: the room
    // is a property of the tower and the column, and neither moves when this card
    // renders more rows. `useFitsRanked` earned that argument once already; this
    // extends it one step rather than inventing a second mechanism.
    const spare = room - atFloor - FONT_GUARD;
    const extra = Math.max(0, Math.floor(spare / rowHeight));
    setFitState({
      ranked: true,
      depth: Math.min(RANKED_FLOOR + extra, RANKED_CAP),
    });
  }, [card, fitState.ranked, fitState.depth]);

  // **`content` is in here because the card mounts EMPTY, and that was a P1.**
  // The card appears with the first TICK; its rows come from the BULK, which
  // arrives on its own poll a moment later. Latching the ranked height on mount
  // therefore latched the height of the EMPTY panel - measured, 114 px against a
  // populated 151 - so at any room between those two numbers the panel committed
  // to `ranked` and then clipped, THEORETICAL included, with no tell: the card is
  // capped by `max-height` and this window hides scrollbars.
  //
  // The observer alone could not save it. It watches the COLUMN, and the column's
  // grid rows do not move when the bests data lands, so it never fired - a forced
  // 1 px viewport change did not re-decide it either. Measured over six fresh
  // mounts per size: 0 px hidden at both settled clients, and 33 px hidden at
  // 1265x650, 23 at 1350x660, 10 at 1350x673. The defect lived exactly between the
  // two numbers anybody had measured, and `WindowSpec.place` makes the client
  // height a continuous function of the screen, so ordinary machines land in it.
  //
  // `RacePaceGrid`'s sibling `fit` already keys on its own content
  // (`grid.columns.length`); the asymmetry between the two WAS the bug.
  // **And the content signature alone is not enough**, which the guard at 1350x660
  // caught: the row count settles and the card still GROWS afterwards, because
  // `--qt-mono` is a web font and the rows are shorter until it swaps in. Measured,
  // 8 px - enough to clip at that client, and invisible to any signature derived
  // from the data.
  //
  // So the observer watches the CARD as well as the COLUMN: one for "the content
  // changed size", one for "the room changed". It still cannot oscillate, because
  // the ranked height is only re-measured while the panel IS ranked - going compact
  // shrinks the card, the observer fires, and `fit` compares the same latched height
  // against the same room and keeps the same answer.
  // **Re-fit once the webfont has actually arrived**, rather than reserving pixels
  // against the possibility. `document.fonts.ready` is the signal; the observer below
  // catches everything else that changes the card's box.
  useEffect(() => {
    let live = true;
    document.fonts?.ready.then(() => {
      if (live) fit();
    });
    return () => {
      live = false;
    };
  }, [fit]);

  useEffect(() => {
    const column = card?.parentElement;
    if (!card || !column) return;
    fit();
    const observer = new ResizeObserver(fit);
    observer.observe(column);
    observer.observe(card);
    return () => observer.disconnect();
  }, [card, fit, content]);

  return fitState;
}

export function BestsPanel({ bulk }: { bulk: Bulk | null }) {
  const bests = sessionBests(bulk);
  const theoretical = theoreticalLap(bests);
  // State rather than a plain ref: the observer has to be attached to the node
  // the FIRST time it exists, and a ref's assignment does not re-run an effect.
  // The same lesson `useEChart` learned when a remounted panel left a dead chart.
  const [card, setCard] = useState<HTMLElement | null>(null);
  // How much this panel HAS to show, which is what its height depends on: the
  // ranked rows across the four sections plus the theoretical. `bulk.rev` would
  // also work and would re-measure ten times more often for no gain.
  const content =
    SECTIONS.reduce(
      (total, { field }) => total + Math.min(bests[field].length, RANKED_CAP),
      0,
    ) + (theoretical === null ? 0 : 1);
  const fit = useFitsRanked(card, content);

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
      {fit.ranked ? (
        <>
          <header className="bests-header">
            <span className="bests-title">BESTS</span>
            {/* The depth is said out loud, because it is not the same at every
                client and two readers comparing panels must not be comparing two
                silently different lists. */}
            <span className="bests-subtitle">
              session, revealed laps only · top {fit.depth}
            </span>
          </header>

          <div className="bests-sections">
            {SECTIONS.map(({ field, label }) => (
              <BestsSection
                key={field}
                label={label}
                entries={bests[field].slice(0, fit.depth)}
              />
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
                {/* The compact form shows nothing BUT leaders, so the same flash
                 * belongs here. Without it the item would be desktop-only and the
                 * two clients that degrade to this form would be the ones with no
                 * cue at all. */}
                <span className="bests-value" key={leader.value}>
                  {formatTime(leader.value)}
                </span>
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

function BestsSection({
  label,
  entries,
}: {
  label: string;
  entries: BestEntry[];
}) {
  const leader = entries[0];

  return (
    <div className="bests-section">
      <div className="bests-field-label">{label}</div>
      {entries.length === 0 ? (
        <div className="bests-empty">—</div>
      ) : (
        entries.map((entry, index) => (
          <div
            key={entry.code}
            className={`bests-row ${index === 0 ? "is-purple" : ""}`}
          >
            <span className="bests-rank">{index + 1}</span>
            <span className="bests-code">{entry.code}</span>
            {/* **Keyed on the VALUE, and only on the leader.** A record deserves
             * to announce itself, and the purple row is what "a new session
             * best" means here. Keying the number rather than the row means the
             * flash lands on the thing that changed while the row keeps its
             * identity through a re-rank; keying only the leader means the four
             * section leaders flash and the forty rows below them do not, which
             * is the difference between an announcement and a busy panel. */}
            <span
              className="bests-value"
              key={index === 0 ? entry.value : undefined}
            >
              {formatTime(entry.value)}
            </span>
            <span className="bests-delta">
              {index === 0 ? "" : formatDelta(entry.value, leader.value)}
            </span>
            <span className="bests-compound">
              {entry.compound ? entry.compound[0] : ""}
            </span>
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
