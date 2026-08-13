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

import type { Bulk } from "../../lib/bridge";
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

export function BestsPanel({ bulk }: { bulk: Bulk | null }) {
  const bests = sessionBests(bulk);
  const theoretical = theoreticalLap(bests);

  return (
    <section className="bests card">
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
    </section>
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

/** `1:25.744` past the minute, `29.412` under it. */
function formatTime(seconds: number): string {
  if (seconds < 60) return seconds.toFixed(3);
  const minutes = Math.floor(seconds / 60);
  const rest = seconds - minutes * 60;
  return `${minutes}:${rest.toFixed(3).padStart(6, "0")}`;
}

/** How far off the section's leader, as the percentage a timing screen shows. */
function formatDelta(value: number, leader: number): string {
  if (leader <= 0) return "";
  return `+${(((value - leader) / leader) * 100).toFixed(2)}%`;
}
