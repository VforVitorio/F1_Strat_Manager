/**
 * The radio / RCM feed: what was said, in the order it was said.
 *
 * The RaceX client's `Race Control Messages` panel with the driver radio
 * merged into it, in the column under the ring.
 *
 * **It shows the WHOLE FIELD, and rivals carry a `BROADCAST` tag.** Víctor's
 * call, and his reason is the one that settles the tier question: a real team
 * receives the public broadcast radio feed, so rendering it is fidelity rather
 * than a privilege we grant ourselves. The tag is the same treatment band 4
 * gives a pinned rival's trace - which is what makes the tier discipline
 * visible on screen instead of asserted in a PDF.
 *
 * **The tier is decided HERE and not by the host.** Which car is ours lives on
 * the tick (`arcade.driver_main`), and the feed rides in the bulk payload,
 * whose signature is (year, location, reveal map). A host that baked the tier
 * in would be serving a payload its own signature does not determine, which is
 * exactly what #934 cost a sprint.
 *
 * Newest first, because a pit wall reads the newest line at the top. What
 * falls off the bottom is the OLDEST, and the header's count says how many
 * there are in total so the cut is never silent.
 */

import type { Bulk, RadioEvent } from "../../lib/bridge";

/**
 * Who said it: a driver's code, or `RCM` for race control.
 *
 * Race control's own rows carry no code on purpose - the message already
 * names the car it is about, code included, on all 492 such rows across the
 * 24 races published, so a badge would print it twice.
 */
function whoLabel(event: RadioEvent): string {
  if (event.kind === "rcm") return "RCM";
  return event.driver ?? "--";
}

/** Every radio that is not our own car's is public-feed material. */
function isBroadcast(event: RadioEvent, driverMain: string | null): boolean {
  return event.kind === "radio" && event.driver !== driverMain;
}

/**
 * A radio row with no transcript still renders, with the words missing.
 *
 * That is the corpus reader's own contract, and it is the COMMON case: 23 of
 * the 24 races published have no transcribed audio and there is not one MP3 on
 * disk. That a radio happened, from whom, and on which lap is information; a
 * dropped row would present a race as quieter than it was.
 */
function bodyText(event: RadioEvent): string {
  if (event.text) return event.text;
  return event.kind === "radio" ? "(no transcript)" : "";
}

export function RadioFeed({ bulk, driverMain }: { bulk: Bulk | null; driverMain: string | null }) {
  const feed = bulk?.radio;
  const events = feed?.events ?? [];
  // Newest first. The array arrives oldest-first because that is the order the
  // race happened in and the order the host reveals it in; reversing here
  // rather than there keeps the payload in the shape a future full-history
  // panel would want.
  const newestFirst = [...events].reverse();

  return (
    <section className="card radio-feed">
      <header className="radio-header">
        <span className="radio-title">RADIO · RCM</span>
        {feed?.available ? (
          <span className="radio-count">{events.length}</span>
        ) : (
          <span className="radio-subtitle">no corpus</span>
        )}
      </header>
      {feed?.available && newestFirst.length === 0 ? (
        <p className="radio-empty">Nothing said yet.</p>
      ) : null}
      <ol className="radio-list">
        {newestFirst.map((event, index) => (
          <li
            className={`radio-row is-${event.kind}`}
            // The corpus has no stable id, and two race-control rows of the
            // same lap really can carry the same text (four identical BLUE
            // FLAG lines for the same car on Melbourne's lap 46). The index
            // into a list that is rebuilt whole on every reveal is the honest
            // key; a text-based one would collapse those four into one.
            key={`${event.lap}-${event.kind}-${index}`}
            title={event.text || undefined}
          >
            <span className="radio-lap">L{event.lap}</span>
            <span className="radio-who">{whoLabel(event)}</span>
            <span className="radio-text">
              {isBroadcast(event, driverMain) ? (
                <span className="radio-tier">BROADCAST </span>
              ) : null}
              {bodyText(event)}
            </span>
          </li>
        ))}
      </ol>
    </section>
  );
}
