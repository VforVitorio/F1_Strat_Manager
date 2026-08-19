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

import { useCallback, useEffect, useRef, useState } from "react";

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

/**
 * What KIND of race-control message this is, in three characters.
 *
 * **`category` and `flag` ride the wire and were read by nothing.** Measured on the
 * live payload: 58 events carrying `category` in {Flag 20, Other 24, SafetyCar 3,
 * Drs 1} and `flag` in {CLEAR 11, BLUE 5, DOUBLE YELLOW 4}. Every one of them
 * rendered in the same dress, so `SAFETY CAR DEPLOYED` looked exactly like the
 * twenty-fourth `FIA STEWARDS: NO FURTHER ACTION` in a ten-row fold. The chip is
 * what makes the fold's content rankable at a glance.
 *
 * `Other` gets no chip: a badge on two thirds of the rows is a badge on none of
 * them, and the absence is then itself a signal.
 */
function categoryChip(
  event: RadioEvent,
): { text: string; tone: string } | null {
  if (event.kind !== "rcm") return null;
  if (event.category === "SafetyCar") return { text: "SC", tone: "is-sc" };
  if (event.category === "Drs") return { text: "DRS", tone: "is-drs" };
  if (event.category === "Flag") {
    // The flag itself when the wire names one, because BLUE and DOUBLE YELLOW are
    // different messages and "FLAG" would flatten them back together.
    const flag = (event.flag ?? "").trim();
    if (flag === "DOUBLE YELLOW") return { text: "2Y", tone: "is-flag" };
    if (flag === "CLEAR") return { text: "CLR", tone: "is-clear" };
    if (flag) return { text: flag.slice(0, 4), tone: "is-flag" };
    return { text: "FLAG", tone: "is-flag" };
  }
  return null;
}

/** One row of the feed, plus how many identical ones it stands for. */
interface FeedRow {
  event: RadioEvent;
  index: number;
  repeats: number;
}

/**
 * The feed, newest first, with consecutive identical messages collapsed.
 *
 * The component's own key comment records why they exist: "two race-control rows of
 * the same lap really can carry the same text (four identical BLUE FLAG lines for
 * the same car on Melbourne's lap 46)". Four slots of a ten-row fold spent saying
 * one thing. Collapsed they spend one and say `x4`.
 *
 * Consecutive only, and deliberately: the same message an hour apart is two events,
 * and merging them would rewrite the race's chronology rather than compress it.
 */
function collapse(events: RadioEvent[]): FeedRow[] {
  const rows: FeedRow[] = [];
  events.forEach((event, index) => {
    const previous = rows[rows.length - 1];
    const same =
      previous !== undefined &&
      previous.event.kind === event.kind &&
      previous.event.text === event.text &&
      previous.event.driver === event.driver &&
      previous.event.lap === event.lap;
    if (same) previous.repeats += 1;
    else rows.push({ event, index, repeats: 1 });
  });
  return rows;
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

export function RadioFeed({
  bulk,
  driverMain,
}: {
  bulk: Bulk | null;
  driverMain: string | null;
}) {
  const feed = bulk?.radio;
  const events = feed?.events ?? [];
  // Newest first. The array arrives oldest-first because that is the order the
  // race happened in and the order the host reveals it in; reversing here
  // rather than there keeps the payload in the shape a future full-history
  // panel would want.
  const newestFirst = [...events].reverse();
  const rows = collapse(newestFirst);
  const [visible, setVisible] = useState<number | null>(null);
  const list = useRef<HTMLOListElement | null>(null);

  /**
   * How many rows are actually ON SCREEN.
   *
   * **The header count says how many events EXIST; it never said the panel was
   * showing about ten of them.** 58 events in a 404 px card is a ten-row fold, and
   * with the scrollbar hidden globally there was nothing to tell a reader the rest
   * was there - the same missing-affordance shape the pace grid's LAPS range exists
   * to fix, one panel over.
   *
   * Measured off each row's own box rather than derived from a row-height constant,
   * for the reason that grid's `measure` gives: the height is CSS, and a second copy
   * of it here is the twin this repo pays for most often.
   */
  const measure = useCallback(() => {
    const box = list.current;
    if (!box) return;
    const items = [...box.querySelectorAll<HTMLElement>("li")];
    if (items.length === 0) return setVisible(null);
    // Rects, not `offsetTop`: that is measured from the nearest POSITIONED ancestor,
    // and `.radio-list` is a plain flex child, so the first version of this counted
    // zero rows visible out of six. Viewport rects need no such assumption.
    const edge = box.getBoundingClientRect().bottom;
    setVisible(
      items.filter((item) => item.getBoundingClientRect().bottom <= edge + 1)
        .length,
    );
  }, []);

  useEffect(() => {
    measure();
    const box = list.current;
    if (!box) return;
    const observer = new ResizeObserver(measure);
    observer.observe(box);
    return () => observer.disconnect();
  }, [measure, rows.length]);

  const hidden = visible === null ? 0 : Math.max(0, rows.length - visible);

  return (
    <section className="card radio-feed">
      <header className="radio-header">
        <span className="radio-title">RADIO · RCM</span>
        {feed?.available ? (
          // `10 / 58`, not `58`. What the panel is showing, over what there is.
          <span className="radio-count">
            {visible === null ? events.length : `${visible} / ${events.length}`}
          </span>
        ) : (
          <span className="radio-subtitle">no corpus</span>
        )}
      </header>
      {feed?.available && newestFirst.length === 0 ? (
        <p className="radio-empty">Nothing said yet.</p>
      ) : null}
      <ol className="radio-list" ref={list} onScroll={measure}>
        {rows.map(({ event, index, repeats }) => (
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
              {(() => {
                const chip = categoryChip(event);
                return chip === null ? null : (
                  <span className={`radio-cat ${chip.tone}`}>{chip.text}</span>
                );
              })()}
              {isBroadcast(event, driverMain) ? (
                <span className="radio-tier">BROADCAST </span>
              ) : null}
              {bodyText(event)}
              {/* One slot, not four. The count is what stops the collapse from
                  hiding that the message really did repeat. */}
              {repeats > 1 ? (
                <span className="radio-repeats"> x{repeats}</span>
              ) : null}
            </span>
          </li>
        ))}
      </ol>
      {/* The fold, said out loud. Scrollbars are hidden globally, so without this
          the older half of the feed is unannounced - which is what it was. */}
      {hidden > 0 ? (
        <p className="radio-fold">+ {hidden} older · scroll</p>
      ) : null}
    </section>
  );
}
