/**
 * Band 1: the slim strip that says what moment the rest of the window shows.
 *
 * Lap, track status, session clock, playback, connection. Everything but the
 * last rides on the tick already; the connection is a property of the socket
 * rather than of the stream, so it is polled separately (`useConnection`).
 *
 * **The PROVISIONAL chip is the plan's rule, not decoration.** `race_order`
 * is meaningless until every car has completed a lap: on frame 0 the field is
 * ordered by millimetres of accumulated distance (HUL "leads" by 6 mm,
 * measured) and through lap 1 each car's fraction is normalised by its OWN
 * first-lap length, which biases the back of the grid so hard that a car
 * starting P7 reads P2. `gaps.py` says as much - "excluding the opening lap,
 * where no classification exists yet" - and the wire publishes the order
 * anyway. A broadcast timing tower marks exactly this state; so does this.
 */

import type { Connection, Tick } from "../../lib/bridge";
import { driverStatus } from "../../lib/driverStatus";
import { trackStatusTreatment } from "../../lib/trackStatus";

interface StatusStripProps {
  tick: Tick | null;
  connection: Connection | null;
  /** The producer is gone and every value on this strip is from before. */
  frozen?: boolean;
}

export function StatusStrip({ tick, connection, frozen = false }: StatusStripProps) {
  const arcade = tick?.arcade;
  const playback = tick?.playback;

  return (
    <header className="status-strip card">
      <span className="strip-lap">
        {arcade?.lap ? `L ${arcade.lap}` : "L —"}
        <span className="strip-lap-total">/{arcade?.total_laps || "—"}</span>
      </span>

      <TrackStatusChip
        label={arcade?.track_status_label ?? null}
        colour={arcade?.track_status_color ?? null}
        frozen={frozen}
      />

      {/* Where the chips already live, so the reader who only checks the top-left
       * corner gets it too. The status bar says the same thing at the bottom. */}
      {frozen ? (
        <span className="strip-chip is-frozen" title="The arcade broadcast stopped">
          DATA FROZEN
        </span>
      ) : null}

      {arcade && isProvisional(arcade.drivers) ? (
        <span className="strip-chip is-provisional" title="No classification exists until every car has completed a lap">
          PROVISIONAL
        </span>
      ) : null}

      <span className="strip-spacer" />

      <StripField label="SESSION" value={sessionClock(arcade)} />
      {/* The last tick's speed is not the replay's speed once the ticks stop, and
       * `2x` is an assertion that it is still advancing. */}
      <StripField label="PLAYBACK" value={frozen ? "—" : playbackLabel(playback)} />
      <span
        className="strip-chip"
        // **The colour arrives with the word.** These were three CSS classes
        // mapping the same three states the AGENTS window mapped in Python,
        // and the two disagreed: "Connecting..." was dim here and WARNING
        // amber there, for one socket, on two windows open side by side. The
        // argument this window had written down - an absence must not borrow
        // the colour of a state - won, and moved into the shared map.
        style={connection ? { color: connection.colour } : undefined}
      >
        {connection?.label ?? "—"}
      </span>
    </header>
  );
}

function StripField({ label, value }: { label: string; value: string }) {
  return (
    <span className="strip-field">
      <span className="strip-field-label">{label}</span>
      <span className="strip-field-value">{value}</span>
    </span>
  );
}

/**
 * The decoded status, or an explicit unknown, in this strip's chip idiom.
 *
 * **The three-state rule moved to `lib/trackStatus.ts` and is shared with the
 * AGENTS header**, which renders the same neutralisation in its own idiom. Both
 * windows are open on one desk; a second copy of this rule is a second chance
 * for them to disagree, which is exactly what the connection chip did.
 *
 * What stays here is the markup, because the two chips are not the same object:
 * this one is a 1465 x 28 strip cell and the other is a header chip.
 *
 * A non-green status is FILLED, and that is the whole of the window's reaction
 * to a safety car. Before this it was an outline chip swapping its text, `GREEN`
 * at 54.3 x 18 px against `SAFETY CAR` at about 86 x 18, and nothing else on the
 * window changed at all: two captures, one green and one under the safety car,
 * differed in no element but that one.
 *
 * The fill is a DIFFERENT amber from the one the pace grid's rail and the race
 * trace's band use for the same state. The wire sends `SAFETY CAR` as
 * rgb(255,140,0) while those two use `palette.WARNING` #f59e0b. Both names live
 * in `palette.py` so neither is a stray literal, and at a glance the two are
 * indistinguishable; recorded so the next reader finds a decision rather than a
 * bug. The wire's colour is the right one to fill with, because it is the one
 * the arcade paints its own banner in on the screen beside this window.
 */
function TrackStatusChip({
  label,
  colour,
  frozen,
}: {
  label: string | null;
  colour: [number, number, number] | null;
  frozen: boolean;
}) {
  const worn = trackStatusTreatment(label, colour, frozen);
  if (worn.kind === "unknown") return <span className="strip-chip is-unknown">{worn.text}</span>;
  return (
    <span
      className={worn.kind === "filled" ? "strip-chip is-filled" : "strip-chip"}
      style={
        worn.kind === "filled"
          ? { background: worn.rgb, borderColor: worn.rgb }
          : { color: worn.rgb, borderColor: worn.rgb }
      }
    >
      {worn.text}
    </span>
  );
}

/**
 * True while a car that is still IN the race has yet to complete a lap.
 *
 * **The retired cars have to come out, and the real race is what says so.**
 * The delivery plan words the rule as "until `laps_completed >= 1` for every
 * driver", and read literally that never becomes false on Melbourne 2025:
 * SAI, DOO and HAD crashed on lap 1 and their `laps_completed` is 0 for the
 * whole race, so the chip measured against all 20 drivers was still lit at
 * lap 23 - checked against the live wire, 3 of 20 under one lap and all
 * three OUT. A permanent PROVISIONAL says nothing, which is worse than not
 * having it: the state it exists to mark is the opening lap.
 *
 * What the rule is really asking is whether a classification exists yet, and
 * a car that stopped will never contribute one.
 */
function isProvisional(drivers: Tick["arcade"]["drivers"]): boolean {
  const contenders = Object.values(drivers).filter((car) => driverStatus(car) !== "out");
  if (!contenders.length) return false;
  return contenders.some((car) => car.laps_completed < 1);
}

/**
 * FastF1 SessionTime as `H:MM:SS`.
 *
 * `t` alone is `frame_index * DT`, which is replay time and means nothing
 * outside this process; `t + global_t_min` is the clock laps.parquet,
 * intervals.parquet and the weather table are all keyed on, which is the one
 * a strategist can compare against anything else.
 */
function sessionClock(arcade: Tick["arcade"] | undefined): string {
  if (!arcade) return "—";
  const total = Math.max(0, Math.floor(arcade.t + arcade.global_t_min));
  const hours = Math.floor(total / 3600);
  const minutes = Math.floor((total % 3600) / 60);
  const seconds = total % 60;
  return `${hours}:${pad(minutes)}:${pad(seconds)}`;
}

function playbackLabel(playback: Tick["playback"] | undefined): string {
  if (!playback) return "—";
  return playback.paused ? "PAUSED" : `${playback.speed}x`;
}

function pad(value: number): string {
  return value.toString().padStart(2, "0");
}
