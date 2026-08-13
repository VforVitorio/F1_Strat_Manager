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

import type { Tick } from "../../lib/bridge";
import { driverStatus } from "../../lib/driverStatus";

interface StatusStripProps {
  tick: Tick | null;
  connection: string | null;
}

export function StatusStrip({ tick, connection }: StatusStripProps) {
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
      />

      {arcade && isProvisional(arcade.drivers) ? (
        <span className="strip-chip is-provisional" title="No classification exists until every car has completed a lap">
          PROVISIONAL
        </span>
      ) : null}

      <span className="strip-spacer" />

      <StripField label="SESSION" value={sessionClock(arcade)} />
      <StripField label="PLAYBACK" value={playbackLabel(playback)} />
      <span className={`strip-chip ${connectionClass(connection)}`}>{connection ?? "—"}</span>
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
 * The decoded status, or an explicit unknown.
 *
 * A null label means the loader has no TrackStatus entry for this lap, which
 * is NOT a green track. Rendering it as GREEN would be a confident answer to
 * a question nobody asked the data.
 */
function TrackStatusChip({
  label,
  colour,
}: {
  label: string | null;
  colour: [number, number, number] | null;
}) {
  if (!label || !colour) return <span className="strip-chip is-unknown">NO STATUS</span>;
  const rgb = `rgb(${colour[0]}, ${colour[1]}, ${colour[2]})`;
  return (
    <span className="strip-chip" style={{ color: rgb, borderColor: rgb }}>
      {label}
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

function connectionClass(connection: string | null): string {
  if (connection === "Connected") return "is-connected";
  if (connection === "Disconnected") return "is-lost";
  return "is-unknown";
}

function pad(value: number): string {
  return value.toString().padStart(2, "0");
}
