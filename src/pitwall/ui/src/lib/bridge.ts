/**
 * The only module that knows `window.pywebview` exists.
 *
 * Everything above it consumes typed functions, so when the transport is
 * upgraded - the named path is an in-process WebSocket, if the pull model is
 * ever measured to hurt - this is the one file that changes.
 */

export interface PlaybackState {
  speed: number;
  paused: boolean;
  frame_index: number;
  total_frames: number;
}

export interface DriverState {
  lap: number;
  dist: number;
  rel_dist: number | null;
  speed: number;
  compound: number;
  tyre_life: number;
  active: boolean;
  has_position: boolean;
  /** Laps completed per the crossing map. The reveal carrier: reveal lap L iff L <= laps_completed. */
  laps_completed: number;
  /** Laps plus fraction of the current lap, the ordering coordinate. Null when the telemetry never places the car (#886). */
  progress: number | null;
  /** Took the chequered flag. OUT = !active && !has_finished (#855); the value's quality is #879. */
  has_finished: boolean;
}

export interface TelemetrySample {
  lap: number;
  t: number;
  dist: number | null;
  speed: number;
  throttle: number;
  brake: number;
  gear: number;
  drs: number;
}

export interface ArcadeState {
  gp_name: string;
  location: string;
  year: number;
  lap: number;
  t: number;
  global_t_min: number;
  total_laps: number;
  circuit_length_m: number;
  driver_main: string;
  driver_rival: string | null;
  drivers: Record<string, DriverState>;
  /** Every published driver, best first - the producer's own ranking, so no consumer re-derives it (#857). */
  race_order: string[];
  /** FastF1 TrackStatus digits for the lap on screen; "" when the loader has no entry, rendered as clear. */
  track_status: string;
  telemetry: {
    /** Every sample the replay clock crossed since the previous tick, oldest first. */
    main: TelemetrySample[];
    rival: TelemetrySample[];
    /** The user seeked backwards: drop what you have, do not append. */
    rewound: boolean;
    /** Frames a forward jump could not carry. Non-zero means a hole in the trace. */
    dropped: number;
  };
}

export interface Tick {
  schema_version: number;
  /** Strictly increasing per message the producer SENT. Not per message this window received. */
  seq: number;
  arcade: ArcadeState;
  playback: PlaybackState;
  strategy: unknown;
}

interface PitwallApi {
  get_tick: (sinceSeq: number) => Promise<Tick | null>;
}

declare global {
  interface Window {
    pywebview?: { api?: PitwallApi };
  }
}

/**
 * pywebview injects `window.pywebview.api` asynchronously and announces it
 * with a `pywebviewready` event. Code that reads the API at module scope
 * therefore sees `undefined` and fails silently, which renders an empty
 * window with nothing in any log.
 */
export function whenBridgeReady(): Promise<void> {
  if (window.pywebview?.api) return Promise.resolve();
  return new Promise((resolve) => {
    window.addEventListener("pywebviewready", () => resolve(), { once: true });
  });
}

/**
 * Ask for the latest tick, or null when this window is already up to date.
 *
 * `sinceSeq` is what makes two windows on independent timers read the same
 * frames: against a blind latest-payload slot they were measured disagreeing
 * on 58 % of polls, with duplicate reads and skips in equal measure.
 */
export async function getTick(sinceSeq: number): Promise<Tick | null> {
  const api = window.pywebview?.api;
  if (!api) return null;
  return api.get_tick(sinceSeq);
}
