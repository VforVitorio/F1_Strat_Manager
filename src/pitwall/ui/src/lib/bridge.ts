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
  /** Per-driver RGB, from the arcade's own palette so no consumer hardcodes a second copy. */
  driver_colors: Record<string, [number, number, number]>;
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
 * Is this page inside a PITWALL window, or in a browser tab?
 *
 * The same bundle serves both. In a window the host injects
 * `window.pywebview.api`; over http there is no such object and the same
 * payload arrives from `/api/tick` on the loopback server the host runs
 * alongside the windows. Everything above this module is unaware of which.
 *
 * `file:` is the giveaway a window ALWAYS has and a browser never does, but
 * it is not sufficient on its own: pywebview announces its API
 * asynchronously, so a page that has finished loading may not have it yet.
 * Hence the wait below rather than a single check here.
 */
const IN_A_WINDOW = window.location.protocol === "file:";

/**
 * pywebview injects `window.pywebview.api` asynchronously and announces it
 * with a `pywebviewready` event. Code that reads the API at module scope
 * therefore sees `undefined` and fails silently, which renders an empty
 * window with nothing in any log.
 *
 * Over http there is nothing to wait for, and waiting for an event that will
 * never fire is how the browser path would render an empty page with nothing
 * in any log - the same failure, one transport over.
 */
export function whenBridgeReady(): Promise<void> {
  if (window.pywebview?.api || !IN_A_WINDOW) return Promise.resolve();
  return new Promise((resolve) => {
    window.addEventListener("pywebviewready", () => resolve(), { once: true });
  });
}

/**
 * One tick, from whichever transport this page has.
 *
 * A network error resolves to null rather than throwing: null already means
 * "nothing new, keep what you have", which is exactly the right behaviour
 * while a server restarts, and it is what the poll loop above already
 * handles. Throwing would kill the loop on the first hiccup.
 */
async function fetchJson<T>(route: string, sinceSeq: number): Promise<T | null> {
  try {
    const response = await fetch(`${route}?since=${sinceSeq}`, { cache: "no-store" });
    if (!response.ok) return null;
    return (await response.json()) as T | null;
  } catch {
    return null;
  }
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
  if (api) return api.get_tick(sinceSeq);
  if (IN_A_WINDOW) return null;
  return fetchJson<Tick>("/api/tick", sinceSeq);
}

/**
 * The whole AGENTS view, from whichever transport this page has.
 *
 * It lives here rather than in `agents.ts` for the same reason `getTick`
 * does: this module is the ONE place that knows how a payload arrives, and
 * the browser fallback would otherwise have to be written twice - which is
 * this repo's dominant defect, one copy fixed and its twin not.
 */
export async function getAgentsView<T>(sinceSeq: number): Promise<T | null> {
  const api = window.pywebview?.api as { get_agents_view?: (s: number) => Promise<T | null> };
  if (api?.get_agents_view) return api.get_agents_view(sinceSeq);
  if (IN_A_WINDOW) return null;
  return fetchJson<T>("/api/agents", sinceSeq);
}
