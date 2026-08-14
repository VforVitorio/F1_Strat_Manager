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
  /**
   * The same digits decoded by the arcade's own rule: GREEN / YELLOW FLAG /
   * VSC / SAFETY CAR / RED FLAG. **Null means the loader has no entry for
   * that lap, which is NOT a green track** and must not render as one.
   *
   * Decoded by the producer for the same reason `driver_colors` is: the
   * priority order and the four labels are a project rule, and decoding the
   * digits here would be a second copy of it in another language.
   */
  track_status_label: string | null;
  /** The label's RGB, from the arcade's own palette. Null exactly when the label is. */
  track_status_color: [number, number, number] | null;
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

/** One row of the lap table, as the BULK channel serves it. Every unknown is null, never 0. */
export interface LapRow {
  lap: number;
  /** Line-crossing time on the tick's clock. The gap column subtracts these. */
  t: number | null;
  lap_time: number | null;
  s1: number | null;
  s2: number | null;
  s3: number | null;
  v1: number | null;
  v2: number | null;
  vfl: number | null;
  vst: number | null;
  position: number | null;
  compound: string | null;
  tyre_life: number | null;
  stint: number | null;
  track_status: string | null;
  pit_in: boolean;
  pit_out: boolean;
  /** A deleted time. Render struck; it never counts towards a best. */
  deleted: boolean;
  /**
   * A row FastF1 synthesised for a car that did not finish the lap. Render
   * the row, count it in NOTHING: their `Time` stamps sort before the whole
   * field, so a naive ranking puts the lap-1 crashers P1-P3.
   */
  generated: boolean;
  pb: boolean;
}

export interface DriverLaps {
  number: string | null;
  /** The mask actually applied, so a panel can assert rather than assume. */
  laps_revealed: number;
  stops: number;
  laps: LapRow[];
  /** lap -> crossing time, real rows only. The lap-quantised gap clock. */
  crossings: Record<number, number>;
  best: {
    lap: number | null;
    lap_time: number | null;
    s1: number | null;
    s2: number | null;
    s3: number | null;
    v1: number | null;
    v2: number | null;
    vfl: number | null;
    vst: number | null;
    compound: string | null;
  };
  theoretical: number | null;
}

export interface Bulk {
  /** Advances whenever the reveal changes IN EITHER DIRECTION. A rewind bumps it too. */
  rev: number;
  /** False when the race has no parquet on disk - the common case on a curated install. */
  available: boolean;
  race: { year: number | null; location: string | null; total_laps: number };
  drivers: Record<string, DriverLaps>;
}

/**
 * The lap a driver is ON, with only the sectors he has already crossed.
 *
 * Every field is null until the replay clock passes that sector's own
 * crossing time, so the tower's columns blank at the line and fill as the car
 * goes round - the way a timing tower does, rather than holding the previous
 * lap for the whole of the next one.
 */
export interface LiveSectors {
  lap: number;
  s1: number | null;
  s2: number | null;
  s3: number | null;
  v1: number | null;
  v2: number | null;
  vfl: number | null;
  /**
   * Whether each value belongs to the lap IN PROGRESS or was carried over
   * from the one before, which the tower dims.
   *
   * **`s3_fresh` is false essentially always, and that is the data, not a
   * bug.** A third sector's crossing IS the end of its lap - measured over
   * the whole race, `Sector3SessionTime` lands a median 55 ms AFTER the lap's
   * own line crossing - so the S3 a strategist sees mid-lap is always the
   * previous lap's. Serving only the current lap made the column permanently
   * empty (#933).
   */
  s1_fresh: boolean;
  s2_fresh: boolean;
  s3_fresh: boolean;
}

export interface LiveLap {
  /** Advances when a sector OPENS or CLOSES. A rewind closes them, and bumps it. */
  rev: number;
  /** Drivers with a lap in progress. A retired or finished car is absent, not empty. */
  drivers: Record<string, LiveSectors>;
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
/**
 * The revealed lap table, or null when this caller's revision is current.
 *
 * `sinceRev` is the `rev` of the last view rendered. **Compare on inequality,
 * not on "greater than", everywhere this value is used** - a rewind takes
 * laps BACK, and treating a lower revision as stale would keep rows the
 * clock has un-revealed. The host applies the same rule on its side; this
 * note is here because the client is where the tempting `>` lives.
 */
export async function getBulk(sinceRev: number): Promise<Bulk | null> {
  const api = window.pywebview?.api as { get_bulk?: (r: number) => Promise<Bulk | null> };
  if (api?.get_bulk) return api.get_bulk(sinceRev);
  if (IN_A_WINDOW) return null;
  return fetchJson<Bulk>("/api/bulk", sinceRev);
}

/**
 * The lap in progress, or null when this caller's revision is current.
 *
 * Same inequality rule as `getBulk`, and here it is what makes a rewind
 * CLOSE a sector again: `>` would keep a cell filled with a time the car has
 * not re-driven yet.
 */
export async function getLiveLap(sinceRev: number): Promise<LiveLap | null> {
  const api = window.pywebview?.api as { get_live_lap?: (r: number) => Promise<LiveLap | null> };
  if (api?.get_live_lap) return api.get_live_lap(sinceRev);
  if (IN_A_WINDOW) return null;
  return fetchJson<LiveLap>("/api/live", sinceRev);
}

/**
 * The socket's state as a word: Connected / Connecting... / Disconnected.
 *
 * No revision, because there is nothing to be current with: when the arcade
 * dies the ticks stop and this is the only thing left that changes. Null only
 * when the transport itself failed, which the caller renders as unknown
 * rather than inventing a state.
 */
export async function getConnection(): Promise<string | null> {
  const api = window.pywebview?.api as { get_connection?: () => Promise<string> };
  if (api?.get_connection) return api.get_connection();
  if (IN_A_WINDOW) return null;
  try {
    const response = await fetch("/api/connection", { cache: "no-store" });
    if (!response.ok) return null;
    return (await response.json()) as string;
  } catch {
    return null;
  }
}

export async function getAgentsView<T>(sinceSeq: number): Promise<T | null> {
  const api = window.pywebview?.api as { get_agents_view?: (s: number) => Promise<T | null> };
  if (api?.get_agents_view) return api.get_agents_view(sinceSeq);
  if (IN_A_WINDOW) return null;
  return fetchJson<T>("/api/agents", sinceSeq);
}
