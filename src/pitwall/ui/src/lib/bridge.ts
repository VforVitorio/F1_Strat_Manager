/**
 * The only module that knows `window.pywebview` exists.
 *
 * Everything above it consumes typed functions, so when the transport is
 * upgraded - the named path is an in-process WebSocket, if the pull model is
 * ever measured to hurt - this is the one file that changes.
 */

/**
 * The socket state, as the host reports it: one word and the colour that word
 * wears, from a single map (`agents_view/panels.CONNECTION_COLOURS`).
 */
export interface Connection {
  label: string;
  colour: string;
}

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
  /** The RAW FastF1 code. 8 means "eligible, not open" - never test it with `> 0`. */
  drs: number;
  /**
   * The wire's own answer to "is the wing open", decoded from
   * `config.DRS_OPEN_CODES` producer-side.
   *
   * This field exists so no consumer here has to know the codes: the panel that
   * charts it refused to fork that set into TypeScript, which is why the DRS lane
   * could not be built until the producer published the answer instead.
   */
  drs_open: boolean;
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
  /**
   * The digits decoded to the label for "the field was NOT racing freely on this
   * lap" - SAFETY CAR, VSC or RED FLAG - and `null` for green, for a lone yellow
   * (sector-local: the cars away from it are racing) and for an unknown status.
   *
   * Decoded by the producer, from `src/arcade/track_status.py`, for the same
   * reason `driver_colors` and the tick's own `track_status_label` are: the
   * priority order and the four labels are a project rule, and a client testing
   * for a `4` would be the second copy of it in another language.
   *
   * Non-null means a per-lap pace ranking over this lap ranks the safety car's
   * queue and not pace. Measured on Melbourne 2025: 22 of 57 laps, and 213 of
   * the 776 cells the grid ranks.
   */
  neutralised: string | null;
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

/**
 * One thing that was said during the race - a team radio or a race-control message.
 *
 * `driver` is the speaker's code on a radio and always `null` on an RCM, whose
 * message already names the car it concerns.
 *
 * **The TIER is not on this object.** Which car is ours lives on the tick, and
 * this rides in the bulk, so the renderer pairs the two: a radio from anyone
 * but `arcade.driver_main` is broadcast-tier and says so on screen.
 */
export interface RadioEvent {
  kind: "radio" | "rcm";
  lap: number;
  driver: string | null;
  /** The transcript, or "" when the audio was never transcribed - which is the common case. */
  text: string;
  category: string | null;
  flag: string | null;
}

export interface Bulk {
  /** Advances whenever the reveal changes IN EITHER DIRECTION. A rewind bumps it too. */
  rev: number;
  /** False when the race has no parquet on disk - the common case on a curated install. */
  available: boolean;
  race: { year: number | null; location: string | null; total_laps: number };
  drivers: Record<string, DriverLaps>;
  /**
   * The radio/RCM feed of the same race, oldest first, masked by the same
   * reveal. It rides in this payload rather than on a channel of its own
   * because it is a function of exactly what signs this one, and a second
   * signature that does not determine its payload is what #934 cost.
   */
  radio: { available: boolean; events: RadioEvent[] };
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
 * How long to wait for pywebview to announce its API before giving up on it.
 *
 * A browser tab never fires `pywebviewready`, so the wait needs a floor, and the
 * floor has to be short enough that a tab's first paint does not visibly stall.
 * 250 ms is one and a half tick polls.
 */
const BRIDGE_READY_TIMEOUT_MS = 250;

/**
 * Wait for `window.pywebview.api`, briefly, whichever page this is.
 *
 * pywebview injects the API asynchronously and announces it with a
 * `pywebviewready` event. Code that reads it at module scope sees `undefined` and
 * fails silently, which renders an empty window with nothing in any log - hence a
 * wait at all.
 *
 * **There used to be an `IN_A_WINDOW` constant here, defined as
 * `location.protocol === "file:"`, and #996 made it a lie.** The windows now load
 * over the host's own loopback server, so a real PITWALL window reports `http:`
 * exactly like a browser tab and the sniff answered "browser" inside the product.
 * It was benign - every getter below dispatches on the OBJECT, so a window used its
 * API as soon as one appeared - but the constant's name, its comment ("`file:` is
 * the giveaway a window ALWAYS has") and its five early-return guards all described
 * a world that no longer exists, and each guard skipped a `fetch` to a route that
 * is now always there.
 *
 * So the protocol is not consulted at all. The event is raced against a timeout: in
 * a window it resolves on the announcement, in a tab on the floor, and neither path
 * needs to know which it is.
 */
export function whenBridgeReady(): Promise<void> {
  if (window.pywebview?.api) return Promise.resolve();
  return new Promise((resolve) => {
    const done = () => resolve();
    window.addEventListener("pywebviewready", done, { once: true });
    window.setTimeout(done, BRIDGE_READY_TIMEOUT_MS);
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
async function fetchJson<T>(
  route: string,
  sinceSeq: number,
  extraQuery = "",
): Promise<T | null> {
  try {
    const response = await fetch(`${route}?since=${sinceSeq}${extraQuery}`, { cache: "no-store" });
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
  return fetchJson<LiveLap>("/api/live", sinceRev);
}

/**
 * The socket's state as a word AND its colour: Connected / Connecting... /
 * Disconnected.
 *
 * The colour rides with the word because it used to ride separately, and the
 * two windows disagreed about "Connecting..." - amber through the AGENTS view,
 * dim grey through the DATA strip's own CSS classes, for one socket.
 *
 * No revision, because there is nothing to be current with: when the arcade
 * dies the ticks stop and this is the only thing left that changes. Null only
 * when the transport itself failed, which the caller renders as unknown
 * rather than inventing a state.
 */
export async function getConnection(): Promise<Connection | null> {
  const api = window.pywebview?.api as { get_connection?: () => Promise<Connection> };
  if (api?.get_connection) return api.get_connection();
  try {
    const response = await fetch("/api/connection", { cache: "no-store" });
    if (!response.ok) return null;
    return (await response.json()) as Connection;
  } catch {
    return null;
  }
}

/**
 * The AGENTS view, or null when this caller's view is current.
 *
 * **Two things the caller holds, not one.** The sequence says which tick it
 * rendered; `sinceConnection` says which connection LABEL it rendered. The
 * second exists because when the producer dies the sequence stops advancing,
 * so a purely sequence-driven view would keep painting the last frame of a
 * dead race behind a green chip - and the host cannot answer "changed since
 * YOU looked" from one field it keeps for everybody. It tried, and with two
 * consumers the second never learned (#950).
 */
export async function getAgentsView<T>(
  sinceSeq: number,
  sinceConnection: string | null,
): Promise<T | null> {
  const api = window.pywebview?.api as {
    get_agents_view?: (s: number, c: string | null) => Promise<T | null>;
  };
  if (api?.get_agents_view) return api.get_agents_view(sinceSeq, sinceConnection);
  const held = sinceConnection === null ? "" : `&connection=${encodeURIComponent(sinceConnection)}`;
  return fetchJson<T>("/api/agents", sinceSeq, held);
}
