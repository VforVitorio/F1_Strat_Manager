/**
 * The AGENTS view, typed, and the hook that polls for it.
 *
 * **There is deliberately no formatting on this side.** Every string,
 * colour and glyph below is computed in Python by the same code that
 * paints the Qt window (`src/pitwall/agents_view/`), so the port is 1:1
 * by construction rather than by inspection. If you find yourself about
 * to write a `toFixed(2)` here, the number belongs in `panels.py`.
 */

import { useEffect, useRef, useState } from "react";
import { getAgentsView, whenBridgeReady } from "./bridge";

/** Matches the producer's own ~10 Hz. Polling faster only returns null more often. */
const POLL_INTERVAL_MS = 100;

export interface CardLine {
  /** Qt rich text: `<b>`, `<br>`, `&nbsp;` and the two badge spans. */
  text: string;
  colour: string;
}

export interface AgentCardView {
  headline: string;
  headline_colour: string;
  lines: CardLine[];
  status: "OK" | "WATCH" | "ALERT" | "IDLE";
  glyph: string;
  glyph_colour: string;
  /** Empty string means no tooltip, which is Qt's own convention. */
  tooltip: string;
}

export interface HeaderView {
  session: string;
  driver: string;
  lap: string;
  playback: string;
  connection: string;
  connection_colour: string;
}

export interface PaceHistoryRow {
  lap: number;
  actual?: number | null;
  pred?: number | null;
  ci_p10?: number | null;
  ci_p90?: number | null;
}

export interface TireHistoryRow {
  lap: number;
  lap_time_s?: number | null;
  tyre_life?: number | null;
  compound?: string | null;
}

export interface OrchestratorView {
  action: string;
  action_colour: string;
  /** null before the first decision; the bar then draws empty. */
  confidence: number | null;
  /** The bar's width in per cent, already clamped and rounded host side. */
  confidence_fill: number;
  confidence_label: string;
  confidence_colour: string;
  pace: string;
  pace_colour: string;
  risk: string;
  risk_colour: string;
  /** Qt rich text: may carry the compound pill. */
  plan: string;
  /** Empty string when the orchestrator did not override the MC winner. */
  guardrail: string;
}

export interface ScenarioRow {
  key: string;
  label: string;
  /** Min-max normalised across the scenarios present, already clamped. */
  fill: number;
  /** The same value as a percentage, rounded host side; this is the width. */
  fill_pct: number;
  score: string;
  is_winner: boolean;
  bar_colour: string;
  label_colour: string;
  score_colour: string;
}

export interface ReasoningSegment {
  text: string;
  colour: string;
  bold: boolean;
}

export interface ReasoningTab {
  key: string;
  label: string;
  /** Already split into coloured runs by the Qt highlighter's own rules. */
  segments: ReasoningSegment[];
}

export interface PaceSeries {
  /** `[lap, seconds]`, already filtered to plausible lap times. */
  actual: [number, number][];
  pred: [number, number][];
  /** `[lap, p10, p90]`. */
  band: [number, number, number][];
  actual_colour: string;
  pred_colour: string;
  band_colour: string;
}

export interface TireStint {
  compound: string;
  colour: string;
  points: [number, number][];
}

export interface TireSeries {
  /** One entry per compound run: the break IS the compound change. */
  stints: TireStint[];
  trend: [number, number][];
  trend_colour: string;
  /** Absolute lap numbers, or null when the projection is out of range. */
  cliff: { lo: number | null; hi: number | null; p50: number | null } | null;
  cliff_colour: string;
  /** Lap numbers where the compound changed: one faint dashed vertical each. */
  boundaries: number[];
  boundary_colour: string;
  boundary_opacity: number;
  x_range: [number, number];
}

export interface AgentsView {
  view_version: number;
  seq: number | null;
  header: HeaderView;
  orchestrator: OrchestratorView;
  scenarios: ScenarioRow[];
  reasoning: ReasoningTab[];
  cards: Record<string, AgentCardView>;
  charts: { pace: PaceSeries; tire: TireSeries };
  history: { pace: PaceHistoryRow[]; tire: TireHistoryRow[] };
  status_bar: { text: string; transient: boolean };
}

export interface AgentsState {
  view: AgentsView | null;
  /** False until the first view lands, so the window can say "waiting" honestly. */
  live: boolean;
}

/**
 * One poll loop, sequenced, exactly like `useTick`.
 *
 * The host returns null when this window is already up to date, and a
 * view anyway when the connection state changed with no new tick - which
 * is the only way the window learns the arcade died, because a dead
 * producer stops advancing `seq`.
 */
export function useAgentsView(): AgentsState {
  const [state, setState] = useState<AgentsState>({ view: null, live: false });
  // A ref, not state: the loop must read the newest sequence without
  // re-subscribing, and a stale closure would ask for the same view forever.
  const lastSeq = useRef(-1);
  // The connection label this caller last RENDERED, for the same reason and by
  // the same mechanism as the sequence beside it. The host cannot hold it: with
  // two consumers of this view - and the loopback server is always one of them -
  // whichever polled first would consume the transition and the other would
  // keep a green chip on a dead race (#950).
  const lastConnection = useRef<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    let timer: number | undefined;

    const poll = async () => {
      const view = await getAgentsView<AgentsView>(lastSeq.current, lastConnection.current);
      if (cancelled) return;
      if (view) {
        if (view.seq !== null) lastSeq.current = view.seq;
        // The header's label is the RAW one the host passed in, not a
        // rendering of it (`build_header` stores `connection` verbatim and
        // derives only the colour), so it is the right thing to hand back.
        lastConnection.current = view.header?.connection ?? null;
        setState({ view, live: true });
      }
      timer = window.setTimeout(poll, POLL_INTERVAL_MS);
    };

    whenBridgeReady().then(() => {
      if (!cancelled) poll();
    });

    return () => {
      cancelled = true;
      if (timer !== undefined) window.clearTimeout(timer);
    };
  }, []);

  return state;
}
