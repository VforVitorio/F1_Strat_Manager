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
  /** `null` means no tooltip. Structured, never markup. See `agent_formatters.py`. */
  tooltip: TooltipView | null;
}

/** A titled group of rows in a tooltip. `lead` is an optional label before the text. */
export interface TooltipSection {
  title: string;
  rows: { lead: string; text: string }[];
}

/**
 * What a card's tooltip SAYS. How it looks is decided here, in the TSX.
 *
 * Python used to return Qt's restricted rich-text dialect and this side
 * rendered it with `dangerouslySetInnerHTML`. Qt is gone; the content still
 * comes from one place, so only presentation can drift.
 */
export interface TooltipView {
  sections: TooltipSection[];
  footer: string | null;
}

export interface HeaderView {
  session: string;
  driver: string;
  lap: string;
  playback: string;
  connection: string;
  connection_colour: string;
  /** The tick's decoded TrackStatus, `null` when the loader has no entry for
   *  the lap. `null` is NOT a green track and must not render as one. */
  track_status: string | null;
  /** The wire's own colour for it, so neither window decodes the digits. */
  track_status_colour: [number, number, number] | null;
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
  /** Carried as TEXT on the panel, not as a fill: every value clears AA there. */
  action_colour: string;
  /** null before the first decision; the bar then draws empty. */
  confidence: number | null;
  /** The bar's width in per cent, already clamped and rounded host side. */
  confidence_fill: number;
  /** The numeral alone (`71%` / `--`); the caption beside it is chrome. */
  confidence_text: string;
  confidence_colour: string;
  pace: string;
  pace_colour: string;
  risk: string;
  risk_colour: string;
  /** Qt rich text: may carry the compound pill. */
  plan: string;
  /** The narrative's first sentence, on the glass. Empty before the first call. */
  why: string;
  /** The whole narrative plus the memory block, one hover or keypress away. */
  why_detail: TooltipView | null;
  /** `was STAY OUT (0.58) · L22`, only on the lap the call moved. */
  changed: string;
}

export interface ScenarioRow {
  key: string;
  label: string;
  /** Min-max normalised across the scenarios present, already clamped. */
  fill: number;
  /** The same value as a percentage, rounded host side; this is the width. */
  fill_pct: number;
  score: string;
  /** The top Monte Carlo score. NOT necessarily the plan that was enacted. */
  is_winner: boolean;
  /** The plan the orchestrator actually published, which a guardrail can move. */
  is_enacted: boolean;
  /** Whether this scenario was scored at all. An unscored row draws no track. */
  is_scored: boolean;
  /** `NOT TAKEN` on a winner the enacted action overruled, empty otherwise. */
  note: string;
  bar_colour: string;
  label_colour: string;
  score_colour: string;
}

/**
 * The six reasoning tabs. **Nothing renders this any more** (#1020): the
 * orchestrator's narrative is the band's WHY module and its tooltip, and the
 * five agent bodies are their own consoles' model detail.
 *
 * It stays on the wire because it is what the cross-surface guard in
 * `test_pitwall_agents_view.py` compares the tooltips AGAINST, and a guard
 * that measured the tooltips against the same builder that fills them would
 * pass on a move that dropped half the content. That is a different case from
 * the guardrail field #974 deleted, which no producer could populate at all.
 */
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
  /** The tyre chart's lap axis, borrowed so the two agree about where a lap is. */
  x_range: [number, number] | null;
  /** Where the car is NOW, marked on both charts. */
  current_lap: number | null;
  cursor_colour: string;
  /**
   * The newest lap carrying a PREDICTION, which is not the newest lap plotted.
   * A tick with no `per_agent` still delivers the actual lap time, so the solid
   * line advances while the dashed one stops - and nothing said so.
   */
  prediction_lap: number | null;
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
  /** The lap-time axis, bounded to the laps plotted. Null leaves it autoranged. */
  y_range: [number, number] | null;
  current_lap: number | null;
  cursor_colour: string;
}

/** One bar on the stint lane. `compound` is null when nobody reported one. */
export interface PlanSegment {
  lo: number;
  hi: number;
  compound: string | null;
  colour: string;
  /** Hollow rather than filled: the stop has not happened yet. */
  planned: boolean;
  left_pct: number;
  width_pct: number;
}

export interface PlanTimelineView {
  /** 0 before the arcade has said how long the race is; the track draws empty. */
  total_laps: number;
  /** Laps before this one are blank track: this window never saw them. */
  first_known_lap: number | null;
  segments: PlanSegment[];
  pit_lap: number | null;
  pit_pct: number | null;
  /** The tyre chart's own band, so one fact is drawn at two zoom levels. */
  cliff: {
    lo: number;
    hi: number;
    colour: string;
    left_pct: number;
    width_pct: number;
  } | null;
  current_lap: number | null;
  current_pct: number | null;
  /** The orchestrator's plan line, verbatim; may carry the compound pill. */
  caption: string;
}

/**
 * One branch plan: what the orchestrator does INSTEAD, and when.
 *
 * Every string is LLM free text and every one of them is rendered as a React
 * text node. Nothing here is markup and nothing here is escaped, which is the
 * same contract the tooltips already carry.
 */
export interface ContingencyRow {
  /** The condition that activates the branch, in the orchestrator's words. */
  trigger: string;
  /**
   * The replacement action through `classify_action`'s own label table, so it
   * reads `PIT NOW` exactly as the badge one card up does.
   *
   * It carries NO colour, deliberately: this branch is not the call, and a
   * hypothetical wearing the live decision's identity colours reads as an
   * announcement.
   */
  switch_to: string;
  /** `HIGH` / `MEDIUM` / `LOW`. The word carries it; there is no alarm hue. */
  priority: string;
  /** One line of why, clamped to two lines on the glass. */
  rationale: string;
  /**
   * The whole of the row, one hover or one keypress away. `null` when the row
   * has nothing to expand, which is also what keeps it out of the tab order
   * rather than offering an empty popup.
   */
  detail: TooltipView | null;
}

export interface ContingenciesView {
  rows: ContingencyRow[];
  /**
   * The orchestrator's risk bullets, rendered in the BODY beside the branches.
   *
   * Empty when it flagged none, and the card then shows no risks block at all
   * rather than an empty heading. They began behind a hover on the title, which
   * made them content nobody would find.
   */
  risks: string[];
  /**
   * What to print instead of rows, and `null` when there ARE rows so a renderer
   * cannot show both. The two sentences differ on purpose: no call has been made
   * yet, versus a call that planned no branches, which is the ordinary state on
   * the no-LLM profile.
   */
  empty: string | null;
}

/** One car the rejoin lands next to. `side` is the word, so no consumer holds a sign. */
export interface PitExitNeighbour {
  driver: string;
  /** Already absolute and formatted, e.g. `4.5s`. */
  gap: string;
  side: "ahead" | "behind";
}

/**
 * Where a stop taken on THIS lap would put us.
 *
 * Two states and no third: `ready` carries the move and the two neighbours,
 * `idle` carries one sentence saying WHY there is nothing to show. There is
 * deliberately no shape in which a slot is present but meaningless, because a
 * number-shaped placeholder reads as data on a glass a strategist scans.
 */
export type PitExitView =
  | {
      state: "ready";
      /** `P1 -> P3`, or `P3` alone when the current position is unknown. */
      headline: string;
      /** `±2` only when the draws disagree, else the empty string. */
      band: string;
      rows: PitExitNeighbour[];
      /** The header's qualifier, which is what makes the rest a hypothesis. */
      qualifier: string;
    }
  | { state: "idle"; note: string };

export interface AgentsView {
  view_version: number;
  seq: number | null;
  header: HeaderView;
  orchestrator: OrchestratorView;
  scenarios: ScenarioRow[];
  reasoning: ReasoningTab[];
  cards: Record<string, AgentCardView>;
  charts: { pace: PaceSeries; tire: TireSeries };
  plan_timeline: PlanTimelineView;
  contingencies: ContingenciesView;
  /**
   * Optional so a NEW bundle tolerates an OLD host. The host is a separate
   * process on the same desk and the two are not upgraded atomically; an
   * absent field has to reach the same idle branch as a host that had nothing
   * to say, which is what makes the wire's no-version-bump answer safe.
   */
  pit_exit?: PitExitView;
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
