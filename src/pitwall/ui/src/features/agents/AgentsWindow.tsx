/**
 * PITWALL · AGENTS — the 1:1 port of the Qt strategy window.
 *
 * The layout is frozen and matches `window.py:141-207`: a header strip, a
 * horizontal split at 540 / 740, the left column carrying the
 * orchestrator card, the scenario bars and the reasoning tabs, the right
 * column a 3x2 grid of agent cards, and a status bar at the bottom.
 *
 * **One Qt bug fixes itself here.** The left column gave the reasoning
 * tabs about 268 px, so the decision-memory counterweight sentence fell
 * below the fold on the one lap it matters. CSS has no fixed heights.
 *
 * Before the first view arrives the window renders what the Qt window
 * shows at startup, rather than a spinner. That is not the same as
 * `update_from(None)`: the badge is ACCENT purple, the plan line uses
 * em-dashes and the scenario scores read "0%" — `_render_idle` is what Qt
 * paints once a tick with no decision has arrived, a different moment.
 * The connection chip is the one field taken from Qt's FIRST PAINTED
 * FRAME rather than from its constructor, because the constructor's grey
 * lasts milliseconds and is not a state anybody sees.
 */

import { useStatusText } from "../../lib/useStatusText";

import { AgentCard } from "./AgentCard";
import { OrchestratorCard } from "./OrchestratorCard";
import { ReasoningTabs } from "./ReasoningTabs";
import { ScenarioBars } from "./ScenarioBars";
import { HeaderBar } from "./HeaderBar";
import { PaceChart } from "./PaceChart";
import { TireChart } from "./TireChart";
import { useAgentsView, type AgentCardView, type AgentsView } from "../../lib/agents";

/** Grid order, reading across then down, exactly as `window.py` adds them. */
const CARDS: ReadonlyArray<readonly [key: string, title: string]> = [
  ["pace", "Pace"],
  ["tire", "Tire"],
  ["situation", "Situation"],
  ["pit", "Pit"],
  ["radio", "Radio"],
  ["rag", "RAG"],
];

const IDLE_CARD: AgentCardView = {
  headline: "--",
  headline_colour: "#ffffff",
  lines: [],
  status: "IDLE",
  glyph: "○",
  glyph_colour: "#9ca3af",
  tooltip: "",
};

const IDLE_VIEW: AgentsView = {
  view_version: 0,
  seq: null,
  header: {
    session: "--",
    driver: "--",
    lap: "L 0/0",
    playback: "-- × · --",
    // Qt builds the chip grey in the constructor and its client thread
    // flips it to amber within milliseconds; the host's own
    // `_connection_label` answers "Connecting..." until something has been
    // rendered. Red "Disconnected" matched none of the three. #871 ported
    // the badge, the plan line and the scores and left this one behind.
    connection: "Connecting...",
    connection_colour: "#f59e0b",
  },
  orchestrator: {
    action: "--",
    // ACCENT: `orchestrator_card.py:100` styles the badge at construction.
    action_colour: "#a78bfa",
    confidence: null,
    confidence_fill: 0,
    confidence_label: "Confidence: --",
    confidence_colour: "#9ca3af",
    pace: "Pace: --",
    pace_colour: "#9ca3af",
    risk: "Risk: --",
    risk_colour: "#9ca3af",
    plan: "Pit: — · Next: — · UCUT: —",
    guardrail: "",
  },
  scenarios: [
    ["STAY_OUT", "STAY"],
    ["PIT_NOW", "PIT"],
    ["UNDERCUT", "UCUT"],
    ["OVERCUT", "OCUT"],
    // `--`, not the `0%` this used to claim. Before the first tick nothing
    // has been simulated, and "0 %" is a measurement — in a unit the live
    // view never uses, since a live absent scenario prints `--`. The same
    // window rendered "no data" three different ways.
  ].map(([key, label]) => ({
    key,
    label,
    fill: 0,
    fill_pct: 0,
    score: "  --",
    is_winner: false,
    is_enacted: false,
    is_scored: false,
    note: "",
    bar_colour: "#d1d5db",
    label_colour: "#d1d5db",
    score_colour: "#d1d5db",
  })),
  reasoning: [
    ["orchestrator", "Orchestrator"],
    ["pace", "Pace"],
    ["tire", "Tire"],
    ["situation", "Situation"],
    ["radio", "Radio"],
    ["pit", "Pit"],
  ].map(([key, label]) => ({ key, label, segments: [] })),
  cards: Object.fromEntries(CARDS.map(([key]) => [key, IDLE_CARD])),
  charts: {
    pace: {
      actual: [],
      pred: [],
      band: [],
      actual_colour: "#3b82f6",
      pred_colour: "#a78bfa",
      band_colour: "#a78bfa",
    },
    tire: {
      stints: [],
      trend: [],
      trend_colour: "#ffffff",
      cliff: null,
      cliff_colour: "#f59e0b",
      boundaries: [],
      boundary_colour: "#9ca3af",
      boundary_opacity: 0.31,
      x_range: [0, 1],
    },
  },
  history: { pace: [], tire: [] },
  status_bar: { text: "Waiting for arcade stream…", transient: false },
};

export function AgentsWindow() {
  const { view } = useAgentsView();
  const shown = view ?? IDLE_VIEW;
  const statusText = useStatusText(shown.status_bar, shown.seq);

  return (
    <div className="agents-window">
      <HeaderBar header={shown.header} />

      <div className="agents-split">
        <div className="agents-left">
          <OrchestratorCard view={shown.orchestrator} />
          <ScenarioBars rows={shown.scenarios} />
          <ReasoningTabs tabs={shown.reasoning} />
        </div>
        <div className="agents-right">
          {CARDS.map(([key, title]) => (
            <AgentCard key={key} title={title} card={shown.cards[key] ?? IDLE_CARD}>
              {key === "pace" ? <PaceChart series={shown.charts.pace} /> : null}
              {key === "tire" ? <TireChart series={shown.charts.tire} /> : null}
            </AgentCard>
          ))}
        </div>
      </div>

      <footer className="status-bar">{statusText}</footer>
    </div>
  );
}
