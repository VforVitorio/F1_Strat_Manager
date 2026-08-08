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
 * Before the first view arrives the window renders the same idle state
 * the Qt one shows at startup, rather than a spinner: same chips, same
 * placeholders, same six hollow cards.
 */

import { AgentCard } from "./AgentCard";
import { HeaderBar } from "./HeaderBar";
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
    connection: "Disconnected",
    connection_colour: "#ef4444",
  },
  cards: Object.fromEntries(CARDS.map(([key]) => [key, IDLE_CARD])),
  history: { pace: [], tire: [] },
  status_bar: { text: "Waiting for arcade stream…", transient: false },
};

export function AgentsWindow() {
  const { view } = useAgentsView();
  const shown = view ?? IDLE_VIEW;

  return (
    <div className="agents-window">
      <HeaderBar header={shown.header} />

      <div className="agents-split">
        <div className="agents-left">{/* orchestrator, scenarios, reasoning */}</div>
        <div className="agents-right">
          {CARDS.map(([key, title]) => (
            <AgentCard key={key} title={title} card={shown.cards[key] ?? IDLE_CARD} />
          ))}
        </div>
      </div>

      <footer className="status-bar">{shown.status_bar.text}</footer>
    </div>
  );
}
