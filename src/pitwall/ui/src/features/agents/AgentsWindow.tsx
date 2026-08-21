/**
 * PITWALL · AGENTS — a decision band over six specialist consoles.
 *
 * **The Qt lineage ends here for the LAYOUT, and only for the layout.** Every
 * string, colour and glyph below still comes out of `src/pitwall/agents_view/`,
 * which is the Qt window's own code; what changed is where they sit. The port
 * was a header strip, a 540 / 740 horizontal split with the decision in the
 * left column and a 3x2 card grid in the right, and a status bar - `window.py`
 * geometry, faithfully. Faithful is what was wrong with it: the split made the
 * decision a PEER of the agent grid, two territories with no reading order, and
 * the most important content on the window shared its column with a reasoning
 * panel measured at 1.9 % ink.
 *
 * Four strata now, top to bottom: header, the DECISION BAND, the agent grid,
 * status bar. The band answers one question per module, left to right, in the
 * order a reader asks them - what are we doing, why, on what evidence, and what
 * happens next - along the line the eye lands on anyway.
 *
 * Before the first view arrives the window renders what the Qt window
 * shows at startup, rather than a spinner. That is not the same as
 * `update_from(None)`: the badge is ACCENT purple and the plan line uses
 * em-dashes — `_render_idle` is what Qt paints once a tick with no
 * decision has arrived, a different moment. The scenario scores are the
 * one place that no longer follows Qt: they read `--` rather than the
 * "0%" it painted, because before the first tick nothing has been
 * simulated and 0 % is a measurement.
 * The connection chip is the one field taken from Qt's FIRST PAINTED
 * FRAME rather than from its constructor, because the constructor's grey
 * lasts milliseconds and is not a state anybody sees.
 */

import { useConnection } from "../../lib/useConnection";
import { useStatusText } from "../../lib/useStatusText";
import { waitingStatus } from "../../lib/waitingCopy";

import { AgentCard } from "./AgentCard";
import { OrchestratorCard } from "./OrchestratorCard";
import { WhyPanel } from "./WhyPanel";
import { ScenarioBars } from "./ScenarioBars";
import { PlanPanel } from "./PlanPanel";
import { HeaderBar } from "./HeaderBar";
import { PaceChart } from "./PaceChart";
import { TireChart } from "./TireChart";
import { useAgentsView, type AgentCardView, type AgentsView } from "../../lib/agents";

/**
 * The six consoles, for the boot view's own card map.
 *
 * No longer a render order: the grid places them by name (`agents.css`'s
 * `grid-template-areas`) because three of the six want a shape the reading
 * order does not give them.
 */
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
  tooltip: null,
};

const IDLE_VIEW: AgentsView = {
  view_version: 0,
  seq: null,
  header: {
    session: "--",
    driver: "--",
    lap: "L 0/0",
    playback: "-- × · --",
    // `null` before a view arrives, which renders the unknown chip. There is no
    // tick yet, so the window genuinely does not know the track status, and
    // "GREEN" here would be a claim it cannot make.
    track_status: null,
    track_status_colour: null,
    // Overwritten from the polled channel below while there is no view. It is
    // a placeholder, not a claim: the host owns both the word and its colour
    // (`agents_view/panels.CONNECTION_COLOURS`), and this literal used to
    // hardcode WARNING amber - so a socket that had come up and not yet
    // delivered a lap read as still connecting, in a colour that means
    // something is wrong, for the whole startup.
    connection: "Connecting...",
    connection_colour: "#9ca3af",
  },
  orchestrator: {
    action: "--",
    // ACCENT: `orchestrator_card.py:100` styles the badge at construction.
    action_colour: "#a78bfa",
    confidence: null,
    confidence_fill: 0,
    confidence_text: "--",
    confidence_colour: "#9ca3af",
    pace: "Pace: --",
    pace_colour: "#9ca3af",
    risk: "Risk: --",
    risk_colour: "#9ca3af",
    plan: "Pit: — · Next: — · UCUT: —",
    why: "",
    why_detail: null,
    changed: "",
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
      x_range: null,
      current_lap: null,
      cursor_colour: "#9ca3af",
      prediction_lap: null,
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
      y_range: null,
      current_lap: null,
      cursor_colour: "#9ca3af",
    },
  },
  plan_timeline: {
    total_laps: 0,
    first_known_lap: null,
    segments: [],
    pit_lap: null,
    pit_pct: null,
    cliff: null,
    current_lap: null,
    current_pct: null,
    caption: "Pit: — · Next: — · UCUT: —",
  },
  history: { pace: [], tire: [] },
  // Replaced by `waitingStatus(connection)` while `view` is null: the idle
  // view cannot know whether the socket is up, and this window has no other
  // channel that does before the first tick arrives (#1004).
  status_bar: { text: "Waiting for arcade stream…", transient: false },
};

/**
 * One console in the grid.
 *
 * `chart` is a single optional node rather than two conditionals at the call
 * site. Two of them made `children` an ARRAY OF NULLS for the four cards
 * without a chart, and an array is truthy, so every text card rendered an
 * empty `.agent-chart` - a 140 px min-height box holding nothing. That phantom
 * box, not the grid, is what put 89-112 px of dead strip under four cards.
 */
function Console({
  slot,
  title,
  card,
  chart,
}: {
  slot: string;
  title: string;
  card: AgentCardView;
  chart?: React.ReactNode;
}) {
  return (
    <AgentCard title={title} card={card} slot={slot}>
      {chart ?? null}
    </AgentCard>
  );
}

export function AgentsWindow() {
  const { view } = useAgentsView();
  const shown = view ?? IDLE_VIEW;
  /**
   * Polled separately, and ONLY while there is no view (#1004).
   *
   * `get_agents_view` returns None until a tick has arrived, so this window has
   * no connection word at all for the whole startup - measured on the real path,
   * None on all 169 samples across 11 s, including the last 3 s during which the
   * socket was up and the arcade was loading its session. The DATA window has
   * polled `get_connection` since #982; this one had nothing to poll it for until
   * the wait itself became the thing worth describing.
   *
   * Once a view exists the view's own label wins: it is host-built and travels
   * WITH the payload (#950), so it cannot disagree with the lap beside it, which
   * a separately-polled word could.
   */
  const connection = useConnection();

  /**
   * The producer is gone, and every card is holding a call from before.
   *
   * **This window was the twin that never got #982.** The DATA window learned to
   * say so; this one, with a real producer killed, still read `PIT NOW ·
   * Confidence: 71% · Pace: PUSH · Risk: AGGRESSIVE` beside `2.00× · PLAYING` at
   * full strength, with a red chip and a blank status bar as the only tells - the
   * exact pair #982's own comment calls insufficient. And a stale strategy CALL is
   * worse to mistake for a live one than a stale lap time is.
   *
   * The label is host-built and travels WITH the view (#950), so there is nothing
   * to poll separately here: `view !== null` means a payload has arrived, and the
   * connection field is what still moves after the ticks stop.
   */
  // The header the window paints: the view's own label once a payload has
  // arrived - it travels WITH the lap beside it, so it cannot disagree with it
  // - and the polled pair before that, which is the only thing that knows.
  const header =
    view === null && connection
      ? { ...shown.header, connection: connection.label, connection_colour: connection.colour }
      : shown.header;

  const frozen = view !== null && shown.header.connection === "Disconnected";
  const statusText = useStatusText(
    frozen
      ? { text: `DATA FROZEN · last tick ${shown.header.lap}`, transient: false }
      : view === null
        ? { text: waitingStatus(connection?.label ?? null), transient: false }
        : shown.status_bar,
    shown.seq,
  );

  return (
    <div className="agents-window">
      <HeaderBar header={header} frozen={frozen} />

      {/* Dimmed, not desaturated: the cards' colour is content here too - the
          alarm red, the WARNING amber on a changed call, the confidence bar.

          ONE class for this state, and it is the shipped one. The elevation
          spec proposed a second, `.is-stale`, keyed off
          `connection !== "Connected"` - which is also true for
          "Connecting...", i.e. the whole ~11 s startup, and which would have
          composed its `brightness()` with this one on a dead producer. Two
          names for one state is this repo's dominant defect class. */}
      <div className={frozen ? "agents-body is-frozen" : "agents-body"}>
        {/* The decision band: what, how sure, why, and what happens next, in
            one left-to-right sweep along the top of the window. It replaces a
            540 px left column that made the decision a PEER of the agent grid
            - two territories with no reading order, the most important content
            sharing a column with the least. */}
        <div className="agents-band">
          <OrchestratorCard view={shown.orchestrator} />
          <WhyPanel view={shown.orchestrator} />
          <ScenarioBars rows={shown.scenarios} />
          <PlanPanel view={shown.plan_timeline} />
        </div>

        {/* The six consoles below it, in named grid areas rather than source
            order: the two chart cards take the tall slots, SITUATION and PIT
            are compact and both feed the decision so they stack beside them,
            and RADIO carries the wordiest content on the window - transcripts -
            so it spans two columns.

            SITUATION and PIT share ONE area through a stack rather than taking
            a grid row each. Given a row each they sat at the top of two equal
            cells and the leftover fell BETWEEN them, which reads as a card
            that failed to load; stacked, the whole remainder is one block of
            window background under the pair. */}
        <div className="agents-grid">
          <Console
            slot="pace"
            title="Pace"
            card={shown.cards.pace ?? IDLE_CARD}
            chart={<PaceChart series={shown.charts.pace} />}
          />
          <Console
            slot="tire"
            title="Tire"
            card={shown.cards.tire ?? IDLE_CARD}
            chart={<TireChart series={shown.charts.tire} />}
          />
          <div className="agents-side">
            <Console slot="situation" title="Situation" card={shown.cards.situation ?? IDLE_CARD} />
            <Console slot="pit" title="Pit" card={shown.cards.pit ?? IDLE_CARD} />
          </div>
          <Console slot="radio" title="Radio" card={shown.cards.radio ?? IDLE_CARD} />
          <Console slot="rag" title="RAG" card={shown.cards.rag ?? IDLE_CARD} />
        </div>
      </div>

      <footer className="status-bar">{statusText}</footer>
    </div>
  );
}
