/**
 * Does the built AGENTS bundle still render the window?
 *
 * **The gap this closes was named by the sprint-3 exit gate**: every one
 * of its render-layer findings sailed through 35 green Python tests,
 * because those pin the view DICT and nothing loaded the bundle. Deleting
 * `<PaceChart/>` from `AgentsWindow.tsx`, or the whole layout rule from
 * the stylesheet, left the suite entirely green.
 *
 * So this asserts EFFECTS in a real engine: the elements exist, the
 * layout the port claims is frozen actually computes to Qt's numbers, and
 * the status bar's 1.5 s auto-clear — typed, documented and read by
 * nothing until #871 — actually fires.
 *
 * It is NOT a visual regression test. Pixels are the screenshot tool's
 * job (`shot-agents.mjs`) and a human's. This is the structural floor.
 *
 *   npm run build && node scripts/smoke-agents.mjs
 */
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { chromium } from "@playwright/test";
import { serveDist } from "./serve-dist.mjs";
import { watchPage } from "./page-guard.mjs";
import { staysStill } from "./settle.mjs";

const UI_DIR = resolve(dirname(fileURLToPath(import.meta.url)), "..");
// An alternate bundle directory, so a negative check can run against a
// deliberately broken copy without touching the real one.
const DIST = resolve(process.argv[2] ?? resolve(UI_DIR, "dist"));

// The client area the product really hands this page, NOT the `WindowSpec`
// size. `place()` opens AGENTS at 1500x870 on the reference desktop and the
// OS keeps 14 px of frame and 37 px of title bar, so the page gets 1486x833.
// Measuring the outer size instead hid 67 px of vertical overflow from every
// assertion in this file; `tests/surfaces/test_pitwall_host.py` now refuses a
// harness viewport larger than the real client.
const CLIENT = { width: 1486, height: 833 };

// The host's own map, so the states this file drives wear the colours the
// product gives them rather than a second set invented here. Kept in step by
// `test_pitwall_tokens.py`, which reads both.
const CONNECTION_COLOURS = {
  Connected: "#10b981",
  "Connecting...": "#9ca3af",
  Disconnected: "#ef4444",
};

/**
 * Enough of a view to render every panel.
 *
 * A literal, not a file: the SHAPE of the view is pinned on the Python
 * side by `tests/surfaces/test_pitwall_agents_view.py`, and duplicating
 * that contract here would be a second source of truth for it. What this
 * file owns is whether the renderer draws what it is handed.
 */
const VIEW = {
  view_version: 1,
  seq: 1,
  header: {
    session: "Melbourne · 2025",
    driver: "NOR",
    lap: "L 23/57",
    playback: "2.00× · PLAYING",
    connection: "Connected",
    connection_colour: "#10b981",
    // Green, so the default page exercises the OUTLINE branch; the filled and
    // unknown branches are driven explicitly further down. A fixture that
    // omitted these would only ever render the unknown one.
    track_status: "GREEN",
    track_status_colour: [16, 185, 129],
  },
  orchestrator: {
    action: "PIT NOW",
    action_colour: "#ef4444",
    confidence: 0.71,
    confidence_fill: 71.0,
    confidence_text: "71%",
    confidence_colour: "#10b981",
    pace: "Pace: PUSH",
    pace_colour: "#ef4444",
    risk: "Risk: AGGRESSIVE",
    risk_colour: "#ef4444",
    plan: "Pit: L24 · Next: HARD · UCUT: RUS",
    why: "the undercut window against RUS opens now.",
    why_detail: {
      sections: [
        {
          title: "Reasoning",
          rows: [
            {
              lead: "",
              text: "the undercut window against RUS opens now. The gap is 1.4 s and his tyres are eight laps older.",
            },
          ],
        },
        { title: "Why this call changed", rows: [{ lead: "", text: "lap 22: STAY_OUT (0.58)" }] },
      ],
      footer: null,
    },
    changed: "was STAY OUT (0.58) · L22",
  },
  scenarios: ["STAY", "PIT", "UCUT", "OCUT"].map((label, index) => ({
    key: label,
    label,
    fill: index === 1 ? 1 : 0.4,
    // #876 moved the bar width host-side to `fill_pct` and did not migrate
    // this stub. React drops `width: undefined%`, so all four bars measured
    // 382 px - winner and losers identical - and the smoke stayed green
    // because it counted rows and never measured one.
    fill_pct: index === 1 ? 100 : 40,
    score: "+0.71",
    is_winner: index === 1,
    // Migrated with the fields, not left to default. `is_scored` undefined is
    // falsy, so every bar would have rendered trackless - the same
    // never-migrated-stub shape the comment above already records once.
    is_enacted: index === 1,
    is_scored: true,
    note: "",
    bar_colour: index === 1 ? "#a78bfa" : "#d1d5db",
    label_colour: "#d1d5db",
    score_colour: "#ffffff",
  })),
  reasoning: ["Orchestrator", "Pace", "Tire", "Situation", "Radio", "Pit"].map((label) => ({
    key: label.toLowerCase(),
    label,
    segments: [{ text: `${label} body`, colour: "#ffffff", bold: false }],
  })),
  cards: Object.fromEntries(
    ["pace", "tire", "situation", "pit", "radio", "rag"].map((key) => [
      key,
      {
        headline: `${key} headline`,
        headline_colour: "#ffffff",
        // The PIT card is deliberately overfull. The scroll-reachability
        // check below needs a body that overflows, and on CI nothing did:
        // the runner's font metrics are shorter than a Windows desktop's, so
        // the four cards that overflow by 14 px locally overflow by nothing
        // there and the check failed its own precondition. A fixture that
        // guarantees the condition beats one that happens to meet it on the
        // machine you wrote it on.
        lines:
          key === "pit"
            ? Array.from({ length: 40 }, (_, i) => ({
                text: `body line ${i} - long enough that this card must scroll anywhere`,
                colour: "#d1d5db",
              }))
            : [{ text: "a body line", colour: "#d1d5db" }],
        status: "OK",
        glyph: "●",
        glyph_colour: "#10b981",
        // Structured, not markup. Sprint 8 turned the two tooltip formatters
        // into data, so a string here would be the stale-stub shape this file
        // already carries one scar from.
        // Five of the six carry model detail now, which is where the reasoning
        // tabs' per-agent bodies went. RAG keeps `null` on purpose: it is the
        // one agent with no `reasoning_lines` builder, and a card with no
        // tooltip is also what proves the tab-stop rule is not "every card".
        tooltip:
          key === "rag"
            ? null
            : {
                sections: [
                  ...(key === "radio"
                    ? [
                        {
                          title: "Radio",
                          rows: [
                            {
                              lead: "NOR PROBLEM",
                              text: "Rear grip is going away, especially through the last sector, and the balance moves every lap.",
                            },
                          ],
                        },
                      ]
                    : []),
                  {
                    title: "Reasoning",
                    rows: [{ lead: "", text: `${key} agent reasoning for this lap` }],
                  },
                  {
                    title: "Model detail",
                    rows: [
                      { lead: `${key}_first`, text: "1.234s" },
                      { lead: `${key}_second`, text: "56.7%" },
                    ],
                  },
                ],
                footer: null,
              },
      },
    ]),
  ),
  charts: {
    pace: {
      actual: [
        [21, 81.2],
        [22, 81.4],
      ],
      pred: [[22, 81.1]],
      band: [[22, 80.6, 81.6]],
      actual_colour: "#3b82f6",
      pred_colour: "#a78bfa",
      band_colour: "#a78bfa",
      // The tyre chart's axis, borrowed, and the same current-lap mark: the
      // two cards measure one quantity side by side.
      x_range: [20.5, 34],
      current_lap: 23,
      cursor_colour: "#9ca3af",
      prediction_lap: 22,
    },
    tire: {
      stints: [
        {
          compound: "MEDIUM",
          colour: "#e6c832",
          points: [
            [21, 81.2],
            [22, 81.4],
          ],
        },
      ],
      trend: [
        [21, 81.3],
        [22, 81.3],
      ],
      trend_colour: "#ffffff",
      cliff: { lo: 26, hi: 31, p50: 28 },
      cliff_colour: "#f59e0b",
      boundaries: [],
      boundary_colour: "#9ca3af",
      boundary_opacity: 0.31,
      x_range: [20.5, 34],
      y_range: [78.8, 83.8],
      current_lap: 23,
      cursor_colour: "#9ca3af",
    },
  },
  // FOUR branches and FIVE risks, which is what a real LLM lap returns:
  // measured on Melbourne 20-22, every lap came back with the orchestrator's
  // maximum of both. A thinner fixture develops the card against a payload a
  // quarter the size of the one it receives, and its empty space then reads as
  // a design problem rather than as a thin stub.
  //
  // They are not interchangeable. The first's `switch_to`
  // differs from its label so the label assertion can fail; the second carries
  // NO rationale, which is the row whose `detail` is null - the fixture that
  // makes "an unexpandable row is not a tab stop" able to fail. The first's
  // rationale is long enough to exceed two lines at this client, without which
  // the clamp assertion is about text that already fits.
  contingencies: {
    rows: [
      {
        trigger: "if RUS pits within two laps",
        switch_to: "PIT NOW",
        priority: "HIGH",
        rationale:
          "the undercut window shuts once he clears traffic and the gap to the car " +
          "behind is inside the pit loss, so the stop has to happen on this lap or " +
          "the position is gone until the second stint and the tyre delta stops " +
          "paying for the track position it cost to build",
        detail: {
          sections: [
            {
              title: "Contingency",
              rows: [
                { lead: "When", text: "if RUS pits within two laps" },
                { lead: "Then", text: "PIT NOW" },
                {
                  lead: "Why",
                  text:
                    "the undercut window shuts once he clears traffic and the gap to " +
                    "the car behind is inside the pit loss, so the stop has to happen " +
                    "on this lap or the position is gone until the second stint and " +
                    "the tyre delta stops paying for the track position it cost to build",
                },
              ],
            },
          ],
          footer: null,
        },
      },
      {
        trigger: "if the safety car is deployed before L28",
        switch_to: "STAY OUT",
        priority: "MEDIUM",
        rationale: "",
        detail: null,
      },
      {
        trigger: "if the front-left graining does not clear by L26",
        switch_to: "PIT NOW",
        priority: "MEDIUM",
        rationale: "the cliff P50 sits at L28 and the degradation slope has doubled since L18",
        detail: {
          sections: [
            {
              title: "Contingency",
              rows: [{ lead: "When", text: "if the front-left graining does not clear by L26" }],
            },
          ],
          footer: null,
        },
      },
      {
        trigger: "if rain reaches the circuit before the stop",
        switch_to: "STAY OUT",
        priority: "LOW",
        rationale: "a dry stop into a wet track wastes the set and the inters window opens later",
        detail: {
          sections: [
            {
              title: "Contingency",
              rows: [{ lead: "When", text: "if rain reaches the circuit before the stop" }],
            },
          ],
          footer: null,
        },
      },
    ],
    risks: [
      "SC probability is elevated and a green stop loses eleven seconds more",
      "rejoin into traffic behind the two-stoppers",
      "the cliff arrives before the stop if the graining does not clear",
      "the undercut from RUS lands first if he pits next lap",
      "no second set of this compound left for a late neutralisation",
    ],
    empty: null,
  },
  plan_timeline: {
    total_laps: 57,
    first_known_lap: 11,
    segments: [
      { lo: 11, hi: 17, compound: "MEDIUM", colour: "#e6c832", planned: false, left_pct: 17.86, width_pct: 12.5 },
      { lo: 18, hi: 23, compound: "HARD", colour: "#e6e6e6", planned: false, left_pct: 30.36, width_pct: 10.71 },
      { lo: 24, hi: 57, compound: "HARD", colour: "#e6e6e6", planned: true, left_pct: 41.07, width_pct: 58.93 },
    ],
    pit_lap: 24,
    pit_pct: 41.07,
    cliff: { lo: 27, hi: 32, colour: "#f59e0b", left_pct: 46.43, width_pct: 8.93 },
    current_lap: 23,
    current_pct: 39.29,
    caption: "Pit: L24 · Next: HARD · UCUT: RUS",
  },
  status_bar: { text: "lap 23 · streaming", transient: true },
};

const failures = [];
// Counted, not written down. Two artifacts in a row shipped a hand-typed
// total that had drifted from the real one ("36 tests" over 35, then
// "13 checks" over 12) - harmless until someone deletes a check and
// trusts the number.
let checks = 0;
const check = (ok, what) => {
  checks += 1;
  if (!ok) failures.push(what);
};

const server = await serveDist(DIST);
const browser = await chromium.launch();
const ctx = await browser.newContext({ viewport: CLIENT });
const page = await ctx.newPage();
watchPage(page, failures);

await page.addInitScript((view) => {
  window.pywebview = {
    api: {
      get_agents_view: async (sinceSeq) => (sinceSeq >= view.seq ? null : view),
      get_tick: async () => null,
      // Stubbed since #1004: `getConnection` falls back to `fetch("/api/connection")`
      // when the injected api has no such method, and the static server answers 404.
      // A missing stub method is a 404 in the console, not a thrown error, so it
      // costs three console failures and no clue about which call made them.
      get_connection: async () => ({ label: "Connected", colour: "#10b981" }),
    },
  };
}, VIEW);

await page.goto(`http://127.0.0.1:${server.address().port}/agents.html`, {
  waitUntil: "domcontentloaded",
});
await page.waitForSelector(".agent-card", { timeout: 5000 });

check((await page.locator(".agent-card").count()) === 6, "six agent cards");
check((await page.locator(".agent-chart canvas").count()) === 2, "two chart canvases");
// The BOXES, not just the canvases inside them. Counting canvases answered
// 2 for years while all six cards carried an `.agent-chart` - the renderer
// passed two conditionals as children, so the four cards without a chart
// received an ARRAY OF NULLS, which is truthy. Each of those four then
// reserved the box's 140 px `min-height` for nothing, which is where the
// dead strip the sprint-8 gate measured actually came from. A check on the
// contents cannot see an empty container.
check(
  (await page.locator(".agent-chart").count()) === 2,
  `only the two cards with a chart have a chart box (${await page.locator(".agent-chart").count()})`,
);
// The reasoning tabs are gone (#1020). What replaces the check is what
// replaces the panel: one sentence on the glass, and the whole narrative plus
// the memory block reachable from it - asserted in the keyboard walk below,
// because reachability is the property that had to survive the deletion.
check(
  (await page.locator(".why-narrative").innerText()).trim().length > 0,
  "the WHY module carries the narrative's first sentence",
);
check(
  (await page.locator(".reasoning-tab").count()) === 0,
  "and the tab panel it replaces is gone",
);
check((await page.locator(".scenario-row").count()) === 4, "four scenario rows");

// ...and the winner's bar is actually wider. Counting rows passed over a
// board where every bar rendered full width.
const barWidths = await page.evaluate(() =>
  [...document.querySelectorAll(".scenario-bar-fill")].map((el) => Math.round(el.getBoundingClientRect().width)),
);
check(barWidths[1] > barWidths[0] && barWidths[0] > 0, `the bars carry their fill (${barWidths})`);

// The fixture's prediction stopped a lap behind the car, so the pace chart owes
// the reader a tag saying where. Asserted on the RENDERED chart rather than on
// the option object: `graphic` is a config, and a config that never reaches the
// canvas is exactly the mechanism-instead-of-effect trap this file keeps.
const staleTag = await page.evaluate(() => {
  const el = document.querySelector(".slot-pace .chart");
  const chart = el && el.__pitwallChart;
  if (!chart) return null;
  const texts = [];
  chart.getZr().storage.traverse((shape) => {
    if (shape.type === "text" && shape.style && shape.style.text) texts.push(shape.style.text);
  });
  return texts;
});
check(
  Array.isArray(staleTag) && staleTag.some((t) => t.includes("prediction to L22")),
  `the pace chart names the lap its prediction stopped at (${JSON.stringify(staleTag)})`,
);
check(await page.locator(".orchestrator").isVisible(), "the orchestrator card");

// --- The four strata -------------------------------------------------------
//
// The 540 px left column is gone with the Qt split it came from, and so is the
// check that pinned it. What replaces it asserts the SHAPE the band claims: one
// row of four modules, in the reader's question order, and six consoles placed
// by name underneath.
const band = await page.evaluate(() => {
  const modules = [...document.querySelectorAll(".agents-band > *")];
  const box = (el) => el.getBoundingClientRect();
  return {
    count: modules.length,
    tops: modules.map((el) => Math.round(box(el).top)),
    lefts: modules.map((el) => Math.round(box(el).left)),
    widths: modules.map((el) => Math.round(box(el).width)),
  };
});
check(band.count === 4, `the band carries four modules (${band.count})`);
check(new Set(band.tops).size === 1, `all four sit on one row (${band.tops})`);
check(
  band.lefts.every((left, i) => i === 0 || left > band.lefts[i - 1]),
  `and in reading order (${band.lefts})`,
);
// PLAN is the `1fr`: it must be the widest, and it must actually absorb the
// slack rather than sitting at some fixed budget. The three fixed columns are
// 340 / 290 / 330 against a 1466 px inner width.
check(
  band.widths[3] > Math.max(band.widths[0], band.widths[1], band.widths[2]),
  `PLAN absorbs the width (${band.widths})`,
);

// --- The action is text, not a button ---------------------------------------
//
// It was a 200x70 filled pill: 14,000 px2 of the action's colour, which for
// PIT_NOW is the same DANGER red that means a fault everywhere else on the
// window. Asserted as PAINTED AREA rather than as "the pill class is gone",
// because the class going away is not the point and a differently-named fill
// would pass that.
const banner = await page.evaluate(() => {
  const action = document.querySelector(".orch-action");
  const card = document.querySelector(".orchestrator");
  const style = getComputedStyle(action);
  const cardStyle = getComputedStyle(card);
  const rect = action.getBoundingClientRect();
  const bandText = [...document.querySelectorAll(".agents-band *")]
    .map((el) => parseFloat(getComputedStyle(el).fontSize))
    .filter((size) => Number.isFinite(size));
  return {
    colour: style.color,
    fill: style.backgroundColor,
    rule: cardStyle.borderLeftColor,
    ruleWidth: parseFloat(cardStyle.borderLeftWidth),
    size: parseFloat(style.fontSize),
    // Every distinct type size in the band, largest first. Comparing against
    // "the largest OTHER size" put the confidence numeral in its own
    // comparison set and asked whether 22 > 22.
    ranks: [...new Set(bandText)].sort((a, b) => b - a),
    confidence: parseFloat(getComputedStyle(document.querySelector(".orch-conf-value")).fontSize),
    painted: Math.round(rect.width * rect.height),
    cardHeight: Math.round(card.getBoundingClientRect().height),
  };
});
// A transparent or fully-unset background is what "no fill" looks like in
// computed style; anything else is a painted rectangle.
check(
  banner.fill === "rgba(0, 0, 0, 0)" || banner.fill === "transparent",
  `the action carries no fill (${banner.fill})`,
);
check(
  banner.colour === banner.rule,
  `the rule and the word carry one identity colour (${banner.colour} vs ${banner.rule})`,
);
// The identity survives at the rule's area instead of the pill's. 4 px times
// the card height against the 14,000 px2 the badge painted.
check(
  banner.ruleWidth * banner.cardHeight < 0.1 * 14000,
  `the identity colour is painted at a fraction of the badge's area (${
    Math.round(banner.ruleWidth * banner.cardHeight)
  } px2)`,
);
// The pair reads as one unit: what, and how much to trust it. The confidence
// number used to render at 11 px, the size of an axis tick, beside a 26 px
// word - so the window's two most decisive values sat five ranks apart.
check(
  banner.ranks[0] === banner.size && banner.ranks[1] === banner.confidence,
  `the action and its confidence are the band's top two type ranks (${banner.ranks})`,
);

// --- The PLAN timeline ------------------------------------------------------
//
// Four rectangles and two rules, so the assertions are the rendered geometry.
// The one thing this lane must never do is make a lap it never saw look like a
// lap it did: measured against the track's own box, not against a colour name.
const plan = await page.evaluate(() => {
  const track = document.querySelector(".plan-lane-stints").getBoundingClientRect();
  const rel = (el) => {
    const r = el.getBoundingClientRect();
    return {
      left: Math.round(((r.left - track.left) / track.width) * 1000) / 10,
      right: Math.round(((r.right - track.left) / track.width) * 1000) / 10,
      fill: getComputedStyle(el).backgroundColor,
    };
  };
  const stints = [...document.querySelectorAll(".plan-stint")].map((el) => ({
    ...rel(el),
    planned: el.classList.contains("is-planned"),
  }));
  const now = document.querySelector(".plan-now");
  const cliff = document.querySelector(".plan-cliff");
  return {
    stints,
    nowLeft: now ? Math.round(((now.getBoundingClientRect().left - track.left) / track.width) * 1000) / 10 : null,
    cliffFill: cliff ? getComputedStyle(cliff).backgroundColor : null,
    trackFill: getComputedStyle(document.querySelector(".plan-lane-stints")).backgroundColor,
    labels: [...document.querySelectorAll(".plan-end, .plan-cursor-label")].map((el) => el.innerText),
    overflow: Math.max(...stints.map((s) => s.right)) - 100,
  };
});
check(plan.stints.length === 3, `three stints on the lane (${plan.stints.length})`);
// The window opened on lap 11, so the first ten laps are TRACK. A lane that
// started its first bar at zero would be claiming a stint nobody reported.
check(
  plan.stints[0].left > 5,
  `the laps this window never saw are blank track (first bar at ${plan.stints[0].left}%)`,
);
// Filled versus hollow, asserted as a computed fill rather than as a class:
// the class is the mechanism, the paint is the encoding.
check(
  plan.stints.filter((s) => s.planned).every((s) => s.fill === "rgba(0, 0, 0, 0)"),
  `the planned stint is hollow (${plan.stints.filter((s) => s.planned).map((s) => s.fill)})`,
);
check(
  plan.stints.filter((s) => !s.planned).every((s) => s.fill !== "rgba(0, 0, 0, 0)"),
  "and the stints already run are filled",
);
// **The empty track may not look like a stint.** This is the pair that came
// out the same tone on the first draft: a run stint over `--qt-elevated` at
// 0.42 alpha was indistinguishable from the ground behind it.
check(
  plan.trackFill !== plan.stints[0].fill,
  `an unrun lap and a run stint are different paint (${plan.trackFill} vs ${plan.stints[0].fill})`,
);
check(plan.overflow <= 0.5, `no bar runs past the flag (${plan.overflow}% over)`);
check(
  plan.labels.includes("L1") && plan.labels.includes("L57") && plan.labels.includes("L23"),
  `the axis names its ends and the current lap (${plan.labels.join(", ")})`,
);
check(
  plan.nowLeft !== null && Math.abs(plan.nowLeft - 39.3) < 1.5,
  `the NOW cursor sits on the current lap (${plan.nowLeft}%)`,
);

// The consoles, by rendered geometry rather than by class name: a card can
// carry the right class and be placed in the wrong area.
const grid = await page.evaluate(() => {
  const box = (selector) => {
    const el = document.querySelector(selector);
    if (!el) return null;
    const r = el.getBoundingClientRect();
    return { left: Math.round(r.left), right: Math.round(r.right), top: Math.round(r.top) };
  };
  return {
    pace: box(".slot-pace"),
    tire: box(".slot-tire"),
    situation: box(".slot-situation"),
    pit: box(".slot-pit"),
    radio: box(".slot-radio"),
    rag: box(".slot-rag"),
  };
});
check(
  grid.pace.right <= grid.tire.left && grid.tire.right <= grid.situation.left,
  "the three columns run pace, tire, then the side stack",
);
check(
  grid.situation.top < grid.pit.top && grid.situation.left === grid.pit.left,
  "SITUATION sits over PIT in one column",
);
check(
  grid.radio.left === grid.pace.left && grid.radio.right >= grid.tire.right,
  "RADIO spans the two chart columns",
);
check(
  grid.rag.left === grid.situation.left && grid.rag.top >= grid.radio.top,
  "and RAG closes the corner",
);

// **Nothing may overflow the client.** The old harness ran at 1320x900 - 67 px
// taller than the window - so a layout that did not fit could not be seen here
// at all. Asserted on the document, which is what the OS window scrolls or
// clips, not on any one panel.
const overflow = await page.evaluate(() => {
  const h = (selector) => {
    const el = document.querySelector(selector);
    return el ? Math.round(el.getBoundingClientRect().height) : null;
  };
  return {
    vertical: document.documentElement.scrollHeight - document.documentElement.clientHeight,
    horizontal: document.documentElement.scrollWidth - document.documentElement.clientWidth,
    // In the failure message, because "it does not fit" is not actionable and
    // "the band is 195 and the grid wants 600" is.
    client: document.documentElement.clientHeight,
    // WHICH element sticks out, not just that something does. `.agent-card`
    // is deliberately `overflow: visible` so nothing can clip its tooltip, so
    // a card whose content exceeds its cell spills into the document rather
    // than scrolling, and the document is where it shows up.
    past: [...document.querySelectorAll(".agents-body *")]
      .map((el) => ({
        what: `${el.tagName.toLowerCase()}.${(el.className || "").toString().split(" ").join(".")}`,
        over: Math.round(el.getBoundingClientRect().bottom - document.documentElement.clientHeight),
      }))
      .filter((e) => e.over > 0)
      .sort((a, b) => b.over - a.over)
      .slice(0, 4),
    strata: {
      header: h(".header-bar"),
      band: h(".agents-band"),
      grid: h(".agents-grid"),
      status: h(".status-bar"),
    },
  };
});
check(
  overflow.vertical <= 0 && overflow.horizontal <= 0,
  `the window fits its client (${JSON.stringify(overflow)})`,
);

// The tooltip must appear only where the host sent content, and NOTHING
// may clip it. Two fixes failed here and the second was passed by a check
// written right here: it asserted the card's `overflow` and the popup's
// DOM placement — the MECHANISM — and never hovered or measured. The
// popup had escaped the card and was being amputated 130 px by
// `.agents-right` instead. So this hovers and measures the rendered box
// against every scrolling ancestor, which is the EFFECT and the only
// thing that was ever in question.
check((await page.locator(".agent-tooltip").count()) === 0, "no tooltip before hover");
await page.locator(".agent-card").nth(4).hover();
await page.waitForTimeout(200);
check((await page.locator(".agent-tooltip").count()) === 1, "one tooltip at a time");

const clip = await page.evaluate(() => {
  const box = document.querySelector(".agent-tooltip").getBoundingClientRect();
  const cut = [];
  for (let node = document.querySelector(".agent-tooltip").parentElement; node; node = node.parentElement) {
    const style = getComputedStyle(node);
    if (style.overflowX === "visible" && style.overflowY === "visible") continue;
    const edge = node.getBoundingClientRect();
    if (edge.left > box.left + 1 || edge.right < box.right - 1) cut.push(node.className || node.tagName);
  }
  return { cut, onScreen: box.left >= 0 && box.right <= window.innerWidth };
});
check(clip.cut.length === 0, `nothing clips the tooltip (cut by ${clip.cut.join(", ")})`);
check(clip.onScreen, "the tooltip stays inside the viewport");

// ...and it must not cover the card it belongs to. The popup used to open at
// the card's own left edge, 32 px down: measured at 68,877 px² of overlap on
// a 216 px card, so hovering a card to read more HID the card. Asserted as
// rendered geometry, because "the tooltip exists and is on screen" was
// already true the whole time it was doing this.
const overlap = await page.evaluate(() => {
  const tip = document.querySelector(".agent-tooltip").getBoundingClientRect();
  const card = document.querySelectorAll(".agent-card")[4].getBoundingClientRect();
  const x = Math.max(0, Math.min(tip.right, card.right) - Math.max(tip.left, card.left));
  const y = Math.max(0, Math.min(tip.bottom, card.bottom) - Math.max(tip.top, card.top));
  return Math.round(x * y);
});
check(overlap === 0, `the tooltip does not cover its own card (${overlap} px2)`);

// **And the same when the popup fits on neither side**, which is the branch a
// wide card reaches and the one that used to place it on its own subject.
// Forced rather than waited for: the width is `max-content`, so whether a real
// tooltip trips it depends on the fonts the machine happens to have - it fired
// on a CI runner and never here, from the same code and the same fixture.
await page.mouse.move(0, 0);
await page.waitForTimeout(150);
const widen = await page.addStyleTag({
  content: ".agent-tooltip { min-width: 1200px !important; }",
});
await page.locator(".agent-card").nth(4).hover();
await page.waitForTimeout(200);
const forced = await page.evaluate(() => {
  const tip = document.querySelector(".agent-tooltip").getBoundingClientRect();
  const card = document.querySelectorAll(".agent-card")[4].getBoundingClientRect();
  const x = Math.max(0, Math.min(tip.right, card.right) - Math.max(tip.left, card.left));
  const y = Math.max(0, Math.min(tip.bottom, card.bottom) - Math.max(tip.top, card.top));
  return {
    overlap: Math.round(x * y),
    onScreen: tip.top >= 0 && tip.bottom <= window.innerHeight,
    width: Math.round(tip.width),
  };
});
check(
  forced.overlap === 0,
  `a popup too wide for either side still clears its card (${JSON.stringify(forced)})`,
);
check(forced.onScreen, `and stays on screen (${JSON.stringify(forced)})`);
await widen.evaluate((node) => node.remove());
await page.mouse.move(0, 0);
await page.waitForTimeout(150);

// Qt's `showMessage(text, 1500)` clears itself. The port typed a
// `transient` flag and read it nowhere until #871.
// The scrollbars are HIDDEN, not deleted. Víctor asked for the bars inside
// the cards to go; the tempting wrong fix is `overflow: hidden`, which puts
// back the clipping the migration README records as a defect of the Qt window
// being replaced ("the right column clipped mid-card"). So this asserts the
// content is still REACHABLE, which is the part that can regress silently -
// the chrome being gone is visible to anyone who looks.
//
// With a REAL WHEEL, not `el.scrollTop = 999`. An `overflow: hidden` element
// is still scrollable from script - only the USER is blocked - so the
// scripted version passed against the exact mutation it was written to
// catch. Same mechanism-instead-of-effect trap the sprint-3 gate found in
// this file's tooltip check.
const overflowing = await page.evaluate(() =>
  [...document.querySelectorAll(".agent-card-body")]
    .map((el, i) => ({ i, over: el.scrollHeight - el.clientHeight }))
    .filter((e) => e.over > 0)
    .map((e) => e.i),
);
check(overflowing.length > 0, `something overflows, or this checks nothing`);

const wheeled = [];
for (const index of overflowing) {
  const box = await page
    .locator(".agent-card-body")
    .nth(index)
    .boundingBox();
  await page.mouse.move(box.x + box.width / 2, box.y + box.height / 2);
  await page.mouse.wheel(0, 200);
  await page.waitForTimeout(120);
  wheeled.push(
    await page.evaluate(
      (i) => document.querySelectorAll(".agent-card-body")[i].scrollTop,
      index,
    ),
  );
}
check(
  wheeled.every((top) => top > 0),
  `a wheel over an overflowing body still reaches its content (${JSON.stringify(wheeled)})`,
);

check(
  (await page.locator(".status-bar").innerText()).includes("lap 23"),
  "the status bar starts with the lap",
);
await page.waitForTimeout(1800);
check((await page.locator(".status-bar").innerText()).trim() === "", "the status bar auto-clears");

// --- Reachable without a mouse --------------------------------------------
//
// The reasoning tabs were `<button>`s, so the per-agent bodies they held were
// reachable with Tab by anyone. The band replaces them with card tooltips, and
// a mouse-only popup would make this window strictly WORSE for a keyboard user
// than the panel it retires: zero of the dumps against six.
//
// Driven with real key presses, not by calling `.focus()`. Focus set from
// script reaches an element the tab order may never visit, which is precisely
// the thing in question.
await page.mouse.move(0, 0);
await page.waitForTimeout(150);

const stops = await page.evaluate(() =>
  [...document.querySelectorAll(".agent-card")].map((el) => ({
    slot: [...el.classList].find((c) => c.startsWith("slot-")),
    tabIndex: el.tabIndex,
    hasTip: el.getAttribute("tabindex") !== null,
  })),
);
check(
  stops.filter((s) => s.tabIndex === 0).length === 5,
  `the five consoles with content are tab stops (${JSON.stringify(stops.map((s) => [s.slot, s.tabIndex]))})`,
);
check(
  stops.find((s) => s.slot === "slot-rag").tabIndex === -1,
  "and the one with no tooltip is not, so it is not an empty stop in the way",
);

// Walk the tab order and collect what each stop reveals. The loop bounds
// itself on the number of stops rather than on a fixed count, so adding a
// console cannot leave one silently unvisited.
const reached = [];
await page.locator(".header-bar").click();
for (let i = 0; i < 40 && reached.length < 6; i += 1) {
  await page.keyboard.press("Tab");
  await page.waitForTimeout(60);
  const seen = await page.evaluate(() => {
    const active = document.activeElement;
    const isCard = active && active.classList.contains("agent-card");
    const isWhy = active && active.classList.contains("why-panel");
    if (!isCard && !isWhy) return null;
    const tip = document.querySelector(".agent-tooltip");
    return {
      slot: isWhy ? "why" : [...active.classList].find((c) => c.startsWith("slot-")),
      described: active.getAttribute("aria-describedby"),
      tipId: tip ? tip.id : null,
      text: tip ? tip.innerText : null,
      clipped: tip ? tip.scrollHeight - tip.clientHeight : null,
    };
  });
  if (seen) reached.push(seen);
}
// Six: the WHY module and the five consoles with content. **The whole reason
// the reasoning tabs could be deleted** - they were `<button>`s, so everything
// they held was reachable with Tab, and the replacement has to be too.
check(
  reached.length === 6,
  `Tab alone reaches WHY and all five consoles (${reached.map((r) => r.slot).join(", ")})`,
);
check(
  reached.filter((r) => r.slot !== "why").every((r) => r.text && r.text.includes("Model detail")),
  "each console opens its model detail on focus",
);
// The orchestrator narrative and the DecisionMemory block: the two things the
// retired panel held that live nowhere else on the window.
{
  const why = reached.find((r) => r.slot === "why");
  check(
    why.text.includes("Reasoning") && why.text.includes("Why this call changed"),
    `and WHY opens the whole narrative and the memory block (${JSON.stringify(why.text ?? "")})`,
  );
}
check(
  reached.every((r) => r.described && r.described === r.tipId),
  "with the popup named by aria-describedby",
);
// **Nothing silently cut.** `.tip-text` used to carry a 4-line clamp, so a
// long row was amputated with no scrollbar and no ellipsis to say so.
check(
  reached.every((r) => r.clipped <= 0),
  `and nothing is clipped inside it (${reached.map((r) => r.clipped).join(", ")})`,
);

// **A popup taller than its 22rem cap must be reachable, by either hand.**
// It has a visible scrollbar and nothing could drive it: the pointer crossing
// the 10 px gap fired the card's `mouseleave` and unmounted it, and the
// keyboard could not reach an element the tab order never visits. A RADIO lap
// with the model-detail sections appended after its rows overflows routinely.
//
// Forced, not waited for - the cap depends on the fonts the machine has.
{
  const tall = await page.addStyleTag({
    content: ".agent-tooltip { max-height: 60px !important; }",
  });
  await page.locator(".agent-card").nth(4).hover();
  await page.waitForTimeout(200);
  const overflowed = await page.evaluate(() => {
    const tip = document.querySelector(".agent-tooltip");
    return tip ? tip.scrollHeight - tip.clientHeight : null;
  });
  check(overflowed > 0, `the probe really does overflow the popup (${overflowed})`);

  // The pointer: from the card into the popup, which must survive the crossing.
  const box = await page.locator(".agent-tooltip").boundingBox();
  await page.mouse.move(box.x + box.width / 2, box.y + box.height / 2);
  await page.waitForTimeout(220);
  check(
    (await page.locator(".agent-tooltip").count()) === 1,
    "the popup survives the pointer crossing the gap into it",
  );
  await page.mouse.wheel(0, 200);
  await page.waitForTimeout(150);
  const wheeled = await page.evaluate(() => document.querySelector(".agent-tooltip")?.scrollTop ?? 0);
  check(wheeled > 0, `and a wheel over it reaches its tail (${wheeled})`);

  // The keyboard: the arrows are forwarded from the card, so the reader never
  // has to reach an element Tab does not visit.
  await page.mouse.move(0, 0);
  await page.waitForTimeout(220);
  // Blur first. `.focus()` on the element that is ALREADY `activeElement` fires
  // no focus event, and the keyboard walk above left focus on this very card -
  // so the probe read a closed popup and blamed the arrows.
  await page.evaluate(() => document.activeElement?.blur());
  await page.waitForTimeout(80);
  await page.locator(".agent-card").nth(4).focus();
  await page.waitForTimeout(200);
  const beforeArrow = await page.evaluate(() => {
    const tip = document.querySelector(".agent-tooltip");
    const active = document.activeElement;
    return {
      open: Boolean(tip),
      id: tip ? tip.id : null,
      over: tip ? tip.scrollHeight - tip.clientHeight : null,
      focused: active ? [...active.classList].join(".") : null,
    };
  });
  check(beforeArrow.open, `the popup is open under keyboard focus (${JSON.stringify(beforeArrow)})`);
  await page.keyboard.press("ArrowDown");
  await page.keyboard.press("ArrowDown");
  await page.waitForTimeout(150);
  const arrowed = await page.evaluate(() => document.querySelector(".agent-tooltip")?.scrollTop ?? 0);
  check(arrowed > 0, `and ArrowDown from the card scrolls it too (${arrowed})`);

  await tall.evaluate((node) => node.remove());
  await page.mouse.move(0, 0);
  await page.waitForTimeout(220);
}

await page.locator(".agent-card").nth(4).focus();
await page.waitForTimeout(200);
await page.keyboard.press("Escape");
await page.waitForTimeout(160);
check(
  (await page.locator(".agent-tooltip").count()) === 0,
  "Escape closes it without moving focus away",
);

await ctx.close();

// --- and it must NOT clear while the producer is still talking --------------
//
// Qt re-arms `showMessage` on every broadcast, so the message is visible
// the whole time it is streaming. Keyed on the text instead of the tick it
// re-armed once per LAP, and the bar sat blank for the other eighty
// seconds of it. The settled stub above cannot see that: it is the dead
// producer, the case that already worked.
const live = await browser.newContext({ viewport: CLIENT });
const livePage = await live.newPage();
watchPage(livePage, failures, "live");
await livePage.addInitScript((view) => {
  let seq = 0;
  window.pywebview = {
    api: {
      // A new sequence every poll, with the SAME status text, which is
      // what a real producer does for the ~85 s a lap lasts.
      //
      // **A DEEP copy, and that is not fussiness.** `AgentsViewBuilder.build`
      // constructs a fresh dict per tick, so every nested object reaching
      // React has a new identity and the charts' `useMemo([series])`
      // recomputes - which is what makes them call `setOption` ten times a
      // second. A shallow spread keeps `charts` identical, the memo never
      // fires, and the stub renders a chart that redraws once and sits still:
      // the animation check below passed against BOTH the defect and its fix
      // until this line changed.
      get_agents_view: async () => ({ ...structuredClone(view), seq: ++seq }),
      get_tick: async () => null,
      get_connection: async () => ({ label: "Connected", colour: "#10b981" }),
    },
  };
}, VIEW);
await livePage.goto(`http://127.0.0.1:${server.address().port}/agents.html`, {
  waitUntil: "domcontentloaded",
});
await livePage.waitForSelector(".status-bar", { timeout: 5000 });
await livePage.waitForTimeout(2500);
check(
  (await livePage.locator(".status-bar").innerText()).includes("lap 23"),
  "the status bar stays visible while the producer streams",
);

// A chart under a streaming producer must be STILL. The data behind it is
// identical on every poll here - only `seq` moves - so any pixel that changes
// between two captures is the chart redrawing itself, not new information.
//
// This is the check that was missing when Víctor reported the dashed cliff
// marker flickering. `notMerge: true` makes each `setOption` look like a
// fresh series, so ECharts runs the ENTRANCE animation, which measured
// ~1200 ms to settle against a ~100 ms push cadence: it never once finished.
// Band 4 had the identical defect and was fixed a sprint earlier - one copy
// fixed, its twin left, which is this repo's dominant defect.
// BOTH cards, not just the one Víctor saw flicker: they share `CHART_BASE`,
// so a guard on one leaves the other free to regress.
check(
  await staysStill(livePage, ".agent-chart .chart"),
  "the AGENTS charts schedule no animation while the producer streams",
);

await live.close();

// --- The IDLE window before the first view, and #1004's two states ------------
//
// `get_agents_view` returns null until a tick has arrived, so this window has no
// connection word at all for the whole startup: measured on the real path, null
// on all 169 samples across 11 s, of which the last 3 s had the socket UP and the
// arcade loading its session. The status bar said "Waiting for arcade stream…"
// throughout, which describes only the first 8 s.
//
// The word is polled with `useConnection` while `view` is null, and the two
// states must be TELLABLE APART from the rendered bar. Reading one of them would
// pass on a build that hardcoded the string.
async function idleStatusBar(connection) {
  const context = await browser.newContext({ viewport: CLIENT });
  const idlePage = await context.newPage();
  watchPage(idlePage, failures, `idle/${connection}`);
  // The PAIR goes in as the argument. `addInitScript` serialises the function
  // and runs it in the page, so a Node-side constant it closes over is simply
  // not there - which surfaced as `CONNECTION_COLOURS is not defined` on three
  // pages at once, and only because every page watches its console now.
  await idlePage.addInitScript((state) => {
    window.pywebview = {
      api: {
        get_agents_view: async () => null,
        get_tick: async () => null,
        get_connection: async () => state,
      },
    };
  }, { label: connection, colour: CONNECTION_COLOURS[connection] });
  await idlePage.goto(`http://127.0.0.1:${server.address().port}/agents.html`, {
    waitUntil: "domcontentloaded",
  });
  await idlePage.waitForSelector(".agents-window", { timeout: 5000 });
  // 1.2 s, because `useConnection` polls at 1 Hz behind `whenBridgeReady`: a
  // shorter wait reads the null-connection frame, which renders the same
  // sentence for both stubs and would make this pass while seeing one branch.
  await idlePage.waitForTimeout(1200);
  const bar = (await idlePage.locator(".status-bar").textContent()) ?? "";
  await context.close();
  return bar;
}

const idleUp = await idleStatusBar("Connected");
const idleDown = await idleStatusBar("Connecting...");
check(
  idleUp !== idleDown,
  `the idle status bar tells the two socket states apart (up: "${idleUp}", down: "${idleDown}")`,
);
check(
  idleUp.includes("Connected"),
  `with the socket up the idle bar says so ("${idleUp}")`,
);

// The view, once it exists, still wins over the polled word: it is host-built and
// travels WITH the payload, so it cannot disagree with the lap beside it. Stubbed
// with a DISAGREEING connection, or a bar that ignored the poll entirely would
// pass this too.
const servedBar = await (async () => {
  const context = await browser.newContext({ viewport: CLIENT });
  const viewPage = await context.newPage();
watchPage(viewPage, failures, "frozen");
  await viewPage.addInitScript(
    ([view, state]) => {
      window.pywebview = {
        api: {
          get_agents_view: async (sinceSeq) => (sinceSeq >= view.seq ? null : view),
          get_tick: async () => null,
          get_connection: async () => state,
        },
      };
    },
    [VIEW, { label: "Connecting...", colour: CONNECTION_COLOURS["Connecting..."] }],
  );
  await viewPage.goto(`http://127.0.0.1:${server.address().port}/agents.html`, {
    waitUntil: "domcontentloaded",
  });
  await viewPage.waitForSelector(".agent-card", { timeout: 5000 });
  await viewPage.waitForTimeout(1200);
  const bar = (await viewPage.locator(".status-bar").textContent()) ?? "";
  await context.close();
  return bar;
})();
check(
  servedBar === "lap 23 · streaming",
  `a rendered view keeps its own host-built status line, not the polled word ("${servedBar}")`,
);

// ── The neutralisation chip, all three treatments ────────────────────────────
//
// The window said nothing about a safety car before this except as RCM prose
// inside the RADIO card. What is asserted here is the EFFECT: a non-green status
// has to be visibly heavier than green, because the failure this replaces is a
// chip that only swapped its text and that a reader misses at a glance.
//
// The unknown branch is asserted too, and it is the one that costs if it is
// wrong: `null` means the loader had no entry for the lap, which is not a green
// track.
const chipFor = async (header) => {
  const ctx = await browser.newContext({ viewport: CLIENT });
  const p = await ctx.newPage();
  watchPage(p, failures);
  await p.addInitScript(
    (view) => {
      window.pywebview = {
        api: {
          get_agents_view: async (s) => (s >= view.seq ? null : view),
          get_tick: async () => null,
          get_connection: async () => ({ label: "Connected", colour: "#10b981" }),
        },
      };
    },
    { ...VIEW, header: { ...VIEW.header, ...header } },
  );
  await p.goto(`http://127.0.0.1:${server.address().port}/agents.html`, {
    waitUntil: "domcontentloaded",
  });
  await p.waitForSelector(".agent-card", { timeout: 5000 });
  await p.waitForTimeout(400);
  const seen = await p.evaluate(() => {
    const chips = [...document.querySelectorAll(".header-bar .chip")];
    const el = chips.find((c) => /GREEN|SAFETY CAR|VSC|RED|YELLOW|NO STATUS/.test(c.textContent));
    if (!el) return null;
    const s = getComputedStyle(el);
    return { text: el.textContent, weight: s.fontWeight, background: s.backgroundColor };
  });
  await ctx.close();
  return seen;
};

const greenChip = await chipFor({ track_status: "GREEN", track_status_colour: [16, 185, 129] });
const scChip = await chipFor({ track_status: "SAFETY CAR", track_status_colour: [255, 140, 0] });
const blindChip = await chipFor({ track_status: null, track_status_colour: null });

check(greenChip?.text === "GREEN", `the header carries the track status ("${greenChip?.text}")`);
check(scChip?.text === "SAFETY CAR", `a safety car reaches the header ("${scChip?.text}")`);
check(
  scChip?.background === "rgb(255, 140, 0)",
  `the safety car chip is FILLED with the wire's own colour (${scChip?.background})`,
);
check(
  Number(scChip?.weight) > Number(greenChip?.weight),
  `a neutralised track is heavier than a green one (${scChip?.weight} vs ${greenChip?.weight})`,
);
check(
  blindChip?.text === "NO STATUS",
  `an absent status says so rather than rendering green ("${blindChip?.text}")`,
);
check(
  blindChip?.background !== "rgb(255, 140, 0)",
  "an absence does not borrow the alarm's weight",
);

// --- Scenario: a card that scrolls SAYS so (#1077) ---------------------------
//
// `agent-card-body` has always been `overflow: auto`, and `qt-base.css` hides
// every scrollbar on purpose - that rule's own comment ends by naming this as
// the debt it leaves. Measured at the 1226 x 593 client a 1280x720 screen gives
// this window, the PACE and TIRE cards each hide 51 px, which is their whole Lap
// axis, and SITUATION and PIT cut a line through the middle of its glyphs.
//
// The assertion is the EFFECT: a body with content beyond its box carries the
// mask, and a body that fits carries none. Reading the CSS rule would pass on a
// build that applied the fade unconditionally, which is the opposite defect and
// just as wrong: a permanent fade on a card that fits says there is more when
// there is not.
{
  const narrow = await browser.newContext({ viewport: { width: 1226, height: 593 } });
  const narrowPage = await narrow.newPage();
  watchPage(narrowPage, failures, "scroll-affordance");
  await narrowPage.addInitScript((view) => {
    window.pywebview = {
      api: {
        get_agents_view: async (sinceSeq) => (sinceSeq >= view.seq ? null : view),
        get_tick: async () => null,
        get_connection: async () => ({ label: "Connected", colour: "#10b981" }),
      },
    };
  }, VIEW);
  await narrowPage.goto(`http://127.0.0.1:${server.address().port}/agents.html`, {
    waitUntil: "domcontentloaded",
  });
  await narrowPage.waitForSelector(".agent-card-body", { timeout: 10000 });
  await narrowPage.evaluate(() => document.fonts?.ready);
  await narrowPage.waitForTimeout(1200);

  const bodies = await narrowPage.evaluate(() =>
    [...document.querySelectorAll(".agent-card-body")].map((node) => ({
      title: node.parentElement?.querySelector(".agent-title")?.textContent?.trim() ?? "?",
      hidden: node.scrollHeight - node.clientHeight,
      masked: getComputedStyle(node).maskImage !== "none",
      reachable: getComputedStyle(node).overflowY !== "hidden",
    })),
  );

  // The discovery step first: this client must actually produce an overflowing
  // card, or every assertion below is about the empty set.
  const overflowing = bodies.filter((b) => b.hidden > 1);
  check(
    overflowing.length > 0,
    `affordance: the 1226x593 client overflows at least one card (${JSON.stringify(bodies)})`,
  );
  for (const body of bodies) {
    check(
      body.masked === body.hidden > 1,
      `affordance: ${body.title} fades exactly when it has more below ` +
        `(hidden ${body.hidden}, masked ${body.masked})`,
    );
    // The mask is a signal, never a substitute for reaching the content. Qt
    // CLIPPED here and the migration README records that as a defect of the
    // window being replaced.
    check(
      body.reachable,
      `affordance: ${body.title} stays scrollable rather than clipped`,
    );
  }
  await narrow.close();
}


// ---------------------------------------------------------------------------
// The hover readout on the two AGENTS cards (#999).
//
// The pointer PARKS, for the reason the DATA guards state at length: an ECharts
// tooltip on these charts is visible 0 times out of 25 with the pointer still
// and 14 out of 14 while it moves, because `notMerge: true` destroys it between
// mousemoves. A sweeping probe passes over a readout that shows a reader
// nothing.
//
// These are also the SECOND surface for the pixel mapping. The stack's grid
// inset is not this one (44/10 here), so a mapping that hardwired one chart's
// constants would answer this chart's laps wrong, and a stack-only guard could
// never see it.
// ---------------------------------------------------------------------------
{
  const hoverCtx = await browser.newContext({ viewport: CLIENT });
  const hoverPage = await hoverCtx.newPage();
  watchPage(hoverPage, failures, "agents-hover");
  await hoverPage.addInitScript((view) => {
    window.pywebview = {
      api: {
        get_agents_view: async (since) => (since >= view.seq ? null : view),
        get_connection: async () => ({ label: "Connected", colour: "#10b981" }),
      },
    };
  }, VIEW);
  await hoverPage.goto(`http://127.0.0.1:${server.address().port}/agents.html`, {
    waitUntil: "domcontentloaded",
  });
  await hoverPage.waitForSelector(".chart canvas", { timeout: 10000 });
  await hoverPage.waitForTimeout(600);

  check(
    (await hoverPage.locator(".chart-hover").count()) === 2,
    "agents hover: both cards carry a hover surface",
  );
  check(
    (await hoverPage.locator(".chart-readout").count()) === 0,
    "agents hover: nothing is shown before the pointer arrives",
  );

  /** Park on a chart at an exact LAP, converted through that chart's own axis. */
  const parkLap = async (index, lap) => {
    const host = hoverPage.locator(".chart").nth(index);
    const hostBox = await host.boundingBox();
    const px = await hoverPage.evaluate(
      (args) =>
        document
          .querySelectorAll(".chart")
          [args[0]].__pitwallChart.convertToPixel({ gridIndex: 0 }, [args[1], 0])[0],
      [index, lap],
    );
    await hoverPage.mouse.move(hostBox.x + px - 1, hostBox.y + hostBox.height * 0.5);
    await hoverPage.waitForTimeout(60);
    await hoverPage.mouse.move(hostBox.x + px, hostBox.y + hostBox.height * 0.5);
    await hoverPage.waitForTimeout(200);
  };
  const readout = async () =>
    (await hoverPage.locator(".chart-readout").allInnerTexts()).map((text) =>
      text.replace(/\s+/g, " ").trim(),
    );

  // --- the pace card, at a lap where EVERY field exists ---------------------
  await parkLap(0, 22);
  const full = await readout();
  check(full.length === 1, `agents hover: exactly one readout is open (${full.length})`);
  check(
    full[0].startsWith("LAP 22"),
    `agents hover: the lap comes from this chart's own axis, not the stack's inset (${full[0]})`,
  );
  check(
    full[0].includes("81.4"),
    `agents hover: the pace card reads the actual at the hovered lap (${full[0]})`,
  );
  check(
    full[0].includes("81.1"),
    `agents hover: and the prediction (${full[0]})`,
  );
  check(
    full[0].includes("80.6-81.6"),
    `agents hover: and the P10-P90 band whole, not one of its ends (${full[0]})`,
  );

  // --- the same card one lap earlier, where TWO of the three are absent ----
  // The fixture's prediction starts at 22 and its actual at 21, so lap 21 is
  // the state where a card that hid the whole box on one missing field, or
  // printed a neighbouring lap's number, would be caught.
  await parkLap(0, 21);
  const partial = await readout();
  check(
    partial.length === 1 && partial[0].startsWith("LAP 21"),
    `agents hover: the box stays open on a lap with missing fields (${partial[0]})`,
  );
  check(
    partial[0].includes("81.2"),
    `agents hover: the actual still reads at lap 21 (${partial[0]})`,
  );
  check(
    (partial[0].match(/—/g) ?? []).length === 2,
    `agents hover: the prediction and the band print an em dash each, not the previous lap's value (${partial[0]})`,
  );

  // --- the tyre card --------------------------------------------------------
  await parkLap(1, 22);
  const tyre = await readout();
  check(
    tyre.length === 1 && tyre[0].startsWith("LAP 22"),
    `agents hover: the tyre card reads its own axis (${tyre[0]})`,
  );
  check(
    tyre[0].includes("81.4"),
    `agents hover: the observed value comes from the stint that covers the lap (${tyre[0]})`,
  );
  check(
    tyre[0].includes("81.3"),
    `agents hover: and the rolling trend (${tyre[0]})`,
  );
  check(
    tyre[0].includes("L26-31"),
    `agents hover: the cliff prints as the lap RANGE it is, not a value at this lap (${tyre[0]})`,
  );

  // --- it leaves ------------------------------------------------------------
  await hoverPage.mouse.move(2, 2);
  await hoverPage.waitForTimeout(250);
  check(
    (await hoverPage.locator(".chart-readout").count()) === 0 &&
      (await hoverPage.locator(".chart-hover-cursor").count()) === 0,
    "agents hover: box and cursor both go when the pointer leaves",
  );

  // --- hovering pushes no options -------------------------------------------
  await hoverPage.evaluate(() => {
    window.__pushes = 0;
    document.querySelectorAll(".chart").forEach((host) => {
      const chart = host.__pitwallChart;
      const real = chart.setOption.bind(chart);
      chart.setOption = (...args) => {
        window.__pushes += 1;
        return real(...args);
      };
    });
  });
  const sweep = await hoverPage.locator(".chart").first().boundingBox();
  for (let i = 0; i < 60; i += 1) {
    await hoverPage.mouse.move(sweep.x + 50 + i * 2, sweep.y + sweep.height * 0.5);
  }
  await hoverPage.waitForTimeout(150);
  const pushes = await hoverPage.evaluate(() => window.__pushes);
  check(
    pushes === 0,
    `agents hover: 60 mousemoves push ZERO options on either card (${pushes})`,
  );
  const stillOff = await hoverPage.evaluate(() =>
    [...document.querySelectorAll(".chart")].map((h) => h.__pitwallChart.getOption().animation),
  );
  check(
    stillOff.length === 2 && stillOff.every((value) => value === false),
    `agents hover: and both cards still carry animation: false (${JSON.stringify(stillOff)})`,
  );

  await hoverCtx.close();
}


// ---------------------------------------------------------------------------
// The CONTINGENCIES card, filling the side column's empty region.
//
// The fixture's two rows are not interchangeable: one has a rationale long
// enough to exceed two lines at this client and a `detail` to expand, the other
// has neither. Without the long one the clamp assertion is about text that
// already fits; without the empty one, "an unexpandable row is not a tab stop"
// cannot fail.
// ---------------------------------------------------------------------------
// **PIT is trimmed for this block, and that is not a convenience.** The shared
// fixture gives PIT 40 body lines deliberately, to guarantee the scroll check
// an overflow - and 40 lines eat the whole side column, so the contingencies
// card correctly collapses and none of its own rendering can be asserted. The
// full-PIT view is kept for the phantom-box block below, where that overfull
// card is exactly what creates the small residual worth testing.
const CTY_VIEW = {
  ...structuredClone(VIEW),
  cards: {
    ...structuredClone(VIEW.cards),
    pit: { ...structuredClone(VIEW.cards.pit), lines: VIEW.cards.pit.lines.slice(0, 2) },
  },
};
{
  const ctyCtx = await browser.newContext({ viewport: CLIENT });
  const ctyPage = await ctyCtx.newPage();
  watchPage(ctyPage, failures, "contingencies");
  await ctyPage.addInitScript((view) => {
    window.pywebview = {
      api: {
        get_agents_view: async (since) => (since >= view.seq ? null : view),
        get_connection: async () => ({ label: "Connected", colour: "#10b981" }),
      },
    };
  }, CTY_VIEW);
  await ctyPage.goto(`http://127.0.0.1:${server.address().port}/agents.html`, {
    waitUntil: "domcontentloaded",
  });
  // `attached`, never `visible`: a collapsed card is an empty `aria-hidden`
  // section, and a build that wrongly collapses it must FAIL a named check
  // rather than hang this waiter for ten seconds and kill the run.
  await ctyPage.waitForSelector(".contingencies", { state: "attached", timeout: 10000 });
  await ctyPage.waitForTimeout(400);

  // --- it renders, and it is a real card at this client ---------------------
  // Presence FIRST, named, and everything after it is gated on the answer.
  // Playwright's `hover` and `innerText` WAIT, so a card that renders nothing
  // becomes a 30-second timeout that kills the run before the failure above is
  // ever printed - a crash reported as a mutation result rather than a finding.
  const hasRows = (await ctyPage.locator(".cty-row").count()) > 0;
  check(hasRows, "contingencies: the card renders its rows at all");
  const shape = await ctyPage.evaluate(() => {
    const card = document.querySelector(".contingencies");
    const style = getComputedStyle(card);
    return {
      collapsed: card.classList.contains("is-collapsed"),
      border: Math.round(parseFloat(style.borderTopWidth)),
      rows: document.querySelectorAll(".cty-row").length,
      count: document.querySelector(".cty-count")?.innerText ?? null,
      switchTo: [...document.querySelectorAll(".cty-switch")].map((n) => n.innerText.trim()),
    };
  });
  check(
    !shape.collapsed && shape.border === 1,
    `contingencies: a real bordered card at the wide client (${JSON.stringify(shape)})`,
  );
  check(
    shape.rows === CTY_VIEW.contingencies.rows.length,
    `contingencies: every branch on the wire is drawn (${shape.rows})`,
  );
  check(
    shape.count === String(CTY_VIEW.contingencies.rows.length),
    `contingencies: the header states the TOTAL, so a scrolled body still says how many (${shape.count})`,
  );
  // The label, not the enum. A row printing PIT_NOW would disagree with the
  // badge two cards up about the same vocabulary.
  check(
    shape.switchTo[0] === "PIT NOW" && !shape.switchTo[0].includes("_"),
    `contingencies: the action reads as the orchestrator's own label (${shape.switchTo[0]})`,
  );

  // --- the clamped copy is not the ONLY copy --------------------------------
  // **`scrollHeight` cannot see this.** `-webkit-line-clamp` sizes the box to the
  // clamped lines, so the scroll height equals the client height and a check on
  // their difference reads 0 whether the text overflows or not. The honest
  // measurement is the same text, same width, same font, WITHOUT the clamp.
  const clamp = await ctyPage.evaluate(() => {
    const el = document.querySelector(".cty-rationale");
    if (!el) return { clamp: "NONE", clipped: -1 };
    const style = getComputedStyle(el);
    const probe = el.cloneNode(true);
    probe.style.webkitLineClamp = "unset";
    probe.style.display = "block";
    probe.style.position = "absolute";
    probe.style.visibility = "hidden";
    probe.style.width = `${el.clientWidth}px`;
    el.parentElement.appendChild(probe);
    const full = probe.getBoundingClientRect().height;
    const shown = el.getBoundingClientRect().height;
    probe.remove();
    return { clamp: style.webkitLineClamp, clipped: Math.round(full - shown) };
  });
  check(clamp.clamp === "2", `contingencies: the rationale is clamped to two lines (${clamp.clamp})`);
  check(
    clamp.clipped > 0,
    `contingencies: and the fixture's text actually EXCEEDS two lines, or the clamp asserts nothing (${clamp.clipped} px hidden)`,
  );
  if (hasRows) await ctyPage.locator(".cty-row").first().hover();
  await ctyPage.waitForTimeout(250);
  const popup = await ctyPage.evaluate(() => {
    const tips = [...document.querySelectorAll(".agent-tooltip")];
    return { n: tips.length, text: (tips[0]?.innerText ?? "").replace(/\s+/g, " ") };
  });
  check(popup.n === 1, `contingencies: hovering a row opens exactly one popup (${popup.n})`);
  check(
    popup.text.includes("the position is gone until the second stint"),
    `contingencies: and it carries the WHOLE rationale, not the clamped half (${popup.text.slice(-60)})`,
  );

  // --- the risks are IN THE BODY, not behind a hover ------------------------
  // They started as the title's tooltip, which made them content nobody would
  // find. Asserted against the wire's own list, never against a count: a block
  // that rendered the right NUMBER of wrong lines would pass a count.
  const risks = await ctyPage.locator(".cty-risk").allInnerTexts();
  check(
    JSON.stringify(risks) === JSON.stringify(CTY_VIEW.contingencies.risks),
    `contingencies: every risk on the wire is a line in the body (${JSON.stringify(risks)})`,
  );
  check(
    await ctyPage.evaluate(() => {
      const block = document.querySelector(".cty-risks");
      return Boolean(block) && Boolean(block.closest(".cty-body"));
    }),
    "contingencies: and the block sits in the scrolling body beside the branches",
  );
  // The title stopped being a tooltip target when they moved, so it must have
  // stopped being a tab stop too - an empty stop is the defect the row check
  // below guards on the other side.
  const titleTab = hasRows
    ? await ctyPage.locator(".cty-title").getAttribute("tabindex")
    : null;
  check(
    titleTab === null,
    `contingencies: the title is not a tab stop now that it expands nothing (${titleTab})`,
  );

  // --- no empty tab stops ---------------------------------------------------
  const stops = await ctyPage.evaluate(() =>
    [...document.querySelectorAll(".cty-row")].map((r) => r.tabIndex),
  );
  check(
    stops[0] === 0 && stops[1] === -1,
    `contingencies: a row with nothing to expand is not a tab stop (${JSON.stringify(stops)})`,
  );

  // --- what is below the fold SAYS so --------------------------------------
  // A real lap returns more branches and risks than the reference client's 214
  // px can hold, so the body scrolls - that is the shipped answer rather than a
  // shortfall. What makes it acceptable is the affordance: this window hides
  // every scrollbar, so without the fade the hidden half is simply lost, which
  // is the defect #1077 fixed on the consoles one card over.
  const fold = await ctyPage.evaluate(() => {
    const body = document.querySelector(".cty-body");
    if (!body) return null;
    return {
      hidden: Math.round(body.scrollHeight - body.clientHeight),
      below: body.classList.contains("has-below"),
      above: body.classList.contains("has-above"),
    };
  });
  check(
    fold !== null && fold.hidden > 0,
    `contingencies: the fixture actually OVERFLOWS, or the affordance below asserts ` +
      `nothing (${JSON.stringify(fold)})`,
  );
  check(
    fold !== null && fold.below && !fold.above,
    `contingencies: and the bottom fade says there is more, with no top fade at ` +
      `the start of the list (${JSON.stringify(fold)})`,
  );
  if (fold && fold.hidden > 0) {
    await ctyPage.locator(".cty-body").evaluate((node) => node.scrollTo(0, node.scrollHeight));
    await ctyPage.waitForTimeout(250);
    const bottom = await ctyPage.evaluate(() => {
      const body = document.querySelector(".cty-body");
      return { below: body.classList.contains("has-below"), above: body.classList.contains("has-above") };
    });
    check(
      bottom.above && !bottom.below,
      `contingencies: scrolled to the end, the fade flips to the top (${JSON.stringify(bottom)})`,
    );
  }

  // --- ⭐ the fit cannot move what it measures ------------------------------
  // The card is `flex: 1 1 0`, so its contents are outside its own sizing
  // calculation and its height is the column's residual. Hiding everything
  // inside it must therefore change nothing. This is the property #1083 does
  // NOT have: the BESTS panel's decision changes the height it re-measures.
  const before = await ctyPage.evaluate(
    () => document.querySelector(".contingencies").getBoundingClientRect().height,
  );
  await ctyPage.addStyleTag({ content: ".cty-body, .cty-title { display: none !important }" });
  await ctyPage.waitForTimeout(250);
  const after = await ctyPage.evaluate(
    () => document.querySelector(".contingencies").getBoundingClientRect().height,
  );
  check(
    Math.abs(before - after) < 0.5,
    `contingencies: emptying the card does not change the height it measures itself by (${before} -> ${after})`,
  );
  await ctyCtx.close();
}

// --- the phantom box: the card exists iff there is room for it ---------------
//
// **Asserted as the RULE, not as a client size.** The first version of this
// expected the laptop client to collapse, and it did with the shared fixture's
// deliberately overfull 40-line PIT card - but with a SHORT pit the same laptop
// has 106 px and correctly shows the card. The thing being tested is the room,
// so the sweep varies both the client and the column's own weight, asserts the
// implication at each point, and then asserts that it actually reached BOTH
// outcomes. Without that last line a build that never rendered the card, or one
// that always did, would pass whichever arm it happened to satisfy.
//
// It deliberately does not read `MIN_ROOM`: a copy of that constant here is the
// twin this project produces most. The implication holds without knowing it.
{
  const shownAt = [];
  for (const [w, h, view, label] of [
    [1485, 913, CTY_VIEW, "1080p, short pit"],
    [1311, 641, CTY_VIEW, "laptop, short pit"],
    [1311, 641, VIEW, "laptop, the overfull pit the product really renders"],
  ]) {
    const ctx = await browser.newContext({ viewport: { width: w, height: h } });
    const page = await ctx.newPage();
    watchPage(page, failures, `contingencies ${label}`);
    await page.addInitScript((fixture) => {
      window.pywebview = {
        api: {
          get_agents_view: async (since) => (since >= fixture.seq ? null : fixture),
          get_connection: async () => ({ label: "Connected", colour: "#10b981" }),
        },
      };
    }, view);
    await page.goto(`http://127.0.0.1:${server.address().port}/agents.html`, {
      waitUntil: "domcontentloaded",
    });
    // `attached`, never the default `visible`: a collapsed card is deliberately
    // an empty `aria-hidden` section, and Playwright is right to call that
    // invisible. Waiting for visible would time out on the very behaviour this
    // block exists to assert.
    await page.waitForSelector(".contingencies", { state: "attached", timeout: 10000 });
    await page.waitForTimeout(500);
    const seen = await page.evaluate(() => {
      const card = document.querySelector(".contingencies");
      const style = getComputedStyle(card);
      return {
        room: Math.round(card.getBoundingClientRect().height),
        rows: document.querySelectorAll(".cty-row").length,
        border: Math.round(parseFloat(style.borderTopWidth)),
        padding: Math.round(parseFloat(style.paddingTop)),
      };
    });
    const drawn = seen.rows > 0;
    shownAt.push(drawn);
    check(
      drawn ? seen.border === 1 : seen.border === 0 && seen.padding === 0,
      `contingencies (${label}): a card with rows has its chrome, one without has NONE - ` +
        `not a heading over a cut-off row (${JSON.stringify(seen)})`,
    );
  }
  check(
    shownAt.some(Boolean) && shownAt.some((drawn) => !drawn),
    `contingencies: the sweep reached BOTH outcomes, so neither arm passed by never ` +
      `running (${JSON.stringify(shownAt)})`,
  );
}


await browser.close();
server.close();

if (failures.length) {
  console.error(`smoke FAILED (${failures.length}):`);
  for (const failure of failures) console.error(`  - ${failure}`);
  process.exit(1);
}
console.log(`smoke OK: ${checks} checks`);
