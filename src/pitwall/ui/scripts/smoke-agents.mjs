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
        tooltip:
          key === "radio"
            ? {
                sections: [
                  {
                    title: "Radio",
                    rows: [
                      {
                        lead: "NOR PROBLEM",
                        text: "Rear grip is going away, especially through the last sector, and the balance moves every lap.",
                      },
                    ],
                  },
                ],
                footer: null,
              }
            : null,
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
      get_connection: async () => "Connected",
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
check((await page.locator(".reasoning-tab").count()) === 6, "six reasoning tabs");
check((await page.locator(".scenario-row").count()) === 4, "four scenario rows");

// ...and the winner's bar is actually wider. Counting rows passed over a
// board where every bar rendered full width.
const barWidths = await page.evaluate(() =>
  [...document.querySelectorAll(".scenario-bar-fill")].map((el) => Math.round(el.getBoundingClientRect().width)),
);
check(barWidths[1] > barWidths[0] && barWidths[0] > 0, `the bars carry their fill (${barWidths})`);
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
check((await page.locator(".agent-tooltip").count()) === 1, "one tooltip, on RADIO only");

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
  [...document.querySelectorAll(".agent-card-body, .reasoning-body")]
    .map((el, i) => ({ i, over: el.scrollHeight - el.clientHeight }))
    .filter((e) => e.over > 0)
    .map((e) => e.i),
);
check(overflowing.length > 0, `something overflows, or this checks nothing`);

const wheeled = [];
for (const index of overflowing) {
  const box = await page
    .locator(".agent-card-body, .reasoning-body")
    .nth(index)
    .boundingBox();
  await page.mouse.move(box.x + box.width / 2, box.y + box.height / 2);
  await page.mouse.wheel(0, 200);
  await page.waitForTimeout(120);
  wheeled.push(
    await page.evaluate(
      (i) => document.querySelectorAll(".agent-card-body, .reasoning-body")[i].scrollTop,
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
      get_connection: async () => "Connected",
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
  watchPage(idlePage, failures, "idle/${connection}");
  await idlePage.addInitScript((state) => {
    window.pywebview = {
      api: {
        get_agents_view: async () => null,
        get_tick: async () => null,
        get_connection: async () => state,
      },
    };
  }, connection);
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
    [VIEW, "Connecting..."],
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

await browser.close();
server.close();

if (failures.length) {
  console.error(`smoke FAILED (${failures.length}):`);
  for (const failure of failures) console.error(`  - ${failure}`);
  process.exit(1);
}
console.log(`smoke OK: ${checks} checks`);
