/**
 * Does the built AGENTS bundle still render the window?
 *
 * **The gap this closes was named by the sprint-3 exit gate**: every one
 * of its render-layer findings sailed through 35 green Python tests,
 * because those pin the view DICT and nothing loaded the bundle. Deleting
 * `<PaceChart/>` from `AgentsWindow.tsx`, or the whole `.agents-split`
 * rule from the stylesheet, left the suite entirely green.
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

const UI_DIR = resolve(dirname(fileURLToPath(import.meta.url)), "..");
// An alternate bundle directory, so a negative check can run against a
// deliberately broken copy without touching the real one.
const DIST = resolve(process.argv[2] ?? resolve(UI_DIR, "dist"));

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
    confidence_label: "Confidence: 71%",
    confidence_colour: "#10b981",
    pace: "Pace: PUSH",
    pace_colour: "#ef4444",
    risk: "Risk: AGGRESSIVE",
    risk_colour: "#ef4444",
    plan: "Pit: L24 · Next: HARD · UCUT: RUS",
    guardrail: "",
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
        tooltip: key === "radio" ? "<b>Radio</b><br>NOR: rear grip" : "",
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
const ctx = await browser.newContext({ viewport: { width: 1320, height: 900 } });
const page = await ctx.newPage();
page.on("pageerror", (error) => failures.push(`pageerror: ${error.message}`));
page.on("console", (message) => {
  if (message.type() === "error") failures.push(`console: ${message.text()}`);
});

await page.addInitScript((view) => {
  window.pywebview = {
    api: {
      get_agents_view: async (sinceSeq) => (sinceSeq >= view.seq ? null : view),
      get_tick: async () => null,
    },
  };
}, VIEW);

await page.goto(`http://127.0.0.1:${server.address().port}/agents.html`, {
  waitUntil: "domcontentloaded",
});
await page.waitForSelector(".agent-card", { timeout: 5000 });

check((await page.locator(".agent-card").count()) === 6, "six agent cards");
check((await page.locator(".agent-chart canvas").count()) === 2, "two chart canvases");
check((await page.locator(".reasoning-tab").count()) === 6, "six reasoning tabs");
check((await page.locator(".scenario-row").count()) === 4, "four scenario rows");

// ...and the winner's bar is actually wider. Counting rows passed over a
// board where every bar rendered full width.
const barWidths = await page.evaluate(() =>
  [...document.querySelectorAll(".scenario-bar-fill")].map((el) => Math.round(el.getBoundingClientRect().width)),
);
check(barWidths[1] > barWidths[0] && barWidths[0] > 0, `the bars carry their fill (${barWidths})`);
check(await page.locator(".orchestrator").isVisible(), "the orchestrator card");

// Qt pins the left panel at 540 px and sends every extra pixel right
// (`setStretchFactor(0, 0)`). A proportional split grew it to 807 at 1920.
const columns = await page.evaluate(
  () => getComputedStyle(document.querySelector(".agents-split")).gridTemplateColumns,
);
check(columns.startsWith("540px"), `left column is 540px (got ${columns})`);

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
const live = await browser.newContext({ viewport: { width: 1320, height: 900 } });
const livePage = await live.newPage();
await livePage.addInitScript((view) => {
  let seq = 0;
  window.pywebview = {
    api: {
      // A new sequence every poll, with the SAME status text, which is
      // what a real producer does for the ~85 s a lap lasts.
      get_agents_view: async () => ({ ...view, seq: ++seq }),
      get_tick: async () => null,
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
await live.close();

await browser.close();
server.close();

if (failures.length) {
  console.error(`smoke FAILED (${failures.length}):`);
  for (const failure of failures) console.error(`  - ${failure}`);
  process.exit(1);
}
console.log(`smoke OK: ${checks} checks`);
