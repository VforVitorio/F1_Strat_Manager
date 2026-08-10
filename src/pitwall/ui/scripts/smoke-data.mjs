/**
 * Does the built DATA bundle still render band 4, and render it RIGHT?
 *
 * The sibling of `smoke-agents.mjs`, and it exists for the reason that one
 * had to be written after the fact: the sprint-3 exit gate found render-layer
 * defects that sailed through 35 green Python tests, because those pin what
 * the host SENDS and nothing loaded the bundle.
 *
 * So every check below is an EFFECT in a real engine:
 *
 * - the four axes are LOCKED by reading each chart's COMPUTED extent, not
 *   the option object this file already knows was passed;
 * - the delta series is the one the interpolation produced, counted against
 *   hand-worked numbers, including the samples it must DROP because the
 *   rival never reached that far;
 * - the shared cursor is looked for in the canvas PIXELS, at the column
 *   ECharts places `cursorX` in - and looked for again where it must not be;
 * - a rewind actually empties the buffer;
 * - the placeholder actually replaces the plot in single-driver mode.
 *
 * It is NOT a visual regression test. Pixels beyond the one cursor column
 * are `shot-data.mjs`'s job and a human's.
 *
 *   npm run build && node scripts/smoke-data.mjs
 */
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { chromium } from "@playwright/test";
import { serveDist } from "./serve-dist.mjs";

const UI_DIR = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const DIST = resolve(process.argv[2] ?? resolve(UI_DIR, "dist"));

const CIRCUIT_M = 5220;



/**
 * A span the delta chart can be checked by hand.
 *
 * The rival covers 100-300 m and the main covers 100-500, so three of the
 * five main samples have a rival time to interpolate against and two do not.
 * Times are chosen so the rival is a flat +2.0 s behind: at x=100 the main
 * is at t=10.0 and the rival at t=12.0, and both advance 1 s per 100 m.
 */
const MAIN_SPAN = [100, 200, 300, 400, 500].map((dist, i) => ({
  lap: 24,
  t: 10 + i,
  dist,
  speed: 200 + i * 10,
  throttle: 50 + i,
  brake: i === 0 ? 100 : 0,
  gear: 6,
  drs: 8,
}));
const RIVAL_SPAN = [100, 200, 300].map((dist, i) => ({
  lap: 24,
  t: 12 + i,
  dist,
  speed: 190 + i * 10,
  throttle: 40 + i,
  brake: 0,
  gear: 6,
  drs: 8,
}));
/** The main driver's own position, which is where the cursor must land. */
const CURSOR_DIST = 500;

function driver(overrides = {}) {
  return {
    lap: 24,
    dist: 120000,
    rel_dist: CURSOR_DIST / CIRCUIT_M,
    speed: 240,
    compound: 2,
    tyre_life: 16,
    active: true,
    has_position: true,
    laps_completed: 23,
    progress: 23.1,
    has_finished: false,
    ...overrides,
  };
}

function tick(seq, { rival = "PIA", main = MAIN_SPAN, rivalSpan = RIVAL_SPAN, rewound = false, mainDriver = {} } = {}) {
  return {
    schema_version: 2,
    seq,
    arcade: {
      gp_name: "Melbourne",
      location: "Melbourne",
      year: 2025,
      lap: 24,
      t: 1400,
      global_t_min: 0,
      total_laps: 57,
      circuit_length_m: CIRCUIT_M,
      driver_main: "NOR",
      driver_rival: rival,
      drivers: { NOR: driver(mainDriver), PIA: driver() },
      race_order: ["NOR", "PIA"],
      driver_colors: { NOR: [255, 128, 0], PIA: [255, 128, 0] },
      track_status: "1",
      telemetry: { main, rival: rivalSpan, rewound, dropped: 0 },
    },
    playback: { speed: 1, paused: false, frame_index: 1000 + seq, total_frames: 154173 },
    strategy: {},
  };
}

const failures = [];
let checks = 0;
const check = (ok, what) => {
  checks += 1;
  if (!ok) failures.push(what);
};

/** Read a chart's computed axis extents through the handle `useEChart` sets. */
const EXTENTS = (index) => `
  (() => {
    const el = document.querySelectorAll(".trace-plot")[${index}];
    const chart = el && el.__pitwallChart;
    if (!chart) return null;
    const axis = (type) => chart.getModel().getComponent(type, 0).axis.scale.getExtent();
    return { x: axis("xAxis"), y: axis("yAxis") };
  })()
`;

const server = await serveDist(DIST);
const browser = await chromium.launch();
const url = `http://127.0.0.1:${server.address().port}/data.html`;

// --- Scenario A: two drivers, a real span -----------------------------------

const ctx = await browser.newContext({ viewport: { width: 1500, height: 950 } });
const page = await ctx.newPage();
page.on("pageerror", (error) => failures.push(`pageerror: ${error.message}`));
page.on("console", (message) => {
  if (message.type() === "error") failures.push(`console: ${message.text()}`);
});

// A MONOTONE script, never "return whatever the caller has not seen". The
// first version of this stub returned the first tick whose seq differed, so
// once tick 2 had rendered it handed back tick 1 again and the poller
// oscillated between them - re-ingesting the span the rewind had just
// evicted, ten times a second. The rewind check failed, correctly, against a
// producer no real arcade could be.
await page.addInitScript((payload) => {
  window.__ticks = [payload];
  window.__cursor = 0;
  window.pywebview = {
    api: {
      get_tick: async (sinceSeq) => {
        if (window.__ticks[window.__cursor].seq === sinceSeq) {
          if (window.__cursor + 1 >= window.__ticks.length) return null;
          window.__cursor += 1;
        }
        return window.__ticks[window.__cursor];
      },
    },
  };
}, tick(1));

await page.goto(url, { waitUntil: "domcontentloaded" });
await page.waitForSelector(".trace-plot canvas", { timeout: 5000 });
await page.waitForTimeout(400);

check((await page.locator(".trace-cell").count()) === 4, "four chart cells");
check((await page.locator(".trace-plot canvas").count()) === 4, "four live canvases");
check(
  (await page.locator(".traces-lap").innerText()).trim() === "LAP 24",
  "the header carries the lap and nothing else",
);
check((await page.locator(".driver-chip").count()) === 2, "main and rival chips");
check(
  (await page.locator(".trace-tier").first().innerText()).trim() === "BROADCAST",
  "the rival legend is labelled broadcast tier",
);

// Qt stretches both columns and both rows equally (`setColumnStretch(_, 1)`).
// A `1fr 1fr` that silently became `auto auto` would put the dead space
// inside the cells and the four plots would divide what the title rows left.
const grid = await page.evaluate(() => {
  const style = getComputedStyle(document.querySelector(".traces-grid"));
  return { cols: style.gridTemplateColumns, rows: style.gridTemplateRows };
});
const equalPair = (value) => {
  const [a, b] = value.split(" ").map(parseFloat);
  return a > 0 && Math.abs(a - b) < 1;
};
check(equalPair(grid.cols), `two equal columns (${grid.cols})`);
check(equalPair(grid.rows), `two equal rows (${grid.rows})`);

// The four locked ranges, read as the extent the axis COMPUTED. This is the
// claim the whole port rests on: only the lines move between updates.
// Order is Qt's: delta, speed, brake, throttle.
const expected = [
  ["delta", [-3, 3]],
  ["speed", [0, 360]],
  ["brake", [-5, 105]],
  ["throttle", [-5, 105]],
];
for (const [index, [name, range]] of expected.entries()) {
  const extents = await page.evaluate(EXTENTS(index));
  check(
    extents !== null &&
      extents.y[0] === range[0] &&
      extents.y[1] === range[1] &&
      extents.x[0] === 0 &&
      extents.x[1] === CIRCUIT_M,
    `${name} axes locked to x[0,${CIRCUIT_M}] y[${range}] (got ${JSON.stringify(extents)})`,
  );
}

// The delta the interpolation produced. Three points, not five: the rival's
// samples stop at 300 m and `lerpSorted` returns null past the end rather
// than extrapolating a flat tail that would look like real data.
const delta = await page.evaluate(() => {
  const el = document.querySelectorAll(".trace-plot")[0];
  return el.__pitwallChart.getOption().series[1].data;
});
check(
  delta.length === 3 && delta.every(([, value]) => Math.abs(value - 2) < 1e-9),
  `the delta is +2.0 s over the rival's three samples only (${JSON.stringify(delta)})`,
);

// The speed trace carries the whole main span.
const speedPoints = await page.evaluate(() => {
  const el = document.querySelectorAll(".trace-plot")[1];
  return el.__pitwallChart.getOption().series[0].data.length;
});
check(speedPoints === 5, `the speed trace holds all five samples (${speedPoints})`);

// The cursor, in PIXELS. It must be at the column ECharts maps 500 m to, and
// must not be at a column the car has not reached - which is also a column
// no trace passes through, so grey there could only be the cursor.
//
// "Grey and bright", not a colour distance: a 1 px dashed line lands
// between two device columns and each gets it at part alpha, so the exact
// hex appears nowhere. Nothing else in the plot area is both neutral and
// bright - the traces are saturated, the panel and the grid are dark - so
// the discriminator is the one that survives antialiasing. The scan stops
// above the x axis for the same reason it has to: the tick labels below it
// are TEXT_SECONDARY grey, and a full-height column would find "3000" and
// call it a cursor.
const cursor = await page.evaluate(
  ({ at, away, gridBottom }) => {
    const el = document.querySelectorAll(".trace-plot")[1];
    const chart = el.__pitwallChart;
    const canvas = el.querySelector("canvas");
    const context = canvas.getContext("2d");
    const ratio = canvas.width / canvas.getBoundingClientRect().width;
    const plotHeight = Math.floor(canvas.height - gridBottom * ratio);
    // A five-pixel window, not one column. `convertToPixel` answers in the
    // chart's coordinate space and the canvas rounds; measured, the line
    // lands one to two device pixels off the computed column, which is a
    // rounding fact and not a placement claim. Five pixels out of a
    // ~700-pixel plot still says "here and nowhere else".
    const column = (metres) => {
      const x = Math.round(chart.convertToPixel({ xAxisIndex: 0 }, metres) * ratio);
      const data = context.getImageData(Math.max(0, x - 2), 0, 5, plotHeight).data;
      let hits = 0;
      for (let i = 0; i < data.length; i += 4) {
        const [r, g, b] = [data[i], data[i + 1], data[i + 2]];
        const neutral = Math.max(r, g, b) - Math.min(r, g, b) < 25;
        const bright = (r + g + b) / 3 > 80;
        if (neutral && bright && data[i + 3] > 200) hits += 1;
      }
      return hits;
    };
    return { at: column(at), away: column(away) };
  },
  { at: CURSOR_DIST, away: 3000, gridBottom: 36 },
);
check(cursor.at > 10, `the cursor is drawn at ${CURSOR_DIST} m (${cursor.at} px)`);
check(cursor.away === 0, `and nowhere else (${cursor.away} px at 3000 m)`);

// A rewind must EMPTY the buffer. The producer sends the flag with an empty
// span, and a distance-keyed store holds samples for track the car has not
// re-driven - nothing else would ever evict them.
await page.evaluate((payload) => window.__ticks.push(payload), tick(2, { main: [], rivalSpan: [], rewound: true }));
await page.waitForTimeout(400);
const afterRewind = await page.evaluate(() => {
  const el = document.querySelectorAll(".trace-plot")[1];
  return el.__pitwallChart.getOption().series[0].data.length;
});
check(afterRewind === 0, `a rewind empties the trace (${afterRewind} points left)`);

// ...and the rewind must NOT look like single-driver mode. This is the twin
// the Qt panel actually shipped: visibility keyed on the buffer rather than
// on the session's rival, so the three rival traces and their legends
// vanished for the whole of a rewind hold and every lap change. The buffer
// is empty right now, which is exactly the moment that bug is visible.
check(
  (await page.locator(".trace-placeholder").count()) === 0,
  "an empty buffer is not single-driver mode",
);
check(
  (await page.locator(".trace-tier").count()) === 4,
  "the rival legend survives an empty buffer on all four charts",
);

// The status bar, which is Qt's `showMessage(f"lap {lap} · live", 1500)`.
// It has to be visible while the producer talks and blank 1.5 s after it
// stops - the AGENTS window shipped BOTH halves of that wrong before #871
// and #874, once by never clearing and once by clearing mid-lap.
check(
  (await page.locator(".status-bar").innerText()).includes("lap 24 · live"),
  "the status bar names the lap while streaming",
);
await page.waitForTimeout(1800);
check(
  (await page.locator(".status-bar").innerText()).trim() === "",
  "and clears itself once the producer goes quiet",
);

await ctx.close();

// --- Scenario B: single driver, and a car the telemetry never placed --------

const solo = await browser.newContext({ viewport: { width: 1500, height: 950 } });
const soloPage = await solo.newPage();
soloPage.on("pageerror", (error) => failures.push(`pageerror(solo): ${error.message}`));

await soloPage.addInitScript((payload) => {
  window.pywebview = { api: { get_tick: async (sinceSeq) => (sinceSeq === payload.seq ? null : payload) } };
}, tick(1, { rival: null, rivalSpan: [], mainDriver: { has_position: false, rel_dist: null } }));

await soloPage.goto(url, { waitUntil: "domcontentloaded" });
await soloPage.waitForSelector(".trace-cell", { timeout: 5000 });
await soloPage.waitForTimeout(400);

check(
  (await soloPage.locator(".trace-placeholder").innerText()).trim() === "single-driver mode",
  "the delta chart collapses to its placeholder",
);
check(
  (await soloPage.locator(".trace-plot canvas").count()) === 3,
  "three canvases, because the placeholder REPLACES the delta plot",
);
check((await soloPage.locator(".driver-chip").count()) === 1, "no rival chip");
check((await soloPage.locator(".trace-tier").count()) === 0, "and no broadcast-tier label");
// #856: the note names the blind car, and it must fire for the MAIN driver
// too - the first version of it read the rival alone.
check(
  (await soloPage.locator(".traces-lap").innerText()).includes("NO POSITION DATA (NOR)"),
  "the header says the car was never placed",
);
// A null `rel_dist` is unknown, not zero. Drawing the cursor at 0 m would put
// the blind car exactly on the start line, a place a real car can be.
const blindCursor = await soloPage.evaluate(() => {
  const el = document.querySelectorAll(".trace-plot")[0];
  return (el.__pitwallChart.getOption().series[0].markLine?.data ?? []).some((m) => "xAxis" in m);
});
check(!blindCursor, "and draws no cursor for a car with no position");

await solo.close();
await browser.close();
server.close();

if (failures.length) {
  console.error(`smoke-data FAILED (${failures.length}):`);
  for (const failure of failures) console.error(`  - ${failure}`);
  process.exit(1);
}
console.log(`smoke-data OK: ${checks} checks`);
