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
import { staysStill } from "./settle.mjs";

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

function tick(
  seq,
  {
    rival = "PIA",
    main = MAIN_SPAN,
    rivalSpan = RIVAL_SPAN,
    rewound = false,
    dropped = 0,
    mainDriver = {},
    drivers = null,
    order = null,
    colors = null,
  } = {},
) {
  const field = drivers ?? { NOR: driver(mainDriver), PIA: driver() };
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
      drivers: field,
      race_order: order ?? Object.keys(field),
      driver_colors:
        colors ?? Object.fromEntries(Object.keys(field).map((code) => [code, [255, 128, 0]])),
      track_status: "1",
      // Decoded by the producer, never by the renderer. The pair is null
      // together when the loader has no entry for the lap, which band 1 must
      // render as unknown rather than as a green track.
      track_status_label: "GREEN",
      track_status_color: [16, 185, 129],
      telemetry: { main, rival: rivalSpan, rewound, dropped },
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

/**
 * Read one LANE's computed axis extents off the stack's single chart.
 *
 * The index is a GRID index now, not a canvas index: the 2x2's four charts became
 * one instance with six grids, so `getComponent(type, index)` asks the question the
 * four separate `__pitwallChart` handles used to answer.
 */
const EXTENTS = (index) => `
  (() => {
    const el = document.querySelector(".trace-stack-plot");
    const chart = el && el.__pitwallChart;
    if (!chart) return null;
    const axis = (type) => chart.getModel().getComponent(type, ${index}).axis.scale.getExtent();
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
      get_bulk: async () => null,
      get_live_lap: async () => null,
      get_connection: async () => "Connected",
    },
  };
}, tick(1));

await page.goto(url, { waitUntil: "domcontentloaded" });
await page.waitForSelector(".trace-stack-plot canvas", { timeout: 5000 });
await page.waitForTimeout(400);

// SIX lanes on ONE canvas. The 2x2 is retired: the agreed layout drawing assigned
// the stacked form to this sprint and calls the 2x2 "the wrong shape".
check(
  (await page.locator(".trace-lane-label").count()) === 6,
  "six lanes, each with its own label row",
);
check(
  (await page.locator(".trace-stack-plot canvas").count()) === 1,
  "on ONE canvas - four ECharts instances became one",
);
check((await page.locator(".trace-cursor").count()) === 1, "and ONE cursor across all six");
check(
  (await page.locator(".traces-lap").innerText()).trim() === "LAP 24",
  "the header carries the lap and nothing else",
);
check((await page.locator(".driver-chip").count()) === 2, "main and rival chips");
check(
  (await page.locator(".trace-tier").first().innerText()).trim() === "BROADCAST",
  "the rival chip is labelled broadcast tier",
);

// --- Band 1: the status strip -----------------------------------------------

check(
  (await page.locator(".strip-lap").innerText()).replace(/\s+/g, "") === "L24/57",
  "band 1 carries the lap out of the total",
);
check(
  (await page.locator(".strip-chip").first().innerText()).trim() === "GREEN",
  "the track status is the label the PRODUCER decoded, not one re-derived here",
);
// The clock is `t + global_t_min` - the FastF1 SessionTime the parquets are
// keyed on. `t` alone is `frame_index * DT`, which means nothing off this
// process. The stub sets t=1400 and global_t_min=0.
check(
  (await page.locator(".strip-field-value").first().innerText()).trim() === "0:23:20",
  "the session clock is SessionTime and not replay seconds",
);
check(
  (await page.locator(".strip-chip.is-connected").innerText()).trim() === "Connected",
  "the connection comes from the host's socket, not from tick freshness",
);
check(
  (await page.locator(".strip-chip.is-provisional").count()) === 0,
  "no PROVISIONAL chip once the running field has completed a lap",
);

// **The lanes must SUM to the box, and the last one carries the rounding.** The
// heights are derived from weights rather than tabulated, precisely so the same six
// lanes land on 666 px at one client and 430 at another - so the assertion is the
// arithmetic, not a pixel table.
const lanes = await page.evaluate(() => {
  const el = document.querySelector(".trace-stack-plot");
  const chart = el && el.__pitwallChart;
  if (!chart) return null;
  const stack = document.querySelector(".trace-stack");
  const grids = chart.getOption().grid;
  return {
    box: stack.clientHeight,
    tops: grids.map((g) => g.top),
    heights: grids.map((g) => g.height),
    lefts: grids.map((g) => g.left),
    rights: grids.map((g) => g.right),
    cursorHeight: parseFloat(getComputedStyle(document.querySelector(".trace-cursor")).height),
  };
});
check(
  lanes !== null &&
    lanes.heights.length === 6 &&
    lanes.heights.every((h) => h > 0) &&
    lanes.tops.every((top, i) => i === 0 || top > lanes.tops[i - 1]),
  `six lanes in order, none collapsed (${JSON.stringify(lanes && lanes.heights)})`,
);
// All six share the plot margins, or the cursor is not one straight line.
check(
  lanes !== null && new Set(lanes.lefts).size === 1 && new Set(lanes.rights).size === 1,
  `every lane shares the plot margins (${JSON.stringify(lanes && [lanes.lefts[0], lanes.rights[0]])})`,
);
// The cursor spans the whole stack, not one lane: a reader's cut must be unbroken.
check(
  lanes !== null && lanes.cursorHeight > lanes.box * 0.85,
  `the cursor spans the lanes rather than one of them (${lanes && lanes.cursorHeight} of ${lanes && lanes.box})`,
);

// The SIX locked ranges, read as the extent each axis COMPUTED. This is the claim
// the whole panel rests on: only the lines move between updates. Order is the
// stack's, speed first - the convention every client the spec surveyed uses.
// BRAKE and THROTTLE hold the same pair from two SEPARATE constants; merging rules
// that agree by coincidence is a defect class this repo has already paid for.
const expected = [
  ["speed", [0, 360]],
  ["delta", [-3, 3]],
  ["throttle", [-5, 105]],
  ["brake", [-5, 105]],
  ["gear", [0, 9]],
  ["drs", [-0.2, 1.2]],
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
// Looked up by NAME, not by index. These two used to read `series[1]` and
// `series[0]`, and the declaration order is exactly what the z-order fix had to
// change - an index-keyed probe turns that into two silently wrong assertions
// about the wrong line.
/**
 * One lane's series off the stack's single chart.
 *
 * The 2x2 named its two series `main` and `rival` inside each of four charts; the
 * stack has twelve on one, named `<lane>-<car>` - so the lookup takes the lane by
 * name rather than a canvas index, which also stops the assertions below depending
 * on the lane ORDER (that is asserted once, in its own check).
 */
const laneSeries = (lane, car) =>
  page.evaluate(
    ([which, side]) => {
      const el = document.querySelector(".trace-stack-plot");
      const found = el.__pitwallChart
        .getOption()
        .series.find((s) => s.name === `${which}-${side}`);
      return found ?? null;
    },
    [lane, car],
  );

const delta = (await laneSeries("delta", "rival")).data;
check(
  delta.length === 3 && delta.every(([, value]) => Math.abs(value - 2) < 1e-9),
  `the delta is +2.0 s over the rival's three samples only (${JSON.stringify(delta)})`,
);

// The speed trace carries the whole main span.
const speedPoints = (await laneSeries("speed", "main")).data.length;
check(speedPoints === 5, `the speed trace holds all five samples (${speedPoints})`);

// **The own car paints ON TOP, on every chart.** ECharts paints in declaration
// order, and the rival's coarse broadcast dashes used to be last: wherever the
// two cars run comparable numbers - which is when the comparison matters - the
// solid pit-wall-grade line was underneath. The race trace one tab away builds
// the opposite rule deliberately, so this is the twin that had it inverted.
const paintOrder = await page.evaluate(() => {
  const el = document.querySelector(".trace-stack-plot");
  const names = el.__pitwallChart.getOption().series.map((s) => s.name);
  // Pairs, in declaration order: every lane must read rival-then-own.
  const pairs = [];
  for (let i = 0; i < names.length; i += 2) pairs.push(`${names[i]}>${names[i + 1]}`);
  return pairs;
});
check(
  paintOrder.length === 6 &&
    paintOrder.every((pair) => {
      const [first, second] = pair.split(">");
      return first.endsWith("-rival") && second.endsWith("-main");
    }),
  `the own car is declared last so it paints over the rival on all six lanes (${paintOrder.join(" | ")})`,
);
// And the lane ORDER itself, asserted once rather than assumed by every lookup.
check(
  paintOrder.map((pair) => pair.split("-")[0]).join(",") ===
    "speed,delta,throttle,brake,gear,drs",
  `speed first, DRS last (${paintOrder.map((pair) => pair.split("-")[0]).join(",")})`,
);

// **The cursor, in pixels, against what ECharts says the same distance is.**
//
// It used to be a per-chart `markLine` and this check scanned canvas columns for a
// grey pixel run. The stack draws ONE div across all six lanes instead - a markLine
// per grid would be six fragments with five gaps, and the panel exists to be read
// with an unbroken vertical cut - so the assertion moved with the mechanism: the
// div's left edge must be where `convertToPixel` puts the car's distance, and the
// div must be tall enough to cross every lane.
//
// This is still the EFFECT and not the mechanism: it asks "is the line at the car's
// position on the axis", which is the claim, and it would fail on an off-by-`left`
// transform, a stale `xMax`, or a cursor scoped to one lane.
const cursor = await page.evaluate(
  ({ at }) => {
    const el = document.querySelector(".trace-stack-plot");
    const chart = el.__pitwallChart;
    const div = document.querySelector(".trace-cursor");
    if (!div) return null;
    const stack = document.querySelector(".trace-stack").getBoundingClientRect();
    const expected = chart.convertToPixel({ xAxisIndex: 0 }, at);
    const actual = div.getBoundingClientRect().left - stack.left;
    const grids = chart.getOption().grid;
    const lastGrid = grids[grids.length - 1];
    return {
      expected: Math.round(expected),
      actual: Math.round(actual),
      height: Math.round(div.getBoundingClientRect().height),
      lanesBottom: lastGrid.top + lastGrid.height,
      firstLaneTop: grids[0].top,
    };
  },
  { at: CURSOR_DIST },
);
check(
  cursor !== null && Math.abs(cursor.expected - cursor.actual) <= 2,
  `the cursor sits where the axis maps ${CURSOR_DIST} m (expected ${cursor && cursor.expected}, got ${cursor && cursor.actual})`,
);
check(
  cursor !== null && cursor.height >= cursor.lanesBottom - cursor.firstLaneTop,
  `and it crosses every lane rather than one (${cursor && cursor.height} px over ${cursor && cursor.lanesBottom - cursor.firstLaneTop})`,
);

// The delta chart's BASELINE has to cross the whole plot, because every value
// on that chart is read against it and Qt draws it with `pg.InfiniteLine`.
// Measured on a real payload, the `{yAxis: 0}` shorthand stopped at 1679 m of
// a 5220 m axis - and 36 green checks, four of which were about this very
// chart, all passed over it. Only a pixel scan along the zero row sees it.
const baseline = await page.evaluate(() => {
  const el = document.querySelector(".trace-stack-plot");
  const chart = el.__pitwallChart;
  const canvas = el.querySelector("canvas");
  const context = canvas.getContext("2d");
  const ratio = canvas.width / canvas.getBoundingClientRect().width;
  // **Lane 1, not lane 0.** The delta is the SECOND lane of the stack now (speed
  // first, per the convention), so asking axis 0 where value 0 sits answers about
  // the speed lane - a row where this line cannot be, which is how this check first
  // reported `null` after the stack landed.
  const row = context.getImageData(
    0,
    Math.round(chart.convertToPixel({ yAxisIndex: 1 }, 0) * ratio),
    canvas.width,
    1,
  ).data;
  const blue = [];
  for (let x = 0; x < canvas.width; x += 1) {
    const i = x * 4;
    // palette.INFO #3b82f6 = (59, 130, 246), with room for antialiasing.
    if (row[i] < 110 && row[i + 1] > 90 && row[i + 2] > 180) blue.push(x);
  }
  if (!blue.length) return null;
  const metres = (px) => chart.convertFromPixel({ xAxisIndex: 1 }, px / ratio);
  return { from: metres(blue[0]), to: metres(blue[blue.length - 1]) };
});
check(
  baseline !== null && baseline.from < 60 && baseline.to > CIRCUIT_M - 60,
  `the zero reference line crosses the whole plot (${JSON.stringify(baseline)})`,
);

// A rewind must EMPTY the buffer. The producer sends the flag with an empty
// span, and a distance-keyed store holds samples for track the car has not
// re-driven - nothing else would ever evict them.
await page.evaluate((payload) => window.__ticks.push(payload), tick(2, { main: [], rivalSpan: [], rewound: true }));
await page.waitForTimeout(400);
const afterRewind = (await laneSeries("speed", "main")).data.length;
check(afterRewind === 0, `a rewind empties the trace (${afterRewind} points left)`);

// ...and the rewind must NOT look like single-driver mode. This is the twin
// the Qt panel actually shipped: visibility keyed on the buffer rather than
// on the session's rival, so the three rival traces and their legends
// vanished for the whole of a rewind hold and every lap change. The buffer
// is empty right now, which is exactly the moment that bug is visible.
check(
  (await page.locator(".trace-lane-caption, .trace-placeholder").count()) === 0,
  "an empty buffer is not single-driver mode",
);
// One tag on the card header, not one per cell: the per-cell legends were
// dropped when band 4 moved into the narrower right column, where the title
// row wrapped onto three lines and ate the plot.
check(
  (await page.locator(".trace-tier").count()) === 1,
  "the broadcast-tier tag survives an empty buffer, on the header chip",
);

// The eviction ORDER, which the accumulator's own docstring calls
// load-bearing and which the rewind above structurally CANNOT see: the
// producer sends `rewound` with an EMPTY span, so clearing before and
// clearing after are the same outcome. A forward jump is the case that
// separates them - `dropped` rides along with a valid post-jump span, and
// clearing afterwards throws it away. Measured on the Qt panel before it was
// fixed there: up to 250 samples, ten seconds of trace the payload had
// already delivered, leaving four blank charts to refill.
const JUMPED = [3100, 3200, 3300].map((dist, i) => ({
  lap: 24,
  t: 40 + i,
  dist,
  speed: 300 + i,
  throttle: 100,
  brake: 0,
  gear: 8,
  drs: 12,
}));
await page.evaluate(
  (payload) => window.__ticks.push(payload),
  tick(3, { main: JUMPED, rivalSpan: [], dropped: 60 }),
);
await page.waitForTimeout(400);
const afterJump = (await laneSeries("speed", "main")).data.map(([x]) => x);
check(
  afterJump.length === 3 && afterJump[0] === 3100,
  `a forward jump keeps the span it arrived with (${JSON.stringify(afterJump)})`,
);

// The delta plot must SURVIVE going away and coming back. Its container is
// unmounted whenever the session drops to single-driver mode and remounted
// when a rival returns, and `useEChart` used to key its init effect on `[]` -
// which runs once per component, not once per container. After one round trip
// the instance pointed at a detached node and the chart was dead for the rest
// of the session, silently. Nothing in the 37 checks before this one could
// see it: they all measured a chart that had never been unmounted.
await page.evaluate((payload) => window.__ticks.push(payload), tick(4, { rival: null, rivalSpan: [] }));
await page.waitForTimeout(400);
check(
  (await page.locator(".trace-lane-caption, .trace-placeholder").count()) === 1,
  "losing the rival shows the placeholder",
);
await page.evaluate((payload) => window.__ticks.push(payload), tick(5));
await page.waitForTimeout(500);
// The delta lane is index 1 in the stack: speed owns 0.
const revived = await page.evaluate(EXTENTS(1));
check(
  (await page.locator(".trace-stack-plot canvas").count()) === 1,
  "the delta plot comes back as a live canvas",
);
check(
  revived !== null && revived.y[0] === -3 && revived.y[1] === 3,
  `and it is a working chart, not a detached one (${JSON.stringify(revived)})`,
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
  window.pywebview = {
    api: {
      get_tick: async (sinceSeq) => (sinceSeq === payload.seq ? null : payload),
      get_bulk: async () => null,
      get_live_lap: async () => null,
      get_connection: async () => "Connected",
    },
  };
}, tick(1, { rival: null, rivalSpan: [], mainDriver: { has_position: false, rel_dist: null } }));

await soloPage.goto(url, { waitUntil: "domcontentloaded" });
await soloPage.waitForSelector(".trace-lane-label", { timeout: 5000 });
await soloPage.waitForTimeout(400);

check(
  (await soloPage.locator(".trace-lane-caption, .trace-placeholder").innerText()).trim() === "single-driver mode",
  "the delta chart collapses to its placeholder",
);
check(
  (await soloPage.locator(".trace-stack-plot canvas").count()) === 1,
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
// Across EVERY series, not the one that happens to carry the marks: they hang
// off whichever series is declared first, and that changed once already.
const blindCursor = await soloPage.evaluate(() => {
  const el = document.querySelector(".trace-stack-plot");
  return el.__pitwallChart
    .getOption()
    .series.flatMap((s) => s.markLine?.data ?? [])
    .some((m) => "xAxis" in m);
});
check(!blindCursor, "and draws no cursor for a car with no position");

await solo.close();

// --- Scenario C: the ring's three states, and the one that reads backwards --

// The trap this exists for, measured on the real session: on the final frame
// `!active` alone reads 19 of the 20 cars as retired, the WINNER included,
// because a car that took the flag stops broadcasting exactly like a car that
// crashed. VER below is that car.
const FIELD = {
  NOR: driver({ rel_dist: 0 }), //           running, main, at the line
  PIA: driver({ rel_dist: 0.25 }), //        running, rival, a quarter round
  VER: driver({ active: false, has_finished: true, rel_dist: 0.5 }), // finished
  HUL: driver({ active: false, has_finished: false, rel_dist: 0.75 }), // out
  HAD: driver({ rel_dist: null, has_position: false }), // never placed, still RUNNING
  // Never placed AND retired, which is what HAD really is on Melbourne 2025 and
  // what no fixture carried: the blind list used to collect him for all 57 laps,
  // so the ring's only telemetry alarm was lit from the first capture to the
  // last. A retired car cannot be a car the telemetry lost.
  SAI: driver({ rel_dist: null, has_position: false, active: false, has_finished: false }),
};

const ring = await browser.newContext({ viewport: { width: 1500, height: 950 } });
const ringPage = await ring.newPage();
ringPage.on("pageerror", (error) => failures.push(`pageerror(ring): ${error.message}`));

await ringPage.addInitScript((payload) => {
  window.pywebview = {
    api: { get_tick: async (sinceSeq) => (sinceSeq === payload.seq ? null : payload) },
  };
}, tick(1, {
  drivers: FIELD,
  order: ["NOR", "PIA", "VER", "HUL", "HAD", "SAI"],
  // **The REAL team colours, including the ones that fail.** `tick()` defaults every
  // car to rgb(255,128,0), which is 6.94:1 on the card - it passes, so a fixture
  // built on the default cannot fail a contrast assertion however the codes are
  // painted. These are the arcade's own values for this session: VER at
  // rgb(6,0,239) is 1.88:1 and HUL's rgb(0,231,0) is 10.35, so the guard has both
  // ends of the range to work with.
  colors: {
    NOR: [255, 128, 0],
    PIA: [255, 128, 0],
    VER: [6, 0, 239],
    HUL: [0, 231, 0],
    HAD: [252, 215, 0],
    SAI: [0, 160, 221],
  },
}));

await ringPage.goto(url, { waitUntil: "domcontentloaded" });
await ringPage.waitForSelector(".ring-dot", { timeout: 5000 });
await ringPage.waitForTimeout(300);

const placed = await ringPage.evaluate(() =>
  [...document.querySelectorAll(".ring-dot circle")].map((el) => ({
    code: el.dataset.code,
    status: el.dataset.status,
    cx: Number(el.getAttribute("cx")),
    cy: Number(el.getAttribute("cy")),
    hollow: el.getAttribute("fill") === "none",
  })),
);
const of = (code) => placed.find((dot) => dot.code === code);

check(placed.length === 4, `four dots for five cars, one of them unplaced (${placed.length})`);
check(of("NOR")?.status === "running", "a car on track is running");
check(of("HUL")?.status === "out" && of("HUL")?.hollow, "a retirement is out, and hollow");
// The whole reason `has_finished` is on the wire.
check(
  of("VER")?.status === "finished" && !of("VER")?.hollow,
  `a chequered flag is finished, not out (${of("VER")?.status})`,
);

// The angle: fraction 0 is twelve o'clock and the lap runs clockwise, so a
// quarter of the way round is three o'clock. Asserted on the rendered
// coordinates, which is the only place the rotation can be wrong.
const CENTRE = 100;
const RADIUS = 78;
check(
  Math.abs(of("NOR").cx - CENTRE) < 0.5 && Math.abs(of("NOR").cy - (CENTRE - RADIUS)) < 0.5,
  `fraction 0 sits at the start line, top centre (${of("NOR").cx}, ${of("NOR").cy})`,
);
check(
  Math.abs(of("PIA").cx - (CENTRE + RADIUS)) < 0.5 && Math.abs(of("PIA").cy - CENTRE) < 0.5,
  `fraction 0.25 is a quarter clockwise (${of("PIA").cx}, ${of("PIA").cy})`,
);

// A car the telemetry never placed is NAMED, never drawn at fraction 0 -
// which is the start line, a position a real car can hold.
check(!of("HAD"), "the unplaced car has no dot");
const blindLine = await ringPage.locator(".ring-blind").innerText();
check(blindLine.includes("HAD"), "and the ring says which car it is");
// **The alarm names only the cars it is ABOUT.** It exists to flag a live car the
// telemetry lost; a retirement has no telemetry by definition and lighting it for
// one turns the alarm into furniture - on the real race it was lit from lap 1 to
// lap 57 by a car that crashed on the first lap.
check(
  !blindLine.includes("SAI"),
  `and not the retired one, which has no telemetry by definition ("${blindLine}")`,
);
check(
  (await ringPage.locator(".ring-code").count()) === 2,
  "only the two featured cars are labelled",
);

// --- Identity, and whether it can be READ -----------------------------------
//
// `driver_colors` are the arcade's own and they are TEAM colours, so colour never
// identified a car here - the CODE does. Six of the twenty fail AA as 11 px text
// on the card they are drawn on (VER and LAW at 1.88:1, ALO and STR at 2.55, HAM
// and LEC at 3.71) and four fail even the 3.0 large-text floor, so the row key of
// this window's primary panel was the part that could not be read.
//
// Asserted over the WHOLE enumeration of wire colours rather than on the codes
// that happen to be on screen, and as a RATIO rather than as a colour name: a
// membership test passes on any hex still in the palette.
const contrast = await ringPage.evaluate(() => {
  const lin = (c) => (c / 255 <= 0.03928 ? c / 255 / 12.92 : ((c / 255 + 0.055) / 1.055) ** 2.4);
  const L = ([r, g, b]) => 0.2126 * lin(r) + 0.7152 * lin(g) + 0.0722 * lin(b);
  const ratio = (a, b) => {
    const [hi, lo] = [L(a), L(b)].sort((x, y) => y - x);
    return (hi + 0.05) / (lo + 0.05);
  };
  const parse = (value) => value.match(/\d+/g).slice(0, 3).map(Number);
  const card = parse(getComputedStyle(document.querySelector(".tower")).backgroundColor);
  // `td` only: the header shares the class and is a column label, not a code.
  const cells = [...document.querySelectorAll("td.col-drv")];
  return {
    cells: cells.length,
    swatches: cells.filter((cell) => cell.querySelector(".drv-swatch")).length,
    // Every code, whatever its team colour: they all read against the card now.
    worstCode: Math.min(...cells.map((cell) => ratio(parse(getComputedStyle(cell).color), card))),
    // And no code is painted in a team colour any more.
    tinted: cells.filter((cell) => {
      const own = getComputedStyle(cell).color;
      const swatch = cell.querySelector(".drv-swatch");
      return swatch !== null && own === getComputedStyle(swatch).backgroundColor;
    }).length,
  };
});
check(
  contrast.worstCode >= 4.5 && contrast.tinted === 0,
  `every driver code reads against its card (worst ${contrast.worstCode.toFixed(2)}:1, ${contrast.tinted} still tinted)`,
);
check(
  contrast.swatches === contrast.cells,
  `and each one keeps its team colour as a swatch (${contrast.swatches}/${contrast.cells})`,
);


// The two labels go on OPPOSITE sides of their dots. The main driver and the
// car chosen to compare against are routinely seconds apart - on the real
// session NOR and PIA sit 0.006 of a lap apart - and with both codes above
// their dots they printed on top of each other. Asserted as rendered
// geometry, because "two labels exist" was already true while they overlapped.
const labels = await ringPage.evaluate(() =>
  [...document.querySelectorAll(".ring-code")].map((el) => {
    const dot = el.parentElement.querySelector("circle");
    return {
      code: el.textContent.trim(),
      above: Number(el.getAttribute("y")) < Number(dot.getAttribute("cy")),
    };
  }),
);
const main = labels.find((l) => l.code === "NOR");
const rival = labels.find((l) => l.code === "PIA");
check(
  main?.above === true && rival?.above === false,
  `the main code sits above its dot and the rival's below (${JSON.stringify(labels)})`,
);
// `textContent`, not `innerText`: these are SVG <text> nodes, which are not
// HTMLElements and have no `innerText` at all.
const lapText = await ringPage.locator(".ring-lap").textContent();
check(lapText?.trim() === "24", `the ring carries the lap counter (${lapText})`);

await ring.close();

// Band 4's charts must be STILL under a streaming producer. The payload here
// is re-served unchanged except for `seq`, so the traces carry no new data
// and every pixel that moves between two captures is the chart redrawing.
//
// This exists because the fix it guards had NO guard. `animation: false` in
// `TraceChart` is what stopped the delta baseline reaching 1328 m of a 5220 m
// axis and restarting forever, and 42 checks stayed green either way - so
// deleting that line would have been invisible. Its twin in the AGENTS charts
// shipped the same defect a sprint later and Víctor is the one who saw it.
const stillCtx = await browser.newContext({ viewport: { width: 1500, height: 950 } });
const stillPage = await stillCtx.newPage();
await stillPage.addInitScript((payload) => {
  let seq = payload.seq;
  window.pywebview = {
    api: {
      // Deep copy per poll, because the host builds a fresh payload every
      // tick: a shallow one keeps the nested identities and React's memos
      // never recompute, so the chart sits still whether or not it animates
      // and the check below would pass against the defect it exists for.
      get_tick: async () => ({ ...structuredClone(payload), seq: ++seq }),
      get_bulk: async () => null,
      get_live_lap: async () => null,
      get_connection: async () => "Connected",
    },
  };
}, tick(1));
await stillPage.goto(url, { waitUntil: "domcontentloaded" });
await stillPage.waitForSelector(".trace-stack-plot canvas", { timeout: 5000 });

check(
  await staysStill(stillPage, ".trace-stack-plot"),
  "the delta trace schedules no animation while the producer streams",
);

await stillCtx.close();

// --- Scenario E: PROVISIONAL is about the opening lap, not about retirements -
//
// The delivery plan words the rule as "until `laps_completed >= 1` for every
// driver", and read literally it never switches off on the race this window
// is developed against: SAI, DOO and HAD crashed on lap 1, so their
// `laps_completed` is 0 for the whole race. Measured on the live wire at lap
// 23, three of twenty drivers were under one lap and all three were OUT - the
// chip built to mark the opening lap was still lit an hour in, which says
// nothing at all. So the test is over the cars still IN the race.
const RETIRED = { laps_completed: 0, progress: 0.4, active: false, has_finished: false };

async function provisionalChips(field) {
  const context = await browser.newContext({ viewport: { width: 1500, height: 950 } });
  const scenarioPage = await context.newPage();
  scenarioPage.on("pageerror", (error) => failures.push(`pageerror(provisional): ${error.message}`));
  await scenarioPage.addInitScript((payload) => {
    window.pywebview = {
      api: {
        get_tick: async (sinceSeq) => (sinceSeq === payload.seq ? null : payload),
        get_bulk: async () => null,
        get_live_lap: async () => null,
      get_live_lap: async () => null,
        get_connection: async () => "Connected",
      },
    };
  }, tick(1, { drivers: field }));
  await scenarioPage.goto(url, { waitUntil: "domcontentloaded" });
  await scenarioPage.waitForSelector(".status-strip", { timeout: 5000 });
  await scenarioPage.waitForTimeout(300);
  const count = await scenarioPage.locator(".strip-chip.is-provisional").count();
  await context.close();
  return count;
}

check(
  (await provisionalChips({
    NOR: driver({ laps_completed: 23 }),
    PIA: driver({ laps_completed: 23 }),
    SAI: driver(RETIRED),
    DOO: driver(RETIRED),
    HAD: driver(RETIRED),
  })) === 0,
  "three lap-1 retirements do NOT keep the tower provisional for the whole race",
);
check(
  (await provisionalChips({
    NOR: driver({ laps_completed: 0, progress: 0.6 }),
    PIA: driver({ laps_completed: 0, progress: 0.5 }),
  })) === 1,
  "and the opening lap, which the chip exists for, still marks itself",
);

// --- Scenario F: the timing tower --------------------------------------------
//
// Twenty rows, and the numbers hand-worked so a wrong branch cannot look
// plausible. The bulk deliberately omits the three retired cars: SAI, DOO and
// HAD have only `FastF1Generated` rows on the real race and reveal nothing, so
// a tower keyed on what the BULK contains renders seventeen. It has to iterate
// `race_order`, which always carries twenty.

const TOWER_ORDER = [
  "NOR", "PIA", "VER", "RUS", "LEC", "TSU", "ALB", "HAM", "GAS", "ALO",
  "ANT", "STR", "HUL", "BOR", "LAW", "OCO", "BEA", "SAI", "DOO", "HAD",
];
const RETIRED_CODES = ["SAI", "DOO", "HAD"];
/** Crossing times at lap 23, chosen so every interval is exact at 2 dp. */
const CROSSING_AT_23 = { NOR: 2070.0, PIA: 2071.24, VER: 2073.07 };

function towerField() {
  const field = {};
  TOWER_ORDER.forEach((code, index) => {
    if (RETIRED_CODES.includes(code)) {
      field[code] = driver({
        laps_completed: 0,
        progress: 0.4,
        active: false,
        has_finished: false,
      });
      return;
    }
    // LAW is a full lap down: the positional laps-down branch must fire for
    // him instead of rendering tens of seconds as if he were racing.
    const progress = code === "LAW" ? 22.3 : 23.9 - index * 0.01;
    field[code] = driver({ laps_completed: code === "LAW" ? 22 : 23, progress });
  });
  return field;
}

/**
 * Per-driver bests, arranged so the three sector tones and the field ranking
 * are all decidable by hand.
 *
 * - NOR owns S1 outright and his last lap MATCHES it       -> purple
 * - PIA owns S2, and his last S1 equals his own best S1     -> green
 * - VER owns S3, and his last S1 is slower than his own     -> yellow
 *
 * Theoretical = NOR's S1 + PIA's S2 + VER's S3 = 29.000 + 18.500 + 25.900.
 * Every other driver is deliberately slower in all four fields, so the
 * podium of each section is fixed.
 */
const TOWER_BESTS = {
  NOR: { s1: 29.0, s2: 19.0, s3: 26.5, lap_time: 85.0, lastS1: 29.0 },
  PIA: { s1: 29.5, s2: 18.5, s3: 26.4, lap_time: 85.4, lastS1: 29.5 },
  VER: { s1: 29.8, s2: 19.2, s3: 25.9, lap_time: 85.9, lastS1: 30.2 },
};

function towerBulk() {
  const drivers = {};
  TOWER_ORDER.filter((code) => !RETIRED_CODES.includes(code)).forEach((code, index) => {
    const revealed = code === "LAW" ? 22 : 23;
    const crossings = {};
    for (let lap = 1; lap <= revealed; lap += 1) {
      crossings[lap] = CROSSING_AT_23[code] ?? 2070 + 5 * index - (23 - lap) * 90;
    }
    if (CROSSING_AT_23[code] !== undefined) crossings[revealed] = CROSSING_AT_23[code];
    const crafted = TOWER_BESTS[code];
    const best = crafted ?? {
      s1: 31 + index * 0.1,
      s2: 21 + index * 0.1,
      s3: 28 + index * 0.1,
      lap_time: 90 + index * 0.1,
      lastS1: 31 + index * 0.1,
    };
    drivers[code] = {
      number: String(index + 1),
      laps_revealed: revealed,
      stops: 2,
      crossings,
      laps: [
        {
          lap: revealed,
          t: crossings[revealed],
          lap_time: 85.744,
          s1: best.lastS1,
          s2: 30.128,
          s3: 26.204,
          v1: 301,
          v2: 289,
          vfl: 280,
          vst: 321,
          position: index + 1,
          compound: "MEDIUM",
          tyre_life: 12,
          stint: 2,
          track_status: "1",
          neutralised: null,
          pit_in: false,
          pit_out: false,
          deleted: false,
          generated: false,
          pb: false,
        },
      ],
      best: {
        lap: revealed,
        lap_time: best.lap_time,
        s1: best.s1,
        s2: best.s2,
        s3: best.s3,
        v1: 301,
        v2: 289,
        vfl: 280,
        vst: 321,
        compound: "MEDIUM",
      },
      theoretical: best.s1 + best.s2 + best.s3,
    };
  });
  return { rev: 1, available: true, race: { year: 2025, location: "Melbourne", total_laps: 57 }, drivers };
}

/**
 * The lap in progress: S1 crossed on THIS lap, S2 and S3 carried over.
 *
 * The cells roll rather than blank, so the fixture carries a value in all
 * three and the flags say which lap each belongs to. **The first version of
 * this fixture set `s2` and `s3` to null and the checks asserted dashes** -
 * which is how the S3 column came to be permanently empty on the real race
 * and no guard noticed: the test and the code agreed with each other and
 * neither agreed with the parquet (#933).
 */
function towerLive() {
  const drivers = {};
  TOWER_ORDER.filter((code) => !RETIRED_CODES.includes(code)).forEach((code, index) => {
    const crafted = TOWER_BESTS[code];
    drivers[code] = {
      lap: code === "LAW" ? 23 : 24,
      s1: crafted ? crafted.lastS1 : 31 + index * 0.1,
      s2: 19.5,
      s3: 26.75,
      v1: 301,
      v2: 288,
      vfl: 279,
      s1_fresh: true,
      s2_fresh: false,
      s3_fresh: false,
    };
  });
  return { rev: 1, drivers };
}

const towerCtx = await browser.newContext({ viewport: { width: 1485, height: 833 } });
const towerPage = await towerCtx.newPage();
towerPage.on("pageerror", (error) => failures.push(`pageerror(tower): ${error.message}`));
await towerPage.addInitScript(
  ([payload, bulk, live]) => {
    window.pywebview = {
      api: {
        get_tick: async (sinceSeq) => (sinceSeq === payload.seq ? null : payload),
        get_bulk: async (sinceRev) => (sinceRev === bulk.rev ? null : bulk),
        get_live_lap: async (sinceRev) => (sinceRev === live.rev ? null : live),
        get_connection: async () => "Connected",
      },
    };
  },
  [tick(1, { drivers: towerField(), order: TOWER_ORDER }), towerBulk(), towerLive()],
);
await towerPage.goto(url, { waitUntil: "domcontentloaded" });
await towerPage.waitForSelector(".tower-row", { timeout: 5000 });
await towerPage.waitForTimeout(700);

// Tolerant on purpose. A defect that DROPS rows makes every later selector
// miss, and a throwing helper kills the harness with a stack instead of
// reporting the failures by name - so the one check that explains the cause
// never gets printed. A missing cell is a named failure, not an exception.
const cell = async (row, column) => {
  try {
    const text = await towerPage
      .locator(`.tower-row:nth-child(${row}) td:nth-child(${column})`)
      .innerText({ timeout: 1000 });
    return text.trim();
  } catch {
    return "<no such cell>";
  }
};

check((await towerPage.locator(".tower-row").count()) === 20, "twenty rows, not the bulk's seventeen");
check((await cell(18, 3)) === "SAI", "a car with no revealed lap still holds its place in the order");

// The four branches of the gap cell, each on the row that exercises it.
check((await cell(1, 4)) === "LEADER", "the leader's GAP says so");
check((await cell(1, 5)) === "—", "and the leader has no interval to anything");
check((await cell(2, 4)) === "+1.24s", "GAP is measured to the LEADER");
check((await cell(3, 4)) === "+3.07s", "and it accumulates down the order");
check((await cell(3, 5)) === "+1.83s", "INT is measured to the car directly ahead");
check((await cell(15, 4)) === "+1 LAP", "a lapped car reads in laps, not in tens of seconds");
check((await cell(18, 4)) === "OUT", "a stopped car never shows a frozen interval");
check((await cell(18, 9)) === "OUT", "and its LAST column says so in place of a lap time");

// The row itself, on the leader.
check((await cell(1, 3)) === "NOR", "the code column");
check(
  (await cell(1, 6)).replace(/\s+/g, " ") === "29.000 301",
  "the sector time and its trap speed, inline",
);
check((await cell(1, 9)) === "1:25.744", "a lap time past the minute reads as m:ss.mmm");
check((await cell(1, 10)) === "321", "ST is the speed trap");
check((await cell(1, 11)) === "M 12", "the compound letter and the set's age");
check((await cell(1, 12)) === "2", "the stop count");
check((await cell(18, 11)) === "—", "a driver the bulk never revealed shows dashes, not zeros");

// EFFECT, not a CSS property: all twenty rows are inside the card. Scrollbars
// are hidden globally, so an overflowing tower does not announce itself - it
// silently stops drawing P15 to P20 and looks complete.
const towerFits = await towerPage.evaluate(() => {
  const cardEl = document.querySelector(".tower");
  const card = cardEl.getBoundingClientRect();
  const rows = [...document.querySelectorAll(".tower-row")];
  const last = rows[rows.length - 1].getBoundingClientRect();
  // The table's NATURAL width against the card's content box. A rect assert
  // on the rendered table is a mechanism check wearing an effect check's
  // clothes: `width: 100%` pins the table to the card, so when the columns
  // no longer fit they compress and their `nowrap` text spills into the
  // padding while every rectangle still says "inside". Measured at a 600 px
  // column: natural 597 against 578 of content box, and the text ends 5 px
  // past the padding edge - a tower touching its own border.
  const table = document.querySelector(".tower-table");
  const applied = table.style.width;
  table.style.width = "max-content";
  const naturalWidth = table.getBoundingClientRect().width;
  table.style.width = applied;
  const style = getComputedStyle(cardEl);
  const contentWidth =
    cardEl.clientWidth - parseFloat(style.paddingLeft) - parseFloat(style.paddingRight);
  return {
    lastBottom: last.bottom,
    cardBottom: card.bottom,
    naturalWidth,
    contentWidth,
    visible: rows.filter((row) => row.getBoundingClientRect().height > 0).length,
  };
});
check(
  towerFits.lastBottom <= towerFits.cardBottom + 1,
  `the last row is inside the card (${towerFits.lastBottom} vs ${towerFits.cardBottom})`,
);
check(
  towerFits.naturalWidth <= towerFits.contentWidth,
  `no column is compressed: the table wants ${towerFits.naturalWidth} and the card gives ${towerFits.contentWidth}`,
);
check(towerFits.visible === 20, `all twenty rows have height (${towerFits.visible})`);

// --- The sector colour code, on the tower ------------------------------------
//
// Purple is fastest of the session outright, green is the driver's own best,
// yellow is slower than his own. The fixture arranges all three on the S1
// column: NOR owns it and matched it, PIA matched his own, VER did not.
const tone = async (row) =>
  towerPage.locator(`.tower-row:nth-child(${row}) td:nth-child(6)`).getAttribute("class");
check((await tone(1)).includes("is-purple"), "the session's fastest sector is purple");
check((await tone(2)).includes("is-green"), "a driver's own best is green");
check((await tone(3)).includes("is-yellow"), "slower than his own best is yellow");
check((await tone(18)).includes("is-plain"), "a sector nobody has set is not coloured at all");

// --- The sectors are the lap IN PROGRESS, not the last completed one ---------
//
// The bulk's completed row carries an S2 and an S3 for every driver here, so a
// tower reading it would print them. The live channel has only S1 open, which
// is what a car that has crossed one sector of the current lap actually knows.
// S2 and S3 carry the PREVIOUS lap's values, dimmed. They are not blank, and
// that is the whole of #933: a cell that blanked until this lap's crossing
// left S3 empty for the entire race, because a third sector's crossing IS the
// end of its lap.
check((await cell(1, 7)).startsWith("19.500"), "S2 shows the previous lap's value, not a dash");
check((await cell(1, 8)).startsWith("26.750"), "and so does S3, which otherwise NEVER fills");
const staleness = async (column) =>
  (await towerPage.locator(`.tower-row:nth-child(1) td:nth-child(${column})`).getAttribute("class"))
    .includes("is-stale");
check((await staleness(6)) === false, "this lap's own sector is not dimmed");
check((await staleness(7)) === true, "a carried-over sector says so by being dimmed");
check((await staleness(8)) === true, "and S3 is carried over essentially always");
check(
  (await cell(1, 9)) === "1:25.744",
  "while LAST still shows the lap the car completed, which is what that column means",
);
check(
  (await cell(18, 6)) === "—",
  "a retired car has no lap in progress, so its sectors are dashes and not its final ones",
);

// --- Band 2 right: the bests panel -------------------------------------------

const bestsRow = async (section, row) =>
  (
    await towerPage
      .locator(`.bests-section:nth-child(${section}) .bests-row:nth-child(${row})`)
      .innerText()
  )
    .replace(/\s+/g, " ")
    .trim();

check((await towerPage.locator(".bests-section").count()) === 4, "four ranked sections");
check(
  (await towerPage.locator(".bests-section:nth-child(1) .bests-row").count()) === 3,
  "top three per section, because twenty would be 1,668 px of a 790 px body",
);
// S1 is NOR 29.000, PIA 29.500, VER 29.800 - ranked across the FIELD, not per
// driver, and the delta is a percentage off the section's leader.
check((await bestsRow(1, 2)).startsWith("1 NOR 29.000"), "the section leader, with no delta");
check((await bestsRow(1, 3)) === "2 PIA 29.500 +1.72%", "second, with its percentage off the top");
check((await bestsRow(1, 4)) === "3 VER 29.800 +2.76%", "and third");
// S2 and S3 are owned by different drivers, so a panel reading one field for
// all four sections would show NOR three times.
check((await bestsRow(2, 2)).startsWith("1 PIA 18.500"), "S2 belongs to whoever set it");
check((await bestsRow(3, 2)).startsWith("1 VER 25.900"), "and so does S3");
check(
  (await bestsRow(4, 2)).replace(/\s+$/, "") === "1 NOR 1:25.000 M",
  "the LAP section carries the compound; the sector sections cannot and do not",
);

// 29.000 + 18.500 + 25.900 = 73.400, from three DIFFERENT drivers. A panel
// summing one driver's own sectors would read 1:14.500 for NOR.
const theoretical = (await towerPage.locator(".bests-theoretical").innerText())
  .replace(/\s+/g, " ")
  .trim();
check(
  theoretical === "THEORETICAL 1:13.400 NOR · PIA · VER",
  `the ideal lap recombines the FIELD's best sectors (${theoretical})`,
);

await towerCtx.close();

// --- The radio / RCM feed, in the column under the ring ----------------------

/**
 * Six events chosen so every branch of the panel is decidable by eye.
 *
 * Oldest first, exactly as the host serves them: the panel reverses, and a
 * fixture already in display order would let a renderer that forgot to reverse
 * pass. LAW's radio is longer than two lines at 260 px, so it also exercises
 * the clamp; HAM's has no transcript, which is what 23 of the 24 published
 * races look like.
 */
const RADIO_EVENTS = [
  { kind: "rcm", lap: 2, driver: null, text: "DOUBLE YELLOW IN TRACK SECTOR 20", category: "Flag", flag: "DOUBLE YELLOW" },
  { kind: "radio", lap: 6, driver: "NOR", text: "Weather update, no significant rain expected.", category: null, flag: null },
  { kind: "radio", lap: 14, driver: "HAM", text: "", category: null, flag: null },
  { kind: "rcm", lap: 20, driver: null, text: "FIA STEWARDS: INCIDENT INVOLVING CAR 22 (TSU) NO FURTHER ACTION - SAFETY CAR INFRINGEMENT", category: "Other", flag: null },
  { kind: "radio", lap: 21, driver: "NOR", text: "Lando, a bit of an update on the safety car window.", category: null, flag: null },
  { kind: "radio", lap: 23, driver: "VER", text: "If there is heavy rain we might need to fit inters, bear in mind.", category: null, flag: null },
];

async function radioPage(radio) {
  const context = await browser.newContext({ viewport: { width: 1485, height: 833 } });
  const rPage = await context.newPage();
  rPage.on("pageerror", (error) => failures.push(`pageerror(radio): ${error.message}`));
  await rPage.addInitScript(
    ([payload, bulk, live]) => {
      window.pywebview = {
        api: {
          get_tick: async (sinceSeq) => (sinceSeq === payload.seq ? null : payload),
          get_bulk: async (sinceRev) => (sinceRev === bulk.rev ? null : bulk),
          get_live_lap: async (sinceRev) => (sinceRev === live.rev ? null : live),
          get_connection: async () => "Connected",
        },
      };
    },
    [
      tick(1, { drivers: towerField(), order: TOWER_ORDER }),
      { ...towerBulk(), radio },
      towerLive(),
    ],
  );
  await rPage.goto(url, { waitUntil: "domcontentloaded" });
  await rPage.waitForSelector(".radio-feed", { timeout: 5000 });
  await rPage.waitForTimeout(500);
  return [context, rPage];
}

const [radioCtx, feedPage] = await radioPage({ available: true, events: RADIO_EVENTS });

// Tolerant for the same reason `cell` is: a defect that DROPS rows makes every
// later selector miss, and a throwing helper kills the harness with a stack
// instead of naming the failures - so the one check that explains the cause is
// never printed. Measured: a mutation that filtered out transcript-less rows
// took the whole run down with a TimeoutError before this was tolerant.
const rowText = async (n) => {
  try {
    const text = await feedPage
      .locator(`.radio-row:nth-child(${n})`)
      .innerText({ timeout: 1000 });
    return text.replace(/\s+/g, " ").trim();
  } catch {
    return "<no such row>";
  }
};

// Six events, six rows. The panel drops nothing of its own accord: the only
// thing allowed to remove an event is the reveal, upstream of here. Measured
// on a mutated copy that filtered out transcript-less radios - the count in the
// header still read 6, because it counts the PAYLOAD, so without this check the
// six-into-five silently passed.
check(
  (await feedPage.locator(".radio-row").count()) === RADIO_EVENTS.length,
  "every revealed event gets a row when they all fit",
);

// Newest FIRST. A pit wall reads the top line; a feed rendered in arrival order
// puts lap 2 there and the freshest message off the bottom of a 416 px card.
check((await rowText(1)).startsWith("L23 VER"), "the newest event is the top row");
check((await rowText(6)).startsWith("L2 RCM"), "and the oldest is the last one");

// The minute boundary `paceLabel` documented and its two siblings did not have.
// Evaluated against the SHIPPED module rather than a copy of its arithmetic.
const boundary = await feedPage.evaluate(async () => {
  const mod = await import("/src/lib/format.ts").catch(() => null);
  return mod === null ? null : [
    mod.formatSeconds(119.9996, 3),
    mod.formatSeconds(59.9996, 3),
    mod.formatSeconds(29.412, 3),
    mod.formatSeconds(119.96, 1, true),
  ];
});
if (boundary !== null) {
  check(
    JSON.stringify(boundary) === JSON.stringify(["2:00.000", "1:00.000", "29.412", "2:00.0"]),
    `the shared formatter rounds before it splits (${JSON.stringify(boundary)})`,
  );
}

// --- The history a reader could not reach ---------------------------------
//
// `.radio-list` and `.radio-feed` were BOTH `overflow: hidden`, which made this
// the one panel in the window whose content was genuinely gone rather than merely
// unadorned: measured on the live page, 10 of 46 events visible and no user input
// that could reach the other 36 - `scrollTop` from the console worked, so the rows
// were there and only the reader had no path in. The window's own rule is the
// opposite: `qt-base.css` hides the scrollBAR and keeps bodies scrollable.
//
// Six events fit, so this needs its own fixture: a fold has to EXIST before
// "can it be reached" is a question, and a guard whose probe sits where the
// content fits cannot see the defect it names.
//
// **`overflowY` is not the mechanism half of this check, it IS the effect.**
// Driving the mutation proved it: with `overflow: hidden` a scripted
// `scrollTop = scrollHeight` still moves the list to 1615 and the oldest row is
// still in view, so the scroll-and-look half passes ON the defect. What
// `overflow: hidden` blocks is the WHEEL, the trackpad and the keyboard - the
// reader's only paths - and the computed value is what says whether they work.
{
  const many = Array.from({ length: 60 }, (_, index) => ({
    kind: index % 3 === 0 ? "rcm" : "radio",
    lap: 40 - Math.floor(index / 2),
    driver: index % 3 === 0 ? null : "NOR",
    text: `event ${index} - long enough to occupy a row of its own on the panel`,
  }));
  const [longCtx, longPage] = await radioPage({ available: true, events: many });
  const reach = await longPage.evaluate(async () => {
    const list = document.querySelector(".radio-list");
    const rows = [...list.querySelectorAll(".radio-row")];
    const folded = list.scrollHeight - list.clientHeight;
    const before = list.scrollTop;
    list.scrollTop = list.scrollHeight;
    await new Promise((done) => setTimeout(done, 60));
    const box = list.getBoundingClientRect();
    const oldest = rows[rows.length - 1]?.getBoundingClientRect();
    return {
      rows: rows.length,
      folded,
      before,
      after: list.scrollTop,
      overflowY: getComputedStyle(list).overflowY,
      oldestReached: !!oldest && oldest.bottom <= box.bottom + 1 && oldest.top >= box.top - 1,
    };
  });
  check(
    reach.folded > 100 && reach.rows === many.length,
    `the long feed really has a fold (${reach.folded} px hidden over ${reach.rows} rows)`,
  );
  check(
    reach.overflowY === "auto" && reach.after > reach.before && reach.oldestReached,
    `and the oldest event can be reached (${reach.overflowY}, scrollTop ${reach.before} -> ${reach.after}, oldest in view: ${reach.oldestReached})`,
  );
  await longCtx.close();
}

// The tier claim, on screen rather than in a PDF. NOR is `driver_main`.
const tiers = await feedPage.evaluate(() =>
  [...document.querySelectorAll(".radio-row")].map((row) => ({
    who: row.querySelector(".radio-who")?.textContent ?? "",
    broadcast: row.querySelector(".radio-tier") !== null,
  })),
);
check(
  tiers.filter((row) => row.who === "VER" || row.who === "HAM").every((row) => row.broadcast),
  "a rival's radio is tagged BROADCAST, exactly as band 4 tags a pinned rival's trace",
);
check(
  tiers.filter((row) => row.who === "NOR").every((row) => !row.broadcast),
  "our own car's radio is not - it is team tier and carries no tag",
);
check(
  tiers.filter((row) => row.who === "RCM").every((row) => !row.broadcast),
  "and race control is neither: it is public by definition, not a rival's channel",
);

// A radio whose audio was never transcribed still occupies a row. Dropping it
// would present a race as quieter than it was, and that is the COMMON case:
// 23 of the 24 published races have no transcript at all.
check((await rowText(4)).includes("no transcript"), "a radio with no words still shows itself");

// The count is what stops an overflow being silent. Scrollbars are hidden
// globally, so a panel that shows nine of forty-two and says nothing looks
// exactly like a panel showing all there is.
check(
  (await feedPage.locator(".radio-count").innerText()).trim() === String(RADIO_EVENTS.length),
  "the header counts every revealed event, not the ones that happen to fit",
);

// EFFECT, not mechanism: the card must not spill past the column it lives in.
const fits = await feedPage.evaluate(() => {
  const card = document.querySelector(".radio-feed");
  const column = document.querySelector(".side-column");
  return {
    overflow: card.getBoundingClientRect().bottom - column.getBoundingClientRect().bottom,
    ringVisible: document.querySelector(".ring").getBoundingClientRect().height > 0,
  };
});
check(fits.overflow <= 1, `the feed stays inside its column (spills ${fits.overflow.toFixed(1)}px)`);
check(fits.ringVisible, "and the ring above it is still there");

await radioCtx.close();

// A race with no corpus SAYS so. An empty list and a missing corpus are the
// same pixel otherwise, which is the twin F7 caught one sprint ago between
// get_bulk and get_live_lap.
const [emptyCtx, emptyPage] = await radioPage({ available: false, events: [] });
check((await emptyPage.locator(".radio-row").count()) === 0, "no rows for a race with no corpus");
check(
  (await emptyPage.locator(".radio-subtitle").innerText()).includes("no corpus"),
  "and the panel says so instead of going quietly blank",
);
await emptyCtx.close();

// --- Band 3: the race-pace grid -------------------------------------------

/**
 * Twenty drivers over fifty-seven laps, with the three lap-1 crashers carrying
 * ONLY generated rows - which is what Melbourne really looks like, and what a
 * grid keyed on the bulk's own keys silently renders as seventeen columns.
 *
 * `TOWER_ORDER` is ranked by POSITION; the numbers below are deliberately not
 * in that order, so a grid that renders its columns in wire order and one that
 * sorts them stably by car number cannot both pass.
 */
const PACE_LAPS = 57;
const PACE_FASTEST = { code: TOWER_ORDER[3], lap: 30, time: 84.111 };
/**
 * One classified car has completed FEWER laps than the rest, which is what a
 * lapped car looks like and what bounds the race trace.
 *
 * LAW is already the lapped car in the tower fixture (`progress: 22.3` against
 * everyone else's 23.9), so the two fixtures agree about who is a lap down.
 * Without him every driver would reveal all 57 and the trace's cap - the last
 * lap the WHOLE classified field has completed - would be indistinguishable
 * from "the last lap anybody has", which is the ragged edge it exists to
 * refuse.
 */
const PACE_LAPPED = { code: "LAW", laps: 55 };
/**
 * A car that RACED and then stopped, unlike `RETIRED_CODES` who never started.
 *
 * The existing retirees carry zero crossings, so removing them from a reference
 * average changes nothing - which is exactly why the fixture was 100 % blind to
 * a reference computed over CURRENT status moving the drawn history under the
 * reader. This one has 30 real laps behind him and then nothing.
 */
const PACE_RETIRED_MIDRACE = { code: TOWER_ORDER[8], laps: 30 };

function paceBulk() {
  const drivers = {};
  // One driver has no car number and one is missing from the bulk entirely.
  // Both are typed as possible (`DriverLaps.number` is `string | null`) and
  // neither appeared in any fixture, so the guards below could not see an
  // intransitive sort or a column keyed on the bulk.
  const NO_NUMBER = TOWER_ORDER[6];
  const ABSENT_FROM_BULK = TOWER_ORDER[9];
  TOWER_ORDER.forEach((code, index) => {
    if (code === ABSENT_FROM_BULK) return;
    const number = code === NO_NUMBER ? null : String((index * 7 + 1) % 88);
    if (RETIRED_CODES.includes(code)) {
      // Generated-only: rendered, counted in nothing.
      drivers[code] = {
        number, laps_revealed: 0, stops: 0, crossings: {},
        laps: [{ lap: 1, t: null, lap_time: null, s1: null, s2: null, s3: null,
          v1: null, v2: null, vfl: null, vst: null, position: null, compound: null,
          tyre_life: null, stint: null, track_status: "1", neutralised: null,
          pit_in: false, pit_out: false, deleted: false, generated: true, pb: false }],
        best: { lap: null, lap_time: null, s1: null, s2: null, s3: null,
          v1: null, v2: null, vfl: null, vst: null, compound: null },
        theoretical: null,
      };
      return;
    }
    const laps = [];
    const crossings = {};
    let elapsed = 0;
    let best = null;
    const revealed =
      code === PACE_LAPPED.code
        ? PACE_LAPPED.laps
        : code === PACE_RETIRED_MIDRACE.code
          ? PACE_RETIRED_MIDRACE.laps
          : PACE_LAPS;
    for (let lap = 1; lap <= revealed; lap += 1) {
      const pitIn = lap === 20 && index % 5 === 0;
      const pitOut = lap === 21 && index % 5 === 0;
      const isFastest = code === PACE_FASTEST.code && lap === PACE_FASTEST.lap;
      // 119.96 s renders "1:60.0" if the minutes are split off BEFORE the
      // tenths are rounded - a non-time that the cell regex below accepts.
      const onBoundary = code === TOWER_ORDER[2] && lap === 50;
      // Laps 2-6 are the safety car, and they are in this fixture because the
      // REAL race has them: measured on Melbourne 2025, 82.4 % of laps sit
      // past +10 % of the session best, so a heat scale anchored on that best
      // paints four fifths of the grid one colour. A tidy fixture cannot tell
      // the two scales apart - this one can.
      // Laps 2-6 are one range; lap 30 is a SECOND, ONE lap long. A live safety
      // car looks like that - only the lap just revealed is marked - and an
      // unpadded one-lap range is `from == to`, which paints a zero-width band.
      // No fixture had one, so the case could not be seen.
      const neutralised = (lap >= 2 && lap <= 6) || lap === 30;
      const green = 85 + (index % 9) * 0.4 + ((lap * 3) % 11) * 0.2;
      const time = onBoundary
        ? 119.96
        : isFastest
          ? PACE_FASTEST.time
          : neutralised
            ? green * 2.4
            : green;
      // Melbourne really carries deleted racing laps - six of them - and every
      // row of this fixture used to say `deleted: false`, so the branch that
      // paints them was never entered by any check.
      const deleted = lap === 44 && index % 3 === 0;
      if (!pitIn && !pitOut && !deleted && (best === null || time < best)) best = time;
      const lapTime = pitIn || pitOut ? time + 22 : time;
      // The crossing clock is the CUMULATIVE lap time, which is what it is on
      // the real wire and what makes the race trace mean anything: a pit stop
      // is +22 s here, so the trace has to render it as a step of about that.
      // Every driver used to carry `crossings: {}` and `t: lap * 90` - the same
      // number for all twenty - so a trace built on this fixture would have
      // been twenty flat lines at zero and every check would have passed.
      elapsed += lapTime;
      crossings[lap] = elapsed;
      // **The fixture used to say `track_status: "1"` on the very laps it calls
      // the safety car.** It knew they were neutralised - `neutralised` above
      // makes their times 2.4x - and told the wire they were green, so the rail
      // and the trace's shaded band had no fixture that could exercise them.
      laps.push({ lap, t: elapsed, lap_time: lapTime,
        s1: 29, s2: 30, s3: 26, v1: 301, v2: 289, vfl: 280, vst: 321,
        position: index + 1, compound: "MEDIUM", tyre_life: lap, stint: 1,
        track_status: neutralised ? "4" : "1",
        neutralised: neutralised ? "SAFETY CAR" : null,
        pit_in: pitIn, pit_out: pitOut, deleted,
        generated: false, pb: false });
    }
    drivers[code] = { number, laps_revealed: revealed, stops: 1, crossings, laps,
      best: { lap: PACE_FASTEST.lap, lap_time: best, s1: 29, s2: 30, s3: 26,
        v1: 301, v2: 289, vfl: 280, vst: 321, compound: "MEDIUM" },
      theoretical: 85 };
  });
  return { rev: 1, available: true, race: { year: 2025, location: "Melbourne", total_laps: PACE_LAPS },
    drivers, radio: { available: true, events: [] } };
}

/**
 * The tower's field, with the mid-race retirement actually marked as retired.
 *
 * `towerField()` cannot carry it: the tower's own checks assert LEC's gaps, and
 * a car that is OUT renders `OUT` instead. The trace reads status from the tick
 * and laps from the bulk, so this is the only fixture where the two have to
 * disagree - a car with 30 laps of real history who is no longer running.
 *
 * Left RUNNING he would pin the cap at 30 of 57, which is the OBS-4 shape the
 * plan documents (a classified car whose telemetry stopped) and a real state -
 * but not the one this scenario is built to test.
 */
function paceField() {
  const field = towerField();
  field[PACE_RETIRED_MIDRACE.code] = driver({
    laps_completed: PACE_RETIRED_MIDRACE.laps,
    progress: PACE_RETIRED_MIDRACE.laps + 0.4,
    active: false,
    has_finished: false,
  });
  return field;
}

const paceCtx = await browser.newContext({ viewport: { width: 1485, height: 833 } });
const pacePage = await paceCtx.newPage();
pacePage.on("pageerror", (error) => failures.push(`pageerror(pace): ${error.message}`));
await pacePage.addInitScript(
  ([payload, bulk, live]) => {
    window.pywebview = { api: {
      get_tick: async (s) => (s === payload.seq ? null : payload),
      get_bulk: async (r) => (r === bulk.rev ? null : bulk),
      get_live_lap: async (r) => (r === live.rev ? null : live),
      get_connection: async () => "Connected",
    } };
  },
  [tick(1, { drivers: paceField(), order: TOWER_ORDER }), paceBulk(), towerLive()],
);
await pacePage.goto(url, { waitUntil: "domcontentloaded" });
await pacePage.waitForSelector(".tab-strip", { timeout: 5000 });

// The ring and the traces own the column until the reader asks for the grid.
check((await pacePage.locator(".ring").count()) === 1, "the TRACES tab opens with the ring on it");
check((await pacePage.locator(".pace-table").count()) === 0, "and the grid is not mounted yet");

await pacePage.getByRole("tab", { name: "RACE PACE" }).click();
await pacePage.waitForSelector(".pace-table", { timeout: 5000 });
await pacePage.waitForTimeout(400);

// Measured, not assumed: with the 260 px ring column still mounted the grid
// gets 555 px, its columns fall to 25.25 px against 25 px of text and 1,101 of
// 1,140 cells clip. There is no arrangement that keeps both.
check((await pacePage.locator(".ring").count()) === 0, "the ring hides on the RACE PACE tab");
check((await pacePage.locator(".radio-feed").count()) === 0, "and the radio feed hides with it");

const paceHead = await pacePage.evaluate(() =>
  [...document.querySelectorAll(".pace-table thead th")].slice(1).map((h) => h.textContent),
);
check(paceHead.length === 20, `every driver the wire names gets a column (${paceHead.length})`);
// One of them is absent from `bulk.drivers` altogether. The design says the
// wire decides WHO races; a grid that reduced over the bulk instead would be
// one column short and nothing would say so. (The reason once given for this -
// "the bulk renders seventeen" - was false: `masked_view` iterates every driver
// it loaded, so the bulk carries all twenty at every reveal. The rule stands;
// its old justification did not, and this check replaces it.)
check(
  paceHead.includes(TOWER_ORDER[9]),
  "a driver the bulk does not name still gets his column from the wire",
);
// Ascending by car number, with the unknown at the END rather than acting as a
// barrier. Two orderings in one comparator is intransitive: measured on the
// real bundle with one unknown at wire index 1, car 44 rendered ahead of car 1.
const paceNumbers = await pacePage.evaluate(
  ([head, bulkNumbers]) => head.map((code) => bulkNumbers[code] ?? null),
  [paceHead, Object.fromEntries(Object.entries(paceBulk().drivers).map(([c, d]) => [c, d.number === null ? null : Number(d.number)]))],
);
const knownNumbers = paceNumbers.filter((n) => n !== null);
check(
  knownNumbers.every((n, i) => i === 0 || knownNumbers[i - 1] <= n),
  `columns ascend by car number (${knownNumbers.join(",")})`,
);
// Every unknown at the END, as a suffix. There are two of them here - the
// driver with a null number and the one the bulk never names - and the
// assertion is that no numbered car sits behind either, which is exactly what
// the intransitive comparator did: it stranded the cars before the unknown.
const firstNull = paceNumbers.indexOf(null);
check(
  firstNull === -1 || paceNumbers.slice(firstNull).every((n) => n === null),
  `unknown numbers sort to the end instead of stranding the cars before them (${paceNumbers.join(",")})`,
);
check(
  new Set(paceHead).size === 20 && paceHead.every((code) => TOWER_ORDER.includes(code)),
  "every driver the wire names gets a column, exactly once",
);
// Stable across the race: `race_order` re-sorts every time two cars swap, and
// a history grid whose columns move is a history nobody can read back.
check(
  JSON.stringify(paceHead) !== JSON.stringify(TOWER_ORDER),
  "the columns are NOT in position order, which changes under the reader",
);

const paceCells = await pacePage.evaluate(() => {
  const cells = [...document.querySelectorAll(".pace-table td")];
  const tone = (name) => cells.filter((c) => c.className === `is-${name}`).length;
  return {
    total: cells.length,
    clipped: cells.filter((c) => c.scrollWidth > c.clientWidth).length,
    best: tone("best"), t1: tone("t1"), t2: tone("t2"), t3: tone("t3"),
    pit: tone("pit"), out: tone("out"), none: tone("none"),
    pitText: cells.find((c) => c.className === "is-pit")?.textContent,
    outText: cells.find((c) => c.className === "is-out")?.textContent,
    rows: document.querySelectorAll(".pace-table tbody tr").length,
  };
});

// EFFECT, not mechanism. `overflow-x` on the container reports zero for every
// variant that clips - only the cell's own scrollWidth sees the digits cut,
// which is how a 0.27 px "fit" measured as a pass right up to the screenshot.
check(paceCells.clipped === 0, `no lap time is cut (${paceCells.clipped}/${paceCells.total} clipped)`);
check(paceCells.rows === PACE_LAPS, `one row per lap of the race (${paceCells.rows})`);
check(paceCells.pitText === "IN PIT" && paceCells.outText === "P.EXIT",
  "the in-lap and the out-lap replace the time, as a timing screen shows them");
// **And neither of them says what the tower says about a RETIRED car.** The
// tower's own docstring refuses to reuse that word for a car that is still
// racing; this grid used it for the out-lap, in the same window, on the same
// screen. Asserted over the whole enumeration of cell texts rather than on the
// one sampled above, so a single tone reverting is still caught.
const paceWords = await pacePage.evaluate(() => {
  const cells = [...document.querySelectorAll(".pace-table td")];
  const tower = [...document.querySelectorAll(".tower-row .col-last")].map((c) => c.textContent);
  return {
    collisions: cells.filter((c) => c.textContent.trim() === "OUT").length,
    towerUsesIt: tower.includes("OUT"),
  };
});
check(
  paceWords.collisions === 0,
  `no pace cell says OUT, which the tower reserves for a retirement (${paceWords.collisions} do)`,
);
check(paceCells.best === 1, `exactly one purple cell - the session's fastest lap (${paceCells.best})`);

// **The range says what is ON SCREEN, and the only way to check that is to
// SCROLL.** It used to render `grid.laps[0]`, which is always 1, so the header
// claimed `LAPS 1-57` over a panel pinned to the bottom showing 8-57. An
// assertion that the string matches the data passes straight over that; this
// one moves the panel and requires the header to move with it. The panel is
// also the one affordance replacing a scrollbar this window hides globally.
//
// The overflow is FORCED rather than borrowed (#958). This block used to lean
// on Melbourne's 57 rows overflowing the box on their own, and here they do -
// by 18 px, one and a half rows of 12. On CI's font metrics the same 57 rows
// fit, `scrollTop` never leaves 0, both reads return the same correct string,
// and the check went red on `dev` the day it shipped. A guard whose probe sits
// a row from the boundary cannot see the defect it names, so this one squeezes
// the panel to a height that guarantees an overflow and PROVES the two
// hypotheses are distinguishable before asserting which one ships.
const squeeze = await pacePage.addStyleTag({
  content: ".pace-scroll { height: 200px !important; flex: none !important; }",
});
const repin = () =>
  pacePage.evaluate(() => {
    const box = document.querySelector(".pace-scroll");
    box.scrollTop = box.scrollHeight;
    box.dispatchEvent(new Event("scroll"));
  });
await repin();
await pacePage.waitForTimeout(150);
const overflow = await pacePage.evaluate(() => {
  const box = document.querySelector(".pace-scroll");
  return {
    hidden: box.scrollHeight - box.clientHeight,
    row: box.querySelector("tbody tr")?.offsetHeight ?? 0,
  };
});
check(
  overflow.hidden > overflow.row,
  `the panel really overflows before the range is asked to follow it (${overflow.hidden} px hidden, ${overflow.row} px per row)`,
);

const rangeAtBottom = await pacePage.locator(".pace-range").innerText();
const scrolled = await pacePage.evaluate(() => {
  const box = document.querySelector(".pace-scroll");
  box.scrollTop = 0;
  box.dispatchEvent(new Event("scroll"));
  return new Promise((done) => setTimeout(() => done(document.querySelector(".pace-range").textContent), 150));
});
check(
  rangeAtBottom !== scrolled,
  `the lap range follows the scroll (bottom "${rangeAtBottom}", top "${scrolled}")`,
);
check(
  /^LAPS 1-\d+ of \d+$/.test(scrolled ?? ""),
  `scrolled to the top it starts at lap 1 (${scrolled})`,
);
check(
  rangeAtBottom.endsWith(`-${PACE_LAPS} of ${PACE_LAPS}`),
  `pinned to the bottom it ends at the newest lap, and says the race length (${rangeAtBottom})`,
);
// Drop the squeeze and put the panel back where it pins itself, so the checks
// below see the state the window actually opens in rather than a 200 px box
// this guard invented.
await squeeze.evaluate((tag) => tag.remove());
await repin();
await pacePage.waitForTimeout(150);

// A lap 40 ms under a minute boundary. Splitting the minutes off BEFORE
// rounding the tenths renders "1:60.0" - a time that does not exist, and one
// the cell regex above accepts without complaint.
const boundaryCell = await pacePage.evaluate(() => {
  const rows = [...document.querySelectorAll(".pace-table tbody tr")];
  const row = rows.find((r) => r.querySelector("th")?.textContent === "50");
  return [...(row?.querySelectorAll("td") ?? [])].map((c) => c.textContent);
});
check(
  boundaryCell.includes("2:00.0") && !boundaryCell.some((t) => /:60\./.test(t ?? "")),
  `a lap just under a minute boundary rounds up to the next minute, never :60 (${boundaryCell.filter((t) => t && t.startsWith("1:5") === false && t.includes(":")).slice(0, 3).join(",")})`,
);

// A deleted time is struck through and carries NO rank. It used to carry the
// FASTEST one: the ranking excludes it, so `indexOf` answered -1, and -1 and
// "top third" shared a branch. Measured on the real race, GAS's lap 54 was
// last of fourteen and rendered the same green as the quickest.
const deletedCells = await pacePage.evaluate(() => {
  const rows = [...document.querySelectorAll(".pace-table tbody tr")];
  const row = rows.find((r) => r.querySelector("th")?.textContent === "44");
  const cells = [...(row?.querySelectorAll("td") ?? [])];
  const struck = cells.filter((c) => c.className === "is-deleted");
  return {
    struck: struck.length,
    line: struck[0] ? getComputedStyle(struck[0]).textDecorationLine : null,
    text: struck[0]?.textContent ?? "",
    fastestToned: cells.filter((c) => c.className === "is-t1").length,
  };
});
check(deletedCells.struck > 0, `the deleted laps reach the grid (${deletedCells.struck})`);
check(
  deletedCells.line === "line-through",
  `a deleted time is struck through, as the tower already shows it (${deletedCells.line})`,
);
check(
  /^\d:\d\d\.\d$/.test(deletedCells.text),
  `and it still shows its time rather than vanishing (${deletedCells.text})`,
);

// The reason the colour ranks each lap against ITSELF. Anchored on the session
// best with fixed percentage bands, 82.4 % of the real Melbourne payload lands
// in one colour, because the race was wet and ran safety cars.
const spread = [paceCells.t1, paceCells.t2, paceCells.t3];
check(
  spread.every((count) => count > 0) && Math.max(...spread) < paceCells.total * 0.75,
  `the heat scale uses all three tones rather than painting one (${spread.join(" / ")})`,
);

// The check that actually separates the two scales, and the ONLY one that
// does - the tidy laps look the same under both. Under a neutralisation the
// whole field is bunched, so any fixed percentage band collapses it into one
// or two tones and stops discriminating at exactly the moment a strategist is
// reading the grid. Measured on a mutated copy that banded at 1.5 % / 4 %:
// 40 / 45 / 0, the slowest tone empty across all five laps. Ranking inside the
// lap splits the field whatever the spread.
const scLaps = await pacePage.evaluate(() => {
  const rows = [...document.querySelectorAll(".pace-table tbody tr")].slice(1, 6);
  const tones = {};
  for (const row of rows) {
    for (const cell of row.querySelectorAll("td")) {
      if (cell.textContent === "") continue;
      tones[cell.className] = (tones[cell.className] ?? 0) + 1;
    }
  }
  return tones;
});
const scSpread = ["is-t1", "is-t2", "is-t3"].map((k) => scLaps[k] ?? 0);
check(
  scSpread.every((count) => count > 0),
  `under the safety car the grid still ranks the field instead of painting it one colour (${scSpread.join(" / ")})`,
);

// **And it SAYS the thirds are the queue on those laps.** The ranking is left
// alone deliberately - excluding the laps would leave holes in a history panel
// and re-ranking them would invent a scale - so the rail is what stops a green
// cell on lap 4 reading as "quick" when it means "at the compressing end of the
// accordion". Measured on the real race: 22 of 57 laps, 213 of the 776 cells the
// grid ranks.
//
// Over the WHOLE enumeration, both directions: every lap the payload marks has a
// rail and no other lap does. A count would pass on a rail drawn on the wrong
// laps.
const rails = await pacePage.evaluate(() => {
  const rows = [...document.querySelectorAll(".pace-table tbody tr")];
  const railed = [];
  const plain = [];
  for (const row of rows) {
    const cell = row.querySelector("th.pace-lapcol");
    const lap = Number(cell.textContent);
    (cell.classList.contains("is-neutralised") ? railed : plain).push(lap);
  }
  return {
    railed,
    plain,
    // The rail is a border, so a text-colour assertion could not see it.
    width: railed.length
      ? getComputedStyle(rows[railed[0] - 1].querySelector("th")).borderLeftWidth
      : "0px",
    legend: document.querySelectorAll(".pace-legend-rail").length,
    title: rows[railed[0] - 1]?.querySelector("th")?.getAttribute("title") ?? "",
  };
});

check(
  JSON.stringify(rails.railed) === JSON.stringify([2, 3, 4, 5, 6, 30]),
  `the neutralised laps carry a rail and only those (${rails.railed.join(",")})`,
);
check(
  rails.plain.length === PACE_LAPS - 6 && !rails.plain.includes(4) && !rails.plain.includes(30),
  `and the other ${rails.plain.length} laps carry none`,
);
check(
  rails.width === "2px" && rails.legend === 1 && rails.title === "SAFETY CAR",
  `the rail is drawn, keyed in the header and named on hover (${rails.width}, ${rails.legend} legend, "${rails.title}")`,
);

// --- The WIDTH axis, which the check above cannot see ---------------------
//
// `paceCells.clipped === 0` is asserted at 1485 x 833, the largest client in the
// fleet, where the columns are 38.75 px and nothing clips. It passed all the way
// through the P0: on a 1080p laptop at 150 % scaling - Windows' own recommended
// scaling for a 13-14" screen - the client is 1265 x 593, the columns fall to
// 27.75 px, and 495 of 514 populated cells lost their last glyph in silence.
//
// A guard whose probe sits at the one size where the defect does not exist is
// the shape this file's own header warns about, so this block PROVES the two
// widths are distinguishable - the column really is narrower - before asserting
// that nothing clips at either.
{
  const narrowCtx = await browser.newContext({ viewport: { width: 1265, height: 593 } });
  const narrow = await narrowCtx.newPage();
  narrow.on("pageerror", (error) => failures.push(`pageerror(pace-narrow): ${error.message}`));
  await narrow.addInitScript(
    ([payload, bulk, live]) => {
      window.pywebview = { api: {
        get_tick: async (s) => (s === payload.seq ? null : payload),
        get_bulk: async (r) => (r === bulk.rev ? null : bulk),
        get_live_lap: async (r) => (r === live.rev ? null : live),
        get_connection: async () => "Connected",
      } };
    },
    [tick(1, { drivers: paceField(), order: TOWER_ORDER }), paceBulk(), towerLive()],
  );
  await narrow.goto(`http://127.0.0.1:${server.address().port}/data.html`, {
    waitUntil: "domcontentloaded",
  });
  await narrow.getByRole("tab", { name: "RACE PACE" }).click();
  await narrow.waitForSelector(".pace-table", { timeout: 5000 });
  await narrow.waitForTimeout(400);

  const narrowCells = await narrow.evaluate(() => {
    const cells = [...document.querySelectorAll(".pace-table td")];
    const populated = cells.filter((c) => c.textContent.trim().length > 0);
    return {
      column: document.querySelector(".pace-table thead th + th")?.clientWidth ?? 0,
      populated: populated.length,
      clipped: populated.filter((c) => c.scrollWidth > c.clientWidth).length,
      sample: populated.find((c) => c.className === "is-t1")?.textContent ?? "",
      pit: cells.find((c) => c.className === "is-pit")?.textContent ?? "",
      subtitle: document.querySelector(".pace-subtitle")?.textContent ?? "",
    };
  });

  check(
    narrowCells.column < 32 && narrowCells.populated > 400,
    `the narrow client really is narrower than the one above (${narrowCells.column} px per column, ${narrowCells.populated} populated cells)`,
  );
  check(
    narrowCells.clipped === 0,
    `and nothing clips there either (${narrowCells.clipped}/${narrowCells.populated} clipped)`,
  );
  check(
    /^\d:\d\d$/.test(narrowCells.sample) && narrowCells.pit === "PIT",
    `the cell coarsens instead of truncating ("${narrowCells.sample}", "${narrowCells.pit}")`,
  );
  check(
    narrowCells.subtitle.includes("to the second"),
    `and the header says the resolution changed ("${narrowCells.subtitle}")`,
  );

  // The same client's OTHER silent clip: the tower's twenty rows are fixed at
  // 437 px, so at a 510 px column the bests card's slot is 63 - and its chrome
  // alone is 79 before a single ranked row exists. It used to render the full
  // 153 px panel and lose 90 of them past the column's edge, THEORETICAL
  // included, with scrollbars hidden globally so nothing said so.
  //
  // Asserted as an EFFECT over the whole enumeration - every element of the card
  // is inside the column - rather than by checking a row count, which would pass
  // on a panel whose one remaining row still hung over the edge.
  const bests = await narrow.evaluate(() => {
    const column = document.querySelector(".left-column").getBoundingClientRect();
    const card = document.querySelector(".bests");
    const parts = [card, ...card.querySelectorAll("*")];
    const outside = parts.filter((el) => {
      const r = el.getBoundingClientRect();
      return r.height > 0 && r.bottom > column.bottom + 0.5;
    });
    return {
      outside: outside.length,
      parts: parts.length,
      worst: outside.length
        ? +(Math.max(...outside.map((el) => el.getBoundingClientRect().bottom)) - column.bottom).toFixed(1)
        : 0,
      // The card is clamped to its slot, so an overflow no longer leaves the
      // column - it becomes a scroll inside the card. Asserting BOTH is what
      // keeps "nothing outside the column" from passing on a panel that simply
      // moved the hiding one level in.
      hidden: card.scrollHeight - card.clientHeight,
      subtitle: document.querySelector(".bests-subtitle")?.textContent ?? "",
      // The theoretical lap is the one value a wall reads off this panel that no
      // other panel carries, so it is the one that must survive the degradation.
      theoretical: document.querySelector(".bests-theoretical-value")?.textContent ?? "",
      ranked: document.querySelectorAll(".bests-row").length,
      leaders: document.querySelectorAll(".bests-leader").length,
    };
  });

  check(
    bests.outside === 0 && bests.hidden === 0,
    `the bests card fits its column at the narrow client with nothing scrolled away (${bests.outside}/${bests.parts} elements over by up to ${bests.worst} px, ${bests.hidden} px hidden)`,
  );
  check(
    /^\d:\d\d\.\d\d\d$/.test(bests.theoretical),
    `and the theoretical lap survives the degradation ("${bests.theoretical}")`,
  );
  check(
    bests.ranked === 0 && bests.leaders === 6 && bests.subtitle === "leaders",
    `it degrades to the four purple holders plus THEO on one titled line, and says so (${bests.leaders} leaders, ${bests.ranked} ranked rows, "${bests.subtitle}")`,
  );
  await narrowCtx.close();
}

// --- The axis is the WHOLE RACE, not the part that has happened --------------
//
// The grid used to stop at the last revealed lap, so the table grew downward and the
// card had to be anchored to the column's bottom to hold the newest row at a stable
// height - which left a 382 px void above it for two thirds of a race, and Victor
// called it out on the shipped window. Drawing the full lap axis is the motorsport
// convention (a tyre-strategy chart plots laps 1..N and lets the stints fill in) and
// it is deliberately NOT the web's loading-skeleton idiom, which signals a fetch.
//
// The reveal is capped at the STUB, because `bridge.ts` uses `window.pywebview`
// whenever it exists and an HTTP route intercepts nothing - the lesson the BESTS
// guard above had to learn the hard way.
{
  const partialCtx = await browser.newContext({ viewport: { width: 1485, height: 833 } });
  const partial = await partialCtx.newPage();
  partial.on("pageerror", (error) => failures.push(`pageerror(skeleton): ${error.message}`));
  const REVEALED_TO = 30;
  await partial.addInitScript(
    ([payload, bulk, live, cap]) => {
      for (const driver of Object.values(bulk.drivers)) {
        driver.laps = driver.laps.filter((lap) => lap.lap <= cap);
      }
      window.pywebview = {
        api: {
          get_tick: async (s) => (s === payload.seq ? null : payload),
          get_bulk: async (r) => (r === bulk.rev ? null : bulk),
          get_live_lap: async (r) => (r === live.rev ? null : live),
          get_connection: async () => "Connected",
        },
      };
    },
    [tick(1, { drivers: paceField(), order: TOWER_ORDER }), paceBulk(), towerLive(), REVEALED_TO],
  );
  await partial.goto(`http://127.0.0.1:${server.address().port}/data.html`, {
    waitUntil: "domcontentloaded",
  });
  await partial.getByRole("tab", { name: "RACE PACE", exact: true }).click();
  await partial.waitForSelector(".pace-table", { timeout: 5000 });
  await partial.waitForTimeout(600);

  const skeleton = await partial.evaluate(() => {
    const lin = (v) => (v / 255 <= 0.03928 ? v / 255 / 12.92 : ((v / 255 + 0.055) / 1.055) ** 2.4);
    const L = ([r, g, b]) => 0.2126 * lin(r) + 0.7152 * lin(g) + 0.0722 * lin(b);
    const ratio = (a, b) => {
      const [hi, lo] = [L(a), L(b)].sort((x, y) => y - x);
      return (hi + 0.05) / (lo + 0.05);
    };
    const parse = (value) => value.match(/[0-9]+/g).slice(0, 3).map(Number);
    const rows = [...document.querySelectorAll(".pace-table tbody tr")];
    const future = rows.filter((row) => row.classList.contains("is-future"));
    const driven = rows.filter((row) => !row.classList.contains("is-future"));
    const newest = driven[driven.length - 1];
    const box = document.querySelector(".pace-scroll").getBoundingClientRect();
    const card = parse(getComputedStyle(document.querySelector(".pace")).backgroundColor);
    // Tolerant of there being NO future rows, because that is exactly the defect
    // this block exists to catch: reading `future[0]` blind turned a red check into
    // a thrown exception, and a stack trace names nothing.
    let futureRatio = null;
    if (future.length > 0) {
      const style = getComputedStyle(future[0].querySelector("th"));
      const alpha = Number(style.opacity);
      const composited = parse(style.color).map((v, i) =>
        Math.round(card[i] + alpha * (v - card[i])),
      );
      futureRatio = +ratio(composited, card).toFixed(2);
    }
    const newestBox = newest.getBoundingClientRect();
    return {
      rows: rows.length,
      driven: driven.length,
      future: future.length,
      futureWithText: future.filter((row) =>
        [...row.querySelectorAll("td")].some((cell) => cell.textContent.trim().length > 0),
      ).length,
      lastLapNumber: Number(rows[rows.length - 1].querySelector("th").textContent),
      newestLap: Number(newest.querySelector("th").textContent),
      newestVisible: newestBox.top >= box.top - 1 && newestBox.bottom <= box.bottom + 1,
      futureRatio,
    };
  });

  check(
    skeleton.rows === PACE_LAPS && skeleton.lastLapNumber === PACE_LAPS,
    `the grid draws the whole race, not the part that has run (${skeleton.rows} rows, last lap ${skeleton.lastLapNumber} of ${PACE_LAPS})`,
  );
  check(
    skeleton.driven === REVEALED_TO && skeleton.future === PACE_LAPS - REVEALED_TO,
    `and it knows which half is which (${skeleton.driven} driven, ${skeleton.future} future)`,
  );
  // The whole point of drawing them: they say how much race is left and NOTHING else.
  check(
    skeleton.futureWithText === 0 && skeleton.newestLap === REVEALED_TO,
    `a lap nobody has driven carries its number and no data (${skeleton.futureWithText} with text, newest driven ${skeleton.newestLap})`,
  );
  // Replaces what the bottom-anchored card was bought for: the row every decision is
  // about stays on screen. Asserted as visibility, not as a scroll offset - the offset
  // is 0 whenever the newest lap is still inside the first viewport.
  check(
    skeleton.newestVisible,
    `the newest driven lap is on screen with the future drawn below it (lap ${skeleton.newestLap})`,
  );
  // A future lap number MEANS something - how much race is left - so it clears the 3:1
  // floor for a meaningful graphic. The first version used `--qt-border`: 1.29:1.
  check(
    skeleton.futureRatio !== null && skeleton.futureRatio >= 3,
    `and a future lap number is legible against its card (${skeleton.futureRatio}:1)`,
  );
  await partialCtx.close();
}

// --- The client heights BETWEEN the two anybody measured ---------------------
//
// The two settled sizes cannot see this and neither could the guard above: at
// 1265x593 the room (63 px) is below even the EMPTY bests panel, so it degrades
// whatever happens, and at 1485x833 the room (303 px) is above the populated one.
// The band is room in [~115, ~151) - the empty card measures 114 px and the
// populated one 151 - and `useFitsRanked` latched the height on MOUNT, when the
// tick had landed and the BULK had not. So the panel committed to `ranked` against
// the empty measurement and then clipped, THEORETICAL included, in silence.
//
// The bulk is HELD at the stub while the card mounts, so these three sizes are
// exercised with the panel settling EMPTY first.
//
// **This guard IS the discriminator, and the paragraph that used to stand here said
// it was not.** That sentence was written while a red/green mutation was still sitting
// in the working tree, so it described a measurement taken against the wrong build;
// the guard was then driven red four times against the un-fixed latch and green four
// times with it, and the comment never caught up. A false comment claiming a guard
// cannot see its own defect is worse than no comment: it invites the next reader to
// delete a working check. Measured either way: 18 px hidden at 1265x650 and 8 px at
// 1350x660 without the fix, 0 with it, on this fixture.
for (const [width, height] of [
  [1265, 650],
  [1350, 660],
  [1350, 673],
]) {
  const ctx = await browser.newContext({ viewport: { width, height } });
  const page = await ctx.newPage();
  page.on("pageerror", (error) => failures.push(`pageerror(bests ${height}): ${error.message}`));
  // **Withheld at the STUB, not with `page.route`.** `bridge.ts` uses
  // `window.pywebview` whenever it exists and only falls back to `fetch`, so the
  // smoke's injected api is the transport and an HTTP route intercepts nothing -
  // the first version of this block held `/api/bulk` and the panel got its rows
  // immediately anyway, which made the guard pass on the defect it was written for.
  await page.addInitScript(
    ([payload, bulk, live]) => {
      window.__holdBulk = true;
      window.pywebview = { api: {
        get_tick: async (s) => (s === payload.seq ? null : payload),
        get_bulk: async (r) => (window.__holdBulk || r === bulk.rev ? null : bulk),
        get_live_lap: async (r) => (r === live.rev ? null : live),
        get_connection: async () => "Connected",
      } };
    },
    [tick(1, { drivers: paceField(), order: TOWER_ORDER }), paceBulk(), towerLive()],
  );
  await page.goto(`http://127.0.0.1:${server.address().port}/data.html`, {
    waitUntil: "domcontentloaded",
  });
  await page.waitForSelector(".bests", { timeout: 5000 });
  await page.waitForTimeout(1200);
  // The card has now mounted and settled with an EMPTY bests panel, which is the
  // 114 px measurement the latch used to keep. Release the rows.
  await page.evaluate(() => {
    window.__holdBulk = false;
  });
  await page.waitForTimeout(1200);

  const fit = await page.evaluate(() => {
    const column = document.querySelector(".left-column").getBoundingClientRect();
    const card = document.querySelector(".bests");
    const box = card.getBoundingClientRect();
    const theo = document.querySelector(".bests-theoretical-value");
    return {
      hidden: card.scrollHeight - card.clientHeight,
      over: +(box.bottom - column.bottom).toFixed(1),
      // The one value the panel's own docstring calls irreplaceable.
      theoreticalBelow: theo === null ? null : +(theo.getBoundingClientRect().bottom - column.bottom).toFixed(1),
      form: document.querySelectorAll(".bests-row").length > 0 ? "ranked" : "leaders",
    };
  });
  check(
    fit.hidden === 0 && fit.over <= 1 && fit.theoreticalBelow !== null && fit.theoreticalBelow <= 1,
    `bests fits at ${width}x${height} whichever form it picks (${fit.form}, ${fit.hidden} px hidden, card ${fit.over} px over, THEORETICAL ${fit.theoreticalBelow} px over)`,
  );
  await ctx.close();
}

// --- Band 3, second panel: the race trace ---------------------------------

await pacePage.getByRole("tab", { name: "RACE TRACE" }).click();
// Waited for and then CHECKED, rather than only waited for. The panel's empty
// state is a real render, so a defect that collapses the trace - a reference
// population bounded by a car the bulk has no laps for is one, and it caps the
// whole chart at lap zero - takes the plot off the page entirely. A bare
// `waitForSelector` turns that into a timeout and a stack trace, which names
// nothing and reads like a flaky harness; this names it.
const tracePlotted = await pacePage
  .waitForSelector(".trace-band-plot", { timeout: 5000 })
  .then(() => true)
  .catch(() => false);
check(tracePlotted, "the race trace draws a plot rather than its empty state");
await pacePage.waitForTimeout(400);

check((await pacePage.locator(".pace-table").count()) === 0, "the pace grid unmounts on the trace tab");
check((await pacePage.locator(".ring").count()) === 0, "and the ring stays hidden here too");

/** The trace's series and axes, read off the live ECharts instance. */
const traceState = () =>
  pacePage.evaluate(() => {
    const el = document.querySelector(".trace-band-plot");
    const chart = el && el.__pitwallChart;
    if (!chart) return null;
    const series = chart.getOption().series;
    const axis = (type) => chart.getModel().getComponent(type, 0).axis.scale.getExtent();
    return {
      x: axis("xAxis"),
      names: series.map((s) => s.name),
      points: Object.fromEntries(series.map((s) => [s.name, s.data])),
      labelled: series.filter((s) => s.endLabel?.show).map((s) => s.name),
      zero: document.querySelector(".pace-subtitle")?.textContent ?? "",
    };
  });

const traceLeader = await traceState();
check(traceLeader !== null, "the race trace mounts a chart");

// **The check the whole design turns on.** The reveal is per driver, so the
// newest laps are ragged: LAW has 55 and everyone else 57. A trace plotted to
// the last lap ANYBODY has would compute its reference at laps 56 and 57 over
// a population of nineteen and then eighteen, and every line would swing on
// the next reveal. It stops at the last lap ALL of them have.
check(
  traceLeader?.x?.[1] === PACE_LAPPED.laps,
  `the trace stops at the last lap the whole field has completed (${traceLeader?.x?.[1]}, expected ${PACE_LAPPED.laps})`,
);
// And the three lap-1 retirements do NOT bound it, which is the other half:
// they completed zero laps, so a population that included them would cap the
// whole trace at lap 0 and render the empty state on every real race with a
// first-lap incident. Melbourne 2025 has three.
check(
  (traceLeader?.x?.[1] ?? 0) > 1,
  "a lap-1 retirement does not collapse the trace to nothing",
);

// The two control strips, whose targets were 15 px and 22 against the ~28 a mouse
// wants - and these three buttons decide what the whole panel is measured against.
const targets = await pacePage.evaluate(() => {
  const height = (sel) => {
    const el = document.querySelector(sel);
    return el ? +el.getBoundingClientRect().height.toFixed(1) : 0;
  };
  return { tab: height(".tab"), ref: height(".ref") };
});
check(
  targets.tab >= 26 && targets.ref >= 26,
  `the tab and reference targets are big enough to hit (${targets.tab} px tab, ${targets.ref} px ref)`,
);

// **The neutralised bands, and the one-lap case a live safety car produces.** A
// lap is a POINT on this axis, so an unpadded one-lap range is `from == to` and
// ECharts paints a zero-width area: measured live at lap 35 with only lap 33
// revealed, the band was invisible exactly while the safety car was out, and its
// label floated over the NOR/PIA end labels - the chart's only identification.
const bands = await pacePage.evaluate(() => {
  const el = document.querySelector(".trace-band-plot");
  const series = el.__pitwallChart.getOption().series;
  const area = series.map((s) => s.markArea).find(Boolean);
  if (!area) return null;
  return {
    label: area.label?.position ?? "",
    ranges: area.data.map(([from, to]) => [from.xAxis, to.xAxis, from.name ?? ""]),
  };
});
if (bands === null) {
  check(false, "the race trace carries the neutralised bands");
} else {
  const widths = bands.ranges.map(([from, to]) => +(to - from).toFixed(2));
  check(
    bands.ranges.length === 2 && widths.every((width) => width >= 1),
    `both bands have real width, the one-lap one included (${JSON.stringify(widths)})`,
  );
  check(
    bands.ranges.every(([, , name]) => name === "SAFETY CAR") &&
      bands.label === "insideTopLeft",
    `each band is named and labelled at its left edge (${bands.label}, ${JSON.stringify(bands.ranges.map((r) => r[2]))})`,
  );
}

// The distance axis' ticks, which the four telemetry charts share. At 1265x593 a
// plot is 152 px and the `1,000`-style labels are ~150 px of glyphs on ~120 px of
// axis: they rendered as one unbroken digit string on the tab the window OPENS on.
// Measured as WIDTH against the axis, in the axis' own font, not as a string
// comparison - a formatter that shortened the labels and still did not fit would
// pass a string test.
{
  const narrowCtx = await browser.newContext({ viewport: { width: 1265, height: 593 } });
  const narrow = await narrowCtx.newPage();
  narrow.on("pageerror", (error) => failures.push(`pageerror(axis): ${error.message}`));
  await narrow.addInitScript(
    ([payload, bulk, live]) => {
      window.pywebview = { api: {
        get_tick: async (s) => (s === payload.seq ? null : payload),
        get_bulk: async (r) => (r === bulk.rev ? null : bulk),
        get_live_lap: async (r) => (r === live.rev ? null : live),
        get_connection: async () => "Connected",
      } };
    },
    [tick(1, { drivers: paceField(), order: TOWER_ORDER }), paceBulk(), towerLive()],
  );
  await narrow.goto(`http://127.0.0.1:${server.address().port}/data.html`, {
    waitUntil: "domcontentloaded",
  });
  await narrow.waitForSelector(".trace-stack-plot canvas", { timeout: 5000 });
  await narrow.waitForTimeout(500);

  const axis = await narrow.evaluate(() => {
    const el = document.querySelector(".trace-stack-plot");
    const chart = el.__pitwallChart;
    const option = chart.getOption();
    const x = option.xAxis[0];
    const ticks = chart
      .getModel()
      .getComponent("xAxis", 0)
      .axis.scale.getTicks()
      .map((entry) => (typeof entry === "object" ? entry.value : entry));
    const format = x.axisLabel.formatter;
    // The bounds are not labelled on a locked axis (`valueAxis`), so they are not
    // glyphs on the axis and must not be counted as if they were.
    const shown = x.axisLabel.showMinLabel === false ? ticks.slice(1, -1) : ticks;
    const labels = shown.map((value) => (format ? format(value) : String(value)));
    const ruler = document.createElement("canvas").getContext("2d");
    ruler.font = `${x.axisLabel.fontSize}px ${getComputedStyle(document.body).fontFamily}`;
    const glyphs = labels.reduce((total, text) => total + ruler.measureText(text).width, 0);
    const span = el.clientWidth - (option.grid[0].left + option.grid[0].right);
    return { labels, glyphs: +glyphs.toFixed(1), span, plot: el.clientWidth };
  });
  check(
    axis.glyphs < axis.span,
    `the distance ticks fit their axis at the narrow client (${axis.labels.join(" ")} = ${axis.glyphs} px on ${axis.span} px of a ${axis.plot} px plot)`,
  );
  await narrowCtx.close();
}

// --- The producer dies, and the window has to stop looking live -------------
//
// `state-dead.png` is why: the board held a full set of confident numbers, the lap
// counter still said L 28/57, the track chip still asserted GREEN and PLAYBACK
// still said 2x, with one 77 x 18 chip and a status bar that had quietly gone
// BLANK as the only tells. The socket label is the only thing that still moves
// once the ticks stop, so the state is known client-side.
//
// Asserted as the EFFECT on the four surfaces that used to lie, and the fixture
// feeds real ticks first so the board is populated before the feed goes quiet -
// a window that never had data cannot demonstrate a window whose data went stale.
{
  const deadCtx = await browser.newContext({ viewport: { width: 1485, height: 833 } });
  const dead = await deadCtx.newPage();
  dead.on("pageerror", (error) => failures.push(`pageerror(dead): ${error.message}`));
  await dead.addInitScript(
    ([payload, bulk, live]) => {
      window.__alive = true;
      window.pywebview = { api: {
        get_tick: async (s) => (s === payload.seq ? null : payload),
        get_bulk: async (r) => (r === bulk.rev ? null : bulk),
        get_live_lap: async (r) => (r === live.rev ? null : live),
        get_connection: async () => (window.__alive ? "Connected" : "Disconnected"),
      } };
    },
    [tick(1, { drivers: paceField(), order: TOWER_ORDER }), paceBulk(), towerLive()],
  );
  await dead.goto(`http://127.0.0.1:${server.address().port}/data.html`, {
    waitUntil: "domcontentloaded",
  });
  await dead.waitForSelector(".tower-row", { timeout: 5000 });
  await dead.waitForTimeout(600);

  const alive = await dead.evaluate(() => ({
    playback: [...document.querySelectorAll(".strip-field")]
      .find((f) => f.textContent.startsWith("PLAYBACK"))
      ?.querySelector(".strip-field-value")?.textContent,
    bar: document.querySelector(".status-bar")?.textContent ?? "",
    frozenChip: document.querySelectorAll(".strip-chip.is-frozen").length,
    filter: getComputedStyle(document.querySelector(".data-main")).filter,
    rows: document.querySelectorAll(".tower-row").length,
  }));
  check(
    alive.rows === 20 && /^\d+x$/.test(alive.playback ?? "") && alive.bar.includes("live"),
    `the board is live and says so before the producer dies (${alive.rows} rows, ${alive.playback}, "${alive.bar}")`,
  );

  // The producer goes quiet. The connection poll is 1 Hz, so this waits for it.
  await dead.evaluate(() => {
    window.__alive = false;
  });
  await dead.waitForTimeout(2200);

  const frozen = await dead.evaluate(() => ({
    playback: [...document.querySelectorAll(".strip-field")]
      .find((f) => f.textContent.startsWith("PLAYBACK"))
      ?.querySelector(".strip-field-value")?.textContent,
    bar: document.querySelector(".status-bar")?.textContent ?? "",
    frozenChip: document.querySelectorAll(".strip-chip.is-frozen").length,
    trackFilled: document.querySelectorAll(".strip-chip.is-filled").length,
    filter: getComputedStyle(document.querySelector(".data-main")).filter,
    rows: document.querySelectorAll(".tower-row").length,
    lost: document.querySelectorAll(".strip-chip.is-lost").length,
  }));

  check(
    frozen.playback === "—" && !frozen.bar.includes("live") && frozen.bar.includes("FROZEN"),
    `nothing on the strip claims the replay is advancing (${frozen.playback}, "${frozen.bar}")`,
  );
  check(
    frozen.frozenChip === 1 && frozen.lost === 1 && frozen.trackFilled === 0,
    `the frozen state is named in the header and the track status gives up its weight (${frozen.frozenChip} frozen, ${frozen.lost} lost, ${frozen.trackFilled} filled)`,
  );
  // The board is treated, and STILL THERE. The last known state is the best
  // information a strategist has; a treatment that hid it would be worse than
  // the lie it replaces.
  check(
    frozen.filter !== "none" && frozen.rows === 20,
    `the board reads as history without becoming unreadable (${frozen.filter}, ${frozen.rows} rows)`,
  );
  // And the bar does not go blank. Its 1.5 s auto-clear is what left the window
  // with no message at all, which is how a dead feed looked like a quiet one.
  await dead.waitForTimeout(2000);
  const later = await dead.evaluate(() => document.querySelector(".status-bar")?.textContent ?? "");
  check(
    later.includes("FROZEN"),
    `and it is still saying so four seconds later, not auto-cleared ("${later}")`,
  );
  await deadCtx.close();
}

// A retired car keeps the laps he drove and gets NO point for the ones he did
// not. These three have no crossings at all, so their series must be EMPTY -
// never a flat line at zero, which is what a missing crossing read as a
// default would draw, and which reads as a car circulating on the leader's
// pace. That is the sentinel-collision class this repo has paid for twice.
const retiredPoints = RETIRED_CODES.map((code) => traceLeader?.points[code]?.length ?? -1);
check(
  retiredPoints.every((count) => count === 0),
  `a car with no crossings draws nothing rather than a flat line at zero (${retiredPoints.join(",")})`,
);
check(
  RETIRED_CODES.every((code) => !(traceLeader?.labelled ?? []).includes(code)),
  "and it is not labelled at the right-hand edge either",
);

// A pit stop is a STEP, and this is measured rather than asserted about the
// mechanism. NOR is the fastest car in the fixture and pits on lap 20-21, each
// of those laps carrying the stop's +22 s. Against the leader he therefore
// falls about 44 s in two laps.
const norLine = Object.fromEntries((traceLeader?.points.NOR ?? []).map(([lap, y]) => [lap, y]));
const step = norLine[21] - norLine[19];
check(
  step <= -30,
  `a pit stop reads as a step down of the stop's own length (${step?.toFixed(1)} s over laps 19-21)`,
);

// **The labels must not land on each other, and this is measured on the
// RENDERED boxes.** An ECharts label is canvas text, so no selector and no
// axis extent can see it - which is why a real overlap (ALB over HAM by 4.5 px
// of a 9 px label) survived every other check on this page and was found by
// looking at a screenshot. This fixture is harsher than the real race: four
// pairs of drivers carry byte-identical cumulative times, so without the
// de-collision their codes render exactly on top of one another.
const labelBoxes = await pacePage.evaluate(() => {
  const chart = document.querySelector(".trace-band-plot")?.__pitwallChart;
  if (!chart) return { count: 0, overlaps: ["no chart"] };
  const boxes = chart
    .getZr()
    .storage.getDisplayList()
    .filter((el) => /^[A-Z]{3}$/.test(el.style?.text ?? ""))
    .map((el) => {
      const rect = el.getBoundingRect().clone();
      rect.applyTransform(el.transform);
      return { code: el.style.text, x: rect.x, y: rect.y, w: rect.width, h: rect.height };
    });
  const overlaps = [];
  for (let i = 0; i < boxes.length; i += 1) {
    for (let j = i + 1; j < boxes.length; j += 1) {
      const a = boxes[i];
      const b = boxes[j];
      const dx = Math.min(a.x + a.w, b.x + b.w) - Math.max(a.x, b.x);
      const dy = Math.min(a.y + a.h, b.y + b.h) - Math.max(a.y, b.y);
      if (dx > 0 && dy > 0) overlaps.push(`${a.code}/${b.code} ${dx.toFixed(1)}x${dy.toFixed(1)}`);
    }
  }
  return { count: boxes.length, overlaps, right: Math.max(...boxes.map((z) => z.x + z.w)),
           canvas: chart.getWidth() };
});
check(
  labelBoxes.overlaps.length === 0,
  `no two driver codes are drawn on top of each other (${labelBoxes.overlaps.join(", ")})`,
);
// And the right-hand margin actually holds them. Measured once at 52 px of
// grid: a label's right edge landed at 804 px on an 803 px canvas.
check(
  labelBoxes.right <= labelBoxes.canvas,
  `no label is clipped by the canvas edge (${labelBoxes.right?.toFixed(1)} of ${labelBoxes.canvas})`,
);

// Twenty lines need twenty identities and a legend for twenty codes eats the
// plot. Each line says its own name at its right-hand end.
// Sixteen, not seventeen: three cars retired on lap 1 and a fourth is in
// `race_order` while the bulk never names him. That fourth is the case that
// used to DELETE the panel - a minimum bounded by a car with no lap data reads
// his lap count as zero and caps the whole trace at lap zero - and the check
// above (x max = 55) is what catches it. This one pins the count itself.
check(
  traceLeader?.labelled?.length === 16,
  `every car the bulk has laps for is labelled at its end (${traceLeader?.labelled?.length}, expected 16)`,
);
check(
  (traceLeader?.points[TOWER_ORDER[9]] ?? []).length === 0,
  "a driver the bulk does not name draws no line, and does not bound the trace either",
);

// The reference switches, and the switch is the panel's whole control surface.
// With OUR car as the zero line every one of its own points is exactly zero -
// which is what "flat at zero" has to mean, and what a reference computed off
// a different clock would miss by milliseconds.
if (tracePlotted) await pacePage.getByRole("tab", { name: "NOR" }).click();
await pacePage.waitForTimeout(300);
const traceOwn = await traceState();
check(
  (traceOwn?.points.NOR ?? []).every(([, y]) => y === 0),
  "with OWN as the reference our own line is flat at exactly zero",
);
check(
  (traceOwn?.zero ?? "").includes("NOR"),
  `and the header says what the zero line is (${traceOwn?.zero})`,
);
// The other lines are NOT all zero, which is the guard that separates a real
// reference switch from a chart that quietly zeroed everything.
check(
  (traceOwn?.points.VER ?? []).some(([, y]) => y !== 0),
  "while the rest of the field is measured against it",
);

// FIELD is a third distinct answer, not a relabelled LEADER. Against the mean
// the leaders sit ABOVE the axis; against the leader nothing can.
if (tracePlotted) await pacePage.getByRole("tab", { name: "FIELD" }).click();
await pacePage.waitForTimeout(300);
const traceField = await traceState();
const aboveField = Object.values(traceField?.points ?? {}).flat().filter(([, y]) => y > 0).length;
const aboveLeader = Object.values(traceLeader?.points ?? {}).flat().filter(([, y]) => y > 0).length;
check(
  aboveLeader === 0,
  `nothing is ahead of the car leading the lap (${aboveLeader} points above zero)`,
);
check(
  aboveField > 0,
  `against the field average the quick cars sit above the axis (${aboveField} points)`,
);

// A trace that stops and a race that ended are the same pixels, so the bound
// says how far behind the race it sits. One car with a mid-race telemetry
// dropout pins it silently otherwise (OBS-4).
const traceRange = await pacePage.locator(".trace-band .pace-range").innerText();
check(
  traceRange.includes(`of ${PACE_LAPS}`),
  `the trace says how far behind the race its bound sits (${traceRange})`,
);

// **A car that RACED and then stopped stays in the reference for the laps he
// drove.** The reference used to be averaged over the CURRENT population, so
// the moment a car retired he left it and every point of every line - back to
// lap 1 - was recomputed without him: measured on the real payload, all 45 of
// NOR's historical points moved by up to 7.6 s at one retirement. It is the
// twin of the lap-axis bound this module already had. Computed here from the
// fixture's own bulk rather than asserted about the mechanism.
const historyStable = await pacePage.evaluate(
  ([bulk, probeLap, code, retired]) => {
    const chart = document.querySelector(".trace-band-plot")?.__pitwallChart;
    if (!chart) return null;
    const mean = (entries) =>
      entries.reduce((sum, v) => sum + v, 0) / entries.length;
    const withCrossing = (codes) =>
      codes
        .map((c) => bulk.drivers[c]?.crossings[probeLap])
        .filter((v) => v !== undefined);
    const all = Object.keys(bulk.drivers);
    const own = bulk.drivers[code].crossings[probeLap];
    const series = chart.getOption().series.find((x) => x.name === code);
    const point = series?.data.find(([lap]) => lap === probeLap);
    return {
      cars: withCrossing(all).length,
      // Everyone who completed this lap, retired-since or not.
      everyone: mean(withCrossing(all)) - own,
      // The refuted alternative: only the cars still classified NOW.
      stillRacing: mean(withCrossing(all.filter((c) => c !== retired))) - own,
      actual: point ? point[1] : null,
    };
  },
  [paceBulk(), 10, TOWER_ORDER[0], PACE_RETIRED_MIDRACE.code],
);

// The two hypotheses must be TELLABLE APART on this fixture, or the assertion
// below is decoration. The first version of this check chose a probe driver
// sitting exactly on the field mean, so removing him moved nothing and the
// guard stayed green against the very defect it names.
check(
  historyStable !== null &&
    Math.abs(historyStable.everyone - historyStable.stillRacing) > 0.5,
  `the fixture can tell the two reference populations apart (${(historyStable?.everyone - historyStable?.stillRacing)?.toFixed(3)} s)`,
);
check(
  historyStable !== null && Math.abs(historyStable.everyone - historyStable.actual) < 1e-9,
  `a retired car still counts in the laps he drove (everyone ${historyStable?.everyone?.toFixed(3)}, still-racing ${historyStable?.stillRacing?.toFixed(3)}, rendered ${historyStable?.actual?.toFixed(3)}, over ${historyStable?.cars} cars)`,
);

await paceCtx.close();

await browser.close();
server.close();

if (failures.length) {
  console.error(`smoke-data FAILED (${failures.length}):`);
  for (const failure of failures) console.error(`  - ${failure}`);
  process.exit(1);
}
console.log(`smoke-data OK: ${checks} checks`);
