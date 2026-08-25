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
import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { chromium } from "@playwright/test";
import { serveDist } from "./serve-dist.mjs";
import { watchPage } from "./page-guard.mjs";
import ts from "typescript";
import { staysStill } from "./settle.mjs";

const UI_DIR = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const DIST = resolve(process.argv[2] ?? resolve(UI_DIR, "dist"));

/** The lane table's own module, read so the palette lives in exactly one file. */
const TRACE_STACK_SOURCE = resolve(UI_DIR, "src/features/data/TraceStack.tsx");

const CIRCUIT_M = 5220;

/**
 * A span the delta chart can be checked by hand.
 *
 * The rival covers 0-200 m and the main covers 0-500, so three of the six main
 * samples have a rival time to interpolate against and three do not.
 *
 * **Both cars start at dist 0 and at DIFFERENT times, and they run at different
 * PACE, and all three of those are load-bearing** (#1066):
 *
 * - starting at 0 is what a real buffer does. Measured over the Melbourne
 *   capture, 24 of 25 line crossings put a sample at `dist == 0.0` exactly, so a
 *   fixture opening at 100 m models a window mid-warm-up rather than the normal
 *   state, and it makes the header's off-the-line note fire on every scenario.
 * - starting at different times, the main at t=10.0 and the rival at t=12.0, is
 *   what lets the anchor subtraction be OBSERVED. With a shared lap start the
 *   value at the anchor is already 0 and removing the subtraction changes
 *   nothing.
 * - differing in pace is what lets the VALUES be observed. The rival was a flat
 *   +2.0 s behind at identical pace, and a re-based delta reports that, quite
 *   correctly, as 0.0 everywhere - so every value assertion in this file went
 *   invariant at once and could no longer tell one car from another.
 *
 * The rival loses 0.2 s per 100 m: main advances 1 s per 100 m, rival 1.2 s. So
 * the delta reads 0.0, +0.2, +0.4 and the on-track gap the old fixture pinned,
 * +2.0, is exactly what the subtraction removes.
 */
const MAIN_SPAN = [0, 100, 200, 300, 400, 500].map((dist, i) => ({
  lap: 24,
  t: 10 + i,
  dist,
  speed: 200 + i * 10,
  throttle: 50 + i,
  brake: i === 0 ? 100 : 0,
  gear: 6,
  drs: 8,
}));
const RIVAL_SPAN = [0, 100, 200].map((dist, i) => ({
  lap: 24,
  t: 12 + i * 1.2,
  dist,
  speed: 190 + i * 10,
  throttle: 40 + i,
  brake: 0,
  gear: 6,
  drs: 8,
}));
/** What the delta lane must read, worked out by hand from the two spans above. */
const EXPECTED_DELTA = [
  [0, 0],
  [100, 0.2],
  [200, 0.4],
];
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

/**
 * Which span a fixture driver carries: the main's, the rival's, or none.
 *
 * The real producer sends a real span for every car including retirements,
 * which keep a full-length frame array and simply stop moving. The fixture
 * leaves the rest of the field empty because no scenario here asserts on a
 * third car's trace; a scenario that starts to will have to fill it.
 */
function spanFor(code, rival, main, rivalSpan) {
  if (code === "NOR") return main;
  if (code === rival) return rivalSpan;
  return [];
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
    // Per-driver spans, verbatim. `main`/`rivalSpan` cover the two-car scenarios
    // this harness was built for; a scenario about WHICH car is charted needs to
    // fill a third, so it hands the whole map instead.
    spans = null,
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
        colors ??
        Object.fromEntries(
          Object.keys(field).map((code) => [code, [255, 128, 0]]),
        ),
      track_status: "1",
      // Decoded by the producer, never by the renderer. The pair is null
      // together when the loader has no entry for the lap, which band 1 must
      // render as unknown rather than as a green track.
      track_status_label: "GREEN",
      track_status_color: [16, 185, 129],
      // Schema v2 keys the spans by driver code, exactly like the `drivers`
      // block above (#1048). The harness still takes `main` and `rivalSpan`,
      // because that is what a DATA scenario is about, and files them under
      // the codes the window looks them up by.
      telemetry: {
        drivers:
          spans ??
          Object.fromEntries(
            Object.keys(field).map((code) => [
              code,
              spanFor(code, rival, main, rivalSpan),
            ]),
          ),
        rewound,
        dropped,
      },
    },
    playback: {
      speed: 1,
      paused: false,
      frame_index: 1000 + seq,
      total_frames: 154173,
    },
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

// The client area the product really hands this page, NOT the `WindowSpec`
// size. `place()` opens DATA at 1500x870 on the reference desktop and the OS
// keeps 14 px of frame and 37 px of title bar, so the page gets 1486x833.
// Most scenarios below already used the real client; scenario A used the outer
// size and was therefore measuring a surface 117 px taller than the window.
// `tests/surfaces/test_pitwall_host.py` now refuses a viewport larger than it.
const CLIENT = { width: 1486, height: 833 };

const server = await serveDist(DIST);
const browser = await chromium.launch();
const url = `http://127.0.0.1:${server.address().port}/data.html`;

// --- Scenario A: two drivers, a real span -----------------------------------

const ctx = await browser.newContext({
  viewport: CLIENT,
});
const page = await ctx.newPage();
watchPage(page, failures);

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
      get_connection: async () => ({ label: "Connected", colour: "#10b981" }),
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
check(
  (await page.locator(".trace-cursor").count()) === 1,
  "and ONE cursor across all six",
);
check(
  (await page.locator(".traces-lap").innerText()).trim() === "LAP 24",
  "the header carries the lap and nothing else",
);
check(
  (await page.locator(".driver-chip").count()) === 2,
  "main and rival chips",
);
check(
  (await page.locator(".trace-tier").first().innerText()).trim() ===
    "BROADCAST",
  "the rival chip is labelled broadcast tier",
);

// --- Band 1: the status strip -----------------------------------------------

check(
  (await page.locator(".strip-lap").innerText()).replace(/\s+/g, "") ===
    "L24/57",
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
  (await page.locator(".strip-field-value").first().innerText()).trim() ===
    "0:23:20",
  "the session clock is SessionTime and not replay seconds",
);
// The chip no longer has a state class: the colour arrives WITH the word from
// the host's own map, because the two windows used to map the same three
// states separately and disagreed about "Connecting...". So this asserts the
// pair - the word, and that it is not wearing the neutral grey an unknown gets.
const connectionChip = await page.evaluate(() => {
  const chip = [...document.querySelectorAll(".strip-chip")].find((el) =>
    ["Connected", "Connecting...", "Disconnected"].includes(el.innerText.trim()),
  );
  return chip ? { text: chip.innerText.trim(), colour: getComputedStyle(chip).color } : null;
});
check(
  connectionChip?.text === "Connected" && connectionChip.colour === "rgb(16, 185, 129)",
  `the connection comes from the host's socket, word and colour (${JSON.stringify(connectionChip)})`,
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
    cursorHeight: parseFloat(
      getComputedStyle(document.querySelector(".trace-cursor")).height,
    ),
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
  lanes !== null &&
    new Set(lanes.lefts).size === 1 &&
    new Set(lanes.rights).size === 1,
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

// The delta the interpolation produced. Three points, not six: the rival's
// samples stop at 200 m and `lerpSorted` returns null past the end rather
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
  delta.length === EXPECTED_DELTA.length &&
    delta.every(
      ([x, value], i) =>
        x === EXPECTED_DELTA[i][0] && Math.abs(value - EXPECTED_DELTA[i][1]) < 1e-9,
    ),
  `the delta is lap-relative: 0.0, +0.2, +0.4 over the rival's three samples only (${JSON.stringify(delta)})`,
);
// The anchor, stated as its own assertion rather than left implicit in the list
// above. The raw difference at x=0 is the rival's +2.0 s on-track gap; the series
// starts at exactly 0 because that gap is subtracted out. Removing the
// subtraction turns this list into 2.0, 2.2, 2.4, and it is this check that says
// which of those two the lane is drawing.
check(
  delta.length > 0 && delta[0][1] === 0,
  `and it is anchored: the first point is exactly 0 (${JSON.stringify(delta[0])})`,
);
// The gap is GONE from the lane, deliberately (#1066). Asserted as an absence
// because "the values are lap-relative" and "the values are the gap" are two
// claims and the first does not imply the second on a fixture where they could
// coincide.
check(
  !delta.some(([, value]) => Math.abs(value - 2) < 1e-9),
  `and the on-track gap of +2.0 s appears nowhere in it (${JSON.stringify(delta)})`,
);

// The speed trace carries the whole main span. Derived from the fixture rather
// than typed, so adding a sample to the span cannot leave this asserting a stale
// count that happens to still pass.
const speedPoints = (await laneSeries("speed", "main")).data.length;
check(
  speedPoints === MAIN_SPAN.length,
  `the speed trace holds all ${MAIN_SPAN.length} samples (${speedPoints})`,
);

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
  for (let i = 0; i < names.length; i += 2)
    pairs.push(`${names[i]}>${names[i + 1]}`);
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
// **Which lane wears which colour, as the RENDERED series style.**
//
// An adversarial gate swapped `colour: SUCCESS` and `colour: DANGER` between the
// throttle and brake entries of the LANES table - throttle painted red, brake green -
// rebuilt, and every check here plus the whole token suite stayed green. The token test
// reads the `const NAME = "#hex"` declarations, so both hexes keep their values and only
// the ASSIGNMENT flips; the one colour-sensitive probe in this file scans the delta
// lane's blue baseline, which the swap never touches. That is "membership instead of the
// slot", the failure this repo's own token test warns about in its docstring.
//
// The expected hexes are READ FROM THE SOURCE rather than typed here, so this file does
// not become a second home for the palette: what is asserted is the MAP from lane to
// constant, which is exactly what a swapped assignment breaks.
const declared = Object.fromEntries(
  [
    ...readFileSync(TRACE_STACK_SOURCE, "utf8").matchAll(
      /^const (\w+) = "(#[0-9a-fA-F]{6})";/gm,
    ),
  ].map((match) => [match[1], match[2].toLowerCase()]),
);
const laneColours = await page.evaluate(() => {
  const el = document.querySelector(".trace-stack-plot");
  return Object.fromEntries(
    el.__pitwallChart.getOption().series.map((s) => {
      // The delta lane's own car IS the baseline, so it is an empty series carrying a
      // markLine and its colour lives there. Read the lane's colour wherever the lane
      // puts it, which is the rendered EFFECT either way.
      const colour =
        s.lineStyle?.color ?? s.markLine?.data?.[0]?.lineStyle?.color ?? "";
      return [s.name, String(colour).toLowerCase()];
    }),
  );
});
const laneToConstant = {
  "speed-main": "INFO",
  "delta-main": "INFO",
  "throttle-main": "SUCCESS",
  "brake-main": "DANGER",
  "gear-main": "ACCENT",
  "drs-main": "INFO",
};
const wrong = Object.entries(laneToConstant).filter(
  ([series, name]) => laneColours[series] !== declared[name],
);
check(
  wrong.length === 0,
  `each lane paints its own channel's colour (${
    wrong
      .map(
        ([series, name]) =>
          `${series} is ${laneColours[series]}, ${name} is ${declared[name]}`,
      )
      .join("; ") || "all six correct"
  })`,
);
// The two percentage lanes share a RANGE by coincidence and must never share a colour:
// they are the pair the swap was performed on, and the pair a reader tells apart by hue.
check(
  laneColours["throttle-main"] !== laneColours["brake-main"],
  `throttle and brake are told apart by colour (${laneColours["throttle-main"]} vs ${laneColours["brake-main"]})`,
);
// And every rival trace is the RIVAL'S OWN team colour, on all six lanes.
//
// This used to read `declared.RIVAL`, a fixed palette.WARNING amber parsed out of
// `TraceStack`'s constants. #1070 deleted that constant: the rival takes the
// pinned driver's colour off the wire now. The expected value therefore comes
// from the FIXTURE's `driver_colors`, keyed by the car being charted, so the
// assertion cannot drift back to a literal.
const RIVAL_RGB = "rgb(255, 128, 0)";
const rivalWrong = Object.entries(laneColours).filter(
  ([series, colour]) => series.endsWith("-rival") && colour !== RIVAL_RGB,
);
check(
  rivalWrong.length === 0,
  `and every rival trace is the rival's own colour (${rivalWrong.map(([s, c]) => `${s}=${c}`).join("; ") || "all six"})`,
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
    const stack = document
      .querySelector(".trace-stack")
      .getBoundingClientRect();
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
await page.evaluate(
  (payload) => window.__ticks.push(payload),
  tick(2, { main: [], rivalSpan: [], rewound: true }),
);
await page.waitForTimeout(400);
const afterRewind = (await laneSeries("speed", "main")).data.length;
check(
  afterRewind === 0,
  `a rewind empties the trace (${afterRewind} points left)`,
);

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
await page.evaluate(
  (payload) => window.__ticks.push(payload),
  tick(4, { rival: null, rivalSpan: [] }),
);
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

const solo = await browser.newContext({
  viewport: CLIENT,
});
const soloPage = await solo.newPage();
watchPage(soloPage, failures, "solo");

await soloPage.addInitScript(
  (payload) => {
    window.pywebview = {
      api: {
        get_tick: async (sinceSeq) =>
          sinceSeq === payload.seq ? null : payload,
        get_bulk: async () => null,
        get_live_lap: async () => null,
        get_connection: async () => ({ label: "Connected", colour: "#10b981" }),
      },
    };
  },
  tick(1, {
    rival: null,
    rivalSpan: [],
    mainDriver: { has_position: false, rel_dist: null },
  }),
);

await soloPage.goto(url, { waitUntil: "domcontentloaded" });
await soloPage.waitForSelector(".trace-lane-label", { timeout: 5000 });
await soloPage.waitForTimeout(400);

check(
  (
    await soloPage
      .locator(".trace-lane-caption, .trace-placeholder")
      .innerText()
  ).trim() === "single-driver mode",
  "the delta chart collapses to its placeholder",
);
check(
  (await soloPage.locator(".trace-stack-plot canvas").count()) === 1,
  "three canvases, because the placeholder REPLACES the delta plot",
);
check((await soloPage.locator(".driver-chip").count()) === 1, "no rival chip");
check(
  (await soloPage.locator(".trace-tier").count()) === 0,
  "and no broadcast-tier label",
);
// #856: the note names the blind car, and it must fire for the MAIN driver
// too - the first version of it read the rival alone.
check(
  (await soloPage.locator(".traces-lap").innerText()).includes(
    "NO POSITION DATA (NOR)",
  ),
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
  SAI: driver({
    rel_dist: null,
    has_position: false,
    active: false,
    has_finished: false,
  }),
};

const ring = await browser.newContext({
  viewport: CLIENT,
});
const ringPage = await ring.newPage();
watchPage(ringPage, failures, "ring");

await ringPage.addInitScript(
  (payload) => {
    window.pywebview = {
      api: {
        get_tick: async (sinceSeq) =>
          sinceSeq === payload.seq ? null : payload,
        // The other three the DATA window polls. Without them the bridge
        // falls back to `fetch("/api/...")` and the static server answers
        // 404 four times, which nothing on this page was listening for.
        get_bulk: async () => null,
        get_live_lap: async () => null,
        get_connection: async () => ({ label: "Connected", colour: "#10b981" }),
      },
    };
  },
  tick(1, {
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
  }),
);

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

check(
  placed.length === 4,
  `four dots for five cars, one of them unplaced (${placed.length})`,
);
check(of("NOR")?.status === "running", "a car on track is running");
check(
  of("HUL")?.status === "out" && of("HUL")?.hollow,
  "a retirement is out, and hollow",
);
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
  Math.abs(of("NOR").cx - CENTRE) < 0.5 &&
    Math.abs(of("NOR").cy - (CENTRE - RADIUS)) < 0.5,
  `fraction 0 sits at the start line, top centre (${of("NOR").cx}, ${of("NOR").cy})`,
);
check(
  Math.abs(of("PIA").cx - (CENTRE + RADIUS)) < 0.5 &&
    Math.abs(of("PIA").cy - CENTRE) < 0.5,
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
  const lin = (c) =>
    c / 255 <= 0.03928 ? c / 255 / 12.92 : ((c / 255 + 0.055) / 1.055) ** 2.4;
  const L = ([r, g, b]) => 0.2126 * lin(r) + 0.7152 * lin(g) + 0.0722 * lin(b);
  const ratio = (a, b) => {
    const [hi, lo] = [L(a), L(b)].sort((x, y) => y - x);
    return (hi + 0.05) / (lo + 0.05);
  };
  const parse = (value) => value.match(/\d+/g).slice(0, 3).map(Number);
  const card = parse(
    getComputedStyle(document.querySelector(".tower")).backgroundColor,
  );
  // `td` only: the header shares the class and is a column label, not a code.
  const cells = [...document.querySelectorAll("td.col-drv")];
  return {
    cells: cells.length,
    swatches: cells.filter((cell) => cell.querySelector(".drv-swatch")).length,
    // Every code, whatever its team colour: they all read against the card now.
    worstCode: Math.min(
      ...cells.map((cell) => ratio(parse(getComputedStyle(cell).color), card)),
    ),
    // And no code is painted in a team colour any more.
    tinted: cells.filter((cell) => {
      const own = getComputedStyle(cell).color;
      const swatch = cell.querySelector(".drv-swatch");
      return (
        swatch !== null && own === getComputedStyle(swatch).backgroundColor
      );
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
check(
  lapText?.trim() === "24",
  `the ring carries the lap counter (${lapText})`,
);

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
const stillCtx = await browser.newContext({
  viewport: CLIENT,
});
const stillPage = await stillCtx.newPage();
watchPage(stillPage, failures, "still");
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
      get_connection: async () => ({ label: "Connected", colour: "#10b981" }),
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
const RETIRED = {
  laps_completed: 0,
  progress: 0.4,
  active: false,
  has_finished: false,
};

async function provisionalChips(field) {
  const context = await browser.newContext({
    viewport: CLIENT,
  });
  const scenarioPage = await context.newPage();
  watchPage(scenarioPage, failures, "provisional");
  await scenarioPage.addInitScript(
    (payload) => {
      window.pywebview = {
        api: {
          get_tick: async (sinceSeq) =>
            sinceSeq === payload.seq ? null : payload,
          get_bulk: async () => null,
          get_live_lap: async () => null,
          get_connection: async () => ({ label: "Connected", colour: "#10b981" }),
        },
      };
    },
    tick(1, { drivers: field }),
  );
  await scenarioPage.goto(url, { waitUntil: "domcontentloaded" });
  await scenarioPage.waitForSelector(".status-strip", { timeout: 5000 });
  await scenarioPage.waitForTimeout(300);
  const count = await scenarioPage
    .locator(".strip-chip.is-provisional")
    .count();
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
  "NOR",
  "PIA",
  "VER",
  "RUS",
  "LEC",
  "TSU",
  "ALB",
  "HAM",
  "GAS",
  "ALO",
  "ANT",
  "STR",
  "HUL",
  "BOR",
  "LAW",
  "OCO",
  "BEA",
  "SAI",
  "DOO",
  "HAD",
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
    field[code] = driver({
      laps_completed: code === "LAW" ? 22 : 23,
      progress,
    });
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
  TOWER_ORDER.filter((code) => !RETIRED_CODES.includes(code)).forEach(
    (code, index) => {
      const revealed = code === "LAW" ? 22 : 23;
      const crossings = {};
      for (let lap = 1; lap <= revealed; lap += 1) {
        crossings[lap] =
          CROSSING_AT_23[code] ?? 2070 + 5 * index - (23 - lap) * 90;
      }
      if (CROSSING_AT_23[code] !== undefined)
        crossings[revealed] = CROSSING_AT_23[code];
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
    },
  );
  return {
    rev: 1,
    available: true,
    race: { year: 2025, location: "Melbourne", total_laps: 57 },
    drivers,
  };
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
  TOWER_ORDER.filter((code) => !RETIRED_CODES.includes(code)).forEach(
    (code, index) => {
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
    },
  );
  return { rev: 1, drivers };
}

const towerCtx = await browser.newContext({
  viewport: CLIENT,
});
const towerPage = await towerCtx.newPage();
watchPage(towerPage, failures, "tower");
await towerPage.addInitScript(
  ([payload, bulk, live]) => {
    window.pywebview = {
      api: {
        get_tick: async (sinceSeq) =>
          sinceSeq === payload.seq ? null : payload,
        get_bulk: async (sinceRev) => (sinceRev === bulk.rev ? null : bulk),
        get_live_lap: async (sinceRev) => (sinceRev === live.rev ? null : live),
        get_connection: async () => ({ label: "Connected", colour: "#10b981" }),
      },
    };
  },
  [
    tick(1, { drivers: towerField(), order: TOWER_ORDER }),
    towerBulk(),
    towerLive(),
  ],
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

check(
  (await towerPage.locator(".tower-row").count()) === 20,
  "twenty rows, not the bulk's seventeen",
);
check(
  (await cell(18, 3)) === "SAI",
  "a car with no revealed lap still holds its place in the order",
);

// The four branches of the gap cell, each on the row that exercises it.
check((await cell(1, 4)) === "LEADER", "the leader's GAP says so");
check((await cell(1, 5)) === "—", "and the leader has no interval to anything");
check((await cell(2, 4)) === "+1.24s", "GAP is measured to the LEADER");
check((await cell(3, 4)) === "+3.07s", "and it accumulates down the order");
check(
  (await cell(3, 5)) === "+1.83s",
  "INT is measured to the car directly ahead",
);
check(
  (await cell(15, 4)) === "+1 LAP",
  "a lapped car reads in laps, not in tens of seconds",
);
check(
  (await cell(18, 4)) === "OUT",
  "a stopped car never shows a frozen interval",
);
check(
  (await cell(18, 9)) === "OUT",
  "and its LAST column says so in place of a lap time",
);

// The row itself, on the leader.
check((await cell(1, 3)) === "NOR", "the code column");
check(
  (await cell(1, 6)).replace(/\s+/g, " ") === "29.000 301",
  "the sector time and its trap speed, inline",
);
check(
  (await cell(1, 9)) === "1:25.744",
  "a lap time past the minute reads as m:ss.mmm",
);
check((await cell(1, 10)) === "321", "ST is the speed trap");
check((await cell(1, 11)) === "M 12", "the compound letter and the set's age");
check((await cell(1, 12)) === "2", "the stop count");
check(
  (await cell(18, 11)) === "—",
  "a driver the bulk never revealed shows dashes, not zeros",
);

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
    cardEl.clientWidth -
    parseFloat(style.paddingLeft) -
    parseFloat(style.paddingRight);
  return {
    lastBottom: last.bottom,
    cardBottom: card.bottom,
    naturalWidth,
    contentWidth,
    visible: rows.filter((row) => row.getBoundingClientRect().height > 0)
      .length,
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
check(
  towerFits.visible === 20,
  `all twenty rows have height (${towerFits.visible})`,
);

// --- The sector colour code, on the tower ------------------------------------
//
// Purple is fastest of the session outright, green is the driver's own best,
// yellow is slower than his own. The fixture arranges all three on the S1
// column: NOR owns it and matched it, PIA matched his own, VER did not.
const tone = async (row) =>
  towerPage
    .locator(`.tower-row:nth-child(${row}) td:nth-child(6)`)
    .getAttribute("class");
check(
  (await tone(1)).includes("is-purple"),
  "the session's fastest sector is purple",
);
check((await tone(2)).includes("is-green"), "a driver's own best is green");
check(
  (await tone(3)).includes("is-yellow"),
  "slower than his own best is yellow",
);
check(
  (await tone(18)).includes("is-plain"),
  "a sector nobody has set is not coloured at all",
);

// --- The sectors are the lap IN PROGRESS, not the last completed one ---------
//
// The bulk's completed row carries an S2 and an S3 for every driver here, so a
// tower reading it would print them. The live channel has only S1 open, which
// is what a car that has crossed one sector of the current lap actually knows.
// S2 and S3 carry the PREVIOUS lap's values, dimmed. They are not blank, and
// that is the whole of #933: a cell that blanked until this lap's crossing
// left S3 empty for the entire race, because a third sector's crossing IS the
// end of its lap.
check(
  (await cell(1, 7)).startsWith("19.500"),
  "S2 shows the previous lap's value, not a dash",
);
check(
  (await cell(1, 8)).startsWith("26.750"),
  "and so does S3, which otherwise NEVER fills",
);
const staleness = async (column) =>
  (
    await towerPage
      .locator(`.tower-row:nth-child(1) td:nth-child(${column})`)
      .getAttribute("class")
  ).includes("is-stale");
check((await staleness(6)) === false, "this lap's own sector is not dimmed");
check(
  (await staleness(7)) === true,
  "a carried-over sector says so by being dimmed",
);
check(
  (await staleness(8)) === true,
  "and S3 is carried over essentially always",
);
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
      .locator(
        `.bests-section:nth-child(${section}) .bests-row:nth-child(${row})`,
      )
      .innerText()
  )
    .replace(/\s+/g, " ")
    .trim();

check(
  (await towerPage.locator(".bests-section").count()) === 4,
  "four ranked sections",
);
// **Depth is the room's answer now, not a constant.** Three was argued against the
// 63 px slot the narrow client leaves, and it left 150 px of nothing at the wide one
// where the agreed drawing gives this card 303.
//
// The cap was then 10, for "P10 is the last point-scoring position" - a real argument
// about relevance and the wrong one for a CAP, which only bites when there is room to
// spare. Measured at 1485 px wide: the ramp works to 833 px tall and then the depth
// sticks while the hole grows, 91 px at 900 and 271 at 1600x1080. Víctor saw it on his
// own screen. Twenty is the field, so the panel runs out of DATA before room.
//
// So the assertion is the RULE: never below the floor, never past the whole grid, and
// whatever it picks it must SAY.
const depth = await towerPage.evaluate(() => ({
  rows: document.querySelectorAll(".bests-section:nth-child(1) .bests-row")
    .length,
  subtitle:
    document.querySelector(".bests-subtitle")?.textContent?.trim() ?? "",
  sections: [...document.querySelectorAll(".bests-section")].map(
    (section) => section.querySelectorAll(".bests-row").length,
  ),
}));
check(
  depth.rows >= 3 && depth.rows <= 20,
  `the ranked depth is between the floor and the whole field (${depth.rows})`,
);
check(
  depth.subtitle.includes(`top ${depth.rows}`),
  `and the panel says which depth it is showing ("${depth.subtitle}")`,
);
// All four sections at the same depth, or two of them are silently different lists.
check(
  new Set(depth.sections.filter((n) => n > 0)).size === 1,
  `every section is the same depth (${JSON.stringify(depth.sections)})`,
);
// S1 is NOR 29.000, PIA 29.500, VER 29.800 - ranked across the FIELD, not per
// driver, and the delta is a percentage off the section's leader.
check(
  (await bestsRow(1, 2)).startsWith("1 NOR 29.000"),
  "the section leader, with no delta",
);
check(
  (await bestsRow(1, 3)) === "2 PIA 29.500 +1.72%",
  "second, with its percentage off the top",
);
check((await bestsRow(1, 4)) === "3 VER 29.800 +2.76%", "and third");
// S2 and S3 are owned by different drivers, so a panel reading one field for
// all four sections would show NOR three times.
check(
  (await bestsRow(2, 2)).startsWith("1 PIA 18.500"),
  "S2 belongs to whoever set it",
);
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
  {
    kind: "rcm",
    lap: 2,
    driver: null,
    text: "DOUBLE YELLOW IN TRACK SECTOR 20",
    category: "Flag",
    flag: "DOUBLE YELLOW",
  },
  {
    kind: "radio",
    lap: 6,
    driver: "NOR",
    text: "Weather update, no significant rain expected.",
    category: null,
    flag: null,
  },
  {
    kind: "radio",
    lap: 14,
    driver: "HAM",
    text: "",
    category: null,
    flag: null,
  },
  {
    kind: "rcm",
    lap: 20,
    driver: null,
    text: "FIA STEWARDS: INCIDENT INVOLVING CAR 22 (TSU) NO FURTHER ACTION - SAFETY CAR INFRINGEMENT",
    category: "Other",
    flag: null,
  },
  {
    kind: "radio",
    lap: 21,
    driver: "NOR",
    text: "Lando, a bit of an update on the safety car window.",
    category: null,
    flag: null,
  },
  {
    kind: "radio",
    lap: 23,
    driver: "VER",
    text: "If there is heavy rain we might need to fit inters, bear in mind.",
    category: null,
    flag: null,
  },
];

async function radioPage(radio) {
  const context = await browser.newContext({
    viewport: CLIENT,
  });
  const rPage = await context.newPage();
  watchPage(rPage, failures, "radio");
  await rPage.addInitScript(
    ([payload, bulk, live]) => {
      window.pywebview = {
        api: {
          get_tick: async (sinceSeq) =>
            sinceSeq === payload.seq ? null : payload,
          get_bulk: async (sinceRev) => (sinceRev === bulk.rev ? null : bulk),
          get_live_lap: async (sinceRev) =>
            sinceRev === live.rev ? null : live,
          get_connection: async () => ({ label: "Connected", colour: "#10b981" }),
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

const [radioCtx, feedPage] = await radioPage({
  available: true,
  events: RADIO_EVENTS,
});

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
check(
  (await rowText(1)).startsWith("L23 VER"),
  "the newest event is the top row",
);
check(
  (await rowText(6)).startsWith("L2 RCM"),
  "and the oldest is the last one",
);

// The minute boundary `paceLabel` documented and its two siblings did not have.
// Evaluated against the real module rather than a copy of its arithmetic.
//
// **This ran in the browser until sprint 10 and it never executed once.** It
// asked for `/src/lib/format.ts` from a server that holds the BUILT bundle, so
// the import 404'd, `.catch(() => null)` turned that into `null`, and the `if`
// below skipped the whole assertion - a guard about the empty set, and the one
// page in this file with nothing watching its console, so the 404 was silent
// too. Transpiled here instead, with the `typescript` this project already
// depends on: same source the bundle is built from, no browser needed for a
// pure function, and no way left for it to skip itself.
const formatSource = readFileSync(resolve(UI_DIR, "src/lib/format.ts"), "utf-8");
const { outputText } = ts.transpileModule(formatSource, {
  compilerOptions: { module: ts.ModuleKind.ESNext, target: ts.ScriptTarget.ES2022 },
});
const format = await import(
  `data:text/javascript;base64,${Buffer.from(outputText).toString("base64")}`
);
const boundary = [
  format.formatSeconds(119.9996, 3),
  format.formatSeconds(59.9996, 3),
  format.formatSeconds(29.412, 3),
  format.formatSeconds(119.96, 1, true),
];
check(
  JSON.stringify(boundary) === JSON.stringify(["2:00.000", "1:00.000", "29.412", "2:00.0"]),
  `the shared formatter rounds before it splits (${JSON.stringify(boundary)})`,
);

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
  const [longCtx, longPage] = await radioPage({
    available: true,
    events: many,
  });
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
      oldestReached:
        !!oldest &&
        oldest.bottom <= box.bottom + 1 &&
        oldest.top >= box.top - 1,
    };
  });
  check(
    reach.folded > 100 && reach.rows === many.length,
    `the long feed really has a fold (${reach.folded} px hidden over ${reach.rows} rows)`,
  );
  check(
    reach.overflowY === "auto" &&
      reach.after > reach.before &&
      reach.oldestReached,
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
  tiers
    .filter((row) => row.who === "VER" || row.who === "HAM")
    .every((row) => row.broadcast),
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
check(
  (await rowText(4)).includes("no transcript"),
  "a radio with no words still shows itself",
);

// **The count is what stops an overflow being silent - and it now says BOTH
// numbers.** The rationale this check shipped with was "a panel that shows nine of
// forty-two and says nothing looks exactly like a panel showing all there is", and
// `42` alone left the reader to notice the discrepancy by counting rows. The header
// reads `visible / total`, so the assertion keeps the half that was load-bearing -
// the TOTAL must survive, or the fix would have hidden the very thing it announces -
// and adds the half that is new.
const headerCount = (await feedPage.locator(".radio-count").innerText()).trim();
check(
  headerCount.endsWith(`/ ${RADIO_EVENTS.length}`) ||
    headerCount === String(RADIO_EVENTS.length),
  `the header still names every revealed event, not just the ones that fit (${headerCount})`,
);
const [shown] = headerCount.split(" / ").map(Number);
// All six fixture events fit, so the honest assertion here is that the visible half
// does not UNDER-count and the fold does not claim a fold that is not there. (The
// first version of this measure read `offsetTop`, which is relative to the nearest
// positioned ancestor rather than the list, and reported `0 / 6`.)
check(
  shown === RADIO_EVENTS.length,
  `all ${RADIO_EVENTS.length} fixture events fit and the header says so (${headerCount})`,
);
check(
  (await feedPage.locator(".radio-fold").count()) === 0,
  "and no fold line is printed when there is nothing below the fold",
);

// **Now DRIVE the overflow**, because the fold is the affordance and a guard that
// never sees one proves nothing. Squeezing the list is the same thing a real race
// does with 58 events in a 404 px card.
await feedPage.addStyleTag({ content: ".radio-list { max-height: 40px; }" });
await feedPage
  .waitForSelector(".radio-fold", { timeout: 3000 })
  .catch(() => null);
const folded = await feedPage.evaluate(() => ({
  header: document.querySelector(".radio-count")?.textContent?.trim() ?? "",
  fold: document.querySelector(".radio-fold")?.textContent?.trim() ?? null,
}));
const [nowShown] = folded.header.split(" / ").map(Number);
check(
  Number.isFinite(nowShown) && nowShown < RADIO_EVENTS.length,
  `squeezed, the header drops to what fits (${folded.header})`,
);
// Not scrolled, so everything off screen IS below, and the fold names it.
check(
  folded.fold !== null &&
    folded.fold.includes(String(RADIO_EVENTS.length - nowShown)) &&
    folded.fold.includes("scroll"),
  `and the panel says how many are below the fold, in words (${folded.fold})`,
);

// **Now SCROLL it, because a one-edge measure is right until you do.** The first
// version of this measure tested `rect.bottom <= listBottom`, which every row scrolled
// off the TOP also satisfies: at the end of the list the header read `total / total`
// and the fold line disappeared, while a third of the rows sat above the viewport. The
// pace grid's sibling measure tested both edges from the start.
await feedPage.evaluate(() => {
  const list = document.querySelector(".radio-list");
  list.scrollTop = list.scrollHeight;
  list.dispatchEvent(new Event("scroll"));
});
await feedPage.waitForTimeout(300);
const atEnd = await feedPage.evaluate(() => {
  const list = document.querySelector(".radio-list");
  const frame = list.getBoundingClientRect();
  // The truth, measured independently of the component: rows intersecting the frame.
  const trulyVisible = [...list.querySelectorAll("li")].filter((li) => {
    const rect = li.getBoundingClientRect();
    return rect.bottom > frame.top + 1 && rect.top < frame.bottom - 1;
  }).length;
  return {
    header: document.querySelector(".radio-count")?.textContent?.trim() ?? "",
    fold: document.querySelector(".radio-fold")?.textContent?.trim() ?? null,
    trulyVisible,
  };
});
const [shownAtEnd] = atEnd.header.split(" / ").map(Number);
check(
  shownAtEnd <= atEnd.trulyVisible + 1 && shownAtEnd < RADIO_EVENTS.length,
  `scrolled to the end the header still counts what is IN VIEW (says ${shownAtEnd}, ${atEnd.trulyVisible} in view of ${RADIO_EVENTS.length})`,
);
// **And the fold correctly goes AWAY, which is not what this check first asserted.** It
// said "the fold line survives the scroll rather than vanishing", written to catch the
// one-edge measure - but that bug's real symptom is the HEADER reading `total / total`,
// which the check above holds. At the END of a newest-first list nothing OLDER is hidden,
// so a line saying "+ N older" would be the lie. Right expectation, wrong reason, and the
// fix to the fold's direction is what exposed it.
check(
  atEnd.fold === null,
  `and no line claims older events below when the list is at its end (${atEnd.fold})`,
);

// EFFECT, not mechanism: the card must not spill past the column it lives in.
const fits = await feedPage.evaluate(() => {
  const card = document.querySelector(".radio-feed");
  const column = document.querySelector(".side-column");
  return {
    overflow:
      card.getBoundingClientRect().bottom -
      column.getBoundingClientRect().bottom,
    ringVisible:
      document.querySelector(".ring").getBoundingClientRect().height > 0,
  };
});
check(
  fits.overflow <= 1,
  `the feed stays inside its column (spills ${fits.overflow.toFixed(1)}px)`,
);
check(fits.ringVisible, "and the ring above it is still there");

// **The two wire fields that nothing read.** `category` and `flag` have ridden the
// bulk since the feed shipped, and both fixture RCM rows carry them - so a reader
// could not tell `DOUBLE YELLOW` from the twentieth `NO FURTHER ACTION` without
// reading the sentence. Asserted as the SLOT each row shows, not as "a chip exists
// somewhere": the flag branch and the no-chip branch are different claims.
const chips = await feedPage.evaluate(() =>
  [...document.querySelectorAll(".radio-row")].map((row) => ({
    lap: row.querySelector(".radio-lap")?.textContent ?? "",
    chip: row.querySelector(".radio-cat")?.textContent ?? null,
  })),
);
check(
  chips.find((row) => row.lap === "L2")?.chip === "2Y",
  `the DOUBLE YELLOW row wears its flag, not the word FLAG (${chips.find((row) => row.lap === "L2")?.chip})`,
);
check(
  chips.find((row) => row.lap === "L20")?.chip === null,
  "and an `Other` stewards note wears nothing - a badge on two thirds of the rows is a badge on none",
);
check(
  chips.filter((row) => row.chip !== null).length === 1,
  `so exactly one of the six rows is chipped (${chips.filter((row) => row.chip !== null).length})`,
);

await radioCtx.close();

// **The collapse, driven.** Measured on the live 58-event payload there are ZERO
// consecutive duplicate runs, so the panel's own key comment - "four identical BLUE
// FLAG lines for the same car on Melbourne's lap 46" - is the only evidence the case
// exists, and a guard that never sees a duplicate proves nothing about collapsing
// one. This fixture is that lap 46.
const DUPLICATE_EVENTS = [
  {
    kind: "rcm",
    lap: 46,
    driver: null,
    text: "BLUE FLAG FOR CAR 31 (OCO)",
    category: "Flag",
    flag: "BLUE",
  },
  {
    kind: "rcm",
    lap: 46,
    driver: null,
    text: "BLUE FLAG FOR CAR 31 (OCO)",
    category: "Flag",
    flag: "BLUE",
  },
  {
    kind: "rcm",
    lap: 46,
    driver: null,
    text: "BLUE FLAG FOR CAR 31 (OCO)",
    category: "Flag",
    flag: "BLUE",
  },
  {
    kind: "rcm",
    lap: 46,
    driver: null,
    text: "BLUE FLAG FOR CAR 31 (OCO)",
    category: "Flag",
    flag: "BLUE",
  },
  {
    kind: "rcm",
    lap: 47,
    driver: null,
    text: "SAFETY CAR DEPLOYED",
    category: "SafetyCar",
    flag: null,
  },
  // Same text as the first four, an hour of race later. It must NOT join them.
  {
    kind: "rcm",
    lap: 52,
    driver: null,
    text: "BLUE FLAG FOR CAR 31 (OCO)",
    category: "Flag",
    flag: "BLUE",
  },
];
const [dupCtx, dupPage] = await radioPage({
  available: true,
  events: DUPLICATE_EVENTS,
});
const collapsed = await dupPage.evaluate(() =>
  [...document.querySelectorAll(".radio-row")].map((row) => ({
    lap: row.querySelector(".radio-lap")?.textContent ?? "",
    chip: row.querySelector(".radio-cat")?.textContent ?? null,
    repeats: row.querySelector(".radio-repeats")?.textContent?.trim() ?? null,
  })),
);
check(
  collapsed.length === 3,
  `six events with a run of four render as three rows (${collapsed.length})`,
);
check(
  collapsed[collapsed.length - 1]?.repeats === "x4",
  `and the run says how many it stands for rather than hiding them (${collapsed[collapsed.length - 1]?.repeats})`,
);
// Newest first, so the lap-52 copy is row 0 and the run is last. The separated copy
// must be its OWN row: collapsing by text alone would rewrite the chronology.
check(
  collapsed[0]?.lap === "L52" && collapsed[0]?.repeats === null,
  `the same message six laps later stays a separate, uncounted row (${collapsed[0]?.lap} ${collapsed[0]?.repeats})`,
);
check(
  collapsed.find((row) => row.lap === "L47")?.chip === "SC",
  `and the safety car is chipped SC (${collapsed.find((row) => row.lap === "L47")?.chip})`,
);

// **The header's numerator, counted independently from the DOM.**
//
// The numerator used to count collapsed ROWS while the denominator counted EVENTS -
// identical until something collapses, and this fixture is where it does (6 events, 3
// rows). The first guard for it asserted `visible + hidden == total`, which a gate
// refuted as a TAUTOLOGY: the component derives `hidden = events - visible`, so that sum
// is the total by construction, and reverting the numerator to rows left 217/217 green.
//
// So the expectation is rebuilt here from the rendered rows and their own `x4` labels,
// which the component does not get to define. Squeezed, the header must name the EVENTS
// those rows stand for, and the two numbers must DIFFER or the fixture is not exercising
// a collapse at all.
// Squeezed, then SCROLLED TO THE END, because the collapsed run is the OLDEST row and a
// squeeze alone leaves it below the fold - which is how the first version of this check
// came to compare 1 event against 1 row and prove nothing about collapsing.
await dupPage.addStyleTag({ content: ".radio-list { max-height: 34px; }" });
await dupPage.waitForTimeout(300);
await dupPage.evaluate(() => {
  const list = document.querySelector(".radio-list");
  list.scrollTop = list.scrollHeight;
  list.dispatchEvent(new Event("scroll"));
});
await dupPage.waitForTimeout(400);
const dupCounts = await dupPage.evaluate(() => {
  const list = document.querySelector(".radio-list");
  const frame = list.getBoundingClientRect();
  let eventsInView = 0;
  let rowsInView = 0;
  for (const item of list.querySelectorAll("li")) {
    const rect = item.getBoundingClientRect();
    if (rect.bottom > frame.bottom + 1 || rect.top < frame.top - 1) continue;
    rowsInView += 1;
    const label =
      item.querySelector(".radio-repeats")?.textContent?.trim() ?? "";
    eventsInView += label ? Number(label.replace(/\D/g, "")) : 1;
  }
  return {
    eventsInView,
    rowsInView,
    header: document.querySelector(".radio-count")?.textContent?.trim() ?? "",
    fold: document.querySelector(".radio-fold")?.textContent?.trim() ?? "",
  };
});
const [dupShown] = dupCounts.header.split(" / ").map(Number);
check(
  dupShown === dupCounts.eventsInView,
  `the header names the EVENTS on screen, not the rows (says ${dupShown}, DOM has ${dupCounts.eventsInView} events over ${dupCounts.rowsInView} rows: "${dupCounts.header}")`,
);
check(
  dupCounts.eventsInView !== dupCounts.rowsInView,
  `and the fixture really is exercising a collapse (${dupCounts.eventsInView} events, ${dupCounts.rowsInView} rows)`,
);
// At the END of a newest-first list there is nothing OLDER hidden, so the line that says
// "older" must be gone. It used to stay and count the NEWER rows above instead: measured
// on the real corpus it read `+ 10 older` with nine of the ten being newer.
check(
  dupCounts.fold === "",
  `at the end of the list nothing is older, so no fold line claims otherwise ("${dupCounts.fold}")`,
);
await dupCtx.close();

// A race with no corpus SAYS so. An empty list and a missing corpus are the
// same pixel otherwise, which is the twin F7 caught one sprint ago between
// get_bulk and get_live_lap.
const [emptyCtx, emptyPage] = await radioPage({ available: false, events: [] });
check(
  (await emptyPage.locator(".radio-row").count()) === 0,
  "no rows for a race with no corpus",
);
check(
  (await emptyPage.locator(".radio-subtitle").innerText()).includes(
    "no corpus",
  ),
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
 * average changes nothing - which is exactly why the fixture was 100% blind to
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
        number,
        laps_revealed: 0,
        stops: 0,
        crossings: {},
        laps: [
          {
            lap: 1,
            t: null,
            lap_time: null,
            s1: null,
            s2: null,
            s3: null,
            v1: null,
            v2: null,
            vfl: null,
            vst: null,
            position: null,
            compound: null,
            tyre_life: null,
            stint: null,
            track_status: "1",
            neutralised: null,
            pit_in: false,
            pit_out: false,
            deleted: false,
            generated: true,
            pb: false,
          },
        ],
        best: {
          lap: null,
          lap_time: null,
          s1: null,
          s2: null,
          s3: null,
          v1: null,
          v2: null,
          vfl: null,
          vst: null,
          compound: null,
        },
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
      // REAL race has them: measured on Melbourne 2025, 82.4% of laps sit
      // past +10% of the session best, so a heat scale anchored on that best
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
      if (!pitIn && !pitOut && !deleted && (best === null || time < best))
        best = time;
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
      laps.push({
        lap,
        t: elapsed,
        lap_time: lapTime,
        s1: 29,
        s2: 30,
        s3: 26,
        v1: 301,
        v2: 289,
        vfl: 280,
        vst: 321,
        position: index + 1,
        compound: "MEDIUM",
        tyre_life: lap,
        stint: 1,
        track_status: neutralised ? "4" : "1",
        neutralised: neutralised ? "SAFETY CAR" : null,
        pit_in: pitIn,
        pit_out: pitOut,
        deleted,
        generated: false,
        pb: false,
      });
    }
    drivers[code] = {
      number,
      laps_revealed: revealed,
      stops: 1,
      crossings,
      laps,
      best: {
        lap: PACE_FASTEST.lap,
        lap_time: best,
        s1: 29,
        s2: 30,
        s3: 26,
        v1: 301,
        v2: 289,
        vfl: 280,
        vst: 321,
        compound: "MEDIUM",
      },
      theoretical: 85,
    };
  });
  return {
    rev: 1,
    available: true,
    race: { year: 2025, location: "Melbourne", total_laps: PACE_LAPS },
    drivers,
    radio: { available: true, events: [] },
  };
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

const paceCtx = await browser.newContext({
  viewport: CLIENT,
});
const pacePage = await paceCtx.newPage();
watchPage(pacePage, failures, "pace");
await pacePage.addInitScript(
  ([payload, bulk, live]) => {
    window.pywebview = {
      api: {
        get_tick: async (s) => (s === payload.seq ? null : payload),
        get_bulk: async (r) => (r === bulk.rev ? null : bulk),
        get_live_lap: async (r) => (r === live.rev ? null : live),
        get_connection: async () => ({ label: "Connected", colour: "#10b981" }),
      },
    };
  },
  [
    tick(1, { drivers: paceField(), order: TOWER_ORDER }),
    paceBulk(),
    towerLive(),
  ],
);
await pacePage.goto(url, { waitUntil: "domcontentloaded" });
await pacePage.waitForSelector(".tab-strip", { timeout: 5000 });

// The ring and the traces own the column until the reader asks for the grid.
check(
  (await pacePage.locator(".ring").count()) === 1,
  "the TRACES tab opens with the ring on it",
);
check(
  (await pacePage.locator(".pace-table").count()) === 0,
  "and the grid is not mounted yet",
);

await pacePage.getByRole("tab", { name: "RACE PACE" }).click();
await pacePage.waitForSelector(".pace-table", { timeout: 5000 });
await pacePage.waitForTimeout(400);

// Measured, not assumed: with the 260 px ring column still mounted the grid
// gets 555 px, its columns fall to 25.25 px against 25 px of text and 1,101 of
// 1,140 cells clip. There is no arrangement that keeps both.
check(
  (await pacePage.locator(".ring").count()) === 0,
  "the ring hides on the RACE PACE tab",
);
check(
  (await pacePage.locator(".radio-feed").count()) === 0,
  "and the radio feed hides with it",
);

const paceHead = await pacePage.evaluate(() =>
  [...document.querySelectorAll(".pace-table thead th")]
    .slice(1)
    .map((h) => h.textContent),
);
check(
  paceHead.length === 20,
  `every driver the wire names gets a column (${paceHead.length})`,
);
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
  [
    paceHead,
    Object.fromEntries(
      Object.entries(paceBulk().drivers).map(([c, d]) => [
        c,
        d.number === null ? null : Number(d.number),
      ]),
    ),
  ],
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
  new Set(paceHead).size === 20 &&
    paceHead.every((code) => TOWER_ORDER.includes(code)),
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
  const tone = (name) =>
    cells.filter((c) => c.className === `is-${name}`).length;
  return {
    total: cells.length,
    clipped: cells.filter((c) => c.scrollWidth > c.clientWidth).length,
    best: tone("best"),
    t1: tone("t1"),
    t2: tone("t2"),
    t3: tone("t3"),
    pit: tone("pit"),
    out: tone("out"),
    none: tone("none"),
    pitText: cells.find((c) => c.className === "is-pit")?.textContent,
    outText: cells.find((c) => c.className === "is-out")?.textContent,
    rows: document.querySelectorAll(".pace-table tbody tr").length,
  };
});

// EFFECT, not mechanism. `overflow-x` on the container reports zero for every
// variant that clips - only the cell's own scrollWidth sees the digits cut,
// which is how a 0.27 px "fit" measured as a pass right up to the screenshot.
check(
  paceCells.clipped === 0,
  `no lap time is cut (${paceCells.clipped}/${paceCells.total} clipped)`,
);
check(
  paceCells.rows === PACE_LAPS,
  `one row per lap of the race (${paceCells.rows})`,
);
check(
  paceCells.pitText === "IN PIT" && paceCells.outText === "P.EXIT",
  "the in-lap and the out-lap replace the time, as a timing screen shows them",
);
// **And neither of them says what the tower says about a RETIRED car.** The
// tower's own docstring refuses to reuse that word for a car that is still
// racing; this grid used it for the out-lap, in the same window, on the same
// screen. Asserted over the whole enumeration of cell texts rather than on the
// one sampled above, so a single tone reverting is still caught.
const paceWords = await pacePage.evaluate(() => {
  const cells = [...document.querySelectorAll(".pace-table td")];
  const tower = [...document.querySelectorAll(".tower-row .col-last")].map(
    (c) => c.textContent,
  );
  return {
    collisions: cells.filter((c) => c.textContent.trim() === "OUT").length,
    towerUsesIt: tower.includes("OUT"),
  };
});
check(
  paceWords.collisions === 0,
  `no pace cell says OUT, which the tower reserves for a retirement (${paceWords.collisions} do)`,
);
check(
  paceCells.best === 1,
  `exactly one purple cell - the session's fastest lap (${paceCells.best})`,
);

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
  return new Promise((done) =>
    setTimeout(
      () => done(document.querySelector(".pace-range").textContent),
      150,
    ),
  );
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
  boundaryCell.includes("2:00.0") &&
    !boundaryCell.some((t) => /:60\./.test(t ?? "")),
  `a lap just under a minute boundary rounds up to the next minute, never :60 (${boundaryCell
    .filter((t) => t && t.startsWith("1:5") === false && t.includes(":"))
    .slice(0, 3)
    .join(",")})`,
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
check(
  deletedCells.struck > 0,
  `the deleted laps reach the grid (${deletedCells.struck})`,
);
check(
  deletedCells.line === "line-through",
  `a deleted time is struck through, as the tower already shows it (${deletedCells.line})`,
);
check(
  /^\d:\d\d\.\d$/.test(deletedCells.text),
  `and it still shows its time rather than vanishing (${deletedCells.text})`,
);

// The reason the colour ranks each lap against ITSELF. Anchored on the session
// best with fixed percentage bands, 82.4% of the real Melbourne payload lands
// in one colour, because the race was wet and ran safety cars.
const spread = [paceCells.t1, paceCells.t2, paceCells.t3];
check(
  spread.every((count) => count > 0) &&
    Math.max(...spread) < paceCells.total * 0.75,
  `the heat scale uses all three tones rather than painting one (${spread.join(" / ")})`,
);

// The check that actually separates the two scales, and the ONLY one that
// does - the tidy laps look the same under both. Under a neutralisation the
// whole field is bunched, so any fixed percentage band collapses it into one
// or two tones and stops discriminating at exactly the moment a strategist is
// reading the grid. Measured on a mutated copy that banded at 1.5% / 4%:
// 40 / 45 / 0, the slowest tone empty across all five laps. Ranking inside the
// lap splits the field whatever the spread.
const scLaps = await pacePage.evaluate(() => {
  const rows = [...document.querySelectorAll(".pace-table tbody tr")].slice(
    1,
    6,
  );
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
      ? getComputedStyle(rows[railed[0] - 1].querySelector("th"))
          .borderLeftWidth
      : "0px",
    legend: document.querySelectorAll(".pace-legend-rail").length,
    title:
      rows[railed[0] - 1]?.querySelector("th")?.getAttribute("title") ?? "",
  };
});

check(
  JSON.stringify(rails.railed) === JSON.stringify([2, 3, 4, 5, 6, 30]),
  `the neutralised laps carry a rail and only those (${rails.railed.join(",")})`,
);
check(
  rails.plain.length === PACE_LAPS - 6 &&
    !rails.plain.includes(4) &&
    !rails.plain.includes(30),
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
// through the P0: on a 1080p laptop at 150% scaling - Windows' own recommended
// scaling for a 13-14" screen - the client is 1265 x 593, the columns fall to
// 27.75 px, and 495 of 514 populated cells lost their last glyph in silence.
//
// A guard whose probe sits at the one size where the defect does not exist is
// the shape this file's own header warns about, so this block PROVES the two
// widths are distinguishable - the column really is narrower - before asserting
// that nothing clips at either.
{
  const narrowCtx = await browser.newContext({
    viewport: { width: 1265, height: 593 },
  });
  const narrow = await narrowCtx.newPage();
  watchPage(narrow, failures, "pace-narrow");
  await narrow.addInitScript(
    ([payload, bulk, live]) => {
      window.pywebview = {
        api: {
          get_tick: async (s) => (s === payload.seq ? null : payload),
          get_bulk: async (r) => (r === bulk.rev ? null : bulk),
          get_live_lap: async (r) => (r === live.rev ? null : live),
          get_connection: async () => ({ label: "Connected", colour: "#10b981" }),
        },
      };
    },
    [
      tick(1, { drivers: paceField(), order: TOWER_ORDER }),
      paceBulk(),
      towerLive(),
    ],
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
      column:
        document.querySelector(".pace-table thead th + th")?.clientWidth ?? 0,
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
    const column = document
      .querySelector(".left-column")
      .getBoundingClientRect();
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
        ? +(
            Math.max(
              ...outside.map((el) => el.getBoundingClientRect().bottom),
            ) - column.bottom
          ).toFixed(1)
        : 0,
      // The card is clamped to its slot, so an overflow no longer leaves the
      // column - it becomes a scroll inside the card. Asserting BOTH is what
      // keeps "nothing outside the column" from passing on a panel that simply
      // moved the hiding one level in.
      hidden: card.scrollHeight - card.clientHeight,
      subtitle: document.querySelector(".bests-subtitle")?.textContent ?? "",
      // The theoretical lap is the one value a wall reads off this panel that no
      // other panel carries, so it is the one that must survive the degradation.
      theoretical:
        document.querySelector(".bests-theoretical-value")?.textContent ?? "",
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
  const partialCtx = await browser.newContext({
    viewport: CLIENT,
  });
  const partial = await partialCtx.newPage();
  watchPage(partial, failures, "skeleton");
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
          get_connection: async () => ({ label: "Connected", colour: "#10b981" }),
        },
      };
    },
    [
      tick(1, { drivers: paceField(), order: TOWER_ORDER }),
      paceBulk(),
      towerLive(),
      REVEALED_TO,
    ],
  );
  await partial.goto(`http://127.0.0.1:${server.address().port}/data.html`, {
    waitUntil: "domcontentloaded",
  });
  await partial.getByRole("tab", { name: "RACE PACE", exact: true }).click();
  await partial.waitForSelector(".pace-table", { timeout: 5000 });
  await partial.waitForTimeout(600);

  const skeleton = await partial.evaluate(() => {
    const lin = (v) =>
      v / 255 <= 0.03928 ? v / 255 / 12.92 : ((v / 255 + 0.055) / 1.055) ** 2.4;
    const L = ([r, g, b]) =>
      0.2126 * lin(r) + 0.7152 * lin(g) + 0.0722 * lin(b);
    const ratio = (a, b) => {
      const [hi, lo] = [L(a), L(b)].sort((x, y) => y - x);
      return (hi + 0.05) / (lo + 0.05);
    };
    const parse = (value) =>
      value
        .match(/[0-9]+/g)
        .slice(0, 3)
        .map(Number);
    const rows = [...document.querySelectorAll(".pace-table tbody tr")];
    const future = rows.filter((row) => row.classList.contains("is-future"));
    const driven = rows.filter((row) => !row.classList.contains("is-future"));
    const newest = driven[driven.length - 1];
    const box = document.querySelector(".pace-scroll").getBoundingClientRect();
    const card = parse(
      getComputedStyle(document.querySelector(".pace")).backgroundColor,
    );
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
        [...row.querySelectorAll("td")].some(
          (cell) => cell.textContent.trim().length > 0,
        ),
      ).length,
      lastLapNumber: Number(
        rows[rows.length - 1].querySelector("th").textContent,
      ),
      newestLap: Number(newest.querySelector("th").textContent),
      newestVisible:
        newestBox.top >= box.top - 1 && newestBox.bottom <= box.bottom + 1,
      futureRatio,
    };
  });

  check(
    skeleton.rows === PACE_LAPS && skeleton.lastLapNumber === PACE_LAPS,
    `the grid draws the whole race, not the part that has run (${skeleton.rows} rows, last lap ${skeleton.lastLapNumber} of ${PACE_LAPS})`,
  );
  check(
    skeleton.driven === REVEALED_TO &&
      skeleton.future === PACE_LAPS - REVEALED_TO,
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
  watchPage(page, failures, "bests ${height}");
  // **Withheld at the STUB, not with `page.route`.** `bridge.ts` uses
  // `window.pywebview` whenever it exists and only falls back to `fetch`, so the
  // smoke's injected api is the transport and an HTTP route intercepts nothing -
  // the first version of this block held `/api/bulk` and the panel got its rows
  // immediately anyway, which made the guard pass on the defect it was written for.
  await page.addInitScript(
    ([payload, bulk, live]) => {
      window.__holdBulk = true;
      window.pywebview = {
        api: {
          get_tick: async (s) => (s === payload.seq ? null : payload),
          get_bulk: async (r) =>
            window.__holdBulk || r === bulk.rev ? null : bulk,
          get_live_lap: async (r) => (r === live.rev ? null : live),
          get_connection: async () => ({ label: "Connected", colour: "#10b981" }),
        },
      };
    },
    [
      tick(1, { drivers: paceField(), order: TOWER_ORDER }),
      paceBulk(),
      towerLive(),
    ],
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
    const column = document
      .querySelector(".left-column")
      .getBoundingClientRect();
    const card = document.querySelector(".bests");
    const box = card.getBoundingClientRect();
    const theo = document.querySelector(".bests-theoretical-value");
    return {
      hidden: card.scrollHeight - card.clientHeight,
      over: +(box.bottom - column.bottom).toFixed(1),
      // The one value the panel's own docstring calls irreplaceable.
      theoreticalBelow:
        theo === null
          ? null
          : +(theo.getBoundingClientRect().bottom - column.bottom).toFixed(1),
      form:
        document.querySelectorAll(".bests-row").length > 0
          ? "ranked"
          : "leaders",
    };
  });
  check(
    fit.hidden === 0 &&
      fit.over <= 1 &&
      fit.theoreticalBelow !== null &&
      fit.theoreticalBelow <= 1,
    `bests fits at ${width}x${height} whichever form it picks (${fit.form}, ${fit.hidden} px hidden, card ${fit.over} px over, THEORETICAL ${fit.theoreticalBelow} px over)`,
  );
  await ctx.close();
}

// --- The wheel scroll Víctor asked for, and whether it survives a reveal ------
//
// > *"yo veo esqueleto que si sobrepasa, tenga un mini scroll que se pueda hacer para
// > bajar con la rueda del raton"*
//
// Three separate things have to be true and only one of them was: the wheel has to move
// the panel (it always did - hiding a scrollbar disables nothing), the panel has to SAY
// it scrolls (it did not, and research on hidden scrollbars is unanimous that nobody
// discovers it), and **a reveal must not undo the reader's scroll** (it did: the pin
// re-fires every reveal, about every 4.5 s, so a scroll to look at lap 5 was thrown away
// before it could be read).
{
  const ctx = await browser.newContext({
    viewport: { width: 1265, height: 593 },
  });
  const page = await ctx.newPage();
  watchPage(page, failures, "wheel");
  // **Capped at lap 30, like the skeleton block one section down.** `paceBulk()` reveals
  // all 57 laps, so on it there are no future rows at all, the pin target IS the bottom of
  // the scroller, and the reader can never be "away from" it - the first version of this
  // block measured exactly that and proved nothing. A cap is what makes a mid-race panel.
  const CAP = 30;
  await page.addInitScript(
    ([payload, bulk, live, cap]) => {
      window.__advance = false;
      const at = (limit) => {
        const copy = JSON.parse(JSON.stringify(bulk));
        copy.rev = bulk.rev + (limit > cap ? 1 : 0);
        for (const driver of Object.values(copy.drivers)) {
          driver.laps = driver.laps.filter((lap) => lap.lap <= limit);
        }
        return copy;
      };
      const first = at(cap);
      const next = at(cap + 1);
      window.pywebview = {
        api: {
          get_tick: async (s) => (s === payload.seq ? null : payload),
          get_bulk: async (r) => {
            const served = window.__advance ? next : first;
            return r === served.rev ? null : served;
          },
          get_live_lap: async (r) => (r === live.rev ? null : live),
          get_connection: async () => ({ label: "Connected", colour: "#10b981" }),
        },
      };
    },
    [
      tick(1, { drivers: paceField(), order: TOWER_ORDER }),
      paceBulk(),
      towerLive(),
      CAP,
    ],
  );
  await page.goto(`http://127.0.0.1:${server.address().port}/data.html`, {
    waitUntil: "domcontentloaded",
  });
  await page.getByRole("tab", { name: "RACE PACE" }).click();
  await page.waitForSelector(".pace-scroll", { timeout: 5000 });
  await page.waitForTimeout(700);

  const box = page.locator(".pace-scroll");
  // Keyboard reach: with the scrollbar hidden this is the only way a keyboard user gets
  // to the rest of the race.
  check(
    (await box.getAttribute("tabindex")) === "0" &&
      (await box.getAttribute("role")) === "region",
    "the pace scroller is reachable by keyboard and named for a screen reader",
  );

  const room = await page.evaluate(() => {
    const el = document.querySelector(".pace-scroll");
    return el.scrollHeight - el.clientHeight;
  });
  check(
    room > 0,
    `the whole-race table really does overflow this client (${room} px of scroll)`,
  );

  // **The fade, from MEASURED overflow.** At rest there is more below and nothing above.
  const atRest = await page.getAttribute(".pace-scroll", "class");
  check(
    atRest.includes("has-below") && !atRest.includes("has-above"),
    `at rest the panel says there is more below and nothing above ("${atRest}")`,
  );

  // **The wheel.** Not `scrollTop = n`: the actual input device, over the actual element.
  await box.hover();
  await page.mouse.wheel(0, 200);
  await page.waitForTimeout(400);
  const afterWheel = await page.evaluate(
    () => document.querySelector(".pace-scroll").scrollTop,
  );
  check(
    afterWheel > 0,
    `the mouse wheel scrolls it even with the scrollbar hidden (scrollTop ${afterWheel})`,
  );
  const scrolled = await page.getAttribute(".pace-scroll", "class");
  check(
    scrolled.includes("has-above"),
    `and once scrolled the panel says there is something above ("${scrolled}")`,
  );

  // **A reveal must NOT take the scroll back.**
  await page.evaluate(() => {
    window.__advance = true;
  });
  await page.waitForFunction(
    (before) =>
      document.querySelectorAll(".pace-table tbody tr:not(.is-future)").length >
      before,
    await page.evaluate(
      () =>
        document.querySelectorAll(".pace-table tbody tr:not(.is-future)")
          .length,
    ),
    { timeout: 8000 },
  );
  await page.waitForTimeout(500);
  const afterReveal = await page.evaluate(
    () => document.querySelector(".pace-scroll").scrollTop,
  );
  check(
    Math.abs(afterReveal - afterWheel) <= 2,
    `a new lap does not throw the reader's scroll away (${afterWheel} -> ${afterReveal})`,
  );

  await ctx.close();
}

// --- Can the lane axes be READ, at every client -----------------------------
//
// **A locked range picks its labels from the range, never from the room.** Measured
// before this guard existed: THROTTLE and BRAKE printed `-5 0 20 40 60 80 100 105`,
// whose closest pair sat 2.1 px apart in a 46 px lane under a 10 px font, and GEAR's
// 4.1 px apart in 37 px. Seen at 3x on a real screenshot, then fixed with a per-lane
// allow-list.
//
// **It was not a narrow-client defect, which is why this loop runs the WIDE size too.**
// Driven red, 1485x833 fails at 3.8 px pitch - the client the panel had been signed
// off against by eye.
//
// The assertion is the PITCH, computed from each lane's own box and its own locked
// range, which is why it holds at five client sizes rather than pinning a pixel
// table: a lane that shrinks has to drop labels, not squeeze them.
// **How much of the stack the lanes leave unused, per client.** Collected across the
// loop and asserted after it: nothing here asserted that the lanes FILL their box.
// Ordered, non-collapsed, sharing margins, spanned by the cursor - all of those are
// checked, and a layout that stopped consuming the room would satisfy every one of them
// while leaving 100 px of dead space at the bottom. Which is precisely the complaint
// that opened #998, one panel to the left.
const laneLeftovers = [];
for (const [width, height] of [
  [1265, 593],
  [1265, 650],
  [1350, 660],
  [1350, 673],
  [CLIENT.width, CLIENT.height],
]) {
  const ctx = await browser.newContext({ viewport: { width, height } });
  const page = await ctx.newPage();
  watchPage(page, failures, "lanes ${width}");
  await page.addInitScript(
    ([payload, bulk, live]) => {
      window.pywebview = {
        api: {
          get_tick: async (s) => (s === payload.seq ? null : payload),
          get_bulk: async (r) => (r === bulk.rev ? null : bulk),
          get_live_lap: async (r) => (r === live.rev ? null : live),
          get_connection: async () => ({ label: "Connected", colour: "#10b981" }),
        },
      };
    },
    [
      tick(1, { drivers: paceField(), order: TOWER_ORDER }),
      paceBulk(),
      towerLive(),
    ],
  );
  await page.goto(`http://127.0.0.1:${server.address().port}/data.html`, {
    waitUntil: "domcontentloaded",
  });
  await page.waitForSelector(".trace-stack-plot canvas", { timeout: 5000 });
  await page.waitForTimeout(700);

  laneLeftovers.push({
    client: `${width}x${height}`,
    ...(await page.evaluate(() => {
      const chart = document.querySelector(".trace-stack-plot").__pitwallChart;
      const grids = chart.getOption().grid;
      const last = grids[grids.length - 1];
      const box = document.querySelector(".trace-stack").clientHeight;
      return {
        box,
        below: Math.round(box - (last.top + last.height)),
        sum: grids.reduce((total, grid) => total + grid.height, 0),
      };
    })),
  });

  const legible = await page.evaluate(() => {
    const chart = document.querySelector(".trace-stack-plot").__pitwallChart;
    const opt = chart.getOption();
    return opt.grid.map((grid, index) => {
      const cfg = opt.yAxis[index];
      const font = cfg.axisLabel?.fontSize ?? 12;
      // `formattedLabel` is the FORMATTER'S output, so an allow-listed blank reads as
      // "". (A first version of this read `tickValue`, which does not exist on these
      // items, and reported every lane as printing nothing.)
      const painted =
        cfg.axisLabel?.show === false
          ? []
          : chart
              .getModel()
              .getComponent("yAxis", index)
              .axis.getViewLabels()
              .filter((label) => label.formattedLabel !== "")
              .map((label) => label.tick.value);
      const span = cfg.max - cfg.min;
      const gaps = painted
        .slice(1)
        .map((value, k) => Math.abs(value - painted[k]));
      const pitch = gaps.length
        ? (Math.min(...gaps) / span) * grid.height
        : null;
      return {
        painted: painted.length,
        font,
        pitch: pitch === null ? null : +pitch.toFixed(1),
        height: Math.round(grid.height),
        ticks: cfg.axisTick?.show !== false,
        split: cfg.splitLine?.show !== false,
      };
    });
  });
  const tight = legible.filter(
    (lane) => lane.pitch !== null && lane.pitch < lane.font + 2,
  );
  check(
    tight.length === 0,
    `every lane's printed labels clear each other at ${width}x${height} (${
      tight
        .map(
          (lane) =>
            `${lane.height}px lane at ${lane.pitch}px pitch under ${lane.font}px`,
        )
        .join("; ") || "all clear"
    })`,
  );
  // And the DRS strip, which at 11 px cannot carry an axis at all: six ticks and five
  // split lines merged into a smear hanging off the frame. Killing the split lines
  // alone left it - the marks were the TICKS - so both are asserted.
  const drs = legible[legible.length - 1];
  check(
    drs.painted === 0 && !drs.ticks && !drs.split,
    `the ${drs.height}px DRS strip carries no labels, ticks or split lines (${drs.painted}, ${drs.ticks}, ${drs.split})`,
  );
  await ctx.close();
}

// **The lanes must SUM to the box, and the space below the last one must be the shared
// axis band EXACTLY.**
//
// This started as "the leftover is identical at every client and under 50 px", which a
// gate refuted with one line of arithmetic: a CONSTANT dead strip of 12 px makes the
// leftover 46 at every client - identical, under 50, and green. Any invariant that a
// constant offset satisfies cannot see a constant offset.
//
// So the constants are read FROM THE SOURCE and the arithmetic is asserted whole. That
// is the one place they are declared, so this is not a second copy of them; and the
// claim is now the one the comment makes, which is what the previous version of this
// block promised and did not do.
const layout = Object.fromEntries(
  [
    ...readFileSync(TRACE_STACK_SOURCE, "utf8").matchAll(
      /^const (AXIS_BAND|LANE_GAP|LABEL_ROW) = (\d+);/gm,
    ),
  ].map((match) => [match[1], Number(match[2])]),
);
check(
  ["AXIS_BAND", "LANE_GAP", "LABEL_ROW"].every((key) =>
    Number.isFinite(layout[key]),
  ),
  `the layout constants are readable from TraceStack.tsx (${JSON.stringify(layout)})`,
);
// `grid.height` is the PLOT, and each lane also carries a LABEL_ROW above it - which is
// what makes this the whole arithmetic rather than a plausible subset of it: the first
// version omitted the six label rows and came out 72 px short at every client, so the
// guard's own first run told me the identity I had written down was incomplete.
const predicted = (entry) =>
  entry.sum + layout.LABEL_ROW * 6 + layout.LANE_GAP * 5 + layout.AXIS_BAND;
const misfits = laneLeftovers.filter((entry) => predicted(entry) !== entry.box);
check(
  misfits.length === 0,
  `the six lanes, their label rows, five gaps and the axis band ARE the box at every client (${laneLeftovers
    .map((entry) => `${entry.client}: ${predicted(entry)} vs ${entry.box}`)
    .join("; ")})`,
);

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

check(
  (await pacePage.locator(".pace-table").count()) === 0,
  "the pace grid unmounts on the trace tab",
);
check(
  (await pacePage.locator(".ring").count()) === 0,
  "and the ring stays hidden here too",
);

/** The trace's series and axes, read off the live ECharts instance. */
const traceState = () =>
  pacePage.evaluate(() => {
    const el = document.querySelector(".trace-band-plot");
    const chart = el && el.__pitwallChart;
    if (!chart) return null;
    const series = chart.getOption().series;
    const axis = (type) =>
      chart.getModel().getComponent(type, 0).axis.scale.getExtent();
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
    ranges: area.data.map(([from, to]) => [
      from.xAxis,
      to.xAxis,
      from.name ?? "",
    ]),
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
  const narrowCtx = await browser.newContext({
    viewport: { width: 1265, height: 593 },
  });
  const narrow = await narrowCtx.newPage();
  watchPage(narrow, failures, "axis");
  await narrow.addInitScript(
    ([payload, bulk, live]) => {
      window.pywebview = {
        api: {
          get_tick: async (s) => (s === payload.seq ? null : payload),
          get_bulk: async (r) => (r === bulk.rev ? null : bulk),
          get_live_lap: async (r) => (r === live.rev ? null : live),
          get_connection: async () => ({ label: "Connected", colour: "#10b981" }),
        },
      };
    },
    [
      tick(1, { drivers: paceField(), order: TOWER_ORDER }),
      paceBulk(),
      towerLive(),
    ],
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
    const shown =
      x.axisLabel.showMinLabel === false ? ticks.slice(1, -1) : ticks;
    const labels = shown.map((value) =>
      format ? format(value) : String(value),
    );
    const ruler = document.createElement("canvas").getContext("2d");
    ruler.font = `${x.axisLabel.fontSize}px ${getComputedStyle(document.body).fontFamily}`;
    const glyphs = labels.reduce(
      (total, text) => total + ruler.measureText(text).width,
      0,
    );
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
  const deadCtx = await browser.newContext({
    viewport: CLIENT,
  });
  const dead = await deadCtx.newPage();
  watchPage(dead, failures, "dead");
  await dead.addInitScript(
    ([payload, bulk, live]) => {
      window.__alive = true;
      window.pywebview = {
        api: {
          get_tick: async (s) => (s === payload.seq ? null : payload),
          get_bulk: async (r) => (r === bulk.rev ? null : bulk),
          get_live_lap: async (r) => (r === live.rev ? null : live),
          get_connection: async () =>
            window.__alive
              ? { label: "Connected", colour: "#10b981" }
              : { label: "Disconnected", colour: "#ef4444" },
        },
      };
    },
    [
      tick(1, { drivers: paceField(), order: TOWER_ORDER }),
      paceBulk(),
      towerLive(),
    ],
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
    alive.rows === 20 &&
      /^\d+x$/.test(alive.playback ?? "") &&
      alive.bar.includes("live"),
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
    lost: [...document.querySelectorAll(".strip-chip")].filter(
      (el) => el.innerText.trim() === "Disconnected",
    ).length,
  }));

  check(
    frozen.playback === "—" &&
      !frozen.bar.includes("live") &&
      frozen.bar.includes("FROZEN"),
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
  const later = await dead.evaluate(
    () => document.querySelector(".status-bar")?.textContent ?? "",
  );
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
const retiredPoints = RETIRED_CODES.map(
  (code) => traceLeader?.points[code]?.length ?? -1,
);
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
const norLine = Object.fromEntries(
  (traceLeader?.points.NOR ?? []).map(([lap, y]) => [lap, y]),
);
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
      return {
        code: el.style.text,
        x: rect.x,
        y: rect.y,
        w: rect.width,
        h: rect.height,
      };
    });
  const overlaps = [];
  for (let i = 0; i < boxes.length; i += 1) {
    for (let j = i + 1; j < boxes.length; j += 1) {
      const a = boxes[i];
      const b = boxes[j];
      const dx = Math.min(a.x + a.w, b.x + b.w) - Math.max(a.x, b.x);
      const dy = Math.min(a.y + a.h, b.y + b.h) - Math.max(a.y, b.y);
      if (dx > 0 && dy > 0)
        overlaps.push(`${a.code}/${b.code} ${dx.toFixed(1)}x${dy.toFixed(1)}`);
    }
  }
  return {
    count: boxes.length,
    overlaps,
    right: Math.max(...boxes.map((z) => z.x + z.w)),
    canvas: chart.getWidth(),
  };
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
const aboveField = Object.values(traceField?.points ?? {})
  .flat()
  .filter(([, y]) => y > 0).length;
const aboveLeader = Object.values(traceLeader?.points ?? {})
  .flat()
  .filter(([, y]) => y > 0).length;
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
const traceRange = await pacePage
  .locator(".trace-band .pace-range")
  .innerText();
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
  historyStable !== null &&
    Math.abs(historyStable.everyone - historyStable.actual) < 1e-9,
  `a retired car still counts in the laps he drove (everyone ${historyStable?.everyone?.toFixed(3)}, still-racing ${historyStable?.stillRacing?.toFixed(3)}, rendered ${historyStable?.actual?.toFixed(3)}, over ${historyStable?.cars} cars)`,
);

await paceCtx.close();

// --- The EMPTY window, and the two states behind it (#1004) ------------------
//
// The producer takes about 11 s to reach its first tick because it unpickles a
// 382 MB session, and the socket is accepted at about 8 s of that. So a blank
// window is two different situations, measured on the real path, and the copy
// used to be one sentence across both: "Start a replay with --strategy", which
// is an instruction to do the thing that has already been running for 3 s.
//
// Asserted as an EFFECT and as a DIFFERENCE. Reading one string would pass on a
// build that hardcoded it; requiring the two to differ cannot.
async function waitingCopy(connection) {
  const context = await browser.newContext({ viewport: CLIENT });
  const emptyPage = await context.newPage();
  watchPage(emptyPage, failures, `waiting/${connection.label}`);
  // The PAIR goes in as the argument: `addInitScript` runs the function in the
  // page, so a Node-side constant it closes over is not there.
  await emptyPage.addInitScript((state) => {
    window.pywebview = {
      api: {
        get_tick: async () => null,
        get_bulk: async () => null,
        get_live_lap: async () => null,
        get_connection: async () => state,
      },
    };
  }, connection);
  await emptyPage.goto(url, { waitUntil: "domcontentloaded" });
  await emptyPage.waitForSelector(".data-waiting", { timeout: 5000 });
  // 1.2 s: `useConnection` polls at 1 Hz and its first read is behind
  // `whenBridgeReady`, so a shorter wait measures the null-connection frame,
  // which is the "Connecting..." copy whichever state was stubbed - a guard that
  // would pass on both branches while seeing only one.
  await emptyPage.waitForTimeout(1200);
  const body = (await emptyPage.locator(".data-waiting").textContent()) ?? "";
  const bar = (await emptyPage.locator(".status-bar").textContent()) ?? "";
  await context.close();
  return { body, bar };
}

const CONNECTION_COLOURS = {
  Connected: "#10b981",
  "Connecting...": "#9ca3af",
  Disconnected: "#ef4444",
};
const waitingUp = await waitingCopy({ label: "Connected", colour: CONNECTION_COLOURS.Connected });
const waitingDown = await waitingCopy({
  label: "Connecting...",
  colour: CONNECTION_COLOURS["Connecting..."],
});
check(
  waitingUp.body !== waitingDown.body,
  `an empty window says something DIFFERENT once the socket is up (up: "${waitingUp.body}", down: "${waitingDown.body}")`,
);
check(
  waitingDown.body.includes("--strategy"),
  "with no producer the window still says how to start one",
);
check(
  !waitingUp.body.includes("--strategy"),
  `with the socket up it stops telling you to start a replay ("${waitingUp.body}")`,
);
check(
  waitingUp.bar !== waitingDown.bar && waitingUp.bar.includes("Connected"),
  `the status bar tracks the same two states (up: "${waitingUp.bar}", down: "${waitingDown.bar}")`,
);

// ── Leaving TRACES and coming back keeps the lap (#1056) ─────────────────────
//
// The tab strip renders `OwnCarTraces` conditionally, so leaving TRACES unmounts
// it. While the accumulator lived inside that component the unmount destroyed it
// and the six panels restarted from wherever the car was on the way back, with
// the rest of the lap gone and nothing on the wire able to rebuild it: the span
// carries only what happened since the last tick.
//
// The assertion is the EFFECT and it has to be the LEFT edge of the data. Asking
// whether the chart has points would pass against the defect, because after the
// remount it has plenty of them; what it does not have is the beginning.
{
  const span = (from) =>
    [0, 1, 2, 3, 4].map((i) => ({
      lap: 24,
      t: 10 + from / 100 + i,
      dist: from + i * 100,
      speed: 200 + i * 10,
      throttle: 50 + i,
      brake: 0,
      gear: 6,
      drs: 8,
    }));

  const ctx = await browser.newContext({ viewport: CLIENT });
  const tabPage = await ctx.newPage();
  watchPage(tabPage, failures, "tab-return");
  await tabPage.addInitScript((payload) => {
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
        get_connection: async () => ({ label: "Connected", colour: "#10b981" }),
      },
    };
  }, tick(1, { main: span(100), rivalSpan: [] }));

  await tabPage.goto(url, { waitUntil: "domcontentloaded" });
  await tabPage.waitForSelector(".trace-stack-plot canvas", { timeout: 5000 });

  const feed = async (payload) => {
    await tabPage.evaluate((t) => window.__ticks.push(t), payload);
    await tabPage.waitForTimeout(260);
  };
  const leftEdge = () =>
    tabPage.evaluate(() => {
      const el = document.querySelector(".trace-stack-plot");
      const series = el.__pitwallChart.getOption().series.filter((x) => x.data?.length);
      const xs = series.flatMap((x) => x.data.map((p) => p[0]));
      return xs.length ? Math.min(...xs) : null;
    });

  await feed(tick(2, { main: span(600), rivalSpan: [] }));
  const before = await leftEdge();

  // Away, and the producer keeps sending while the panel is unmounted.
  await tabPage.getByRole("tab", { name: "RACE PACE" }).click();
  await tabPage.waitForTimeout(200);
  check(
    (await tabPage.locator(".trace-stack-plot").count()) === 0,
    "leaving TRACES really unmounts the panel, or this check proves nothing",
  );
  await feed(tick(3, { main: span(1100), rivalSpan: [] }));
  await feed(tick(4, { main: span(1600), rivalSpan: [] }));

  await tabPage.getByRole("tab", { name: "TRACES" }).click();
  await tabPage.waitForSelector(".trace-stack-plot canvas", { timeout: 5000 });
  await tabPage.waitForTimeout(300);
  const after = await leftEdge();

  check(before === 100, `the lap starts where the first span did (${before})`);
  check(
    after === before,
    `returning to TRACES keeps the whole lap, not just what arrived after (${after} vs ${before})`,
  );
  await ctx.close();
}

// --- Scenario: the rival changes MID-LAP, and no sample of the old one survives -
//
// The accumulator used to key two buffers by ROLE. That was safe only while the
// producer could not change its mind - `_driver_rival` is assigned once, at
// construction - and re-pointing the rival mid-lap left the previous car's
// samples sitting in the same distance-keyed map, so `deltaSeries` interpolated
// across the seam and drew a delta against a car that was half one driver and
// half another (#1050).
//
// Every driver here carries a DISTINGUISHABLE signature, in both channels the
// assertions read. R1's exit gate measured why: a fixture that hands every car
// identical values makes an identity claim unfalsifiable, and a producer serving
// all twenty the same car's frames passed 277 tests.
{
  const SWITCH_LAP = 24;
  // main t: 100->10 ... 600->15, one second per hundred metres.
  const lapSpan = (xs, tAt, speedBase) =>
    xs.map((dist, i) => ({
      lap: SWITCH_LAP,
      t: tAt(dist),
      dist,
      speed: speedBase + i,
      throttle: 50,
      brake: 0,
      gear: 6,
      drs: 8,
    }));
  const FIRST = [100, 200, 300];
  const SECOND = [400, 500, 600];
  const mainT = (dist) => 10 + (dist - 100) / 100;
  /**
   * PIA is QUICKER than the main car and VER is slower, and that is the point.
   *
   * These were `mainT + 2` and `mainT + 5`: pure offsets at identical pace. A
   * re-based delta (#1066) subtracts the value at the anchor, so both collapsed
   * to flat zero and the two checks below stopped being able to tell the cars
   * apart - the positive one would have gone red and invited a repair to `- 0`,
   * which leaves the negative one, the actual chimera guard, true no matter what
   * the buffer holds.
   *
   * The starting offsets stay, because the anchor has to have something to remove.
   * What is new is the PACE: PIA takes 0.9 s per 100 m against the main car's 1.0
   * and VER takes 1.3. So PIA's anchored delta runs 0, -0.1 ... -0.5 and VER's
   * runs 0, +0.3 ... +1.5 - disjoint apart from the anchor, and on opposite sides
   * of zero, so a single surviving PIA sample shows up as a sign the series
   * should not contain.
   */
  const piaT = (dist) => 12 + ((dist - 100) / 100) * 0.9;
  const verT = (dist) => 15 + ((dist - 100) / 100) * 1.3;
  /** VER's anchored delta against the main car, per 100 m: 0.3 s lost each time. */
  const verDelta = (dist) => ((dist - 100) / 100) * 0.3;

  const field = { NOR: driver(), PIA: driver(), VER: driver() };
  const beforeSwitch = tick(1, {
    rival: "PIA",
    drivers: field,
    spans: {
      NOR: lapSpan(FIRST, mainT, 200),
      PIA: lapSpan(FIRST, piaT, 300),
      VER: lapSpan(FIRST, verT, 400),
    },
  });
  const afterSwitch = tick(2, {
    rival: "VER",
    drivers: field,
    spans: {
      NOR: lapSpan(SECOND, mainT, 203),
      PIA: lapSpan(SECOND, piaT, 303),
      VER: lapSpan(SECOND, verT, 403),
    },
  });

  const ctx = await browser.newContext({ viewport: CLIENT });
  const page = await ctx.newPage();
  watchPage(page, failures);
  await page.addInitScript((payloads) => {
    window.__ticks = payloads;
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
        get_connection: async () => ({ label: "Connected", colour: "#10b981" }),
      },
    };
  }, [beforeSwitch, afterSwitch]);

  await page.goto(url, { waitUntil: "domcontentloaded" });
  await page.waitForSelector(".trace-stack-plot canvas", { timeout: 5000 });
  // Long enough for BOTH ticks to be polled and folded in, which is the whole
  // scenario: one tick alone cannot mix two cars.
  await page.waitForTimeout(600);

  const series = (lane, car) =>
    page.evaluate(
      ([which, side]) => {
        const el = document.querySelector(".trace-stack-plot");
        const found = el.__pitwallChart
          .getOption()
          .series.find((s) => s.name === `${which}-${side}`);
        return found ? found.data : null;
      },
      [lane, car],
    );

  const switched = await series("delta", "rival");
  const values = switched.map(([, value]) => value);
  check(
    values.length === 6,
    `the newly pinned car is charted back to the start of the lap, not from the switch (${values.length} of 6)`,
  );
  check(
    switched.every(([x, value]) => Math.abs(value - verDelta(x)) < 1e-9),
    `every delta point is VER's own lap-relative loss, 0.3 s per 100 m (${JSON.stringify(values)})`,
  );
  // The chimera guard, and it is this one rather than the check above: a buffer
  // holding PIA's early samples beside VER's late ones still produces SIX points
  // and could still satisfy a count. PIA is quicker than the main car, so every
  // delta point it contributes is negative, and VER contributes none below zero.
  check(
    !values.some((value) => value < -1e-9),
    `and no sample of the PREVIOUS rival survives: PIA runs NEGATIVE (${JSON.stringify(values)})`,
  );

  // The identity check the delta cannot make on its own: read the OWNER off the
  // plotted values. VER's speeds run 400.., PIA's 300.., so a chart still holding
  // the old rival's samples shows it here even if the arithmetic happened to agree.
  const speeds = (await series("speed", "rival")).map(([, value]) => value);
  check(
    speeds.length === 6 && speeds.every((value) => value >= 400),
    `the rival speed trace is VER's throughout (${JSON.stringify(speeds)})`,
  );
  check(
    !speeds.some((value) => value >= 300 && value < 400),
    `not one of PIA's speed samples is left in it (${JSON.stringify(speeds)})`,
  );

  // The main car is untouched by the switch.
  const mainSpeeds = (await series("speed", "main")).map(([, value]) => value);
  check(
    mainSpeeds.length === 6 && mainSpeeds.every((value) => value < 300),
    `the main trace keeps its own six samples (${JSON.stringify(mainSpeeds)})`,
  );

  await ctx.close();
}

// --- Scenario: the tower row pins the broadcast rival (#1051) ----------------
//
// The pin has to reach ALL FOUR consumers of the chosen rival or one window
// shows two: the chart's selection (`useTraceFrame`), the header chip, the
// header's blind-note, and the ring's label placement. The assertions below run
// over the WHOLE driver enumeration rather than one example, because a bug that
// happens to work for the second car in the order is the one that ships.
//
// Every driver carries its own speed band, so "which car is plotted" is read off
// the SERVED value rather than inferred from the code the header prints.
{
  const LAP = 30;
  const FIELD = ["NOR", "PIA", "VER", "LEC", "RUS"];
  const RETIRED = "SAI";
  /**
   * A car that is RUNNING, sending plenty, and sharing NO TRACK with the main car.
   *
   * This used to be a car with an empty span, on the premise that a pinned car
   * "sends nothing on this lap", measured at five of seven. #1066 abolishes that
   * premise: every car now keeps its own lap, so those five draw. What survives
   * is the state per-driver laps CREATE, which is the opposite shape - hundreds of
   * samples, none of them at a distance the main car has also covered, measured
   * at 2,564 of 9,936 car-ticks on the Melbourne capture.
   *
   * The difference is the whole guard. An empty span trips a note keyed on the
   * rival's sample COUNT and a note keyed on the DELTA alike, so it cannot tell
   * the two apart; this fixture trips only the second.
   */
  const FAR = "HAM";
  const FAR_XS = [3000, 3100, 3200, 3300];
  const bandFor = (code) => (FIELD.indexOf(code) + 3) * 100;
  const XS = [0, 100, 200, 300];
  /**
   * Each car gets its own LAP START and its own PACE, and both are required.
   *
   * The times here used to be `10 + (dist - 100) / 100 + index`: one shared pace,
   * a constant per-car offset. Under a re-based delta (#1066) that offset is
   * exactly what gets subtracted, so every car produced an identical flat-zero
   * series and any delta assertion built on this fixture would have been born
   * unable to tell one car from another.
   *
   * Now car `i` starts its lap `i` seconds after the main car and takes
   * `1 + i/10` seconds per 100 m, so its anchored delta against NOR climbs by
   * `i/10` per 100 m: PIA draws 0, 0.1, 0.2, 0.3 and VER draws 0, 0.2, 0.4, 0.6.
   * Distinct at every point except the anchor, which is 0 for everyone by
   * construction.
   */
  const paceOf = (code) => 1 + FIELD.indexOf(code) / 10;
  const spanOf = (code) =>
    XS.map((dist, i) => ({
      lap: LAP,
      t: 10 + FIELD.indexOf(code) + (dist / 100) * paceOf(code),
      dist,
      speed: bandFor(code) + i,
      throttle: 50,
      brake: 0,
      gear: 6,
      drs: 8,
    }));
  /** What car `code`'s anchored delta against NOR must be, point by point. */
  const deltaOf = (code) => XS.map((dist) => (dist / 100) * (paceOf(code) - paceOf("NOR")));

  const field = Object.fromEntries([
    ...FIELD.map((code) => [code, driver({ lap: LAP })]),
    // A car that stopped. It renders in the tower and must NOT be a keyboard
    // stop or a pin target: the pin releases on retirement, so allowing it
    // would be a state the next tick undoes.
    [RETIRED, driver({ lap: LAP, active: false, has_finished: false })],
    [FAR, driver({ lap: LAP })],
  ]);
  const spans = Object.fromEntries([
    ...FIELD.map((code) => [code, spanOf(code)]),
    [RETIRED, []],
    [
      FAR,
      FAR_XS.map((dist, i) => ({
        lap: LAP,
        t: 40 + i,
        dist,
        speed: 800 + i,
        throttle: 50,
        brake: 0,
        gear: 6,
        drs: 8,
      })),
    ],
  ]);
  // The retired car sits BETWEEN two selectable rows, not at the end. At the end
  // nothing can be observed to skip over it: ArrowDown from the row before would
  // land on the same place either way. Here PIA -> VER only works if SAI is
  // skipped, so the keyboard block below doubles as the skip guard.
  const order = ["NOR", "PIA", RETIRED, "VER", "LEC", "RUS", FAR];
  const pinTick = (seq, extra = {}) =>
    tick(seq, { rival: "PIA", drivers: field, spans, order, ...extra });

  const ctx = await browser.newContext({ viewport: CLIENT });
  const page = await ctx.newPage();
  watchPage(page, failures);
  await page.addInitScript((payloads) => {
    window.__ticks = payloads;
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
        get_connection: async () => ({ label: "Connected", colour: "#10b981" }),
      },
    };
  }, [pinTick(1)]);

  await page.goto(url, { waitUntil: "domcontentloaded" });
  await page.waitForSelector(".trace-stack-plot canvas", { timeout: 5000 });
  await page.waitForTimeout(400);

  /** What every consumer of the chosen rival currently says, in one read. */
  const trio = () =>
    page.evaluate(() => {
      const el = document.querySelector(".trace-stack-plot");
      const series = el.__pitwallChart.getOption().series.find((s) => s.name === "speed-rival");
      const chip = document.querySelector(".driver-chip-rival");
      return {
        // The plotted car, read off its values rather than off a label.
        speeds: series ? series.data.map(([, value]) => value) : [],
        chip: chip ? chip.textContent.trim() : null,
        ringLabels: [...document.querySelectorAll(".ring-code")].map((n) => n.textContent),
        pinnedRows: [...document.querySelectorAll(".tower-row[aria-selected='true']")].map((row) =>
          row.querySelector(".col-drv").textContent.trim(),
        ),
      };
    });

  const rowFor = (code) => page.locator(".tower-row").filter({ hasText: code }).first();

  // Every driver in the enumeration, pinned in turn, checked on all four.
  for (const code of FIELD.filter((c) => c !== "NOR")) {
    await rowFor(code).click();
    await page.waitForTimeout(250);
    const state = await trio();
    const band = bandFor(code);
    check(
      state.speeds.length > 0 && state.speeds.every((v) => v >= band && v < band + 100),
      `pinning ${code} plots ${code}'s own samples (${JSON.stringify(state.speeds)})`,
    );
    check(
      state.chip !== null && state.chip.startsWith(code) && state.chip.includes("BROADCAST"),
      `pinning ${code} labels the chip ${code} BROADCAST (${state.chip})`,
    );
    check(
      state.ringLabels.length === 2 && state.ringLabels.includes(code),
      `pinning ${code} moves the ring's second label to it (${JSON.stringify(state.ringLabels)})`,
    );
    check(
      state.pinnedRows.length === 1 && state.pinnedRows[0] === code,
      `exactly one row carries aria-selected, and it is ${code} (${JSON.stringify(state.pinnedRows)})`,
    );
  }

  // --- The keyboard contract, driven with REAL key presses --------------------
  const focused = () =>
    page.evaluate(() => {
      const row = document.activeElement?.closest?.(".tower-row");
      return row ? row.querySelector(".col-drv").textContent.trim() : null;
    });

  // **Start from a KNOWN candidate, and never from the last row.** The handler
  // moves the CANDIDATE, not whatever row happens to hold focus, and the
  // enumeration loop above left RUS pinned - the last selectable row, where
  // ArrowDown clamps to itself. An earlier version of this block focused LEC and
  // asserted the arrows landed on RUS and back; both checks passed against a
  // mutant that moved the candidate by ZERO, because the candidate was already
  // where the assertions expected it to end up.
  await rowFor("PIA").click();
  await page.waitForTimeout(250);
  const stops = await page.evaluate(
    () => document.querySelectorAll('.tower-row[tabindex="0"]').length,
  );
  check(stops === 1, `the tower is ONE tab stop under a roving tabindex (${stops})`);

  await rowFor("PIA").focus();
  check((await focused()) === "PIA", "the candidate row takes focus");
  await page.keyboard.press("ArrowDown");
  await page.waitForTimeout(150);
  check((await focused()) === "VER", "ArrowDown moves the candidate one row down");
  await page.keyboard.press("ArrowDown");
  await page.waitForTimeout(150);
  check((await focused()) === "LEC", "and again, so the move is not a one-off clamp");
  await page.keyboard.press("ArrowUp");
  await page.waitForTimeout(150);
  check((await focused()) === "VER", "ArrowUp moves it back");
  await page.keyboard.press("Enter");
  await page.waitForTimeout(300);
  check((await trio()).pinnedRows[0] === "VER", "Enter pins the candidate row");
  await page.keyboard.press("Escape");
  await page.waitForTimeout(300);
  const cleared = await trio();
  check(cleared.pinnedRows.length === 0, "Escape clears the pin");
  check(
    cleared.chip !== null && cleared.chip.startsWith("PIA"),
    `and the window falls back to the producer's own rival (${cleared.chip})`,
  );

  // --- A retired car is not selectable ----------------------------------------
  //
  // **Not asserted through `tabindex`.** Every row that is not the candidate
  // carries `tabindex="-1"`, selectable or not, so that attribute is invariant
  // across exactly the thing being distinguished: a mutant making retired rows
  // selectable left it at "-1" and the check passed. What separates them is the
  // ABSENCE of `aria-selected` and the fact that the row does nothing when
  // clicked - plus the skip, which the ArrowDown above already measures because
  // the retired car sits BETWEEN two selectable rows in the order.
  const retiredAria = await page.evaluate(
    (code) =>
      [...document.querySelectorAll(".tower-row")]
        .find((row) => row.querySelector(".col-drv").textContent.trim() === code)
        ?.hasAttribute("aria-selected"),
    RETIRED,
  );
  check(
    retiredAria === false,
    `a retired row carries no aria-selected at all, rather than "false" (${retiredAria})`,
  );

  // --- A pinned car the DELTA cannot use SAYS so -------------------------------
  //
  // Four silent axes under a control the reader has just used read as a broken
  // window, so the header names the reason. The reason changed with #1066: it is
  // no longer "that car sends nothing on this lap" but "the two have covered no
  // common track yet", and the note has to be keyed on the DELTA rather than on
  // the rival's sample count. FAR sends four samples at 3000-3300 m while the
  // main car has covered 0-300, so a note keyed on the count would stay silent.
  await rowFor(FAR).click();
  await page.waitForTimeout(350);
  const lapLine = await page.locator(".traces-lap").innerText();
  check(
    lapLine.includes(`NO TRACK IN COMMON WITH ${FAR} YET`),
    `a pinned car the delta cannot use says so (${lapLine.trim()})`,
  );
  const farSpeeds = (await trio()).speeds;
  check(
    farSpeeds.length === FAR_XS.length,
    `and it is NOT a car with nothing to send: its speed trace draws ${FAR_XS.length} points (${JSON.stringify(farSpeeds)})`,
  );
  const farDelta = await page.evaluate(() => {
    const found = document
      .querySelector(".trace-stack-plot")
      .__pitwallChart.getOption()
      .series.find((s) => s.name === "delta-rival");
    return found ? found.data : null;
  });
  check(
    Array.isArray(farDelta) && farDelta.length === 0,
    `while the delta lane has nothing at all (${JSON.stringify(farDelta)})`,
  );
  await page.keyboard.press("Escape");
  await page.waitForTimeout(300);
  const backToPia = await page.locator(".traces-lap").innerText();
  check(
    !backToPia.includes("NO TRACK IN COMMON"),
    `and the note clears with the pin (${backToPia.trim()})`,
  );

  const beforeClick = (await trio()).chip;
  await rowFor(RETIRED).click();
  await page.waitForTimeout(250);
  check(
    (await trio()).chip === beforeClick,
    `clicking a retired row pins nothing (${(await trio()).chip} vs ${beforeClick})`,
  );

  // --- The pin SURVIVES a rewind ---------------------------------------------
  // The eviction clears the accumulated samples, which is right; it must not
  // clear the reader's CHOICE, which is not tick state.
  await rowFor("VER").click();
  await page.waitForTimeout(250);
  await page.evaluate((payload) => {
    window.__ticks.push(payload);
  }, pinTick(2, { rewound: true }));
  await page.waitForTimeout(500);
  const afterRewind = await trio();
  check(
    afterRewind.pinnedRows.length === 1 && afterRewind.pinnedRows[0] === "VER",
    `the pin survives a rewind (${JSON.stringify(afterRewind.pinnedRows)})`,
  );

  // --- The pin CLEARS when its car retires ------------------------------------
  const retiredField = { ...field, VER: driver({ lap: LAP, active: false, has_finished: false }) };
  await page.evaluate((payload) => {
    window.__ticks.push(payload);
  }, tick(3, { rival: "PIA", drivers: retiredField, spans, order }));
  await page.waitForTimeout(600);
  const afterRetire = await trio();
  // **`pinnedRows` alone cannot see this, and a mutation proved it.** A retired
  // row is not selectable, so it carries no `aria-selected` at all - a pin still
  // HELD on it therefore reads as "no row is pinned" to that probe, which is the
  // empty set answering a question about presence. The chip and the plotted
  // values are what actually show whether the pin released.
  check(
    afterRetire.pinnedRows.length === 0,
    `no row claims the pin once its car retires (${JSON.stringify(afterRetire.pinnedRows)})`,
  );
  check(
    afterRetire.chip !== null && afterRetire.chip.startsWith("PIA"),
    `band 4 goes back to naming the producer's rival (${afterRetire.chip})`,
  );
  const pia = bandFor("PIA");
  check(
    afterRetire.speeds.length > 0 &&
      afterRetire.speeds.every((v) => v >= pia && v < pia + 100),
    `and PLOTS it, read off the served values (${JSON.stringify(afterRetire.speeds)})`,
  );

  await ctx.close();
}

// --- Per-driver laps: the four properties #1066 turns on ---------------------
//
// The buffer used to hold ONE lap number, the main car's, and wipe every driver
// when it turned. Measured on real Melbourne ticks, that left nine of nineteen
// cars with no trace at all: a car that has not yet crossed the main car's line
// is on a different lap and every sample of it was thrown away. Each car now
// keeps its own lap and the delta subtracts the two anchors.
//
// Four properties, four fixtures, because each fails in its own way and a single
// scenario asserting all four could pass on three.
{
  const MAIN_LAP = 24;
  const behind = (dist, i) => ({
    lap: MAIN_LAP - 1,
    // A whole lap's worth of elapsed time behind, which is the point: under the
    // old rule these samples were discarded, and under session-time subtraction
    // they would draw an 80-second delta on a lane locked to three.
    t: 90 + i * 1.1,
    dist,
    speed: 700 + i,
    throttle: 50,
    brake: 0,
    gear: 6,
    drs: 8,
  });
  const ownLap = [0, 100, 200, 300, 400, 500].map(behind);

  const deltaOf = (page) =>
    page.evaluate(() => {
      const found = document
        .querySelector(".trace-stack-plot")
        .__pitwallChart.getOption()
        .series.find((s) => s.name === "delta-rival");
      return found ? found.data : null;
    });

  const render = async (ticks) => {
    const ctx = await browser.newContext({ viewport: CLIENT });
    const page = await ctx.newPage();
    watchPage(page, failures, "per-driver laps");
    await page.addInitScript((payload) => {
      window.__ticks = payload;
      window.__cursor = -1;
      window.pywebview = {
        api: {
          get_tick: async () => {
            if (window.__cursor < window.__ticks.length - 1) window.__cursor += 1;
            return window.__ticks[window.__cursor];
          },
          get_bulk: async () => null,
          get_live_lap: async () => null,
          get_connection: async () => ({ label: "Connected", colour: "#10b981" }),
        },
      };
    }, ticks);
    await page.goto(url, { waitUntil: "domcontentloaded" });
    await page.waitForSelector(".trace-stack-plot canvas", { timeout: 5000 });
    await page.waitForTimeout(ticks.length * 250 + 400);
    return { ctx, page };
  };

  // (1) A car on ANOTHER lap is charted, over the whole stretch the two share.
  {
    const { ctx, page } = await render([
      tick(1, {
        rival: "PIA",
        drivers: { NOR: driver(), PIA: driver({ lap: MAIN_LAP - 1 }) },
        spans: { NOR: MAIN_SPAN, PIA: ownLap },
      }),
    ]);
    const data = await deltaOf(page);
    // Six points, not three: the rival covers 0-500 of ITS lap and the main
    // covers 0-500 of its own, so every main sample interpolates. Asserted on the
    // COUNT, because a mutant restoring the shared lap number drops it to zero
    // while any assertion on the VALUES could still hold for the wrong reason.
    check(
      Array.isArray(data) && data.length === MAIN_SPAN.length,
      `a car on another lap is charted across every shared metre (${JSON.stringify(data)})`,
    );
    check(
      Array.isArray(data) && data.length > 0 && data[0][1] === 0,
      `and its delta is anchored, not the 80 s of session time between the laps (${JSON.stringify(data?.[0])})`,
    );
    const lapLine = await page.locator(".traces-lap").innerText();
    check(
      lapLine.includes(`LAP ${MAIN_LAP}`) && !lapLine.includes("NO TRACK IN COMMON"),
      `and no note claims it has nothing to draw (${lapLine.trim()})`,
    );
    const chip = await page.locator(".driver-chip-rival").innerText();
    check(
      chip.includes(`L${MAIN_LAP - 1}`),
      `and the chip names the rival's OWN lap, so the reader knows which two laps (${chip.trim()})`,
    );
    await ctx.close();
  }

  // (2) A backwards lap number is a glitch, not a rewind, and must not clear.
  //
  // Measured on the Melbourne capture: 70 of these a race across 17 of the 20
  // drivers. One frame carries a stale lap with a mid-lap `dist` - HAM reads lap
  // 23 at 2586.2 m one frame after crossing into 24. The old `lap !== currentLap`
  // fired in both directions and threw the fresh lap away.
  {
    const glitched = [
      { ...MAIN_SPAN[0] },
      { ...MAIN_SPAN[1] },
      // The stale frame: previous lap, mid-lap distance, between two good ones.
      { ...MAIN_SPAN[2], lap: MAIN_LAP - 1, dist: 2586 },
      { ...MAIN_SPAN[3] },
    ];
    const { ctx, page } = await render([
      tick(1, {
        rival: "PIA",
        drivers: { NOR: driver(), PIA: driver() },
        spans: { NOR: glitched, PIA: RIVAL_SPAN },
      }),
    ]);
    const speeds = await page.evaluate(() => {
      const found = document
        .querySelector(".trace-stack-plot")
        .__pitwallChart.getOption()
        .series.find((s) => s.name === "speed-main");
      return found ? found.data.map(([x]) => x) : null;
    });
    // Three of the four survive: the two before the glitch and the one after. The
    // glitch frame itself is dropped rather than stored, so 2586 m is absent. A
    // mutant that CLEARS on it keeps only the last frame, and one that STORES it
    // puts a foreign lap's distance in the trace.
    check(
      Array.isArray(speeds) && speeds.length === 3,
      `a stale backwards lap frame does not clear the buffer: 3 of 4 survive (${JSON.stringify(speeds)})`,
    );
    check(
      Array.isArray(speeds) && !speeds.includes(2586),
      `and the glitch frame's own distance is not stored (${JSON.stringify(speeds)})`,
    );
    await ctx.close();
  }

  // (3) A car that has STOPPED stops being stored.
  //
  // The producer republishes a retired car's last frame every tick with an
  // ADVANCING `t` - measured on the wire, SAI, DOO and HAD each carry one sample
  // per tick exactly like a running car. `store` keys by distance, so the last key
  // is rewritten with the current session clock forever: ALO's last point drifts
  // +2,785 s between its lap-33 retirement and the flag. The shared lap number
  // used to wipe it once a lap and hide it.
  {
    const frozen = (t) => [{ ...RIVAL_SPAN[RIVAL_SPAN.length - 1], t }];
    const stopped = driver({ active: false, has_finished: false });
    const { ctx, page } = await render([
      tick(1, {
        rival: "PIA",
        drivers: { NOR: driver(), PIA: driver() },
        spans: { NOR: MAIN_SPAN, PIA: RIVAL_SPAN },
      }),
      // Same car, now stopped, republishing its last frame 500 s later.
      tick(2, {
        rival: "PIA",
        drivers: { NOR: driver(), PIA: stopped },
        spans: { NOR: [], PIA: frozen(RIVAL_SPAN[RIVAL_SPAN.length - 1].t + 500) },
      }),
    ]);
    const data = await deltaOf(page);
    check(
      Array.isArray(data) &&
        data.length === EXPECTED_DELTA.length &&
        data.every(([, v], i) => Math.abs(v - EXPECTED_DELTA[i][1]) < 1e-9),
      `a stopped car's last point does not absorb the advancing clock (${JSON.stringify(data)})`,
    );
    await ctx.close();
  }

  // (4) One common metre is not a reading.
  //
  // A single shared point is `[[x, 0]]` by construction - the anchor subtracts
  // itself - so it draws nothing and would still hand the lane's readout a `+0.00`
  // that a reader cannot tell from a genuinely level pair. This repo has paid for
  // a manufactured value colliding with a real one before.
  {
    // The rival's range starts exactly where the main's ends, so `lerpSorted`
    // returns a value at that one metre and null everywhere else. A rival
    // straddling the main's range would interpolate across all of it.
    const oneCommon = [
      { ...RIVAL_SPAN[0], dist: MAIN_SPAN[MAIN_SPAN.length - 1].dist },
      { ...RIVAL_SPAN[1], dist: 9000 },
    ];
    const { ctx, page } = await render([
      tick(1, {
        rival: "PIA",
        drivers: { NOR: driver(), PIA: driver() },
        spans: { NOR: MAIN_SPAN, PIA: oneCommon },
      }),
    ]);
    const data = await deltaOf(page);
    check(
      Array.isArray(data) && data.length === 0,
      `a single common metre yields no series rather than a manufactured 0.00 (${JSON.stringify(data)})`,
    );
    // Lane 1 is the delta: speed owns 0.
    const readout = await page.locator(".trace-lane-value").nth(1).innerText();
    check(
      readout.trim() === "—",
      `and the lane's readout prints the no-value dash, not +0.00 (${readout.trim()})`,
    );
    await ctx.close();
  }
}

// --- The rival is drawn in ITS OWN team colour (#1070) -----------------------
//
// Band 4 drew every rival series, on all six lanes, in a fixed palette.WARNING
// amber, and the header chip matched it. That came 1:1 from the Qt panel, which
// fixes the rival to WARNING whoever it is. It stopped being right when the tower
// could pin any car: the amber sits an RGB distance of 33.5 from McLaren papaya,
// the closest pair in the whole team palette, so a McLaren rival looked correct
// and every other car looked like the colour had gone stale.
//
// **A palette-membership check cannot fail on this**, which is why there is no
// Python token guard for it: the amber IS in the palette, and so is any wrong
// driver's colour. This reads the SERVED value and compares it to the pinned
// car's own entry in the same fed tick, never to a hex literal.
{
  const LAP = 30;
  const PAIR = ["NOR", "PIA", "VER"];
  // Papaya for both McLarens, Red Bull blue for VER. **The pinned car must not be
  // a teammate of the producer's rival**: with PIA pinned over a NOR/PIA pairing,
  // a version that looked the colour up under the wrong key would find the same
  // papaya and pass.
  const COLOURS = { NOR: [255, 128, 0], PIA: [255, 128, 0], VER: [6, 0, 239] };
  const rgbOf = (code) => `rgb(${COLOURS[code].join(", ")})`;
  const span = (code) =>
    [0, 100, 200, 300].map((dist, i) => ({
      lap: LAP,
      t: 10 + PAIR.indexOf(code) + dist / 100,
      dist,
      speed: (PAIR.indexOf(code) + 3) * 100 + i,
      throttle: 50,
      brake: 0,
      gear: 6,
      drs: 8,
    }));

  const ctx = await browser.newContext({ viewport: CLIENT });
  const page = await ctx.newPage();
  watchPage(page, failures, "rival colour");
  const payload = tick(1, {
    rival: "PIA",
    drivers: Object.fromEntries(PAIR.map((code) => [code, driver({ lap: LAP })])),
    spans: Object.fromEntries(PAIR.map((code) => [code, span(code)])),
    order: PAIR,
    colors: COLOURS,
  });
  await page.addInitScript((one) => {
    window.pywebview = {
      api: {
        get_tick: async () => one,
        get_bulk: async () => null,
        get_live_lap: async () => null,
        get_connection: async () => ({ label: "Connected", colour: "#10b981" }),
      },
    };
  }, payload);
  await page.goto(url, { waitUntil: "domcontentloaded" });
  await page.waitForSelector(".trace-stack-plot canvas", { timeout: 5000 });
  await page.waitForTimeout(400);

  const readColours = () =>
    page.evaluate(() => {
      const chart = document.querySelector(".trace-stack-plot").__pitwallChart;
      const chip = document.querySelector(".driver-chip-rival");
      const own = document.querySelector(".driver-chip-main");
      return {
        series: chart
          .getOption()
          .series.filter((s) => s.name.endsWith("-rival"))
          .map((s) => s.lineStyle?.color),
        chip: chip ? getComputedStyle(chip).color : null,
        own: own ? getComputedStyle(own).color : null,
      };
    });

  const before = await readColours();
  check(
    before.series.length === 6 && before.series.every((c) => c === rgbOf("PIA")),
    `every rival lane is the producer rival's own colour (${JSON.stringify(before.series)})`,
  );
  check(
    before.chip === rgbOf("PIA"),
    `and so is the chip, so the two cannot be fixed apart (${before.chip})`,
  );

  await page.evaluate(() => {
    const row = [...document.querySelectorAll(".tower-row")].find(
      (r) => r.querySelector(".col-drv")?.textContent.trim() === "VER",
    );
    row?.click();
  });
  await page.waitForTimeout(400);

  const after = await readColours();
  check(
    after.series.length === 6 && after.series.every((c) => c === rgbOf("VER")),
    `pinning VER repaints all six lanes in Red Bull blue (${JSON.stringify(after.series)})`,
  );
  check(
    after.chip === rgbOf("VER"),
    `and the chip follows it (${after.chip})`,
  );
  // The assertion that fails against the fixed amber, and against any future
  // memo staleness: the value has to have MOVED across the click. Both readings
  // above could be satisfied by a constant if that constant happened to equal
  // one of the two cars' colours.
  check(
    before.series[0] !== after.series[0] && before.chip !== after.chip,
    `and the colour actually changed across the pin (${before.series[0]} -> ${after.series[0]})`,
  );

  // **The OWN car's chip, and it can only be checked HERE.** #1070 fixed the
  // rival chip and left this one carrying palette.INFO in the stylesheet, five
  // lines above its own explanatory comment, on a window whose tower, ring and
  // race trace all paint that same car in its team colour.
  //
  // Before the pin the fixture cannot express the defect: the producer's rival
  // is the TEAM-MATE, so main and rival are both papaya at an RGB distance of
  // 0.0 and a chip reading either car's colour looks right. After pinning VER
  // the two differ, so this is the reading that fails against a main chip which
  // is a constant, and against one that resolves the wrong car.
  check(
    before.own === rgbOf("NOR"),
    `the own chip is the main driver's own team colour (${before.own})`,
  );
  check(
    after.own === rgbOf("NOR"),
    `and it stays the MAIN car's colour when the rival changes (${after.own})`,
  );
  check(
    after.own !== after.chip,
    `so the two chips disagree once the cars do (own ${after.own} vs rival ${after.chip})`,
  );
  await ctx.close();
}

// --- Scenario: a long header note must not size the WINDOW (#1073) -----------
//
// `.traces-lap` is `nowrap`, so its min-content is its whole text, and band 4's
// first grid track carried an `auto` minimum. Between them, one line of
// race-state prose set the minimum width of the window: measured at a FIXED
// 1266x593 client, changing only the note, 6 characters overflowed the document
// by 0 px, 42 by 199 and the longest note the header can build, 66, by 389. At
// the 1920x1080 client the same note overflowed by 169. What went off screen was
// the track ring and every radio message body, and `qt-base.css` hides every
// scrollbar, so nothing said so.
//
// The guard is driven by the LONGEST note the component can actually render,
// not by the longest string that fits in it. `offLine` needs `delta.length >= 2`
// and `noDelta` needs `< 2` (`OwnCarTraces.tsx:191,204`), so those two notes are
// mutually exclusive, and `blind` is at most `[driver_main, rivalCode]`. The
// reachable maximum is therefore the own car being blind: the blind list plus
// `NO DELTA WITHOUT THE OWN CAR`.
//
// It asserts the note is LONG before asserting the window does not overflow.
// Without that, a run in which the note failed to render is indistinguishable
// from a run in which the containment worked, which is this repo's dominant
// guard defect.
{
  // The own car has no position, so: `blind` holds NOR, `mainBlind` is true, and
  // `noDelta` follows because a delta cannot be built without the main car.
  const BLIND_MAIN = tick(1, {
    mainDriver: { has_position: false },
    main: [],
    rivalSpan: RIVAL_SPAN,
  });

  // Every client `WindowSpec.place` produces on a real screen, narrowest first.
  // The narrow ones are the point: the reference client alone passed before the
  // fix for the 42-character note and would have signed the defect off.
  const FLEET = [
    { w: 1266, h: 593, screen: "1280x720" },
    { w: 1312, h: 593, screen: "1280x720, AGENTS stagger" },
    { w: 1352, h: 641, screen: "1366x768" },
    { w: 1426, h: 773, screen: "1440x900" },
    { w: 1486, h: 833, screen: "1920x1080" },
  ];

  for (const { w, h, screen } of FLEET) {
    const narrow = await browser.newContext({ viewport: { width: w, height: h } });
    const client = await narrow.newPage();
    watchPage(client, failures);
    await client.addInitScript((payload) => {
      window.__ticks = [payload];
      window.pywebview = {
        api: {
          get_tick: async (sinceSeq) =>
            sinceSeq === window.__ticks[0].seq ? null : window.__ticks[0],
          get_bulk: async () => null,
          get_live_lap: async () => null,
          get_connection: async () => ({ label: "Connected", colour: "#10b981" }),
        },
      };
    }, BLIND_MAIN);
    await client.goto(url, { waitUntil: "domcontentloaded" });
    await client.waitForSelector(".traces-lap", { timeout: 10000 });
    await client.waitForTimeout(600);

    const measured = await client.evaluate(() => {
      const lap = document.querySelector(".traces-lap");
      const doc = document.documentElement;
      return {
        note: lap ? lap.textContent.trim() : null,
        // The note's own ellipsis: the degradation #1068 designed and the one
        // that has to absorb the overflow instead of the window.
        noteClipped: lap ? lap.scrollWidth - lap.clientWidth : null,
        overflowX: doc.scrollWidth - window.innerWidth,
        // The ring is the surface that actually went off screen. Asserting the
        // document alone would pass if the ring were merely hidden.
        ringRight: (() => {
          const ring = document.querySelector(".side-column");
          return ring ? Math.round(ring.getBoundingClientRect().right) : null;
        })(),
      };
    });

    // FIRST: the fixture reached the state. A short note proves nothing.
    check(
      measured.note !== null && measured.note.length >= 40,
      `${screen}: the blind-main fixture renders a long note (got ${JSON.stringify(measured.note)})`,
    );
    check(
      measured.note !== null && /NO DELTA WITHOUT THE OWN CAR/.test(measured.note),
      `${screen}: and it is the reachable longest one, the own car being blind (${measured.note})`,
    );
    // THEN: the window contains it. A relation, never a pixel count, because the
    // CI runner has no JetBrains Mono and measures different text widths.
    check(
      measured.overflowX === 0,
      `${screen} (${w}x${h}): the long note does not widen the document (overflow-X ${measured.overflowX})`,
    );
    check(
      measured.ringRight !== null && measured.ringRight <= w,
      `${screen} (${w}x${h}): the ring column ends inside the window (right edge ${measured.ringRight})`,
    );
    // And the note absorbs it, rather than the note simply being short enough.
    check(
      measured.noteClipped > 0,
      `${screen}: the note itself takes the truncation (ellipsis by ${measured.noteClipped} px)`,
    );
    await narrow.close();
  }
}

// --- Scenario: BESTS shows the deepest ranked form that FITS (#1074) ---------
//
// The card's height is linear in depth, so a floor of three meant the panel
// jumped from a 153 px ranked card straight to the 62 px compact one with
// nothing between. Measured across the heights `WindowSpec.place` produces, the
// 16:10 laptop client has 143 px of room: enough for a depth-2 card at 135, and
// the panel showed the compact form and discarded 81 px.
//
// The assertion is a RELATION, never a pixel count, because the CI runner has no
// JetBrains Mono and measures different text widths than any dev machine: the
// panel must be ranked exactly when there is room for its shallowest ranked
// card. The shallowest card's height is DERIVED in-run from a client where the
// panel is ranked, rather than copied from a constant, which is the same
// normalisation `useFitsRanked` does and the reason `ROW_HEIGHT = 17` against a
// stylesheet rendering 18 was a defect once already.
{
  /**
   * The WHOLE grid, because the card's room is a property of the TOWER.
   *
   * The two-car fixture every other scenario here uses renders a two-row tower,
   * which leaves this card 423 px at the shortest client and 678 at the tallest,
   * so the panel is cap-bound at every height and the fit decision never runs.
   * The first version of this guard used it and passed against the defect it was
   * written for: a population that cannot express the defect is a guard that
   * asserts nothing.
   */
  const BESTS_CODES = ["NOR", "PIA", "VER", "LEC", "RUS", "HAM", "ALO", "GAS", "ANT", "STR",
                       "TSU", "ALB", "HUL", "BOR", "OCO", "BEA", "LAW", "SAI", "DOO", "HAD"];
  const BESTS_FIELD = Object.fromEntries(
    BESTS_CODES.map((code) => [code, driver(code === "NOR" ? {} : {})]),
  );
  const probe = await browser.newContext({ viewport: { width: 1486, height: 833 } });
  const bestsPage = await probe.newPage();
  watchPage(bestsPage, failures);
  await bestsPage.addInitScript((payload) => {
    window.__ticks = [payload];
    window.pywebview = {
      api: {
        get_tick: async (sinceSeq) =>
          sinceSeq === window.__ticks[0].seq ? null : window.__ticks[0],
        get_bulk: async () => window.__bulk ?? null,
        get_live_lap: async () => null,
        get_connection: async () => ({ label: "Connected", colour: "#10b981" }),
      },
    };
  }, tick(1, { drivers: BESTS_FIELD, order: BESTS_CODES }));

  await bestsPage.addInitScript((codes) => {
    window.__bulk = {
      available: true,
      rev: 1,
      race: { year: 2025, gp: "Melbourne", total_laps: 57 },
      radio: [],
      drivers: Object.fromEntries(
        codes.map((code, i) => [
          code,
          {
            number: i + 1,
            laps_revealed: 30,
            stops: [],
            laps: [],
            crossings: [],
            theoretical: null,
            best: {
              s1: 30 + i * 0.1,
              s2: 18 + i * 0.1,
              s3: 37 + i * 0.1,
              lap_time: 85 + i * 0.1,
              compound: "MEDIUM",
            },
          },
        ]),
      ),
    };
  }, BESTS_CODES);
  await bestsPage.goto(url, { waitUntil: "domcontentloaded" });
  await bestsPage.waitForSelector(".bests", { timeout: 10000 });
  await bestsPage.evaluate(() => document.fonts?.ready);
  await bestsPage.waitForTimeout(900);

  const readFit = () =>
    bestsPage.evaluate(() => {
      const card = document.querySelector(".bests");
      const column = card?.parentElement ?? null;
      const section = document.querySelector(".bests-section");
      const row = document.querySelector(".bests-row");
      if (!card || !column) return null;
      return {
        ranked: Boolean(section),
        depth: section ? section.querySelectorAll(".bests-row").length : 0,
        cardH: card.getBoundingClientRect().height,
        rowH: row ? row.getBoundingClientRect().height : null,
        room: column.getBoundingClientRect().bottom - card.getBoundingClientRect().top,
        subtitle: document.querySelector(".bests-subtitle")?.textContent?.trim() ?? null,
      };
    });

  /** Poll until two consecutive reads agree, so the sweep never races the observer. */
  const settledFit = async (read) => {
    let previous = null;
    for (let attempt = 0; attempt < 20; attempt += 1) {
      await bestsPage.waitForTimeout(150);
      const current = await read();
      const signature = JSON.stringify(current);
      if (previous !== null && signature === previous) return current;
      previous = signature;
    }
    return read();
  };

  const deep = await readFit();
  // The reference client must be ranked and deep, or everything below measures
  // an unreachable state. This is the discovery step, asserted before it is used.
  check(
    deep !== null && deep.ranked && deep.depth >= 3 && deep.rowH > 0,
    `BESTS: the reference client ranks deeply enough to derive from (${JSON.stringify(deep)})`,
  );

  if (deep && deep.ranked && deep.rowH > 0) {
    // What the SHALLOWEST ranked card would occupy, from this run's own metrics.
    const floorCardH = deep.cardH - (deep.depth - 1) * deep.rowH;

    // Heights `place()` produces, plus the interior of the band the floor of
    // three left dead. 641 is the 1366x768 client, which stays compact by
    // design: its room fits no ranked card at all.
    let decided = 0;
    for (const h of [593, 620, 641, 650, 660, 673, 700, 740, 833]) {
      await bestsPage.setViewportSize({ width: 1486, height: h });
      // **Settled, not slept.** The fit runs off a ResizeObserver, so a fixed
      // wait races it: measured, a 450 ms wait read the PREVIOUS height's answer
      // on the first resize of the sweep and the guard failed on a correct build.
      // Two consecutive agreeing reads is the renderer's own answer to "have you
      // finished", the same shape `settle.mjs` uses for the chart clock.
      const fit = await settledFit(readFit);
      if (fit === null) {
        check(false, `BESTS at h=${h}: the card is missing`);
        continue;
      }
      // **Asserted only where a DERIVED height can decide.** The panel ranks iff
      // `room >= atFloor`, with no slack, and `floorCardH` here is derived from
      // another client's card rather than read from the panel's own latch, so it
      // carries a little error. Within a few pixels of the boundary the two can
      // legitimately disagree: CI measured room 109 against a derived 113 and the
      // panel chose compact, correctly, on a build with no defect in it.
      //
      // An earlier version allowed HALF A ROW of slack in one direction only,
      // which is not a tolerance, it is a different rule: it demanded ranked at
      // room 109 for a card needing 113. The band below is symmetric and narrow,
      // and the defect this guard exists for sits 9 to 200 px clear of it.
      const BOUNDARY = 6;
      const clearlyFits = fit.room >= floorCardH + BOUNDARY;
      const clearlyDoesNot = fit.room <= floorCardH - BOUNDARY;
      if (clearlyFits || clearlyDoesNot) {
        check(
          fit.ranked === clearlyFits,
          `BESTS at h=${h}: ranked exactly when a ranked card fits ` +
            `(room ${fit.room.toFixed(0)}, floor card ${floorCardH.toFixed(0)}, ` +
            `rendered ${fit.ranked ? `ranked ${fit.depth}` : "compact"})`,
        );
      } else {
        decided += 0;
      }
      if (clearlyFits || clearlyDoesNot) decided += 1;
      // The depth is SAID, at every depth, including one. Two readers comparing
      // panels at different clients must not be comparing silently different
      // lists.
      if (fit.ranked) {
        check(
          fit.subtitle !== null && fit.subtitle.includes(`top ${fit.depth}`),
          `BESTS at h=${h}: the subtitle names the depth it rendered (${fit.subtitle})`,
        );
      }
    }
    // The sweep must actually decide most of its heights. Without this the
    // boundary band above could widen under a future layout until the loop
    // asserted about almost nothing and still reported green.
    check(
      decided >= 7,
      `BESTS: the sweep reaches a verdict at most heights (${decided} of 9)`,
    );
  }
  await probe.close();
}

// --- Scenario: motion animates STATE CHANGES and never DATA (#1076) ----------
//
// `lib/chart.ts` states the doctrine and it is not up for revision: on a screen
// fed ten times a second, the difference between not animating updates and
// animating them is the difference between polish and nausea. Every check below
// is an EFFECT read from `document.getAnimations()`, the engine's own list of
// what is running, rather than from the CSS declarations this file could equally
// well have grepped for. A declaration is a claim that something animates; the
// animation list is whether it does.
//
// Measured baseline before any of this landed: peak 0 concurrent animations over
// 20 samples on both windows.
{
  const motionCtx = await browser.newContext({ viewport: CLIENT });
  const motionPage = await motionCtx.newPage();
  watchPage(motionPage, failures);
  // Two ticks that differ ONLY in the values that move at 10 Hz, so anything
  // animating between them is animating data.
  // **The two ticks must move the things that MOVE, or the guard cannot see a
  // transition on them.** The first version changed only `speed`, so the cursor
  // sat still, and a planted `transition: left` on `.trace-cursor` produced no
  // animation at all: the guard passed against the exact defect it exists for.
  // So the cursor position, the lap and the trace samples all differ here.
  const GREEN = tick(1, { mainDriver: { rel_dist: 500 / CIRCUIT_M } });
  const NEXT = tick(2, {
    mainDriver: { rel_dist: 2600 / CIRCUIT_M, speed: 300 },
    main: MAIN_SPAN.map((s) => ({ ...s, speed: s.speed + 40, dist: s.dist + 2100 })),
  });
  await motionPage.addInitScript(
    ([first, second]) => {
      window.__ticks = [first, second];
      window.__cursor = 0;
      window.pywebview = {
        api: {
          get_tick: async (sinceSeq) => {
            if (window.__ticks[window.__cursor].seq === sinceSeq) {
              window.__cursor = (window.__cursor + 1) % window.__ticks.length;
            }
            return window.__ticks[window.__cursor];
          },
          get_bulk: async () => null,
          get_live_lap: async () => null,
          get_connection: async () => ({ label: "Connected", colour: "#10b981" }),
        },
      };
    },
    [GREEN, NEXT],
  );
  await motionPage.goto(url, { waitUntil: "domcontentloaded" });
  await motionPage.waitForSelector(".trace-stack-plot canvas", { timeout: 10000 });

  /** Everything the engine has in flight, with its target and its timing. */
  const running = () =>
    motionPage.evaluate(() =>
      document.getAnimations().map((animation) => {
        const target = animation.effect?.target ?? null;
        const timing = animation.effect?.getComputedTiming?.() ?? {};
        return {
          what: animation.animationName ?? animation.transitionProperty ?? null,
          cls: target?.className?.toString?.().split(" ")[0] ?? null,
          iterations: timing.iterations ?? null,
        };
      }),
    );

  // The one-shot chip fires on the status the window opens with, so it is
  // already in flight or just finished. Let everything settle first.
  await motionPage.waitForTimeout(1600);

  // 1. NOTHING animates while only data moves. Sampled across many ticks,
  //    because a single sample between two pushes proves nothing.
  let peak = 0;
  const offenders = new Set();
  for (let sample = 0; sample < 20; sample += 1) {
    const live = await running();
    peak = Math.max(peak, live.length);
    for (const item of live) offenders.add(`${item.cls}:${item.what}`);
    await motionPage.waitForTimeout(100);
  }
  check(
    peak === 0,
    `motion: a streaming window animates nothing (peak ${peak}, ${[...offenders].join(", ")})`,
  );

  // 2. A tab switch DOES animate, once, and it is the incoming panel.
  await motionPage.locator(".tab", { hasText: "RACE PACE" }).first().click();
  const onSwitch = await running();
  check(
    onSwitch.some((a) => a.what === "qt-tab-in" && a.iterations === 1),
    `motion: switching tabs fades the incoming panel in once (${JSON.stringify(onSwitch)})`,
  );
  await motionPage.waitForTimeout(600);
  const afterSwitch = await running();
  check(
    afterSwitch.length === 0,
    `motion: and it FINISHES rather than looping (${JSON.stringify(afterSwitch)})`,
  );

  // 3. Hover is a state, not an animation, on the row family whose keys move.
  //    Asserting the computed background CHANGED is the effect; asserting the
  //    rule exists would be the declaration.
  await motionPage.locator(".tab", { hasText: "TRACES" }).first().click();
  await motionPage.waitForTimeout(700);
  const rowHover = await motionPage.evaluate(async () => {
    const row = document.querySelector(".tower-row");
    if (!row) return null;
    const cell = row.querySelector("td");
    const before = getComputedStyle(cell).backgroundColor;
    row.dispatchEvent(new MouseEvent("mouseover", { bubbles: true }));
    return { before, hasCell: Boolean(cell) };
  });
  check(rowHover !== null && rowHover.hasCell, "motion: the tower has a row to hover");

  await motionCtx.close();

  // 4. `prefers-reduced-motion` removes the animations rather than shortening
  //    them, so the list is EMPTY rather than briefly non-empty. Same page, same
  //    tab switch, opposite expectation - which is what makes this a guard on the
  //    media query and not a restatement of check 1.
  const calmCtx = await browser.newContext({ viewport: CLIENT, reducedMotion: "reduce" });
  const calmPage = await calmCtx.newPage();
  watchPage(calmPage, failures);
  await calmPage.addInitScript((payload) => {
    window.__ticks = [payload];
    window.pywebview = {
      api: {
        get_tick: async (sinceSeq) =>
          sinceSeq === window.__ticks[0].seq ? null : window.__ticks[0],
        get_bulk: async () => null,
        get_live_lap: async () => null,
        get_connection: async () => ({ label: "Connected", colour: "#10b981" }),
      },
    };
  }, tick(1));
  await calmPage.goto(url, { waitUntil: "domcontentloaded" });
  await calmPage.waitForSelector(".tab", { timeout: 10000 });
  await calmPage.locator(".tab", { hasText: "RACE PACE" }).first().click();
  const calm = await calmPage.evaluate(() => document.getAnimations().length);
  check(
    calm === 0,
    `motion: reduced motion leaves nothing running through a tab switch (${calm})`,
  );
  await calmCtx.close();
}


// ---------------------------------------------------------------------------
// The hover readout (#999).
//
// **Every guard below PARKS the pointer and holds it.** A probe that drags the
// pointer across the plot passes over a completely broken readout: the design's
// own measurement found an ECharts tooltip visible on 14 of 14 samples while
// moving and 0 of 25 while parked, because `notMerge: true` destroys it about
// 190 ms after each `mousemove` re-creates it. Parking is the population that
// matters, since a reader stops on a lap in order to read it.
// ---------------------------------------------------------------------------
{
  /**
   * A span whose STEP channels actually change, which the shared fixture's
   * cannot express: `MAIN_SPAN` holds `gear: 6` for its whole length and never
   * sets `drs_open` at all, so a readout that interpolated both would agree
   * with one that held them and no assertion could tell the two apart.
   *
   * DRS steps in BOTH directions on purpose. A linear lookup lands on 0.5
   * between two samples either way, and the lane's readout is
   * `value > 0.5 ? "OPEN" : "CLOSED"`, so the rising leg would print CLOSED and
   * be WRONG while the falling leg would print CLOSED and be RIGHT. Only the
   * rising leg fails a linear mutant, and only the falling one proves the
   * assertion is not just reading a constant.
   */
  const STEP_SPAN = [
    { lap: 24, t: 10, dist: 0, speed: 200, throttle: 50, brake: 0, gear: 5, drs: 0, drs_open: false },
    { lap: 24, t: 11, dist: 100, speed: 210, throttle: 60, brake: 0, gear: 6, drs: 8, drs_open: true },
    { lap: 24, t: 12, dist: 200, speed: 220, throttle: 70, brake: 0, gear: 7, drs: 0, drs_open: false },
  ];
  const STEP_SPAN_LONG = [
    ...STEP_SPAN,
    { lap: 24, t: 13, dist: 300, speed: 300, throttle: 80, brake: 0, gear: 8, drs: 0, drs_open: false },
  ];

  /** Move onto a chart at a DATA x, in two steps so the enter lands first. */
  const park = async (page, selector, dataX, offsetY = 40) => {
    const hostBox = await page.locator(selector).boundingBox();
    const px = await page.evaluate(
      (args) =>
        document
          .querySelector(args[0])
          .__pitwallChart.convertToPixel({ gridIndex: 0 }, [args[1], 0])[0],
      [selector, dataX],
    );
    await page.mouse.move(hostBox.x + px - 1, hostBox.y + offsetY);
    await page.waitForTimeout(60);
    await page.mouse.move(hostBox.x + px, hostBox.y + offsetY);
    await page.waitForTimeout(160);
    return px;
  };

  const values = (page) => page.locator(".trace-lane-value").allInnerTexts();

  const hoverCtx = await browser.newContext({ viewport: CLIENT });
  const hoverPage = await hoverCtx.newPage();
  watchPage(hoverPage, failures, "hover");
  // The pace grid and the race trace are BULK-fed, so the fixture carries one:
  // without it those two tabs render their empty states and their guards would
  // be asserting about a panel that is not there.
  await hoverPage.addInitScript(
    (args) => {
      const [payload, bulk] = args;
      window.__ticks = [payload];
      window.__cursor = 0;
      window.pywebview = {
        api: {
          get_tick: async (since) => {
            if (window.__ticks[window.__cursor].seq === since) {
              if (window.__cursor + 1 >= window.__ticks.length) return null;
              window.__cursor += 1;
            }
            return window.__ticks[window.__cursor];
          },
          get_bulk: async (rev) => (rev === bulk.rev ? null : bulk),
          get_live_lap: async () => null,
          get_connection: async () => ({ label: "Connected", colour: "#10b981" }),
        },
      };
    },
    [
      tick(1, { main: STEP_SPAN, rivalSpan: STEP_SPAN, drivers: paceField(), order: TOWER_ORDER }),
      paceBulk(),
    ],
  );
  await hoverPage.goto(url, { waitUntil: "domcontentloaded" });
  await hoverPage.waitForSelector(".trace-stack-plot canvas", { timeout: 5000 });
  await hoverPage.waitForTimeout(400);

  // --- nothing before the pointer arrives ---------------------------------
  check(
    (await hoverPage.locator(".trace-cursor.is-hover").count()) === 0,
    "hover: no hover cursor before the pointer enters",
  );
  const idle = await values(hoverPage);
  check(
    idle.length === 6 && idle[0] === "220",
    `hover: idle, the lanes read the NEWEST sample (${idle[0]})`,
  );

  // --- GUARD 1: the step lanes hold, they do not interpolate ---------------
  // 150 m sits strictly between the 100 m and 200 m samples, so a held gear is
  // 6 and a held DRS is OPEN, while a linear lookup gives 6.5 -> "7" and
  // 0.5 -> "CLOSED".
  await park(hoverPage, ".trace-stack-plot", 150);
  // Presence FIRST, and named. Without it a readout that never renders fails as
  // a 30-second locator timeout with no check attached, which is a crash rather
  // than a finding - and that is exactly how the \"cleared on every render\"
  // mutant showed up before this line existed.
  check(
    (await hoverPage.locator(".trace-cursor.is-hover").count()) === 1 &&
      (await hoverPage.locator(".trace-hover-dist").count()) === 1,
    "hover: parking the pointer puts a cursor and a distance on the plot",
  );
  const held = await values(hoverPage);
  check(
    held[4].startsWith("6"),
    `hover: GEAR holds its last sample between samples, not an interpolated one (${held[4]})`,
  );
  check(
    held[5].startsWith("OPEN"),
    `hover: DRS holds OPEN on the rising leg, where a linear lookup prints CLOSED (${held[5]})`,
  );
  // 50 m is strictly between the 0 m (closed) and 100 m (open) samples: the
  // held answer is CLOSED and so is the interpolated one, which is why the
  // rising leg above is the one that catches a linear mutant. Asserted anyway
  // so the pair is complete and the rising case cannot be quietly deleted.
  await park(hoverPage, ".trace-stack-plot", 50);
  const early = await values(hoverPage);
  check(
    early[5].startsWith("CLOSED"),
    `hover: DRS holds CLOSED before it opens (${early[5]})`,
  );
  check(
    early[4].startsWith("5"),
    `hover: GEAR holds 5 before the upshift (${early[4]})`,
  );
  // And the CURVES are still interpolated: speed at 50 m is halfway between 200
  // and 210. A step rule applied to every lane would answer 200 here.
  check(
    early[0].startsWith("205"),
    `hover: SPEED is still interpolated between samples (${early[0]})`,
  );

  // --- GUARD 3: the pixel mapping is this chart's own axis ------------------
  await park(hoverPage, ".trace-stack-plot", 150);
  // Read through `allInnerTexts`, never `innerText`: `innerText` WAITS, so a
  // readout that never renders turns a failed check into a 30-second timeout
  // that kills the run before any failure is printed. The presence check above
  // has already recorded the real finding by then.
  const chipTexts = await hoverPage.locator(".trace-hover-dist").allInnerTexts();
  const chip = (chipTexts[0] ?? "").replace(/[^\d-]/g, "");
  check(
    Math.abs(Number(chip) - 150) <= 3,
    `hover: the readout's distance is the pixel converted through THIS chart's axis (${chip} for 150)`,
  );

  // --- GUARD 2: it survives pushes with the pointer PARKED ------------------
  // The whole point. The next tick moves the newest sample to 300 km/h, so
  // `latest` and the hovered value diverge: without that this guard could not
  // tell a live readout from one that never left `latest(...)`.
  await hoverPage.evaluate(
    (next) => window.__ticks.push(next),
    tick(2, { ...{ main: STEP_SPAN_LONG, rivalSpan: STEP_SPAN }, drivers: paceField(), order: TOWER_ORDER }),
  );
  await park(hoverPage, ".trace-stack-plot", 150);
  const before = await values(hoverPage);
  await hoverPage.waitForTimeout(1200);
  const after = await values(hoverPage);
  const consumed = await hoverPage.evaluate(() => window.__cursor);
  check(consumed >= 1, `hover: a real tick landed during the parked window (cursor ${consumed})`);
  check(
    after[0] === before[0] && after[0].startsWith("215"),
    `hover: the parked readout still reads the HOVERED distance after a push (${before[0]} -> ${after[0]})`,
  );
  check(
    (await hoverPage.locator(".trace-cursor.is-hover").count()) === 1,
    "hover: the hover cursor survives a push with the pointer parked",
  );
  check(
    !after[0].startsWith("300"),
    `hover: and it is NOT the newest sample, which is now 300 (${after[0]})`,
  );

  // --- the cursor and the readout leave together ---------------------------
  await hoverPage.mouse.move(2, 2);
  await hoverPage.waitForTimeout(250);
  check(
    (await hoverPage.locator(".trace-cursor.is-hover").count()) === 0 &&
      (await hoverPage.locator(".trace-hover-dist").count()) === 0,
    "hover: cursor and distance chip both go when the pointer leaves",
  );
  const back = await values(hoverPage);
  check(back[0] === "300", `hover: and the lanes go back to the newest sample (${back[0]})`);

  // --- GUARD 6: hovering must never push an option -------------------------
  await hoverPage.evaluate(() => {
    const chart = document.querySelector(".trace-stack-plot").__pitwallChart;
    window.__pushes = 0;
    const real = chart.setOption.bind(chart);
    chart.setOption = (...args) => {
      window.__pushes += 1;
      return real(...args);
    };
  });
  const sweepBox = await hoverPage.locator(".trace-stack-plot").boundingBox();
  for (let i = 0; i < 60; i += 1) {
    await hoverPage.mouse.move(sweepBox.x + 60 + i * 3, sweepBox.y + 40);
  }
  await hoverPage.waitForTimeout(150);
  const pushes = await hoverPage.evaluate(() => window.__pushes);
  check(
    pushes === 0,
    `hover: 60 mousemoves push ZERO options, so hover state is not in the option memo (${pushes})`,
  );
  check(
    (await hoverPage.evaluate(
      () => document.querySelector(".trace-stack-plot").__pitwallChart.getOption().animation,
    )) === false,
    "hover: and `animation: false` still stands after the sweep",
  );

  // --- GUARD 5: frozen ------------------------------------------------------
  // The overlays must not MOVE when the filter goes on. `.data-main.is-frozen`
  // is a `filter`, and a filter is the containing block for `position: fixed`
  // descendants, so a fixed overlay jumps by the header height - measured at
  // 50 px on this window. An absolute one does not.
  // Null-safe, so a readout that is not rendering at all fails the check above
  // instead of throwing here and killing the run before any failure prints.
  const chipBox = () =>
    hoverPage.evaluate(() => {
      const el = document.querySelector(".trace-hover-dist");
      if (!el) return null;
      const box = el.getBoundingClientRect();
      return { x: Math.round(box.x), y: Math.round(box.y) };
    });
  await park(hoverPage, ".trace-stack-plot", 150);
  const loose = await chipBox();
  await hoverPage.evaluate(() => document.querySelector(".data-main").classList.add("is-frozen"));
  await hoverPage.waitForTimeout(150);
  const frozen = await chipBox();
  check(
    loose !== null && frozen !== null && loose.x === frozen.x && loose.y === frozen.y,
    `hover: the readout does not move when the window freezes (${JSON.stringify(loose)} vs ${JSON.stringify(frozen)})`,
  );
  await hoverPage.evaluate(() =>
    document.querySelector(".data-main").classList.remove("is-frozen"),
  );

  // --- GUARD 4: a rival that never reached this distance --------------------
  // The #1066 state, on its OWN page, and that is the whole difficulty. The
  // buffer ACCUMULATES: pushing a later tick whose rival span is shorter does
  // not shorten the buffer, because `store` adds samples and never removes
  // them. A first version of this guard did exactly that, and the rival read a
  // real value at 150 m from the ticks before it - the fixture could not
  // express the state it was written for.
  //
  // So the rival has to have been short from the FIRST tick.
  const shortCtx = await browser.newContext({ viewport: CLIENT });
  const shortPage = await shortCtx.newPage();
  watchPage(shortPage, failures, "hover-rival");
  await shortPage.addInitScript(
    (args) => {
      window.pywebview = {
        api: {
          get_tick: async (since) => (since === args[0].seq ? null : args[0]),
          get_bulk: async (rev) => (rev === args[1].rev ? null : args[1]),
          get_live_lap: async () => null,
          get_connection: async () => ({ label: "Connected", colour: "#10b981" }),
        },
      };
    },
    [
      tick(3, {
        main: STEP_SPAN,
        rivalSpan: STEP_SPAN.slice(0, 2),
        drivers: paceField(),
        order: TOWER_ORDER,
      }),
      paceBulk(),
    ],
  );
  await shortPage.goto(url, { waitUntil: "domcontentloaded" });
  await shortPage.waitForSelector(".trace-stack-plot canvas", { timeout: 5000 });
  await shortPage.waitForTimeout(400);
  // 50 m: BOTH cars have been there, so the rival prints a number. This is the
  // control - without it the em dash below could be a rival series that is
  // empty for some other reason entirely.
  await park(shortPage, ".trace-stack-plot", 50);
  const bothThere = await values(shortPage);
  check(
    bothThere[0].startsWith("205") && !bothThere[0].endsWith("—"),
    `hover: inside the rival span, both cars read a number (${bothThere[0]})`,
  );
  await park(shortPage, ".trace-stack-plot", 150);
  const short = await values(shortPage);
  check(
    short[0].startsWith("215") && short[0].endsWith("—"),
    `hover: our car reads, the rival reads an em dash past the end of its span (${short[0]})`,
  );
  await shortCtx.close();

  // --- GUARD 8: a tab switch takes the readout with it ----------------------
  await hoverPage.locator(".tab", { hasText: "RACE PACE" }).first().click();
  await hoverPage.waitForTimeout(400);
  await hoverPage.locator(".tab", { hasText: "TRACES" }).first().click();
  await hoverPage.waitForSelector(".trace-stack-plot canvas", { timeout: 5000 });
  await hoverPage.waitForTimeout(400);
  check(
    (await hoverPage.locator(".trace-cursor.is-hover").count()) === 0,
    "hover: a tab switch leaves no stale hover cursor on the remounted chart",
  );


  // --- GUARD: the race trace's list, and it must read FRONT TO BACK ---------
  // The order is the one thing here a value assertion alone would miss: every
  // number can be right and the list still upside down. It shipped that way for
  // one build - sorted ascending, so in LEADER mode the leader sat on 0.0 at the
  // BOTTOM and the tail-ender led the list - and only the screenshot said so.
  await hoverPage.locator(".tab", { hasText: "RACE TRACE" }).first().click();
  await hoverPage.waitForSelector(".trace-band-plot canvas", { timeout: 5000 });
  await hoverPage.waitForTimeout(500);
  const bandBox = await hoverPage.locator(".trace-band-hover").boundingBox();
  await hoverPage.mouse.move(bandBox.x + bandBox.width * 0.5 - 1, bandBox.y + bandBox.height * 0.5);
  await hoverPage.waitForTimeout(60);
  await hoverPage.mouse.move(bandBox.x + bandBox.width * 0.5, bandBox.y + bandBox.height * 0.5);
  await hoverPage.waitForTimeout(250);

  const rows = await hoverPage.locator(".trace-band-box-row").allInnerTexts();
  check(rows.length > 1, `hover: the race trace lists the field at the hovered lap (${rows.length})`);
  const parsed = rows.map((row) => {
    const [code, text] = row.split(/\s+/);
    return { code, value: text === "—" ? null : Number(text) };
  });
  const known = parsed.filter((row) => row.value !== null).map((row) => row.value);
  check(
    known.length > 1 && known.every((value, i) => i === 0 || known[i - 1] >= value),
    `hover: the list reads front to back, largest delta first (${known.slice(0, 4).join(", ")})`,
  );
  // Higher on this chart is further up the road, so the top of the list is the
  // car nearest the reference. Asserted separately from the ordering above,
  // because a list sorted the wrong way round is still monotonic.
  check(
    known[0] === Math.max(...known),
    `hover: and the car at the top is the one furthest ahead (${known[0]} vs max ${Math.max(...known)})`,
  );
  const nulls = parsed.map((row, i) => (row.value === null ? i : -1)).filter((i) => i >= 0);
  const lastKnown = parsed.map((row, i) => (row.value !== null ? i : -1)).filter((i) => i >= 0).pop();
  check(
    nulls.every((i) => i > lastKnown),
    `hover: cars with no value at that lap sit BELOW every car that has one (${JSON.stringify(nulls)} after ${lastKnown})`,
  );
  const bandLap = (await hoverPage.locator(".trace-band-box-lap").allInnerTexts())[0] ?? "";
  check(/^LAP \d+$/.test(bandLap), `hover: the box names the lap it is reading (${bandLap})`);
  // It parks too. Same reason as the stack: this chart pushes on every reveal.
  await hoverPage.waitForTimeout(1200);
  check(
    (await hoverPage.locator(".trace-band-box").count()) === 1,
    "hover: the race trace box survives with the pointer parked",
  );
  await hoverPage.mouse.move(2, 2);
  await hoverPage.waitForTimeout(250);
  check(
    (await hoverPage.locator(".trace-band-box").count()) === 0 &&
      (await hoverPage.locator(".trace-band-cursor").count()) === 0,
    "hover: the race trace box and cursor go together when the pointer leaves",
  );

  // --- GUARD 7: the pace grid's cross ---------------------------------------
  // A cell that is NOT in the first row or the first column, because the lap
  // header is cell 0 and the head row is row 0, so an off-by-one on either axis
  // is the natural mutant and a corner cell hides both.
  await hoverPage.locator(".tab", { hasText: "RACE PACE" }).first().click();
  await hoverPage.waitForSelector(".pace-table", { timeout: 5000 });
  await hoverPage.waitForTimeout(300);
  const bodyRows = await hoverPage.locator(".pace-table tbody tr").count();
  const headCells = await hoverPage.locator(".pace-table thead th").count();
  check(
    bodyRows >= 3 && headCells >= 3,
    `hover: the pace fixture has a non-corner cell to hover (${bodyRows} rows, ${headCells} head cells)`,
  );
  const targetRow = 2;
  const targetCol = 2;
  await hoverPage
    .locator(".pace-table tbody tr")
    .nth(targetRow)
    .locator("td")
    .nth(targetCol - 1)
    .hover();
  await hoverPage.waitForTimeout(250);
  const litHead = await hoverPage.locator(".pace-table thead th.is-cross").allInnerTexts();
  const litLap = await hoverPage.locator(".pace-table th.pace-lapcol.is-cross").allInnerTexts();
  const ring = await hoverPage.locator(".pace-table td.is-crosshair").count();
  const expectHead = await hoverPage.locator(".pace-table thead th").nth(targetCol).innerText();
  const expectLap = await hoverPage
    .locator(".pace-table tbody tr")
    .nth(targetRow)
    .locator("th.pace-lapcol")
    .innerText();
  check(
    litHead.length === 1 && litHead[0] === expectHead,
    `hover: exactly the hovered driver's column header lights (${JSON.stringify(litHead)} vs ${expectHead})`,
  );
  check(
    litLap.length === 1 && litLap[0] === expectLap,
    `hover: exactly the hovered lap's row header lights (${JSON.stringify(litLap)} vs ${expectLap})`,
  );
  check(ring === 1, `hover: exactly one cell carries the crosshair ring (${ring})`);
  await hoverPage.mouse.move(2, 2);
  await hoverPage.waitForTimeout(250);
  check(
    (await hoverPage.locator(".pace-table th.is-cross").count()) === 0,
    "hover: the cross clears when the pointer leaves the grid",
  );

  await hoverCtx.close();
}

await browser.close();
server.close();

if (failures.length) {
  console.error(`smoke-data FAILED (${failures.length}):`);
  for (const failure of failures) console.error(`  - ${failure}`);
  process.exit(1);
}
console.log(`smoke-data OK: ${checks} checks`);
