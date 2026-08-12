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

// The delta chart's BASELINE has to cross the whole plot, because every value
// on that chart is read against it and Qt draws it with `pg.InfiniteLine`.
// Measured on a real payload, the `{yAxis: 0}` shorthand stopped at 1679 m of
// a 5220 m axis - and 36 green checks, four of which were about this very
// chart, all passed over it. Only a pixel scan along the zero row sees it.
const baseline = await page.evaluate(() => {
  const el = document.querySelectorAll(".trace-plot")[0];
  const chart = el.__pitwallChart;
  const canvas = el.querySelector("canvas");
  const context = canvas.getContext("2d");
  const ratio = canvas.width / canvas.getBoundingClientRect().width;
  const row = context.getImageData(
    0,
    Math.round(chart.convertToPixel({ yAxisIndex: 0 }, 0) * ratio),
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
  const metres = (px) => chart.convertFromPixel({ xAxisIndex: 0 }, px / ratio);
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
const afterJump = await page.evaluate(() => {
  const el = document.querySelectorAll(".trace-plot")[1];
  return el.__pitwallChart.getOption().series[0].data.map(([x]) => x);
});
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
  (await page.locator(".trace-placeholder").count()) === 1,
  "losing the rival shows the placeholder",
);
await page.evaluate((payload) => window.__ticks.push(payload), tick(5));
await page.waitForTimeout(500);
const revived = await page.evaluate(EXTENTS(0));
check(
  (await page.locator(".trace-plot canvas").count()) === 4,
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
  HAD: driver({ rel_dist: null, has_position: false }), // never placed
};

const ring = await browser.newContext({ viewport: { width: 1500, height: 950 } });
const ringPage = await ring.newPage();
ringPage.on("pageerror", (error) => failures.push(`pageerror(ring): ${error.message}`));

await ringPage.addInitScript((payload) => {
  window.pywebview = {
    api: { get_tick: async (sinceSeq) => (sinceSeq === payload.seq ? null : payload) },
  };
}, tick(1, { drivers: FIELD, order: ["NOR", "PIA", "VER", "HUL", "HAD"] }));

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
check(
  (await ringPage.locator(".ring-blind").innerText()).includes("HAD"),
  "and the ring says which car it is",
);
check(
  (await ringPage.locator(".ring-code").count()) === 2,
  "only the two featured cars are labelled",
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
    },
  };
}, tick(1));
await stillPage.goto(url, { waitUntil: "domcontentloaded" });
await stillPage.waitForSelector(".trace-plot canvas", { timeout: 5000 });

check(
  await staysStill(stillPage, ".trace-plot"),
  "the delta trace schedules no animation while the producer streams",
);

await stillCtx.close();
await browser.close();
server.close();

if (failures.length) {
  console.error(`smoke-data FAILED (${failures.length}):`);
  for (const failure of failures) console.error(`  - ${failure}`);
  process.exit(1);
}
console.log(`smoke-data OK: ${checks} checks`);
