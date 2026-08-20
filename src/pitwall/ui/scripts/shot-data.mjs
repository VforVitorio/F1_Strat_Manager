/**
 * Screenshot the built DATA window headless, against a stubbed bridge.
 *
 * A dev tool, not part of the product. It exists because the acceptance test
 * for this port is visual - the Qt telemetry window rendered beside the React
 * one - and because a pywebview window cannot be captured without opening it
 * on somebody's desktop.
 *
 * What it captures is the REAL bundle rendering REAL producer output: the
 * ticks come off `scripts/dev_pitwall_producer.py`, which drives the actual
 * `TelemetryStreamServer` from the actual Melbourne 2025 session, and the
 * only thing faked is `window.pywebview`, which the OS shell would otherwise
 * inject.
 *
 * The input is an ARRAY of consecutive ticks, replayed in order at the
 * window's own poll rate, because band 4 accumulates: one frozen tick draws
 * a trace three samples long and tells you nothing about the panel.
 *
 *   npm run build
 *   node scripts/shot-data.mjs <ticks.json> <out.png> [width] [height]
 *
 * The Qt side of the comparison is captured with `QWidget.grab()` and
 * `Qt.WA_DontShowOnScreen` - never a screen rectangle, which returns
 * whatever is physically in front, and never the `offscreen` platform
 * plugin, which ships no font database on Windows and renders every glyph
 * as tofu.
 */
import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { chromium } from "@playwright/test";
import { serveDist } from "./serve-dist.mjs";

const UI_DIR = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const TICKS = JSON.parse(readFileSync(process.argv[2], "utf-8"));
const OUT = resolve(process.argv[3] ?? resolve(UI_DIR, "data.png"));
// The client area the product really hands this page, NOT the `WindowSpec`
// size. `place()` opens DATA at 1500x870 on the reference desktop and the OS
// keeps 14 px of frame and 37 px of title bar, so the page gets 1486x833. The
// defaults were the outer size, so every capture was 117 px taller and 14 px
// wider than the window it claimed to show. `smoke-data.mjs` was already close
// to the real client, which is how the two disagreed.
const CLIENT = { width: 1486, height: 833 };
const WIDTH = Number(process.argv[4] ?? CLIENT.width);
const HEIGHT = Number(process.argv[5] ?? CLIENT.height);

const ticks = Array.isArray(TICKS) ? TICKS : [TICKS];
const server = await serveDist(resolve(UI_DIR, "dist"));
const browser = await chromium.launch();
const context = await browser.newContext({ viewport: { width: WIDTH, height: HEIGHT } });
const page = await context.newPage();
page.on("console", (message) => console.log(`[page:${message.type()}] ${message.text()}`));
page.on("pageerror", (error) => console.error(`[pageerror] ${error.message}`));

// Monotone, exactly like the smoke's stub: hand back the current tick until
// the window has rendered it, then advance. A stub that returns "anything you
// have not seen" walks backwards and re-feeds spans the panel already evicted.
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
    },
  };
}, ticks);

await page.goto(`http://127.0.0.1:${server.address().port}/data.html`, {
  waitUntil: "domcontentloaded",
});
await page.waitForSelector(".trace-stack-plot canvas", { timeout: 10000 });
// Wait for the LAST tick to be consumed, then shoot almost immediately.
// A fixed trailing margin overshot the status bar's 1.5 s auto-clear, so
// every capture showed an empty bar - a window that looks like its producer
// has died, which is the one state a reference capture must not show by
// accident.
await page.waitForFunction(() => window.__cursor + 1 >= window.__ticks.length, null, {
  timeout: ticks.length * 200 + 5000,
});
await page.waitForTimeout(250);

const consumed = await page.evaluate(() => window.__cursor + 1);
await page.screenshot({ path: OUT, fullPage: false });
console.log(`shot-data: ${consumed}/${ticks.length} ticks replayed -> ${OUT}`);

await context.close();
await browser.close();
server.close();
