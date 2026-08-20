/**
 * Screenshot the built AGENTS window headless, against a stubbed bridge.
 *
 * A dev tool, not part of the product. It exists because the acceptance
 * test for this port is visual - the Qt window rendered beside the React
 * one - and because a pywebview window cannot be captured without opening
 * it on somebody's desktop.
 *
 * What it captures is the REAL bundle rendering REAL host output: the
 * view JSON comes from `AgentsViewBuilder` (see the sibling recipe in
 * `~/.claude/FRONTEND_VISUAL_VERIFICATION.md`), and the only thing faked
 * is `window.pywebview`, which the OS shell would otherwise inject.
 *
 *   npm run build
 *   node scripts/shot-agents.mjs <view.json> <out.png> [width] [height]
 *
 * The Qt side of the comparison is captured with `QWidget.grab()` and
 * `Qt.WA_DontShowOnScreen` - never a screen rectangle, which returns
 * whatever is physically in front, and never the `offscreen` platform
 * plugin, which ships no font database on Windows and renders every
 * glyph as tofu.
 */
import { readFileSync, readdirSync } from "node:fs";
import { createServer } from "node:http";
import { dirname, extname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { chromium } from "@playwright/test";
import { watchPage } from "./page-guard.mjs";

const MIME = {
  ".html": "text/html",
  ".js": "text/javascript",
  ".css": "text/css",
  ".json": "application/json",
};

/**
 * Serve `dist/` over http, from a map read once at startup.
 *
 * Two reasons it is not decoration and not a file server. First,
 * chromium refuses `<script type="module">` over `file://` (CORS treats
 * it as origin `null`), so the bundle loads nothing and the screenshot
 * is a blank page with two console errors. pywebview reaches the same
 * files through the OS webview, which does allow it.
 *
 * Second, nothing here turns a request into a path. The first version
 * joined the URL onto the root and CodeQL called it a path injection -
 * correctly, because a URL can carry `../`. Reading the bundle into
 * memory removes the question rather than guarding it, and a built
 * bundle is a few hundred kilobytes.
 */
function serveDist(root) {
  const files = new Map();
  // `withFileTypes` answers directory-or-file from the directory read
  // itself. Asking again with `stat` is a second look at something that
  // can change in between, which CodeQL calls a file-system race and is
  // right to.
  const walk = (dir, prefix) => {
    for (const entry of readdirSync(dir, { withFileTypes: true })) {
      const full = join(dir, entry.name);
      const url = `${prefix}/${entry.name}`;
      if (entry.isDirectory()) walk(full, url);
      else files.set(url, { body: readFileSync(full), type: MIME[extname(entry.name)] });
    }
  };
  walk(root, "");

  const server = createServer((req, res) => {
    const file = files.get(req.url.split("?")[0]);
    if (!file) {
      res.statusCode = 404;
      res.end();
      return;
    }
    res.setHeader("Content-Type", file.type ?? "application/octet-stream");
    res.end(file.body);
  });
  return new Promise((ready) => server.listen(0, "127.0.0.1", () => ready(server)));
}

const UI_DIR = resolve(dirname(fileURLToPath(import.meta.url)), "..");

// The client area the product really hands this page, NOT the `WindowSpec`
// size. `place()` opens AGENTS at 1500x870 on the reference desktop and the OS
// keeps 14 px of frame and 37 px of title bar, so the page gets 1486x833.
// Shooting the outer size instead means every capture is 67 px taller than the
// window, which is where a vertical overflow would show.
const CLIENT = { width: 1486, height: 833 };

const [viewPath, out, width = String(CLIENT.width), height = String(CLIENT.height)] =
  process.argv.slice(2);

if (!viewPath || !out) {
  console.error("usage: node scripts/shot-agents.mjs <view.json> <out.png> [w] [h]");
  process.exit(2);
}

const view = JSON.parse(readFileSync(viewPath, "utf8"));
const server = await serveDist(resolve(UI_DIR, "dist"));
const bundle = `http://127.0.0.1:${server.address().port}/agents.html`;

const browser = await chromium.launch();
const ctx = await browser.newContext({ viewport: { width: +width, height: +height } });
const page = await ctx.newPage();
// A capture with a console error is not a capture of the product. This used to
// print and carry on, which is how every AGENTS shot since #1004 was taken with
// `useConnection` falling through to `fetch("/api/connection")` and a 404.
const failures = [];
watchPage(page, failures);

await page.addInitScript((payload) => {
  window.pywebview = {
    api: {
      // The host returns null once the window is up to date; the stub does
      // the same, so the poll loop settles instead of re-rendering at 10 Hz
      // under the screenshot.
      get_agents_view: async (sinceSeq) => (sinceSeq >= payload.seq ? null : payload),
      get_tick: async () => null,
      // Polled by `useConnection` while there is no view. Without it the
      // bridge falls back to `fetch("/api/connection")`, the static server
      // answers 404, and the window renders an unknown connection.
      get_connection: async () => "Connected",
    },
  };
}, view);

// `domcontentloaded`, not `networkidle`: a file:// bundle has no network to
// go idle, and Vite's HMR socket keeps `networkidle` from ever firing in dev.
await page.goto(bundle, { waitUntil: "domcontentloaded" });
await page.waitForTimeout(1200);
await page.screenshot({ path: resolve(out), fullPage: false });
await ctx.close();
await browser.close();
server.close();

if (failures.length) {
  console.error(`shot-agents FAILED (${failures.length}):`);
  for (const failure of failures) console.error(`  - ${failure}`);
  process.exit(1);
}
console.log(`saved ${out}`);
