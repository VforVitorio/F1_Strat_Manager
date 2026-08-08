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
import { readFileSync, readdirSync, statSync } from "node:fs";
import { createServer } from "node:http";
import { dirname, extname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { chromium } from "@playwright/test";

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
  const walk = (dir, prefix) => {
    for (const entry of readdirSync(dir)) {
      const full = join(dir, entry);
      const url = `${prefix}/${entry}`;
      if (statSync(full).isDirectory()) walk(full, url);
      else files.set(url, { body: readFileSync(full), type: MIME[extname(entry)] });
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
const [viewPath, out, width = "1320", height = "900"] = process.argv.slice(2);

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
page.on("console", (m) => console.log(`[console:${m.type()}] ${m.text()}`));
page.on("pageerror", (e) => console.log(`[pageerror] ${e.message}`));

await page.addInitScript((payload) => {
  window.pywebview = {
    api: {
      // The host returns null once the window is up to date; the stub does
      // the same, so the poll loop settles instead of re-rendering at 10 Hz
      // under the screenshot.
      get_agents_view: async (sinceSeq) => (sinceSeq >= payload.seq ? null : payload),
      get_tick: async () => null,
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
console.log(`saved ${out}`);
