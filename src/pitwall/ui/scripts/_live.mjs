/** Throwaway: screenshot the LIVE host over loopback - real producer, real bundle. */
import { chromium } from "@playwright/test";
const [url, out] = process.argv.slice(2);
const browser = await chromium.launch();
const ctx = await browser.newContext({ viewport: { width: 1485, height: 833 } });
const page = await ctx.newPage();
page.on("pageerror", (e) => console.log(`[pageerror] ${e.message}`));
page.on("console", (m) => { if (m.type() === "error") console.log(`[console] ${m.text()}`); });
await page.goto(url, { waitUntil: "domcontentloaded" });
await page.waitForTimeout(6000);
const state = await page.evaluate(() => ({
  lap: document.querySelector(".header-lap, .head-lap")?.textContent ?? null,
  action: document.querySelector(".orch-badge")?.textContent?.trim() ?? null,
  charts: document.querySelectorAll(".agent-chart").length,
  status: document.querySelector(".status-bar")?.textContent ?? null,
  cardHeights: [...document.querySelectorAll(".agent-card")].map((c) => Math.round(c.getBoundingClientRect().height)),
}));
console.log(JSON.stringify(state));
await page.screenshot({ path: out, fullPage: false });
await ctx.close(); await browser.close();
