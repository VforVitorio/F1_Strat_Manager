/**
 * Watch a Playwright page for the failures a harness cannot see any other way.
 *
 * **The class this exists for is a 404, not an exception.** `bridge.ts` calls
 * `window.pywebview.api.<method>` when the shell provides one and falls back to
 * `fetch("/api/…")` when it does not - and the fallback swallows a bad status
 * (`if (!response.ok) return null`), because in the product that is a server
 * restarting and null already means "keep what you have". So a harness stub
 * missing a method does not throw and does not fail an assertion: the window
 * quietly renders the unknown state and chromium logs one console error.
 *
 * Four stubs had drifted that way by sprint 10 - `smoke-agents.mjs`'s live
 * page and `shot-agents.mjs` had never gained `get_connection` after #1004
 * wired `useConnection` into the AGENTS window, and `shot-data.mjs` and one
 * `smoke-data.mjs` page carried `get_tick` alone. Only one page per file was
 * listening to the console, so 18 of the 22 pages across the four harnesses
 * could not have reported it.
 *
 * `pageerror` alone is not enough and never was: a failed `fetch` is not a
 * page error.
 */

/**
 * Attach the two listeners every page in a harness needs.
 *
 * @param {import("@playwright/test").Page} page
 * @param {string[]} failures - the harness's own list; a smoke fails on it,
 *   a screenshot script exits non-zero on it.
 * @param {string} [label] - which page this is, for a legible message when a
 *   harness drives a dozen of them.
 */
export function watchPage(page, failures, label = "") {
  const where = label ? `(${label})` : "";
  page.on("pageerror", (error) => failures.push(`pageerror${where}: ${error.message}`));
  page.on("console", (message) => {
    if (message.type() === "error") failures.push(`console${where}: ${message.text()}`);
  });
}
