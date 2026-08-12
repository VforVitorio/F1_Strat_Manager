/**
 * Wait for an element to stop changing, then check it STAYS stopped.
 *
 * Shared by both smoke scripts because both need it and a second copy is
 * this repo's dominant defect.
 *
 * The question these guards ask is "does this chart redraw itself when no
 * new information arrives?", and the obvious way to ask it - capture, wait,
 * capture, compare - is wrong in a way that only shows up on someone else's
 * machine. A page has a one-off late settle in it (fonts, a resize observer,
 * the scrollbar appearing) that lands at ~1200 ms locally and later on a slow
 * CI runner, so a fixed warm-up either passes by luck or fails for a reason
 * that has nothing to do with the defect. It failed on the runner exactly
 * that way, WITH the fix in place.
 *
 * So: poll until two consecutive captures match - that is quiescence,
 * whenever it arrives - and only then assert it persists. Against a chart
 * that restarts its animation ten times a second, two consecutive captures
 * never match and the wait runs out, which is the failure the guard is for.
 */
import { createHash } from "node:crypto";

const digest = (buffer) => createHash("sha1").update(buffer).digest("hex");

/**
 * @returns the settled hash, or null when the element never stopped moving.
 */
export async function settled(page, locator, { tries = 40, gap = 150 } = {}) {
  let previous = null;
  for (let attempt = 0; attempt < tries; attempt += 1) {
    const hash = digest(await locator.screenshot());
    if (hash === previous) return hash;
    previous = hash;
    await page.waitForTimeout(gap);
  }
  return null;
}

/**
 * True when the element settles and is still identical `holdMs` later.
 *
 * The second half matters: an animation slow enough to look settled between
 * two captures 150 ms apart would still be caught 450 ms on.
 */
export async function staysStill(page, locator, { holdMs = 450, ...options } = {}) {
  const first = await settled(page, locator, options);
  if (first === null) return false;
  await page.waitForTimeout(holdMs);
  return digest(await locator.screenshot()) === first;
}
