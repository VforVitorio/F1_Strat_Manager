/**
 * Is a chart redrawing itself when no new information has arrived?
 *
 * Shared by both smoke scripts because both need it and a second copy is
 * this repo's dominant defect.
 *
 * **This asks the renderer, not the pixels, and the reason is a failure.**
 * The first version captured the element twice and compared bytes, which is
 * how the defect was actually FOUND: the TIRE card differed at 80, 150, 300
 * and 600 ms after one view landed and only matched itself at ~1200 ms,
 * against a ~100 ms push cadence. But as a CI guard it was unusable - it
 * failed on the runner WITH the fix in place, and no amount of waiting for
 * quiescence helped, because two captures of the same canvas on that machine
 * are not reliably byte-identical to begin with. A guard that cannot pass
 * where it runs is worse than no guard: it teaches everyone to ignore a red.
 *
 * zrender keeps every running animation in one clock, and `isFinished()` is
 * that clock's own answer. With `animation: false` nothing is ever scheduled,
 * so it reports finished on every poll; with the entrance animation live it
 * is mid-flight almost always, because a new `setOption` arrives every 100 ms
 * and the entrance needs about twelve times that. Sampling it repeatedly is
 * what makes the difference unambiguous rather than a coin flip.
 */

/**
 * Fraction of samples in which ANY matched chart had work in flight.
 *
 * Every chart the selector matches, not one of them: both AGENTS cards share
 * `CHART_BASE` and both DATA cells share `TraceChart`, so a guard aimed at a
 * single card would leave its sibling free to regress - which is the shape of
 * the defect this file exists for.
 */
export async function animatingFraction(page, selector, { samples = 12, gap = 60 } = {}) {
  let busy = 0;
  for (let sample = 0; sample < samples; sample += 1) {
    const inFlight = await page.evaluate((query) => {
      const hosts = [...document.querySelectorAll(query)];
      const charts = hosts.map((host) => host.__pitwallChart).filter(Boolean);
      if (!charts.length) return null;
      return charts.some((chart) => {
        const clock = chart.getZr().animation;
        return typeof clock.isFinished === "function" ? !clock.isFinished() : true;
      });
    }, selector);
    // A null means no handle was found at all, which is a broken probe rather
    // than a still chart - say so instead of passing.
    if (inFlight === null) return 1;
    if (inFlight) busy += 1;
    await page.waitForTimeout(gap);
  }
  return busy / samples;
}

/** True when the chart never had an animation in flight across the samples. */
export async function staysStill(page, selector, options = {}) {
  return (await animatingFraction(page, selector, options)) === 0;
}
