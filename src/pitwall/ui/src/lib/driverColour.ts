/**
 * A driver's own colour, off the wire, in the one place that conversion lives.
 *
 * `driver_colors` rides on every tick as an RGB triple precisely so no consumer
 * keeps a second copy of a palette this repo has already found five copies of
 * (#857). That solved the PALETTE and left the CONVERSION duplicated: the tower,
 * the ring and the race trace each had their own three-line `rgb(...)` builder,
 * and band 4 was about to get a fourth when its rival trace stopped being a
 * fixed token (#1070). Three near-identical helpers is how the fourth one gets a
 * different fallback nobody notices.
 *
 * **The fallback is a parameter, not a default, because the three callers
 * genuinely need different ones and each reason is worth keeping:**
 *
 * - the tower wants `--qt-border`, because as a swatch a bright `--qt-fg-1`
 *   reads as a team colour rather than as the absence of one;
 * - the ring wants `--qt-fg-2`, its own dim dot;
 * - anything drawn on an ECharts CANVAS cannot use a CSS custom property at
 *   all - `var(--qt-fg-1)` does not resolve there and the series silently falls
 *   back to ECharts' own palette, the one colour set on this window that answers
 *   to nothing.
 *
 * So a caller that passes a `var(--...)` fallback is promising the value lands
 * in CSS, and a canvas caller must pass a literal.
 */

/** The wire's per-driver palette: `arcade.driver_colors`. */
export type DriverColours = Record<string, [number, number, number]>;

/**
 * `rgb(r, g, b)` for `code`, or `fallback` when the wire has no colour for it.
 *
 * A missing entry is a real state, not a bug to paper over: a car the producer
 * has never placed has no colour, and painting it in some default would claim a
 * team it does not have.
 */
export function driverColour(
  colours: DriverColours,
  code: string | null,
  fallback: string,
): string {
  if (code === null) return fallback;
  const rgb = colours[code];
  return rgb ? `rgb(${rgb[0]}, ${rgb[1]}, ${rgb[2]})` : fallback;
}
