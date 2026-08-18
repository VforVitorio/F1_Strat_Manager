/**
 * Lap and sector times, in the one place that owns the arithmetic.
 *
 * **There were three copies and two of them carried a bug the third had already
 * fixed and documented.** `racePace.ts`'s `paceLabel` says it in as many words:
 * splitting the minutes off BEFORE rounding the remainder renders a non-time in
 * the window under every minute boundary, and it rounds first for that reason.
 * `TimingTower`'s `formatLapTime` and `BestsPanel`'s `formatTime` were
 * byte-identical duplicates of each other and neither got the fix. Executed on
 * the shipped arithmetic:
 *
 *     formatLapTime(119.9996) -> "1:60.000"
 *     formatLapTime(59.9996)  -> "60.000"     (should be 1:00.000)
 *
 * The second one is a separate defect in the same three lines and it was not in
 * the finding: the `seconds < 60` branch is decided on the RAW value, so a time
 * that rounds up to a minute takes the no-minutes branch and prints sixty
 * seconds. Both come from the same cause, which is deciding anything before the
 * rounding.
 *
 * At three decimals the window is 0.5 ms per boundary, so no lap on the one race
 * on disk lands in it. Three sector times per lap per car over a season will.
 */

/**
 * `m:ss.ddd`, or `ss.ddd` below a minute unless `alwaysMinutes`.
 *
 * **Rounded FIRST, then split.** Every caller passes the decimals its own surface
 * shows - three for the tower and the bests panel, one for the pace grid, none for
 * the pace grid's narrow form - and the minute test is applied to the ROUNDED
 * value, which is what stops `59.9996` printing as sixty seconds.
 *
 * `alwaysMinutes` is the pace grid's convention: its cells are a fixed-width
 * column and a form that sometimes carries a minute and sometimes does not would
 * change width mid-grid.
 */
export function formatSeconds(seconds: number, decimals: number, alwaysMinutes = false): string {
  const scale = 10 ** decimals;
  const ticks = Math.round(seconds * scale);
  const perMinute = 60 * scale;
  const minutes = Math.floor(ticks / perMinute);
  const rest = (ticks - minutes * perMinute) / scale;
  const width = decimals > 0 ? 3 + decimals : 2;
  if (minutes === 0 && !alwaysMinutes) return rest.toFixed(decimals);
  return `${minutes}:${rest.toFixed(decimals).padStart(width, "0")}`;
}
