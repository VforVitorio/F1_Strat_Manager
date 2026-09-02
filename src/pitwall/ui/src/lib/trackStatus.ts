/**
 * How a track status is WORN, decided once for both windows.
 *
 * The two windows sit side by side on one desk and show the same neutralisation,
 * so a second copy of this rule is a second chance for them to disagree about
 * the race's most decision-changing fact. The connection chip was exactly that
 * and cost a sprint: `Connecting...` rendered amber through one window and dim
 * grey through the other's own CSS.
 *
 * What is shared is the DECISION, not the markup. The two windows have different
 * chip idioms (`.strip-chip` in the DATA strip, `.chip` in the AGENTS header) and
 * forcing one component on both would move a layout choice into a shared module
 * to save a rule that fits in twenty lines.
 *
 * The colour is always the wire's own `track_status_color`, decoded by the
 * producer out of `palette.py`, so neither window spends a constant on it.
 */

export type TrackStatusTreatment =
  | { kind: "unknown"; text: string }
  | { kind: "outline"; text: string; rgb: string }
  | { kind: "filled"; text: string; rgb: string };

/**
 * The three states, in the order they have to be tested.
 *
 * - **No label or no colour is `unknown`**, and it says `NO STATUS` rather than
 *   guessing green. The producer sends `null` when the loader has no entry for
 *   the lap, which is not the same as a green track.
 * - **A frozen feed keeps its LABEL and gives up its WEIGHT.** The track may have
 *   gone red in the seconds since the last tick, and a frozen filled `SAFETY CAR`
 *   would read as a live alarm. An absence is not an alarm either way.
 * - **A non-green status is FILLED.** Before that it was an outline chip swapping
 *   its text, and two captures of the same window, one green and one under the
 *   safety car, differed in no element but that one.
 */
export function trackStatusTreatment(
  label: string | null,
  colour: [number, number, number] | null,
  frozen: boolean,
): TrackStatusTreatment {
  if (!label || !colour) return { kind: "unknown", text: "NO STATUS" };
  if (frozen) return { kind: "unknown", text: label };
  const rgb = `rgb(${colour[0]}, ${colour[1]}, ${colour[2]})`;
  return label === "GREEN"
    ? { kind: "outline", text: label, rgb }
    : { kind: "filled", text: label, rgb };
}
