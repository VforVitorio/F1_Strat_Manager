/**
 * Running, finished, or out — the one place that predicate exists.
 *
 * It is in `lib/` before its second caller exists, deliberately. Band 4's
 * ring needs it now and sprint 5's timing tower needs the identical
 * predicate, and two components each writing `!active && !has_finished`
 * inline is this repo's dominant defect: one copy of a rule gets fixed and
 * its twin does not. The Qt telemetry panel shipped exactly that pair.
 *
 * **`active` alone is not the answer, and the failure is spectacular.**
 * Measured on the final frame of Melbourne 2025: `!active` reads 19 of the
 * 20 cars as retired, the winner included, because a car that has taken the
 * chequered flag stops broadcasting just like a car that crashed. Only one
 * driver is still `active` at `t_max` (BEA, who owns it). Adding
 * `has_finished` — FastF1's official classification since #879 — recovers
 * the real 13 finished / 6 retired split, which is the official 14/6 once
 * BEA is counted.
 *
 * Mid-race the same frame reads 17 running / 3 OUT / 0 finished, which is
 * the other half of the invariant: `has_finished` is a static fact about
 * the whole race, not a state, so it MUST be gated behind `!active` or
 * every eventual finisher reads as finished from lap 1.
 */

/** A car on track, a car that took the flag, or a car that stopped. */
export type DriverStatus = "running" | "finished" | "out";

export interface StatusInputs {
  active: boolean;
  has_finished: boolean;
}

export function driverStatus(car: StatusInputs): DriverStatus {
  if (car.active) return "running";
  return car.has_finished ? "finished" : "out";
}
