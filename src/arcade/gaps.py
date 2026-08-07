"""At-the-line intervals between cars, read off the replay's own lap crossings.

A gap is a time, and the honest way to get one is to compare two cars at the
same point on track. The line is the one point every car passes, the replay
already knows when each of them crossed it, and the difference of those two
instants is the interval a timing screen shows.

What this replaces was a distance difference divided by a hardcoded
55.56 m/s (200 km/h) for every car, everywhere on track, in every condition:
about 13 % too large on a fastest lap and about 57 % too small under a
Safety Car, on a project whose stated thesis is data fidelity.

Why the line and not a live, sub-lap gap
----------------------------------------
A live gap needs a track coordinate the cars share, and `FrameData` does not
offer one. Measured on Melbourne 2025 against `laps.parquet` (an artefact
independent of the telemetry these arrays come from):

- **race-cumulative `dist`** is per car, not per track: each driver
  accumulates the distance *they* drove, so two cars at the same corner hold
  different numbers and the difference grows all race. Median error 1.97 s,
  p95 12.5 s.
- **`(lap - 1) + rel_dist`** looks right at the median (5 ms) but `lap` is a
  rounded interpolation of a step function, so it flips up to a few frames
  away from where `rel_dist` resets and the coordinate spikes by a whole lap.
- **counting `rel_dist` resets** gives a 6.7 ms median and then collapses:
  the resampler interpolates `rel_dist` *through* its own 1 -> 0 reset, so
  the fall is spread over up to 23 frames and a single-step detector misses
  it. p95 92 s.
- **last crossing of the back car's `rel_dist`** was worse still, 80 s median.

The line-crossing interval below is the same quantity all four were trying to
approximate, taken where it is actually observable. Measured over 13,854
driver-pair comparisons across the full race: **median 17 ms, p95 101 ms,
p99 160 ms, worst 568 ms**, with no tail. It costs one property: it updates
once a lap and steps at the line. That is what a real timing screen does, and
it is labelled on screen so nobody reads it as live.
"""

from __future__ import annotations

import numpy as np

from src.arcade.config import DT
from src.arcade.data import SessionData


class RaceGapCalculator:
    """Intervals between cars, from the lap-line crossings in `SessionData`.

    Built once per session: finding the crossings walks every driver's
    frames, and each query afterwards is two dict lookups.

    Invariants:

    - A crossing is keyed by the lap it **ends**, so `crossings[7]` is when
      that driver completed lap 7. The frame is where the `lap` field first
      increments, which is within one frame of the true crossing; that
      +-40 ms is the whole error budget of the interval and it is what the
      measured 17 ms median reflects.
    - Two cars are only comparable on a lap they have **both** finished, so
      the interval a panel can show mid-lap is the one from the last shared
      completed lap. It is not stale data, it is the most recent instant at
      which the two were measurable at the same place.
    """

    def __init__(self, session: SessionData) -> None:
        self._crossings: dict[str, dict[int, float]] = {
            code: self._lap_crossings(frames) for code, frames in session.frames_by_driver.items()
        }
        self._circuit_length_m = float(session.circuit_length_m or 0.0)

    @staticmethod
    def _lap_crossings(frames: list) -> dict[int, float]:
        """Map each completed lap to the replay time the driver crossed the line.

        A jump of more than one lap (a gap in the source telemetry) records
        only the lap it lands on; the laps skipped simply have no crossing
        and every interval that would need them answers None rather than
        interpolating one.
        """
        if not frames:
            return {}
        lap_numbers = np.fromiter((f.lap for f in frames), dtype=int, count=len(frames))
        increments = np.flatnonzero(np.diff(lap_numbers) > 0) + 1
        return {int(lap_numbers[i]) - 1: float(i) * DT for i in increments}

    def interval_at_line(self, front_code: str, back_code: str, lap: int) -> float | None:
        """Seconds between two cars at the line ending `lap`, front car first.

        Returns None rather than a number when the answer does not exist:
        either car unknown, either car not having completed that lap, or a
        negative result, which means `front_code` was not in front. **That
        last case must not clamp to zero**: zero is a legitimate interval,
        two cars level, so clamping would turn an inverted call into a
        plausible reading instead of a visible gap in the data.
        """
        front = self._crossings.get(front_code)
        back = self._crossings.get(back_code)
        if front is None or back is None:
            return None
        front_t = front.get(lap)
        back_t = back.get(lap)
        if front_t is None or back_t is None:
            return None
        seconds = back_t - front_t
        if seconds < -DT:
            return None
        return max(0.0, seconds)

    @staticmethod
    def last_shared_lap(front_lap: int, back_lap: int) -> int:
        """The most recent lap both cars have finished.

        A car showing lap L has completed L - 1. Before anyone has crossed
        the line this is 0, which no crossing map contains, so the panel
        reads "N/A" for the opening lap instead of inventing an interval.
        """
        return max(0, min(int(front_lap), int(back_lap)) - 1)

    def laps_down(self, front_dist: float, back_dist: float) -> int:
        """Whole laps of race distance separating two cars, 0 when on the same lap.

        Deliberately computed from `dist` and not from the two lap numbers:
        lap numbers differ by one for the whole time one car has crossed the
        line and the other has not, which is most of a lap and is not being
        lapped. Race distance does not have that ambiguity, and the per-car
        accumulation drift that ruins it as a live coordinate (tens of
        metres) cannot move a boundary measured in circuit lengths.
        """
        if self._circuit_length_m <= 0.0:
            return 0
        return int(max(0.0, front_dist - back_dist) // self._circuit_length_m)
