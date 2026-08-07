"""Race order and at-the-line intervals, read off the replay's own lap crossings.

Two quantities live here and they share one coordinate:

- **where a car is in the race**, as laps completed plus fraction of the
  current lap, which orders the field and says who is a lap down;
- **the interval between two cars**, as the difference of their lap-line
  crossing times, which is what a timing screen shows.

What this replaces was a distance difference divided by a hardcoded
55.56 m/s (200 km/h) for every car, everywhere on track, in every condition:
about 13 % too large on a fastest lap and about 57 % too small under a
Safety Car, on a project whose stated thesis is data fidelity.

Why `dist` is not the coordinate, despite being the obvious one
---------------------------------------------------------------
`FrameData.dist` is race-cumulative metres, so it looks like a race
progress axis and the old code sorted on it. It is not one: **each car
accumulates the distance IT drove**, so two cars at the same corner hold
different numbers, and the difference grows all race. Measured on
Melbourne 2025, the drift reaches **1877 m for a single car and 1469 m
between two cars on a 5220 m circuit** — 28 % of a lap, not the "tens of
metres" an earlier version of this docstring claimed.

Sampled every 500 frames against the timing classification rebuilt from
the crossings, a descending `dist` sort puts **the wrong car in the lead
on 85 % of frames**. The coordinate below gets it wrong on **2.0 %** and
reproduces the whole order exactly on 239 of 305 frames — and of the six
disagreements, all are cars racing within a tenth of a lap of each other,
where "leader" legitimately differs between on-track order and the
at-the-line classification.

`dist` is still what the fraction is *measured* in; what makes it usable
is normalising it per car, by that car's own previous lap, so the drift
cancels instead of accumulating. See `progress`.

The same drift is why `laps_down` cannot be a `dist` difference over a
circuit length: it disagrees with the positional answer on 4.9 % of
same-corner pairs, always in the direction that makes a lapped car read
as a same-lap car.

Why the interval is at the line and not live
--------------------------------------------
A live sub-lap gap needs a track coordinate valid *between* two cars at
different points, and every candidate in `FrameData` was measured and
refuted (against `laps.parquet` line-crossing times, over the full race):

| coordinate | median err | tail |
|---|---|---|
| race-cumulative `dist` | 1.97 s | p95 12.5 s |
| `(lap - 1) + rel_dist` | 5 ms | +-1 lap spikes: `lap` is a rounded interpolation of a step function |
| counting `rel_dist` resets | 6.7 ms | p95 92 s: the resampler interpolates `rel_dist` THROUGH its own reset |
| last crossing of the back car's `rel_dist` | 80 s | worse throughout |
| **line crossings (this module)** | **17 ms** | **p95 101 ms, worst 568 ms, no tail** |

Two honest caveats on that headline figure, both established by an
adversarial gate that reproduced it exactly:

1. **`laps.parquet` is not an independent artefact.** It is `session.laps`,
   the table that slices the very telemetry these arrays come from, so the
   comparison is a self-consistency check, not an external validation. The
   residual is one-sided (+22 ms).
2. **The error budget is not "one frame".** 14.9 % of crossings land more
   than 40 ms from the parquet time and the worst is 486 ms, because the
   `lap` field is `np.interp` plus `round` over a step function rather than
   a true line detector.

It costs one property: the interval updates once a lap and steps at the
line. That is what a real timing screen does, it is labelled `(L)` on
screen, and it beats a live number that is right 90 % of the time and
300 s out for the rest.
"""

from __future__ import annotations

import numpy as np

from src.arcade.config import DT
from src.arcade.data import SessionData


class RaceGapCalculator:
    """Race order and intervals, from the lap-line crossings in `SessionData`.

    Built once per session: finding the crossings walks every driver's
    frames. Each query afterwards is a `searchsorted` or a dict lookup.

    Invariants:

    - A crossing is keyed by the lap it **ends**, so `crossings[7]` is when
      that driver completed lap 7. Verified against `laps.parquet`: 907 of
      907 crossings keyed correctly, no off-by-one, lap 1 included.
    - The **first** frame of each lap increment is kept, not the last. The
      resampled `lap` field can hold an intermediate value for a few frames
      around the line; taking the first occurrence measured p95 error
      83 ms -> 68 ms and systematic lag +25 ms -> +16 ms.
    - Only real lap-field increments are crossings. The chequered flag is
      not one, so the last lap of a race shows the interval from the
      previous line; crediting it as a completed lap would hand every
      retirement a lap it never drove.
    - Unknown is `None`, never a number a caller could also legitimately
      compute. This applies to `laps_down` as much as to `interval_at_line`:
      an earlier version returned `0` for an inverted call, which is also
      what it returns for "same lap".
    """

    def __init__(self, session: SessionData) -> None:
        self._crossings: dict[str, dict[int, float]] = {}
        self._crossing_frames: dict[str, np.ndarray] = {}
        self._dist: dict[str, np.ndarray] = {}
        self._default_lap_m = float(session.circuit_length_m or 0.0)
        for code, frames in session.frames_by_driver.items():
            crossings = self._lap_crossings(frames)
            self._crossings[code] = crossings
            self._crossing_frames[code] = np.array(
                sorted(round(t / DT) for t in crossings.values()), dtype=int
            )
            # `np.maximum.accumulate` removes the float seams the per-lap
            # accumulator leaves at lap boundaries. Measured on Melbourne
            # 2025: every driver carries 9 to 385 backward steps, the worst
            # a single 0.11 m and the worst total deviation 0.60 m across a
            # 300 km race, against a 2.8 m frame at racing speed. It is
            # removing float noise, not physics.
            self._dist[code] = np.maximum.accumulate(
                np.fromiter((f.dist for f in frames), dtype=float, count=len(frames))
            )

    @staticmethod
    def _lap_crossings(frames: list) -> dict[int, float]:
        """Map each completed lap to the replay time the driver crossed the line.

        **Only real increments of the lap field count.** An earlier version
        also recorded the lap the driver was on when their telemetry ended,
        so that the chequered flag had an interval too. That is right for a
        finisher and wrong for everyone else: a car that crashes 1700 m
        into lap 1 has completed no laps, and the synthetic entry credited
        it with one, which put it a whole lap up the order. The final lap
        of a race still renders an interval, taken from the last line both
        cars actually crossed.
        """
        if not frames:
            return {}
        lap_numbers = np.fromiter((f.lap for f in frames), dtype=int, count=len(frames))
        crossings: dict[int, float] = {}
        for i in np.flatnonzero(np.diff(lap_numbers) > 0) + 1:
            crossings.setdefault(int(lap_numbers[i]) - 1, float(i) * DT)
        return crossings

    # --- Where a car is in the race -----------------------------------------

    def laps_completed(self, code: str, frame_idx: int) -> int:
        """How many laps this driver had finished as of `frame_idx`."""
        frames = self._crossing_frames.get(code)
        if frames is None or not len(frames):
            return 0
        return int(np.searchsorted(frames, frame_idx, side="right"))

    def progress(self, code: str, frame_idx: int) -> float | None:
        """Laps completed plus fraction of the current lap, or None if unknown.

        This is the coordinate the field is ordered on, and it is a real
        one. The integer part comes from the line crossings. The fractional
        part is how far the car has driven into its current lap **measured
        against its own previous lap**, which is what makes it comparable
        between cars: each is normalised by the distance it actually
        covers, so the per-car accumulation drift cancels instead of
        accumulating.

        `rel_dist` is deliberately NOT used, even though it looks like
        exactly this number. FastF1 leaves it NaN for a whole driver on
        Melbourne 2025 (HAD), and worse, the 25 Hz resampler clamps it past
        a driver's last real sample: DOO's saturates at 1.000 from frame
        1500 while his `dist` shows him stopped 1717 m into lap 1. Ranking
        on `rel_dist` drew a car that had crashed at turn 1 as the race
        leader for 68 seconds of replay; this form draws him where his
        distance says he is, with no threshold to tune.
        """
        dist = self._dist.get(code)
        if dist is None or frame_idx < 0 or frame_idx >= len(dist):
            return None
        completed = self.laps_completed(code, frame_idx)
        start, lap_length = self._current_lap_bounds(code, completed)
        if lap_length <= 0.0:
            return None
        fraction = (float(dist[frame_idx]) - start) / lap_length
        return completed + min(1.0, max(0.0, fraction))

    def _current_lap_bounds(self, code: str, completed: int) -> tuple[float, float]:
        """Distance at the start of the current lap, and how long that lap is.

        The current lap has no end yet, so its length is taken from the
        driver's previous lap, and from the circuit length on lap 1 where
        there is no previous. Both are that driver's own scale, which is
        the point.
        """
        dist = self._dist[code]
        frames = self._crossing_frames[code]
        start = float(dist[frames[completed - 1]]) if completed > 0 else 0.0
        if completed < len(frames):
            length = float(dist[frames[completed]]) - start
        elif completed > 1:
            length = start - float(dist[frames[completed - 2]])
        else:
            length = self._default_lap_m
        return start, length if length > 0.0 else self._default_lap_m

    @staticmethod
    def laps_down(front_progress: float | None, back_progress: float | None) -> int | None:
        """Whole laps of track separating two cars, 0 when on the same lap.

        Deliberately positional rather than a difference of lap numbers.
        Lap numbers differ by one for the entire window between the two
        cars crossing the line, which is not being lapped, and the live
        `lap` field differs by one for however long one car is ahead within
        the lap. The positional form has neither ambiguity: it asks whether
        the car in front is more than a full lap of track ahead.

        None rather than 0 when either progress is unknown or the pair is
        inverted, because 0 is a legitimate answer meaning "same lap".
        """
        if front_progress is None or back_progress is None:
            return None
        difference = front_progress - back_progress
        if difference < 0.0:
            return None
        return int(difference)

    # --- The interval between two cars --------------------------------------

    def interval_at_line(self, front_code: str, back_code: str, lap: int) -> float | None:
        """Seconds between two cars at the line ending `lap`, front car first.

        Returns None rather than a number when the answer does not exist:
        either car unknown, either car not having completed that lap, or a
        negative result, which means `front_code` was not in front. **That
        last case must not clamp to zero**: zero is a legitimate interval,
        two cars level, so clamping would turn an inverted call into a
        plausible reading instead of a visible gap in the data. Measured on
        Melbourne 2025, an inverted call under such a clamp returned
        `0.000` on five of six sampled laps while the truth was 1 to 25
        seconds the other way.

        Differences within one frame period of zero are reported as 0.0
        rather than None, because the crossing itself is only resolved to a
        frame and a sub-frame negative is noise, not an inversion.
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

    def last_shared_lap(self, front_code: str, back_code: str, frame_idx: int) -> int:
        """The most recent lap both cars had finished as of `frame_idx`.

        Read from the crossings rather than from the live `lap` field,
        which is a rounded interpolation and can sit a few frames away from
        the real crossing. 0 means nobody has completed a lap yet, and no
        crossing map contains lap 0, so the panel reads "N/A" on the
        opening lap instead of inventing an interval.
        """
        return min(
            self.laps_completed(front_code, frame_idx),
            self.laps_completed(back_code, frame_idx),
        )
