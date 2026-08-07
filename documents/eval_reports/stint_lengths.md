# stint_lengths

- harness `0156673` · schema v1 · generated 2026-08-06T14:34:35+00:00
- era 2022-2025 · dataset data/raw laps 2023-2025 (RAW, not featured) · seed deterministic · llm none
- artifacts: —

## Real green-flag stint lengths by compound (2023-2025 raw laps)

Every completed stint that ended in a real green-flag pit stop, counted in
tyre-age laps: the `TyreLife` reading at the moment of the stop, the exact
field `apply_guard_rails` compares against its minimum-stint bound. A stint
that ended because the race finished, or because the driver retired, is not a
decision to stop and is excluded by construction: neither ever produces a
lap with `PitInTime` set.

The last row is INTERMEDIATE and WET together. They carry no entry of their
own in `_MIN_STINT_LAPS`, so the rail's `.get(compound, _DEFAULT_MIN_STINT)`
resolves them to the fallback -- a minimum-stint bound like any other, and
one this report used to drop rather than measure.

| compound | n | threshold (laps) | shorter than threshold | min | p1 | p5 | p10 | p25 | median | p75 | max |
|---|---|---|---|---|---|---|---|---|---|---|---|
| SOFT | 322 | 2 | 3.4% | 1.0 | 1.0 | 2.0 | 5.0 | 9.0 | 15.0 | 21.0 | 50.0 |
| MEDIUM | 885 | 7 | 4.6% | 1.0 | 1.0 | 7.0 | 9.0 | 14.0 | 19.0 | 25.0 | 55.0 |
| HARD | 535 | 8 | 4.9% | 1.0 | 1.7 | 8.0 | 13.0 | 18.0 | 24.0 | 30.0 | 72.0 |
| WET | 110 | 6 | 4.5% | 2.0 | 3.1 | 6.0 | 7.9 | 11.0 | 12.0 | 19.0 | 44.0 |

"shorter than threshold" is the share of real stints the current guard rail
would have overridden to STAY_OUT had a strategist tried to make that exact
call: `TyreLife < the bound`, the same strict inequality `apply_guard_rails`
itself uses.

This share IS the calibration of a proscriptive bound, and the number the
bounds are set from (#716). A bound exists so a generative model cannot emit
nonsense, so it has to sit where real strategy essentially never goes; once it
is vetoing a meaningful share of what professional pit walls actually did, it
is separating unusual from usual rather than absurd from sane. The ceiling the
bounds are held to is **5%**, and every row above is expected to clear it.

- real green-flag stints counted: 1852 across 70 races
- stops dropped for missing compound/tyre-life data: 5

