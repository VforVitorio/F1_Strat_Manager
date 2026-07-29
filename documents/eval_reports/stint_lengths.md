# stint_lengths

- harness `af96fb6` · schema v1 · generated 2026-07-29T09:39:31+00:00
- era 2022-2025 · dataset data/raw laps 2023-2025 (RAW, not featured) · seed deterministic · llm none
- artifacts: —

## Real green-flag stint lengths by compound (2023-2025 raw laps)

Every completed stint that ended in a real green-flag pit stop, counted in
tyre-age laps: the `TyreLife` reading at the moment of the stop, the exact
field `apply_guard_rails` compares against `_MIN_STINT_LAPS`. A stint that
ended because the race finished, or because the driver retired, is not a
decision to stop and is excluded by construction: neither ever produces a
lap with `PitInTime` set.

| compound | n | threshold (laps) | shorter than threshold | min | p1 | p5 | p10 | p25 | median | p75 | max |
|---|---|---|---|---|---|---|---|---|---|---|---|
| SOFT | 341 | 8 | 15.5% | 1.0 | 1.0 | 2.0 | 5.0 | 9.0 | 15.0 | 21.0 | 50.0 |
| MEDIUM | 896 | 12 | 17.0% | 1.0 | 1.0 | 7.0 | 9.0 | 14.0 | 19.0 | 25.0 | 55.0 |
| HARD | 548 | 15 | 12.2% | 1.0 | 1.9 | 8.0 | 13.0 | 18.0 | 24.0 | 30.0 | 72.0 |

"shorter than threshold" is the share of real stints the current guard rail
would have overridden to STAY_OUT had a strategist tried to make that exact
call: `TyreLife < _MIN_STINT_LAPS[compound]`, the same strict inequality
`apply_guard_rails` itself uses. Close to zero means the bound sits where
real strategy essentially never goes; anywhere else means it is vetoing
calls a real pit wall has actually made.

- real green-flag stints counted: 1785 across 71 races
- wet-compound stops dropped (INTERMEDIATE/WET is not a dry-tyre-life question): 110
- stops dropped for missing compound/tyre-life data: 5

