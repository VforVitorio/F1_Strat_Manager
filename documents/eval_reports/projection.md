# projection

- harness `a97f5dd` · schema v1 · generated 2026-08-06T13:58:14+00:00
- era 2022-2025 · dataset data/raw laps 2025 (RAW, not featured) · seed deterministic · llm none
- artifacts: —

## Position projection against real pit stops

| quantity | value |
|---|---|
| seasons scored | 2025 |
| green-flag stops projected | 552 |
| races covered | 24 |
| within one position | 86.1% |
| exactly right | 59.6% |
| mean signed error (positions) | +0.567 |
| mean absolute error (positions) | 0.668 |

Every real stop is a labelled example of the claim the projection makes, so
this is measured accuracy and not a proxy. Neutralised stops are excluded
because the pit-loss reconstruction, not the projection, is wrong under a
Safety Car.

**The seasons row is not decoration.** 2023 and 2024 are TRAINING seasons for
every model in the stack and 2025 is the holdout the shipped system infers on,
so a figure that mixes them is partly the system reading back its own training
data. Quote the season scope on the same line as the number, always.

## Measured tables the scorer reads

| table | answers | cells |
|---|---|---|
| clean_air | seconds a lap a follower gains once the car directly ahead pits, per circuit | 34 |
| gap_density | seconds between consecutive cars, so a projected gap maps to a place | 12 |
| neutralisation_rate | chance a Safety Car arrives while a stop is being deferred | 29 |
| sc_window | green laps left inside the 5-lap decision window once neutralised | 2 |
| status_mix | share of laps that are actually racing, the denominator for the rest | 5 |
| stop_hazard | chance a rival stops in the window, by tyre life | 36 |
| undercut_band | undercut success against the gap to the target | 15 |

Counted off 70 races (2023, 2024, 2025) of raw laps. The raw parquet is the source
and not the featured one, which drops the neutralised and pit laps these tables
are about.

