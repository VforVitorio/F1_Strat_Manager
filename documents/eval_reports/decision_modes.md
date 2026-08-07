# decision_modes

- harness `ea9c319` · schema v1 · generated 2026-08-06T18:57:01+00:00
- era 2022-2025 · dataset data/raw laps, stratified 6-race subset (RAW, not featured) · seed deterministic · llm none
- artifacts: —

| Metric | Value | Meaning |
| --- | --- | --- |
| Stops scored | 66 of 178 (37.1%) | real green-flag stops the tier could grade |
| Exact lap | 21.2% | chose the lap the team chose |
| Within 1 lap | 37.9% | same call, one lap either side |
| Within 2 laps | 51.5% | same strategic window |
| Mean signed error | -1.97 laps | negative = earlier than the team. **Do not quote as a system property** — still moves with `DECISION_WINDOW_LAPS` (measured -0.33 / -1.29 / -2.50 at w=3/5/10 on one race), because a wider window admits more distant, and therefore earlier, transitions |
| Mean absolute error | 2.42 laps | magnitude, same width caveat |
| Coverage verdict | **masked** | `masked` when under 60% of eligible stops were scored |

### Buckets

| Bucket | Stops |
| --- | --- |
| `closing_laps` | 4 |
| `min_stint` | 5 |
| `no_boundary_in_window` | 31 |
| `no_call_in_window` | 72 |
| `scored` | 66 |

`opening_laps` / `closing_laps` / `min_stint` are stops the guard rails make
impossible to agree with, so they are excluded from the headline rather than
counted as misses. `no_data` is a car that had already retired, so the stack
never evaluated the window at all. `no_call_in_window` is different and is the
number to watch: the stack looked and declined to stop anywhere near the real
lap. Charging a retirement to the model as a missed call would flatter neither
side honestly, so the two are never merged.

`no_boundary_in_window` is the third case and it is the one this tier used to
get wrong (#752). It means only this: the stack asked to stop somewhere in the
window, but no STAY_OUT -> PIT transition could be located inside it. Read it
as **no locatable decision**, never as a description of what the stack did.
Three different shapes land here and they are not the same finding:

- already asking when the window opened, and still asking on every lap;
- already asking when the window opened, then **withdrawing** later - on the
  measured 2025 Monza sample this was 4 of 4 occupants, one of them flipping to
  STAY_OUT on the exact lap the team really stopped;
- a lap inside the window that was never evaluated, so the only pit ask has no
  witness for its predecessor.

What they share is that the earliest pit ask has no evaluated non-pit lap before
it, so any lap reported would be the window's left edge rather than the model's
choice - which is why `mean_signed_error` used to move with the window width
instead of with the model. A stop here is counted as looked-at and left unscored.
The same applies at the opening guard rail: a transition on lap
5 or earlier is the rail releasing, not the model deciding,
so it is bucketed here too.

### Inputs: the canonical RaceState, since #829

This tier used to build its own ``RaceState`` instead of calling
``src/agents/race_state_builder.build_race_state``, and two of the fields it
invented were constants: ``gap_ahead_s`` came from a key the driver dict does
not carry, so it was **2.0 s on every lap of every race**, and ``pace_delta_s``
was hardcoded to 0.0. Both feed N27's overtake scoring and the Monte Carlo this
tier grades, so every figure published before this fix described a stack fed two
constants. **Figures generated before 2026-08-06 are not comparable to these.**

### Scope

- Sampled races (6 measured): 2025 Barcelona, 2025 Monaco, 2025 Silverstone, 2025 Marina_Bay, 2025 Lusail, 2025 Monza.
- **All six are 2025, deliberately.** 2023 and 2024 are training seasons for
  every model in the stack, so a decision tier scored there is partly reading
  back its own training data. An earlier version of this list took four of its
  six races from those seasons; the archetypes are unchanged, only the year.
- A full sweep of the real-stop sample is roughly 11.5 h of wall clock at
  0.51 s per lap through the stack, so this is a stratified subset by circuit
  archetype and **not** full coverage. Read every figure above as conditional
  on these races.
- Decisions come from `profile="no-llm"`: the deterministic Monte Carlo layer
  plus the guard rails, never the LLM synthesis.
- Agreement with the real pit wall is evidence, not correctness. The team can
  be wrong, and this tier cannot tell when it was.

