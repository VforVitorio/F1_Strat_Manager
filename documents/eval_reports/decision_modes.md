# decision_modes

- harness `80f1fa7` · schema v1 · generated 2026-07-29T08:43:26+00:00
- era 2022-2025 · dataset data/raw laps, stratified 6-race subset (RAW, not featured) · seed deterministic · llm none
- artifacts: —

| Metric | Value | Meaning |
| --- | --- | --- |
| Stops scored | 40 of 198 (20.2%) | real green-flag stops the tier could grade |
| Exact lap | 30.0% | chose the lap the team chose |
| Within 1 lap | 37.5% | same call, one lap either side |
| Within 2 laps | 47.5% | same strategic window |
| Mean signed error | -2.23 laps | negative = stops earlier than the team |
| Mean absolute error | 2.48 laps | magnitude |
| Coverage verdict | **masked** | `masked` when under 60% of eligible stops were scored |

### Buckets

| Bucket | Stops |
| --- | --- |
| `closing_laps` | 4 |
| `min_stint` | 22 |
| `no_call_in_window` | 128 |
| `opening_laps` | 4 |
| `scored` | 40 |

`opening_laps` / `closing_laps` / `min_stint` are stops the guard rails make
impossible to agree with, so they are excluded from the headline rather than
counted as misses. `no_data` is a car that had already retired, so the stack
never evaluated the window at all. `no_call_in_window` is different and is the
number to watch: the stack looked and declined to stop anywhere near the real
lap. Charging a retirement to the model as a missed call would flatter neither
side honestly, so the two are never merged.

### Scope

- Sampled races (6 measured): 2023 Barcelona, 2023 Monaco, 2024 Silverstone, 2024 Marina_Bay, 2025 Lusail, 2025 Monza.
- A full sweep of the real-stop sample is roughly 11.5 h of wall clock at
  0.51 s per lap through the stack, so this is a stratified subset by circuit
  archetype and **not** full coverage. Read every figure above as conditional
  on these races.
- Decisions come from `profile="no-llm"`: the deterministic Monte Carlo layer
  plus the guard rails, never the LLM synthesis.
- Agreement with the real pit wall is evidence, not correctness. The team can
  be wrong, and this tier cannot tell when it was.

