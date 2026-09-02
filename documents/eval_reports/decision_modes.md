# decision_modes

- harness `83c59629` · schema v1 · generated 2026-08-30T17:54:11+00:00
- era 2022-2025 · dataset data/raw laps, every 2025 race (RAW, not featured) · seed deterministic · llm none
- artifacts: none

| Metric | Value | Meaning |
| --- | --- | --- |
| Stops scored | 204 of 573 (35.6%) | real green-flag stops the tier could grade |
| Exact lap | 18.6% | chose the lap the team chose |
| Within 1 lap | 34.3% | same call, one lap either side |
| Within 2 laps | 50.0% | same strategic window |
| Mean signed error | -2.21 laps | negative = earlier than the team. **Do not quote as a system property** — it still moves with `DECISION_WINDOW_LAPS`, because a wider window admits more distant, and therefore earlier, transitions. The levels this caveat used to quote (-0.33 / -1.29 / -2.50 at w=3/5/10) were measured on ONE race and on the constant-fed inputs #829 retired, so they are withdrawn rather than restated; the direction is arithmetic and survives, the magnitudes are unmeasured on this input set |
| Mean absolute error | 2.48 laps | magnitude, same width caveat |
| Coverage verdict | **masked** | `masked` when under 60% of eligible stops were scored |

### Buckets

| Bucket | Stops |
| --- | --- |
| `closing_laps` | 4 |
| `min_stint` | 16 |
| `no_boundary_in_window` | 121 |
| `no_call_in_window` | 224 |
| `opening_laps` | 4 |
| `scored` | 204 |

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
- already asking when the window opened, then **withdrawing** later. Re-measured
  on 2025 Monza (2026-08-06, on these inputs): this is 4 of 4 occupants, and all
  four were STILL ASKING on the exact lap the team really stopped, flipping to
  STAY_OUT only the lap after - VER asks laps 32-37 for a lap-37 stop, NOR 41-46
  for 46, HAD 27-32 for 32, PIA 40-45 for 45. So this bucket is holding four
  cases where the stack agreed with the team and the metric cannot say so;
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
was hardcoded to 0.0. Both reach the orchestrator's synthesis prompt, which is
what this tier grades, so every figure published before this fix described a
synthesis told on every lap that the car ahead sat exactly 2.0 s away and matched
its pace. **Figures generated before 2026-08-06 are not comparable to these.**

They do not reach N27, which derives its own pair gap from ``laps_df``, nor the
Monte Carlo, which takes the rivals list from the lap state. An earlier wording
here said they did.

### Scope

- Sampled races (24 measured): 2025 Austin, 2025 Baku, 2025 Barcelona, 2025 Budapest, 2025 Imola, 2025 Jeddah, 2025 Las_Vegas, 2025 Lusail, 2025 Marina_Bay, 2025 Melbourne, 2025 Mexico_City, 2025 Miami_Gardens, 2025 Monaco, 2025 Montréal, 2025 Monza, 2025 Sakhir, 2025 Shanghai, 2025 Silverstone, 2025 Spa-Francorchamps, 2025 Spielberg, 2025 Suzuka, 2025 São_Paulo, 2025 Yas_Island, 2025 Zandvoort.
- **Every one is 2025, deliberately.** 2023 and 2024 are training seasons for
  every model in the stack, so a decision tier scored there is partly reading
  back its own training data.
- This is the whole 2025 season, not a stratified subset. It used to be six
  races, on a runtime estimate of 11.5 h for a full sweep that assumed a
  full-race replay per driver; the tier replays only the scoring windows, so
  the whole season is 8,393 replayed laps at 0.213 s each, about half an hour.
  The subset was representative for the headline, 40.4% declines against 39.1%
  here, and not per race: decline runs from 4.8% at Suzuka to 88.9% at
  Melbourne, only 3 of the 24 races land within three points of the season
  figure, and the retired subset's own six spanned 13.0% to 74.2%. Read any
  per-circuit claim off the per-race numbers, never off this headline.
- Season coverage is **not** the same thing as the coverage verdict above. Every
  eligible 2025 stop is now looked at; the verdict says how many of them the
  metric could locate a decision inside.
- Decisions come from `profile="no-llm"`: the deterministic Monte Carlo layer
  plus the guard rails, never the LLM synthesis.
- Agreement with the real pit wall is evidence, not correctness. The team can
  be wrong, and this tier cannot tell when it was.

