# 2025 LLM-mode measurement session: running log

Append-only. Written as the session runs so a crash does not lose it.
Date opened: 2026-08-06. Branch base: `dev` @ `0cb150d`.

---

## Step 0: the three thesis races, located and verified

Source: `~/Desktop/Documents/Cuarto Año/TFG/Docs/Memoria/capitulos/05_resultados.tex`, section
5.5 "Casos de uso" (lines 498-560).

| section | race | surface | driver / rival | window | nature of the case |
|---|---|---|---|---|---|
| 5.5.1 | Australia 2025 (Melbourne) | Streamlit | PIA + NOR | V13-V25 | chat + MCP tool calling + report. The thesis itself says the validation "es cualitativa, no cuantitativa" |
| 5.5.2 | Hungary 2025 (Budapest) | CLI | LEC / PIA | V17-V19 | undercut cover on the hard compound; tests DIVERGENCE from the real wall |
| 5.5.3 | Qatar 2025 (Lusail) | Arcade | PIA / VER | V7 | SC pit-window; the press calls it the season's worst wall error |

### The thesis contradiction, RESOLVED, and my first reading of it was wrong

**Correction, made after checking the press against the parquet.** I first wrote that the
thesis section intro ("la parada de Leclerc en V20") was wrong and its body (V19) was right.
That was reading one source. The press accounts of the race number the same two stops
**PIA lap 19, LEC lap 20**; the parquet numbers them **PIA 18, LEC 19**. Neither is an error:
`PitInTime` sits on the lap the car was completing when it entered the pit lane, and the
timing-screen convention the press follows counts the lap the car spends in the pit lane. It
is a one-lap indexing offset, not a factual disagreement, and both describe the identical
event.

What that means for the thesis: the intro is right in press indexing, the body is right in the
indexing the system actually runs on, and only the second one can be used to build a window.
The intro's other claim, "las cinco vueltas", is genuinely inconsistent with the three-lap
command in the same section, and that one is worth fixing.

The press claim the case rests on does hold: Leclerc on the compound, verbatim, *"The hard was
much more difficult, but you discover this a bit too late. We were trying to do the undercut,
and it was working at one stage, you anticipate the pit stop and you can't react."*

### The stops, in the indexing the system runs on

The section intro (line 501) says *"las cinco vueltas en torno a la parada de Leclerc en V20"*.
Section 5.5.2 and the command say **V17-V19**, with Piastri stopping V18 and Ferrari covering
V19.

**The data settles it: the body is right and the intro is wrong.** From
`data/raw/2025/Budapest/pitstops.parquet`, in-lap (`PitInTime` non-null) per driver:

| driver | stop 1 | stop 2 |
|---|---|---|
| VER | 17 | 48 |
| PIA | **18** | 45 |
| LEC | **19** | 40 |
| RUS | 19 | 43 |
| NOR | 31 | - |

Leclerc's covering stop is lap **19**, not 20, and the window is three laps, not five.
The intro paragraph of section 5.5 needs correcting in the thesis source.

### Qatar: which of the three brakes survived

The thesis (5.5.3) reports a first run recommending `STAY_OUT` at 92%, through three brakes,
and then reports a cross-integration module that fixed it. That module **is in the shipped
code today**: `sc_currently_active` is parsed from the RCM stream in
`race_situation_agent.py:1833`, threaded through the lap state, consumed by
`pit_strategy_agent.py:1444` and `:1530`, and it suspends the minimum-stint rail.
Verified again per-brake in step 3 below.

---

## Step 1: the cost probe (DONE)

Command, exactly the thesis command from section 5.5.2:

```
f1-sim Budapest LEC Ferrari --year 2025 --laps 17-19 --rival PIA --provider openai
```

Instrumented with `src/strategy/eval/token_meter.py` (new): patches
`ChatOpenAI.__init__` once on the class object so every one of the five clients the stack
builds is metered, including the sub-agent calls made inside LangGraph's ReAct loop where
there is no call site to edit.

Raw measurement: `documents/audits/PROBE_llm_cost.json`.

| quantity | measured |
|---|---|
| wall clock, total | 72.4 s |
| of which inside `run_lap` | 47.8 s |
| of which boot + render | 24.6 s (radio corpus already cached) |
| **mean seconds per lap** | **15.93 s** (max 17.38) |
| **LLM calls per lap** | **6.0** (5 sub-agent + 1 orchestrator) |
| **prompt tokens per lap** | **8,080** |
| **completion tokens per lap** | **814** |
| cached prompt tokens | **0** |

Per model over the three laps:

| model | role | calls | prompt | completion |
|---|---|---|---|---|
| `gpt-4.1-mini-2025-04-14` | the five sub-agents | 15 | 14,044 | 999 |
| `gpt-5.4-mini-2026-03-17` | the orchestrator (layer 3) | 3 | 10,196 | 1,442 |

**Zero cached prompt tokens.** The prompt is rebuilt per lap and drifts numerically on every
line, so no prefix is stable enough for the provider's cache to hit. Prompt cost is paid in
full on every lap.

### What the probe rules out

The no-llm decision tier replays, per DRIVER, a contiguous span from
`first_stop - 6` to `last_stop + 5`. On a two-stop 2025 race that is roughly 38 laps per
driver, ~760 laps per race with a full grid. At 15.93 s/lap that is **3.4 hours per race**,
so the six-race stratified subset in LLM mode would be **~20 hours** and ~5.4 M prompt tokens.

A per-lap sweep of the whole 2025 season is not on the table either: 22,760 featured laps at
15.93 s is **101 hours**.

**Conclusion carried into the methodology brief: the LLM tier cannot reuse the no-llm tier's
sample. The window has to shrink, the driver set has to shrink, or both, and whichever is
chosen has to be stated on the same line as every number it produces.**

### ⚠️ CORRECTION: the three-lap probe was too short, and three of its numbers are wrong

**Written after the 75-lap repeat run finished.** A probe of three laps is not a distribution,
which is this project's own most-repeated lesson and I walked into it anyway. Measured over the
completed 75-lap run instead:

| quantity | 3-lap probe (wrong) | **75 laps (correct)** |
|---|---|---|
| LLM calls per lap | 6.00 | **7.91** (6.91 sub-agent + 1 orchestrator) |
| prompt tokens per lap | 8,080 | **13,226** |
| completion tokens per lap | 814 | **1,014** |
| cached share of prompt tokens | **0%** | **34.8%** (39.0% on `gpt-4.1-mini`, 23.5% on `gpt-5.4-mini`) |
| cost per lap | $0.0071 | **$0.0080** |

Two distinct errors, and neither was a rounding difference:

1. **"Zero cacheable" was an artefact of the probe's length.** OpenAI's prefix cache only serves
   a prompt it has already seen, and three laps is too short to populate it. Over 75 laps a
   third of all prompt tokens come back cached. The sentence "the prompt drifts numerically
   every lap so no prefix is stable" is wrong about what is cacheable: the drifting part is the
   numeric block, but the sub-agent system prompts and tool schemas in front of it do not move,
   and they are what the cache holds.
2. **Calls per lap is variable and the probe caught a low draw.** The sub-agents are LangGraph
   ReAct loops with an unbounded number of tool turns, so 6.00 was three laps that happened to
   take one turn each. The real mean is 7.91.

**The cost-per-lap conclusion survives** ($0.0080 against $0.0071) because the higher token count
and the caching discount very nearly cancel. That is luck, not method: the two errors happened to
point in opposite directions.

**What does NOT change**: the binding constraint is still wall clock, not money. 1,090 laps is
**$8.72** and hours.

### What the probe costs in money, and the surprise

Published list prices read 2026-08-06 (state them with this date; they move):

| model | input / 1M | output / 1M |
|---|---|---|
| `gpt-4.1-mini` | $0.40 | $1.60 |
| `gpt-5.4-mini` | $0.75 | $4.50 |

Per evaluated lap that is **$0.0024** on the sub-agents plus **$0.0047** on the orchestrator,
so **$0.0071 per lap**, or **$0.43 per hour of running**.

**The binding constraint is WALL CLOCK, not money.** 1,500 laps is about **$11** and about
**6.6 hours**. Any framing of this session as expensive in API spend is wrong; it is expensive
in time. Prompt caching would help the money and not the clock, and it does not apply here
anyway because the measured cached-prompt count is zero.

---

## Step 2: the executor

`scripts/measure_llm_windows.py`. Two decisions worth recording:

1. **It drives the real CLI in-process and wraps `run_lap`**, rather than calling `run_lap`
   itself. Between a bare engine call and the product sit the radio corpus, the Safety Car
   tracker that re-asserts a deployment on the laps between RCMs, and the `DecisionMemory`
   block that enters the orchestrator prompt. Reimplementing those is exactly the
   second-copy-of-race-state-shaping defect this repo keeps paying for, and the numbers would
   then describe the copy.
2. **Every lap is appended to JSONL and flushed on return**, and a restart skips
   `(race, driver, lap, pass)` keys already on disk. Verified: a second invocation over a
   finished spec re-billed nothing and left the row count unchanged.

Rehearsed deterministically over two windows in one process (Budapest 17-19, Monza 20-22):
6 rows, correct shape, `run()` re-entrant across windows.

---

## Step 3: the Qatar case, verified against the data before re-measuring

From `data/raw/2025/Lusail/laps.parquet`, which confirms the thesis narrative:

- `TrackStatus` carries `4` on laps **7, 8, 9, 10**: the Safety Car is out for four laps.
- **16 cars pit on lap 7**, one more on 8 and one on 9. The thesis says 17; the count on the
  lap itself is 16, 18 across the neutralisation.
- **VER pits on lap 7. PIA pits on lap 24, NOR on lap 25.** Both McLarens stayed out, exactly
  as reported.
- PIA's first stint is 24 laps and NOR's is 25, against the Pirelli 25-lap maximum, so the
  forced green-flag stop the thesis describes is visible in the data.

### Which of the three brakes survived: none of them

| brake, as the thesis reports it | state today |
|---|---|
| the SC LightGBM predicts future events, read 10% with an SC deployed | **defused.** `race_situation_agent.py:1809-1835` forces `sc_prob_3lap` to 1.00 on an RCM-confirmed deployment, sets `sc_currently_active`, and says so in the reasoning string. It also forces `overtake_prob` to 0.0 under Art. 55.8 / 56.6 |
| the pit agent's 12-lap minimum stint discarded any stop | **defused twice.** `_MIN_STINT_LAPS` is now `{SOFT: 2, MEDIUM: 7, HARD: 8}` (default 6) after #716, and `apply_guard_rails(..., sc_active=True)` suspends the opening and minimum-stint bounds outright. The end-of-race bound is deliberately NOT suspended, on Art. 55.17 |
| the RAG only activates above 30% SC probability | **defused as a consequence.** `sc_prob_threshold` is still 0.30, but the override puts the input at 1.00, so the condition is met |

So Qatar V7 is a **regression test of a shipped fix**, not an open question. Measured below.

---

## Step 4: the 2025-only projection number, measured directly

`measure_projection_ground_truth` takes a `years` tuple. Measured today on the regenerated,
de-duplicated artefact:

| sample | races | stops | within one | exact | mean signed | mean abs |
|---|---|---|---|---|---|---|
| 2023+2024+2025 (corrected) | 70 | 1,768 | 86.31% | 59.22% | +0.575 | 0.662 |
| **2025 only (the holdout)** | **24** | **552** | **86.05%** | **59.60%** | **+0.567** | **0.668** |
| 2023+2024 (training seasons) | 46 | 1,216 | 86.43% | 59.05% | +0.579 | 0.660 |

Two things to say about it, and the second is the one nobody has said:

1. The published **86.5% / 59.1% / 1,810 / 71** is retired on both counts: it counted the
   duplicated 2023 Spanish GP, and it mixed training seasons into the headline.
2. **The train/holdout gap on this metric is 0.4 points, and that is not a good result, it
   is a statement about what the metric is.** The projection scorer runs on a fixed
   `ProjectionConfig(window_laps=2, racing_laps=2.0, fresh_gain_s=0, cliff_loss_s=0,
   neutralisation_saving_s=0)` and consumes **no learned model and none of the seven measured
   tables**. So the season scope cannot leak into it, which is why restricting to 2025 barely
   moves the number.

That also settles a question the rescope note left open. The seven measured tables
(`clean_air`, `gap_density`, `neutralisation_rate`, `sc_window`, `status_mix`, `stop_hazard`,
`undercut_band`) are counted off three seasons, and the note asked whether re-scoping the
metric without re-scoping the tables is half a fix. **For this metric it is not a fix at all
in either direction: the scorer never reads them.** They feed the served Monte Carlo layer,
which is what the decision tier measures instead. Recorded so nobody re-opens it.

---

## Step 5: the two thesis decision windows, LLM path against deterministic path

Same windows, same data, same process, only `profile` differs. 22 laps each.

| lap | real event | **LLM (`rich`)** | **deterministic (`no-llm`)** | MC score STAY / PIT |
|---|---|---|---|---|
| Budapest 14-18 | | STAY_OUT x5 | STAY_OUT x5 | |
| **Budapest 19** | **LEC pits (Ferrari covers PIA)** | **STAY_OUT** (conf 0.90) | **PIT_NOW** | -1.69 / -1.65 |
| Budapest 20-24 | | STAY_OUT x5 | STAY_OUT x5 | |
| Lusail 4-6 | | STAY_OUT x3 | STAY_OUT x3 | |
| **Lusail 7** | **SC deployed, 16 cars pit** | **STAY_OUT** (conf 0.97) | STAY_OUT | **0.30 / 0.30** |
| **Lusail 8** | SC still out | **STAY_OUT** (0.93) | **PIT_NOW** | -1.37 / +0.04 |
| **Lusail 9** | SC still out | **STAY_OUT** (0.93) | **PIT_NOW** | -1.76 / -0.76 |
| Lusail 10-14 | SC in on 10 | STAY_OUT x5 | STAY_OUT x5 | |

### Finding 1: the LLM layer declined on 22 of 22 laps, and it cost an exact agreement

The deterministic layer chose **lap 19** at Budapest, which is **the exact lap Ferrari stopped**:
offset 0, the best result this metric can produce. The LLM layer, given the same inputs on the
same lap, answered STAY_OUT with confidence 0.90 and the stop was never called. In the decision
tier's own vocabulary the deterministic path scores `scored, offset 0` and the LLM path scores
`no_call_in_window`.

That is the first direct evidence that **the two paths do not agree with each other**, and it
runs in the direction nobody had checked: the layer that ships is more reluctant than the layer
that was measured.

**Confirmed by the scorer rather than by reading the action list** — the distinction matters,
because "emitted PIT_NOW on lap 19" and "the tier's transition rule locates the decision on lap
19" are different statements and only the second is the metric:

```
thesis_windows_nollm.jsonl  Budapest LEC  actual_lap 19  chosen_lap 19  offset 0  bucket 'scored'
thesis_windows.jsonl        Budapest LEC  actual_lap 19  chosen_lap None          bucket 'no_call_in_window'
```

### Finding 2: Qatar. The shipped SC fix works, and the LLM vetoes it on a regulation that does not exist

The deterministic layer **does** call the Safety Car stop, on laps 8 and 9. So the
cross-integration module the thesis describes is alive and doing its job: the RCM is parsed,
`sc_prob_3lap` is forced to 1.00, the minimum-stint rail is suspended, and the Monte Carlo
flips PIT_NOW positive inside the neutralisation. Every one of the thesis's three brakes is
gone.

The LLM layer overrides it. Its own reasoning on laps 7, 8 and 9 says so in as many words:

> "under Article 54.3d) [it is] forbidden [to change] to a different specification while the SC
> orange lights are on, **so we cannot take the MC-favoured PIT_NOW**"

And the `regulation_context` field it was handed, which comes from the RAG agent, reads:

> "During a Safety Car period, it is compulsory that drivers do not change to a different
> specification of tyres or use any other specification of tyres while the Safety Car is on the
> track. A penalty under Article 54.3d) will be imposed if a driver changes tyres to a different
> specification during this time (Article 54.3)."

**No such rule exists.** Searched all three rulebooks in `data/rag/documents/`
(2023: 309,732 chars; 2024: 320,974; 2025: 371,849). Article 54.3 is the **penalties** article:
54.3a) to 54.3g) enumerate penalty TYPES, which is why Art. 4.2 refers to "a penalty applied
under the Code or Article 54.3" and Art. 17.3 forbids appeals against "penalties imposed under
Articles 54.3a) ... 54.3g)". The phrase "different specification" appears six times in the 2025
book and every one is either a replacement-component clause (Art. 40.2) or the two-compound race
requirement (Art. 30.2c). Not one of them is about a Safety Car.

**The empirical refutation is in the same parquet the run reads: sixteen cars, Verstappen
included, changed tyres on lap 7 of this exact Safety Car.** If the rule existed, sixteen cars
took a penalty.

So this is a regulation **fabricated by the RAG agent** out of penalty-article chunks and
tyre-specification chunks, and it is **decisive**: it is the stated reason the flagship case
comes out wrong.

### Finding 3: the Monte Carlo is a point mass on the deployment lap

On Lusail lap 7, the lap the whole case is about, `scenario_scores` reads
`STAY_OUT: E=P10=P90=0.30` and `PIT_NOW: E=P10=P90=0.30`. Identical, zero variance, tie.
UNDERCUT and OVERCUT are ineligible. The layer supplies **no discrimination at all** on the most
consequential lap of the 2025 season, and the argmax picks STAY_OUT on tie order.

A related and separate defect: the LLM's reasoning calls PIT_NOW "MC-favoured" on that lap.
The scores it was given are tied. **The narrative misdescribes the numbers in the same prompt.**

### The three repeat passes, and the correction they forced

I wrote the stability claim with **two** of the three passes on disk. The third refutes part of
it. Corrected:

| window | lap-decisions | STAY_OUT | pit-class |
|---|---|---|---|
| Budapest LEC, 3 passes over both window definitions (14-24 and the exact thesis 17-19) | 42 | **42** | 0 |
| Lusail PIA L4-L14, 3 passes | 33 | 31 | **2** (pass 2, laps 9 and 10) |
| **Lusail lap 7 alone, the SC deployment lap** | 3 | **3** | **0** |

- **Budapest is fully stable.** The exact agreement the deterministic layer achieves is lost on
  every pass, under both window definitions. The window-start and memory confound is therefore
  ruled out: the exact thesis command returns the same answer as the wider window.
- **Qatar needs restating, and the restatement is worse.** Lap 7, the lap the case is about and
  the lap sixteen cars pitted on, is STAY_OUT on every pass (0.86, 0.96, 0.92). The stop is
  asked for on laps 9 and 10 in one pass of three.

### And the sharper version of finding 2

**The fabricated article is in the RAG output on every pass. What is unstable is how the LLM
reads it.** Passes 0 and 1: *"under Article 54.3d) [it is] forbidden ... so we cannot take the
MC-favoured PIT_NOW"*. Pass 2, recommending PIT_NOW: *"Regulation Article 54.3 allows the stop
only once the Safety Car orange lights are extinguished"*. Same non-existent rule, read once as
a prohibition and once as a timing condition.

That is a stronger argument for grounding the citation than a stable veto would have been: a
stable veto is a bug with a fixed sign, and this is a bug with a random one.

### Stability of the planning fields, over 22 repeated laps

| field | laps differing across passes |
|---|---|
| `action` | **9.1%** |
| `compound_next` | 50.0% |
| `pit_lap_target` | **68.2%** |
| `pace_mode` | 77.3% |
| `confidence` | **100%** |

The pre-committed 20% gate applies to `action`, and `action` passes it at 9.1%, so a
single-pass agreement headline is quotable with the discordance stated beside it.

**But `pit_lap_target` changing on two runs in three is its own finding, and it is the field
the thesis quotes.** Section 5.5.2 reports the system "planifica la siguiente parada en la
vuelta 22". That number is a draw, not a plan. The discrete action is roughly stable; every
field that describes the plan around it is not.

### What is deliberately NOT concluded

The thesis reports `PIT_NOW` at 97% confidence for Qatar V7 after the fix, on the **Arcade**
surface and before PRs 1-6 changed the served data. This run is the **CLI**, on the corrected
data. The measured statement is therefore: *under the CLI surface, on the regenerated 2025
data, across three passes, lap 7 does not reproduce that recommendation.* Whether the Arcade
surface differs is unmeasured, and the data has changed underneath either way, so this is not
stated as "the thesis is wrong".

---

## Step 6: the methodology, and the nine forks

Design in `documents/audits/MEASUREMENT_2025_METHODOLOGY.md` (design only, zero LLM calls spent,
no repository file touched but its own report).

**The draw was reproduced and verified independently before a single lap was paid for.** All 91
Tier-A rows were parsed out of the report's tables and checked against `data/raw/2025/`: every
row is a real green-flag stop by `green_flag_stops`, every window is exactly
`[stop-6, stop+5]` clipped, every stop is rail-eligible by `guard_rail_block`, every team string
matches the parquet. **91 windows, 1,090 evaluated laps, zero problems.** (The first parse
returned 90: a regex missed the bolded Budapest LEC L19 row, which is the single most important
window in the design. Caught by the lap-count check, not by reading.)

### The nine forks, and what I did with each

| # | fork | decision |
|---|---|---|
| 1 | memory ON (cold-open) vs OFF | **ON**, as designed. The CLI is the flagship surface and it accumulates `DecisionMemory`; measuring the stateless surface would measure something the thesis does not describe |
| 2 | railed stops excluded vs included | **excluded**, as designed. Keeps the population identical in kind to the no-llm headline's |
| 3 | unweighted vs cluster-reweighted headline | **unweighted is primary**, reweighted reported beside it. Fable recommends against reweighting as the headline and I agree: it rests on a representativeness assumption the draw cannot support |
| 4 | transition rule at laps <= 5 | **keep the no-llm rule**, so the paired contrast stays rule-identical |
| 5 | ALERT as non-pit vs its own bucket | **not a fork.** The action is recorded per lap, so both readings are computable from the same rows after the run. Reported both ways |
| 6 | wet + TD-forced windows inside or outside the headline | **not a fork either**, same reason. Reported both ways, with the sub-regime lines either way |
| 7 | probe whether `gpt-5.4-mini` honours `seed` | **skipped.** The orchestrator's client never passes a seed, so a positive result would not change this run. Noted as future work rather than paid for |
| 8 | pre-committed stopping rule on stability | **pre-committed, below, before any Tier C data exists** |
| 9 | the Lusail maximum-stint directive is inferred from its data signature | verified in the press step before any prose quotes it as regulation |

### The pre-commitment on stability, registered before the data

**Written now, with zero repeat-pass verdicts computed.** If the repeat passes disagree on the
window-level verdict for more than 20% of repeated windows, the single-pass agreement figure is
**not quotable on its own**: it must be reported with the discordance rate on the same line, and
any comparison against the deterministic tier must be stated as a comparison against one draw
from a distribution. Deciding this after seeing the number is the exact manoeuvre this project's
own doctrine forbids elsewhere, so it is fixed here first.

## Step 7: the harness cross-check found a defect in the PUBLISHED tier

Fable's design specifies a V0 cross-check: score the deterministic preflight with the new
analyser and compare against the verdicts already published in
`documents/eval_reports/decision_modes.json`. Result on the 37 stops both cover:

- same bucket: **32 of 37**
- same chosen lap, among those: **27 of 32**

That gap is not noise and it is not a bug in the new analyser. **The published tier does not
use the product's race state.**

`decision_modes.lap_inputs` builds its own `RaceState`. Measured on 2025 Barcelona, ALB:

| lap | field | eval harness | `build_race_state` (the product) |
|---|---|---|---|
| 1-5 | `gap_ahead_s` | **2.0 every lap** | 0.431, 0.914, 0.865, 0.961, 1.075 |
| 1-3 | `pace_delta_s` | **0.0 every lap** | 0.431, 0.483, -0.049 |

`lap_inputs` reads `car.get("gap_ahead_s") or 2.0`, and **`gap_ahead_s` is not a key in the lap
state's driver dict at all** (it carries `gap_to_leader_s`). So the fallback is not a fallback:
it is the value, on every lap of every race. `build_race_state` derives the real interval from
`lap_state["rivals"]`, which `lap_inputs` never reads.

Both fields feed N27's overtake scoring, the orchestrator prompt and the Monte Carlo the tier
grades. **So the published 67-of-178 / 31.3% exact / 78-decline figures describe a stack told,
on every lap, that the car ahead sits exactly 2.0 s away and matches its pace exactly.** And
2.0 is inside the plausible range (the real gaps above run 0.43 to 1.08), so nothing downstream
could tell it from a measurement. Filed as **#829**.

Root cause is the family this repo keeps paying for: #784 replaced three drifted RaceState
copies with one canonical builder, and `lap_inputs` is a fifth copy written into the eval
harness afterwards.

**What it means for this session:** it vindicates the decision to drive the real CLI rather than
reimplement the mapping. **Both of my arms use the product's race state**, so they are
comparable to each other, and neither is comparable to the published tier for this reason as
well as for the sample.

---

### One design-invalidating collision the design could not have known about

The methodology allocates **13 windows / 155 laps to Monza**, and Monza is one of the three races
served another race's radio and RCM corpus (issue #825, found earlier this session). Imola's
corpus puts a Safety Car on laps 29 and 46; several Monza windows span those laps.

**Decision: Monza runs with the radio corpus suppressed (`--no-real-radios`, synthetic generator
off), in both the LLM and the paired deterministic arm.** The pairing stays internally valid
because both arms get the same treatment, and Monza's rows carry a flag saying their inputs are
radio-free. The alternative, running it as-is, would measure the system's response to a Safety
Car that never happened and report it as 2025 Monza.




