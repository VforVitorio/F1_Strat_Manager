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

### The thesis contradiction: my first reading was wrong, and so was my correction of it

I wrote three things here in sequence and the middle one was the worst.

**First** I said the thesis intro ("la parada de Leclerc en V20") was simply wrong against the
parquet's 19. **Then** I "corrected" that: press accounts number the stops PIA 19 / LEC 20, so
it is a one-lap indexing convention, not an error, and the intro is right in its own frame.
**The adversarial gate refuted the correction, and it is right.**

What is true, and it is only the mechanics: `PitInTime` sits on the lap the car was completing
when it entered the pit lane; `PitOutTime`, the stationary time and the new compound all sit on
the next lap. Verified on the data: LEC's in-lap 19 carries a 1:24.552 with the pit-entry
deceleration and his lap-20 row carries 1:39.083 on a HARD at `TyreLife` 1. Both cars crossed
the timing line about 1.7 s **after** entering the pit lane, so "the lap spent in the pit lane"
genuinely is 20 for LEC.

**What is NOT true is that the press follows that as a convention.** Reachable accounts
disagree with each other. Wikipedia's race report: *"On lap 18, Piastri pitted from second to
take hard tyres"* and *"Leclerc responded the following lap"* — **18 / 19, the parquet's
numbering**. Other accounts say 19 / 20. There is no single press convention to appeal to.

**And the decisive test is inside this very document.** Step 3 below reads *"16 cars pit on lap
7"* straight off `PitInTime` at Lusail and treats the thesis's *"diecisiete coches pitan en esa
misma vuelta"* as the same event with **no offset**. If a systematic +1 existed, the press would
call those stops lap 8. It does not (Sky Sports: *"pitting on Lap 7 provided key strategic
gain"*, and *"McLaren's decision not to pit their two cars at the end of that lap"*, which is
in-lap numbering said out loud). Lusail's `PitInTime` on lap 6 is **empty**, so "vuelta 7"
cannot be press-plus-one under any reading.

**So I answered the same convention question two different ways in one session, each time in
the direction that made the thesis right: +1 at Budapest, 0 at Qatar.** Each reading is locally
defensible and no single rule generates both, which means it was not a reconciliation. It was
rescuing a sentence.

**The honest statement:** the parquet's numbering is the one the system runs on and the one the
thesis body uses ("al cierre de la vuelta 19", which is literally what the data shows). The
preamble's "V20" is most simply an error, and Occam agrees: under the plain reading, "tres
vueltas en torno a la parada de Leclerc en V19" is exactly the executed `--laps 17-19`, so
**one consistent re-indexing fixes the sentence**. Under the rescue reading, "five laps around
V20" is 18-22, which is still not the window, so it needs two repairs and leaves the thesis
carrying two lap numbers for one stop.

The press claim the case rests on does hold: Leclerc on the compound, verbatim, *"The hard was
much more difficult, but you discover this a bit too late. We were trying to do the undercut,
and it was working at one stage, you anticipate the pit stop and you can't react."*

### The stops, in the indexing the system runs on

From `data/raw/2025/Budapest/pitstops.parquet`, in-lap (`PitInTime` non-null) per driver. This
is the indexing the replay engine, the windows and every number below use:

| driver | stop 1 | stop 2 |
|---|---|---|
| VER | 17 | 48 |
| PIA | **18** | 45 |
| LEC | **19** | 40 |
| RUS | 19 | 43 |
| NOR | 31 | - |

In this indexing Leclerc's covering stop is lap **19** and Piastri's undercut is lap **18**,
which is what section 5.5.2 says and what Wikipedia's race report says. The recommended thesis
edit is the single consistent one: the preamble becomes "las tres vueltas en torno a la parada
de Leclerc en V19", which then matches both the body and the executed command.

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

### What a lap costs in money, corrected

Published list prices read 2026-08-06 (state them with this date; they move):

| model | input / 1M | output / 1M | cached input / 1M |
|---|---|---|---|
| `gpt-4.1-mini` | $0.40 | $1.60 | $0.10 |
| `gpt-5.4-mini` | $0.75 | $4.50 | $0.075 |

Over the completed 75-lap run: **$0.0080 per lap**, $0.263 on the sub-agents and $0.337 on the
orchestrator. The orchestrator is the larger share despite making one call in eight, because
its completions are long and priced at $4.50/1M.

**Per hour of running: about $1.3 to $1.6**, depending on where in the measured latency range
the run sits (157 laps/h at 22.9 s/lap, 199 laps/h at 18.1 s/lap).

> ⚠️ **A fourth error the adversarial gate caught, which I had missed.** This paragraph used to
> say **"$0.43 per hour of running"**. That is wrong by roughly a factor of four and, worse, it
> contradicted a figure three lines below it in the same paragraph: "1,500 laps is about $11 and
> about 6.6 hours" implies $1.67/hour. **Two numbers in one paragraph that cannot both be true,
> and I published both.** Neither the correction pass I did on the token counts nor my own
> re-reading caught it; the gate did, by dividing.

**The binding constraint is still WALL CLOCK, not money.** The full 1,090-lap Tier A sample is
**$8.72** and hours. Any framing of this session as expensive in API spend is wrong.

### The wall-clock figure is the head of a right-tailed distribution, not its middle

The gate is right about this too and it changes the budget. Measured per-lap seconds:

| sample | mean | max |
|---|---|---|
| the 3-lap probe | **15.93** | 17.38 |
| `thesis_windows`, n=22 | **18.07** | 35.60 |
| `thesis_repeats`, n=75 | **22.92** | 55.63 |

The medians of the larger samples sit near the probe's value, so the probe was not unlucky, it
was **short**: it sampled three laps from the head of a distribution with a long right tail
(one lap reached 77,484 prompt tokens against a median of 8,322). Calls per lap behaves the
same way: **6 is the MODE, not the mean** (28% of laps take 10 to 21 calls as the ReAct loops
iterate), and the mean is 7.6 to 7.9.

Consequence for the design's budget: Tier A's "4.82 h at 15.93 s/lap" is realistically **5.5 to
7 h**, and its "12.33 M tokens" is realistically **15 to 18 M**. Every completed sample sits
above the probe on every axis, always in the direction that made the plan look cheaper.

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

> ⚠️ **The MC column below is the DETERMINISTIC arm's, and the gate was right that presenting
> it as shared was misleading.** In `rich` mode the sub-agents' outputs feed the Monte Carlo, so
> the scores the orchestrator is handed are themselves perturbed per pass. On Budapest 19,
> `PIT_NOW` is frozen at -1.645 across all eight stored samples, but `STAY_OUT` ranges from
> -0.968 to -1.727 across the seven rich ones. **On the very pass this table quotes (confidence
> 0.90) the MC favoured STAY_OUT, -0.968 against -1.645** — so on that pass the LLM was agreeing
> with its own Monte Carlo, not overriding a pit-favouring one. Over the seven rich samples the
> ordering favours PIT in four and STAY in three. The result stands; the phrase "given the same
> inputs" does not, and Qatar is unaffected because there the MC favoured PIT on laps 8 and 9 on
> every sample.

| lap | real event | **LLM (`rich`)** | **deterministic (`no-llm`)** | MC score STAY / PIT (no-llm arm) |
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
offset 0, the best result this metric can produce. The LLM layer, over the same window and the
same replayed race, answered STAY_OUT on every lap and every pass, and the stop was never
called. In the decision
tier's own vocabulary the deterministic path scores `scored, offset 0` and the LLM path scores
`no_call_in_window`.

That is the first direct evidence that **the two paths do not agree with each other**.

> ⚠️ **DO NOT read "the LLM layer is more reluctant" out of this.** I wrote that sentence here
> first and the wider sample refutes it. See "the two-window generalisation, refuted" below:
> over 180 paired laps of Tier A the LLM arm asks to pit on **37.2%** of laps and the
> deterministic arm on **41.1%** — near-identical rates. Budapest and Lusail are two windows,
> and a system property drawn from two windows is the exact error this session exists to avoid.

**Confirmed by the scorer rather than by reading the action list** — the distinction matters,
because "emitted PIT_NOW on lap 19" and "the tier's transition rule locates the decision on lap
19" are different statements and only the second is the metric:

```
thesis_windows_nollm.jsonl  Budapest LEC  actual_lap 19  chosen_lap 19  offset 0  bucket 'scored'
thesis_windows.jsonl        Budapest LEC  actual_lap 19  chosen_lap None          bucket 'no_call_in_window'
```

### Finding 2: Qatar. The shipped SC fix works, and the LLM vetoes it on a real rule read out of its condition

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

> ⚠️ **I first wrote here that no such rule exists and that the RAG had fabricated it. That was
> wrong.** The adversarial gate refuted it and I verified the refutation against the PDF myself.
> The corrected finding is below, and it is a **distortion**, not a fabrication. The
> six-occurrence classification I published ("every one is a replacement-component clause or the
> two-compound requirement, not one is about a Safety Car") was also false: the count of six was
> right and occurrence five is the Safety Car rule. I wrote executed-evidence-shaped prose from a
> sample of the occurrences instead of from all of them, which is the exact shape of
> `feedback_grep_is_not_an_audit`.

**The rule exists, conditionally.** 2025 Sporting Regulations, **Article 30.5 n)**, read out of
`data/rag/documents/sporting_regs_2025.pdf` verbatim:

> "n) If the formation lap is started behind the safety car in accordance with Article 49.1a, or
> the sprint session or race is resumed in accordance with Article 58.1, the use of wet-weather
> tyres until the safety car orange lights are extinguished and it returns to the pit lane is
> compulsory. **A penalty under Article 54.3d) will be imposed on any driver whose tyre(s) are
> changed for a different specification or who uses any other specification of tyres whilst the
> safety car is on the track at such times.**"

(Art. 30.5 o) in 2023, 30.5 n) in 2024.)

**The transformation is amputating the condition.** "At such times" points back to a wet
formation-lap start behind the Safety Car, or a resumption. The RAG's sentence lifts the second
half and re-scopes it to every Safety Car period. Almost every phrase in its output is a
quotation: "changed for a different specification", "uses any other specification of tyres",
"whilst the safety car is on the track", "a penalty under Article 54.3d)". Even the article
number it cites is the real rule's own cross-reference, so **citing 54.3d) is quotation, not
invention** — 54.3 is indeed the penalties article, and 30.5 n) is what points at it.

The stored rows show the retriever reached the right chunk: `regulation_context` cites
`(Article 30.5)` on lap 7, `(Article 58.10)` on lap 8 — 30.5's own resumption cross-reference in
the 2023/2024 wording — and reproduces the 2025 phrase "orange lights are extinguished and it
returns to the pit lane" on lap 9.

**The empirical refutation still holds, against the re-scoped version: sixteen cars, Verstappen
included, changed tyres on lap 7 of this Safety Car**, which was an ordinary dry mid-race
deployment. Under the real rule that is legal; under the RAG's version, sixteen cars took a
penalty.

So the defect is **decisive and real**, and it is a condition dropped in summarisation rather
than a rule invented from nothing. That changes the fix: grounding the CITATION would not have
caught it, because every article number involved was genuinely retrieved. **Grounding the
CONDITION is what would.**

### Finding 3: the Monte Carlo is a point mass on the deployment lap

On Lusail lap 7, the lap the whole case is about, `scenario_scores` reads
`STAY_OUT: E=P10=P90=0.30` and `PIT_NOW: E=P10=P90=0.30`. Identical, zero variance, tie.
UNDERCUT and OVERCUT are ineligible. The layer supplies **no discrimination at all** on the most
consequential lap of the 2025 season, and the argmax picks STAY_OUT on tie order.

> ⚠️ **A second claim of mine the gate refuted, and I verified the refutation.** I wrote here
> that "the LLM's reasoning calls PIT_NOW 'MC-favoured' on that lap; the scores it was given are
> tied; the narrative misdescribes the numbers in the same prompt." **False.** The
> "MC-favoured PIT_NOW" phrase is on **lap 9**, not lap 7, and on lap 9 the Monte Carlo genuinely
> did favour pitting (-0.76 against -1.76). Every stored lap-7 narrative describes the tie
> correctly: *"STAY_OUT and PIT_NOW are tied on score"*, *"MC is neutral between stay out and
> pit"*, *"the MC tie"*, *"The MC result is effectively neutral"*. I attributed a lap-9 sentence
> to lap 7 and built a finding on it.

What survives, and it is the half that matters: the tied point mass on lap 7 is real and
reproduces on every sample. The layer supplies no discrimination on the lap the case is about,
and the LLM correctly says so before overriding it on the regulation instead.

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

| field | pooled across window definitions | **same-context (3 passes of one window)** |
|---|---|---|
| `action` | 9.1% | **9.1%** |
| `compound_next` | 50.0% | **50.0%** |
| `pit_lap_target` | 68.2% | **59.1%** |
| `pace_mode` | 77.3% | **72.7%** |
| `confidence` | 100% | **100%** |

> ⚠️ **The right-hand column is the honest one and I published the left.** Three of the 22
> repeated laps (Budapest 17-19) carry SIX observations rather than three, because they appear
> in both the 14-24 window and the cold-opened 17-19 window. A lap 17 cold-opened at the window
> start carries different `DecisionMemory` than a lap 17 warmed from lap 14 — a distinction this
> very document draws when it rules out the memory confound for `action`. Counting those six as
> repeats of one lap charges **context variation to path noise**, and it inflates
> `pit_lap_target` by 9.1 points and `pace_mode` by 4.6. `action`, `confidence` and
> `compound_next` are identical either way, so the pre-committed 20% gate outcome does not move.
> Caught by the gate; the recorder now stores each row's window bounds so the two cannot be
> pooled again by accident.

The pre-committed 20% gate applies to `action`, and `action` passes it at 9.1%, so a
single-pass agreement headline is quotable with the discordance stated beside it.

**But `pit_lap_target` changing on three runs in five is its own finding, and it is the field
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

> ⚠️ **Corrected after the gate.** I wrote that this gap "is not noise and is not a bug in the
> new analyser: the published tier does not use the product's race state." The second half is
> true and independently verified; **using it to explain the 5 disagreeing verdicts is not.**
> Three differences sit between the two artefacts, not one: the published `decision_modes.json`
> was regenerated on 2026-08-03 and the featured artefacts on 2026-08-06; the preflight drives
> the real CLI and therefore feeds RCM events into the Safety Car tracker while
> `_decisions_in_window` passes no radio or RCM at all, which points in the causal direction of
> a verdict flip on any window touching a neutralisation; and the race-state fields. Separating
> them needs the five stops re-run pairwise. The finding below stands on its own evidence and is
> only **consistent with** the disagreement.

What the gap did lead to, and it holds independently: **the published tier does not use the
product's race state.**

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





---

## Step 8: the two-window generalisation, refuted by my own sample

**Updated at 639 paired laps (from the 180 this section first reported). Still partial: three
races dominate the subset and three are not yet measured at all. Both facts are stated below
rather than left for a reader to discover.**

Budapest and Lusail both showed the LLM arm declining where the deterministic arm committed, and
I wrote "the layer that ships is more reluctant than the layer that was measured" into this log.
**The wider sample does not support it.** Over the 180 laps both arms have evaluated so far:

| | LLM (`rich`) | deterministic (`no-llm`) |
|---|---|---|
| **pit-class share of laps** | **27.2%** | **29.9%** |
| identical action | **75.7%** of laps | (same figure, both arms) |

**And the direction is not even consistent across races**, which is the strongest evidence
against a systematic reluctance:

| race | paired laps | LLM pit-class | deterministic pit-class | identical action |
|---|---|---|---|---|
| Barcelona | 169 | 29.0% | 26.6% | 74.6% |
| Budapest | 164 | 33.5% | 38.4% | 66.5% |
| Monza | 152 | 26.3% | 36.8% | 80.3% |
| Montréal | 56 | 32.1% | 25.0% | 71.4% |
| Lusail | 53 | 3.8% | 0.0% | 96.2% |
| Monaco | 45 | 22.2% | 28.9% | 80.0% |
| **all** | **639** | **27.2%** | **29.9%** | **75.7%** |

The LLM arm is the more willing one at Barcelona and Montréal and the less willing one at
Budapest, Monza and Monaco. A single ordering does not survive the race breakdown.

**Two things that bias this table and are not fixed by more laps:**

1. **Three races carry 76% of it** (Barcelona, Budapest, Monza) because they finished first.
   Silverstone (the wet regime), Suzuka and Mexico City are absent entirely. The rates will
   move when they land, and the wet race is the one most likely to move them.
2. **Monza's rows are radio-free on BOTH arms** (the #825 workaround), so its inputs differ from
   every other race in the table. The pairing stays internally valid, and Monza is also the
   race with the largest gap in the table, which is exactly the kind of coincidence worth
   naming rather than hoping nobody checks.

The earlier per-action counts at 180 laps are superseded by the table above.

Near-identical willingness to stop. What actually differs is **which** stop:

| deterministic said | LLM said | laps |
|---|---|---|
| STAY_OUT | STAY_OUT | 90 |
| UNDERCUT | STAY_OUT | 20 |
| UNDERCUT | PIT_NOW | 18 |
| PIT_NOW | PIT_NOW | 18 |
| UNDERCUT | UNDERCUT | 15 |
| STAY_OUT | PIT_NOW | 13 |
| STAY_OUT | UNDERCUT | 3 |
| PIT_NOW | STAY_OUT | 3 |

Identical action on **68.3%** of laps; same pit/no-pit class on **78.3%**. The deterministic
layer reaches for `UNDERCUT` (53 laps) where the LLM reaches for `PIT_NOW` (49), and the single
largest disagreement is the deterministic layer proposing an undercut that the LLM either
converts to a plain stop (18) or declines outright (20).

**Two lessons, and the second is about me.**

1. The interesting question is not "does the LLM stop less" but "does the LLM pick a *different
   kind* of stop", and the answer so far is yes. That was not on the list of things this session
   set out to measure.
2. **Two windows are not a system.** I generalised from Budapest and Lusail inside the same
   document that opens by warning against exactly that, and the correction came from my own
   sample rather than from a reviewer. The Budapest and Lusail findings stand as **case
   results** — they are about specific, consequential laps, and the Qatar mechanism (#826) is a
   real defect either way. They are not a rate.

`ALERT` has not appeared once in 336 LLM laps, so the fork about how to bucket it dissolves: it
is not in the measured distribution.

---

## Step 9: the deterministic arm is complete, and its lap shortfall is accounted for

All nine races, **1,066 distinct laps of the 1,090 the sample asks for (97.8%)**, 80 measured
`(race, driver)` window groups covering the 91 spec windows.

| race | planned | evaluated | coverage | what is missing |
|---|---|---|---|---|
| Barcelona | 191 | 185 | 97% | ALB 28-32 (retired on lap 27), ANT 54 |
| Budapest | 168 | 168 | 100% | - |
| Lusail | 60 | 56 | 93% | BEA 42-45 |
| Mexico_City | 84 | 84 | 100% | - |
| Monaco | 96 | 88 | 92% | GAS 8-13 and two more |
| Montréal | 84 | 84 | 100% | - |
| Monza | 155 | 153 | 99% | ALO 25, STR 53 |
| Silverstone | 132 | 128 | 97% | HAM 16, SAI 16, VER 15-16 (inside the lap 13-21 neutralisation) |
| Suzuka | 120 | 120 | 100% | - |

**19 of the 24 gaps are the replay engine declining a lap it cannot serve**: a retired car keeps
yielding lap states with an empty driver dict, and a lap with no position is skipped by design
because a sentinel position has already collided with a real one in this codebase.

> ⚠️ **The other five are not explained, and I first wrote that all of them were.** The gate
> checked and found five laps where the car was running with a valid position: four Silverstone
> laps inside a Safety Car block and one Monaco pit in-lap. Whatever declines those is a
> different mechanism, and "every gap is X" was a generalisation from the majority. It is
> 1.8% of the deterministic arm and it does not move any figure here, but a shortfall with a
> confident single cause is exactly the shape #827 taught this session to distrust.

Naming this here because #827 proved that a broken arm and a thin sample look identical from a
row count, so a shortfall gets an explanation rather than a shrug — and, per the five above,
gets one only as far as the evidence reaches.

### Its own numbers, and what they may NOT be compared with

| | this arm (9 races, product race state) | published `decision_modes.md` (6 races) |
|---|---|---|
| eligible stops | 100 | 178 |
| scored | 39 (**39.0%**) | 67 (37.6%) |
| coverage verdict | **masked** | **masked** |
| exact lap | **12.8%** | 31.3% |
| within one lap | **30.8%** | 47.8% |
| mean signed error | -2.31 | -1.52 |

**These two columns are NOT comparable and nothing in this session should be read as comparing
them.** Three things differ at once: a different race sample, a different window construction
(per-stop rather than a contiguous per-driver span), and the published column is measured on the
constant 2.0 s gap and constant 0.0 pace delta of #829 while this one goes through the product's
own `build_race_state`. **Do not read the lower exact-lap rate here as "the fix made it worse".**
Which of the three moves it, and in which direction, is unmeasured, and saying otherwise would be
the same unproven attribution the gate already caught once in this document.

---

## Step 10: the run STOPPED, and not because of a bug

At **17:17:2x-17:17:41** all four running batches stopped writing rows, within 13 seconds of
each other. A single external event, not four independent failures. Diagnosed by re-running one
window with the CLI's stdout captured, which the batch runs had sent to `/dev/null`:

```
RateLimitError: Error code: 429 - {'error': {'message': 'You have no credits ...
```

**The OpenAI account ran out of credit.** Every subsequent lap raised inside the agent stack and
the CLI's per-lap `except Exception` rendered it as a red row, which is the exact silent-shape
of #827 arriving by a different route: three races produced **zero** rows while their batches
walked all their windows and reported `done`. Mexico City wrote 5 laps of 84, Silverstone 0 of
132, Suzuka 0 of 120.

**This is a hard stop and it is Víctor's call**, not something to work around: continuing needs
credit added to the account. Everything already measured is on disk, the resume logic skips it,
and re-running costs only the missing laps.

### Where the measurement ended up

| arm | races covered | laps of 1,090 |
|---|---|---|
| **deterministic (`no-llm`)** | **9 of 9** (complete) | **1,066 (97.8%)** |
| LLM (`rich`) | 7 of 9 (Silverstone and Suzuka absent, Mexico City at 5 laps) | 669 (61.4%) |
| thesis windows, 3 repeat passes | complete | 97 |

Report: `documents/eval_reports/llm_2025/REPORT.{md,json}`. It names the races each arm actually
covers on its own second line, because the design is nine races and quoting the design as the
population is the error this whole session was built to avoid.

### The paired result, on the 67 stops BOTH arms measured

| | |
|---|---|
| same bucket in both arms | **42 of 67 (62.7%)** |
| scored by both | 23, of which **the same chosen lap in 5 (21.7%)** |
| deterministic scored, LLM did not | **7**, of which **1 was an exact agreement the LLM lost** (Budapest LEC 19) |
| LLM scored, deterministic did not | **18** |

**The headline finding, and it is not the one this session set out to find.** The LLM arm
locates a decision **more** often than the deterministic arm (41 scored of 67 eligible, 61.2%,
verdict `ok`, against 39 of 100, 39.0%, `masked`) and lands on the **right lap less** often
(exact 4.9% against 12.8%; within one 22.0% against 30.8%).

So the two layers fail in opposite directions: **the deterministic layer declines more and is
right more often when it does commit; the LLM layer commits more often and is further out.**
That is a sharper statement than "the LLM is more reluctant", which is what I wrote from two
windows and then had to retract twice.

Read with the caveats that apply and are not optional: the two arms cover different race sets,
the LLM arm is missing the wet race entirely, `mean_signed_error` is a property of
`DECISION_WINDOW_LAPS` and not of the system, and agreement with the real wall is evidence
rather than a verdict, because the wall can be wrong and Qatar is in this sample precisely
because the press says it was.

### What is left, and the sample stays unfinished ON PURPOSE

**Standing rule set by Víctor on 2026-08-06, after this session drained the account: no runs
that spend LLM credits without asking first.** So the three missing races are not a TODO waiting
for a spare moment, they are a decision he takes when he wants the number. `measure_llm_windows.py`
now refuses a paid run by default and prints the bill; `--yes-spend` is the opt-in.

1. **The three missing races**: Silverstone (132 laps, and it is the WET regime, absent
   entirely), Suzuka (120), Mexico City (79). About 331 laps, **$2.65**, 1.5 to 2 h. The resume
   skips everything on disk. **Ask before running.**
2. Then regenerate `REPORT.md`; the paired numbers will move.
3. ~~Fix #829 and re-measure~~ **DONE** (PR #830): every accuracy band dropped about ten points
   once the tier stopped receiving a constant 2.0 s gap.
4. ~~#825~~ **DONE** (PR #830): the corpora are rebuilt and Monza's phantom Safety Car is gone.
5. **#826 stays open**, and the fix it needs is not the one the issue proposed. Measured on the
   retriever alone, at zero cost: the chunk that carried the rule **starts mid-word**, so the
   article's applicability clause is in the previous chunk and the agent never had it. The
   prompt changes shipped in #830 are a measured mitigation (the bad chunk falls from rank 1 to
   outside the top 5); the real fix is clause-aware re-chunking plus labelling from the
   containing heading. **Its end-to-end verification is deliberately NOT run**, because that
   costs credits and the rule above applies.
