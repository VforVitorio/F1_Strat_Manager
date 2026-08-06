# ADVERSARIAL GATE — 2025 LLM-mode measurement claims

Gate over CLAIMS, not code. Date: 2026-08-06. Branch: `feat/measure-2025-llm-mode`.
Material: `MEASUREMENT_SESSION_2025_LOG.md`, `MEASUREMENT_2025_METHODOLOGY.md`,
`PROBE_llm_cost.json`, `documents/eval_reports/llm_2025/*.jsonl` (thesis_* complete,
tierA_* partial — no conclusions drawn from tierA totals), issues #825/#826/#827/#829,
PR #828 body.

Written incrementally. Each verdict appended as confirmed or refuted.

## Checklist

- [ ] A. Cost claim (15.93 s/lap, 6 calls, 8,080+814 tokens, ZERO cacheable, $0.0071)
- [ ] B. The fabricated regulation (#826) — fabrication vs distortion; what Art. 54.3 is
- [ ] C. Monza's phantom Safety Car (#825) — Imola discriminator; slug routing; decision-modes immunity
- [ ] D. Eval race-state defect (#829) — gap_ahead_s 2.0 on 100% of laps; pace_delta_s hardcode; rival branch
- [ ] E. 2025 projection figures (552/24/86.05/59.60) + the causal "reads no model/tables" half
- [ ] F. Qatar + Budapest results recount from JSONL
- [ ] G. Non-determinism percentages recount + denominator check
- [ ] H. Population discipline across the log + PR #828 body
- [ ] I. Lap-indexing correction (press 20 vs parquet 19)

---

## Verdicts

### A. Cost claim — PARTLY REFUTED. The probe is a probe, and the completed runs contradict four of its five numbers.

Verified first: the $/lap arithmetic against the stated prices is right at the probe's own
numbers ((14,044x0.40 + 999x1.60)/3M = $0.00241; (10,196x0.75 + 1,442x4.50)/3M = $0.00471;
total $0.00712/lap). The probe file is internally consistent (18 calls = 6/lap x 3; token sums
match; per-lap rows sum to totals).

What does NOT survive contact with the completed runs (`thesis_windows.jsonl` n=22,
`thesis_repeats.jsonl` n=75, both `profile="rich"`, plus their `.err` token-meter dumps):

| quantity | probe (n=3) | thesis_windows (n=22) | thesis_repeats (n=75) |
|---|---|---|---|
| seconds/lap mean | **15.93** (max 17.38) | **18.07** (max 35.60) | **22.92** (max 55.63) |
| LLM calls/lap | **6.0** flat | mean 7.59 (mode 6; 7/22 laps at 10-15) | mean 7.91 (mode 6; 21/75 laps at 10-21) |
| prompt tokens/lap | **8,080** | mean 11,165 (median 8,332, max 31,134) | mean 13,226 (median 8,322, max 77,484) |
| completion/lap | 814 | 1,017 | 1,014 |
| cached prompt tokens | **0** | **44,800 (18.2% of prompt)** | **345,344 (34.8% of prompt)** |
| $/lap at stated prices | $0.0071 | $0.0083-0.0089 | $0.0081-0.0097 |

1. **"ZERO cacheable" is refuted, and so is its stated mechanism.** The log asserts *"The
   prompt is rebuilt per lap and drifts numerically on every line, so no prefix is stable
   enough for the provider's cache to hit"* and the methodology repeats it (*"Zero prompt
   tokens are cacheable (measured)"*, section 0, and again in 4.5). The completed runs cache
   18.2% and 34.8% of prompt tokens — including **62,720 cached tokens on the orchestrator
   model** (`gpt-5.4-mini`) in `thesis_repeats.err`. The numeric drift claim confuses the
   suffix with the prefix: the per-lap numbers drift, but the sub-agent/orchestrator system
   preambles are a stable prefix, and once a process runs long enough (or revisits laps) the
   provider cache hits. n=3 laps in one short process was exactly too small to see it.
   Corrected statement: caching is real, grows with run length, and the "prompt cost is paid
   in full on every lap" sentence is wrong; the money conclusion barely moves (see 3).
2. **6 calls/lap is the MODE, not the number.** 28% of thesis-window laps take 10-21 calls
   (ReAct tool loops firing). Mean is 7.6-7.9. The methodology's 1.5 ("One run_lap = 6 LLM
   calls") and its 8.3k-calls budget line inherit the error.
3. **The per-lap wall clock and token figures are underestimates of the served
   distribution.** 15.93 s -> 18-23 s measured (Budapest repeats mean 22.8 s); 8,080 prompt
   -> 11.2-13.2k mean. The medians sit at the probe's values — the probe sampled 3 laps from
   the head of a right-tailed distribution (max 77k tokens on one lap). Budget consequence:
   Tier A's "4.82 h" at 15.93 s/lap is more realistically 5.5-7 h, and "12.33 M tokens" for
   1,386 laps is more realistically 15-18 M. The project's own doctrine names this failure:
   *a probe is not a distribution*.
4. **"$0.43 per hour of running" is an arithmetic error in the log's own paragraph.** At the
   claimed $0.00711/lap and 15.93 s/lap, one hour is 226 laps = **$1.61/hour**. The same
   paragraph's other figures ("1,500 laps is about $11 and about 6.6 hours") imply
   $11/6.6 h = $1.67/hour and contradict the $0.43 three lines above them. The qualitative
   conclusion (wall clock binds, not money) survives — at $1.6/h the 6-hour design is ~$10.

Population note: thesis_repeats' higher means partly reflect Budapest re-runs (its laps carry
longer memory/radio context: Budapest repeats mean 15,136 prompt tokens/lap vs Lusail 10,795),
so 22.92 s is not "the" per-lap figure either — but every completed sample sits above the
probe on every axis, in the direction that makes the budget optimistic.

Severity: MEDIUM (budget and two published sentences wrong; no decision reversed).
Evidence: `documents/audits/PROBE_llm_cost.json`; `documents/eval_reports/llm_2025/thesis_windows.jsonl`
+ `.err`; `thesis_repeats.jsonl` + `.err`; recomputation shown above.

*(Coordinator confirmed A accepted and corrected upstream; not revisited below.)*

### D. The eval race-state defect (#829) — CORE CLAIM VERIFIED, with sharper conditions. The BLAST-RADIUS attribution ("the 5-of-37 disagreement is this") is UNPROVEN and confounded.

**The constant-2.0 claim survives every attack I ran, including the rival-branch one.**

1. **The key genuinely never exists.** `src/simulation/` contains zero occurrences of
   `gap_ahead_s` (ripgrep over the tree); `RaceStateManager.get_driver_state`
   (`race_state_manager.py:387-`) emits `gap_to_leader_s` (line 421) and no `gap_ahead_s` —
   the issue's 26-key list is accurate. Runtime check over three full 2025 race replays
   (Barcelona/ALB, Budapest/LEC, Lusail/PIA — 154 lap states with valid position):
   `'gap_ahead_s' in state['driver']` was True on **0 of 154**. So
   `decision_modes.py:313` `car.get("gap_ahead_s") or 2.0` serves 2.0 on 100% of laps —
   confirmed, not just on lap 1.
2. **The rival-specific branch is NOT needed for the divergence.** `build_race_state`
   (`race_state_builder.py:328-333`) computes the gap from the POSITIONAL car ahead
   (`_car_ahead` -> `_gap_to_car_ahead`) whenever the caller passes `gap_ahead_s=None`; the
   `rival` branch (`:335-341`, `_targeting_against_rival`) is an OVERRIDE on top of that,
   not the source. No `--rival` is required for the product to serve a real gap.
3. **Measured divergence on the served distribution, not one hand-built row:**

   | race/driver | laps | builder gap != 2.0 | builder pace != 0.0 | leader laps | interval-missing laps |
   |---|---|---|---|---|---|
   | Barcelona/ALB | 27 | 100% | 100% | 0 | 0 |
   | Budapest/LEC | 70 | 100% | 60% | 28 | 0 |
   | Lusail/PIA | 57 | 100% | 42% | 33 | 0 |

   The exact divergence conditions, which the issue does not spell out: (a) **leader laps**
   diverge as honest-0.0 vs fabricated-2.0 (`_gap_to_car_ahead` returns 0.0 when no car is
   ahead — 28 of LEC's 70 laps, 33 of PIA's 57); (b) **non-leader laps** diverge as
   real-interval vs 2.0; (c) the only gap-agreement case is a car ahead whose
   `interval_to_driver_s` is None (builder falls back to the same 2.0) — measured **zero**
   such laps in 154; (d) `pace_delta_s` AGREES on leader laps (both 0.0,
   `_pace_delta_vs_car_ahead:195-199`) and diverges elsewhere. So "100% of laps" is right
   for the gap, and pace divergence is 42-100% depending on how much of the race the driver
   led.
4. One forward-looking note on the fix: `or 2.0` also coerces a STORED 0.0. If the fix ever
   adds the key to the driver dict instead of calling the builder, the leader's honest 0.0
   becomes 2.0 again. The fix direction in the issue (call `build_race_state`) avoids this;
   any other fix must replace `or` with an `is None` read.

**What does NOT survive: the closing attribution.** #829 ends: *"the two arms disagreed on
5 of 37 shared verdicts, and the disagreement is this"*, and the log's Step 7 says the
32-of-37 gap "is not noise and it is not a bug in the new analyser. The published tier does
not use the product's race state." That leap is unproven. THREE differences sit between the
compared artefacts, not one:

- **Data vintage.** The published `decision_modes.json` was last regenerated at `fce6fd9`
  (2026-08-03 11:32); the featured artefacts were regenerated at `c561124`
  (2026-08-06 13:51, PR 6 — de-dup Spain, impute Las Vegas), and the V0 preflight ran after
  that. The log itself says, about Qatar, "the data has changed underneath either way" — the
  same sentence applies to the V0 comparison and is not applied there.
- **RCM/radio injection.** The preflight (`scripts/measure_llm_windows.py`) drives the real
  CLI path, which feeds RCM events into the SC tracker (`sc_currently_active`, rail
  suspension); `_decisions_in_window` (`decision_modes.py:349-351`) calls
  `run_lap(race_state, laps_df, state, profile="no-llm")` with NO rcm/radio argument. Under
  any window that touches a neutralisation (Barcelona L49 windows against the L54-60 SC,
  Silverstone's two blocks, Lusail 7-10), the two runs feed different SC state to the same
  stack. This difference is in the CAUSAL direction of verdict flips (rail suspension changes
  PIT_NOW eligibility).
- The race-state fields (#829 itself).

Any of the three can move a bucket. Until the 5 disagreeing stops are re-run pairwise with
one difference at a time, "the disagreement is this" is a plausible hypothesis written as a
finding. The defect stands on its own evidence; the 5-of-37 sentence should be downgraded to
"consistent with".

Severity: the defect itself HIGH (already filed, correctly); the causal attribution MEDIUM
(it is in an issue body, and a false claim in an issue is scope).
Evidence: executed replay measurement above; `decision_modes.py:280-353`;
`race_state_builder.py:99-199, 270-341`; `git log` dates as quoted.

### B. The "fabricated" Article 54.3 rule (#826) — REFUTED AS A FABRICATION. It is a DISTORTION of a real rule, Art. 30.5 n), and the issue's own supporting search claim is false.

I extracted all three PDFs myself (pypdf; 2023: 315,467 chars, 2024: 327,704, 2025: 379,357 —
close to the issue's counts, different extractor) and searched them independently.

**The rule the issue says does not exist, exists — conditionally — in all three rulebooks.**
2025 Sporting Regulations, Article 30.5 n) (verbatim from my extraction):

> "n) If the formation lap is started behind the safety car in accordance with Article 49.1a,
> or the sprint session or race is resumed in accordance with Article 58.1, the use of
> wet-weather tyres until the safety car orange lights are extinguished and it returns to the
> pit lane is compulsory. **A penalty under Article 54.3d) will be imposed on any driver whose
> tyre(s) are changed for a different specification or who uses any other specification of
> tyres whilst the safety car is on the track at such times.**"

The same rule is Art. 30.5 o) in 2023 and 30.5 n) in 2024 ("...will be imposed on any driver
who does not use wet weather tyres whilst the safety car is on the track at such times",
resumption cross-ref there being Article 58.10a)).

Put the RAG's sentence next to it: *"...compulsory that drivers do not change to a different
specification of tyres or use any other specification of tyres while the Safety Car is on the
track. A penalty under Article 54.3d) will be imposed..."*. The n-grams "changed for a
different specification", "uses any other specification of tyres", "whilst/while the safety
car is on the track", "penalty under Article 54.3d)", "is compulsory", and (in the lap 7/9
variants) "orange lights are extinguished and it returns to the pit lane" are all LIFTED from
30.5 n). The single transformation is **amputating the applicability condition** (wet
formation-lap start behind the SC, or a resumption — "at such times") and re-scoping the
sentence to every Safety Car period.

**The stored rows prove the retriever surfaced this very rule.** From
`thesis_windows.jsonl` / `thesis_repeats.jsonl`, `regulation_context` on Lusail:

- lap 7 (windows + all 3 repeat passes): ends "...such as wet conditions requiring
  wet-weather tyres **(Article 30.5)**" — the RAG cites the real rule's own article number.
- lap 8: cites "**(Article 58.10)**" — the resumption cross-reference exactly as spelled in
  the 2023/2024 wording of 30.5 n)/o).
- lap 9 and the repeats: carry the 2025 wording's "orange lights are extinguished and it
  returns to the pit lane".

So the mechanism in #826's root-cause link 1 — penalty chunks (54.3) plus tyre-specification
chunks (30.x) "fused into a rule" — is the wrong mechanism. The retriever almost certainly
returned the 30.5 n) chunk itself (the one chunk in the book combining SC + tyre
specification + penalty); the synthesis then dropped its condition. (Direct confirmation from
the retrieval payload is not possible from disk: the JSONL persists `agents.rag` as a boolean,
and probing the live Qdrant store was off-limits while measurement processes hold it.)

Specific verdicts on the issue's supporting claims:

1. *"No such rule exists" / "fabricated regulation"* — **REFUTED as phrased.** The corrected
   statement: no rule restricts tyre specification during an ORDINARY mid-race Safety Car;
   the only SC tyre-specification restriction (30.5 n)) applies at wet starts behind the SC
   and at resumptions, where wet-weather tyres are compulsory until the SC's orange lights
   are extinguished. A distortion and a fabrication are different findings, and this is the
   first.
2. *"'different specification' appears 6 times in the 2025 book and every one is either the
   replacement-component clause or the two-compound requirement. Not one of them is about a
   Safety Car."* — **REFUTED by executed enumeration.** My count is also 6; occurrence 5 is
   30.5 n), which is exactly about the Safety Car. The count was right and the
   classification of the six was wrong — the sentence reads as if all six were read, and at
   least one was not.
3. *"Article 54.3 is the penalties article"* — **CONFIRMED** by reading 54.3's own body
   ("The stewards may impose any one of the penalties below on any driver involved in an
   Incident: a) five seconds... b) ten seconds..."), plus the 4.2 and 17.3 cross-references.
   Note the corollary the issue misses: 54.3d) is cross-referenced BY the real rule, so the
   RAG citing it is not invention either — it is quotation.
4. The empirical refutation (16 cars pitted on lap 7, no penalties) — **VALID against the
   generalized version** and consistent with the real rule: Qatar's SC was a mid-race
   deployment in a dry race, so 30.5 n) did not apply. The cars' behaviour refutes the RAG's
   sentence, not the existence of the rule.
5. **Consequence for the fix list, and it matters:** fix direction 1 ("build
   `regulation_context` from `reg_out.articles` so an article number the retriever never
   returned cannot appear") **would not have prevented this defect** — 54.3d), 30.5 and 58.10
   all plausibly sit in the retrieved articles. The load-bearing fix is grounding the
   CONDITION: quote the retrieved chunk text with its conditional clause intact (or extract
   condition+rule pairs), and reword the SC question (fix 2), which remains correct as
   written.

**A second refutation, in the log's Finding 3:** *"the LLM's reasoning calls PIT_NOW
'MC-favoured' on that lap [Lusail 7]. The scores it was given are tied. The narrative
misdescribes the numbers in the same prompt."* — **REFUTED.** All four stored lap-7 samples
describe the tie correctly ("MC is neutral between stay out and pit", "the MC tie", "The MC
result is effectively neutral"). The "MC-favoured PIT_NOW" quote is from **lap 9**, where the
MC genuinely favoured PIT (-0.763 vs -1.765). No stored lap-7 narrative misdescribes its
numbers. (The tied point mass itself — E=P10=P90=0.30 both actions — is confirmed on every
lap-7 sample; that half of Finding 3 stands.)

The defect remains real and decisive: the distorted rule is false for the case it was applied
to, and it is the stated reason the flagship recommendation comes out wrong, with the
condition read once as a prohibition (passes 0-1) and once as a timing gate (pass 2 laps
9-10, verified: "allows the stop only once the Safety Car orange lights are extinguished").
But #826's headline, mechanism, and one of its three fixes need rewriting.

Severity: HIGH (an issue asserting "fabrication" where the truth is "distortion" sends the
fix at the wrong target, and the false 6-occurrence classification is executed-evidence-shaped
prose that was not executed).
Evidence: scratchpad extractions of all three PDFs; contexts quoted above at 2025 offsets
~117,031-117,600 and 2023/2024 equivalents; `thesis_windows.jsonl` Lusail 7-9
`regulation_context`; `thesis_repeats.jsonl` lap-7 reasonings (all passes).

### C. Monza's phantom Safety Car (#825) — CORE CONFIRMED on every axis I attacked, with two wrong sentences in the issue's own prose.

Confirmed, each by execution:

1. **Byte-identity, not similarity.** md5 over `rcm.parquet`: `italy_monza` == `italy_imola`
   (`e2e33ff113`), and `united_states_austin` == `united_states_las_vegas` ==
   `united_states_miami` (`9424fe9c4b`). "Identical rather than merely similar" holds at
   file-hash level for the RCM tables.
2. **63 is really Imola's discriminator — and the date column is an even harder one.**
   Census over `data/raw/2025/*/laps.parquet`: exactly one 2025 race has 63 laps (Imola);
   Monza has 53. Independently, the corpus rows are dated **2025-05-18** — Imola's race day;
   the real Monza raced in September. For the US group, 57 laps is NOT unique (Lusail,
   Melbourne, Miami, Sakhir are all 57), so the issue's `total_laps=57` line has no
   discriminating power on its own there — the identification rests on `session_key=10033`
   plus the corpus date **2025-05-04** (Miami's race day; Austin ran in October, Las Vegas in
   November). The conclusion is right; the load-bearing evidence for the US pair is the
   session key and date, not the lap count.
3. **Monza really has zero neutralised laps of its own.** `TrackStatus` scan over raw Monza:
   0 laps containing status 4/6/7.
4. **The slug map routes it.** `src/f1_strat_manager/gp_slugs.py:65` maps `"Monza":
   "italy_monza"`, so every surface that loads the corpus by slug serves Imola's session to
   Monza.
5. **`f1-eval decision-modes` immunity — verified, not just re-asserted.**
   `_decisions_in_window` (`decision_modes.py:346-351`) constructs `RaceState` with only
   driver/pace/risk plus `lap_inputs` fields; `RaceState.radio_msgs` and `rcm_events` default
   to empty lists (`strategy_orchestrator.py:261-262`); `run_lap` has no radio parameter at
   all (`engine.py:119-127`). No RCM ever reaches the situation agent on that path, so the
   published deterministic numbers cannot see the phantom.

Refuted or corrected in the issue's prose:

- **"Imola's corpus carries `SAFETY CAR DEPLOYED` on lap 29 and again on 46" is wrong about
  lap 29.** The lap-29 message is `VIRTUAL SAFETY CAR DEPLOYED` (VSC, ending lap 31); the
  full SC is lap 46 (in on 53). The methodology brief repeats the error ("Imola's corpus puts
  a Safety Car on laps 29 and 46"). Functionally the phantom still fires
  (`sc_currently_active` deliberately covers both, `vsc_active` distinguishes, #471), but SC
  and VSC have different strategic semantics (the pit-time saving halves), and prose that
  says "Safety Car on lap 29" describes the wrong regime for two of the phantom laps.
- **"15 of 24 races of 2025 carry an RCM `SAFETY CAR DEPLOYED`" is a substring artefact
  counted over partly-wrong corpora.** Executed count over the 24 corpus folders: **10**
  carry a full `SAFETY CAR DEPLOYED`; **5 more** carry only `VIRTUAL SAFETY CAR DEPLOYED`,
  whose text CONTAINS the searched substring — 10+5 = the quoted 15. Of the 10, one is the
  phantom itself (italy_monza), so the real full-SC race count is 9. The blast-radius
  conclusion ("not a Qatar-only path") survives, because the SC branch fires on
  `sc_currently_active`, which a VSC also sets — but the sentence is wrong about what it
  counted, and it counted it over a corpus three folders of which the same issue declares
  wrong.

Severity: the defect HIGH (already filed); the two prose corrections LOW-MEDIUM (they are in
an issue body and one is repeated in the methodology).
Evidence: md5 hashes, lap census, TrackStatus scan, corpus date/message dumps, code reads as
cited — all executed this session.

### F. Qatar and Budapest results — RECOUNT CONFIRMS EVERY NUMBER. One framing sentence does not survive.

Recounted from the three complete JSONL files:

- **Budapest, LLM arm: 42 of 42 STAY_OUT** in `thesis_repeats.jsonl` (14 lap-rows x 3 passes,
  the 14 = window 14-24 plus the 17-19 thesis window re-runs), plus 11/11 STAY_OUT in
  `thesis_windows.jsonl`. 53/53 across every stored rich sample. Confirmed.
- **Deterministic arm picks lap 19, and it is a scored, offset-0 transition, not merely a
  PIT_NOW.** `thesis_windows_nollm.jsonl`: laps 14-18 STAY_OUT, lap 19 PIT_NOW, 20-24
  STAY_OUT. Under the tier's own rule (`_pit_decision_lap`, `decision_modes.py:410-419`):
  predecessor lap 18 evaluated and non-pit, 19 > `_NO_PIT_BEFORE_LAP` (= 5,
  `guard_rails.py:67`), first pit action in [14, 24] -> transition at 19; actual stop 19 ->
  offset 0, scored. Confirmed.
- **Determinism of the no-llm arm independently reproduced:** `tierA_nollm.jsonl`'s Budapest
  LEC window (13-24, different window start) emits the identical verdict AND identical MC
  scores to 3 decimals on lap 19 (STAY -1.687 / PIT -1.645). Two separate runs, same output.
- **Lusail lap 7: STAY_OUT on all 3 repeat passes** (conf 0.86, 0.96, 0.92) and on the
  windows run (0.97). **Deterministic arm: STAY_OUT on 7, PIT_NOW on 8 and 9.** The repeat
  passes' only pit-class rows are pass 2, laps 9 and 10 — exactly as the log states. All MC
  score values in the log's table match the rows (tie 0.30 quadruple on lap 7; -1.37/+0.04 on
  8; -1.76/-0.76 on 9). Confirmed.

**The framing that does not survive: "given the same inputs on the same lap" (Finding 1).**
In rich mode the LLM sub-agents' outputs feed the Monte Carlo, so the MC scores the
orchestrator is handed are themselves perturbed per pass. On Budapest 19, PIT_NOW's score is
frozen at -1.645 across all 8 stored samples (both arms), but STAY_OUT's varies from -0.968
to -1.727 across the 7 rich samples. Consequence: **on the exact pass the log's table quotes
(conf 0.90), the MC favoured STAY_OUT (-0.968 vs -1.645)** — on that pass the LLM was
agreeing with its own MC, not declining a PIT-favoured one. Across the 7 rich samples the MC
ordering favours PIT in 4 and STAY in 3. The conclusion (the LLM path never calls the
Budapest stop; the deterministic path does, on the exact lap) is solid — but the mechanism
sentence "same inputs" is wrong at the scenario-score level, and the table's single "MC score
STAY / PIT" column (-1.69/-1.65) is the no-llm arm's pair presented as if both arms saw it.

Severity: LOW-MEDIUM (all headline numbers verified; one mechanism framing needs a
correction that slightly weakens the "LLM vetoes the MC" narrative at Budapest — the Qatar
veto, where the MC genuinely favoured PIT on laps 8-9 and the LLM cited the distorted rule,
is unaffected).
Evidence: recounts above, executed on the three complete JSONL files.

### G. Non-determinism percentages — ARITHMETIC CONFIRMED, and the recount identifies WHICH computation produced them: the one that pools two different contexts into "repeated laps".

Recount over `thesis_repeats.jsonl`, discordance per unique (race, lap), n = 22 (11 Budapest
14-24 + 11 Lusail 4-14):

| field | pooled (log's method): Bud 17-19 carry 6 obs from BOTH window definitions | passes-only (3 obs/lap, same context) |
|---|---|---|
| action | 2/22 = **9.1%** | 2/22 = 9.1% |
| confidence | 22/22 = **100%** | 22/22 = 100% |
| pit_lap_target | 15/22 = **68.2%** | 13/22 = **59.1%** |
| pace_mode | 17/22 = **77.3%** | 16/22 = **72.7%** |
| compound_next | 11/22 = **50.0%** | 11/22 = 50.0% |

The log's exact figures (9.1 / 100 / 68.2 / 77.3 / 50.0) reproduce ONLY under the pooled
reading: for Budapest laps 17-19 it counts six observations — three passes of the 14-24
window AND three passes of the cold-opened 17-19 window — as repeats of the same lap. Those
are not the same context: a lap 17 cold-opened at the window start carries different
`DecisionMemory` than a lap 17 warmed from lap 14 (the log itself makes this exact
distinction when it rules out the "window-start and memory confound" for the ACTION field).
Two of the three double-window laps are concordant within each window definition and
discordant only ACROSS definitions — that is context variation counted as path noise.

- The pre-committed 20% gate applies to `action`, which is 9.1% either way: **the gate
  outcome is unaffected.**
- `pit_lap_target` and `pace_mode` are overstated by 9.1 and 4.6 points respectively as
  "path noise"; the clean same-context figures are 59.1% and 72.7%. The qualitative sentence
  ("the field the thesis quotes is a draw, not a plan") survives comfortably at 59.1%.
- The denominator statement "over 22 repeated laps" should disclose that 3 of the 22 carry
  heterogeneous observation counts (6 vs 3) from two window definitions.

Severity: LOW (numbers right under their own method; the method mixes populations and the
prose does not say so).
Evidence: recount executed above, both ways, reproducing the log's figures exactly under
reading B.

