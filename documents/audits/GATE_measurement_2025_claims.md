# ADVERSARIAL GATE — 2025 LLM-mode measurement claims

Gate over CLAIMS, not code. Date: 2026-08-06. Branch: `feat/measure-2025-llm-mode`.
Material: `MEASUREMENT_SESSION_2025_LOG.md`, `MEASUREMENT_2025_METHODOLOGY.md`,
`PROBE_llm_cost.json`, `documents/eval_reports/llm_2025/*.jsonl` (thesis_* complete,
tierA_* partial — no conclusions drawn from tierA totals), issues #825/#826/#827/#829,
PR #828 body.

Written incrementally. Each verdict appended as confirmed or refuted.

## Checklist

- [x] A. Cost claim (15.93 s/lap, 6 calls, 8,080+814 tokens, ZERO cacheable, $0.0071)
- [x] B. The fabricated regulation (#826) — fabrication vs distortion; what Art. 54.3 is
- [x] C. Monza's phantom Safety Car (#825) — Imola discriminator; slug routing; decision-modes immunity
- [x] D. Eval race-state defect (#829) — gap_ahead_s 2.0 on 100% of laps; pace_delta_s hardcode; rival branch
- [x] E. 2025 projection figures (552/24/86.05/59.60) + the causal "reads no model/tables" half
- [x] F. Qatar + Budapest results recount from JSONL
- [x] G. Non-determinism percentages recount + denominator check
- [x] H. Population discipline across the log + PR #828 body + the two docs pages
- [x] I. Lap-indexing correction (press 20 vs parquet 19)
- [x] J. Step 8's paired-lap refutation of my own two-window generalisation
- [x] K. The deterministic arm's own numbers vs the published `decision_modes` set

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

### E. The 2025 projection figures — ALL THREE ROWS REPRODUCE EXACTLY. The causal half is TRUE and I could not break it; but its COMPARISON is a subset against its own superset, and the sample has no power to detect the leak it claims to rule out.

**The arithmetic, re-measured independently** (`measure_projection_ground_truth(years=...)`,
executed this session, no API calls):

| scope | races | stops | within one | exact | mean signed | mean abs |
|---|---|---|---|---|---|---|
| 2025 only | 24 | 552 | 86.051% | 59.601% | +0.5670 | 0.6685 |
| 2023-2025 | 70 | 1,768 | 86.312% | 59.219% | +0.5752 | 0.6623 |
| 2023-2024 | 46 | 1,216 | 86.431% | 59.046% | +0.5789 | 0.6595 |

Every figure in the log's Step 4 table and in `documents/eval_reports/projection.{md,json}`
matches to the digit printed. `projection.json` carries `scoring_years: [2025]`,
`sample_size 552`, `races 24`, `within_one 0.860507`, `exact 0.596014` — consistent with the
md and with my recount. Nothing to refute here.

**The causal claim survives the hardest test I could build for it.** I did not stop at
reading; I ran the measurement with every measured-artefact reader on
`src/agents/position_projection.py` monkeypatched to `raise`
(`measured_tables`, `measured_undercut_band_s`, `measured_neutralisation_rate`,
`measured_clean_air_s`, `measured_racing_laps`, `traversal_seconds`, `_traversal_table`)
and with `builtins.open` + `Path.read_text` traced:

```
2025 with every measured reader booby-trapped: 552 24 86.051 59.601
suspicious files opened: []   (mc_measured_v1.json, model_config.json, *.pkl/*.joblib/*.pt)
distinct non-parquet files opened: []
total files opened: 24        (one laps.parquet per race)
```

A static call graph agrees: from `project_positions` the reachable set is exactly
`{_usable_rivals, driver_time_delta, rival_time_deltas, _terminal_gaps, _tyre_cost_s,
_stop_residual_s}` and **no** measured-artefact reader. The `ProjectionConfig` fields read on
that path are `racing_laps, fresh_gain_s, cliff_loss_s, neutralisation_saving_s, deg_cost_s,
future_neutralisation_prob, mandatory_stop_pending, clean_air_gain_s,
neutralisation_onset_rate` — the first four pinned by `_GROUND_TRUTH_CONFIG`, the rest inert
defaults (`None`/`0.0`). **The one field whose default IS a frozen copy of a measured table
(`undercut_band_s = DEFAULT_UNDERCUT_BAND_S = 4.91`, the `undercut_band` table's `u_band_s`)
is not read on this path at all** — it is only used by `undercut_targets`
(`position_projection.py:842`), which the ground truth never calls. I went looking for exactly
this and it is genuinely unreachable. So *"consumes no learned model and none of the seven
measured tables"* is **CONFIRMED**, and the published sentence is safe.

**What I do dispute, and it is the reasoning rather than the result:**

1. **The comparison in the code comment is a subset against the superset that contains it.**
   `src/strategy/eval/projection.py:56-57` argues the point with *"86.05% against 86.31%"* —
   but the 552 stops of 2025 are 31% of the 1,768 stops of 2023-2025, so those two numbers
   are algebraically pulled together and the 0.26-point gap is not a train/holdout contrast at
   all. The **disjoint** contrast is 86.05% (2025) against 86.43% (2023-2024), 0.38 points. The
   log's Step 4 table does have the disjoint row; the comment that will be read by the next
   person editing the scope does not, and it is the one phrased as an argument.
2. **The sample could not have detected the leak the sentence rules out.** At n=552 the
   within-one 95% interval is 83.16%-88.94% (half-width **2.89 points**), and the disjoint
   train/holdout difference is **z = 0.21** (0.38 pp against a 1.77 pp standard error; exact:
   z = 0.22). Treating stops as independent flatters this — they cluster within race and
   driver, so the true interval is wider still. So "the gap is 0.4 points, therefore nothing
   leaked" is an argument this design cannot make in either direction: **a real leak of 2
   points would also have printed as 'barely moves'.** The code-reading argument (point above)
   is the one carrying the claim; the smallness of the gap is corroboration with no power, and
   the prose currently presents it as the evidence.
3. **"six tables" vs seven.** `src/strategy/eval/projection.py:5-6` says
   *"`data/mc_measured_v1.json` carries six tables the Monte Carlo scorer reads at runtime"*.
   The file carries **seven** (`clean_air, gap_density, neutralisation_rate, sc_window,
   status_mix, stop_hazard, undercut_band`), `_TABLE_PURPOSE` has seven entries, the report
   renders seven rows, and the session log says "seven" — the module docstring is the only
   place that says six.

**On the "seasons scored" row being misread — the markdown is safe, the JSON and the HEADER
are not.**

- The md is fine: the accuracy section says `seasons scored | 2025`, and the tables section
  independently says *"Counted off 70 races (2023, 2024, 2025)"*. A reader of the md cannot
  conflate them.
- **`projection.json` carries no season scope for the tables at all.** Its payload is
  `scoring_years: [2025]` plus a `tables` list of `{table, answers, cells, present}`. The
  tables' own 70-race / 3-season scope exists only in the prose the JSON does not carry, and
  `projection.py:85-86` states the JSON is the machine-read surface. A consumer joining
  `scoring_years` to `tables` gets the wrong scope with no way to notice — the exact "half a
  fix" the docstring at `build_projection_report` (`:427-431`) warns about, left open in the
  machine-readable half.
- **The report HEADER makes a season claim that is false for its own second half.**
  `projection.md:4` reads `dataset data/raw laps 2025 (RAW, not featured)` and governs the
  whole document, including the tables section counted off 2023-2025. `build_header` is called
  once with the ground truth's scope (`projection.py:444`).

Severity: the numbers LOW (nothing wrong); the reasoning MEDIUM (a no-power comparison
published as the evidence, and the subset/superset framing sits in the comment that governs
future edits); the JSON/header scope MEDIUM-LOW (machine-readable surface asserts one scope
over two populations, which is the defect the whole rescope exists to remove); "six tables"
LOW.
Evidence: booby-trapped runtime measurement and file trace above; AST call graph over
`position_projection.py`; `data/mc_measured_v1.json` key list; `documents/eval_reports/projection.{md,json}`;
binomial arithmetic shown.

### I. The lap-indexing reconciliation — THE OFFSET IS SOURCEABLE, BUT "the press convention" IS REFUTED. It is variation between accounts, and the session applies it selectively: +1 at Budapest to rescue the thesis, 0 at Qatar to rescue the thesis.

**The parquet mechanics are exactly as described.** `data/raw/2025/Budapest/laps.parquet`:

| driver | in-lap (`PitInTime`) | out-lap (`PitOutTime`) | in-lap `LapTime` | out-lap `LapTime` | compound / stint on the out-lap |
|---|---|---|---|---|---|
| PIA | **18** @ 01:22:35.491 | **19** @ 01:22:56.632 | 1:24.668 | 1:38.914 | HARD, stint 2 |
| LEC | **19** @ 01:23:55.108 | **20** @ 01:24:16.388 | 1:24.552 | 1:39.083 | HARD, stint 2, TyreLife 1 |

FastF1's convention is confirmed on the data, not assumed: the in-lap carries `PitInTime` and only
the pit-entry deceleration (LEC 1:24.552 against 1:22.316 the lap before), and the **out-lap**
carries `PitOutTime`, the stationary time (1:39.083) and the new compound. And the physical claim
behind the offset holds at this circuit: LEC crossed the timing line **1.723 s after** entering the
pit lane (lap-19 `Time` 01:23:56.831 vs `PitInTime` 01:23:55.108), PIA 1.648 s after. Both cars were
already inside the pit lane when their lap counter ticked over, so "the lap the car spends in the
pit lane" is genuinely lap 20 / lap 19.

**The thesis body is in parquet indexing and is precise about it.** `05_resultados.tex`, 5.5.2:
*"Charles Leclerc lidera desde la pole position hasta la **vuelta 18**, cuando Oscar Piastri entra a
boxes... Ferrari cubre el movimiento **al cierre de la vuelta 19**"*. "Al cierre de la vuelta 19" is
literally what the data shows. Only the section-5.5 preamble says *"la parada de Leclerc en V20"*.

**Now the part that does not survive.** The log states this as a fact about the press: *"The press
accounts of the race number the same two stops PIA lap 19, LEC lap 20"*, and derives from it a
systematic rule (*"the timing-screen convention the press follows counts the lap the car spends in
the pit lane... It is a one-lap indexing offset, not a factual disagreement"*). I checked reachable
accounts and **they do not agree with each other**:

| account | PIA | LEC |
|---|---|---|
| Wikipedia, 2025 Hungarian Grand Prix race report (direct fetch): *"On lap 18, Piastri pitted from second to take hard tyres, re-joining the race in fifth"* / *"Leclerc responded the following lap"* | **18** | **19** |
| Autosport live text: *"he pitted on lap 19 and rejoined behind Alonso"* | **19** | — |
| Secondary race report surfaced by search: *"Piastri pitted on Lap 19 ... the Ferrari covered him off with a stop one lap later"* (search summary, not a direct fetch — weaker evidence) | 19 | 20 |
| formula1.com's own race report (direct fetch) | no lap number | no lap number |

So **both numberings are in the press for this race**, and the most timing-derived account uses the
parquet's. "The press convention" does not exist as a single thing; the offset is an
account-to-account variation, and the log promotes one sample of accounts to a rule.

**The decisive test is inside this session's own material: Qatar.** Step 3 of the log reads *"16
cars pit on lap 7"* straight off `PitInTime` and treats the thesis's *"Diecisiete coches pitan en
esa misma vuelta [7]"* as describing the same event with **no offset applied**. I verified the
structure is identical to Budapest's: at Lusail, `PitInTime` counts are 16 on lap 7 / 1 on 8 / 1 on
9, and `PitOutTime` counts are 16 on lap 8 / 1 on 9 / 1 on 10. If the +1 convention were real, the
press would number those stops **lap 8**. It does not — Sky Sports: *"pitting on **Lap 7** provided
key strategic gain"*, and *"It was McLaren's decision not to pit their two cars **at the end of that
lap**"*, which is in-lap numbering stated explicitly. `PitInTime` lap 6 at Lusail is **empty**, so
the thesis's "vuelta 7" cannot be press-plus-one indexing under any reading.

**The consequence is the finding.** Within one session, the same convention question is answered
two different ways, each time in the direction that makes the thesis right: +1 at Budapest (so V20
is not an error) and 0 at Qatar (so "vuelta 7" is not an error). That is the shape the task asked me
to look for, and it is here. It is not fraud — each reading is locally defensible — but it is not a
reconciliation either, because no single rule generates both.

**And Occam points the other way for the thesis edit.** The preamble sentence is *"las cinco vueltas
en torno a la parada de Leclerc en V20"*; the executed command is `--laps 17-19`, three laps. Under
the press reading, "five laps around V20" is laps 18-22 — still not the window, so **two** repairs
are needed (V20 stays, "cinco"→"tres", and the window still does not centre on it). Under the plain
reading, "three laps around V19" is 17-19 exactly, so **one** repair fixes the sentence
(V20→V19, cinco→tres — a single consistent re-indexing). The log's recommendation, that only "las
cinco vueltas" is worth correcting, leaves the thesis with two lap numbers for one stop and no
statement of which convention each uses.

Severity: MEDIUM. No measured number moves; what moves is the thesis-correction list and a stated
mechanism that is presented as executed fact ("the press accounts number...") on a sample of
accounts that disagree.
Evidence: Budapest/Lusail parquet reads shown above (in-lap/out-lap, `PitInTime` vs lap `Time`,
per-lap stop censuses); `05_resultados.tex` §5.5 preamble and §5.5.2 body; the four press sources
listed, two by direct fetch.

### J. Step 8's refutation — THE CONCLUSION SURVIVES A 3x LARGER SAMPLE, but the refutation was computed on TWO RACES and eight drivers. It commits the error it corrects.

**My recount** (snapshot 2026-08-06, all `tierA_llm*.jsonl` against `tierA_nollm.jsonl`, paired on
`(race, driver, lap)`, first occurrence per key). The files are live: the paired count was **554**
on one command and **577** eleven minutes later, which is itself the reason no rate from these
files is quotable without a snapshot marker — Step 8 carries none.

At n = 554 paired laps (6 races, 43 race-driver windows):

| | LLM (`rich`) | deterministic (`no-llm`) | log's n=180 |
|---|---|---|---|
| STAY_OUT | 400 | 379 | 113 / 106 |
| PIT_NOW | 95 | 55 | 49 / 21 |
| UNDERCUT | 59 | 120 | 18 / 53 |
| **pit-class share** | **27.8%** | **31.6%** | 37.2% / 41.1% |
| identical action | **76.9%** | | 68.3% |
| same pit/no-pit class | **82.1%** | | 78.3% |

The log's own arithmetic is internally correct (every row and the crosstab sum to 180, and all five
derived percentages reproduce from its cells). **The direction of its conclusion also survives and is
now better supported**: the arms differ by 3.8 points on willingness to stop (against 3.9 at n=180),
and the "different KIND of stop" finding is much stronger at scale — the deterministic arm reaches
for `UNDERCUT` 120 times against the LLM's 59, and the largest disagreements are still
`UNDERCUT` to `STAY_OUT` (37) and `UNDERCUT` to `PIT_NOW` (29). `ALERT` remains 0 of 649 LLM rows.
**But every absolute number in Step 8 is now wrong by 9-11 points.**

**Which laps the refutation was actually computed on — recovered, not guessed.** The
`_merged_llm.jsonl` / `_merged_det.jsonl` pair (mtime **2026-08-06 16:21:20**) is the snapshot
immediately preceding Step 8: 164 paired laps, LLM `{STAY_OUT 100, UNDERCUT 18, PIT_NOW 46}`,
DET `{STAY_OUT 95, UNDERCUT 48, PIT_NOW 21}`, and a crosstab with the **same eight cells in the same
order** as the log's (80/18/17/16/15/12/3/3 against the log's 90/18/20/18/15/13/3/3, +16 laps).
`UNDERCUT 18` on the LLM side and `PIT_NOW 21` on the deterministic side are identical in both. It is
the same population, one snapshot earlier. Its composition:

- **Two races: Budapest 96 laps and Barcelona 68.** Nothing else.
- **Eight drivers: ALB, ALO, ANT, BEA, BOR, COL, GAS, HAM** — the alphabetical prefix of the field,
  with HAM contributing 4 laps. **No LEC and no PIA**, which are the two drivers the claim being
  refuted was about.
- **Monza's 103 LLM laps were silently dropped from the pairing**, because `_merged_det.jsonl` held
  no Monza rows at that moment (Monza's deterministic arm ran as a separate, radio-suppressed spec).
  A third of the available LLM evidence was invisible to the count.

**So the refutation of "two windows are not a system" was computed on two races, one of which
(Budapest) IS one of the two windows.** Sixty per cent of its laps come from the race the original
claim was drawn from. That is not a wider sample; it is the same window plus one neighbour, and the
document says of itself *"I generalised from Budapest and Lusail inside the same document that opens
by warning against exactly that"* — while doing it again two paragraphs later.

**And the race mix does bias the rates, measurably.** Per-race pit-class shares on my 554:

| race | n | LLM | DET | arm gap |
|---|---|---|---|---|
| Barcelona | 147 | 29.3% | 28.6% | **+0.7** (LLM higher) |
| Budapest | 164 | 33.5% | 38.4% | -4.9 |
| Monza | 152 | 27.0% | 36.8% | **-9.8** |
| Montreal | 34 | 26.5% | 20.6% | **+5.9** (LLM higher) |
| Monaco | 26 | 23.1% | 26.9% | -3.8 |
| Lusail | 31 | 0.0% | 0.0% | 0.0 |

**The sign of the arm difference flips by race**, and its range (-9.8 to +5.9) is nearly three times
the aggregate the log quotes (-3.9). An aggregate over an unbalanced race mix is not an estimate of a
system property; it is an estimate of this mix. Two further population facts belong on the same line
as any rate from this sample: **three of the nine Tier A races (Mexico_City, Silverstone, Suzuka —
332 deterministic laps) contribute zero paired laps**, and **Monza, which is 27% of the paired sample
and shows the largest arm gap, is the one race deliberately run with the radio corpus suppressed**.
The pairing stays internally valid (both arms got the treatment) but the aggregate silently mixes two
input regimes.

**Verdict on the refutation as a refutation:** the original claim ("the LLM layer is more reluctant")
was over-general and deserved retracting, and at n=554 it is still not supported. But **a partial,
non-uniformly-covered sample supports the refutation no better than two windows supported the claim**
— at the moment it was written it *was* two windows. The honest form is: *"on the laps measured so
far the two arms ask to pit at similar rates, and the Budapest/Lusail results are case results, not
a rate"* — the retraction without the counter-rate. The counter-rate is the same mistake with the
sign flipped.

Severity: MEDIUM-HIGH. The retraction is right; the evidence offered for it is the error it
retracts, and the rate is now published in a PR comment.
Evidence: recounts above; `_merged_{llm,det}.jsonl` reconstruction with mtimes; per-race
decomposition; the two live snapshots 554 then 577.

### K. The deterministic arm's own numbers — REPRODUCE EXACTLY. Non-comparability IS stated (credit where due), but the 24-lap shortfall is unexplained and the explanation on offer covers only 19 of 24.

**Reproduced to the digit** via `src.strategy.eval.llm_decision.measure(tierA_nollm.jsonl, 2025,
data/raw)`:

```
laps_measured 1066   rows_on_disk 1261   windows 80
eligible 100   scored 39 (39.0%)   coverage_verdict masked
exact 0.1282   within_one 0.3077   within_two 0.4615
mean_signed -2.308   mean_absolute 2.615
buckets: scored 39, no_call_in_window 35, no_boundary_in_window 24, min_stint 2
```

**On comparability, the writing is BETTER than the brief assumed and I could not find a place that
implies it.** `REPORT_partial.md` says it twice — *"it is not comparable to the published
`decision_modes.md` numbers, whose sample is different"* in the population paragraph, and *"The
published `decision_modes.md` numbers are NOT comparable to these: different sample"* in the paired
section — and the log's Step 7 says *"neither is comparable to the published tier"*. That is real
discipline and it should be kept.

**What is wrong is the stated REASON, and the trap it leaves open.** Put the two side by side:

| | Tier A `no-llm` | published `decision_modes.md` |
|---|---|---|
| scored share | 39.0% | 37.6% |
| coverage verdict | masked | masked |
| **exact lap** | **12.8%** | **31.3%** |
| **within one** | **30.8%** | **47.8%** |
| within two | 46.2% | 61.2% |
| mean signed | -2.31 | -1.52 |

**The two rows that look comparable are, and the rows that matter are 2.4x apart.** A reader scanning
"about 38% scored, masked" in both tables has every invitation to read the accuracy rows as
commensurate too, and "different sample" does not tell them the exact rate more than halves. Nobody
has quantified how much of the gap the sample explains, so the disclaimer disclaims without
informing.

**I tried to explain the gap mechanically and FAILED — reporting that rather than a plausible story,
because an unproven attribution written as a finding is exactly what verdict D downgraded.**

- *Window width, the obvious candidate*: **not it.** `llm_decision.score_window:135-136` uses the
  same imported `DECISION_WINDOW_LAPS = 5` as `decision_modes.py:546-547`. The scoring windows are
  identical by construction and the identity is test-asserted.
- *Partial window coverage*, since Tier A replays only `[drawn_stop-6, drawn_stop+5]` while
  `decision_modes` replays a contiguous span covering all of a driver's stops: **measured and
  largely refuted.** 81 of 98 eligible windows (82.7%) are fully covered, and restricting the scored
  set to fully-covered windows moves exact 12.8% to 14.3% and within-one 30.8% to 34.3%. That is
  about 1.5 of the roughly 18.5-point exact gap. It is not the mechanism.
- One real but small population note it did surface: **7 of the 98 eligible stops were never in the
  draw** (mean window coverage 0.45) — they are a driver's *other* stops that happened to fall inside
  a replayed span, and `score_window` grades a stop when as little as one of its eleven laps was
  evaluated. `REPORT_partial.md` describes the population as *"drawn by a seeded uniform draw"*;
  about 7% of what is graded was not drawn.

**The 24-lap shortfall is not explained anywhere, and the general explanation on offer is
incomplete.** Neither the log nor `REPORT_partial.md` mentions 1,066 — the report's coverage table
quotes an older snapshot (913 of 1,090, 83.8%). I reconstructed the planned set from
`spec_tierA_all.json` (91 windows, 80 distinct race-drivers, **exactly 1,090 laps**, 0 extra rows)
and diffed it:

| cause | laps | matches the report's stated explanation? |
|---|---|---|
| no lap row at all in the raw parquet (car retired) | 18 | yes — "a retired car" |
| lap row present, `Position` is NaN (Barcelona ANT L54) | 1 | yes — "a lap with no position" |
| **lap row present, valid Position, still not served** | **5** | **NO** |

The five: **Silverstone VER L15 (P2), VER L16 (P2), HAM L16 (P8), SAI L16 (P11) — all under
`TrackStatus 4`, a Safety Car** — and **Monaco GAS L8 (P19), a pit in-lap under `TrackStatus 12`**.
The report's sentence *"A gap in BOTH arms is the replay engine declining to serve a lap (a retired
car, a lap with no position)"* accounts for 19 of 24. The remaining 5 are a third mechanism — the
neutralised-and-pit-lap filter — and they are not incidental: four of them are Safety Car laps at
Silverstone, the race whose deterministic arm scores 1 of 11 with 10 `no_call_in_window`.

Severity: MEDIUM. The numbers are right and the non-comparability is stated; the shortfall figure is
absent, its stated cause is incomplete, and the near-identical scored shares are an unguarded trap.
Evidence: `measure()` output above; planned-vs-served diff against `spec_tierA_all.json` and
`data/raw/2025/*/laps.parquet`; window-coverage measurement; `llm_decision.py:135-136` against
`decision_modes.py:546-547`.

### H. Population discipline, swept — THE LOG AND THE GENERATED REPORT ARE DISCIPLINED. The PR BODY is not, one "(measured)" sentence is hardcoded and false, and the 70-race de-dup fixed one twin of two pairs.

**Credit first, because it is the majority of the material.** `REPORT_partial.md` opens with
*"**Population, stated once and applying to every rate below**"* naming the draw, the seed, the nine
races and the non-comparability — the correct pattern. `docs/pages/thesis.md:31` states *"Over 552
green-flag stops across the 24 races of 2025"* with the population in the same sentence as the rate,
`:33` explains why 2025, and `:41` states the tables' wider scope explicitly and adds *"The two
scopes are different on purpose and each is stated where it applies."* `decision_modes.md`'s Scope
section names its six races, its season, and its arm. The log labels both arms in every table.

**H1 (HIGH). `scripts/report_llm_2025.py:156` hardcodes a false claim and stamps it "(measured)".**

```
"$0.75/$4.50 per 1M. Zero prompt tokens are cacheable here (measured), so the "
"prompt bill is paid in full on every lap."
```

It is a string literal, not a computed value. **The JSONL rows carry no cached-token field at all**
(keys: `agents, completion_tokens, driver, lap, llm_calls, pass_index, profile, prompt_tokens, race,
recommendation, seconds, state`), so the report is structurally incapable of measuring what it says
it measured. The claim is also false — verdict A measured 18.2% and 34.8% cached on the completed
runs. This is the defect shape the brief names: **executed-evidence-shaped prose that was not
executed**, with the word "(measured)" doing the work, in the auto-generated artefact where a reader
trusts it most, and it will be re-emitted into every future report until the line is deleted. Note
the same report already prints 10.1 calls/lap in the paragraph above it, refuting the probe's 6 —
so the file contradicts itself.

**H2 (MEDIUM-HIGH). PR #828's BODY still carries five refuted claims.** The corrections exist only
as comments underneath it:

| in the body | status |
|---|---|
| "15.93 s/lap (max 17.4)" | refuted (18.07 / 22.92 measured) |
| "6/lap: 5 on `gpt-4.1-mini`, 1 on `gpt-5.4-mini`" | refuted (mode, not mean; 7.91) |
| "8,080 prompt + 814 completion per lap" | refuted (13,226 / 1,014) |
| "cacheable prompt tokens **zero** (the prompt drifts numerically every lap)" | refuted, and the mechanism is wrong |
| "$0.0071/lap, **~$0.43 per hour of running**" | refuted; off by ~4x and self-contradictory |
| "#826 — the RAG returns a **fabricated** Safety Car tyre rule" | refuted by verdict B: a distortion of the real Art. 30.5 n) |

A merged PR body is the durable summary; a reader who does not scroll into the comments gets the
retracted version of every cost figure and the wrong characterisation of #826.

**H3 (MEDIUM). The refutation comment publishes a rate with no population.** PR #828's third comment
states *"Over 180 paired Tier A laps"* with pit-class shares of 37.2% / 41.1% and never names the
races. Per verdict J that population is **Budapest + Barcelona, eight drivers, Monza excluded** — and
the comment's whole subject is that a rate must not be drawn from two windows. (Cosmetic, same
comment: one table cell renders as `\multicolumn — 68.3% of laps`, a broken LaTeX artefact in
Markdown.)

**H4 (MEDIUM). One figure measured in `no-llm` IS presented as an LLM-path property.**
`docs/pages/multi-agent.md:192` — a page whose first line defines the subject as *"a LangGraph
pipeline of six sub-agents"* — states *"the scored sample rises from 54 to 67 of 178, and agreement
within two laps rises from 51.9% to 61.2%"* **without saying anywhere on the page that those figures
come from `profile="no-llm"`, the layer with the LLM synthesis switched off.** `decision_modes.md`
says it plainly (*"never the LLM synthesis"*); the page quoting its numbers does not. This session
has now measured that the two arms differ (identical action on 76.9% of paired laps; the LLM arm
never calls the Budapest stop the deterministic arm hits exactly), so the omission is load-bearing
rather than pedantic. Note the shape: **a caveat was added to that exact sentence today** (the #829
race-state constants block) and it names the constants but not the arm — one defect fixed on a line
whose other defect went untouched.

**H5 (MEDIUM). The 70-race de-dup fixed one member of two pairs.** `mc_measured_v1.json` reports
`races_measured: 70`, and `projection.md` says *"Counted off 70 races"*. But:

| file:line | says | correct |
|---|---|---|
| `docs/pages/thesis.md:41` | "70 races of raw laps across 2023 to 2025" | ok |
| **`docs/pages/thesis.md:61`** | "The measured tables themselves, from **71 races** of raw laps" | STALE |
| `docs/pages/multi-agent.md:175` | "1,852 of them across the **70 races** of 2023-2025" | ok |
| **`docs/pages/multi-agent.md:300`** | "measured across **71 races**, the median gap between consecutive cars is 2.23 s" | STALE |

Both files were edited today; in both, the corrected line and the stale line describe **the same
artefact**. 71 is the retired count that included the duplicated 2023 Spanish GP.

**H6 (LOW). The train/holdout gap is published as two different numbers.**
`docs/pages/thesis.md:35` says *"the gap ... is **0.3 points**"*; PR #828's body and the session log
say *"the **0.4**-point train/holdout gap"*. The measured difference is 86.431 - 86.051 =
**0.380**. The page subtracted the rounded values (86.4 - 86.1) instead of rounding the difference.
Immaterial to the argument, but it is the same quantity printed two ways in two published places.

Severity: H1 HIGH, H2/H3/H4/H5 MEDIUM, H6 LOW.
Evidence: `scripts/report_llm_2025.py:156` and the JSONL key list; `gh pr view 828` body and
comments; `docs/pages/{thesis,multi-agent}.md` at the lines cited; `data/mc_measured_v1.json`
`races_measured`; arithmetic shown.

---

## Findings from this continuation (E, H, I, J, K), ranked

| # | sev | finding | `file:line` | concrete failing scenario |
|---|---|---|---|---|
| 1 | **HIGH** | A false claim is hardcoded into the generated report and stamped "(measured)". The rows it reads carry no cached-token field, so it cannot have been measured; the true value is 18-35%. | `scripts/report_llm_2025.py:156` | PR 7 quotes "zero cacheable, the prompt bill is paid in full on every lap" out of the auto-generated report, and it regenerates on every future run. |
| 2 | **HIGH** | Decision-agreement figures measured with the LLM switched off are published on the multi-agent page with no arm stated. | `docs/pages/multi-agent.md:192` | A reader takes "agreement within two laps 61.2%" as the shipped LangGraph system's accuracy. This session measured the arms agreeing on only 76.9% of paired laps, and the LLM arm never calls the Budapest stop the deterministic arm hits exactly. |
| 3 | **MED-HIGH** | Step 8's counter-rate was computed on **two races** (Budapest 96 + Barcelona 68 laps) and **eight alphabetically-first drivers**, with Monza's 103 LLM laps silently unpaired — inside the correction whose subject is "two windows are not a system". | `MEASUREMENT_SESSION_2025_LOG.md` Step 8; PR #828 comment 3 | The rate is published, cited as a refutation, and its own population is the error it refutes. All its absolutes are already 9-11 points stale. |
| 4 | **MED-HIGH** | PR #828's body still carries five refuted cost claims plus "fabricated" for #826. Corrections live only in comments. | PR #828 body | The merged PR body is the durable record; a reader who does not scroll gets $0.43/hour, 6 calls/lap and "zero cacheable". |
| 5 | **MEDIUM** | The published argument for "no leakage" is a **subset-against-its-own-superset** comparison, and the sample has **no power** to detect the leak (z = 0.21; 95% half-width 2.89 points). | `src/strategy/eval/projection.py:56-57` | A future edit widens the scope, sees "barely moves", and concludes no leakage — for a metric that *does* read a model, where the same non-comparison would be silent. |
| 6 | **MEDIUM** | "The press convention" is asserted as a systematic +1 offset; reachable accounts disagree with each other, and the session applies +1 at Budapest and 0 at Qatar, each time in the direction that makes the thesis right. | `MEASUREMENT_SESSION_2025_LOG.md` Step 0 vs Step 3 | The thesis keeps two lap numbers for one stop (V20 preamble, "cierre de la vuelta 19" body) with neither convention stated, and the correction list omits it. |
| 7 | **MEDIUM** | The 24-lap Tier A shortfall (1,066 of 1,090) is stated nowhere, and the offered mechanism accounts for only 19 of 24. Five laps had a running car with a valid position. | `REPORT_partial.md` coverage section | Silverstone VER L15/L16, HAM L16, SAI L16 (all `TrackStatus 4`) and Monaco GAS L8 (pit in-lap) are dropped by a third mechanism nobody has named — and Silverstone is the race that scores 1 of 11. |
| 8 | **MEDIUM** | The 70-race de-dup fixed one member of two pairs; the stale 71 sits on the same pages as the corrected 70, describing the same artefact. | `docs/pages/thesis.md:61`; `docs/pages/multi-agent.md:300` | Someone regenerating the tables reads "71 races" and believes the duplicate is still in. |
| 9 | **MEDIUM** | `projection.json` carries `scoring_years: [2025]` and a `tables` list with **no season scope**, and `projection.md:4`'s header asserts "data/raw laps 2025" over a document whose second half is 2023-2025. | `documents/eval_reports/projection.{md,json}`; `src/strategy/eval/projection.py:444` | A machine consumer joins the two and attributes 2025 to seven tables counted off 70 races — the exact half-a-fix the code's own docstring warns about, left open in the machine-readable half. |
| 10 | **LOW** | Module docstring says the file carries **six** tables; it carries seven. | `src/strategy/eval/projection.py:5-6` | — |
| 11 | **LOW** | The train/holdout gap is published as 0.3 (docs) and 0.4 (PR body, log); measured 0.380. Rounded values were subtracted instead of the difference being rounded. | `docs/pages/thesis.md:35` | — |
| 12 | **LOW** | The population is described as "drawn by a seeded uniform draw", but ~7% of the stops actually graded (7 of 98) were never in the draw — neighbours caught inside a replayed span. `score_window` grades a stop when 1 of its 11 laps was evaluated. | `REPORT_partial.md`; `src/strategy/eval/llm_decision.py:130-133` | — |
| 13 | **LOW** | Broken table cell renders as `\multicolumn — 68.3% of laps`. | PR #828 comment 3 | — |

## Fix list, ordered by value then risk

1. **Delete or compute the cacheability sentence** (`scripts/report_llm_2025.py:156`). Either drop it, or persist the cached-token count into the JSONL row and render the real share. Do this before PR 7 quotes the report. *(finding 1)*
2. **State the arm** on `docs/pages/multi-agent.md:192` — "measured on `profile="no-llm"`, the deterministic layer with the LLM synthesis off" — beside the existing #829 caveat. *(finding 2)*
3. **Rewrite Step 8 and PR #828 comment 3** to the retraction without the counter-rate, or re-derive the rate over the completed sample and print the race and driver composition on the same line. *(finding 3)*
4. **Edit PR #828's body** to the corrected cost table and "distortion of Art. 30.5 n)", so the merged record is not the retracted one. *(finding 4)*
5. **Re-argue the no-leak claim from the code, not the gap** (`projection.py:56-59`): cite the disjoint 2023-2024 figure (86.43%), say the scorer reaches no measured artefact — which is now verified by execution — and say the sample could not resolve a gap of that size either way. *(finding 5)*
6. **Correct 71 to 70** at `docs/pages/thesis.md:61` and `docs/pages/multi-agent.md:300`, and grep for further instances before PR 7. *(finding 8)*
7. **State the shortfall**: 1,066 of 1,090 served, 18 retired-car laps, 1 no-position lap, 5 neutralised/pit laps — and add the third mechanism to the report's explanatory paragraph. *(finding 7)*
8. **Carry the tables' season scope into `projection.json`** (a `tables_years` / `tables_races` field) and stop the report header asserting the ground truth's dataset over the tables section. *(finding 9)*
9. **Decide the thesis lap-indexing edit deliberately**: either re-index the §5.5 preamble to V19 (one consistent repair, matches the body and the Qatar case), or keep V20 and state the convention. Do not leave both unflagged. *(finding 6)*
10. Housekeeping: "six tables" to seven; 0.3 to 0.4; the `\multicolumn` cell; and a line in `REPORT_partial.md` noting that a small share of graded stops were not drawn. *(findings 10-13)*

## What I tried to break and could NOT

Listed so the coordinator knows which parts need no re-auditing.

- **The three projection figures.** Re-measured independently: 2025 = 552/24/86.051/59.601/+0.5670/0.6685; 2023-2025 = 1,768/70/86.312/59.219; 2023-2024 = 1,216/46/86.431/59.046. Every digit the log and `projection.{md,json}` print is right.
- **The causal claim that the scorer reads no learned model and none of the seven measured tables.** I attacked it three ways and it held every time: an AST call graph from `project_positions` reaches no measured-artefact reader; monkeypatching all seven readers to `raise` leaves the number unchanged at 552/24/86.051/59.601; and tracing `open` + `Path.read_text` shows **24 files opened, all `laps.parquet`, zero non-parquet**. I specifically hunted the one field whose default is a frozen measured value (`undercut_band_s = 4.91`) and it is genuinely unreachable — only `undercut_targets` reads it, which the ground truth never calls. **The claim is true as published.**
- **Claim K's numbers.** 100 eligible, 39 scored, 39.0%, `masked`, 12.8% exact, 30.8% within one, -2.308 mean signed, 1,066 laps — all reproduce exactly from `llm_decision.measure`.
- **Two candidate mechanisms for the Tier A vs `decision_modes` accuracy gap.** Window width: refuted, both tiers use the same imported `DECISION_WINDOW_LAPS = 5`. Partial window coverage: measured and largely refuted, 82.7% of eligible windows are fully covered and restricting to them moves exact only 12.8% to 14.3%. **I have no explanation for the gap and I am not offering one.**
- **A private copy of the shipped scorer inside the eval harness** (this repo's signature defect). I looked: `llm_decision.py` imports `green_flag_stops`, `_neutralised_laps`, `guard_rail_block`, `DECISION_WINDOW_LAPS` and the verdict/aggregate types rather than restating them. No fifth copy on this path.
- **The FastF1 lap mechanics behind the indexing offset.** Confirmed on the data, not assumed: the in-lap carries `PitInTime` and only pit-entry deceleration, the out-lap carries `PitOutTime`, the stationary time and the new compound, and both Budapest cars entered the pit lane ~1.7 s *before* the timing line. The physical basis of the offset is real; only its promotion to "the press convention" is not.
- **Step 8's internal arithmetic.** Every cell of both tables sums to 180 and all five derived percentages reproduce from its own crosstab. The numbers were right about the laps they described.
- **Step 8's qualitative conclusion.** At 554 paired laps the arms still ask to pit at similar rates (27.8% vs 31.6%) and the "different kind of stop" asymmetry is stronger, not weaker. The retraction of "the LLM layer is more reluctant" stands.
- **A statement anywhere implying the Tier A and `decision_modes` numbers are comparable.** I went looking for one and there is none — the opposite is stated three times.
- **The Tier A draw's own arithmetic.** `spec_tierA_all.json` expands to **exactly 1,090** distinct planned laps across 91 windows and 80 race-drivers, and the deterministic run served **zero** laps outside the plan.

*No repository file was modified except this report. No LLM API call and no simulator run was made; every number above came from pandas, AST, `gh`, four web fetches and the committed JSONL.*
