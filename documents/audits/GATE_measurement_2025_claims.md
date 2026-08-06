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

