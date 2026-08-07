# GATE — PR #830, "four defects from the measurement"

**Role.** Adversarial gate over the FIXES and the CLAIMS made about them. Not an implementer:
no repository file is modified except this report.

**Branch under test.** `fix/four-defects-from-the-measurement` (9 commits, `a22c97d..28eeaa7`), base `dev` @ `aeeb7fc`.

**Prior art the author must not repeat.** `documents/audits/GATE_measurement_2025_claims.md` refuted
six claims by the same author on the measurement session this PR is fixing.

**Cost constraint honoured.** Zero OpenAI calls. Everything below is `profile="no-llm"`,
embeddings-only retrieval, OpenF1 HTTP, or pure-python reproduction.

---

## Checklist

| # | Claim | Verdict |
|---|---|---|
| A | #829 — `decision-modes` uses `build_race_state`; the levels are 21.2 / 37.9 / 51.5, 72 declines, 31 no_boundary | **CONFIRMED.** Re-ran `f1-eval decision-modes`: **all 178 verdicts and every aggregate reproduce bit-for-bit.** |
| A′ | …and the ~10-point drop is caused by `gap_ahead_s` + `pace_delta_s` **alone** | **REFUTED — F4** (a third input, `rainfall`, also changed) and **F5** (the stated mechanism is backwards on the served distribution) |
| A2 | `lap_inputs` free of agent imports; tests run without model weights | **CONFIRMED** — importing `decision_modes` pulls zero heavy modules |
| B | #825 — radio corpus fetched by circuit; the three corpora are their own race | **CONFIRMED on local disk; REFUTED for every other consumer — F3** (the Hub still serves the poisoned parquets) |
| B2 | `_disambiguate_by_circuit` raising is not a new crash path | **CONFIRMED** (only 2 callers, both safe) — with **F11**, **F12** |
| C | #826 — reworded SC question drops the bad chunk; root cause is a chunk that starts mid-word | **CONFIRMED in full**, at the chunk boundaries in Qdrant. Side effects **F6**, **F15** |
| C2 | The rewording makes no OTHER regulation question worse | **CONFIRMED for questions** (no other branch changed); **F7**, **F8** are prompt-level risks that this session cannot execute |
| D | #827 — a locked store raises `portalocker…BaseLockException` and the guard fires | **REFUTED — F1.** It raises `builtins.RuntimeError`; the guard never fires. **F2**: its four tests are vacuous |
| E | Regenerated report + docs match the JSON; no retired figure survives | **Numbers all match; retired figures DO survive — F9, F10, F14** |

---

## Findings

### F1 — HIGH — #827 is not fixed. `QdrantClient` raises `RuntimeError`, never `BaseLockException`, so the guard never fires.

`src/agents/strategy_orchestrator.py:1519-1543` (`_run_rag_agent_or_degrade`).

The fix catches `portalocker.exceptions.BaseLockException`. The library does not let that
exception out. At `.venv/Lib/site-packages/qdrant_client/local/qdrant_local.py:148-151`
(qdrant-client **1.16.2**, the pinned version) the lock acquisition is:

```python
except portalocker.exceptions.LockException:
    raise RuntimeError(
        f"Storage folder {self.location} is already accessed by another instance of Qdrant client."
        ...
    )
```

The portalocker exception is caught *inside qdrant* and re-raised as a bare
`builtins.RuntimeError` (no `from`, so `__cause__` is None). `except BaseLockException`
can never match it.

**Executed evidence — a real cross-process collision, no LLM, no real store.** Process A
holds `QdrantClient(path=D)`; process B calls `_run_rag_agent_or_degrade` with
`run_rag_agent` replaced by the one line `RagRetriever.__init__` actually executes
(`QdrantClient(path=D)`):

```
holder: HOLDER_READY
RESULT: THE EXCEPTION ESCAPED _run_rag_agent_or_degrade
   type: builtins.RuntimeError
   msg : Storage folder ...gate830e2e... is already accessed by another instance of Qdrant client.
```

Same result same-process and cross-process; `isinstance(e, BaseLockException)` is `False`
in both.

**So the symptom #827 describes is unchanged**: the exception still leaves `run_lap`, the
CLI's per-lap `except Exception` still renders a red row for the whole race, and the cause
is still named nowhere. The PR says `closes #827`.

**Where the wrong exception came from, and why nothing caught it.** `src/rag/retriever.py:299-301`
says *"A second `QdrantClient(path=...)` on the same storage directory raises `AlreadyLocked`"*.
That docstring is wrong, and the fix trusted it instead of executing it — a comment naming
the wrong mechanism, propagated into code.

### F2 — HIGH — the four tests for F1 assert about a failure mode that cannot occur, and one of them locks in the bug.

`tests/agents/test_rag_degrades_when_locked.py:35,47`.

All four tests monkeypatch `run_rag_agent` to raise `AlreadyLocked` / `LockException`
— exceptions the real path never produces. Green, and about the empty set.

`test_any_other_failure_still_surfaces` (`:60-74`) is worse than vacuous. Its premise is
"only the lock family is swallowed", and it proves it by raising
`RuntimeError("collection 'fia_regulations' not found")` and asserting it propagates. But
the real lock failure **is a `RuntimeError` too** — only the message differs. So the test
that is supposed to keep the `except` narrow is in fact the test that guarantees the real
lock error keeps escaping. Any correct fix (catching `RuntimeError` and discriminating on
the message, or catching at the `get_retriever` boundary) has to reckon with this test,
and its docstring — *"widening it to `Exception` makes this test fail"* — currently reads
as a defence of the defect.

### F3 — HIGH — #825's code fix ships; its data fix does not. The Hub still serves the poisoned corpora.

`data/.gitignore:2` (`/processed`) · `src/f1_strat_manager/data_cache.py:59,137,455`.

The PR states: *"The three corpora are rebuilt (`italy_monza` → session 9912, 53 laps,
zero Safety Car messages)"*, and `closes #825`.

On the author's disk that is true, and I verified it:

| slug | local `session_key` | local `total_laps` | local SC msgs |
|---|---|---|---|
| `italy_monza` | 9912 | 53 | **0** |
| `united_states_austin` | 9888 | 56 | 2 |
| `united_states_las_vegas` | 9858 | 50 | 4 |

But `data/processed/**` is gitignored and every consumer gets these parquets from
`VforVitorio/f1-strategy-dataset` via `ensure_radio_corpus` → `snapshot_download`.
Fetched from the Hub just now:

| slug on the Hub | `session_key` | `total_laps` | SAFETY CAR msgs at laps |
|---|---|---|---|
| `italy_monza/rcm.parquet` | **9987 (Imola)** | **63** | **29, 31, 46, 52, 53, 55** |
| `united_states_austin/rcm.parquet` | **10033 (Miami)** | **57** | 2, 3, 29, 30, 33, 34 |
| `united_states_las_vegas/rcm.parquet` | **10033 (Miami)** | **57** | 2, 3, 29, 30, 33, 34 |

Austin and Las Vegas are still byte-identical (9471 bytes each). **Monza's phantom Safety
Car is intact for everyone except the author.** Nothing in the PR re-uploads, and the
measurement log's "what is left" list does not mention it either.

The new data guard cannot catch this: `tests/audit/test_radio_corpus_is_its_own_race.py:78-81`
skips when `data/processed/race_radios` is absent, which is the CI case, and passes on the
one machine where the local rebuild happened. On a fresh clone it would fail — correctly —
but no configuration in this repo runs it there.

### F4 — MEDIUM — the causal attribution for the ten-point drop is incomplete: a third input changed, and it is never mentioned.

`src/strategy/eval/decision_modes.py:352` (`build_race_state(...)`).

Every artefact in this PR enumerates **two** changed inputs. The measurement log
(`MEASUREMENT_SESSION_2025_LOG.md`) makes it singular and causal: *"every accuracy band
dropped about ten points **once the tier stopped receiving a constant 2.0 s gap**."*

I enumerated all ten `RaceState` fields the two mappings produce, over the exact eligible
population — six races, every driver with a green-flag stop, every lap inside the replay
spans, **2744 laps**:

| field | laps that differ | note |
|---|---|---|
| `gap_ahead_s` | **2744 / 2744 (100%)** | old was 2.0 on every lap — claim confirmed |
| `pace_delta_s` | **2470 / 2744 (90.0%)** | claim confirmed |
| `rainfall` | **86 / 2744 (3.1%)** | **not mentioned anywhere** — all at 2025 Silverstone |
| `lap`, `total_laps`, `position`, `compound`, `tyre_life`, `air_temp`, `track_temp` | 0 | identical |

The old harness never passed `rainfall`, so it took the `RaceState` model default `False`
on every lap of every race — including a wet Silverstone. `build_race_state` reads it from
the weather state, and `RaceState`'s own docstring (`strategy_orchestrator.py:238-239`)
says the weather fields *"are forwarded to N14 (SC model) as contextual features"*. So a
third model input moved.

It is small — Silverstone accounts for 2 of the 51 changed verdicts — so the headline
survives. The defect is the enumeration, not the level. Fix the sentence, or measure the
rainfall contribution and say it is negligible; do not leave a report that names its own
input change and omits one third of it.

### F5 — HIGH — the *mechanism* the PR gives for the drop is measured backwards on the served distribution.

PR #830 body, §#829: *"Coherent rather than suspicious: **a real gap of 0.4-1.0 s is far
closer than 2.0**, so N27 sees a tighter fight, the stack commits more often and lands
further from the real lap."* The retired caveat this descends from (removed from
`decision_modes.py::_render_table` in this PR) sourced it as *"Measured against the builder
on 2025 Barcelona, the real gaps over the same laps run 0.431 to 1.075 s."*

That was a probe. Measured on all 2744 laps the metric actually runs on:

```
count 2744   mean 4.778   std 6.554
min 0.000   25% 1.012   50% 2.507   75% 5.686   max 54.327

share of laps with real gap < 2.0 : 44.2 %
share of laps with real gap > 2.0 : 55.8 %
share inside the quoted 0.431-1.075 band : 16.5 %
```

**The median real gap (2.507 s) is WIDER than the 2.0 s constant, and the mean is 2.4x it.**
Excluding the 199 leader laps where 0.0 means "nobody ahead", 60.2% of laps are wider.
Even 2025 Barcelona — the race the 0.431-1.075 figure was taken from — has median 3.294 s
and only 11.5% of its laps inside that band; per-race medians are Silverstone 1.52,
Lusail 1.91, Monza 2.64, Marina_Bay 2.86, Monaco 3.04, Barcelona 3.29.

So "N27 sees a tighter fight" is false for the majority of laps. The direction of the
result may still be right, but the explanation offered for it is not the one the data
supports, and it is the sentence a reader will quote when asked why the numbers fell. This
is the repo's own "a probe is not a distribution" shape, carried forward into the
explanation of the fix for it.

Committed footprint: `documents/audits/MEASUREMENT_SESSION_2025_LOG.md:579` — *"the real gaps
above run 0.43 to 1.08"*. `decision_modes.md` and `docs/pages/multi-agent.md` do **not** repeat
the mechanism, which limits the damage; the PR body and the log do.

### F6 — MEDIUM — #826's fix makes N30's structured citation list *less* topical, and the PR's table hides it.

`src/agents/rag_agent.py:223` · `src/agents/strategy_orchestrator.py:1554-1565`.

PR table for the reworded question: *"top hit | Art. **55.3**, 0.750"*, *"rest of top 5 |
**55.10, 55.14**"*.

Re-run at the configured `top_k = 5` (so the top 5 IS the whole retrieval), embeddings only:

| rank | NEW question — payload `article` | score | what the chunk is |
|---|---|---|---|
| 1 | **`''` (empty)** | 0.7503 | text containing "55.3 The safety car may be brought into operation…" |
| 2 | **`''`** | 0.7445 | continuation of 55.3 |
| 3 | **`''`** | 0.7209 | Art. 15.4, *stewards must be present* |
| 4 | **`Article 14.6.1`** | 0.7148 | **Driver Cooling System** — not a safety-car article |
| 5 | `Article 55.14` | 0.7115 | genuinely relevant |

Two problems the table conceals:

1. **"Art. 55.3" is not what the system records.** The rank-1 chunk's `article` payload is the
   empty string; "55.3" is only visible by reading the chunk body. `rag_agent.py:223` builds the
   citation list as `list(dict.fromkeys(c.article for c in chunks if c.article))`, and
   `rag_agent.py:79` says explicitly *"the LLM may hallucinate them. Use the `articles` field
   instead."* Executed:

   ```
   OLD: raw labels ['Article 54.3','Article 5.3','Article 54.3','','Article 49']
        ctx.articles = ['Article 54.3', 'Article 5.3', 'Article 49']
   NEW: raw labels ['','','','Article 14.6.1','Article 55.14']
        ctx.articles = ['Article 14.6.1', 'Article 55.14']
   ```

   So after the fix, the authoritative citation list for *"what is the procedure when the
   safety car is deployed"* leads with the **Driver Cooling System** article, and **Art. 55.3 —
   the hit the PR names as the win — is not in it at all.**

2. **Ranks 3 and 4 are omitted from the PR's table.** "rest of top 5: 55.10, 55.14" reports two
   of the four non-top hits and drops the two least favourable, including the irrelevant 14.6.1
   that outscores the relevant 55.14.

The retrieved *prose* is better; the retrieved *metadata* is worse. Since #826's second half is
precisely "a chunk carries a wrong article label and that is why the citation looked
authoritative", shipping a state where the good chunks carry no label at all is a new instance
of the same class, not a mitigation of it.

### F7 — MEDIUM — the fix for #826 ships a worked example that violates its own new rule, six lines below it.

`src/agents/strategy_orchestrator.py:1829-1832` (rule) vs `:1844-1847` (example).

New rule 3:

> "If regulation_context cites an article, quote it AND the condition under which that article
> applies. **If it states no condition, say the regulation context was inconclusive** rather
> than inventing one, and do NOT let it override the numeric evidence."

The shipped few-shot example, in the same prompt string, labelled *"Example of a rich reasoning
paragraph … use as shape"*:

> "The mandatory two-compound rule (see the regulation context above for the article, it is
> renumbered between seasons) requires no fewer than two dry compounds used, so switching to
> HARD satisfies it."

That sentence states a regulation rule **with no condition whatsoever** and lets it support the
action — exactly what rule 3 forbids. And the two-compound rule is itself conditional (it does
not bind when wet-weather tyres have been used), so the example is not merely unconditioned, it
is unconditioned about a *conditional* rule — the very failure #826 is about.

A model handed a rule and a contradicting exemplar marked "use as shape" follows the exemplar.
This is the `AUDIT_A3_prompt_vs_code` shape recurring inside the fix for it.

**Unmeasurable here, and I am saying so rather than implying otherwise:** the effect of rule 3
and of the new system prompt on real answers cannot be checked without LLM calls, which are
forbidden this session. What is executed above is the prompt text itself.

### F8 — MEDIUM — a further risk in rule 3 that nobody has measured: it penalises correctly-quoted UNCONDITIONAL rules.

`src/agents/strategy_orchestrator.py:1831`.

"If it states no condition, say the regulation context was inconclusive" is symmetric where the
regulations are not. An article that genuinely binds without qualification now has to be
reported as inconclusive, and the instruction adds "do NOT let it override the numeric
evidence". #826 was a rule applied too widely; the correction as written creates the opposite
failure, and there is no measurement in this PR bounding how often it fires. Flagging as a
design risk with its mechanism named, not as a confirmed defect.

### F9 — MEDIUM — the report declares pre-2026-08-06 figures incomparable and then keeps quoting three of them; one stale claim is provably about a different population.

`src/strategy/eval/decision_modes.py::_render_table` → `documents/eval_reports/decision_modes.md`.

The new section states: *"**Figures generated before 2026-08-06 are not comparable to these.**"*
The same generated file then carries, as hand-written prose in the same `_render_table`:

- `decision_modes.md:14` — *"still moves with `DECISION_WINDOW_LAPS` (measured **-0.33 / -1.29 /
  -2.50** at w=3/5/10 on one race)"*. Measured on the retired input set.
- `decision_modes.md:39-41` — *"on the measured 2025 Monza sample this was **4 of 4 occupants**,
  one of them flipping to STAY_OUT on the exact lap the team really stopped"*.

The second is checkable and the population has moved underneath it. Monza's
`no_boundary_in_window` occupants, from the two JSONs:

```
OLD (dev): 2 occupants  -> [('STR', 49), ('PIA', 45)]
NEW (PR):  4 occupants  -> [('VER', 37), ('NOR', 46), ('HAD', 32), ('PIA', 45)]
```

The claim already disagreed with the report it sat in on `dev` (which had 2, not 4), and it now
matches the *count* by coincidence while three of the four drivers are different. Neither the
"4 of 4 withdrew" behaviour nor the "flipped on the exact lap" detail has been re-checked
against this run.

### F10 — MEDIUM — the file the PR edits to say "#829 DONE" still tabulates the retired figures as the live baseline.

`documents/audits/MEASUREMENT_SESSION_2025_LOG.md:723-731`.

The PR's last commit edits this file to mark #829 **DONE**. Eighty lines above, untouched, the
session's headline comparison table reads:

| | this arm (9 races, product race state) | **published `decision_modes.md` (6 races)** |
|---|---|---|
| scored | 39 (39.0%) | **67 (37.6%)** |
| exact lap | 12.8% | **31.3%** |
| within one lap | 30.8% | **47.8%** |
| mean signed error | -2.31 | **-1.52** |

`decision_modes.md` no longer publishes any of those. The column is honestly *labelled*
downstream ("the published column is measured on the constant 2.0 s gap … of #829"), so this is
not a false claim — but its header says "published `decision_modes.md`" and it is not, and the
substantive conclusion moved: the LLM-vs-deterministic exact-lap gap was 12.8 vs 31.3 and is now
12.8 vs **21.2**, less than half as wide. Updating one artefact and leaving its comparison table
is the twin-not-fixed shape.

### F14 — HIGH — the public docs page now compares 54 (old bounds, old inputs) with 66 (new bounds, NEW inputs), and its defence covers only half of what it defends.

`docs/pages/multi-agent.md:192`.

Before this PR: *"the `min_stint` exclusion bucket falls from 17 stops to 5, and the scored
sample rises from **54 to 67** of 178, and agreement within two laps rises from 51.9% to 61.2%"*.
Both 54 and 67 came from the same constant-fed harness, so it was a **single-variable** claim
about the #716 recalibration.

After this PR: *"the `min_stint` exclusion bucket falls from 17 stops to 5 and the scored sample
rises from **54 to 66** of 178."*

The 54 is unchanged — its provenance is `MEASURE_744b_decision_effect.md:13` /
`GATE_716_calibration.md:232`, pre-recalibration and pre-#829. The 66 is post-recalibration
**and** post-input-fix. Two variables moved between the two numbers, in the one sentence on the
page that exists to attribute an effect to the recalibration. The controlled comparison that
sentence used to make no longer exists anywhere: nobody re-ran decision-modes with the OLD
bounds and the NEW inputs.

And the caveat box directly beneath defends it with: *"The recalibration effect this paragraph
describes still holds in direction (**a smaller exclusion bucket is arithmetic**)."* That is a
true statement — and it covers only the 17 → 5 half. Measured, the guard-rail buckets are
input-independent and the scored count is not:

```
guard-rail buckets   closing_laps  dev=4   PR=4      min_stint  dev=5   PR=5
model-dependent      scored        dev=67  PR=66     no_call    dev=78  PR=72
                     no_boundary   dev=24  PR=31
```

51 of the 178 verdicts changed bucket or offset when the inputs changed. So "arithmetic"
justifies the bucket number and is silently extended to the scored number, which is the one the
sentence is actually arguing from. A true statement carrying a claim it does not support — the
shape this repo names first in its own lessons.

**Fix:** either restore a single-variable pair (re-run with old bounds + new inputs, cheap: one
constant change and one `f1-eval decision-modes`), or drop the "scored sample rises" clause and
keep only the 17 → 5 arithmetic the caveat actually covers.

### F15 — MEDIUM — the reworded question fixed the hallucination by moving off the topic that motivates the query.

`src/agents/strategy_orchestrator.py:1554-1565`.

`_build_rag_question`'s Safety Car branch exists to inform a **pit decision** — every sibling
branch of the same function asks about tyres, compounds and stops. The rewording changes the
subject from "pit stops and tyre changes during a Safety Car" to "the procedure for drivers and
teams".

Measured on the retriever, top 10, counting chunks that mention the pit lane at all:

```
OLD question: ranks [1, 2, 3, 5, 6, 7, 8, 9, 10]   (9 of 10)
NEW question: ranks [5, 8]                          (2 of 10)
```

**Stated fairly: the old question did not retrieve the right rule either** — its top 5 was the
mislabelled 30.5 n) chunk, a 2023 duplicate, a tyre-supply chunk and Art. 49. So this is not
"the fix broke retrieval". It is that the regulation block handed to the orchestrator on a
Safety Car lap now contains deployment conditions (55.3), SC duration (55.10/55.14), stewards'
attendance (15.4) and the Driver Cooling System (14.6.1), and the rules a strategist needs —
pit-lane status and Art. 55.8 overtaking — are in the corpus (`55.13` in 1 chunk, `55.14` in 3,
"overtaking is forbidden" in 2, all 2025) but not at the top of either ranking.

The PR is right to keep #826 open. This belongs on it: the target the re-chunking work should
be measured against is "does the block answer the pit question", not "is the bad chunk gone".

### F11 — LOW — `resolve_session`'s twin: the non-Race branch still takes `sessions[0]`.

`src/data_extraction/openf1/radio_dataset_builder.py:344`.

The disambiguation is applied only inside `if session_type == "Race":`. Any other
`session_type` still falls through to `return sessions[0]`, which for Italy or the United
States is the same country-keyed lottery #825 is about. No caller uses a non-Race type today
(`build_and_write`, `prepare_session_bundle` and the `__main__` demo all take the default), so
this is latent, not live — but the fix's own docstring calls this "the one-twin-fixed shape
this repo keeps paying for" while leaving a twin one line below.

Also: `resolve_session`'s docstring (`:296-317`) was not updated. It documents neither the new
`circuit_short_name` keyword nor the new `Raises`, and its Sprint paragraph still explains
`sessions[0]` as the hazard when a second, larger `sessions[0]` hazard now has its own method.

### F12 — LOW — the CLI's documented fallback is dead in the only case it exists for.

`scripts/build_radio_dataset.py:548-559`.

```python
bundle = self._builder.prepare_session_bundle(..., circuit_short_name=race.circuit_short_name)
# Fall back to the bundle's session payload if discover_races somehow saw a row
# that was missing the field.
circuit_short_name = race.circuit_short_name or bundle.session.get("circuit_short_name")
```

If `race.circuit_short_name` is falsy, `prepare_session_bundle` raises three lines earlier for
exactly the multi-race countries the fallback is written for. For single-race countries the
fallback works but is not needed. It is a comment describing a safety net that cannot deploy.
Contained: the caller's `except Exception` records `status="failed"`, so this degrades a GP, not
the run.

### F13 — LOW — `_disambiguate_by_circuit`'s docstring calls a VSC a Safety Car.

`src/data_extraction/openf1/radio_dataset_builder.py:358-359` says Monza *"was being served a
Safety Car on laps 29 and 30"*. Imola's RCM (verified on disk) has `VIRTUAL SAFETY CAR DEPLOYED`
at lap 29 / `ENDING` at 31, and the only real `SAFETY CAR DEPLOYED` at lap 46. `rcm_state.py:158`
tracks the two separately (`sc_kind == "VSC"`), and Art. 56 makes a VSC materially different for
pit-stop value — a distinction this codebase already paid for in #471. The conclusion is
unaffected (both set `sc_active`), and the PR body's separate "laps 29 and 46" is about Imola's
own deployments and is correct. Naming precision only.




---

## Reproduction of the headline (claim A)

`f1-eval decision-modes` re-run from a clean worktree at `28eeaa7`, same `data/`, ~55 min:

```
committed agreement == my rerun agreement   (all 9 fields, exact)
committed buckets   == my rerun buckets     {closing 4, min_stint 5, no_boundary 31, no_call 72, scored 66}
verdicts differing between committed and my rerun: 0   (of 178)
```

The published levels are real and deterministic. Everything I found is about what is *said*
around them, and about #825 / #827.

---

## Findings ranked

| # | Sev | One line | Where |
|---|---|---|---|
| F1 | **HIGH** | #827's guard catches an exception qdrant never raises; the bug is unchanged | `strategy_orchestrator.py:1519-1543` |
| F2 | **HIGH** | Its four tests assert about the empty set; one of them locks the bug in | `tests/agents/test_rag_degrades_when_locked.py:35,47,60` |
| F3 | **HIGH** | #825's data fix is local-only; the Hub still serves Imola-as-Monza and Miami-as-Austin/Vegas | `data/.gitignore:2`, `data_cache.py:137,455` |
| F5 | **HIGH** | The stated mechanism for the drop is backwards on the served distribution | PR body §#829 · `MEASUREMENT_SESSION_2025_LOG.md:579` |
| F14 | **HIGH** | The docs page now compares 54 (old bounds, old inputs) with 66 (new bounds, new inputs) | `docs/pages/multi-agent.md:192` |
| F4 | MEDIUM | A third input (`rainfall`) changed and is enumerated nowhere | `decision_modes.py:352` |
| F6 | MEDIUM | N30's citation list now leads with the Driver Cooling System article | `rag_agent.py:223` |
| F7 | MEDIUM | The new prompt rule is contradicted by its own worked example six lines below | `strategy_orchestrator.py:1829` vs `:1844` |
| F8 | MEDIUM | The new rule penalises correctly-quoted unconditional rules (unmeasured) | `strategy_orchestrator.py:1831` |
| F9 | MEDIUM | The report declares old figures incomparable, then quotes three; one is provably stale | `decision_modes.py::_render_table` |
| F10 | MEDIUM | The file edited to say "#829 DONE" still tabulates the retired figures as baseline | `MEASUREMENT_SESSION_2025_LOG.md:723-731` |
| F15 | MEDIUM | The rewording fixed retrieval by leaving the topic the query exists for | `strategy_orchestrator.py:1554` |
| F11 | LOW | `resolve_session`'s non-Race branch still takes `sessions[0]` | `radio_dataset_builder.py:344` |
| F12 | LOW | The CLI's documented fallback is unreachable in the only case it is for | `build_radio_dataset.py:548-559` |
| F13 | LOW | A VSC is called a Safety Car in the new docstring | `radio_dataset_builder.py:358` |

---

## Fix list, ordered by value then risk

1. **F1 + F2 — make the #827 guard actually catch the failure.** `except RuntimeError` narrowed
   on the message qdrant emits (`"is already accessed by another instance"`), or better, catch at
   the `get_retriever()` / `RagRetriever.__init__` boundary where the lock is taken rather than
   around the whole agent call. Rewrite the four tests to provoke a **real** cross-process
   collision (a subprocess holding `QdrantClient(path=tmp)`), not a synthesised `AlreadyLocked`.
   Rewrite `test_any_other_failure_still_surfaces` so its "other failure" is not itself a
   `RuntimeError`. **And fix `src/rag/retriever.py:299-301`**, the docstring that caused this.
   Do not close #827 until a two-process run is shown to produce laps.
2. **F3 — re-upload the three corpora to `VforVitorio/f1-strategy-dataset`**
   (`scripts/upload_radio_corpus.py`), verify by re-fetching `italy_monza/rcm.parquet` and
   asserting `session_key == 9912`, and only then close #825. Consider running the corpus guard
   against a Hub fetch in a scheduled job, since it can never fire in PR CI.
3. **F14 — restore a single-variable comparison on the docs page**, or drop the "scored sample
   rises" clause. One re-run with the old bounds and the new inputs settles it.
4. **F5 + F4 — replace the mechanism sentence with the measured distribution** (median 2.507 s,
   mean 4.778 s, 55.8% of laps *wider* than the 2.0 s constant), and add `rainfall` to the
   enumeration of what changed, with its 86/2744 magnitude.
5. **F9 + F10 — sweep the retired levels**: mark or delete the `-0.33 / -1.29 / -2.50` and the
   "4 of 4 Monza occupants" prose in `_render_table`, and relabel the log's comparison column so
   it does not claim to be `decision_modes.md`.
6. **F6 + F15 — put both on #826**, which is correctly left open. The re-chunking target should
   be measured as "does the block answer the pit question and carry a correct article label",
   not "is the bad chunk gone".
7. **F7 — fix the worked example** so it carries a condition, or drop the two-compound sentence.
8. **F8 — bound the false-inconclusive rate** before the LLM arm is measured again.
9. **F11, F12, F13** — small, no urgency.

---

## What I tried to break and could NOT

Stated so the rest of this report is worth its length.

- **The headline levels.** Re-ran the whole tier. 178 of 178 verdicts identical to the committed
  JSON, all nine aggregate fields identical. Not approximately — exactly.
- **The `gap_ahead_s` = 2.0 claim.** I expected an overstatement. It is literal: over all 2744
  eligible laps of the six races, the old value's `value_counts()` is `{2.0: 2744}`. One value,
  every lap.
- **`pace_delta_s` = 0.0.** Confirmed; differs from the builder on 90.0% of laps.
- **Seven of the ten `RaceState` fields.** `lap`, `total_laps`, `position`, `compound`,
  `tyre_life`, `air_temp`, `track_temp` differ on **zero** of 2744 laps. The compound and
  tyre-life defaults the old code guarded never fired on this population, so the docstring
  paragraphs about them are inert rather than wrong.
- **The 0.0 s gap for a leader.** `_gap_to_car_ahead` returns 0.0 when no car sits one position
  ahead, and I expected the classic sentinel collision — a car with rivals ahead getting 0.0
  because that specific rival was missing from the lap. Measured: 199 laps take that branch and
  **all 199 are `position == 1`**. Zero fabricated zeros.
- **#826's root cause, in full.** Every part checks out at the Qdrant chunk boundaries. Chunk
  `id=1699` (labelled `Article 49.1`) ends: *"n) If the formation lap is started behind the
  safety car in accordance with Article 49.1a, or the … race is resumed in accordance with
  Article 58.1, the use of wetweather tyres until the safety car orange lights are extinguished
  and"*. Chunk `id=1700` (labelled `Article 54.3`) begins mid-word: *"er tyres until the safety
  car orange lights are extinguished and it returns to the pit lane is compulsory…"*. The
  applicability clause is in the previous chunk; the retrieved chunk is Art. **30.5 n)** text
  (the item letter is right there in 1699); and its `Article 54.3` label is the only article
  number in its own body, a cross-reference. Exactly as claimed.
- **The retrieval deltas.** Old question → the bad chunk at rank 1, score 0.6604 (PR says 0.660).
  New question → top score 0.7503 (PR says 0.750), and the bad chunk is not merely outside the
  top 5, it is **outside the top 20**. Understated, not overstated.
- **Which OpenF1 session belongs to which circuit.** Queried the API directly. Italy 2025 returns
  Imola (9987) **first** and Monza (9912) second; the US returns Miami (10033), Austin (9888),
  Las Vegas (9858) after the `session_name == "Race"` filter. `[0]` really was Imola and Miami.
- **The "failed on three of five" claim.** Executed against the Hub parquets: `italy_monza`
  63≠53, `united_states_austin` 57≠56, `united_states_las_vegas` 57≠50 — exactly three of five,
  with Imola and Miami passing. Correct, and it is still the present tense off this machine.
- **The two Safety Car lap numbers.** I suspected the docstring's "laps 29 and 30" contradicted
  the PR body's "laps 29 and 46". They have different subjects — the laps Monza was *served* vs
  Imola's own deployments — and both are right. Only the VSC-vs-SC naming (F13) is loose.
- **The keyword-only signatures.** All five changed methods take `circuit_short_name` after `*`,
  so no positional caller can break. No notebook, test or arcade path calls them.
- **`harness_sha`.** I expected the report's stamp not to match the tree that produced it, since
  a previous gate found exactly that at `06a8f32`. `decision_modes.py` changed only in `a22c97d`,
  and `ea9c319` already contains the new `_render_table`. The stamp is honest.
- **The spend guard.** Exercised `_confirm_spend` across six argument combinations: it proceeds
  only for `--no-llm`, `lmstudio`, or an explicit `--yes-spend`, and refuses every paid
  combination including unknown providers. Fails safe. The `$0.0080`/lap constant matches the
  log's own 75-lap measurement (the `$0.0071` elsewhere is the superseded 3-lap probe).
- **`_run_conditional_agents`' dedent.** Moving `regulation_context` / `rag_dict` out of the
  `if "N30" in active` block and under `if reg_out is not None` is behaviour-preserving.
- **Any other library raising a portalocker exception.** Swept `.venv`: only
  `qdrant_client/local/{qdrant_local,async_qdrant_local}.py` import portalocker at all, and both
  convert to `RuntimeError`. So F1's `except` is unreachable code, not merely unlikely.
- **The touched tests.** `tests/agents/test_rag_degrades_when_locked.py` +
  `tests/audit/test_radio_corpus_is_its_own_race.py`: 14 passed.
  `tests/eval/test_decision_modes.py` + `test_llm_measurement.py`: no failures observed. They
  pass — F2 is about what they assert, not about them being red.

---

## Method note

Run from an isolated `git worktree` at `28eeaa7` with `data/` junctioned to the real tree, so the
user's working tree (on `chore/input-wiring-hygiene`, PR #831's branch, with another gate active)
was never touched. The re-run overwrote `documents/eval_reports/decision_modes.{json,md}`
**inside that scratch worktree only**; it reproduced the committed bytes, and the worktree is
disposable. No OpenAI call was made at any point: retrieval was `get_retriever().query(...)`
(embeddings only), the tier ran `profile="no-llm"`, and the lock experiments used a temporary
Qdrant directory rather than `data/rag/qdrant_local`.
