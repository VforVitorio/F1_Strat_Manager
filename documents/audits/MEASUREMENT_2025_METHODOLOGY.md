# Measurement methodology — LLM mode (`profile="rich"`), 2025 season

**Status: COMPLETE (2026-08-06). All eight sections present.**

Author: methodology-design agent, 2026-08-06. Branch `dev`.
Scope: design only. Nothing here was executed; no LLM calls were spent. All lap counts, windows and costs are computed from the actual 2025 raw data and the actual seeded draw, not estimated.

## Contents

1. The sample — races, drivers, lap windows, and why each window over its neighbours
2. Stratification rationale — the 9/12/1/2 cluster imbalance and what the thesis races cover
3. The metrics — definition, population, what each cannot be read as, comparability to no-llm numbers
4. LLM-specific problems — non-determinism, repeats vs single pass, budget impact
5. The press-contrast axis — designed so it can never be read as accuracy
6. Australia — decided, not deferred
7. The cut list — ranked, with the claim-strength cost of each cut
8. What I could not resolve — the forks Víctor has to pick personally

---

## 0. Design in one paragraph, and the budget

Three tiers plus one free verification pass. **Tier A** is the population axis: 91 stop-anchored windows drawn by a **seeded random draw** (seed `20250806`) from the rail-eligible green-flag stops of 9 races chosen to cover all four circuit clusters, scored with the *same* transition metric and buckets as `decision_modes.py`. **Tier B** is the case-study axis: the two quantitative thesis windows (Budapest V17-V19, Qatar V7), run as named case studies and a regression test, never folded into any rate. **Tier C** is the stability axis: a full second LLM pass over two races' Tier-A windows, because the LLM path is measurably non-deterministic and every other number needs an error bar. Before any LLM lap is paid for, the **entire Tier A+B sample runs once in `no-llm` mode** (~10 min, zero tokens): that pass is simultaneously the harness/keyspace check and the paired deterministic baseline, so the LLM tier is compared against `no-llm` **on identical windows** instead of against the published `decision_modes.md` numbers (whose sample is different).

| Component | Windows | Evaluated laps | Wall clock @15.93 s/lap | Tokens @8,894/lap |
|---|---|---|---|---|
| Tier A (population) | 91 | 1,090 | 4.82 h | 9.69 M |
| Tier B (case studies) | 3 | 32 | 8.5 min | 0.28 M |
| Tier C (stability re-pass) | 22 (repeats) | 264 | 1.17 h | 2.35 M |
| **Total LLM** | **94 (+22 repeats)** | **1,386** | **6.13 h** | **12.33 M** (~11.20 M prompt / 1.13 M completion) |
| no-llm paired preflight | 94 | 1,122 | ~9.5 min @0.51 s/lap | 0 |

Plus ~13 process boots (one per race per pass) x ~25 s = 5.5 min. Zero prompt tokens are cacheable (measured: `cached_prompt_tokens: 0`; the prompt drifts numerically every lap), so the token bill is paid in full. The total sits at **92% of the stated 1,000-1,500-lap ceiling**; section 7 is the pressure valve, ranked. Runs can be split across 2-3 concurrent processes to roughly halve wall clock; each extra process costs one 25 s boot and shares the same token bill.

Two structural decisions made here rather than inherited:

- **Windows are anchored at real stops, +/-5 laps plus one predecessor lap** (`[stop-6, stop+5]`, clipped to race length, merged per driver when overlapping). +/-5 is `DECISION_WINDOW_LAPS`, the Monte Carlo's own decision horizon, so the question posed is the one the system was built to answer. The predecessor lap exists so the first lap of the scoring window can witness a transition (the `_replay_span` lesson from #752). Laps *between* a driver's two stops are **not** evaluated, unlike the no-llm tier's contiguous span: per-stop scoring only ever reads laps inside `[stop-5, stop+5]` plus each lap's predecessor, so the verdicts are identical by construction and the laps the contiguous span spends between windows are not bought at 15.93 s each.
- **The sample is drawn, not chosen.** Newsworthiness selection is the project's own named trap; a seeded uniform draw within each race is the strongest defence, and it still yields a fully named table (below). The press enters only in Tier B and in *post-run* annotation of disagreements (section 5).

---

## 1. The sample

### 1.1 Selection protocol (the rules that generated the tables)

1. **Sample frame:** `green_flag_stops()` from `src/strategy/eval/projection.py` — the shared sample definition both existing tiers already grade — over `data/raw/2025/<race>/laps.parquet`, for the 9 races in 1.2.
2. **Eligibility:** stops where `guard_rail_block(...)` returns `None` (the rails do not make agreement impossible). Railed stops are ~5% of green stops by the calibration ceiling (13 of 265 across the 9 races); spending 15.93 s/lap on windows whose headline outcome is fixed in advance buys nothing. Consequence for the population statement: every Tier-A rate describes **rail-eligible green-flag stops**, and 8.2 records the alternative reading.
3. **The draw:** per race, a seeded permutation (`numpy.default_rng(20250806)`, races processed in the 1.2 order) of the eligible stops, accepted in order until the race's allocation is filled, **skipping any stop whose lap already has 3 accepted windows in that race** (`MAX_SAME_LAP = 3`). The cap exists because of Lusail: the event's maximum-stint directive put half the field into a stop at exactly tyre-life 25 (lap 32), and an uncapped draw spent 9 of 11 Lusail windows re-measuring one regulatory event. The cap is part of the population statement, not a tweak: *at most 3 windows per (race, stop-lap)*.
4. **Windows:** `[max(1, stop-6), min(total_laps, stop+5)]`, merged per driver. Replay via `RaceReplayEngine(data/raw/2025/<folder>, driver, team, interval_seconds=0)`; agents fed the GP-scoped `laps_featured_2025` frame through `run_lap(..., profile="rich", return_agent_outputs=True)`. All five signature traps in `reference_drive_orchestrator_offline` apply verbatim (hyphenated profile, raw-vs-featured split, `augment_featured_laps(df, year)`, mandatory temps, `_out`-suffixed keys).
5. **Keyspace check (mandatory preflight):** all 9 raw folder names verified today to resolve through `resolve_gp_key` into the featured keyspace (`Mexico_City -> 'Mexico City'`, `Montréal -> 'Montréal'`). The preflight must still assert per race that the scoped frame satisfies `len(set(scoped.GP_Name)) == 1`, because `_scope_laps_to_gp`'s failure mode is returning the FULL season frame with only a warning — the exact silent wrong-race trap this project has already paid for once.

### 1.2 Tier A — 91 windows, by race (raw-folder keyspace)

Per-row reasons: the window's position is structural (anchored at the real stop; the "neighbour" laps ARE the window), so the per-row column gives the strategic context that makes the row informative and flags the named sub-regimes. `life` = tyre life at the stop; `P` = position on the stop lap. All rows below are the actual output of the seeded draw, not an illustration.

**Barcelona — cluster 1, conventional/high-deg, 16 windows, 191 laps.** The largest allocation: most green stops of any 2025 race (41), archetypal two-stop strategy race, and shared with the no-llm subset (harness cross-check V0, section 3).

| Driver (team) | Stop | Window | Context / why informative |
|---|---|---|---|
| ALB (Williams) L6 | 1/3 | 1-11 | Earliest drawn stop of the sample (SOFT off at life 9); probes the early-stop regime just above the opening rail |
| ALB (Williams) L27 | 3/3 | 21-32 | Three-stopper's last stop, backmarker P19 |
| ALO (Aston Martin) L42 | 2/2 | 36-47 | Midfield second stop, MEDIUM life 28 |
| ANT (Mercedes) L21 | 1/2 | 15-26 | Front-runner P4 first stop in the main undercut phase |
| ANT (Mercedes) L49 | 2/2 | 43-54 | Window overlaps the L54-60 SC: post-stop laps land against the neutralisation edge; deliberate edge probe |
| BOR (Kick Sauber) L19 | 1/2 | 13-24 | Midfield first stop |
| BOR (Kick Sauber) L49 | 2/2 | 43-54 | Same SC-edge shape as ANT L49 |
| COL (Alpine) L14 | 1/2 | 8-19 | Early first stop, life 14 |
| HAM (Ferrari) L16 | 1/2 | 10-21 | Front-runner P5, SOFT life 19 |
| LAW (Racing Bulls) L44 | 2/2 | 38-49 | Midfield second stop |
| NOR (McLaren) L21 | 1/2 | 15-26 | P2 covering the leader — highest-stakes drawn Barcelona window |
| PIA (McLaren) L49 | 2/2 | 43-54 | Race leader's final stop |
| SAI (Williams) L9 | 1/2 | 3-14 | Early stop at life 9 |
| SAI (Williams) L34 | 2/2 | 28-39 | SOFT-to-SOFT second stop |
| TSU (Red Bull Racing) L8 | 1/3 | 2-13 | MEDIUM off at life 8 — one lap above the recalibrated MEDIUM min-stint (7); rail-boundary probe |
| TSU (Red Bull Racing) L24 | 2/3 | 18-29 | Three-stopper mid-race |

**Budapest — cluster 1, thesis race 5.5.2, 14 windows, 168 laps.** Zero neutralised laps (a clean strategic race), and the seeded draw landed on both strategically famous stops on its own: LEC L19 (the thesis window) and NOR L31 (the winning one-stop).

| Driver (team) | Stop | Window | Context / why informative |
|---|---|---|---|
| ALB (Williams) L14 | 1/2 | 8-19 | Early SOFT stop |
| ALB (Williams) L38 | 2/2 | 32-43 | Two-stop second leg |
| ALO (Aston Martin) L39 | 1/1 | 33-44 | One-stopper, MEDIUM stretched to life 40 |
| BEA (Haas F1 Team) L30 | 1/2 | 24-35 | Midfield first stop |
| BOR (Kick Sauber) L40 | 1/1 | 34-45 | One-stopper P5 at the stop |
| COL (Alpine) L13 | 1/2 | 7-18 | Early two-stopper |
| COL (Alpine) L35 | 2/2 | 29-40 | HARD off at life 22 |
| GAS (Alpine) L32 | 1/1 | 26-37 | One-stopper from HARD |
| HAM (Ferrari) L42 | 1/1 | 36-47 | One-stopper, HARD to life 42 |
| **LEC (Ferrari) L19** | 1/2 | **13-24** | **The thesis window (5.5.2): Ferrari covers Piastri's V18 undercut on the HARD, the compound Leclerc criticised.** Drawn by seed; doubles as the population-side anchor of the Tier-B contrast |
| **NOR (McLaren) L31** | 1/1 | 25-36 | **The stop that won the race** — the overcut the real wall got right; tests whether the system can find the non-obvious long first stint |
| RUS (Mercedes) L19 | 1/2 | 13-24 | Same-lap cover of LEC's stop from P3 — paired decision, different car state |
| RUS (Mercedes) L43 | 2/2 | 37-48 | Second stop onto HARD |
| SAI (Williams) L51 | 2/2 | 45-56 | Latest drawn Budapest stop |

**Monza — cluster 1, low-downforce/fewest-stops, 13 windows, 155 laps.** The one-stop regime where the no-llm tier's `no_boundary` shapes were originally measured; keeping it makes the LLM/no-llm bucket contrast interpretable against the published Monza anatomy.

| Driver (team) | Stop | Window | Context / why informative |
|---|---|---|---|
| ALB (Williams) L41 | 1/1 | 35-46 | Long HARD first stint (life 41) |
| ALO (Aston Martin) L20 | 1/2 | 14-25 | The rare Monza two-stopper |
| ANT (Mercedes) L28 | 1/1 | 22-33 | Mid-pack one-stop |
| BOR (Kick Sauber) L20 | 1/1 | 14-25 | Early one-stop |
| COL (Alpine) L33 | 1/1 | 27-38 | Standard one-stop |
| HAD (Racing Bulls) L32 | 1/1 | 26-37 | HARD-first inversion |
| LEC (Ferrari) L33 | 1/1 | 27-38 | Podium fight P4 |
| NOR (McLaren) L46 | 1/1 | 40-51 | Leader's very late stop (life 46) — the classic Monza track-position hold |
| PIA (McLaren) L45 | 1/1 | 39-50 | P2 mirror of NOR's hold, one lap earlier |
| SAI (Williams) L30 | 1/1 | 24-35 | Midfield reference |
| STR (Aston Martin) L49 | 1/1 | 43-53 | Latest stop in the whole sample, 4 laps from the flag; verified rail-eligible |
| TSU (Red Bull Racing) L19 | 1/1 | 13-24 | Earliest Monza one-stop |
| VER (Red Bull Racing) L37 | 1/1 | 31-42 | Race winner's stop from P1 |

**Lusail — cluster 0, thesis race 5.5.3, 5 windows, 60 laps.** Deliberately small: the maximum-stint directive (field-wide stops at exactly tyre-life 25 — the data signature is unambiguous; verify the TD text in the press step before quoting it as regulation) makes most Lusail "decisions" regulation-forced. Three capped L32 windows measure whether the system reproduces a forced stop **whose cause is outside its observables** (the event TD is not in the RAG corpus, which holds the season rulebooks); they are reported as a named sub-regime line, never silently inside the headline.

| Driver (team) | Stop | Window | Context / why informative |
|---|---|---|---|
| BEA (Haas F1 Team) L40 | 2/3 | 34-45 | A genuinely chosen stop (HARD off at life 8) |
| COL (Alpine) L32 | 1/1 | 26-37 | TD-forced regime (life exactly 25) — capped trio 1/3 |
| HAD (Racing Bulls) L32 | 1/2 | 26-37 | TD-forced — capped trio 2/3, from P7 |
| HAM (Ferrari) L32 | 1/1 | 26-37 | TD-forced — capped trio 3/3 |
| OCO (Haas F1 Team) L34 | 1/1 | 28-39 | Life-25 stop displaced to L34 (set fitted under the L7-10 SC) |

**Silverstone — cluster 0, wet, 11 windows, 132 laps.** Every drawn stop bar one is on INTERMEDIATE: this race is the sample's wet regime, and it is also where the one **documented rich/no-llm divergence by design** lives — `_DEFAULT_MIN_STINT` (INTERMEDIATE/WET minimum 6) exists only on the offline path; the prompts never state it (`guard_rails.py`, "KNOWN DIVERGENCE"). Silverstone windows are the place that divergence can actually show. Reported in the headline AND as a separate wet-regime line (8.6 records the exclusion reading).

| Driver (team) | Stop | Window | Context / why informative |
|---|---|---|---|
| ALB (Williams) L42 | 1/1 | 36-47 | Inter-to-slick crossover phase |
| HAM (Ferrari) L11 | 1/2 | 5-16 | Wet first stop between the two neutralisation blocks (laps 1-7, 13-21) |
| HUL (Kick Sauber) L42 | 2/2 | 36-47 | The podium run — P3 at the stop |
| PIA (McLaren) L43 | 2/2 | 37-48 | Race leader's crossover call |
| RUS (Mercedes) L10 | 1/2 | 4-15 | The one slick-off stop (HARD at life 10) — the failed-gamble shape |
| RUS (Mercedes) L38 | 2/2 | 32-43 | Earliest crossover of the drawn set |
| SAI (Williams) L11 | 1/2 | 5-16 | Wet first stop |
| SAI (Williams) L41 | 2/2 | 35-46 | Crossover |
| TSU (Red Bull Racing) L41 | 2/2 | 35-46 | Crossover, backmarker |
| VER (Red Bull Racing) L11 | 1/2 | 5-16 | Front-runner wet stop |
| VER (Red Bull Racing) L41 | 2/2 | 35-46 | Front-runner crossover |

**Suzuka — cluster 0, clean one-stop classic, 10 windows, 120 laps.** Zero neutralised laps; the STAY_OUT-heavy regime where the no-llm layer declines most — the cleanest place to see whether the LLM tier declines the same way (M4).

| Driver (team) | Stop | Window | Context / why informative |
|---|---|---|---|
| ANT (Mercedes) L31 | 1/1 | 25-36 | Leading at the stop |
| BEA (Haas F1 Team) L23 | 1/1 | 17-28 | Midfield one-stop |
| BOR (Kick Sauber) L31 | 1/1 | 25-36 | HARD one-stop |
| DOO (Alpine) L15 | 1/1 | 9-20 | Earliest Suzuka stop (SOFT life 15, P19) |
| GAS (Alpine) L24 | 1/1 | 18-29 | Midfield reference |
| HAM (Ferrari) L30 | 1/1 | 24-35 | HARD-first inversion |
| NOR (McLaren) L21 | 1/1 | 15-26 | P2 chasing — undercut-pressure shape |
| SAI (Williams) L33 | 1/1 | 27-38 | Late one-stop |
| STR (Aston Martin) L30 | 2/2 | 24-35 | The rare Suzuka two-stopper's second leg |
| VER (Red Bull Racing) L21 | 1/1 | 15-26 | Race winner from P1, same lap as NOR — paired leader/chaser decision |

**Montréal — cluster 2 (the ONLY member), 7 windows, 84 laps.** Without this race, cluster 2 has zero coverage anywhere in the project's measured story.

| Driver (team) | Stop | Window | Context / why informative |
|---|---|---|---|
| ANT (Mercedes) L14 | 1/2 | 8-19 | The eventual winner's first stop |
| COL (Alpine) L14 | 1/1 | 8-19 | Same-lap one-stop commitment from P19 |
| GAS (Alpine) L53 | 1/1 | 47-58 | Extreme stint stretch (HARD life 53) |
| HUL (Kick Sauber) L19 | 1/1 | 13-24 | Early one-stop |
| LAW (Racing Bulls) L38 | 1/2 | 32-43 | Mid-race HARD stop |
| LEC (Ferrari) L53 | 2/2 | 47-58 | Late second stop, P6 |
| SAI (Williams) L57 | 1/1 | 51-62 | Latest first stop in the sample (life 57) |

**Monaco — cluster 3, street/track-position, 8 windows, 96 laps.** The 2025 mandatory-two-stop event: stops are partly regulation-forced, but unlike Lusail's TD the rule IS in the 2025 sporting regulations, so the RAG corpus can in principle see it — a genuinely interesting asymmetry between the two forced regimes. Shared with the no-llm subset (V0 cross-check).

| Driver (team) | Stop | Window | Context / why informative |
|---|---|---|---|
| ALB (Williams) L40 | 2/2 | 34-45 | Second mandated stop, mid-race |
| GAS (Alpine) L8 | 1/1 | 2-13 | Earliest Monaco stop (MEDIUM at life 7, exactly the recalibrated MEDIUM minimum) |
| HAD (Racing Bulls) L14 | 1/2 | 8-19 | Early first mandated stop |
| HUL (Kick Sauber) L12 | 1/2 | 6-17 | Early stop from P19 |
| LEC (Ferrari) L22 | 1/2 | 16-27 | Podium contender's first stop |
| RUS (Mercedes) L62 | 2/3 | 56-67 | The strategic-games phase (Mercedes' late double-stop) |
| SAI (Williams) L48 | 1/2 | 42-53 | Extremely delayed first stop (HARD life 48) — the hold-position-at-all-costs shape |
| TSU (Red Bull Racing) L73 | 1/1 | 67-78 | Latest green stop in the sample: rule-compliance stop 5 laps from the flag |

**Mexico_City — cluster 3, 7 windows, 84 laps.** Featured keyspace `Mexico City` (verified resolving). Covers the altitude outlier so cluster 3 is not represented by Monaco alone.

| Driver (team) | Stop | Window | Context / why informative |
|---|---|---|---|
| ALO (Aston Martin) L20 | 1/2 | 14-25 | SOFT first stop |
| BEA (Haas F1 Team) L48 | 2/2 | 42-53 | The podium run (P3 at the stop) |
| GAS (Alpine) L33 | 1/1 | 27-38 | One-stop reference |
| LEC (Ferrari) L31 | 1/1 | 25-36 | Front-runner P2, SOFT stretched to 31 |
| RUS (Mercedes) L48 | 2/2 | 42-53 | Late second stop |
| STR (Aston Martin) L26 | 1/2 | 20-31 | Midfield SOFT stop |
| VER (Red Bull Racing) L37 | 1/1 | 31-42 | Front-runner one-stop |

### 1.3 Tier B — the case-study windows (3 windows, 32 laps)

| Race (folder) | Driver | Laps | What it is, and the reason this window |
|---|---|---|---|
| Budapest | PIA (McLaren) | 12-23 | **Thesis 5.5.2, the attacking half:** Piastri initiates the undercut at V18. Together with the Tier-A LEC 13-24 window (drawn by seed) both sides of the V17-V19 exchange are measured. The thesis intro's "parada de Leclerc en V20" is wrong; the body's V17-V19 is right (verified: VER L17, PIA L18, LEC L19, RUS L19, NOR L31) |
| Lusail | PIA (McLaren) | 3-12 | **Thesis 5.5.3, regression test — NOT an open question.** SC deployed laps 7-10 (verified from TrackStatus). Asserts the shipped cross-integration module: on the RCM-confirmed deployment lap, SC probability forced to 1.0 (`race_situation_agent.py` ~1833), min-stint rail suspended (`pit_strategy_agent.py:1444`, `:1530`), and the recommendation on L7-L8 is a pit-class action for a car that really stayed out. Pre-SC laps 3-6 give the memory a warm-up and show the *before* state |
| Lusail | VER (Red Bull Racing) | 3-12 | The control arm: Verstappen's wall DID pit under the SC (his L32 green stop at life 25 confirms the set was fitted at L7). Same laps, opposite real-wall action — the pair makes the contrast readable without any press sentence |

Tier B verdicts are **named case narratives plus binary regression assertions**. They never enter M1-M4. The Qatar assertions are pass/fail against the shipped cross-integration module; a fail is a regression, not a finding about strategy quality.

### 1.4 Tier C — the stability re-pass (22 windows, 264 laps)

A full second `rich` pass, identical inputs, over **all Tier-A windows of Budapest (14) and Monaco (8)**. Two whole races rather than a scatter of windows, so within-race conditions are held fixed and the two most different regimes (clean strategic dry race; forced-two-stop street race) each get a stability estimate. Yields ~264 per-lap action pairs and 22 per-window verdict pairs. Why it must exist, and what it feeds, is 4.1.

### 1.5 What one evaluated lap is (so the cost arithmetic is checkable)

One `run_lap(..., profile="rich")` = 6 LLM calls (5x `gpt-4.1-mini` sub-agents + 1x `gpt-5.4-mini` orchestrator), 8,080 prompt + 814 completion tokens, 15.93 s mean / 17.38 s max, measured on this machine on the shipped CLI path (`documents/audits/PROBE_llm_cost.json`). All lap counts above are evaluated laps (window laps + the predecessor lap, clipped, merged per driver), computed from the actual draw — not estimates.

---

## 2. Stratification rationale

### 2.1 The design is two-stage, and the two stages have different natures

- **Stage 1 (races) is purposive and stated, not random.** 9 of 24 races, chosen for: thesis anchors (Budapest, Lusail), overlap with the no-llm `SAMPLED_RACES` subset so the V0 cross-check exists (Barcelona, Monaco, Silverstone, Lusail, Monza — 5 of its 6; only Marina_Bay is dropped), regime coverage the subset lacked (Suzuka: clean one-stop; Montréal: sole cluster-2 member; Mexico_City: the altitude outlier), and cluster completeness. A random stage 1 at n=9 would have been theatre: with 4 clusters of sizes 9/12/1/2, most random draws either miss a cluster or land on races with no anchoring value.
- **Stage 2 (stops within a race) is seeded random.** This is where newsworthiness selection would otherwise creep in, so this is where randomness is spent.

Every population statement is therefore conditional on stage 1: "rail-eligible green-flag stops **in these 9 races**". The reweighted secondary line (2.3) extrapolates beyond that only under the stated assumption that each sampled race represents its cluster.

### 2.2 What was done about 9/12/1/2

"One race per cluster" was rejected: it hands Montréal (1 race, 30 green stops) the same weight as 12 races and 274 green stops of cluster 1. Instead, window allocation tracks each cluster's share of the season's **green stops** (not its race count — stops are the unit the metric grades), with deliberate mild oversampling of the two tiny clusters so their estimates exist at all:

| Cluster | Races (season) | Green stops (share) | Races sampled | Windows (share) | Over/under |
|---|---|---|---|---|---|
| 0 | 9 | 202 (35.3%) | Lusail, Silverstone, Suzuka | 26 (28.6%) | under: the TD-forced Lusail cut (1.1 step 3) came out of this cluster |
| 1 | 12 | 274 (47.8%) | Barcelona, Budapest, Monza | 43 (47.3%) | proportional |
| 2 | 1 | 30 (5.2%) | Montréal | 7 (7.7%) | oversampled x1.5 |
| 3 | 2 | 67 (11.7%) | Monaco, Mexico_City | 15 (16.5%) | oversampled x1.4 |
| total | 24 | 573 | 9 | 91 | |

### 2.3 What the weights mean, exactly

Two aggregate lines are reported, in this order:

1. **Primary (unweighted):** rate over the 91 sampled windows as drawn. Population: rail-eligible green-flag stops in the 9 sampled races, under the max-3-per-lap cap. No extrapolation claim.
2. **Secondary (cluster-reweighted):** each window weighted by `w_c = (cluster's share of season green stops) / (cluster's share of sampled windows)` — w0=1.23, w1=1.01, w2=0.68, w3=0.71. This line answers "what would the rate look like if the sample had followed the season's cluster mix", **under the assumption that sampled races stand for their clusters**. That assumption is strong for cluster 2 (Montréal IS the cluster), exact by construction; weakest for cluster 0, where 3 sampled races stand for 9 and one of the three is a wet race. The weighted line must never be quoted without this sentence attached.

### 2.4 What the thesis races do and do not cover

The three thesis races cover clusters {0, 1} only, and only one of the three is a quantitative stop case at all (Budapest); Qatar is a neutralised SC event outside the green-flag population by construction, and Australia is a qualitative chat validation (section 6). This design keeps the two quantitative thesis anchors exactly where they are strongest — Budapest inside the population axis (its famous windows were drawn by seed anyway), Qatar as a regression test — and adds clusters 2 and 3 through Tier A, which the thesis never touched.

---

## 3. The metrics

Format per metric: definition -> the exact population it describes -> what it cannot be read as -> comparability to existing numbers.

**V0 — harness identity check (runs before anything is paid for).** The no-llm preflight re-runs every Tier A+B window with `profile="no-llm"`. For the ~5 races shared with `decision_modes.md`, every stop that appears in both samples must reproduce its published per-stop verdict (bucket and, when scored, chosen lap) from `documents/eval_reports/decision_modes.json` **exactly** — the path is deterministic, so any mismatch is harness or config drift, and the LLM run does not start until it is explained. Population: shared stops. This is a gate, not a result; it also proves the keyspace assertions of 1.1(5) on the two renamed folders.

**M1 — decision agreement (exact / within-1 / within-2 / mean signed error).** Definitions imported verbatim from `decision_modes.py`: the chosen lap is the first STAY_OUT->pit transition inside `[stop-5, stop+5]` whose predecessor lap was evaluated and non-pit and whose lap is > `_NO_PIT_BEFORE_LAP`; error = chosen - actual, signed. Population: the **scored subset** of the 91 Tier-A windows (the LLM analogue of "67 of 178"). Cannot be read as: correctness (the team can be wrong); a counterfactual ("stopping earlier would have gained"); a full-season figure (9 races, conditional on stage 1); or a deterministic property of the system (single pass of a non-deterministic path — every M1 number carries the Tier-C stability band, 4.1). Comparability: **not** comparable to `decision_modes.md`'s 31.3/47.8/61.2 — different races, different stops, different draw protocol. The comparable object is M3.

**M2 — bucket distribution, scored share, coverage verdict.** Same buckets (`scored`, `no_call_in_window`, `no_boundary_in_window`, `no_data`; the guard-rail buckets are empty by eligibility filtering), same `MIN_SCORED_SHARE = 0.60` verdict. Population: all 91 windows. Cannot be read as: a quality score — a `masked` verdict means the headline is drawn from a minority of stops, which is a *finding about the system's willingness to call stops*, not a measurement failure. Comparability: the shape (not the numbers) is comparable to the published bucket anatomy; the numbers only through M3.

**M3 — paired tier contrast (the one genuinely comparable number).** For each of the 91 windows, the (LLM verdict, no-llm verdict) pair from identical inputs: a 2x2-and-beyond contingency of bucket agreement, and on windows both tiers score, the distribution of `chosen_llm - chosen_no_llm`. Population: the 91 windows, paired by construction. This is the ONLY place a sentence like "the LLM tier calls stops the deterministic tier declines" (or the reverse) may be written, because it is the only place the samples are genuinely the same rather than merely looking it. Cannot be read as: either tier being right — it measures where the two layers part company, not who is correct. Note the mechanism asymmetry it carries: the no-llm action passes `apply_guard_rails` post-hoc; the rich action is railed only by prompt prose. M3 therefore measures the shipped products as shipped, not two implementations of one policy.

**M4 — decline rate.** Share of the 91 windows in `no_call_in_window`: the tier looked and never asked to stop. Population: sampled rail-eligible green-flag windows in the 9 races. Cannot be read as: the published 43.8% (78/178) — that figure lives on the no-llm tier over a different sample; the same-sample paired figure comes free from the preflight and THAT is the printed comparison. Whether the LLM's decline rate is higher or lower than the deterministic layer's on identical windows is arguably the single most product-relevant unknown in this measurement.

**M5 — prompt-rail compliance (LLM-only, free from the same runs).** For every evaluated rich lap (~1,090), post-hoc: would `apply_guard_rails(action, lap, total_laps, compound, tyre_life)` have overridden the emitted action, and which rail. Population: evaluated laps in Tier-A windows (clustered — laps within a window are not independent; report per-window too). Cannot be read as: a defect rate of the rails (in rich mode the rails are prose in the N28 prompt, not code — a "violation" is the LLM not honouring its own instructions, or exercising the INTERMEDIATE/WET bound it was never told about); nor as a strategy-quality measure. Comparability: no no-llm analogue exists (0 by construction there). Silverstone rows are the interesting stratum (the `_DEFAULT_MIN_STINT` divergence).

**M6 — recommendation-integrity panel (the 12 LLM-originated fields, free from the same runs).** Per evaluated lap: (a) `pit_lap_target` plausibility (`> lap`, `<= total_laps`) and lap-to-lap churn within a window (the #646 churn measurement, now in the mode that ships); (b) `undercut_target` membership in live rivals — expected 0 violations, a regression check on #462's guard; (c) `target_lap_time_s` sitting at a CI bound (clamp-hit rate for #433's guard); (d) action distribution including ALERT (an action the MC layer never scores); (e) `contingencies` count distribution; (f) `confidence` distribution. Population: evaluated laps, clustered as in M5. Cannot be read as: field *accuracy* — nothing here has ground truth; these are consistency and guard-exercise measures. Comparability: none exists and none is claimed; this is the first measurement of these fields in the shipping mode, which is the stated purpose of the whole exercise.

**M7 — stability (Tier C).** Between the two passes: per-lap action discordance rate (n≈264 pairs), per-window verdict discordance (bucket changes or chosen-lap changes, n=22), |delta confidence| and |delta pit_lap_target| distributions. Population: Budapest + Monaco Tier-A windows, i.e. clusters {1,3} — the band is extrapolated to the other clusters as an assumption, stated where used. Cannot be read as: sampling error (it is *path* noise at fixed inputs, a different thing from the draw's binomial width); nor as temperature (temperature is requested at 0.0 and discarded by `gpt-5.4-mini`). Comparability: the prior 41-lap Lusail measurement (36/41 confidence, 23/41 pit target, 1/41 action) was no-llm-prompt-era and single-race; M7 supersedes it for the shipping mode.

**M8 — case-study verdicts (Tier B).** Budapest: does the system, sitting where Ferrari sat on V17-V19 (LEC window) and where McLaren sat on V13-V23 (PIA window), diverge from the real calls, and in which direction — written as a narrative with the per-lap action/target/reasoning fields quoted. Qatar: the four regression assertions of 1.3. Population: n=1 events. Cannot be read as: rates, accuracy, or evidence the system is better or worse than the real wall (section 5 fixes the exact allowed sentence). Comparability: to the thesis's own published diagnosis of the same windows — that is the point of keeping them.

**Statistical honesty for all of M1-M4:** 91 windows clustered in 9 races and 44 driver-spans; laps within a window are serially dependent. Every rate is reported with a binomial CI computed at the WINDOW level (n = windows in the denominator, not laps), and the report must print the CI next to the rate, because at the plausible scored counts (30-60 of 91, if the LLM tier's scored share lands anywhere near the no-llm tier's 37.6%) a within-1 rate carries a +/-12-17 pp Wilson interval. A number that wide is still worth having — it is the first of its kind — but it must be typeset with its width, never naked.

---

## 4. The LLM-specific problems the no-llm tier never had

### 4.1 Non-determinism: single seeded pass rejected, single pass + measured spread chosen

The path is not a function: temperature is requested at 0.0, `gpt-5.4-mini` discards it, and the measured 41-lap run had two identical passes disagree on `confidence` in 36 laps, `pit_lap_target` in 23, and produce opposite actions on 1. A "single seeded pass" is therefore **not on offer as a determinism guarantee** — no seed parameter is known to be honoured on this path, and pretending one run is "the" answer would print noise as signal.

Decision: **one pass for the full sample, plus Tier C's full second pass over 22 windows (24% of Tier A), reported as a spread.** Rationale against the alternatives:

- *n repeats over everything* at n=2 doubles the bill to ~2.5k laps (~11 h) or forces the window count down to ~45, and 45 windows cannot carry cluster strata at all. Breadth buys strata; repeats only narrow a band whose width Tier C measures anyway.
- *Repeats only where passes disagree* is data-dependent stopping — it biases toward instability exactly where the estimate is used.
- The per-lap action flip rate from the prior measurement (~2.4%) compounds over an 11-lap window to a non-trivial probability that a window's *verdict* differs between passes; the honest response is to measure the verdict-level discordance directly (M7, n=22) and attach it to M1-M4 as a stability band: "a re-run of this measurement would be expected to move the headline by up to X windows".
- Consequence for language: every Tier-A number is written as "one pass of a stochastic path, verdict stability Y% measured on 22 windows", never as "the system's agreement rate".

### 4.2 The rails move from code to prose

`no-llm` applies `apply_guard_rails` post-hoc in code; `rich` applies rails only as prompt text (engine.py: `guardrail_reason: None  # rich mode applies rails via the LLM prompt, not post-hoc`). Three concrete consequences the design must carry:

1. The transition-scoring rule that rejects transitions at `lap <= _NO_PIT_BEFORE_LAP` exists because the no-llm rail *forces* STAY_OUT there, making a lap-5 "transition" a rail artefact. In rich mode no such force exists, so an early transition could be a real LLM decision. **Kept identical anyway** for M1/M3 comparability; rich-mode transitions rejected by this rule are counted and footnoted (expected rare: only windows touching laps <= 6 qualify — Barcelona ALB L6, SAI L9, TSU L8; Monaco GAS L8, HUL L12; Silverstone RUS L10 area). 8.4 records the alternative.
2. M5 exists (prompt-rail compliance) and is free.
3. The INTERMEDIATE/WET minimum stint (`_DEFAULT_MIN_STINT = 6`) is enforced offline and **absent from every prompt** — the documented divergence. Silverstone is where it bites; M3's Silverstone stratum and M5's min_stint rows are its measurement.

### 4.3 Decision memory: the product carries state the probe cannot fully reproduce

The CLI product accumulates `DecisionMemory` across the whole race; a windowed probe cannot afford the pre-window laps that would populate it (that is the 3.4 h/race trap this design exists to escape). Decision: **memory ON, cold-opened at the window's first lap** — the accumulator starts empty, warms over the 1-6 lead-in laps, and is live by the scoring window. Stated limitation: recommendations near window start have less memory context than the same lap would have in a full-race run; the contingency echo (the load-bearing memory field) operates intra-window only. The alternative (memory OFF, matching the stateless `/recommend` surface) measures a different product; 8.1 hands the choice to Víctor because it changes what "the product" means in the paper's sentence.

### 4.4 ALERT: an action the scorer has no bucket for

The v2 enum allows ALERT; the MC layer never scores it and `_PIT_ACTIONS` does not contain it, so under the imported scoring rule an ALERT lap counts as "not asking to stop". That is a convention, not a fact. ALERT laps are counted separately in M6(d); if they exceed a trivial share (>2% of laps), the M1-M4 tables must carry an ALERT footnote line. 8.5 records the alternative convention (treat ALERT as an abstention bucket of its own).

### 4.5 Cost mechanics worth restating once

Zero cacheable prompt tokens (measured), so cost scales linearly in laps with no cache relief; 6 calls/lap means ~8.3k calls for the full design — rate limits, not wall clock, may bind if parallelised aggressively; each process re-pays ~25 s boot; a killed run leaves orphaned processes holding the Qdrant lock (known trap — kill them before re-running). The harness must **persist per-lap JSON incrementally** (fields, timings, token usage) so a crash at lap 900 keeps 900 laps — the same rule this report is written under.

---

## 5. The press-contrast axis, designed so it cannot be read as accuracy

**Where the press enters, and only there:**

1. **Tier B narratives.** Both windows already have published diagnoses (thesis 5.5.2/5.5.3); the run adds what the shipped system now says on those laps. The contrast is between three parties — the system, the real wall, and the chronicle — and none of the three is ground truth.
2. **Post-run annotation of Tier-A disagreements.** After M1-M4 are computed, the largest-|error| scored windows and a sample of `no_call` windows may be annotated against race reports. Selection happened before any chronicle was opened (the seed guarantees it), so annotation cannot contaminate the sample — this is the discipline that keeps the decline rate a population number while still letting the paper tell stories.

**The sentence a reader is allowed to write** (verbatim template, to be printed in the report):

> "On [window], the system recommended [action/lap]; the team did [real action/lap]; contemporary race coverage judged the team's call [favourably/unfavourably]. These are three opinions; the counterfactual outcome of the system's call was not simulated and cannot be, because the replay engine replays real laps by design."

**Sentences the report must never contain, in either direction:** "the system would have beaten the wall at [race]"; "the system agreed with the press N% of the time" (a rate over press-selected events is a rate over newsworthiness); "the system was right/wrong at [race]" (requires the full-field propagator that does not exist). The Qatar case gets one extra allowed sentence because it is a regression test, not a contrast: "the cross-integration module [did/did not] fire as shipped."

**Population discipline:** any number derived from press-annotated windows describes "windows selected for annotation", and the report prints that phrase on the same line. The 43.8%-style decline figures never mix with it.

---

## 6. Australia: decided

**Excluded from every quantitative tier, retained as the thesis's qualitative chat validation, unchanged.** Not carried for symmetry, and not silently dropped — excluded for three independent, stated reasons:

1. **It validates a different surface.** Thesis 5.5.1 is Streamlit chat + MCP tool calling + report generation over V13-V25 — the conversational product, not the per-lap recommendation path this measurement is about. The thesis itself says the validation "es cualitativa, no cuantitativa". Folding it in would blur the one boundary this document exists to draw (which population, which product).
2. **Its data cannot feed the decision metric.** Melbourne 2025: 82 stops, only 9 green-flag (22 neutralised laps — wet chaos). Nearly every window would land in excluded buckets; the budget it would consume buys `no_data` rows.
3. **The chat surface deserves its own measurement, not a seat in this one.** If chat quality is ever to be measured, that is an eval over tool-calling and answer fidelity with its own design; a lap-window sample is the wrong instrument.

Consequence: cluster 0's wet representation comes from Silverstone instead, where the decision metric can actually score stops.

---

## 7. The cut list, ranked (if the budget halves: 1,386 -> ~690 laps)

Ordered by claim damage per lap saved — take cuts from the top until the budget fits. Truncation of a race's draw is deterministic: drop the LAST stops accepted by the seeded protocol, so the surviving sample is still "the first k of the seeded order", not a re-choice. One protection: if truncation would drop a thesis-anchor window (Budapest LEC L19, NOR L31), that window moves to Tier B rather than disappearing — it stops being a population row and becomes a case row, and M1's n reflects that honestly.

| # | Cut | Laps saved | Running total | What it costs in claim strength |
|---|---|---|---|---|
| 1 | Tier C: drop the Monaco re-pass | 96 | 1,290 | Stability band measured on cluster 1 only (n=14 window pairs); its transfer to street/forced-stop regimes becomes an assumption stated in every M1 caveat |
| 2 | Barcelona 16 -> 8 windows | ~95 | ~1,195 | Cluster-1 CI widens; the SC-edge probes (ANT/BOR L49) likely truncate away; Barcelona keeps enough rows to anchor V0 |
| 3 | Suzuka 10 -> 6 | 48 | ~1,147 | The clean STAY_OUT-heavy regime thins; M4's most interpretable stratum loses half its resolution |
| 4 | Monza 13 -> 8 | 59 | ~1,088 | One-stop-hold regime depth; still represented, and the published no-llm Monza anatomy remains for shape comparison |
| 5 | Budapest 14 -> 9 (Tier C shrinks with it) | ~120 | ~968 | Thesis-race population rows thin AND the stability n drops; protected windows survive via the Tier-B rule |
| 6 | Mexico_City 7 -> 4 | 36 | ~932 | Cluster 3 leans harder on Monaco; altitude outlier keeps only a token presence |
| 7 | Montréal 7 -> 4 | 36 | ~896 | Cluster 2 becomes near-anecdotal (n=4); below this, do not cut further here — at 0 the cluster vanishes from the project's entire measured story |
| 8 | Drop Silverstone entirely | 132 | ~764 | The wet regime and the `_DEFAULT_MIN_STINT` divergence go unmeasured; **every published rate must then say "dry races only"** — this is the largest single claim downgrade on the list, which is why it sits this low despite the big saving |
| 9 | Tier B: drop the VER Qatar control arm | 10 | ~754 | The regression test keeps its assertions but loses the same-lap opposite-call contrast that makes it readable |
| 10 | Monaco Tier A 8 -> 5 | 36 | ~718 | Cluster 3 thins further |
| 11 | Lusail 5 -> 2 (drop the TD-forced trio) | 36 | ~682 | The forced-stop sub-regime line dies; the two genuinely chosen Lusail stops survive |

**Do not cut, at any budget:** the no-llm preflight and V0 (free, and they are the licence to trust everything else); Tier C below ~14 window pairs (without a stability band, no Tier-A number is publishable at all — see 4.1); the Qatar PIA regression window (it is the only guard on an already-published diagnosis).

---

## 8. What I could not resolve (Víctor decides; each fork changes the numbers)

Places where two reasonable readings give different results and no measurement settles the choice. My recommendation is marked, but these are deliberately not decided here.

1. **Which product is under test: memory ON (cold-open) or OFF?** The CLI accumulates `DecisionMemory`; `/recommend` and the MCP tool are stateless by design. ON with cold-open (my recommendation, 4.3) measures the flagship surface but with weaker memory than a full-race run would have; OFF measures the stateless surface exactly. The paper's sentence "the system recommends X" means a different thing under each. *Changes: every metric.*
2. **Railed stops: excluded from the draw (as designed) or included as probes?** Excluded keeps the M1/M3 population identical in kind to the no-llm headline's and spends no budget on foreclosed windows; included would measure whether the LLM *agrees with stops the deterministic rails forbid* — a real question about the prose-vs-code rail gap that M5's lap-level compliance only partially covers (M5 sees the system's own laps, not the real wall's forbidden stops). ~13 stops across the 9 races. *Changes: the population statement and M2's denominator.*
3. **Which aggregate is THE number in the thesis/paper: unweighted (primary, as designed) or cluster-reweighted?** Unweighted is the honest conditional-on-these-races number; reweighted reads as a season estimate but rests on the stage-1 representativeness assumption (2.3). Quoting the reweighted one as the headline is defensible and I recommend against it. *Changes: the quoted headline by an unknown few points.*
4. **Transition rule at laps <= `_NO_PIT_BEFORE_LAP` in rich mode: keep the no-llm rule (as designed) or relax it?** Kept, an early real LLM decision can be unscored as a rail artefact that no longer exists; relaxed, M3's pairing is no longer rule-identical. Footnote counts are designed in either way. *Changes: a handful of early-stop windows (6-8 candidates).*
5. **ALERT convention: non-pit action (as designed) or its own abstention bucket?** As non-pit, an ALERT-heavy window can land in `no_call_in_window` and inflate the decline rate; as its own bucket, M2 gains a bucket the no-llm tier never had and the paired contrast loses symmetry. *Changes: M2/M4 if ALERT is common; unknown until run.*
6. **Wet (Silverstone) and TD-forced (Lusail trio) windows: inside the headline with sub-regime lines (as designed), or excluded from the headline?** Inside says "2025 as it happened, including regimes the models never trained on and causes the system cannot observe"; excluded says "the population the system was built for". Both are defensible population definitions and they will produce different headline rates. *Changes: M1-M4 headline; the sub-regime lines exist either way.*
7. **Spend 2-6 paid calls probing whether `gpt-5.4-mini` honours a `seed` parameter before sizing Tier C?** If a seed is honoured, Tier C shrinks to a confirmation and the saved 264 laps buy ~22 more Tier-A windows; if not (the measured behaviour suggests not), the probe cost is noise. I cannot run it — it spends LLM calls. *Changes: budget allocation between Tiers A and C.*
8. **Pre-committed stopping rule on stability: if Tier C's window-verdict discordance exceeds ~20%, is the single-pass M1 headline publishable at all?** Pre-committing (my recommendation) forfeits flexibility but is the only version a reviewer cannot call post-hoc; deciding after seeing the number is exactly the manoeuvre the project's own doctrine forbids elsewhere. *Changes: whether the measurement publishes on the first budget or needs a second.*
9. **The Lusail maximum-stint directive is asserted here from its data signature** (field-wide stops at exactly tyre-life 25), not from a document — the TD text is not in the repo, and the RAG corpus holds season rulebooks only. Before 5.5.3-adjacent prose quotes it as regulation, verify it in the press/FIA record during the annotation step. Not a measurement fork, but the one factual claim in this design resting on inference. *Changes: one sentence of framing; flagged so it is never quoted unverified.*

---

*Report complete. Draw artefacts (full census of all 24 races, per-stop eligibility, the seeded draw, and per-driver replay spans) are reproducible from seed 20250806 via the protocol in 1.1; the generating script is in the session scratchpad and is intentionally NOT committed — the protocol above is the specification, and re-running it yields the identical sample.*
