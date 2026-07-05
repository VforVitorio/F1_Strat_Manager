# AUDIT ML-EVAL - ML / multi-agent evaluation and robustness framework

**Auditor:** Fable 5 · **Date:** 2026-07-05 · **Repo:** `F1_Strat_Manager` (read-only pass, no code changed)
**Scope:** the scientific evaluation of the seven ML predictors (N06-N16), the six sub-agents (N25-N30), and the orchestrator N31 (MoE routing + 500-sample Monte Carlo + LLM synthesis): per-model metrics and calibration, test-set hygiene and leakage checks, orchestrator-layer validation (MC, routing, guardrails, 14-field output), robustness / failure-mode cataloging, and the ablation framework the IEEE TETCI paper and the future Rival Agent TFM need.
**Hard constraints honored in every remedy:** plan only, no code; UNTOUCHABLE (additive entry points only, duplicate before modifying): `src/agents/` internals, `scripts/run_simulation_cli.py`, `notebooks/**`, `legacy/**`; LLM = OpenAI / LM Studio, never Anthropic; no eval run ever calls Anthropic.
**Inputs read:** `src/agents/*.py` (orchestrator, pace, tire, situation, pit, radio, rag), every `model_config*.json` / `feature_manifest*.json` / calibration JSON under `data/models/` and `data/processed/`, `strategy_orchestrator_config_v1.json`, notebooks N30B / N31-viz / N32 / N33 (headers and eval cells), `tests/` (all files), `documents/audits/AUDIT_2026_REG_CONCEPT_DRIFT.md`, `AUDIT_TESTING_QA.md`, `AUDIT_P2B_CORE_COMPUTE.md`, `AUDITS_BACKLOG.md`.

---

## 1. Framing

The TFG is defended (10.0) and the tribunal unanimously recommended the IEEE paper. The paper's own Discussion section concedes the system's weakest scientific flank: end-to-end validation rests on three curated case studies, not a systematic protocol, and calibration was asserted but never verified. This audit designs the evaluation harness and ablation framework that turns "the system works on Melbourne, Hungary and Qatar" into "the system's quality is measured, calibrated, ablated and reproducible over a full season". Everything here is additive: a new `src/strategy/eval/` package plus `tests/eval/` goldens, never edits to agent internals, notebooks or the PMV.

**Boundary with the two sibling efforts (read this before converting to issues):**

- **AUDIT_2026_REG_CONCEPT_DRIFT Phase 1 ("measurement layer", findings F-03/F-04)** already specifies the calibration verification harness (reliability + Brier for the 3 classifiers, quantile coverage for N15, MC-sigma ratio for the TCN) and the per-GP drift monitors. This audit does NOT re-specify them. It treats that layer as the shared substrate and adds what a *paper* needs on top: leakage and threshold-provenance verification, per-slice breakdowns as citable reports, orchestrator-layer validation, guardrail conformance, the season-scale replay protocol and the ablation matrix. One codebase (`src/strategy/eval/`), two consumers: drift monitoring (2026-reg) and scientific reporting (here).
- **AUDIT_TESTING_QA** owns pass/fail PR gates. Its fixtures (FakeOpenAI stub T-3, canned lap_states / mini parquet T-6, engine golden scenarios T-2/T-10, guard-rail table tests) are the same artifacts this audit's harness consumes. Division of labor: tests assert contracts and gate PRs; the eval harness produces *measurements* (metrics, distributions, rates, tables) that gate *claims*. The fixtures carve-out already landed (`tests/fixtures/README.md`).

---

## 2. Executive summary

The system's predictive layer has good notebook-era evaluation hygiene (temporal splits 2023-2024 train / 2025 test, documented leakage exclusions in the manifests, a dedicated thresholds-and-calibration notebook N33, a quantitative RAG benchmark N30B) but **none of it is reproducible, automated, or complete enough to defend the paper's claims**. Headline metrics live scattered across `model_config` JSONs, notebook cells and the thesis, with at least one divergence between sources (pace MAE 0.392 vs the thesis-final 0.4104). The provenance of the three published decision thresholds and of every historical-aggregate feature is unrecorded, so test-set contamination of the published numbers cannot currently be ruled out. That is a pre-submission blocker for the paper.

The orchestrator layer is worse off: it has **zero quantitative evaluation of any kind**. The Monte Carlo layer draws pace samples it never uses (`strategy_orchestrator.py:685`, `noqa: F841`), runs on a fixed seed 42 embedded in the function (`:637`), claims a convergence property in a docstring that was never measured (`:92-93`), feeds a 3-lap SC probability into a 5-lap Bernoulli window (`:668`, `:687`, `WINDOW_LAPS` at `:545`), and rests on five hardcoded physics constants (`:545-549`) whose sensitivity was never tested. The MoE routing runs on a scalar SC threshold (`:108`, `:521`) while the shipped config documents per-cluster thresholds that were never wired in. The strategic guardrails are mostly prompt text (`:862-877`); their conformance rate under a real LLM has never been measured, and the RCM-SC failure (Qatar V7, fixed by RCMContextResolver) proved that this layer can silently invert an outcome. No ablation exists for any architecture claim, which is a problem because the three-layer orchestrator IS the paper's novelty hook.

Plan: 5 phases. Phase 1 builds the metrics registry + per-model eval harness on the shared calibration substrate. Phase 2 is the hygiene audit (threshold provenance + aggregate-feature leakage), which must land before the paper's results table freezes. Phase 3 validates the MC and routing layers (deterministic, no LLM needed). Phase 4 measures guardrail conformance, output semantic quality and robustness. Phase 5 delivers the season-scale counterfactual replay protocol and the ablation matrix: the tables the paper cites and the baseline the Rival Agent TFM compares against.

---

## 3. Current evaluation state inventory

### 3.1 What exists (and its ceiling)

| Artifact | What it gives | Ceiling |
|---|---|---|
| `model_config*.json` per model (`data/models/*/`) | Frozen headline metrics: overtake AUC-PR 0.5491 / threshold 0.7976; SC AUC-PR 0.0723 / threshold 0.2335 + Platt coef; undercut AUC-PR 0.6739 / threshold 0.522 + Platt coef; pit P50 MAE 0.487 / **P05-P95 coverage 0.7047 vs 0.90 nominal**; train/test season declarations | Static snapshots, not regenerable; no per-slice breakdowns; provenance of thresholds unrecorded; the broken pit coverage sits published inside the config |
| Feature manifests (`data/processed/feature_manifest_laptime.json`, `tiredeg_feature_manifest.json`) | Documented leakage exclusions (`features_out.leakage`, `leaky_columns`) and the lag rule for `DegradationRate`/`DegAcceleration` | The exclusions are documentation; nothing verifies the inference path honors them |
| `N33_thresholds_and_calibration.ipynb` | Threshold sweeps for N12/N14/N16 + MC-Dropout empirical coverage, visual | One-shot notebook, untouchable, not regenerable as a gate; which split the sweeps ran on is exactly the E-02 question |
| `N30B_rag_benchmark.ipynb` | 15 Spanish queries with manual article-level ground truth; Precision@k, MRR, latency across 3 retriever configs | Small n, notebook-only; does not cover the 3 canned production questions of `_build_rag_question` (`strategy_orchestrator.py:717-738`) |
| `N31_mc_visualization.ipynb`, `N32_smoke_test.ipynb` | MC distribution plots on one real lap; per-agent smoke pass criteria | Visualization and smoke only; no acceptance metrics |
| `data/models/tire_degradation/mc_dropout_calibration.json` | Per-compound frozen sigmas (C2/C4/C5/C6) | Frozen on validation-era data; the sigma-vs-realized-error ratio is never re-measured (2026-reg F-03 territory) |
| `scripts/bench_*.py` (pace baselines, sub-agent latency, NLP CPU, Whisper) | Ad hoc latency/baseline probes; Radio P95 59.4 ms is a thesis RNF number | Latency is P2b's domain; not integrated into any report format |
| `tests/` | Qatar 2025 V7 SC-override regression (`test_smoke.py:59`, data-gated), structural agent tests, `test_cli_no_llm.py` fixture smoke; `tests/fixtures/` carve-out README landed | Contracts, not measurements; nearly all data-gated in CI (Testing audit T-4) |

### 3.2 What "defensible" requires that does not exist

1. **Reproducibility:** one command that regenerates every published metric from the frozen artifacts + the HF dataset, with versioned output. Today the numbers are archaeology.
2. **Hygiene proof:** documented, checked provenance for thresholds and aggregate features. Today it is trust.
3. **Orchestrator measurement:** any number at all about Layer 1 routing, Layer 2 MC, or Layer 3 synthesis quality. Today there is none.
4. **Ablation evidence:** any measured delta supporting "the MoE routing helps", "the MC layer helps", "guardrails help", "the LLM synthesis helps". Today there is none.
5. **Scale:** validation on ~24 races, not 3. The paper's own future-work list names this.

---

## 4. Findings register (P0-P3)

| ID | P | Finding | Why / risk | Size |
|---|---|---|---|---|
| **E-01** | **P0** | **End-to-end validation is 3 case studies; no systematic protocol exists.** Melbourne + Bahrain/Hungary + Qatar demos carry every architecture claim; the paper's Discussion admits it and its future work names a systematic multi-race protocol | The IEEE paper's central claims are anecdotally supported; a reviewer asking "over how many races?" has no good answer. Also blocks the Rival Agent TFM, which needs this system as a measured baseline | **L** |
| **E-02** | **P0** | **Threshold and aggregate-feature provenance unrecorded; test-set contamination of published metrics cannot be ruled out.** `optimal_threshold: 0.7976` (`data/models/overtake_probability/model_config.json`), `best_threshold: 0.2335` (`safety_car_probability/feature_list_v1.json`), `best_threshold: 0.522` (`pit_prediction/model_config_undercut_v1.json`) do not record which split selected them; historical aggregates consumed as features (`circuit_sc_rate`, `circuit_undercut_rate`, `team_x_undercut_rate`, `team_year_median`, `year_circuit_median`, `team_pace_rank`, `Cluster`) do not record their computation window relative to the 2025 test season | If any threshold was tuned on test-2025 or any aggregate includes test-season rows, the headline numbers destined for the paper's results table are optimistic and the "strict temporal split" claim is false. Must be verified BEFORE submission; cheap to check, catastrophic to discover in review | **M** |
| **E-03** | **P0** | **Calibration is asserted, not verified** (thesis limitation admits it; pit P05-P95 coverage already measured broken at 0.7047 vs 0.90 in `pit_prediction/model_config.json` "eval"). SHARED with 2026-reg F-03: that audit specifies the verification harness; this audit requires its outputs to become citable, versioned report artifacts with acceptance bands, because the orchestrator's MC layer consumes these distributions directly (`_run_mc_simulation`, `strategy_orchestrator.py:609-710`) | The paper says probabilities are calibrated (Platt as the integration bridge is a named design decision); one of the four calibration families is already known to be wrong. Cross-reference, do not duplicate: build once in the shared measurement layer | **M** (shared) |
| **E-04** | **P1** | **The Monte Carlo layer is scientifically unvalidated.** (a) pace samples drawn then discarded: `pace_s = rng.normal(...)  # noqa: F841` (`strategy_orchestrator.py:685`), so N25's uncertainty contributes nothing to MC scores; (b) fixed `seed=42` inside the function (`:637`): identical noise stream every lap, seed-sensitivity never estimated; (c) docstring claims "500 draws keep variance of the mean below 0.01 position units" (`:92-93`), never measured; (d) horizon mismatch: `sc_prob_3lap` drives the Bernoulli SC draw (`:668`, `:687`) over a `WINDOW_LAPS = 5` window (`:545`); (e) five hardcoded constants `FRESH_GAIN 0.25 / CLIFF_LOSS 0.80 / POS_GAP_S 1.50 / SC_PIT_BONUS 8.0 / WINDOW_LAPS 5` (`:545-549`) with no sensitivity analysis; (f) when N28 is not routed, MC silently substitutes prior Triangular(2.2, 2.8, 3.8) and `undercut_prob = 0.5` (`:681-683`), a coin-flip that still scores UNDERCUT every lap | The MC layer is the paper's Layer 2 and the score `alpha*E + (1-alpha)*P10` (`:702`) decides actions in no-LLM mode outright. Unmeasured convergence, dead inputs and unsourced constants are reviewer bait; the coin-flip fallback can dominate decisions on quiet laps without anyone knowing how often | **M** |
| **E-05** | **P1** | **Guardrail conformance is unmeasured.** The six strategic rails are prompt text (`strategy_orchestrator.py:862-877`); only two code-level guards exist (SC-active STAY_OUT to PIT_NOW override, `pit_strategy_agent.py:1006-1008`; the no-LLM guards, `scripts/run_simulation_cli.py:1529-1560`). No adversarial battery exercises the rails; no violation rate exists. The Qatar V7 incident (fixed by RCMContextResolver, regression pinned in `tests/test_smoke.py:59`) is proof this layer can invert an outcome | "Guardrails encode F1 realism" is a paper claim with zero supporting measurement. Prompt-level constraints are exactly the kind an LLM violates a few percent of the time; without a rate, the claim is unfalsifiable | **M** |
| **E-06** | **P1** | **MoE routing correctness never evaluated, and config/code diverge.** `_decide_agents_to_call` (`strategy_orchestrator.py:475-537`) has truth-table unit tests (Testing audit) but no data-driven evaluation (did N28 activate ahead of actual pit events? how often does N30 fire?); the code uses scalar `sc_prob_threshold = 0.30` (`:108`, used `:521`) while `data/models/agents/strategy_orchestrator_config_v1.json` documents `sc_prob_threshold_by_cluster` 0.20-0.35 and its own `v09_handoff` note says the cluster-aware swap is still pending (N26 DID get its cluster-aware cliff thresholds: `tire_agent.py:246-257`) | Routing determines which distributions the MC gets real vs fallback (see E-04f), so routing errors propagate numerically. And the paper/docs must not describe cluster-aware SC routing that production does not perform | **S-M** |
| **E-07** | **P1** | **Zero ablation evidence for any architecture claim.** No with/without-agent, with/without-guardrails, with/without-RCMContextResolver, alpha sweep, MC-sample sweep, or LLM-vs-no-LLM comparison has ever been run | The three-layer orchestrator is the paper's novelty hook; without an ablation table it reads as engineering description, not evaluated architecture. The tribunal-recommended venue (IEEE TETCI) will expect Table-style ablations | **M** (framework) |
| **E-08** | **P1** | **No single reproducible metrics registry.** Numbers live in model_config JSONs, notebook cells, thesis chapters and memory files; at least one divergence exists (pace MAE 0.392 notebook-era vs 0.4104 thesis-final); the paper plan already had to declare a metrics authority by fiat | Every downstream document (paper, AEPIA 5-pager, docs site, model cards) copies numbers by hand today. One regenerable registry ends the drift | **S** |
| **E-09** | **P2** | **14-field output semantic quality unmeasured.** Pydantic guarantees types (`StrategyRecommendation`, `strategy_orchestrator.py:317`), but nothing measures: `pit_lap_target` within race bounds, `target_lap_time_s` inside the pace CI as the prompt itself demands (`:932-933`), `compound_next` consistent with rail 5, `undercut_target` an actual rival, contingency triggers concrete, reasoning-rubric adherence (`:898-905`: must cite cliff P50, pace delta, a situational signal, a regulation article when present). The `confidence` field has never been compared to outcomes (Qatar first run: 92% confident in the wrong action) | Output quality is what the demo surfaces show users and what the paper quotes; today its quality is vibes. Confidence calibration is a cheap, novel-ish result for the paper's Discussion | **M** |
| **E-10** | **P2** | **Robustness catalog is informal.** Known edge classes are handled ad hoc: lap-1 degenerate distributions crash-fixed by `_clamp_triangular` (`strategy_orchestrator.py:642-660`, the comment documents the original crash), FastF1 incomplete-lap guard (`run_simulation_cli.py:1825`), missing-rival / empty-radio behavior undocumented, unseen-category behavior untested (N15 LabelEncoder raises, per 2026-reg F-06). The single-driver boundary (full telemetry for our driver, timing-only for rivals: `strategy_orchestrator_config_v1.json` "data_boundary", enforced in `RaceStateManager`) has no information-leakage test asserting agents never consume `rival_fields_NOT_available` | Failure modes get discovered in demos (the Qatar way). A catalog + executable battery converts each one into a regression the moment it is found. The boundary test is also a thesis-integrity check: the "realistic pit wall" claim depends on it | **M** |
| **E-11** | **P2** | **RAG evaluation is small and unwired.** N30B exists (15 queries, P@k/MRR, manual ground truth, 3 configs) but is notebook-only; the production orchestrator asks only 3 canned question shapes (`_build_rag_question`, `strategy_orchestrator.py:717-738`) which the benchmark does not cover verbatim; the paper's limitations list already concedes the RAG ground-truth gap | The regulation citation in the Qatar case (Article 36.3) is a headline demo moment resting on an unevaluated retrieval path for exactly those production questions | **S-M** |
| **E-12** | **P2** | **NLP pipeline evaluation frozen at notebook-era.** N17-N24 metrics (sentiment, intent, NER F1 0.4151, latency) are one-shot; radio-alert intents feed routing (`_decide_agents_to_call` consumes PROBLEM/WARNING/PENALTY intents, `strategy_orchestrator.py:517-525`) but alert precision on the live pipeline was never measured; no label-stability regression exists (Testing T-11 covers the test side) | A noisy radio-alert channel silently degrades routing; measuring it also quantifies how much the Radio agent contributes (feeds the E-07 ablation) | **S** |
| **E-13** | **P3** | **N33's threshold sweeps and coverage plots are not regenerable.** Visual, notebook-bound, split-provenance unclear (part of E-02) | Once the harness exists, N33's content becomes `f1-eval` reports; notebook stays as the historical record | **S** |
| **E-14** | **P3** | **Latency benches are ad hoc and separate.** `bench_subagent_latency.py` etc. produce thesis numbers but no versioned report; P2b owns runtime optimization | Fold bench outputs into the same report format so RNF claims regenerate too; no new measurement work here | **S** |
| **E-15** | **P3** | **No era/version labeling convention for eval outputs** (2026-reg F-15 names the docs side) | Every report must carry model-artifact hashes, dataset version, era tag and LLM model+version from day one, or 2026 retraining makes all reports ambiguous | **S** |

---

## 5. Target harness design (additive only)

**Home:** new package `src/strategy/eval/` (the empty `src/strategy/training/` sibling is 2026-reg Phase 0's home; same package family, consistent with `src/strategy/README.md` marking the old jupytext exports as stale). Deterministic goldens that gate PRs live in `tests/eval/` and follow the Testing audit's tier markers. Heavy runs are launched by an additive `f1-eval` console script (pyproject entry point, like `f1-strat`/`f1-sim`).

**Modules (conceptual, one concern each):**

| Module | Concern | Consumes |
|---|---|---|
| `metrics_registry` | Regenerate + version every headline metric; emit the single citable table | frozen artifacts under `data/models/`, HF dataset holdouts |
| `calibration` | The 2026-reg F-03 substrate: reliability/Brier/ECE, quantile coverage, MC-sigma ratio | same; shared with drift monitors |
| `hygiene` | Threshold-provenance + aggregate-feature-window + manifest-lag verification (E-02) | notebooks read-only as evidence, featured parquets |
| `mc_eval` | Convergence, seed variance, horizon analysis, constants sensitivity, fallback-usage tracking (E-04) | `_run_mc_simulation` via public import, canned sub-agent outputs |
| `routing_eval` | Activation stats vs realized events; scalar-vs-cluster threshold comparison (E-06) | `_decide_agents_to_call` via public import, 2025 replays |
| `conformance` | Guardrail battery, 14-field semantic validator, reasoning-rubric scorer, confidence calibration (E-05/E-09) | orchestrator entry points + FakeOpenAI (plumbing) / real LLM (rates) |
| `robustness` | Edge-input battery + single-driver boundary leakage test (E-10) | canned/mutated lap_states |
| `replay` | Season-scale counterfactual protocol (E-01) | `RaceStateManager` replays, full-season data |
| `ablation` | Knockout/sweep matrix runner + LaTeX table emitter (E-07) | all of the above |

**Untouchable-boundary technique:** everything runs through the public entry points (`run_*_from_state`, `run_strategy_orchestrator_from_state`, `_run_mc_simulation` and `_decide_agents_to_call` are importable pure functions, a pattern `tests/test_agents.py` already uses). Knockouts and sweeps are done by (a) constructing inputs, (b) monkeypatching at the entry-point seam from eval code, (c) config-side variation. If a sweep genuinely needs a parameter the internals hardcode (the MC seed is the known case, `strategy_orchestrator.py:637`), the eval code wraps or monkeypatches; proposing an internal edit requires the same case-by-case ruling as 2026-reg open question 4.

**Outputs:** small versioned reports (markdown + CSV) in `documents/eval_reports/`, committed; heavy per-lap parquets in `data/eval/` (gitignored, HF-synced like the rest of `data/`). Every report header carries: artifact hashes, dataset snapshot, era tag, seed policy, LLM provider+model+version (OpenAI gpt-4.1-mini or the LM Studio local model; never Anthropic), and the git SHA of the harness (E-15).

---

## 6. Ablation framework (the paper's tables)

All arms run the same 2025 replay set (subset for LLM arms, full season for no-LLM arms) and report the same outcome metrics (defined in Phase 5): action distribution, agreement rate with actual wall decisions, pit-call lead time, guardrail violation rate, mean MC score of chosen action, confidence calibration.

| Ablation | Arms | Answers | LLM needed |
|---|---|---|---|
| Sub-agent knockout | full system vs minus-N25 / minus-N26 / minus-N28 (forces the Triangular fallback, E-04f) / minus-N29 / minus-N30 | contribution of each expert; formalizes the MoA claim | no-LLM arm first, LLM arm confirmatory |
| Guardrails on/off | prompt rails present vs stripped; no-LLM guards on vs off | violation rate delta; "guardrails encode realism" evidence | yes (rails are prompt-level) |
| RCMContextResolver on/off | resolver active vs bypassed on SC-containing races | the Qatar case generalized: how many SC laps flip decision | no (deterministic path dominates) |
| Alpha sweep | risk_tolerance in {0, 0.25, 0.5, 0.75, 1.0} | decision-frontier sensitivity of `score = alpha*E + (1-alpha)*P10` | no |
| MC sample count | n_sim in {50, 100, 250, 500, 1000, 2000} | validates the ":92-93" convergence claim; latency/quality trade (feeds P2b) | no |
| Classifier thresholds | +/- sweep around 0.7976 / 0.2335 / 0.522 | decision stability vs threshold choice; pairs with E-02 re-derivation | no |
| Layer 3 on/off | LLM synthesis vs no-LLM argmax path | what the LLM adds beyond MC argmax (action deltas, override frequency) | yes |
| LLM model swap | gpt-4.1-mini vs LM Studio local model | provider robustness of Layer 3; supports the provider-agnostic claim | yes |

The knockout, alpha, n_sim and threshold ablations are deterministic (seed-controlled, below Layer 3) and therefore cheap and CI-friendly at small scale; the LLM arms are data-tier runs with recorded transcripts.

---

## 7. Robustness / failure-mode catalog (initial population)

Each entry becomes an executable battery case in Phase 4; the catalog is append-only and grows from every future demo incident.

| # | Failure mode | Status today | Evidence |
|---|---|---|---|
| R-1 | SC active but wall-bias STAY_OUT replicated (RCM signal not propagated) | FIXED by RCMContextResolver; regression exists but data-gated | `tests/test_smoke.py:59`; resolver contract in `race_situation_agent.py` (`_sc_active_from_rcm`) |
| R-2 | Lap-1 degenerate distributions (identical p10/p50/p90) crashed the MC | FIXED by `_clamp_triangular`; behavior at the clamp never characterized | `strategy_orchestrator.py:642-660` |
| R-3 | N28 not routed: coin-flip undercut prior silently scores UNDERCUT | LIVE, unmeasured | `strategy_orchestrator.py:681-683` |
| R-4 | LLM violates a prompt rail (pit lap 1-4, wrong compound-vs-laps, REACTIVE_SC on prediction) | Unmeasured; only the SC-active rail has a code backstop | `strategy_orchestrator.py:862-877`; `pit_strategy_agent.py:1006` |
| R-5 | Missing / partial rival rows, empty radio window, NaN telemetry fields | Undocumented behavior | `run_simulation_cli.py:1825` guards one FastF1 case; agents' tolerance unknown |
| R-6 | Unseen categories (new team/compound/GP) raise or silently mis-encode | Known future break (2026-reg F-06); no current-era test that the failure is LOUD | N15 LabelEncoder classes in `pit_prediction/model_config.json` |
| R-7 | Single-driver boundary breach (an agent consuming rival-forbidden fields) | Never asserted | `strategy_orchestrator_config_v1.json` "data_boundary" `rival_fields_NOT_available` |
| R-8 | Confidence miscalibration (high confidence on wrong action) | One anecdote (Qatar 92%); no rate | E-09 |
| R-9 | Regulation citation wrong-season or hallucinated article | Unmeasured; N30B checks retrieval, not the cited-article faithfulness in synthesis | `_build_rag_question` + reasoning rubric step 3 |
| R-10 | Future resolver classes: red flag, applied penalty, pit-lane closed | Not implemented (thesis future-work); catalog placeholders so batteries exist the day they land | RCM resolver extension list |

---

## 8. Phased, chunkable plan (each phase = one GitHub sub-issue set)

Ordering rationale: registry before hygiene (you need regeneration to re-derive corrected numbers), hygiene before the paper's results freeze, deterministic orchestrator eval before LLM-dependent conformance, everything before the season-scale protocol that consumes it. Phases 1-2 are the paper's minimum; 3-5 are what makes the paper strong and the TFM baseline real.

**Phase 1 - Metrics registry + per-model eval harness (M)**
- `src/strategy/eval/` skeleton + `f1-eval` entry point + report/versioning conventions (E-15 header contract).
- Regenerate the seven predictors' headline metrics from frozen artifacts on the 2025 holdout; add per-season, per-circuit, per-cluster and per-compound breakdown slices (none exist today).
- Wire the calibration substrate WITH the 2026-reg Phase 1 work (E-03: reliability/Brier/ECE, N15 coverage, MC-sigma ratio); one implementation, two report consumers.
- Emit the consolidated metrics registry (versioned JSON + markdown table): the single citable source replacing scattered numbers (E-08); reconcile the 0.392-vs-0.4104 class of divergences explicitly.
- Deliverable: `f1-eval models` reproduces every `model_config` headline number within stated tolerance, or documents the delta.

**Phase 2 - Hygiene: threshold provenance + leakage verification (M)**
- Trace the derivation split of the three published thresholds through N12/N14/N16/N33 (read-only); record a verdict per threshold: val-derived / test-derived / undocumented.
- Trace the computation window of every historical-aggregate feature (`circuit_sc_rate`, `circuit_undercut_rate`, `team_x_undercut_rate`, `team_year_median`, `year_circuit_median`, `team_pace_rank`, `Cluster` membership) relative to the 2025 test season.
- Verify the tire-deg lag rule (manifest notes: `DegradationRate`/`DegAcceleration` must enter lagged) holds in the production inference path (`src/strategy/inference/tire_predictor.py`) and in each agent's feature assembly.
- If contamination is found: re-derive thresholds on val-2024 only, recompute affected metrics, publish registry v2 with corrected numbers BEFORE the paper's results table freezes; if clean: record the provenance in the model_configs' successor registry so the question is answered forever.
- Deliverable: a signed hygiene report (clean / contaminated / underdocumented per item) + registry v2 if needed. **This phase gates the IEEE submission.**

**Phase 3 - Orchestrator decision-layer eval: MC + routing (M)**
- MC validation suite (E-04): convergence curve over n_sim in {50...2000} with score standard errors (tests the `:92-93` claim); seed-variance estimate (multiple seeds via wrapper/monkeypatch, since seed 42 is hardcoded); quantified impact of the 3-vs-5-lap SC horizon mismatch; one-at-a-time sensitivity on the five constants (`:545-549`); fallback-prior usage frequency measured over a season replay (how often N28-absent laps' decisions would flip with real quantiles).
- Ruling item for Víctor: the dead pace samples (`:685`): either document "N25 uncertainty deliberately excluded from MC, prompt-only" as a design decision in the paper, or plan the additive fix; the ablation (Phase 5) measures which answer is true.
- Routing evaluation (E-06): activation confusion analysis on 2025 replays (N28 activations vs actual pit events within k laps; N30 firing rate and trigger mix); scalar-vs-cluster SC threshold comparison to resolve the config/code divergence with data (then either wire the cluster thresholds additively or delete them from the config and docs).
- Deliverable: MC validation report + routing report; both deterministic, no LLM required.

**Phase 4 - Conformance, output quality and robustness batteries (M-L)**
- Guardrail battery (E-05): adversarial lap_state set per rail (6 rails x boundary variants); plumbing determinism via the Testing audit's FakeOpenAI; measured conformance RATE via the real-LLM data tier (OpenAI gpt-4.1-mini and the LM Studio model; recorded transcripts; never Anthropic); code-backstop rails asserted as hard PR-gating tests in `tests/eval/`.
- Semantic output validator (E-09): deterministic cross-checks of the 14 fields (ranges, CI containment, rail-5 compound consistency, rival existence, contingency-trigger concreteness) + reasoning-rubric adherence scorer (string-level checks first: does reasoning cite cliff P50, pace delta, a named signal, an article when regulation_context is present); confidence-vs-outcome calibration over replay results.
- Robustness battery (E-10): execute the §7 catalog (R-2/R-3/R-5/R-6 edge inputs, loud-failure asserts for unseen categories) + the single-driver boundary leakage test (R-7).
- RAG production-question eval (E-11): extend N30B's ground-truth method to the 3 canned `_build_rag_question` shapes as harness cases, plus cited-article faithfulness checks in synthesis output (R-9).
- NLP alert-precision probe (E-12): measure radio-alert intent precision on the committed transcript pairs; feeds the N29 knockout arm.
- Deliverable: conformance report with per-rail violation rates; robustness battery green or catalogued-red; batteries wired to the Testing audit's tier markers.

**Phase 5 - Season-scale replay protocol + ablation matrix (L)**
- The systematic protocol the paper's future work names: counterfactual replay over the 2025 season (~24 races) through `RaceStateManager` + orchestrator entry points; no-LLM arm full-season, LLM arm on a budgeted subset.
- Outcome metrics defined and frozen BEFORE running (agreement rate with actual wall pit decisions, pit-call lead time distribution, action distribution per race phase, guardrail violation rate, confidence calibration at scale, per-race failure harvest into §7).
- The §6 ablation matrix executed; LaTeX-ready tables emitted for the paper (knockouts, alpha, n_sim, thresholds, rails, resolver, Layer 3 on/off, model swap).
- Freeze the whole run (configs, seeds, transcripts, reports) as the **Rival Agent TFM baseline**: the TFM's rival-aware system must beat these numbers on the same protocol.
- Deliverable: the paper's evaluation section artifacts + a tagged baseline snapshot.

Dependency spine: Phase 1 first (Phase 2 needs regeneration; 3-5 need the report/versioning substrate). Phase 2 gates the paper freeze. Phase 3 is parallel-safe with Phase 2. Phase 4 needs the Testing audit's FakeOpenAI + fixtures (its Phase 1). Phase 5 needs all of the above.

---

## 9. Open questions (need Víctor's decision)

1. **Paper timeline coupling:** which phases must land before TETCI submission? Minimum defensible: Phases 1-2 (registry + hygiene). Strongly recommended: Phase 3 + the no-LLM ablation arms of Phase 5, because the novelty hook needs at least one ablation table. Decide the cut line against the writing calendar.
2. **Reference LLM for reported numbers:** conformance rates and Layer-3 ablations depend on the model. Propose: gpt-4.1-mini as the citable reference (stable, cheap), LM Studio local as the secondary arm; every report records model+version. Ratify.
3. **Ground-truth oracle for agreement metrics:** actual wall pit decisions (from race data) are a biased oracle; the Qatar case is precisely one where the wall was wrong. Proposal: report agreement AND divergence-with-outcome analysis (when we diverge, did the MC-projected outcome materialize?), not agreement alone. A full Heilmeier-style outcome simulator is out of scope here (TFM territory). Ratify the framing.
4. **If Phase 2 finds contamination:** silently replace numbers, or report both with an erratum note in the thesis-to-paper delta? Proposal: registry v2 with corrected numbers + one honest sentence in the paper (the N12B negative-result precedent shows honesty reads well). Decide.
5. **LLM budget for Phase 4-5:** the full-season LLM arm at ~50-70 decisions/race x 24 races is thousands of calls; propose budgeted subset (2 races per cluster archetype, 8 races) for LLM arms, full 24 for no-LLM. Approve the subset design.
6. **Seed-policy ruling (E-04b):** exposing the MC seed needs either a wrapper/monkeypatch (pure-eval, zero prod change) or a tiny additive parameter on the entry point (agent-internals adjacent). Same decision gate as 2026-reg open question 4; the wrapper works without any ruling if preferred.
7. **Where eval reports live long-term:** `documents/eval_reports/` in-repo (proposed) vs HF; and whether the report format should anticipate the `pitlab` dashboard contract (2026-reg §6) now or later. Proposal: in-repo markdown/CSV now, format kept dumb enough that pitlab can ingest it later.

---

## 10. Verification protocol (when this plan is executed)

- **Phase 1:** `f1-eval models` regenerates every headline metric in the current model_configs within documented tolerance from frozen artifacts + HF data on a clean machine; the registry report is committed and diffable; the known pit-coverage 0.7047 appears in the calibration report (the harness must "find" the already-known breakage, same retro-validation trick as 2026-reg Phase 1).
- **Phase 2:** every threshold and every aggregate feature has a written verdict with notebook-cell evidence; any "contaminated" verdict has a corrected val-only number in registry v2; the tire-deg lag rule is verified in the production path or filed as a bug issue (issue-first per house rules).
- **Phase 3:** the convergence report either confirms or refutes the docstring claim at `:92-93` with standard errors; the routing report includes the activation confusion analysis over at least 5 races; the config/code threshold divergence (E-06) is resolved in one direction with a data-backed rationale.
- **Phase 4:** per-rail conformance rates reported with LLM model+version and n; the two code-level guards have hard tests that fail on regression; the R-7 boundary leakage test runs in the hermetic CI tier; every §7 catalog row is either green or an open catalogued issue.
- **Phase 5:** the full no-LLM season replay completes without unexplained error frames; ablation tables compile in the paper's LaTeX; every architecture claim in the paper's methodology maps to at least one table or report artifact; the baseline snapshot is tagged and reproducible for the TFM.
