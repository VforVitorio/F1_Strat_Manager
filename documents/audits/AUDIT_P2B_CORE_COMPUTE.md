# AUDIT P2b — Core compute & inference efficiency (CLI + Arcade in-process engine)

> **Auditor**: Fable 5 · **Date**: 2026-07-04 · **Mode**: read-only, decision-grade, NO code.
> **Scope**: the shared strategy engine that CLI (`scripts/run_simulation_cli.py`) and Arcade
> (`src/arcade/strategy.py` → `strategy_pipeline.py`) run **in-process**: the 6 sub-agents' per-lap
> inference, orchestrator N31 (MoE routing + 500-sample Monte Carlo + LLM synthesis),
> `RaceStateManager.get_lap_state`, and `arcade/strategy_pipeline.py`.
> **Out of scope**: boot/model-download/Whisper warmup (audit P2), FastAPI HTTP layer (P1), Arcade UI (P3), CLI UX (P4).
> **Constraints honored**: `src/agents/` internals UNTOUCHABLE (additive entry points / wrappers / caches only);
> `scripts/run_simulation_cli.py` UNTOUCHABLE (duplicate-and-improve); LLM = OpenAI / LM Studio, never Anthropic.
> **Caveat**: all wall-clock figures are **engineering estimates** pending Phase-0 instrumentation. Anchors used:
> the in-repo comment "one LLM call per lap, ~5-10 s" (`src/arcade/strategy.py:328`), the N24 NLP latency
> measurement (47.8 ms mean / 59.4 ms P95 GPU per message), and the code paths read line-by-line below.

---

## 1. Executive summary

The per-lap cost of a running sim is **dominated by LLM round-trips, not by ML inference and not by
the 500-sample Monte Carlo**. The MC layer — the component the backlog flags by name — costs
**~2-5 ms per lap (≈0.01% of the budget)**; it is a non-problem. The real budget:

- Every always-on sub-agent except N25 makes at least one LLM call per lap; N26/N27 are full ReAct
  loops (~3 LLM turns each), N28 up to 4 turns, N30 2-3 turns, N29 and N31 one structured call each.
  A quiet lap in LLM mode issues **~7-8 LLM turns through the orchestrator alone**.
- The CLI **doubles** that: it runs all four always-on agents once for its detail panel
  (`_probe_core_agents`, `run_simulation_cli.py:1961`) and then again inside
  `run_strategy_orchestrator_from_state` (line 1964). ~**40-45% of every LLM-mode lap is duplicate work**,
  acknowledged in the probe's own docstring (lines 489-495).
- The documented fast path, `--no-llm`, is **currently broken**: `_run_no_llm` unpacks 2 values from
  `_run_conditional_agents` (CLI line 1508) which has returned a 3-tuple since commit `bfe5b46`
  (2026-05-09, `strategy_orchestrator.py:1165`). Every no-LLM lap raises
  `ValueError: too many values to unpack` and lands as an `[ERROR]` row.

Estimated steady-state lap, LLM mode on LM Studio: **~17-50 s quiet lap, ~25-70 s on N28/N30-activated
laps** (~15 LLM turns and ~20 with conditionals). ML-only compute is **~0.3-0.7 s/lap**. The path to a
fast sim is therefore: (1) stop running agents twice, (2) cut LLM turns per agent from 3-4 to 1 (or 0),
(3) cache the 3-question RAG, (4) skip the radio LLM on silent laps — all achievable through **one shared
additive engine module** that also retires the known `arcade/strategy_pipeline.py` ↔
`agents/strategy_orchestrator.py` duplication.

Target end-state (same models, same outputs modulo reasoning prose): **quiet lap ~4-8 s in LLM mode,
~0.3-0.7 s in no-LLM mode.**

---

## 2. Per-lap latency budget (steady state, LM Studio local, estimates)

### 2.1 CLI, LLM mode, quiet lap (no N28/N30 activation) — current code

| # | Step | Where | LLM turns | Est. wall | Share |
|---|---|---|---|---|---|
| 1 | Probe N25 pace (XGBoost, no LLM) | CLI `_probe_core_agents` → `pace_agent.run` | 0 | ~5 ms | ~0% |
| 2 | Probe N26 tire (ReAct: 2 tools, sequential) | CLI probe → `tire_agent._run_core:1162` | ~3 | 3-9 s | ~20% |
| 3 | Probe N27 situation (ReAct: 2 tools) | CLI probe → `race_situation_agent._run_core:1145` | ~3 | 3-9 s | ~20% |
| 4 | Probe N29 radio (structured synthesis, runs even with 0 messages) | CLI probe → `radio_agent.run_radio_agent:996` | 1 | 1-3 s | ~7% |
| 5 | Orchestrator N25 ∥ N27 (2-thread pool) | `strategy_orchestrator._run_always_on_agents_from_state:1072` | ~3 | 3-9 s | ~20% |
| 6 | Orchestrator N26 (sequential after pool) | same, line 1079 | ~3 | 3-9 s | ~20% |
| 7 | Orchestrator N29 (sequential) | same, line 1080 | 1 | 1-3 s | ~7% |
| 8 | MoE routing (`_decide_agents_to_call`) | pure Python set logic | 0 | <1 ms | 0% |
| 9 | **Monte Carlo, 500 × 4 candidates** | `_run_mc_simulation:694-698` (Python loop) | 0 | **2-5 ms** | **~0.01%** |
| 10 | N31 LLM synthesis (~2.3 k-token prompt + strict schema w/ contingencies) | `strategy_orchestrator:1299/1417` | 1 | 3-8 s | ~15% |
| 11 | DataFrame hygiene: ≥4 full copies of the 22,760-row season parquet + timedelta conversions + stint feature rebuilds | tire:1087, radio:1045, situation `_ensure_timedelta_laps:519`, tools | 0 | 0.2-0.6 s | ~1.5% |
| 12 | `RaceStateManager.get_lap_state` (masks + rival `iterrows`) | `race_state_manager.py:338` | 0 | 2-5 ms | ~0% |
| 13 | Rich rendering / TCP broadcast | CLI / arcade | 0 | ~10-30 ms | ~0% |
| | **Total quiet lap** | | **~15** | **~17-50 s** | 100% |

**Activated lap adds**: N28 pit ReAct (3 tools → up to 4 turns, 4-12 s) + N30 RAG (ReAct 2-3 turns
**plus a second direct retrieval** — `rag_agent.py:193-199`, documented as intentional — with BGE-M3
query embedding paid twice, 3-10 s) → **~25-70 s/lap**. Under confirmed SC, routing forces N28+N30 every
lap (`_decide_agents_to_call:533`), so SC phases run at the slowest cadence exactly when the user wants
responsiveness.

### 2.2 Arcade, LLM mode

Same engine minus the probe duplication (its `run_strategy_pipeline` returns `(rec, agent_outputs)` in a
single pass): rows 5-13 only → **~8-30 s quiet lap**, consistent with the in-repo "~5-10 s per LLM call"
comment. Arcade already has two good cost-control mechanisms the CLI lacks: it **blocks the agent thread
until the replay reaches the lap** (`_wait_for_arcade`) and **skips stale laps on seek**
(`_should_skip_stale`, saves "~17 LLM calls" on a V2→V20 seek per its docstring).

### 2.3 CLI, `--no-llm` mode — intended vs actual

| | Intended | Actual today |
|---|---|---|
| Sub-agent ML inference (XGBoost + TCN 1+50 forwards + LightGBM + NLP) | ~0.2-0.5 s | runs, then… |
| LLM attempts | none | **every agent still constructs a client and attempts the call**; with LM Studio down each ReAct invoke pays openai-client default retries (2 retries + backoff) ≈ **1.5-2 s of sleep per agent, ~5-8 s/lap of pure retry backoff** |
| Conditional stage | pit=None fallback | **crashes**: 2-value unpack of a 3-tuple (`run_simulation_cli.py:1508` vs `strategy_orchestrator.py:1165`) → per-lap `[ERROR]` row |
| Semantics | deterministic, offline | **if LM Studio happens to be up, "--no-llm" silently makes real LLM calls** (attempt-and-catch design, `_is_llm_unavailable:429`) |

The README/CLAUDE.md-recommended fast invocation (`f1-sim Budapest NOR McLaren --no-real-radios --no-llm`)
exercises exactly this broken path.

---

## 3. The Monte Carlo question, answered directly

The backlog asks whether the 500-sample MC "can be vectorized / reduced / cached". Findings:

- **Cost**: 4 strategies × 500 draws = 2,000 calls to the scalar `simulate_lap_window` inside a Python
  list comprehension (`strategy_orchestrator.py:694-698`), plus 4×500 RNG draws. **~2-5 ms/lap.** It is
  4-5 orders of magnitude below the LLM cost; **n_sim=500 is not a lever and should not be reduced**
  (the config docstring's variance argument — mean variance <0.01 position units — holds).
- **Vectorizable?** Yes, trivially (branch-free `np.where`/`np.select` over the sample vectors would take
  it to ~0.1 ms), and the `pace_s` vector is drawn and never used (`:685`, `noqa: F841` — 500 wasted
  normal draws). Worth doing only as hygiene when the engine module is built. **P3.**
- **Cacheable?** It is already deterministic per inputs (fixed `seed=42`, `:637`), so identical sub-agent
  outputs reproduce identical scores; caching would save ~3 ms and add complexity. **Not worth it.**
- **The real "MC" cost is elsewhere**: the tire agent's MC-Dropout runs **50 sequential batch-1 TCN
  forwards in a Python loop** (`tire_agent.py:906-909`). Repeating the input 50× in the batch dimension
  gives per-sample dropout masks in **one** forward → est. 50-150 ms/lap saved (×2 with the CLI probe
  duplication). **P3** (real, but small next to LLM turns).

**Verdict: the MC layer is a myth as a bottleneck. Keep 500 samples. Spend the effort on LLM turns.**

---

## 4. Findings (P0 → P3)

### P0 — broken or dominant, high-certainty fixes

**F1 · CLI runs all four always-on agents twice per lap (probe + orchestrator).**
`run_simulation_cli.py:1961-1964`: `_probe_core_agents` (N25/N26/N27/N29 sequential, incl. their LLM
chains) feeds the detail panel, then `run_strategy_orchestrator_from_state` re-runs the same four because
the orchestrator returns only the final `StrategyRecommendation`. The probe docstring (489-495)
acknowledges the duplication (it's why N28/N30 are *not* probed). ~40-45% of every LLM-mode lap is
duplicate compute — including duplicate TCN/LightGBM/NLP inference and duplicate full-DF copies.
**Remedy**: a shared *verbose* entry point that returns `(recommendation, agent_outputs)` in one pass —
exactly the contract `arcade/strategy_pipeline.run_strategy_pipeline` already implements. Consumed via a
**duplicated** CLI (PMV stays frozen). Expected win: **−40-45% LLM-mode lap time**, zero fidelity change.

**F2 · `--no-llm` fast path is broken (3-tuple vs 2-tuple).**
Since `bfe5b46` (2026-05-09) `_run_conditional_agents` returns `(pit_out, regulation_context, rag_dict)`
(`strategy_orchestrator.py:1165`); the CLI still unpacks two (`run_simulation_cli.py:1508`). ValueError
isn't LLM-related → re-raised → every no-LLM lap prints `[ERROR]`. This kills the only fast demo mode and
the CI-friendly smoke path. **Remedy**: per the standing repo rule, **file the bug as a GitHub issue
first** (symptom · cause · fix · where), then either (a) Víctor sanctions a 1-line unpack fix on the PMV,
or (b) the fix lands in the Phase-1 duplicate that delegates to the shared engine. The audit recommends
(b) with (a) as an interim hotfix **only if Víctor explicitly approves touching the PMV**.

### P1 — major latency levers

**F3 · ReAct tool loops inflate LLM turns 3-4× per sub-agent for zero information gain.**
N26/N27/N28 wrap deterministic model calls in ReAct agents whose tool *arguments are already known before
the LLM sees the prompt* (driver, compound_id, tyre_life are injected into the user message —
`tire_agent._run_core:1158`, `pit_strategy_agent._run_core` prompt build). With
`parallel_tool_calls=False` (LM Studio), each agent costs ~3-4 LLM turns to (1) decide to call tools it
was told to call, (2) echo tool output, (3) summarize. Numeric outputs are already parsed from
ToolMessages, *not* from the LLM (`_parse_tool_outputs`) — the LLM only contributes the `reasoning`
string. N29 already demonstrates the right shape: run models deterministically first, then **one**
structured synthesis call (`radio_agent.py:976-999`).
**Remedy (additive)**: a "direct mode" in the shared engine that calls the same instance methods the
tools wrap (the import-private pattern is already established by `strategy_pipeline.py` docstring and the
backend's `_run_no_llm_path`), then either (i) one structured synthesis call per agent for the reasoning
string, or (ii) **zero** sub-agent LLM calls, letting N31 synthesize from the numeric blocks (the N31
prompt already embeds every number verbatim; sub-agent reasoning strings are *re-summarized* by N31
anyway). Expected: quiet lap from ~7-8 orchestrator turns to **2-5 turns (option i) or 1 turn (option ii)**.
Fidelity note: option (ii) loses per-agent prose in the UI panels — offer it as a `fast` profile, keep
`rich` profile with option (i).

**F4 · N30 RAG is 100% cacheable and currently double-retrieves.**
The orchestrator only ever asks **three canned questions** (`_build_rag_question:717-738` — SC procedure,
compound-change restriction parametrized by compound, generic dry-compound rule). Answers are
race-invariant. Yet each activation runs a ReAct loop + Qdrant retrieval + a **second** direct
`retriever.query` for typed chunks (`rag_agent.py:193-199`). **Remedy**: additive
`cached_run_rag_agent(question)` (per-process `lru_cache` keyed by the question string) used by the
engine; optionally pre-warm the 2-3 questions at boot. Expected: N30 cost → **~0 after first activation**
(and the first activation can be moved out of the race into warmup).

**F5 · N29 pays an LLM synthesis on silent laps.**
`run_radio_agent` invokes the structured LLM even when `radio_msgs` and `rcm_events` are both empty
(stage 3 unconditional, `radio_agent.py:987-999`); alerts (the load-bearing output) are built
deterministically *before* the LLM. Most laps are silent. **Remedy**: engine-level guard — empty inputs →
deterministic `RadioOutput` with canned reasoning, no LLM call. Saves 1-3 s on the majority of laps.

**F6 · Always-on phase is mostly sequential.**
`_run_always_on_agents_from_state:1071-1081` parallelizes only N25∥N27; N26 and N29 run after,
serialized on a "PyTorch/MLX thread-safety" rationale. The dominant wait is LLM I/O, not torch. The four
agents hold **distinct** model objects (TCN vs transformers vs LightGBM), so cross-agent inference races
don't exist; the one genuinely racy pattern is `model.train()`/`.eval()` toggling *within* the tire agent,
which stays single-threaded inside its own call. **Remedy**: engine runs all four in a 4-worker pool
(keep a config flag to fall back to the current split). Expected: always-on phase = max(agents) instead of
pair-sum ≈ **−30-50% of that phase**. **Caveat to verify in Phase 0**: a single LM Studio instance may
serialize concurrent completions server-side — the win is certain on OpenAI, partial on LM Studio.

### P2 — moderate levers / structural

**F7 · Full-season DataFrame passed and copied per lap.**
`laps_featured_2025.parquet` = 22,760 rows × 53 cols ≈ 14 MB, 24 GPs. CLI (`:1614`) and Arcade
(`strategy.py:487-492`) pass it **unfiltered** into every agent, which then copy it per lap
(tire `:1087`, radio `:1045`, pit `:931`, situation via `_ensure_timedelta_laps:519` — copy + timedelta
conversion + column backfills each call). ×2 under F1's double-run. Est. 0.2-0.6 s/lap of pure churn, and
every internal filter scans 24 GPs instead of 1. **Correctness flag to verify (separate issue)**: tire
`_get_driver_stint:811-824` filters by Driver+Compound+TyreLife only — on a season-wide frame this can
mix stints from *different races*; situation's `fastest_lap_s` (`:1098`) is the season-wide minimum, not
the race's. **Remedy**: filter to the current GP once at engine construction, normalize dtypes
(`lap_time_s`, Timedelta, TrackStatus) once, hand every lap the same immutable frame. Win: ~0.2-0.5 s/lap
+ removes the correctness ambiguity.

**F8 · "no-LLM" is attempt-and-catch, not offline.**
Every agent constructs `ChatOpenAI` and attempts the call; unavailability is detected by exception
matching (`_is_llm_unavailable:429-447`). Consequences: (a) ~1.5-2 s retry backoff per agent per lap when
the backend is down (openai default `max_retries=2`; no override anywhere in `src/agents/`), (b) if LM
Studio is up, `--no-llm` silently becomes LLM mode. **Remedy**: the engine's no-LLM profile never
constructs clients — it calls model predictors directly and composes outputs deterministically (the
guard-rail logic in `_run_no_llm` already encodes the deterministic decision policy to reuse). Win:
no-LLM lap **~0.3-0.7 s** and true determinism.

**F9 · Warmup gap: steady state starts late.**
`_prewarm_agents` (CLI `:448-481`) and Arcade `_warmup_models` materialize model singletons, but
explicitly **not** the LLM clients/ReAct graphs ("they need a live LLM connection", `:459`), not the
Qdrant client + BGE-M3 embedder (first N30 activation — often mid-race under SC — pays retriever
construction + embedder load, est. 5-20 s spike at the worst moment), and LM Studio itself may cold-load
the model on the first call (10-60 s). **Remedy**: optional engine warmup stage — build LLM clients +
graphs, fire one 1-token ping, call `get_retriever()` and pre-answer the 3 canned RAG questions (ties into
F4). Boundary note: warmup *mechanics* belong to audit P2; what P2b owns is the definition of "warm" for
the compute path (clients + graphs + retriever + first-token, not just weights).

**F10 · Arcade ↔ orchestrator pipeline duplication (known #1 debt) — resolve INTO the shared engine.**
`arcade/strategy_pipeline.py` is a body-copy of `run_strategy_orchestrator_from_state` importing seven
private helpers, with a "mirror the change here" comment (`:19`) that F2 proves is a real failure mode
(the same 3-tuple change that broke the CLI was mirrored here but not there). The Arcade copy exists
*only* because the orchestrator's public API discards `agent_outputs` — the same root cause as F1.
**Remedy**: one additive module — proposed home `src/strategy/inference/engine.py` (the package already
exists and is nearly empty) — exposing `run_lap(race_state, laps_df, lap_state, *, profile, return_agent_outputs=True) → (rec, agent_outputs, stage_timings)`.
Arcade's `strategy_pipeline.py` becomes a thin delegate (Arcade is editable); the CLI duplicate (P4's
deliverable) and the backend simulator's `_run_no_llm_path` (P1's domain) consume the same function.
One fast path, three surfaces, duplication retired.

**F11 · N31 synthesizes every lap even when nothing decision-relevant changed.**
On cruise laps the recommendation is STAY_OUT with near-identical numbers, yet the full ~2.3 k-token
prompt + strict-schema generation (contingencies, key_risks — the slowest structured fields on local
models) runs every lap. **Remedy (product-level, Arcade-first)**: event-triggered synthesis — re-run the
N31 LLM only when (routing set changed) ∨ (MC argmax changed) ∨ (tire warning tier changed) ∨ (alerts
non-empty) ∨ (K laps elapsed, K≈3-5); between triggers, reuse the previous recommendation with refreshed
`scenario_scores` (attached in code, not by the LLM — already the design). Fidelity trade-off is explicit
and configurable; MC + models still run every lap so the panels stay live. Win: LLM cost on cruise
sequences → ~1/K.

### P3 — micro / hygiene (do inside the engine work, don't schedule separately)

**F12 · TCN MC-Dropout: 50 sequential batch-1 forwards → 1 batched forward** (`tire_agent.py:906-909`).
~50-150 ms/lap. Via the import-private pattern in the engine's direct mode (no agent edits).
**F13 · Tire stint features rebuilt 4×/lap** (each of 2 tools calls `_build_stint_tensor` *and* a second
`_build_stint_features` for `deg_rate` — `:865/872`, `:902/920`; ×2 under F1). Memoize per
`(driver, compound_id, tyre_life)` within the lap. ~50-200 ms/lap.
**F14 · MC vectorization + drop unused `pace_s` draws** (`:685`). ~3 ms; hygiene only; keep n_sim=500.
**F15 · `get_lap_state` micro-costs** (per-lap boolean masks over the full frame + rival `iterrows`,
`race_state_manager.py:237-281`): pre-group by LapNumber at construction. ~2-4 ms/lap; do only if the
frame stays large after F7.
**F16 · 2-worker `ThreadPoolExecutor` constructed per lap** (`:1072`): negligible (~ms); folds into F6.

### What NOT to optimize (explicit non-goals)
- **n_sim=500** — statistically justified, costs milliseconds. Leave it.
- **RaceStateManager preprocessing** — one-shot at construction, <50 ms, correct design.
- **Rich rendering / TCP broadcast** — tens of ms, invisible next to LLM turns.
- **Whisper / NLP model loading** — boot-time, owned by audit P2 (transcription is JSON-cached; per-lap
  radio lookup is a dict hit).

---

## 5. Warmup vs steady state (compute-path view)

| Stage | Warm today? | First-payment moment | Owner |
|---|---|---|---|
| XGBoost / LightGBM / TCN bundles / pit models | ✅ prewarmed (CLI `:448`, Arcade `_warmup_models`) | boot | P2 |
| NLP (sentiment/intent/NER) | ✅ eager on import (`run_simulation_cli.py:113`) | boot | P2 |
| Whisper + radio corpus | ✅ eager, JSON-cached | boot | P2 |
| LLM clients + ReAct graph construction | ❌ lazy singletons | lap 1 (spike) | **P2b** F9 |
| LM Studio model residency (server-side) | ❌ | first LLM call (10-60 s worst case) | **P2b** F9 |
| Qdrant client + BGE-M3 embedder | ❌ | first N30 activation — typically mid-race under SC | **P2b** F4/F9 |
| RAG answers (3 canned questions) | ❌ | every N30 activation | **P2b** F4 |

## 6. Cross-lap batching opportunities

| Opportunity | Verdict |
|---|---|
| Batch LLM decisions across laps | ✗ Not viable live — decisions are causally sequential. The valid form is **cadence** (F11). |
| Reuse previous recommendation between triggers | ✓ F11 — the main cross-lap lever. |
| RAG answer reuse across laps/races | ✓ F4 — the purest cache in the system. |
| Tire stint features incremental (lap N extends N-1) | ✓ Within-lap memo first (F13); cross-lap incremental only if profiling still shows it. |
| NLP batching across a lap's radio messages | Marginal (~50 ms/msg, msgs/lap ≤ 3). Skip. |
| MC result reuse | ✗ Deterministic already; costs ms. Skip. |

## 7. The shared fast path (the audit's central recommendation)

**One additive module, three consumers.** `src/strategy/inference/engine.py`:

- **API**: `run_lap(race_state, laps_df, lap_state, *, profile, return_agent_outputs=True) → (StrategyRecommendation, agent_outputs, stage_timings)`.
- **Profiles**: `rich` (current fidelity: ReAct sub-agents or 1-call synthesis per agent, N31 every lap),
  `fast` (direct-mode tools, numbers-only sub-agents, N31 event-triggered), `no-llm` (zero clients,
  deterministic policy = today's `_run_no_llm` guard-rails, fixed 3-tuple).
- **Inside**: GP-filtered normalized frame built once (F7); RAG cache (F4); silent-radio guard (F5);
  4-way parallel always-on with fallback flag (F6); per-stage `perf_counter` timings (F16/Phase 0);
  direct-mode tool execution via the established import-private pattern (F3).
- **Consumers**: Arcade `strategy_pipeline.py` → thin delegate (kills F10); **CLI duplicate** (P4's
  duplicate-and-improve calls the engine instead of probe+orchestrator, killing F1 and F2); backend
  simulator `_run_no_llm_path` (P1) migrates when convenient.
- **Untouchability**: `src/agents/` unmodified — the engine imports entry points and (where already
  precedented) private helpers; new behavior lives entirely in the new module.

## 8. Phased plan (chunkable → issues/PRs; S/M/L effort)

### Phase 0 — Measure & unblock (S) — 1 short sprint
| Chunk | What | Effort | Exit criterion |
|---|---|---|---|
| 0.1 | **File the F2 bug issue** (no-LLM 3-tuple crash; symptom·cause·fix·where). Decide hotfix policy with Víctor (1-line sanctioned PMV fix vs fix-in-duplicate). | S | Issue open; decision recorded |
| 0.2 | Timing harness: thin additive wrapper (or engine skeleton) that times each stage (per-agent, MC, N31, DF ops) and logs per lap; run Budapest/NOR baseline in LLM + no-LLM(-post-fix) modes | S | Measured budget table replaces §2 estimates |
| 0.3 | Verify F7 correctness flag (cross-GP stint mixing; season-wide `fastest_lap_s`) with a targeted probe; file as separate issue if confirmed | S | Confirmed/refuted + issue |
| 0.4 | Verify LM Studio concurrency behavior (2 parallel completions) to size F6 | S | Go/no-go for 4-way parallel on lmstudio profile |

### Phase 1 — Shared engine fast path (M) — the structural sprint
| Chunk | What | Effort | Expected win |
|---|---|---|---|
| 1.1 | `src/strategy/inference/engine.py` with `run_lap` + profiles + stage timings; Arcade delegates; CLI duplicate consumes it | M | CLI LLM lap **−40-45%** (F1); duplication retired (F10) |
| 1.2 | True no-LLM profile (no clients; 3-tuple handled; guard-rails preserved) | S | no-LLM lap ~0.3-0.7 s, deterministic (F2/F8) |
| 1.3 | GP-filter + dtype-normalize frame once at engine construction | S | ~0.2-0.5 s/lap + correctness (F7) |
| 1.4 | RAG `lru_cache` + optional pre-answer of the 3 canned questions at warmup | S | N30 laps −3-10 s after first hit (F4) |
| 1.5 | Silent-lap radio guard | S | −1-3 s on most laps (F5) |

### Phase 2 — LLM turn diet (M/L) — the latency sprint
| Chunk | What | Effort | Expected win |
|---|---|---|---|
| 2.1 | Direct-mode sub-agents: deterministic tool execution + single structured synthesis (rich) / numbers-only (fast) | L | N26/N27: 3→1 turns; N28: 4→1; quiet lap → **~4-8 s** |
| 2.2 | 4-way parallel always-on (behind flag; per Phase-0.4 result) | M | always-on phase → max(agents) |
| 2.3 | Warmup completion: LLM ping + retriever + graphs (coordinate with P2 audit) | S | lap-1 and first-SC spikes removed (F9) |

### Phase 3 — Cadence & micro (S-M) — the fidelity-dial sprint
| Chunk | What | Effort | Expected win |
|---|---|---|---|
| 3.1 | Event-triggered N31 synthesis (Arcade first; config K + trigger set) | M | cruise sequences → ~1/K LLM cost (F11) |
| 3.2 | TCN MC batched forward + within-lap stint-feature memo | S | −0.1-0.3 s/lap (F12/F13) |
| 3.3 | MC vectorize + drop unused `pace_s`; RSM pre-group if still relevant | S | hygiene (F14/F15) |

**Dependency chain**: 0.1-0.4 → 1.1 (everything hangs off the engine) → {1.2-1.5 parallelizable} →
2.x → 3.x. Every chunk is a single-concern PR candidate per the repo's issue→PR→sprint rhythm.

## 9. Risk register

| Risk | Mitigation |
|---|---|
| Engine drifts from orchestrator (new silent copy) | Engine *imports* orchestrator helpers (no body copies); contract test asserting engine(rec) == orchestrator(rec) on a fixture lap in both modes |
| Direct mode changes reasoning fidelity | Ship as profiles; `rich` stays default in CLI demo contexts; A/B one GP replay and diff actions/decisions lap-by-lap (actions must match; prose may differ) |
| Parallel LLM calls serialize on LM Studio | Phase-0.4 gate; profile flag keeps pairwise split on lmstudio |
| torch train()/eval() races under 4-way parallel | Each agent owns distinct model objects; tire's toggle is confined to its own thread; add a regression test with concurrent tire+radio calls |
| PMV untouchability | All CLI-side changes land in the P4 duplicate; the only PMV question is the sanctioned F2 hotfix, which is Víctor's call via the filed issue |

## 10. Alignment with sibling audits

- **P2 (loading)**: owns boot mechanics; P2b hands it the "warm = clients + graphs + retriever +
  first token" definition (F9) and the RAG pre-answer hook (F4).
- **P3 (Arcade)**: inherits F10's resolution (delegate module) — its "#1 heavy" item closes here.
- **P4 (CLI)**: its duplicate-and-improve plan should *start from the engine* (F1/F2 are the first two
  things the duplicate fixes by construction).
- **P1 (backend)**: `_run_no_llm_path` migrates to the engine's no-llm profile when P1 executes.

---

### Appendix A — Evidence index (file:line)

| Claim | Evidence |
|---|---|
| CLI double-run of always-on agents | `scripts/run_simulation_cli.py:1961-1964`; ack in `:489-495` |
| no-LLM 3-tuple crash | `scripts/run_simulation_cli.py:1508` vs `src/agents/strategy_orchestrator.py:1165`; commit `bfe5b46` 2026-05-09 |
| MC = Python loop, 500×4, seed 42, unused pace draws | `strategy_orchestrator.py:637, 685, 694-698` |
| MoE routing forces N28+N30 under SC | `strategy_orchestrator.py:527-537` |
| 2-thread pool only for N25∥N27; N26/N29 serial | `strategy_orchestrator.py:1049-1082` |
| Tire ReAct + 2 tools; features rebuilt twice per tool; n_mc=50 loop | `tire_agent.py:1157-1163, 865/872, 902/920, 906-909; n_mc default :190` |
| Situation ReAct + RCM override | `race_situation_agent.py:1144-1173` |
| Pit ReAct, 3 tools, per-lap `laps_df.copy()+astype` | `pit_strategy_agent.py:931-933, 996` |
| Radio: NLP-first, LLM synthesis unconditional, per-lap `LAPS = laps_df.copy()` | `radio_agent.py:976-999, 1043-1051` |
| RAG double retrieval (documented), 3 canned questions | `rag_agent.py:193-199`; `strategy_orchestrator.py:717-738` |
| Arcade single-pass verbose pipeline + private-helper precedent | `src/arcade/strategy_pipeline.py:11-20, 42-121` |
| Arcade lap gate + stale-skip + "~5-10 s" anchor | `src/arcade/strategy.py:228-258, 328` |
| Season-wide frame: 22,760×53 ≈ 14 MB, 24 GPs; loaded unfiltered | measured from `data/processed/laps_featured_2025.parquet`; CLI `:1614`; Arcade `strategy.py:487-492` |
| attempt-and-catch no-LLM semantics | `run_simulation_cli.py:386-447, 1435-1441` |
| Prewarm excludes LLM/ReAct | `run_simulation_cli.py:448-481` (esp. `:459`) |
| RSM per-lap masks + rival iterrows | `src/simulation/race_state_manager.py:237-281, 338-374` |
| NLP latency anchor 47.8 ms mean | N24 measurement (project memory) |
