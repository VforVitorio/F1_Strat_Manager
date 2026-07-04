# AUDIT — Testing & QA strategy (whole system)

> **Auditor:** Fable 5 (senior test architect) · **Date:** 2026-07-04 · **Mode:** read-only, decision-grade, NO code.
> **Scope:** testability of the full system: multi-agent strategy engine (`src/agents/`, N25-N31), the 3 surfaces (CLI `scripts/`, Arcade `src/arcade/`, Streamlit+FastAPI `src/telemetry/`), backend (chat SSE, FastMCP, voice), NLP (`src/nlp/`), RAG (`src/rag/`), and the incoming React SPA (`MIGRATION_PLAN.md`).
> **Constraints honored:** `src/agents/` internals and `scripts/run_simulation_cli.py` are UNTOUCHABLE (tested via public entry points and as black boxes; importing them in tests is fine and already precedented); LLM = OpenAI / LM Studio, never Anthropic; **no test ever calls a real model**; no code in this document.
> **Inputs read:** every test file in both repos, both CI workflows, `AUDIT_P1_BACKEND.md`, `AUDIT_P2_LOADING.md`, `AUDIT_P2B_CORE_COMPUTE.md`, `MIGRATION_PLAN.md` + `SPRINTS_AND_ISSUES.md`, the migration dossier (`SCREENS.md` + 37 `live_*` PNGs), parent + submodule `pyproject.toml`/`pytest.ini`/conftests.

---

## 0. Executive summary

The system has **~100 test functions on paper and near-zero effective automated verification in practice**. The two structural problems:

1. **The only substantive unit tests in the codebase (45 hermetic tests in the submodule: chat engine + MCP bridge) run in NO CI anywhere.** The submodule's CI is a single ruff-critical lint job; the parent's CI never checks out the submodule. They pass only when someone runs them locally by hand.
2. **Everything that touches the strategy engine, data, or an LLM is skip-guarded on assets CI doesn't have** (7-8 GB HF dataset, model weights, the submodule checkout). On a GitHub runner the parent suite reduces to: dependency imports + directory-structure asserts + Pydantic schema defaults. **CI has been green through every bug the sibling audits found** - the `--no-llm` crash (broken since 2026-05-09, 8 weeks, in the *documented* fast command), the `/voice/synthesize` +175% rate bug, the CLI double-run, the repo-root walker that breaks every bare-metal backend.

None of those bugs needed sophisticated tests. Each maps to a **unit test of one pure function or one contract assertion** (§3). The gap is not test-writing skill - the existing tests are well-written - it is (a) no CI wiring, (b) no committed fixtures, and (c) **no stub LLM**, so nothing between "pure helper unit test" and "manual run with LM Studio live" can exist.

The single highest-leverage enabler discovered: **the entire system already speaks the OpenAI wire protocol at `localhost:1234`** (backend `llm_service.py:26-28` via `LM_STUDIO_HOST`; every agent's `ChatOpenAI(base_url='http://localhost:1234/v1')`). One in-process **FakeOpenAI fixture server** on that port stubs the chat engine, all six sub-agents, and orchestrator N31 **without touching a single untouchable file**. That one fixture unlocks the whole integration tier.

Plan: 5 phases (§10). Phase 0 is one afternoon (CI wiring + 3 trivial unit tests that retro-cover the found bugs). Phase 1 builds the fixture layer (FakeOpenAI, recorded SSE transcripts, mini race parquet, canned agent outputs). Phases 2-4 build the pyramid up through Playwright e2e + visual regression seeded from the 37 dossier PNGs, timed to the SPA migration sprints.

---

## 1. Current state inventory

### 1.1 Parent repo (`F1_Strat_Manager/tests/`, pytest config in `pyproject.toml`, testpaths=["tests"])

| File | Tests | What it covers | Runs in CI? |
|---|---|---|---|
| `test_dep_imports.py` | 20 | 3-tier dependency smoke (behavioural API surface, import-only, known-breakage compat). Good Dependabot gate. | ✅ fully |
| `test_smoke.py` | 6 | Directory structure, `RaceStateManager` import, Melbourne lap_state golden check, Qatar 2025 V7 SC-override regression | ⚠️ 4 of 6 skip (data-gated) |
| `test_agents.py` | 22 | Agent module importability, entry-point existence, output dataclass fields, RCM/SC-override pure helpers, `_decide_agents_to_call` routing, backend request schemas | ❌ effectively all skip (`_skip_no_models`: no `data/models/`; `_skip_no_backend`: submodule not checked out - parent CI uses plain `actions/checkout@v6`, no `submodules:`) |
| `test_simulation.py` | 4 | `simulate_race` generator contract (start→lap→summary), `/strategy/simulate` SSE via TestClient, `SimulateRequest` defaults + provider pattern (rejects "anthropic" ✅) | ❌ 2 data-gated skip; 2 schema tests skip (no submodule) |
| `test_arcade_dashboard_imports.py` | 3 (+param) | PySide6 dashboard import smoke | ⚠️ runs if PySide6 wheel installs on runner |

**CI (parent):** jobs `test` / `lint` / `typecheck` (mypy scoped to `src/rag/` only). `uv run pytest -v`, no coverage flag (pytest-cov is a declared dev dep, never wired), no xdist, no tiers/markers used (`unit`/`integration`/`slow` markers declared in the submodule's pytest.ini, unused).

### 1.2 Submodule (`F1_Telemetry_Manager` = `src/telemetry/`, own `pytest.ini`, testpaths=tests)

| File | Tests | What it covers | Runs in CI? |
|---|---|---|---|
| `tests/test_chat_engine.py` | 33 | Pure chat-engine helpers: message/text extraction, tool-call pick, display-type resolution, prefix strip, LLM payload trimming, tool-result payload, fallback summary, done-metadata | ❌ **no test job exists in the submodule CI** |
| `tests/test_mcp_bridge.py` | 12 | Schema normalization, OpenAI tool wrapping, tool-argument coercion (malformed JSON, arrays) | ❌ same |
| `backend/test_voice_api.py` | (dev script) | Manual probe, not a pytest suite; P1 F-13(e) says relocate | ❌ n/a |

**CI (submodule):** one job, `lint (python)`, ruff `--select=E9,F63,F7,F82` only. **Zero pytest execution.** The comment in the workflow itself says "broadened once the SPA lands (see #25)".

### 1.3 What a green CI actually proves today

| Claim a green build makes | True? |
|---|---|
| Dependencies import and their API surface works | ✅ (the one genuinely covered area) |
| The strategy engine produces a recommendation | ❌ never exercised in CI |
| `--no-llm` mode works | ❌ (broken 8 weeks while CI stayed green) |
| The chat SSE stream is well-formed | ❌ (45 relevant unit tests exist, run nowhere) |
| Voice endpoints behave | ❌ (zero tests; rate bug shipped) |
| Backend routes return correct status codes | ❌ (1 route of 38 has a test, and it's data-gated) |
| Data paths resolve outside Docker | ❌ (walker bug; zero path tests) |
| Arcade replays a race | ❌ (import smoke only) |
| NLP pipeline / RAG answer correctly | ❌ (zero tests) |

### 1.4 Coverage snapshot (qualitative - no cov report exists to cite)

| Area | Unit | Integration | E2E | Verdict |
|---|---|---|---|---|
| `backend/services/chatbot/` | good (45) | none | manual | best-covered module; CI-orphaned |
| `backend` other 30+ routes, voice, telemetry services, simulation service | none | 1 (data-gated) | manual | **critical gap** |
| `src/agents/` (via entry points) | field/routing asserts only | none | manual demo runs | **critical gap** |
| `src/simulation/` (RSM, replay engine) | 2 golden (data-gated) | none | — | thin |
| `scripts/` CLI (PMV + menu + pickers) | none | none | manual | **critical gap** (the PMV is the thesis artifact) |
| `src/arcade/` | import smoke | none | manual | accepted-thin (native UI), but `strategy_pipeline.py` drift unguarded |
| `src/nlp/`, `src/rag/` | none | none | notebook-era validation | gap |
| Streamlit frontend | none | none | dossier screenshots | accept (sunset by migration) |
| React SPA (incoming) | n/a yet | n/a | n/a | must be born tested (§6) |

---

## 2. Why the escaped bugs escaped (root-cause pattern)

All three named bugs share one shape: **a contract between two components changed or was wrong, and no executable artifact pinned that contract.**

1. **`--no-llm` 3-tuple** (P2b F2): `_run_conditional_agents` grew a third return element (`bfe5b46`, 2026-05-09); the Arcade mirror was updated by hand, the CLI consumer was not. Three hand-mirrored copies of the no-LLM path exist (CLI `_run_no_llm`, `arcade/strategy_pipeline.py`, backend `simulator._run_no_llm_path`) and **no test executes any of them in CI**. Worse: the one test that runs the backend no-LLM path (`test_simulate_race_emits_start_lap_summary`) explicitly *tolerates* intermediate `error` events - it is designed to be unable to catch a per-lap crash - and skips in CI anyway.
2. **Voice rate** (P1 F-12): `TTSRequest.rate` documented as wpm (default 175), consumed by `tts_service._format_rate` as a percent. A two-assert unit test of a pure function. Zero voice tests exist.
3. **CLI double-run** (P2b F1): behavioral property ("each agent runs once per lap") never asserted anywhere; the duplication was even acknowledged in a docstring.

Corollary finding: the repo-root walker bug (P1 F-1) is the same class - 5 hand-copied implementations of one path contract, no test with a temp-dir tree that has a `.git` *file* in a subdir.

**Design consequence for the target harness:** prioritize *contract tests at the seams that are hand-duplicated or hand-documented* (tuple shapes, SSE grammars, path resolution, unit conventions, provider config) over line-coverage anywhere else. And treat the P2b engine consolidation as a *testability* project: one engine = one place to test the no-LLM path instead of three.

---

## 3. The retro-test table (exact tests that would have caught each found bug)

This is the "prove the harness earns its keep" list. Each row = one small test, named by intent.

| Bug (audit ref) | The test that catches it | Tier | Effort |
|---|---|---|---|
| `--no-llm` 3-tuple crash (P2b F2) | (a) `test_no_llm_lap_produces_decision_without_error` - run `simulate_race(SimConfig(no_llm=True))` on the **committed mini race fixture** and assert **zero `error` frames** (tighten the existing tolerant assert); (b) `test_cli_no_llm_smoke` - subprocess `f1-sim <fixture-race> --no-llm --laps 5-7`, assert exit 0 and no `[ERROR]` in stdout (data tier / nightly until the mini-fixture race works for the CLI); (c) once the P2b shared engine lands: `test_engine_no_llm_profile_matches_orchestrator_actions` (the P2b §9 parity test) | integration + e2e | S |
| Voice rate +175% (P1 F-12) | `test_format_rate_default_is_not_a_speed_multiplier` - assert `_format_rate(TTSRequest().rate)` produces an Edge-TTS rate string in a sane band (e.g. −50%..+100%); plus `test_synthesize_contract_default_rate` via TestClient with Edge-TTS stubbed, asserting the rate string handed to the adapter | unit + contract | S |
| CLI double-run (P2b F1) | `test_each_agent_invoked_once_per_lap` - monkeypatch counters onto the public `run_*_from_state` entry points, drive one lap through the engine/pipeline, assert call-count == 1 per agent. Pre-engine, mark `xfail(reason="known double-run, P2b F1")` against the CLI path so the debt is *executable*, not prose | integration | S |
| Repo-root walker (P1 F-1) | `test_data_root_ignores_gitlink_file` - temp tree with `.git` **file** in `sub/` and `.git` **dir** at root; assert the resolver returns root; parametrize over the 5 walker sites (collapses to 1 site after F-1) + `test_data_root_honors_env_override` (`F1_STRAT_DATA_ROOT`) | unit | S |
| LM Studio no-timeout hang (P1 F-2a) | `test_llm_requests_always_have_timeout` - assert the effective timeout is not `None` for both provider values | unit | S |
| Hidden `year` query param (P1 F-15) | `test_strategy_posts_expose_no_year_query_param` - inspect the generated OpenAPI schema | contract | S |
| Silent-empty fallbacks / 200-with-error (P1 F-11/F-16) | per-route contract tests asserting 404/503 for unknown driver / missing parquet, non-2xx when the LLM path fails - written **test-first** against the F-11 error envelope before it lands | contract | M |
| SSE grammar drift (P1 F-19, migration risk 3) | recorded-transcript contract tests: server output must parse against the committed `.sse` fixtures; the SPA's TS parser unit-tests against the **same files** | contract | M |

---

## 4. Findings (P0 → P3)

### P0 — the harness does not exist where it matters

| ID | Finding | Evidence | Decision |
|---|---|---|---|
| **T-1** | **45 hermetic, high-quality unit tests run in no CI.** Submodule CI = lint-only; parent CI never checks out the submodule. | `src/telemetry/.github/workflows/ci.yml` (single `lint` job); parent `ci.yml` checkout without `submodules:` | Add a `test` job to the **submodule** CI running its pytest suite (needs nothing but Python + backend deps-lite; the 45 tests import only `backend.models`/`backend.services.chatbot`). Make it a required context. **One afternoon, highest ROI in this audit.** |
| **T-2** | **The strategy engine has zero behavioral tests.** Existing agent tests assert importability, signatures, and dataclass fields - never a decision. The three duplicated no-LLM consumers are all unverified; the only engine-adjacent test is data-gated AND tolerates per-lap errors. | `tests/test_agents.py` (all structural); `tests/test_simulation.py:82` docstring "Intermediate ``error`` events are tolerated" | Build the **engine behavioral suite** (§5 L2): golden-scenario tests through public entry points with the FakeOpenAI stub + mini fixture. Tighten the tolerant assert now (S). |
| **T-3** | **No stub LLM exists**, so no test can cover any LLM-touching path without a live LM Studio. This single absence explains why coverage stops at pure helpers. | agents hardcode `base_url='http://localhost:1234/v1'` (e.g. `pace_agent.py:610`); backend `llm_service.py:26-28` honors `LM_STUDIO_HOST`, port fixed 1234 | **FakeOpenAI fixture**: in-process OpenAI-compatible HTTP server bound to `127.0.0.1:1234` (works today with zero prod changes because both layers default there), serving scripted completions/tool_calls per test. Enabler PR (S, backend-local, aligns with P1 F-34): make port/base-URL env-configurable so tests don't need to own :1234. Never a real provider, never Anthropic. |
| **T-4** | **CI-green is uncorrelated with product health.** Every data/model/backend-dependent test silently skips on runners; nobody sees the delta between local-green (with data) and CI-green. CI was green through all four escaped bugs. | skip guards in `test_agents.py:24-36`, `test_simulation.py:33-50`; no skip-report step | Formalize **test tiers with markers** (`unit` / `contract` / `data` / `gpu` / `llm`); CI prints a skip summary and **fails if the `unit`+`contract` tiers collected < N tests** (guards against accidental mass-skip). `data` tier runs nightly/on-demand on Víctor's machine (workflow_dispatch + local pre-release script), not on runners. |

### P1 — must land before / with the SPA typed client and the P1 backend contract freeze

| ID | Finding | Decision |
|---|---|---|
| **T-5** | **38-route API surface has ~0 contract tests** while the migration is about to generate a typed TS client from it and P1 Phase 2 is about to break response shapes (error envelope, fallback removal). Untested breaking changes on a surface two clients depend on. | Per-router **contract suite** (TestClient, no data: parquet-backed routes tested for their 404/503 contract with fixtures/monkeypatched cache): every route ≥ 1 happy-path + 1 error-path test. Write the F-11 envelope tests **test-first**; they double as the spec. = P1 audit Phase 6 item 22, expanded. |
| **T-6** | **No committed fixtures.** Every non-trivial test gates on the 7-8 GB HF dataset. The "data not in git" rule (CLAUDE.md §0.3) has no carve-out for test fixtures, so nobody committed any. | Explicit carve-out decision (needs Víctor's OK): `tests/fixtures/` may hold **small derived artifacts** (< ~200 KB each): a truncated race parquet (one GP, ~10-15 laps, 4-6 drivers - source parquet is only 2.3 MB, a slice is KBs), canned lap_state JSONs, recorded SSE transcripts, canned agent-output JSONs, a 3-doc Qdrant mini-collection. Committed, versioned with the contracts they pin. |
| **T-7** | **SSE grammars are informally specified, in two dialects,** and the SPA's `RaceFeed` client (A5-2) will be written against guesses. The migration plan itself asks for W0 transcript fixtures (risk 3). | **Record real transcripts now** (one `/chat/tool-message-stream` run incl. a tool call + error case; one `/strategy/simulate` run) into `.sse` text fixtures. Python side: server-output-parses-as-fixture-grammar tests. TS side (W0): parser unit tests against the same files. When P1 F-19 unifies framing, regenerate fixtures in the same PR - the fixture diff IS the migration note. |
| **T-8** | **Voice: zero tests** on a service with a confirmed unit-convention bug, unenforced size caps (F-26), and a hardcoded GPU device. | Voice unit+contract mini-suite: `_format_rate` bounds, size-cap 413 contract, `/voices` shape, health degrade semantics; STT/TTS adapters stubbed (no network, no GPU). Pairs with the P1 Phase 4 fixes - again test-first. |
| **T-9** | **Path/boot resolution logic untested** (5 walker copies, `data_cache` selection logic, arcade's bypass F-11/P2). This is the highest-frequency "works in Docker, broken on metal" class. | Unit tests on temp-dir trees for the shared resolver (after P1 F-1 lands, one site) + `data_cache` unit tests with `HF_HUB` mocked (download logic is testable without downloading). |
| **T-10** | **The strategy engine's numeric spine is deterministic and never asserted**: MC seed 42, `_decide_agents_to_call` (partially tested ✅), guard rails, score formula `α·E+(1−α)·P10`. Determinism is a free oracle nobody is using. | **Golden decision tests**: canned sub-agent outputs → orchestrator MC + routing → assert exact scenario scores and argmax (pure Python, no LLM, no data). Guard-rail table tests (no-boxing-lap-1 class rules from `guard_rails.py` and `[[project_strategic_guardrails]]`). These freeze the thesis-defended behavior against refactors - critical since P2b is about to rebuild this into an engine module. |

### P2 — should build, schedulable

| ID | Finding | Decision |
|---|---|---|
| **T-11** | NLP pipeline untested. Model inference needs GPU (data/gpu tier), but the RCM parser, alert composition, and pipeline glue are pure. | Unit tests for RCM parsing + `analyze_radio_message` composition with model calls stubbed; one `data`-tier test running the real pipeline on 2-3 committed audio-transcript pairs asserting label stability (guards model-file drift). |
| **T-12** | RAG untested; the orchestrator only ever asks 3 canned questions (P2b F4). | Retriever contract test against a 3-document fixture collection (Qdrant on-disk, built in a fixture or committed); assert the 3 canned questions return chunks with expected article IDs. Doubles as the cache-correctness test when P2b F4's `lru_cache` lands. |
| **T-13** | `arcade/strategy_pipeline.py` (the known #1 drift debt) has no parity test vs the orchestrator - exactly how the 3-tuple mirror rot happened. | Until the shared engine retires it: a parity test running both paths on one fixture lap (FakeOpenAI) asserting identical actions + agent-output field sets. After the engine: the P2b §9 engine-vs-orchestrator contract test replaces it. |
| **T-14** | mypy scope excludes everything the tests would lean on (`src/telemetry/` wholesale; only `src/rag/` checked). Type-checking is a cheap co-tester for contract drift (the 3-tuple would have been a mypy error under checked signatures). | Adopt P1 F-17's phased plan; add `src/simulation/` + `src/f1_strat_manager/` to the parent scope (small, typed-ish already). |
| **T-15** | pytest-cov installed, never wired; no coverage signal exists at all. | Wire `--cov` report-only in both repos' CI (artifact + PR summary). Ratchet later (§8) - never start with a hard global gate on a legacy base. |
| **T-16** | Parent CI doesn't checkout the submodule, so even the backend-schema tests that could run, skip. | `actions/checkout` with `submodules: true` in the parent `test` job (submodule is code-only, small). Keeps repo-per-repo ownership: the submodule's own CI stays the primary gate for backend tests. |

### P3 — hygiene / accepted gaps

| ID | Finding | Decision |
|---|---|---|
| **T-17** | No pytest-xdist. | Skip until the hermetic suite exceeds ~60 s (PROJECT_BOOTSTRAP §4 caveat 2); then `-n auto --dist=loadfile`. |
| **T-18** | Streamlit frontend untested. | **Accept.** It is being deleted (migration W10). Do not invest; parity is verified visually against the dossier per the migration DoD. |
| **T-19** | `sys.path.insert` repeated inside individual tests (`test_agents.py:192+`), duplicated backend-path logic across parent tests and submodule conftest. | Consolidate into the parent `tests/conftest.py` when touching those files. |
| **T-20** | Arcade rendering/pyglet loop untestable headlessly without disproportionate effort. | Accept import + dashboard-offscreen smoke (QT_QPA_PLATFORM=offscreen for the PySide6 side); rely on the engine tests for logic and on manual replay smoke per the P2 §6 protocol. |
| **T-21** | Benchmarks (`scripts/bench/`, P2b Phase-0 timing harness) are adjacent but distinct from tests. | Keep separate; optionally publish nightly timing trends from the data-tier run against the P2 §4 budgets (regression *alerting*, not gating). |

---

## 5. Target test pyramid

Principles: hermetic-by-default (no network, no GPU, no HF data in tiers that gate PRs); the LLM is **always** the FakeOpenAI stub; untouchables tested only via public entry points / subprocess; every hand-duplicated seam gets a contract test; determinism (seed 42) exploited as a free oracle.

```
            ┌────────────────────────────────────────────┐
  nightly / │  L4  Visual regression (Playwright          │  37 dossier-seeded states
  on-demand │      snapshots) + full-data e2e smokes      │  + f1-sim data-tier matrix
            ├────────────────────────────────────────────┤
   PR gate  │  L3  Surface e2e: SPA Playwright (MSW-      │  ~15-25 flows
   (SPA CI) │      mocked backend), CLI fixture smoke     │
            ├────────────────────────────────────────────┤
   PR gate  │  L2  Integration/contract: FastAPI Test-    │  ~80-120 tests
            │      Client per route, SSE vs transcripts,  │
            │      engine golden scenarios w/ FakeOpenAI, │
            │      RSM + simulator on mini fixture        │
            ├────────────────────────────────────────────┤
   PR gate  │  L1  Unit: pure helpers everywhere          │  ~200+ tests
            │      (chat engine ✅, bridge ✅, voice rate,│
            │      path resolver, guard rails, MC math,   │
            │      routing, pickers, SSE framing, TS lib) │
            └────────────────────────────────────────────┘
```

### L1 — Unit (hermetic, < 15 s total, both repos, every PR)

| Area | Targets (representative, not exhaustive) |
|---|---|
| Backend | existing 45 ✅ · `tts_service._format_rate` · voice size-cap validator · SSE frame builders (both dialects → unified) · error-envelope helpers (F-11) · `laps_cache` selection logic (parquet IO monkeypatched) · repo-root resolver (temp trees) · `mcp_tools` alias/driver-code resolution |
| Parent | `_decide_agents_to_call` ✅ (extend the truth table) · `_sc_active_from_rcm` ✅ · guard-rail decision table · MC scoring: canned distributions → exact scores (seed 42) · `_build_rag_question` parametrization · `gp_slugs` · `data_cache` pattern/selection logic · CLI `pickers` pure parts · `stream.py` newline-JSON framing |
| SPA (`webapp/src/lib/`) | SSE parser vs recorded transcripts (the same `.sse` files as Python) · comparison interpolation math vs a captured `/comparison/compare` fixture (migration risk 2 - **the plan already mandates this**) · `toolResult→EChartsOption` dispatcher table (F1-1) · URL/state codecs |

### L2 — Integration / contract (hermetic via fixtures + FakeOpenAI, < 2-3 min, every PR)

| Suite | Design |
|---|---|
| **Backend route contracts** | Bare `FastAPI()` + one router per test module (the pattern `test_simulation.py` already proves - keeps FastMCP/voice imports out). Every route: 1 happy path (fixture-backed or service-stubbed) + 1 error contract (404/422/503, envelope shape). Health semantics pinned. OpenAPI assertions: `operation_id` present, no leaked `year` query param, `response_model` non-generic (guards A5-4 type generation). |
| **Chat SSE end-to-end-in-process** | TestClient stream of `/chat/tool-message-stream` with FakeOpenAI scripting: text-only answer · tool-call → MCP dispatch (FastMCP in-process) → summary · LLM error mid-stream. Assert the event grammar against the recorded-transcript fixtures. This is the suite that protects the SPA's primary transport. |
| **Simulate SSE** | Existing test, tightened (no error frames on happy path) and unskipped by pointing it at the mini race fixture instead of `data/raw/2025/Melbourne`. Add: bad GP/driver → pre-stream 4xx once P1 F-18 lands (test-first). |
| **Engine golden scenarios** | 4-6 canned lap_states (quiet lap, SC-active lap, pit-window lap, radio-alert lap, cliff-warning lap, lap-1 guard-rail) → `run_strategy_orchestrator_from_state` (and later the P2b engine, same fixtures) with FakeOpenAI returning scripted syntheses → assert: routing set, action, guard-rail overrides, MC scenario scores (deterministic), Pydantic 14-field completeness. Plus the **agent-call-count spy** (kills the double-run class) and the **arcade-pipeline parity test** (T-13). |
| **RSM / replay** | `RaceStateManager` on the mini parquet: lap_state schema (driver full-telemetry vs rivals timing-only boundary asserted **explicitly** - it is the thesis's single-driver contract), edge laps (1, last), missing-rival handling. |
| **NLP / RAG** | RCM parser + pipeline glue with model stubs; retriever vs 3-doc fixture collection; the 3 canned RAG questions. |
| **Voice contracts** | transcribe/synthesize/voice-chat with STT/TTS/LLM stubbed: rate string, size cap, health degrade, voice-per-call (F-23b, test-first). |

### L3 — Surface e2e (fast, gates the owning repo's PRs)

| Surface | Design |
|---|---|
| **SPA (Playwright, MSW-mocked backend)** | One spec per page mirroring the §6 parity checklists: selectors → load → chart renders → interaction (click-to-load, tab switch, stream + Stop, report download). MSW serves the same fixture JSONs / SSE transcripts the backend tests use - **one fixture set, two consumers**, so client and server can't drift apart silently. Combobox interactions keyboard-driven (migration risk 7). |
| **SPA (live-backend smoke, optional job)** | 2-3 flows against docker-compose backend with FakeOpenAI as the "LM Studio": dashboard load, one chat turn, one 3-lap simulate. Nightly or pre-release, not per-PR. |
| **CLI** | Fixture-tier: `f1-sim <fixture-gp> --no-llm --laps 5-7` subprocess, assert exit 0 / no `[ERROR]` / summary printed. LLM-mode variant with FakeOpenAI on :1234 (`F1_LLM_PROVIDER=lmstudio` - zero CLI changes needed). If the PMV's data expectations make the mini fixture unusable for it, this test lives in the data tier until the P4 duplicate exists, and the engine no-LLM test covers the logic per-PR. |
| **Arcade** | Import smoke (exists) + dashboard construction under `QT_QPA_PLATFORM=offscreen` + engine tests carry the logic. Full replay stays a scripted manual smoke (P2 §6). |

### L4 — Visual regression + full-data (nightly / on-demand / cutover gates)

- **Visual regression seeded from the 37 dossier PNGs.** Honest design: the dossier PNGs are *Streamlit* screenshots - they cannot be literal pixel baselines for a different stack. Their role, phased:
  1. **State catalog + naming contract** (now): the 37 `live_*` names (5 chat, 3 comparison, 3 dashboard, 8 model-lab, 10 race-analysis, 7 strategy, 1 theme) define the exact reachable states every parity PR must reproduce and screenshot. The Playwright harness gets one navigation recipe per state, named identically.
  2. **Side-by-side parity evidence** (sprints 2-5): each parity PR attaches its SPA capture next to the dossier PNG - already the migration DoD; the harness just automates the capture half.
  3. **Self-baseline pixel regression** (from cutover): the SPA captures become `toHaveScreenshot` baselines (fixed 1440×900 viewport matching the dossier method, self-hosted fonts, `animations: disabled`, rAF/`Date` frozen for replay states, MSW-pinned data so chart series are byte-identical, orb/WebGL states masked or reduced-motion). Per-page `maxDiffPixelRatio` tolerance; baseline updates are explicit PR artifacts, reviewed like code. This is migration backlog item **E9** - schedule it at cutover, not before (pre-cutover the UI churns too fast for pixel gates).
- **Full-data smokes (Víctor's machine, workflow_dispatch or pre-release script):** `f1-sim` matrix (2 GPs × {no-llm, FakeOpenAI-llm}) asserting zero error rows + budget alerting vs P2 §4; Arcade cached-race load smoke; backend boot + one real-parquet strategy call; NLP label-stability probe; the Qatar V7 regression (already written ✅ - it is exactly the right kind of test, it just never runs).

---

## 6. Fixtures strategy (the load-bearing layer)

| Fixture | Contents | Producer / refresh policy | Consumers |
|---|---|---|---|
| **FakeOpenAI stub server** | In-process ASGI/HTTP server implementing `/v1/chat/completions` (+`/v1/models`), scripted per-test: plain text, tool_calls, structured-output JSON matching `StrategyRecommendation` and per-agent syntheses, error/timeout modes. Deterministic; matches by request sequence or marker strings in prompts. | Hand-written once (S/M). Extend when agents' expected shapes change. | Engine golden tests, chat SSE tests, CLI llm-mode smoke, SPA live-backend smoke |
| **Recorded SSE transcripts** (`.sse` text) | 1× chat stream happy path (stages+tokens+tool_result+done), 1× chat tool-error path, 1× simulate stream (start/laps/summary), 1× simulate error frame | Captured once from a live local run; **regenerated in the same PR as any grammar change** (F-19 unification) - fixture diff = contract diff | Python grammar tests, TS `RaceFeed` parser unit tests (migration W0), MSW mocks |
| **Canned lap_state JSONs** (4-6) | Quiet / SC-active / pit-window / radio-alert / cliff / lap-1, each schema-complete (driver, rivals, weather, session_meta, rcm_events) | Extracted once from RSM on real data, then frozen; schema-validated in a test so drift in RSM output is caught | Engine goldens, backend strategy-route contracts, agent entry-point tests |
| **Canned agent outputs** | One JSON per agent output dataclass + one full `StrategyRecommendation` | Frozen from a real run | MC/scoring goldens, CLI/Arcade/Backend rendering & serialization tests, SPA fixture data |
| **Mini race parquet** (< ~150 KB) | One GP sliced to ~10-15 laps × 4-6 drivers (+ minimal `rcm.parquet` with the SC event), matching real schema/dtypes | Small generation script under `tests/fixtures/` (script committed, artifact committed too - regenerable but never a CI dependency on HF) | RSM tests, simulate SSE, engine tests, CLI fixture smoke |
| **RAG mini-collection** | 3-5 regulation chunks (the articles behind the 3 canned questions) in an on-disk Qdrant dir or built by fixture from committed text | Built in a session-scoped fixture from committed markdown snippets (avoids committing binary Qdrant files) | Retriever contract, canned-question cache tests |
| **Comparison telemetry fixture** | One captured `/comparison/compare` response JSON | Captured once | SPA interpolation math tests (risk 2), MSW |
| **STT/TTS stubs** | Whisper adapter stub returning canned transcription; Edge-TTS stub capturing the rate/voice arguments | monkeypatch at service seam | Voice contracts |

**Governance:** fixtures live in each repo next to their tests (`tests/fixtures/`); anything > ~200 KB or regenerable-from-HF stays in the data tier instead. The carve-out to the "data not in git" rule (T-6) needs Víctor's explicit sign-off - file it as a decision issue before Phase 1.

---

## 7. Per-repo CI gates

### 7.1 Parent `F1_Strat_Manager`

| Job | Content | Tier | Gate status |
|---|---|---|---|
| `test` (rename → `test-unit`) | L1 + hermetic L2 (markers `unit or contract`), submodule checked out (`submodules: true`), `--cov` report-only, collected-count floor (T-4) | PR | **required** (already is) |
| `lint` | unchanged (ruff check + format) | PR | required (already is) |
| `typecheck` | mypy `src/rag/` + add `src/simulation/`, `src/f1_strat_manager/` | PR | required (already is) |
| `sim-smoke-data` | `data`+`gpu`+`llm` markers: f1-sim matrix, Qatar V7, NLP stability, budget alerts | `schedule` (nightly/weekly) + `workflow_dispatch`; effectively runs on Víctor's machine via a documented `scripts/` runner if hosted runners can't hold the data | **non-blocking, alerting** |

### 7.2 Submodule `F1_Telemetry_Manager` (ties to epic #25)

| Job | Content | When introduced | Gate status |
|---|---|---|---|
| `test` | the 45 existing + backend contract suite + voice + SSE grammar (all hermetic) | **Phase 0 (now)** | **required immediately** - the tests already pass |
| `lint` | broaden beyond E9/F63/F7/F82 once baselined (per the workflow's own TODO) | Phase 1 | required |
| `typecheck` | submodule `[tool.mypy]`, F-17 tranches (models/utils/chatbot → endpoints) | with P1 Phase 6 | required per tranche |
| `webapp-ci` (lint, tsc, vitest, build) | SPA quality gates - **already specified in migration S1-1**; vitest includes the SSE-parser + interpolation suites | migration W0 | required from S1-1 merge |
| `webapp-e2e` | Playwright vs MSW (no backend container needed) | Sprint 2 (first real page) | required from Sprint 2 |
| `webapp-visual` | dossier-state capture per PR (artifact for review) → pixel gate post-cutover (E9) | captures from Sprint 2; gate at W10 | artifact-only → required post-cutover |

**Branch protection:** update required contexts in both repos' setup scripts as each job stabilizes (PROJECT_BOOTSTRAP §2/§10). Path-filter gating (§4.1) only if the hermetic suites exceed ~3 min - not before.

---

## 8. Coverage targets (pragmatic, per-module, ratcheted)

Global blanket %-gates on this codebase would be noise (untouchable files, GPU model code, a UI being deleted). Targets are per-area, and the gate is a **ratchet** (fail if coverage drops > 1 pt vs main) introduced only after Phase 2, per repo.

| Area | Today (est.) | 6-week target | 12-week target | Notes |
|---|---|---|---|---|
| `backend/services/chatbot/` | ~80% (orphaned) | 85% + in CI | 85% | keep; add engine-flow integration |
| `backend/api/v1/endpoints/` | ~0% | **route coverage 100%** (≥1 contract test/route), lines ~60% | 70% | route-count assert, not just lines |
| `backend/services/` (voice, telemetry, simulation) | ~0% | 55% | 70% | stubs at adapter seams |
| `src/simulation/` | thin | 80% | 85% | mini-fixture powered |
| `src/f1_strat_manager/` (data_cache, gp_slugs) | 0% | 65% | 75% | network mocked |
| `src/agents/` (via entry points) | n/a | **no line target** - behavioral: 6 golden scenarios + routing truth table + guard-rail table green | +parity test vs engine | untouchable; lines are the wrong metric |
| Engine module (new, P2b) | n/a | born ≥ 85% | 90% | test-first, it's new code |
| `src/nlp/` pure parts | 0% | 45% | 60% | model inference in data tier |
| `src/rag/` | 0% | 60% | 70% | mypy already covers it |
| SPA `src/lib/` (parsers, math, dispatchers) | n/a | born ≥ 80% (interpolation + SSE parser ~100%) | keep | vitest |
| SPA pages/components | n/a | e2e flow coverage = §6 parity rows ✅ | +visual gate | not line-measured |

---

## 9. Cover-these-first (the prioritized first ~20 tests, in order)

The exact paths that would have caught the escaped bugs, plus the seams most likely to break next (P1 Phase 2 contract changes, migration W0). Each row is small enough to be a single PR or bundled 2-3 per PR.

| # | Test | Catches / protects | Repo | Effort |
|---|---|---|---|---|
| 1 | Submodule CI `test` job (no new tests - wire the 45) | everything the chat engine already guards | sub | S |
| 2 | `_format_rate` unit + synthesize contract (stubbed TTS) | **the voice rate bug** (F-12) | sub | S |
| 3 | Repo-root resolver temp-tree tests (gitlink file, env override, /app fallback) | **the walker bug** (F-1) - write against the new shared resolver, test-first | sub | S |
| 4 | Tighten `test_simulate_race_emits_start_lap_summary`: zero `error` frames on happy path; retarget to mini fixture so it stops skipping | **the `--no-llm` class** on the backend copy | parent | S |
| 5 | CLI `--no-llm` subprocess smoke (data tier now; fixture tier when possible) | **the `--no-llm` bug** on the PMV itself | parent | S |
| 6 | Agent-call-count spy per lap (xfail on CLI path until engine; hard assert on engine/pipeline) | **the CLI double-run** (P2b F1) | parent | S |
| 7 | LLM timeout-not-None unit test (both providers) | the LM Studio infinite-hang (F-2a) | sub | S |
| 8 | FakeOpenAI fixture server (+ env-configurable port enabler PR) | unlocks 9-15 | both | M |
| 9 | Chat SSE stream integration (happy, tool-call, error) vs recorded transcripts | SPA transport (A5-2), F-19 unification | sub | M |
| 10 | Error-envelope contract tests, test-first for P1 F-11 (incl. `/chat/tool-message` non-2xx on failure, `lap-range` 404, `available-gps` 503) | the 200-with-error class | sub | M |
| 11 | OpenAPI meta asserts: operation_ids, no `year` query param (F-15), typed response models (F-14) | typed-client generation (A5-4) | sub | S |
| 12 | Engine golden scenarios (4-6 canned lap_states → routing/action/guard-rails/MC scores) | the whole decision spine; regression bed for P2b's engine build | parent | M |
| 13 | MC determinism + score-formula goldens (canned distributions) | silent numeric drift in the thesis-defended math | parent | S |
| 14 | Arcade-pipeline ↔ orchestrator parity on one fixture lap | the mirror-rot class (how F2 happened) | parent | S |
| 15 | RSM lap_state schema + single-driver boundary asserts on mini parquet | the `lap_state` contract all agents and both UIs consume | parent | S |
| 16 | Voice size-cap + health-degrade contracts (with F-26 fixes) | unbounded upload / CPU-crash paths | sub | S |
| 17 | Simulate pre-validation contract (bad GP → 4xx pre-stream), test-first for F-18 | SSE error-inside-200 class | sub | S |
| 18 | RAG canned-question retrieval vs mini-collection | P2b F4 cache correctness | parent | M |
| 19 | SPA W0: SSE parser vs the same transcripts + interpolation math vs comparison fixture | migration risks 2 & 3 | sub (webapp) | M |
| 20 | Dossier-state Playwright capture harness (37 named states, capture-only) | parity evidence automation → future E9 baselines | sub (webapp) | M |

---

## 10. Phased plan (chunkable → issues/PRs; S/M/L)

Aligned with the sibling audits' sprints: Phase 0-1 land before/with P1 Phase 2 (contract freeze) and migration W0; Phase 3 tracks the SPA sprints; Phase 4 gates cutover.

### Phase 0 — Stop the bleeding (S total, ~1 day) — *this week*
1. **[S]** Submodule CI `test` job + make required (T-1). *(#9-list item 1)*
2. **[S]** The three retro-tests: voice rate, timeout-not-None, tighten simulate-SSE assert. *(items 2, 4, 7)*
3. **[S]** Markers + tiers (`unit/contract/data/gpu/llm`) in both repos; skip-summary + collected-count floor in CI (T-4).
4. **[S]** Parent CI: `submodules: true`; wire `--cov` report-only (T-15/T-16).
5. **[S]** File the fixtures carve-out decision issue (T-6) + per the house rule, issues for each escaped-bug test gap.

### Phase 1 — Fixture foundation (M total, 1 sprint) — *before P1 Phase 2 & migration W0*
6. **[M]** FakeOpenAI stub fixture (+ S enabler: env-configurable LLM port/base-URL, aligns F-34). *(item 8)*
7. **[S]** Record SSE transcripts; grammar tests (T-7). *(item 9 partial)*
8. **[S]** Mini race parquet + generation script; canned lap_states; canned agent outputs (T-6).
9. **[S]** Repo-root resolver tests, test-first with the F-1 PR. *(item 3)*
10. **[S]** RSM schema/boundary tests on the mini fixture. *(item 15)*

### Phase 2 — Contract & engine suites (M/L, 1-2 sprints) — *with P1 Phases 2-4 and P2b Phase 1*
11. **[M]** Backend route-contract suite incl. error envelope test-first (T-5). *(items 10, 11, 17)*
12. **[M]** Chat SSE integration suite with FakeOpenAI. *(item 9)*
13. **[M]** Engine golden scenarios + MC goldens + call-count spy + arcade parity (T-2/T-10/T-13). *(items 6, 12, 13, 14)* — co-developed with P2b's `engine.py` so the engine is born tested.
14. **[S]** Voice contract mini-suite (T-8). *(item 16)*
15. **[S]** CLI no-llm smoke wiring (fixture tier if viable, else data tier). *(item 5)*
16. **[S]** Coverage ratchet on, both repos.

### Phase 3 — Surfaces & SPA test bed (M, tracks migration sprints S1-S4)
17. **[M]** SPA W0 test stack: vitest + SSE-parser/interpolation suites + `webapp-ci` job (= S1-1/S1-2 DoD). *(item 19)*
18. **[M]** Playwright e2e per page with MSW fixtures, added sprint-by-sprint as pages land (S2-S4); dossier-state capture harness. *(item 20)*
19. **[S]** NLP pure-part units + RAG mini-collection contract (T-11/T-12). *(item 18)*
20. **[S]** Arcade offscreen dashboard smoke (T-20).

### Phase 4 — Nightly data tier & visual gate (M, around cutover W10)
21. **[M]** `sim-smoke-data` workflow + local runner script: f1-sim matrix, Qatar V7, NLP stability, P2-budget alerting (T-4/T-21).
22. **[M]** Visual regression promotion (E9): SPA self-baselines from the 37-state captures, `toHaveScreenshot` gate post-cutover, baseline-update review protocol.
23. **[S]** mypy expansion tranches complete (T-14); xdist if suite > 60 s (T-17).

**Dependency spine:** 1 → (2-5 parallel) → 6 → {7,8} → everything in Phase 2; Phase 3 tracks the migration calendar independently once 6-8 exist; Phase 4 gates cutover.

---

## 11. Risks & explicit non-goals

- **Untouchable ≠ untestable.** Tests import agent modules and private helpers already (precedent: `_decide_agents_to_call`, `_build_pit_prompt` in `test_agents.py`); the CLI is tested strictly as a subprocess black box. No proposal here edits either. If an agent-internal helper's import path changes, only tests break - acceptable.
- **FakeOpenAI fidelity risk:** a stub can't catch prompt-quality regressions or provider quirks. Mitigation: the data/llm tier keeps 1-2 real-LM-Studio smoke runs (local, non-gating); the stub's job is *plumbing* correctness (shapes, tuples, grammars, routing), which is where every escaped bug lived. Never Anthropic, in stub or smoke.
- **Fixture staleness:** mini parquet / transcripts / canned outputs can drift from reality. Mitigation: each fixture has a producer script + a schema-validation test against the live shape (data tier) and the rule that contract-changing PRs regenerate fixtures in-PR.
- **Port :1234 collisions** if a dev's real LM Studio is running while tests run - the enabler PR (env-configurable port) removes this; until then the fixture skips-with-warning if the port is occupied by a non-fake server.
- **Visual-gate flake** is the classic Playwright failure mode. Mitigations baked into §5 L4 (frozen time/animation, MSW-pinned data, masked WebGL, per-page thresholds, artifact-review workflow); the gate is deliberately deferred to post-cutover.
- **Non-goals:** no Streamlit test investment (sunset); no load/perf gating in CI (budgets are alerting-only, P2 owns them); no mutation testing (revisit if the suite matures); no test-runner change (pytest + vitest + Playwright only); no auth/multi-user scenarios (out of the product's local-first posture).

### Alignment with sibling audits
- **P1:** items 10, 11, 14, 16, 17 here are the test-first halves of its Phases 2-4; its Phase 6 item 22 is superseded by §5 L2's fuller contract suite.
- **P2:** the data-tier smoke doubles as its §6 verification protocol + §4 budget alerting.
- **P2b:** the engine golden suite (item 13) is the regression bed its Phase 1 needs; its §9 parity test and Phase 0.1 bug issue are incorporated here (items 14, and Phase 0 step 5).
- **Migration:** items 17-20 = the concrete content of S1-1's "CI (lint/typecheck/build/vitest)", risks 2/3/7 mitigations, and backlog E9.
