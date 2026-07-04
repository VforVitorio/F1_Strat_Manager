# Fable 5 — independent audit backlog (F1 StratLab)

> Each entry below is an **independent** Fable 5 audit of one subsystem: assessment → prioritized
> improvement plan (findings P0–P3), no code. Run them one at a time, in priority order (adjustable).
> The frontend **migration** is a separate, cohesive run — its detailed brief is `./FABLE_BRIEF.md`.
> Frontend *improvement* topics are folded INTO that migration run (its audits 2/3/4 = UX/IA,
> visual/design-system, performance), so there is no separate "front improvements" audit here.
>
> Common rules for every audit: plan only, no code; back stays FastAPI; LLM = OpenAI/LM Studio, never
> Anthropic; respect UNTOUCHABLE files (duplicate before improving); each audit reads its existing
> memory backlog first and builds on it; output = one `AUDIT_<name>.md` with findings + a phased,
> chunkable improvement plan the human can turn into issues/PRs/sprints.

---

## Priority order (proposed — reorder freely)

| # | Audit | Why here | Runs |
|---|---|---|---|
| **P0** | **Frontend migration** (Streamlit→local web app) | Stated priority #1 of the core; biggest surface | Cohesive run — briefed in `FABLE_BRIEF.md` |
| **P1** | **Backend (FastAPI / web layer)** | Serves the web SPA (migration). NOTE: CLI+Arcade do NOT hit it — they run the core in-process (→ P2b) | Independent |
| **P2** | **Loading & performance** (cross-cutting) | "Carga de las cosas" — the most felt pain (boot times, warmup, HF downloads) | Independent, cross-cutting |
| **P2b** | **Core compute & inference efficiency** (CLI + Arcade in-process engine) | The shared strategy engine (agents + orchestrator 500-sample MC + RaceStateManager per-lap) that CLI+Arcade run directly = the real "backend" for those two; runtime latency, not just boot | Independent |
| **P3** | **Arcade** | Heavy tech debt already identified (pipeline duplication) | Independent |
| **P4** | **CLI simulation** | PMV untouchable → audit yields a duplicate-and-improve plan | Independent |
| **P5** | **Data & new features** | Additive/future (new data features + 2026 retraining prep) | Independent |

---

## P1 — Backend audit
- **Scope**: `src/telemetry/backend/` — FastAPI app, `api/v1/endpoints/`, services (chatbot, strategy),
  `mcp_tools.py` (FastMCP single source of truth), SSE endpoints, voice.
- **Why**: the new frontend (P0) consumes this HTTP contract; auditing it de-risks the migration. NOTE: the CLI and
  Arcade do NOT go through FastAPI — they run the strategy core in-process, so their runtime speed is **P2b**, not this.
  Keep pragmatic, not a rewrite.
- **Produces**: findings on endpoint design/consistency, async I/O vs blocking calls, dead/deprecated code
  cleanup (old chat handlers/router were marked deprecated — confirm removal), error contract, mypy scope
  (currently only `src/rag/`), caching, and a stable API surface spec the frontend can code against.
- **Inputs**: `[[project-chat-mcp-refactor]]`, `[[project-streamlit-refactor-backlog]]` (backend items),
  `[[project_fastmcp_architecture]]`, `[[project_v09_architecture]]`.
- **Constraint**: `src/agents/` internals UNTOUCHABLE (additive entry points only). Skill to lean on later: `refactor-fastapi`.

## P2 — Loading & performance audit (cross-cutting)
- **Scope**: boot/startup + data loading across all surfaces. `f1-sim` pre-warms Whisper + 7 agents before
  lap 1; Arcade `SessionLoader` (multiprocessing.Pool(8)); lazy HF model download on first run
  (`data_cache.py`); Whisper JSON transcription cache; Streamlit reruns/session_state churn; duplicated image
  encoding; model warmup banners.
- **Why**: "carga de las cosas" is the most visible friction. Cross-cutting wins (lazy-load, cache, parallelize,
  prewarm) help CLI + Arcade + Streamlit at once.
- **Produces**: bottleneck → cause → remedy → expected win table, per surface; a shared caching/prewarm strategy;
  what to lazy-load vs eager. Prioritized P0–P3.
- **Inputs**: `[[project_v09_architecture]]`, `[[project_hf_models_restructure]]`, `[[project_cli_distribution_plan]]`
  (lazy first-run HF download), Arcade SessionLoader notes in `MEMORY.md`.

## P2b — Core compute & inference efficiency audit (CLI + Arcade in-process engine)
- **Scope**: the SHARED strategy engine that the CLI (`run_simulation_cli.py`) and Arcade run **in-process** (NOT via
  FastAPI): the 6 sub-agents' per-lap inference, orchestrator N31 (MoE routing + **Monte Carlo over 500 samples** + LLM
  synthesis), `RaceStateManager.get_lap_state` per lap, and `arcade/strategy_pipeline.py`. This is the "backend" that
  actually determines CLI/Arcade responsiveness during a running sim.
- **Why**: the gap the user flagged — P1 is the FastAPI HTTP layer (web only) and P2 is I/O/boot; neither covers the
  **runtime compute** cost of running the agents + the 500-sample MC every lap, which is what makes a live sim feel
  fast or sluggish in CLI/Arcade.
- **Produces**: a per-lap latency budget (which agent/step dominates), MC-sampling cost + whether it can be
  vectorized/reduced/cached, warmup vs steady-state, cross-lap batching, and a shared fast-path both surfaces reuse.
  Prioritized P0–P3. Keep "loading" (P2) separate from "compute" (here).
- **Inputs**: `[[project_v09_architecture]]`, `[[project_arcade_refactor_backlog]]` (the `strategy_pipeline`
  duplication), `[[project_orchestrator_v2_schema]]` (MC design), `[[project_strategic_guardrails]]`.
- **Constraint**: `src/agents/` internals UNTOUCHABLE → optimize via additive entry points, caching and batching
  wrappers, not by editing agent internals. Owns the **shared compute core**; P3/P4 own the surface-level UX.

## P3 — Arcade audit
- **Scope**: `src/arcade/` — pyglet 2D replay + PySide6 3-window dashboard + TCP stream (`stream.py`),
  `strategy_pipeline.py`, `dashboard/`, `RaceReplayEngine`/`RaceStateManager` usage.
- **Why**: heavy debt already flagged — `arcade/strategy_pipeline.py` **duplicates**
  `agents/strategy_orchestrator.py` (the "#1 heavy" item). Native PySide6 (not browser) → screenshot via Qt
  `grab()`, not Playwright.
- **Produces**: a decoupling plan for the pipeline duplication (single source via an additive entry point),
  rendering/perf findings, dashboard structure cleanup, and any UX polish. P0–P3.
- **Inputs**: `[[project_arcade_refactor_backlog]]` (read first — it's the audit's baseline),
  `[[project_v09_architecture]]`, `[[arcade_ui_design]]`.
- **Constraint**: `src/agents/` internals UNTOUCHABLE; reconcile duplication by extracting an additive shared
  entry point, not by editing agent internals.

## P4 — CLI simulation audit
- **Scope**: `scripts/run_simulation_cli.py` (**UNTOUCHABLE — the TFG's PMV**), `scripts/f1_cli.py` (menu),
  `src/f1_strat_manager/` (data_cache, gp_slugs), the Rich Live panel.
- **Why**: high-value surface but the core sim script cannot be edited in place → the audit must yield a
  **duplicate-and-improve** plan (post-defense refactor always on a copy). Boot time + UX are the targets.
- **Produces**: a duplicate-first refactor plan (what to extract/clean on a copy), boot-time improvements
  (ties into P2), CLI UX/menu polish, and distribution notes.
- **Inputs**: `[[project_cli_refactor_backlog]]` (read first), `[[feedback_cli_intocable]]`,
  `[[project_v09_cli_panel]]`, `[[project_cli_distribution_plan]]`.
- **Constraint**: NEVER modify `run_simulation_cli.py` or `src/agents/` internals in place — duplicate before touching.

## P5 — Data & new-features audit
- **Scope**: `src/shared/data_extraction/` (fastf1/openf1 extractors, augmentation), `data/` layout, HF Hub
  dataset/models (`VforVitorio/f1-strategy-dataset` → org `f1stratlab`), the `lap_state` contract, and NEW
  data-feature ideas (extra telemetry-derived features, rival data, etc.).
- **Why**: additive/forward-looking. Also the entry point for the **2026-regulation retraining** strategy
  (historical + FP/Qualy/Sprint → race; clustering K=4; incremental per-GP) and the Rival Agent's data needs.
- **⭐ 2026-reg / concept-drift sub-scope (Víctor, 2026-07-04):** HIGH RIGOR but **NOT priority and VERY future** — yet
  it **WILL have to be done**. When run, the Fable brief MUST state up front: *not priority, very future, but it is going
  to have to be done — plan with maximum rigor.* Covers what breaks under 2026, retraining on **FP/libres + Qualy +
  Sprint → race**, drift detection (RevIN / year-embedding), and it feeds the **model-training tool (`pitlab` Studio)**.
  See [[project_future_vision]] + memory `project_fable_audit_backlog`.
- **Produces**: a prioritized backlog of new data features + their sourcing/feasibility, a data-pipeline
  improvement plan, HF layout/restructure follow-through, and the retraining-data plan for 2026.
- **Inputs**: `[[project_future_vision]]` (2026 retraining + ecosystem datasets), `[[project_hf_models_restructure]]`,
  `[[project_radio_ingestion_plan]]`, `[[reference_tfg_memoria_location]]` (thesis future-work).
- **Note**: overlaps the post-TFG ecosystem (`radiogate`, `gridmind`, `pitlab`) — scope this audit to the CORE
  repo's data layer, not the separate ecosystem repos.

---

## After each audit (Fase C, orchestrator does this)
Convert the audit's phased plan into GitHub artifacts in the **owning repo** (parent `F1_Strat_Manager` for
CLI/Arcade/backend/data; submodule `F1_Telemetry_Manager` for frontend, tied to issue #25): epic → milestones
(sprints) → single-concern typed issues (`Closes #N`) → PR strategy. Then a short `project_*` memory file per
audit, linked from `MEMORY.md`. Per-issue implementation later uses `ui-skills` audits (frontend) +
`refactor-fastapi`/`simplify`/`code-review` + screenshot/verify.
