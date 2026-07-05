# AUDIT P4 - CLI simulation surface (duplicate-and-improve plan)

**Auditor:** Fable 5 · **Date:** 2026-07-05 · **Repo:** `F1_Strat_Manager` (read-only pass, no code changed)
**Scope:** `scripts/run_simulation_cli.py` (the PMV, audit only), `scripts/f1_cli.py` + `scripts/cli/` (interactive menu), `src/f1_strat_manager/` (`data_cache.py`, `gp_slugs.py`), the Rich Live panel, argument handling, distribution (`pyproject.toml` entry points, `uv tool install`).
**Out of scope (owned elsewhere):** boot/download mechanics and warmup ordering -> **P2** (`AUDIT_P2_LOADING.md`); per-lap compute, the shared engine, MC sampling -> **P2b** (`AUDIT_P2B_CORE_COMPUTE.md`); test scaffolding (FakeOpenAI, fixtures, goldens) -> **Testing audit** (`AUDIT_TESTING_QA.md`, epic #179); broken doc commands -> **Docs-accuracy audit** (`AUDIT_DOCS_ACCURACY.md` F-01/F-02/F-14/F-16).

## Hard constraint (governs every remedy below)

`scripts/run_simulation_cli.py` is the **TFG's PMV and is UNTOUCHABLE**: no edits, no new imports, no reordering, no "small fixes" (per `feedback_cli_intocable`, rule escalated 2026-04-14). Same for `src/agents/` internals (additive entry points only) and `notebooks/**`. Therefore this audit produces a **duplicate-and-improve plan**: every structural remedy lands on a **copy** (working name `run_simulation_cli_v2.py`, final home `src/f1_strat_manager/cli/`) or on the **editable** satellites (`scripts/f1_cli.py`, `scripts/cli/*`, `src/f1_strat_manager/*`, `pyproject.toml`). The original stays byte-identical and remains what `f1-sim` points at until Víctor explicitly flips the entry point (open question Q1).

What is editable vs frozen, explicitly:

| File | Status |
|---|---|
| `scripts/run_simulation_cli.py` (2,421 lines) | **FROZEN** - reference only |
| `src/agents/**` | **FROZEN internals** - additive entry points only |
| `scripts/f1_cli.py`, `scripts/cli/{pickers,runner,theme}.py` | editable |
| `src/f1_strat_manager/{data_cache,gp_slugs}.py` | editable (data_cache changes owned by P2 Phase 1) |
| `pyproject.toml` | editable, but the `f1-sim` entry-point line is a decision gate (Q1) |

---

## 0. Executive summary

1. **The advertised fast path is broken.** `--no-llm` has crashed every lap since 2026-05-09: `run_simulation_cli.py:1508` unpacks two values from `_run_conditional_agents`, which returns three since orchestrator commit `bfe5b46`. Tracked as issue **#166**, xfail-tested in `tests/test_cli_no_llm.py`, root-caused as P2b **F2**. The fix home is the P4 duplicate delegating to the P2b shared engine (or a sanctioned one-line PMV hotfix, Q2).
2. **Every invocation, including `--help` and argument typos, pays ~30 s** because the orchestrator import chain (with 3 NLP models loading at import) runs at module level before argparse (`run_simulation_cli.py:117-148`). Measured and root-caused by P2 (F-01, C1). The duplicate parses arguments first and imports lazily.
3. **The PMV is a 2,421-line monolith with a ~685-line `run()` god function and zero tests**, which is exactly why it cannot be safely edited and why the refactor must happen on a copy validated by golden-diff. The 53-day-old `project_cli_refactor_backlog` memory was re-verified in this audit: its line anchors still hold.
4. **The menu is the best-shaped code on this surface** (clean theme/pickers/runner split) but it validates nothing before spawning a 40-60 s subprocess: a driver-code typo or malformed lap range burns a full boot before failing. It also re-pays the full boot on every run (P2 F-03). One stale-claim correction to P2: Head-to-Head is now a **single** subprocess with `--rival` (`scripts/cli/runner.py:152-161`), not two boots.
5. **Distribution works but is untidy**: `uv tool install` ships generic top-level packages `scripts` and `src` into site-packages, the menu makes `cli` importable as a top-level name via a sys.path hack, and the banner hardcodes "v0.9" while the package is at 1.6.2. All fixable without touching the PMV.

The through-line: **P2b builds the shared engine, P2 fixes loading, P4 builds the CLI that consumes both.** The duplicate is not a rewrite for its own sake; it is the delivery vehicle for the #166 fix, the double-inference fix (P2b F1), the boot-order fix (P2 C1/C3), and the warm re-run loop (P2 C2), none of which can land on a frozen file.

---

## 1. Current shape (what exists today)

**Two entry points** (`pyproject.toml:113-117`): `f1-strat` -> `scripts.f1_cli:main` (wizard: banner -> first-run check -> race/driver/laps/provider pickers -> subprocess) and `f1-sim` -> `scripts.run_simulation_cli:main` (headless argparse sim). Plus `f1-arcade` / `f1-streamlit` outside this audit's scope.

**The PMV pipeline** (`run_simulation_cli.py`): env + `.env` load (:102-110) -> heavy imports with fd-level output suppression (:117-148) -> `run()` (:1584) does provider env -> first-run HF check -> parquet + `RaceReplayEngine` load (:1613-1615) -> radio corpus + eager Whisper (:1625-1663) -> `_prewarm_agents` (:1668-1669) -> header panel -> Rich Live lap loop (:1798-2153) -> summary panel (:2155-2268). The Live contract is correct and load-bearing: history rows are printed via `live.console.print` above a fixed-height inference panel inside Live (`project_v09_cli_panel` memory; rationale documented in `_make_table`'s docstring :563-582).

**The menu** (`scripts/f1_cli.py` + `scripts/cli/`): 152-line launcher, `pickers.py` (arrow-key menus with tty fallback, driver->team parquet resolution), `runner.py` (argv builder + `subprocess.run` per simulation), `theme.py` (palette + banner). Every simulation is a fresh subprocess (`runner.py:79-86,117,161`).

**Support modules** (`src/f1_strat_manager/`): `data_cache.py` (data-root resolution with env override -> repo walker -> `~/.f1-strat/data/`; first-run HF snapshot; lazy per-GP race and radio pulls) and `gp_slugs.py` (friendly GP name -> corpus slug, single source of truth, clean). Both editable; `data_cache` remedies are owned by P2 Phase 1 (F-04, F-12) and only cross-referenced here.

**Tests:** one black-box xfail smoke (`tests/test_cli_no_llm.py`) that documents #166. Nothing else covers this surface (`project_cli_refactor_backlog` "NO HAY TESTS", still true).

---

## 2. Findings register (P0 -> P3)

Anchors verified against current code on 2026-07-05. Cross-references name the owning audit where the root cause lives; this register only re-states what P4 must consume, not re-own.

### P0 - broken or felt on every invocation

**C-01 · `--no-llm` crashes every lap (issue #166).**
`run_simulation_cli.py:1507-1516` unpacks `pit_out, rag_text = _run_conditional_agents(...)`; the orchestrator returns a 3-tuple since 2026-05-09 (`strategy_orchestrator.py:1165`). The ValueError is not LLM-related, so `_is_llm_unavailable` re-raises and every lap lands in the `[ERROR]` row path (:2118-2149). This kills the fast demo mode documented in README, INSTALL and CLAUDE.md §5, and the CI-friendly smoke path. Root cause + remedy options: **P2b F2**. P4's position: the durable fix is Phase D below (duplicate delegates to the shared engine, which returns the shape both surfaces need); an interim one-line PMV hotfix exists only if Víctor sanctions it (Q2).

**C-02 · ~30 s paid before argument parsing; `--help` and typos cost a full model load.**
Module-level `from src.agents.strategy_orchestrator import ...` at :136 (inside the fd-suppression block :117-148) executes before `_parse_args()` is ever reached (:2402-2417). A wrong GP name additionally reaches `sys.exit("[FATAL] Race directory not found ...")` only at :1603, after the 30 s import. Root cause is P2 **F-01/C1** (eager NLP load at import inside `src/agents/radio_agent.py`). P4 owns the CLI-side half: the duplicate parses args, validates inputs, prints help, and only then imports the heavy chain (inside the staged warmup).

### P1 - structural debt and major UX levers

**C-03 · Monolith with a god function and no seams (the reason duplicate-first exists).**
2,421 lines; `run()` spans :1584-2268 (~685 lines: env setup, first-run, data load, radio corpus, prewarm, header, lap loop, summary, each concern leaking into the next); the lap loop nests 4-5 levels (:1804-2153); the rendering pipeline is ~30 helpers mutating Rich tables by reference (:563-1363); the agent-stub blocks are duplicated nearly verbatim (`_probe_core_agents` :513-542 vs `_run_no_llm` :1443-1472). All items from `project_cli_refactor_backlog` (quick wins :1814-1841/:2123-2148 status rows, :1685-1725 header grids, :1943-1959 nested try/except; mediums: `run()` split, `_process_lap` extraction, :1995-2008 two-shape result handling; heavies: RadioSource protocol :1625-1663, panel builder :563-1363, `_run_no_llm` decomposition :1408-1576) were re-verified and fold into Phase C/E below.

**C-04 · LLM mode runs all four always-on agents twice per lap.**
`_probe_core_agents` (:1961-1963) feeds the detail panel, then `run_strategy_orchestrator_from_state` (:1964) re-runs the same four internally because the orchestrator's public API discards agent outputs. Root cause + remedy: **P2b F1/F10** (verbose engine entry point `run_lap(...) -> (rec, agent_outputs, timings)` proposed at `src/strategy/inference/engine.py`). P4 consumes it in Phase D; expected win per P2b: -40-45% LLM-mode lap time in the duplicate, with the PMV untouched.

**C-05 · Menu re-pays the full 40-60 s boot on every simulation.**
`runner.py:79-86` runs each sim as a fresh subprocess; "Run another simulation?" (`pickers.py:354-360`) loops back into another full boot. Owned by **P2 F-03/C2**; P4 owns the delivery: an in-process wizard loop in the duplicate (models load once per session, N races per process), with the subprocess path kept as an isolation fallback (Q4). **Correction to P2 §2.1/C2:** Head-to-Head today launches **one** subprocess with `--rival` (`runner.py:120-161`), not two sequential boots; the "2x for H2H" claim is stale against current code.

**C-06 · `--no-llm` is not offline, and the provider env is set even when no LLM is requested.**
`run()` sets `F1_LLM_PROVIDER` unconditionally (:1586); agents attempt real LLM calls and rely on exception matching (`_is_llm_unavailable` :429-445) to degrade to stubs. Consequences (root cause **P2b F8**): retry backoff per agent per lap when the backend is down, and if LM Studio happens to be up, `--no-llm` silently becomes LLM mode. The duplicate's no-LLM profile must never construct LLM clients (engine no-LLM profile, Phase D).

### P2 - argument handling, menu UX, distribution

**C-07 · `--year` does not propagate; the CLI is hardwired to 2025 in five places.**
`--year` (default 2025, :2318-2322) only affects the tire-allocation lookup and the radio corpus. The data paths ignore it: `_default_raw_dir()` hardcodes `raw/2025` (:2288) and `_default_featured()` hardcodes `laps_featured_2025.parquet` (:2303). The menu hardcodes `year=2025` (`f1_cli.py:102`), the picker title says "Available races (2025):" (`pickers.py:271`), and `build_sim_cmd` defaults `year=2025` (`runner.py:50`). Running `--year 2024` today would silently read 2025 data. Matters for the 2026-regulation season (epic #189): the duplicate resolves every path from one `year` value; the menu discovers years from `<data_root>/raw/*/`.

**C-08 · Argument surface: no validation, no friendly failure, no `--version`.**
`--laps "abc"` or `"15..40"` raises an uncaught ValueError traceback (:1673-1676; `main()` catches only KeyboardInterrupt :2411-2417). A reversed range (`40-15`) silently simulates zero laps and prints an empty summary. `team` is a required positional (:2316) even though the menu proves it is derivable from the parquet (`pickers.py:216-248`). A GP-name case mismatch dies with a bare `[FATAL]` (:1603) instead of suggesting candidates (the known-names list already exists in `gp_slugs.COUNTRY_SLUG_BY_GP:34-63` and `discover_races`). No `--version` flag. The duplicate gets: validated lap-range type in argparse, optional team with parquet resolution, did-you-mean GP suggestions, a top-level error boundary rendering friendly Rich errors with distinct exit codes.

**C-09 · Menu pickers accept free text and fail only after a full boot.**
`pick_driver` (`pickers.py:275-306`) takes any 3-letter string; a typo (e.g. "NOO") resolves no team, asks for manual team entry, then spawns the subprocess which boots ~40-60 s before failing per-lap. `pick_laps` (:331-338) applies no format check. Race list is alphabetical rather than calendar order (:206). Remedies (all editable, no duplicate needed): arrow-pick the driver from the parquet-derived per-race driver list (free text as fallback for new signings), validate the lap-range regex at prompt time, sort races by calendar round, and preflight the provider choice (warn at pick time if `OPENAI_API_KEY` is missing or LM Studio is unreachable, instead of discovering it mid-race via stub rows). The full-calendar picker with download-on-select is P2 **F-12** (owned there, Phase 1 item 5); do not double-schedule.

**C-10 · Packaging pollutes site-packages with generic top-level names.**
`[tool.setuptools.packages.find] include = ["src*", "scripts*"]` (`pyproject.toml:140-142`) installs `src` and `scripts` as top-level importable packages in the tool venv; `f1_cli.py:54-55` additionally inserts `scripts/` into `sys.path` so the menu imports `from cli.pickers import ...`, making `cli` a third generic top-level name. Any co-installed package owning `src`, `scripts` or `cli` collides. Target end-state (Phase F): CLI code lives under `src/f1_strat_manager/cli/`, imports are absolute package imports, and entry points reference the package. The `f1-sim = "scripts.run_simulation_cli:main"` line stays untouched until Q1 is decided; the new code gets its own entry point first.

**C-11 · Banner advertises "v0.9" while the package is at 1.6.2.**
`scripts/cli/theme.py:75,89` hardcode "Multi-Agent Race Intelligence System · v0.9"; `pyproject.toml:7` says 1.6.2 and release-please bumps it continuously. Editable quick win: read `importlib.metadata.version("f1-strat-manager")` with a dev-checkout fallback. (The docs site already solved the same problem with the `__DOCS_VERSION__` injection pattern.)

**C-12 · First-run download UX (silent 7-8 GB pass, doubled metadata sweep).**
`data_cache.py:389-398` runs the full snapshot silently under a spinner, then a second "progress" pass. Owned by **P2 F-04** (Phase 1 item 1); P4 only notes two extras to fold into that same PR: `_render_header`/`ensure_setup` build a private `Console()` (:380-382) instead of sharing the CLI console, and the `HF_HUB_DISABLE_PROGRESS_BARS` env flip (:293-307) is process-global.

**C-13 · Palette and driver-color constants exist in three or four copies.**
`theme.py:28-32` (F1_*), `run_simulation_cli.py:630-635` (COL_*), `data_cache.py:47-52` (COL_* copy, comment says "mirrored ... kept verbatim"), plus the driver-color map duplicated from the telemetry backend (:190-218, comment admits the mirror). Sanctioned duplication under the untouchable rule, but the duplicate + editable modules should consume one additive `src/f1_strat_manager/theme.py`; the PMV keeps its private copies forever.

### P3 - hygiene (fold into duplicate work, do not schedule separately)

**C-14 · Dead symbols in the PMV to drop from the copy:** `_AGENT_DISPLAY` (:648-655) is never referenced after definition; `_make_inference_panel(active_agents=...)` accepts the parameter and never uses it in the body (:1266-1363); `_prewarm_agents(no_llm)` ignores its argument (:448-481).
**C-15 · Defensive noise:** four nested try/except AttributeError/TypeError guards around `race_state` buffer appends (:1943-1959) collapse to one `_safe_extend_race_state` helper on the copy (memory quick-win 3).
**C-16 · Two-shape result handling:** `isinstance(result, dict)` branching to extract tactical fields and summary counters (:1995-2008, :2099-2109) plus private-ish `_pit_out`/`_rag_text`/`_radio_out` dict keys (:1562-1576); the engine's typed return (P2b F10) dissolves this on the copy.
**C-17 · Stale H2H wording:** `f1_cli.py:11` says "two drivers, same race, shown back-to-back"; the implementation is one sim with a tracked rival (`runner.py:120-127`). One-line editable doc fix.
**C-18 · Unseeded synthetic radio RNG:** module-level `_random` with no seed (:227-317) makes `--radio-every` runs non-reproducible. The duplicate should accept `--seed` (also what makes golden-diff runs deterministic when synthetic events are on).

---

## 3. Boot time: what P4 owns vs what it consumes (no duplication of P2)

P2 already measured, root-caused and scheduled the loading work. The division of labor:

| Item | Root-cause owner | P4's role |
|---|---|---|
| 30 s import-before-argparse (C-02) | P2 F-01/C1 (lazy `RadioAgentCFG` decision gate, P2 Phase 2 item 7) | Duplicate parses args first, defers heavy imports into warmup (Phase C) |
| Serial boot phases (radio corpus, Whisper, prewarm never overlap) | P2 F-08/C3; prewarm façade X-03 (`src/f1_strat_manager/prewarm.py`, P2 Phase 2 item 9) | Duplicate calls `prewarm(profile="sim", on_stage=...)` and renders staged banners (Phase E) |
| Whisper cost per first GP run | P2 F-02/C4 (ship `transcripts.json` in the HF dataset, P2 Phase 1 item 2) | Duplicate makes Whisper opt-in (`--retranscribe`) once transcripts ship (Phase E) |
| Menu re-boots per run (C-05) | P2 F-03/C2 | Duplicate provides the in-process wizard loop; menu keeps subprocess fallback (Phase E) |
| First-run silent download (C-12) | P2 F-04 (Phase 1 item 1) | Fold the two console/env nits from C-12 into that PR |
| Fresh-install picker shows one race | P2 F-12 (Phase 1 item 5) | Menu-side integration only (Phase B references it) |

P2 §4's cold-start budgets are the acceptance bars for P4's Phases C-E: `--help` < 1 s, warm sim to lap 1 <= 20 s, 2nd+ menu run <= 5 s.

---

## 4. Phased duplicate-and-improve plan

Each phase is sized to become one GitHub sub-issue under a P4 epic (S / M / L). Order respects dependencies; A and B are independent of the duplicate and can land immediately. Nothing in any phase edits `run_simulation_cli.py` or `src/agents/` internals.

### Phase A - Editable quick wins, zero risk (S)
1. Dynamic version in the banner via `importlib.metadata`, dev-checkout fallback (C-11, `theme.py`).
2. Fix the H2H docstring drift (C-17, `f1_cli.py`).
3. Sort the race picker by calendar order instead of alphabetical (C-09, `pickers.py:206`).
- Deliverable: one small PR, no behavior risk, visible polish for demos.

### Phase B - Menu input hardening (M)
1. Driver picker becomes an arrow-select over the parquet-derived driver list for the chosen race, with free-text fallback (C-09, `pickers.py`).
2. Lap-range validation at prompt time (regex + reversed-range check) so malformed input never reaches the subprocess (C-08/C-09).
3. Provider preflight: mark OpenAI unavailable when `OPENAI_API_KEY` is absent; optionally ping LM Studio's endpoint before committing to a 60 s boot (C-09).
4. Integrate with (do not re-implement) P2 F-12's full-calendar picker + `ensure_race` download-on-select when that lands.
- Deliverable: no more 40-60 s boots wasted on typos. All in `scripts/cli/`, PMV untouched.

### Phase C - The duplicate skeleton: `run_simulation_cli_v2.py` (L)
1. Create the copy with the mandated header comment (`# DUPLICATE from scripts/run_simulation_cli.py L1-L2421 - CLI intocable, sync manual`).
2. Argparse and input validation before any heavy import (C-02, C-08: validated lap range, optional team via parquet, GP did-you-mean, `--version`, `--seed`, top-level error boundary with friendly Rich errors and distinct exit codes).
3. Decompose `run()` into `_setup_run` / `_load_race_data` / `_init_radio_source` / `_init_live_render` / `_run_lap_loop` / `_render_summary`; extract `_process_lap` to flatten the loop to two nesting levels; extract `_make_agent_stubs()`, `_build_status_row()`, header-grid helpers (C-03, all six memory backlog items).
4. Drop dead symbols and defensive noise on the copy (C-14, C-15).
5. **Preserve the Rich Live contract exactly**: history rows via `live.console.print`, fixed-height panel inside Live, never a growing Group in Live (`project_v09_cli_panel`). This is a stated invariant, not a refactor target.
6. Validate by golden-diff against the original (see §6). No behavior change intended in this phase; #166 remains reproduced on the copy until Phase D.
- Deliverable: a structurally clean, tested twin that renders byte-comparable output. Registered as a separate entry point (e.g. `f1-sim2`) or invoked by module path; `f1-sim` keeps pointing at the PMV.

### Phase D - Wire the copy to the shared engine (M, depends on P2b Phase 1)
1. Replace `_probe_core_agents` + orchestrator double-run with the engine's verbose `run_lap(...) -> (rec, agent_outputs, timings)` (C-04 / P2b F1/F10).
2. This closes **#166**: the engine's conditional-stage contract supersedes the broken 2-tuple unpack (C-01 / P2b F2). The xfail in `tests/test_cli_no_llm.py` flips to a passing test against the v2 path.
3. Adopt the engine's true no-LLM profile: no LLM clients constructed, deterministic outputs, guard-rails preserved (C-06 / P2b F8; the guard-rail policy at :1533-1560 is the reference behavior to keep).
4. Retire the two-shape result handling on the copy (C-16): one typed return for both modes.
- Deliverable: LLM-mode laps ~40-45% faster than the PMV, no-llm mode working and offline-true, engine-level goldens (Testing #182) as the regression net.

### Phase E - Boot staging + warm wizard loop (M, depends on P2 X-03 prewarm façade)
1. Staged parallel warmup in the copy via `prewarm(profile="sim")` with Rich status banners per stage (consumes P2 C3/F-08 remedy).
2. Whisper becomes opt-in (`--retranscribe`) once P2 F-02 ships transcripts in the HF dataset.
3. In-process wizard loop: after a race finishes, offer the next configuration without exiting Python; menu `runner.py` gains a "warm mode" that drives the v2 in-process, keeping `subprocess` as the fallback/isolation flag (C-05, Q4).
- Deliverable: P2 §4 budgets met on this machine's class of hardware (`--help` < 1 s; 2nd+ run <= 5 s).

### Phase F - Packaging + distribution home (M)
1. Move the v2 + menu code under `src/f1_strat_manager/cli/` (`sim.py`, `menu.py`, `pickers.py`, `theme.py`); absolute package imports; delete the `sys.path` hacks in the new code (C-10).
2. Consolidate palette/driver-color tokens into `src/f1_strat_manager/theme.py`, consumed by the new package and `data_cache._render_header`; PMV keeps its private copies (C-13).
3. Narrow `[tool.setuptools.packages.find]` toward package-only distribution for the new code; keep `scripts*` shipping unchanged so the PMV entry point still resolves (decision gate Q1 controls when `f1-sim` flips to the v2).
4. Distribution notes to verify in a scratch tool install: `uv tool install` end-to-end on a machine without a checkout, data landing in `~/.f1-strat/data/`, `F1_STRAT_DATA_ROOT`/`F1_STRAT_OFFLINE`/`F1_STRAT_NO_FIRST_RUN` honored. README/INSTALL command corrections are already owned by the Docs-accuracy audit (its F-01/F-02/F-14/F-16); coordinate, do not duplicate.
- Deliverable: a clean `pip`/`uv` surface where the only top-level import this project owns going forward is `f1_strat_manager` (long-term; `src`/`scripts` remain until the PMV is retired).

### Phase G - Year generality for 2026 (S-M, feeds epic #189)
1. All data paths derived from `--year` in the v2 (`raw/<year>`, `laps_featured_<year>.parquet` with existence fallback) (C-07).
2. Menu discovers available years from `<data_root>/raw/*/` and passes the year through (`f1_cli.py`, `pickers.py`, `runner.py`).
- Deliverable: the CLI stops being 2025-hardwired before the 2026-regulation retraining work needs it.

Dependency graph: A, B anytime -> C (standalone) -> D (needs P2b engine) -> E (needs P2 prewarm façade; the wizard loop part only needs C) -> F, G after C (F's entry-point flip after D+E prove parity).

---

## 5. Open questions (decision gates for Víctor)

- **Q1 - When does `f1-sim` flip to the v2?** Editing `pyproject.toml:115` changes what the command runs without touching the frozen file. Proposal: only after Phase D+E pass the §6 protocol on at least two GPs in both modes, and as its own PR so it is trivially revertible. Until then the v2 ships under a separate name (`f1-sim2`) or module invocation.
- **Q2 - Interim one-line hotfix for #166 on the PMV?** P2b F2 offers it as option (a) with explicit approval only. P4's recommendation stands with P2b: prefer the Phase D fix; take the hotfix only if a demo needs `--no-llm` before the engine lands.
- **Q3 - Where does the duplicate live first?** Recommendation: `scripts/run_simulation_cli_v2.py` during Phases C-E (keeps the golden-diff trivial and the mandated DUPLICATE header meaningful), then relocate into `src/f1_strat_manager/cli/` in Phase F. Alternative: start in the package immediately and accept a slightly noisier diff story.
- **Q4 - Warm wizard loop vs subprocess isolation.** In-process re-runs mean a crashed sim can take the menu down and GPU memory accumulates across runs. Proposal: warm mode as the default happy path with a `--isolated` flag (or automatic fallback after a crash) that restores today's subprocess behavior.
- **Q5 - How much LLM-mode goldenness is required?** LLM prose is nondeterministic. Proposal: structural goldens (columns, actions, counters, panel skeleton) for LLM mode using the FakeOpenAI stub from Testing #181, byte-level goldens only for `--no-llm` (post-D) and rendering-only laps.

---

## 6. Verification protocol (per phase, extends `project_cli_refactor_backlog` §Verificación)

1. **Original untouched:** `git diff --stat scripts/run_simulation_cli.py` empty in every P4 PR; the file is also a required-reviewer flag in PR descriptions.
2. **Golden-diff (Phase C):** run original and v2 back-to-back on the same scenario, e.g. `Melbourne NOR McLaren --no-real-radios --laps 5-15` (LLM mode against FakeOpenAI from Testing #181 for determinism) with a fixed terminal width (`COLUMNS=200`), capture output, diff. Acceptance: identical history rows, header, summary; panel differences only where a finding explicitly changed them. Avoid `--radio-every` in golden runs until `--seed` exists (C-18).
3. **No-llm regression (Phase D):** the `tests/test_cli_no_llm.py` xfail flips to pass against the v2 invocation; add the LLM-mode variant per Testing audit §CLI row (subprocess, exit 0, no `[ERROR]`, summary rendered).
4. **Boot budgets (Phase E):** re-run P2 §1's timed probes and record in the PR: v2 `--help` < 1 s; warm sim to lap 1 <= 20 s; 2nd wizard run <= 5 s. Compare against the PMV on the same machine in the same session.
5. **Live-panel invariant:** one long run (`--laps 1-40`) in a small terminal window to confirm no repaint-from-top regression (the `project_v09_cli_panel` failure mode) after any change near the Live region.
6. **Distribution smoke (Phase F):** scratch `F1_STRAT_DATA_ROOT` + `uv tool install` from a branch; confirm first-run download UX, sentinel race runs, no import of top-level `cli` in the new code path, and `f1-strat` wizard end-to-end.
7. **Menu changes (Phases A-B):** drive the wizard with a scripted tty session (or manually per the checklist): typo driver, malformed laps, missing API key; assert the failure happens at prompt time, not post-boot.

---

## 7. Alignment with sibling audits (one line each)

- **P2 (loading):** owns F-01/F-02/F-03/F-04/F-08/F-12/X-03; P4 consumes them in Phases C/E/B and corrects its stale H2H-2x claim (C-05).
- **P2b (compute):** owns the shared engine (F10) and the #166 root cause (F2); P4's Phase D is the CLI-side consumer P2b's plan explicitly reserves for "P4's deliverable".
- **Testing (epic #179):** #181 FakeOpenAI and #182 engine goldens are the regression net under Phases C-D; the CLI fixture-tier subprocess test is specified in that audit's §CLI row.
- **Docs-accuracy:** owns the broken README/INSTALL commands and repo-slug drift; P4 only flags that Phase F/G change the documented surface and must ping that backlog.
- **2026-reg (epic #189):** Phase G removes the CLI-side 2025 hardwiring it will need.
