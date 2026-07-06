# AUDIT - DevEx & contributor onboarding (the first 30 minutes)

**Auditor:** Fable 5 · **Date:** 2026-07-06 · **Repo:** `F1_Strat_Manager` (read-only pass, no code changed)
**Scope:** the end-to-end experience of a brand-new contributor or user: `git clone` + submodule init, `uv sync --all-extras` (including the CUDA-pinned torch on non-CUDA machines), `.env` setup, first run (HF lazy download, Whisper prewarm, boot), the Docker path, and running tests / lint / typecheck as documented.
**Out of scope (owned elsewhere, cross-referenced only):** boot/download mechanics and budgets (P2 loading audit, `AUDIT_P2_LOADING.md`, issues #167/#168); wrong or stale documented commands and claims (docs-accuracy audit, `AUDIT_DOCS_ACCURACY.md`); download UX mechanisms, data-root resolvers and data-status endpoints (P5 data-engineering audit, `AUDIT_P5_DATA_ENGINEERING.md`); CLI packaging internals and the duplicate-and-improve plan (P4 CLI audit, `AUDIT_P4_CLI.md`).
**Hard constraints honored in every remedy:** plan only, no code; LLM = OpenAI / LM Studio, never Anthropic; UNTOUCHABLE (duplicate before modifying / additive entry points only): `scripts/run_simulation_cli.py`, `src/agents/` internals, `notebooks/**`, `legacy/**`.

---

## 1. Framing: docs fix vs setup/tooling gap

Each finding below is tagged one of:

- **[docs]** - the fix is words. Where the docs-accuracy audit already registered the exact line, this audit references its finding ID and does not re-plan it.
- **[tooling]** - a real setup, packaging, dependency, or automation gap that no docs edit can close.
- **[mixed]** - both, split explicitly inside the finding.

Ownership boundaries with the sibling audits:

| Topic | Owner |
|---|---|
| Broken copy-paste commands (`f1-sim VER Melbourne ...`, `--lap-range`, bootstrap import), stale bootstrap/provider/slug/version claims | **Docs-accuracy audit** (F-01/F-02/F-03, F-11/F-12/F-14/F-15/F-16/F-20) |
| Silent 7-8 GB first-run download, doubled metadata sweep, import-time model loading, Whisper mid-boot pulls, boot budgets | **P2 loading audit** (F-01/F-02/F-04, X2; issues #167/#168) |
| Data-manager facade, backend data-status endpoints, second resolver collapse, `download_data.py` 31.7 GB trap, full-calendar picker | **P5 data audit** (F-06, F-12; Phase 2) |
| Top-level `scripts`/`cli` packaging names, distribution smoke per change, duplicate CLI | **P4 CLI audit** (C-10, C-12, Phase F) |
| `--no-llm` broken (3-tuple unpack) | **Issue #166** (open, fix lands on the P4 duplicate) |
| **The newcomer walk itself: quickstart integrity per path, dependency landmines, CPU-only install story, dev-env-by-copy-paste, preflight/doctor, toolchain parity, standing fresh-install verification** | **THIS audit** |

---

## 2. Executive summary

Modeling a fresh contributor on a clean machine, all three documented entry paths fail inside the first 30 minutes, each before the product ever runs:

1. **The Docker/Streamlit quickstart fails three times in a row.** `git clone && docker compose up` (README.md:88-91, INSTALL.md:80-84) aborts first because `.env` does not exist yet (`docker-compose.yml:10-11` declares `env_file: ./.env` as required), then because the build context `./src/telemetry` is an empty directory (the clone commands omit `--recurse-submodules`; only CONTRIBUTING.md:26 initializes the submodule), and finally, even fully booted, the backend has no data: `./data` is empty on a fresh clone (gitignored, HF-only) and mounted read-only (`docker-compose.yml:19`), so nothing inside the container can fetch it and no bootstrap step is documented for this path.
2. **There is no working zero-cost verification command.** INSTALL's post-install sanity check is broken twice at the docs level (wrong positional order, nonexistent `--lap-range` flag; docs audit F-01/F-02), and the corrected command still fails because `--no-llm` itself has been broken since commit `bfe5b46` (open issue #166, 3-tuple unpack in `_run_no_llm`). Until #166 lands, a newcomer cannot confirm their install without an OpenAI key or a running LM Studio.
3. **`uv sync` installs landmines.** The dependency list ships `fitz 0.0.1.dev2`, a 2017 dummy package that drags nibabel, nipype, pyxnat, httplib2 and configobj into every install (uv.lock:1540-1557) and is imported by nothing live; `pypdf`, which `scripts/build_rag_index.py:34` actually imports, is neither declared nor locked; and `experta` pins `frozendict==1.2` (uv.lock:1610-1612), which cannot import on any supported Python (3.10-3.12, `collections.Mapping` removal), a fact acknowledged only in a pyproject comment ("install manually after") that every `uv sync` silently reverts.
4. **Non-CUDA machines are second-class with no documented path.** `[tool.uv.sources]` routes Windows and Linux unconditionally to the cu128 index (pyproject.toml:160-168); a CPU-only Linux contributor downloads the full nvidia wheel set (about 3-5 GB, uv.lock nvidia-* entries) they can never use, macOS is routed to CPU wheels but untested, and the INSTALL claim that `uv tool install` resolves the CUDA wheel via `[tool.uv.sources]` (INSTALL.md:20-23) is unverified.
5. **Nothing checks the environment before the pain starts.** There is no preflight/doctor, the fresh `uv tool install` first-run flow has never been verified end-to-end on a clean machine (memory `project_hf_models_restructure`, "Pendiente de testear"), and the felt first-run wall (silent 7-8 GB pull, 30 s import tax, surprise Whisper checkpoint) is fully diagnosed by P2 but not yet expectation-managed anywhere a newcomer reads.

The good news: the pieces to fix this are cheap and mostly editable surface. The quickstarts are one docs+compose PR; the dependency landmines are one re-lock; the preflight doctor composes with P5's planned data-manager facade instead of competing with it.

---

## 3. The first 30 minutes, walked end to end

What actually happens today on a clean machine, per documented path. "Blocked" means the step fails with an error the newcomer must debug; "wall" means a long silent wait with no progress signal.

### Path A - contributor clone (CONTRIBUTING.md:23-35)

| Step | Command (as documented) | What happens | Verdict |
|---|---|---|---|
| A1 | `git clone .../F1_Strat_Manager.git` | Works (old slug redirects; docs audit F-14) | OK |
| A2 | `git submodule update --init --recursive` | Works; submodule is small | OK |
| A3 | `uv sync` | Resolves and installs about 6-10 GB of wheels (torch cu128 on Win/Linux). **No pytest, no voice deps** - extras are not included, though the comment says "installs every dependency" (CONTRIBUTING.md:27) | Wall + trap |
| A4 | `cp .env.example .env` + key | Works; but the example documents a dead variable and omits the live `F1_STRAT_*` overrides (DX-09) | Trap |
| A5 | `python -c "from f1_strat_manager.data_cache import ..."` | **ModuleNotFoundError** - wrong import root (docs audit F-03) | Blocked |
| A6 | `pytest tests/ -x` (PR checklist) | **`pytest` not installed** after plain `uv sync` (needs `--all-extras` or `--extra dev`) | Blocked |

### Path B - `uv tool install` user (README.md:72-77, INSTALL.md:29-51)

| Step | What happens | Verdict |
|---|---|---|
| B1 | `uv tool install git+...` builds and installs the wheel + full ML stack into a tool venv (multi-GB) | Wall (undocumented size) |
| B2 | Whether the tool venv gets the cu128 torch wheel at all depends on uv honoring `[tool.uv.sources]` for git installs; INSTALL.md:20-23 asserts it, never verified (OQ-1) | Unknown |
| B3 | `f1-strat` first run: 7-8 GB HF snapshot behind a spinner with progress disabled, metadata pass twice (P2 F-04, issue #168) | Wall |
| B4 | INSTALL verification: `f1-sim VER Melbourne "Red Bull Racing" --year 2025 --no-llm --lap-range 1 1` fails on arg order, then on the flag (docs audit F-01/F-02), then on #166 | Blocked x3 |
| B5 | First real sim: about 30 s import tax before anything paints (P2 F-01, issue #167), Whisper 1.5 GB checkpoint mid-boot on a first GP (P2 X2) | Wall |

### Path C - Docker/Streamlit (README.md:86-91, INSTALL.md:78-92)

| Step | What happens | Verdict |
|---|---|---|
| C1 | `git clone ... && cd ... && docker compose up` | **Compose aborts: `env file ./.env not found`** (docker-compose.yml:10-11; no `cp .env.example .env` step in this path) | Blocked |
| C2 | After creating `.env`: **build fails, `./src/telemetry` is empty** - clone command has no `--recurse-submodules` and this path never mentions the submodule | Blocked |
| C3 | After submodule init: images build, both ports open, but `data/` is empty and mounted `:ro` (docker-compose.yml:14-19), so the backend 404s with no remediation hint (P5 F-06) and `ensure_setup` could not write even if something called it | Blocked (silent) |

---

## 4. Findings register (P0 - P3)

Priorities: **P0** = a documented path a newcomer will follow that fails or dead-ends. **P1** = makes the environment wrong or unverifiable even when commands are followed correctly. **P2** = friction, drift, and misdirection. **P3** = polish.

| ID | P | Tag | Finding (what a newcomer hits, why) | Evidence (anchors) | Size |
|---|---|---|---|---|---|
| DX-01 | **P0** | mixed | **Docker/Streamlit quickstart fails 3x on a fresh clone:** (a) required `env_file: ./.env` missing - compose aborts before building; (b) clone commands omit `--recurse-submodules`, so build context `./src/telemetry` is empty ("failed to read dockerfile"); (c) `./data` is empty (gitignored, HF-only) and mounted read-only, and no data bootstrap exists or is documented for this path. (a)+(b) are docs+compose edits; (c) is a tooling gap whose mechanism (backend `/data/status` + `ensure` endpoints, data-manager facade) is owned by P5 Phase 2 item 8 - this audit only requires the documented host-side bootstrap one-liner in the interim | `README.md:86-91`; `INSTALL.md:80-84`; `docker-compose.yml:10-11,14-19`; `src/telemetry/backend/Dockerfile`; P5 F-06 | S (a+b) / S interim (c) |
| DX-02 | **P0** | mixed | **No zero-cost way to verify an install.** The documented sanity command is broken twice at the docs level (docs audit F-01/F-02 own the wording) and the corrected form still crashes: `--no-llm` unpacks 2 values from a 3-tuple since `bfe5b46` (issue #166, open; PMV untouchable so the fix lands on the P4 duplicate). Every first-timer either burns OpenAI tokens, must stand up LM Studio, or concludes the project is broken | `INSTALL.md:131-132`; issue #166; `scripts/run_simulation_cli.py` (untouchable); docs audit Phase 1 | S (docs) + cross-ref #166 (M, owned elsewhere) |
| DX-03 | **P0** | docs | **The first-run wall is undocumented.** Silent 7-8 GB pull under a spinner, about 30 s import before first paint, surprise 1.5 GB Whisper checkpoint mid-boot on a first GP. Mechanics and fixes are fully owned by P2 (F-04/#168, F-01/#167, X2) - the DevEx gap is that no doc a newcomer reads sets size/time expectations, so the correct mental model is "hung" | P2 §1-§2; `INSTALL.md:8-24` (prerequisites list no disk/time budget) | S |
| DX-04 | **P1** | tooling | **CPU-only and non-CUDA machines have no supported install.** `[tool.uv.sources]` sends `linux` and `win32` unconditionally to the cu128 index; a CPU-only Linux contributor downloads the full nvidia-* wheel set (uv.lock lines 1042-1072, 3594+) for nothing; the only escape is "edit the URL" in a pyproject comment. macOS resolves CPU wheels but the path is untested. INSTALL's claim that `uv tool install` applies `[tool.uv.sources]` from a git install is unverified (OQ-1) | `pyproject.toml:144-168`; `uv.lock` nvidia-* entries; `INSTALL.md:20-23` | M |
| DX-05 | **P1** | tooling | **Dependency landmines in the core list:** (a) `fitz==0.0.1.dev2` is a 2017 placeholder that pulls a neuroimaging stack (nibabel, nipype, pyxnat, httplib2, configobj) into every install and is imported by nothing live (only the tier-2 import list references it); (b) `pypdf`, actually imported by the RAG index builder, is not declared and not in the lock - `python scripts/build_rag_index.py` = ModuleNotFoundError on any fresh env; (c) `experta` locks `frozendict==1.2`, which raises on import on every supported Python; the workaround lives in a pyproject comment and is undone by every `uv sync`; the tier-2 dep test explicitly excludes experta for this reason. All three are dependency-only fixes; `src/agents/` stays untouched (its experta modules are imported by nothing live) | `pyproject.toml:50,107`; `uv.lock:1540-1557,1610-1612`; `scripts/build_rag_index.py:34,212`; `tests/test_dep_imports.py:331-336,345`; grep: no live `import fitz`, no live import of `src/agents/strategy_agent` | S/M |
| DX-06 | **P1** | mixed | **Following the dev setup verbatim does not yield a working dev env.** `uv sync` without extras installs no pytest, so the PR checklist's `pytest tests/ -x` fails "command not found"; the data-bootstrap one-liner fails to import (docs audit F-03 owns the exact command); CONTRIBUTING's "installs every dependency" claim is wrong. The canonical block should be `uv sync --all-extras` (what CI's test job and CLAUDE.md §5 use) plus the corrected bootstrap line | `CONTRIBUTING.md:27,34,118`; `.github/workflows/ci.yml:57`; `pyproject.toml:119-138` | S |
| DX-07 | **P1** | tooling | **No preflight, and the fresh-install flow has never been verified end-to-end.** Nothing checks Python version, `.env` presence, submodule presence, disk headroom (about 15-20 GB all-in), GPU/CUDA optionality, or HF reachability before the multi-GB commitment; the `uv tool install` first-run + sentinel-race flow is untested on a clean machine (memory "Pendiente de testear", 2026-04-09, never closed). P4's Phase F smoke is per-change verification; there is no standing, repeatable onboarding check | memory `project_hf_models_restructure`; `src/f1_strat_manager/data_cache.py:194-231` (checks data only, after commit); P4 audit Phase F | M |
| DX-08 | **P2** | tooling | **Lint toolchain is unpinned and drifts.** CI runs `uvx ruff` (latest-at-runtime) while the dev extra floor is `ruff>=0.0.286` (2023-era); a new ruff release can redden `lint` on untouched code, and a contributor's local ruff can disagree with CI on format. One pinned version used by both (`uvx ruff@X.Y.Z` + matching dev-extra pin) closes it | `.github/workflows/ci.yml:84-88`; `pyproject.toml:125` | S |
| DX-09 | **P2** | mixed | **`.env.example` documents a dead variable and omits the live ones.** `LM_STUDIO_BASE_URL` is read by nothing (backend reads `LM_STUDIO_HOST`, `llm_service.py:26`; agents hardcode `http://localhost:1234/v1` defaults); the file omits `F1_STRAT_DATA_ROOT` / `F1_STRAT_OFFLINE` / `F1_STRAT_NO_FIRST_RUN` (the documented power-user overrides) and `HF_TOKEN`. Provider-default contradictions across docs are owned by docs audit F-11 | `.env.example:17-18`; `src/telemetry/backend/services/chatbot/llm_service.py:26`; `src/agents/tire_agent.py:954,985`; `data_cache.py:24-35` | S |
| DX-10 | **P2** | mixed | **Error paths misdirect newcomers.** The wizard's no-races fallback says "Run scripts/download_data.py first" - the uncurated 31.7 GB pull (P5 F-12 owns that script); the backend's missing-parquet error carries no remediation (P5 F-06). Caller-side message fixes are cheap and editable; the mechanisms stay with P5 | `scripts/f1_cli.py:104-107`; `backend/utils/laps_cache.py:37` | S |
| DX-11 | **P2** | tooling | **CI cold-start pays the CUDA wheel set on CPU-only runners.** The `test` job's `uv sync --all-extras --frozen` on ubuntu-latest installs cu128 torch plus every nvidia-* wheel (GBs) on each cache miss; a CPU torch resolution for CI would cut cold installs substantially. Blocked on the DX-04 decision (single lockfile constraint) | `.github/workflows/ci.yml:40-57`; `uv.lock` nvidia-* entries | M |
| DX-12 | **P3** | docs | **Prerequisites are silent on budgets and platform gotchas:** no disk (about 15-20 GB total), RAM/VRAM, or first-boot time expectations; no "GPU optional, CPU fallback works" statement; no Windows long-path (`MAX_PATH`) note for the deep HF snapshot trees; ffmpeg only needed for backend voice (Dockerfile installs it) but never mentioned for local voice work | `INSTALL.md:8-24`; `src/telemetry/backend/Dockerfile:8-11` | S |
| DX-13 | **P3** | tooling | **Installed top-level packages are generic (`scripts`, `cli` via sys.path insert)** - collision-prone in the tool venv and confusing to contributors. Owned by P4 audit C-10 (Phase F); registered here only so the onboarding lens is on record | `pyproject.toml:140-142`; `scripts/f1_cli.py:53-61`; P4 C-10 | cross-ref |

---

## 5. Phased, chunkable plan (each phase = one future GitHub sub-issue; S/M/L)

Ordering rationale: unblock the three documented paths first (words and one compose edit), then make the installed environment truthful (deps, dev block, env vars), then the structural investments (CPU story, doctor, standing smoke). Phases 1-3 are independent of every other audit's execution; Phases 4-5 have explicit cross-audit hooks.

**Phase 1 - Unbreak the three quickstarts [S]** (DX-01 a+b, DX-02 docs sliver, DX-03, DX-12)
- README + INSTALL: all clone commands become `git clone --recurse-submodules ...` (or add the submodule line); every path gains the `cp .env.example .env` step; coordinate with docs-accuracy Phases 1 and 7 so this is one PR, not two conflicting ones.
- `docker-compose.yml`: mark `env_file` optional (`required: false`) OR keep it required and document the step; add the interim host-side data bootstrap one-liner (corrected `ensure_setup` import per docs audit F-03) to the Streamlit section with a note that the container cannot self-fetch (read-only mount).
- INSTALL prerequisites: add the budget table (download sizes, disk, first-boot time, "GPU optional") and the Windows long-path note.
- Acceptance: Path C walkthrough (bare clone, follow README verbatim) reaches both ports with data present; no step errors.

**Phase 2 - Dependency hygiene and re-lock [S/M]** (DX-05)
- Remove `fitz`; add `pypdf` (the RAG builder's real import); resolve the experta/frozendict pair: preferred = drop `experta` from dependencies (nothing live imports it; the modules that do are dead code inside untouchable `src/agents/`, which stays byte-identical), fallback = a `frozendict>=2.4.0` constraint/override so the lock stops shipping a package that cannot import.
- Extend `tests/test_dep_imports.py` tier 3: `import pypdf` API check; a guard that fails if `nipype`/`nibabel`/`pyxnat` ever reappear in the lock (dummy-package canary).
- Acceptance: fresh `uv sync --all-extras` contains no neuroimaging wheels; `python -c "import pypdf"` and `python scripts/build_rag_index.py --help` work; CI green.

**Phase 3 - A dev environment by copy-paste [S]** (DX-06, DX-09, DX-10)
- CONTRIBUTING dev-setup block: `uv sync --all-extras` (with the "every dependency" claim corrected), the fixed bootstrap one-liner (lands via docs audit Phase 1; reference, don't duplicate), and a "expect a 7-8 GB pull on first bootstrap" line.
- Rewrite `.env.example`: drop `LM_STUDIO_BASE_URL`, add `LM_STUDIO_HOST` (backend), `F1_STRAT_DATA_ROOT`, `F1_STRAT_OFFLINE`, `F1_STRAT_NO_FIRST_RUN`, `HF_TOKEN`, each with a one-line comment naming which surface reads it.
- Retarget the two misdirecting error messages: `f1_cli.py:106` points at the first-run flow / `ensure_setup` instead of `download_data.py`; backend missing-parquet message names the bootstrap command (full remediation endpoint stays with P5 Phase 2).
- Acceptance: Path A walkthrough verbatim ends with `uv run pytest` collecting and passing (data-gated skips fine).

**Phase 4 - A supported no-CUDA install path [M]** (DX-04, DX-11)
- Decide the support stance (OQ-2), then implement the chosen mechanism: documented CPU recipe (env-scoped index override or an extras split) for Linux/Windows CPU-only machines, and a tested macOS statement.
- Verify OQ-1 (`uv tool install` + `[tool.uv.sources]`) on a clean machine and correct INSTALL.md:20-23 if the claim does not hold.
- Optional follow-up once the mechanism exists: CI `test` job resolves CPU torch, cutting multi-GB cold installs (DX-11).
- Acceptance: a CPU-only Linux container completes `uv sync` without nvidia wheels and runs the sentinel race `--no-llm` (post-#166); INSTALL documents the switch in one table.

**Phase 5 - Preflight doctor + standing fresh-install smoke [M]** (DX-07, closes DX-03's loop)
- Additive `scripts/doctor.py` (surfaced as `f1-strat --doctor` from the editable wizard, or a fifth console script): checks Python version, `.env` presence + provider reachability (OpenAI key set / LM Studio port answering), submodule populated, disk headroom vs the budget table, GPU/CUDA presence (informational), data status. Data checks delegate to P5's `data_manager.status()` when it lands; until then, `data_cache.is_first_run()` + `_CRITICAL_MODEL_FILES` cover it.
- A `workflow_dispatch` (optionally monthly) GitHub workflow: scratch `F1_STRAT_DATA_ROOT`, `uv tool install` from the ref under test, pattern-limited first-run download, sentinel race `--no-llm --laps 1-1`; this finally closes the "Pendiente de testear" from memory and becomes the standing onboarding regression (P2 §6's first-run protocol, automated). Frequency/cost per OQ-4.
- Acceptance: doctor exits non-zero with an actionable line per missing prerequisite; the smoke workflow passes from a clean runner.

**Phase 6 - Toolchain parity pinning [S]** (DX-08)
- Pin ruff to one exact version in both CI (`uvx ruff@X.Y.Z`) and the dev extra; document "local = CI" quality commands in CONTRIBUTING (ruff check, ruff format --check, mypy `src/rag/`, pytest) as a single block.
- Acceptance: `grep` finds exactly one ruff version across `ci.yml` + `pyproject.toml`; a deliberate local downgrade reproduces CI's verdict.

Dependency notes: Phase 1 and 2 are independent and immediate. Phase 3 depends only on docs-accuracy Phase 1 landing (shared command fixes). Phase 5's smoke needs #166 fixed for the `--no-llm` leg (interim: run the sentinel with `--provider openai --laps 1-1` behind a secret, or assert boot-to-lap-1 only). Phase 4 gates the CI slimming half of DX-11.

---

## 6. Open questions (need Víctor's decision)

1. **OQ-1 - Does `uv tool install git+...` actually honor `[tool.uv.sources]`?** INSTALL.md:20-23 asserts the CUDA wheel is auto-selected in tool-install mode; if uv ignores project sources for git installs, Path B users on Windows silently get the default PyPI torch. Must be tested on a clean machine before Phase 4 writes the final wording.
2. **OQ-2 - Support stance for CPU-only Linux/Windows and macOS:** officially supported (docs + tested recipe + maybe extras split) or explicit best-effort ("CUDA machine recommended, CPU fallback undocumented")? Phase 4's size depends on this.
3. **OQ-3 - experta: drop the dependency or override frozendict?** Nothing live imports it and its `src/agents/` consumers are dead code (untouchable but never imported); dropping is cleaner, overriding is zero-risk. Either way `src/agents/` is not edited.
4. **OQ-4 - Standing smoke cost:** the fresh-install workflow pulls 7-8 GB from HF per run (pattern-limiting can cut it to about 1-2 GB: models minus NLP backbones + sentinel race). Monthly scheduled, manual-only, or gated to release branches?
5. **OQ-5 - Docker path investment level:** given the Streamlit surface is slated for replacement (frontend migration epic #25), is Path C worth more than Phase 1's unbreak + interim bootstrap line (e.g. a compose-profile bootstrap service), or should INSTALL steer newcomers to the local `uv run f1-streamlit` flow until the SPA lands?
6. **OQ-6 - Where the doctor lives:** flag on the editable wizard (`f1-strat --doctor`), a fifth console script (`f1-doctor`), or fold into the P5 data-manager CLI when that ships? One home, not three.

---

## 7. Verification protocol (when this plan is executed)

- **Clean-machine walkthroughs (the acceptance bar for Phases 1-4):** scripted runs of Paths A, B and C exactly as documented, on (a) Windows + CUDA and (b) a CPU-only Linux container. Record wall-clock clone-to-first-lap before/after. Acceptance: zero steps fail without an actionable message; no silent phase over 5 s (P2 §4 budget); Path C reaches ports 8000/8501 with data present.
- **Dependency assertions (Phase 2):** `uv pip list` contains no `nipype|nibabel|pyxnat|fitz`; `python -c "import pypdf"` succeeds; the frozendict/experta resolution imports (or the dep is gone); `tests/test_dep_imports.py` tier-3 canaries green on CI.
- **Dev-env assertion (Phase 3):** on a fresh clone, the CONTRIBUTING block verbatim ends with `uv run pytest` collecting >= 40 nodes (CI floor) and `uvx ruff@<pin> check .` matching CI's verdict.
- **Command lint (shared with docs-accuracy Phase 1 acceptance):** every fenced shell command in README/INSTALL/CONTRIBUTING at least parses (`--help` or dry-run) against the current wheel.
- **First-run smoke (Phase 5):** the workflow-dispatch job passes from a scratch `F1_STRAT_DATA_ROOT` on a clean runner: single progress-visible download, sentinel race boots to lap 1, exit 0. Re-run after any change near `data_cache.py`, `pyproject.toml` dependency edits, or compose files.
- **Nothing near the boot path ships without** the P2 §6 re-timing probes, and nothing touching sim behavior ships without the established no-LLM regression diff (`python scripts/run_simulation_cli.py Sakhir HAM Mercedes --no-llm --laps 1-10`, post-#166).
