# AUDIT - Packaging, distribution, release-engineering and CI/CD

**Auditor:** Fable 5 · **Date:** 2026-07-06 · **Repo:** `F1_Strat_Manager` (read-only pass, no code changed; one throwaway `uv build` + a released-wheel download were run to inspect artifact contents, then deleted)
**Scope:** the "how this ships and releases" machinery. Four areas: (1) distribution of the three surfaces (`f1-strat`/`f1-sim`/`f1-arcade`/`f1-streamlit` entry points, `uv tool install`, pipx, PyPI, HF-Hub-on-first-run); (2) the standalone-desktop end-state feasibility; (3) release engineering (release-please, the CHANGELOG-duplication bug, parent + `src/telemetry` submodule coherence, HF-artifact versioning); (4) CI/CD architecture (`.github/workflows/*`, job design, path-filter gating, submodule handling, caching, CUDA torch in CI, scanner-stack completeness).
**Refinement honored (Víctor):** the **CI/CD half is INCREMENTAL tuning of the current pipeline**, not a redesign - caching, path-filter gating, cancel-in-progress (already present), trimming redundant steps, submodule checkout depth, xdist only if it pays. **Packaging / distribution / desktop-bundling may be forward-looking.**
**Out of scope (owned elsewhere, cross-referenced only):**

| Topic | Owner |
|---|---|
| CPU-only install path, CI resolving CPU torch, ruff version pinning, dependency landmines (`fitz`/`pypdf`/`frozendict`), preflight doctor, fresh-install smoke workflow | **DevEx audit** (`AUDIT_DEVEX.md` DX-04/DX-05/DX-08/DX-07/DX-11) |
| Generic top-level package names (`scripts`/`src`/`cli`), per-change distribution smoke, banner version drift, the duplicate CLI | **P4 CLI audit** (`AUDIT_P4_CLI.md` C-10/C-11, Phase F) |
| HF revision pinning, dataset org migration, dataset card, data-release mechanics | **P5 data audit** (`AUDIT_P5_DATA_ENGINEERING.md` F-07, Phase 3) |
| Extending the security-scanner stack to the **submodule** repo | **Security audit** (`AUDIT_SECURITY.md` S-15/E3) |
| Test hermeticity, FakeOpenAI stub, engine goldens, mass-skip floor | **Testing audit** (`AUDIT_TESTING_QA.md`, epic #179) |

**Hard constraints honored in every remedy:** plan only, no code; backend stays FastAPI; LLM = OpenAI / LM Studio, never Anthropic; UNTOUCHABLE (duplicate before modifying / additive only): `scripts/run_simulation_cli.py`, `src/agents/` internals, `notebooks/**`, `legacy/**`.

---

## 0. Executive summary

The pipeline is healthy on the parts it covers: CI is green on `main` (3 jobs, ~1m55s wall), `cancel-in-progress` concurrency is already in place, uv wheel caching is keyed off `uv.lock`, and release-please + `publish-wheel` cut tags and attach artifacts on every `main` bump (latest: `v1.6.5`). The gaps are concentrated in three places, and one of them is a **silent ship-breaker.**

1. **The released wheel is missing the entire Streamlit surface, so `f1-streamlit` is dead on any `uv tool install`.** The `publish-wheel` job checks out **without submodules** (`release-please.yml:36`), so `src/telemetry/**` is empty at build time. Verified: the released `v1.6.5` wheel contains **0** `src/telemetry` files, while a local `uv build` (submodule checked out) bakes in **737** of them. `scripts/run_streamlit.py` resolves its entrypoint to `src/telemetry/frontend/app/main.py`, which is absent from the distributed artifact, so `f1-streamlit` exits with "cannot find Streamlit entrypoint". The wheel also silently changes shape depending on whether the builder happened to have the submodule checked out. This is the top finding and it decides a release-strategy question (should the CLI wheel even carry the web UI, or is Streamlit a separate Docker release per `project_release_strategy`?).

2. **Every release-please version bump will red the next CI run until someone re-locks by hand.** `release-please.yml` has **no `sync-uv-lock` catch-up job** (the exact caveat flagged in `project_release_please_ci_fails`). release-please bumps `pyproject.toml` + the manifest but never the root package's own `version` field inside `uv.lock`; CI installs with `uv sync --frozen`, which fails when the lock is out of sync with `pyproject.toml`. The uncommitted `M uv.lock` in the working tree today (lock self-version lagging at 1.6.1 while `pyproject.toml` is at 1.6.4) is exactly this manual chore being paid.

3. **The parent repo has almost none of the `PROJECT_BOOTSTRAP.md` §6 scanner stack.** No CodeQL workflow (default setup is `not-configured`), no OSV-Scanner, no gitleaks, no `pip-audit` job. Only Dependabot (pip + github-actions) and GitHub-native secret scanning / push protection / Dependabot alerts are on. For a repo heading to a public multi-repo ecosystem this is the biggest structural gap after the wheel bug.

Beyond those: CI has no path-filter gating (a docs-only or config-only PR pays the full ML-stack install + suite), Dependabot is missing the `gitsubmodule` and `npm` ecosystems (the `src/telemetry` pointer is never auto-bumped), the standalone-desktop end-state in the thesis conflicts with the current three-release plan and has no decision on record, and the code-release / submodule-gitlink / HF-data-revision trains have no coherence contract binding them together.

**Incremental-CI verdict up front (per the refinement):** the current 3-job design is sound; do **not** redesign it. The high-value increments are **path-filter gating** and **fixing the submodule-in-wheel build**. `pytest-xdist` is **not** worth adding (suite is ~51s wall and mostly data-gated skips; `PROJECT_BOOTSTRAP.md` §4 says skip xdist under 60s). Submodule shallow/`--filter=blob:none` is **not** needed yet (`src/telemetry` is small, ~MB); revisit only if it grows.

---

## 1. What exists today (the shipping machinery)

**Entry points** (`pyproject.toml:113-117`): `f1-strat` -> `scripts.f1_cli:main`, `f1-sim` -> `scripts.run_simulation_cli:main`, `f1-arcade` -> `src.arcade.main:main`, `f1-streamlit` -> `scripts.run_streamlit:main`. Build backend is `setuptools` (`pyproject.toml:1-3`); packages via `[tool.setuptools.packages.find] where=["."], include=["src*","scripts*"]` (`:140-142`). No `src/__init__.py` or top-level `src` package exists, so `src.arcade.main` resolves via PEP 420 namespace packaging and `src`/`scripts` land as generic importable top-level names (packaging hygiene owned by **P4 C-10 / DevEx DX-13** - referenced, not re-planned).

**Torch routing** (`pyproject.toml:150-168`): two explicit `[[tool.uv.index]]` blocks (`pytorch-cu128`, `pytorch-cpu`) and `[tool.uv.sources]` markers route Win/Linux to cu128, macOS to CPU. `uv.lock` reflects it (`uv.lock:1451-1457`). Plain pip ignores these tables (documented in the pyproject comment).

**Distribution surfaces:**
- **CLI** (`f1-strat`/`f1-sim`): pure-Python, ships in the wheel, first-run HF lazy download via `data_cache.ensure_setup` (`project_cli_distribution_plan`). The realistic install story.
- **Arcade** (`f1-arcade`): ships in the wheel (`src/arcade/**`, 25 files); native PySide6 + pyglet; per `project_release_strategy` R2 targets a container/Modal deploy.
- **Streamlit** (`f1-streamlit`): a thin `python -m streamlit run` wrapper (`scripts/run_streamlit.py`) pointing at the **submodule** app `src/telemetry/frontend/app/main.py`; per `project_release_strategy` R3 is a separate Docker release.

**Release engineering** (`release-please.yml`): `release-type: python`, manifest mode (`.release-please-manifest.json` = `{".":"1.6.4"}`), `include-v-in-tag`, `extra-files: ["pyproject.toml"]`, custom `changelog-sections` (feat/fix/perf/bench/eval/docs/refactor visible). `token: GITHUB_TOKEN` (PAT dropped, `project_release_please_ci_fails` RESOLVED). A `publish-wheel` job runs on `release_created`, does `uv build`, and `gh release upload dist/*`.

**CI** (`ci.yml`): push on `main`/`dev`/`test`/`feat|fix|docs/**`, PR to `main`/`dev`; `concurrency` cancel-in-progress present; 3 jobs - `test` (checkout `submodules: true`, setup-uv cache keyed `uv.lock`, py3.12, `uv sync --all-extras --frozen`, collected-count floor >=40, `pytest -v --cov`), `lint` (`uvx ruff check/format`, no submodules), `typecheck` (`uv sync --extra dev`, `.mypy_cache` keyed to `pyproject.toml`+`src/rag/**`, `mypy src/rag/`). Other workflows: `docs.yml` (gh-pages, path-filtered, version-injected), `auto-update-prs.yml` (rebases `area: deps`/`area: ci-cd` PRs), `labeler.yml` (`pull_request_target`, `actions/labeler@v6`).

**Submodule** (`src/telemetry`, `.gitmodules` -> `F1_Telemetry_Manager`, gitlink `3bf3b1b`): its own `pyproject.toml` (`f1-telemetry-manager` 0.1.0, static), its own minimal CI (`lint (python)` critical-select only + `test` deps-lite). **No** release-please, CHANGELOG, or manifest. PR triggers still name deleted branches `testVictor`/`testSanti`.

**Security posture (parent, verified via API):** Dependabot pip + github-actions; native secret scanning + push protection + Dependabot security updates + vulnerability alerts all enabled. CodeQL default-setup `not-configured`; no CodeQL/OSV/gitleaks/pip-audit workflow files.

---

## 2. Findings register (P0 -> P3)

Priorities: **P0** = a shipped artifact or release path is broken. **P1** = release/CI correctness or security gap that bites predictably. **P2** = friction, incompleteness, forward-looking distribution. **P3** = hygiene.

| ID | P | Area | Finding (what / why) | Evidence (anchors) | Size |
|---|---|---|---|---|---|
| **PK-01** | **P0** | Distribution | **The released wheel omits the whole `src/telemetry` submodule, so `f1-streamlit` is dead on any published install.** `publish-wheel` checks out with no `submodules:`, so the build sees an empty submodule dir. Verified: released `v1.6.5` wheel = **0** `src/telemetry` files; a local `uv build` with the submodule present = **737**. `scripts/run_streamlit.py` targets `src/telemetry/frontend/app/main.py`, absent from the artifact -> the wrapper prints "cannot find Streamlit entrypoint" and exits 2. The wheel's contents also depend on the builder's checkout state (non-reproducible). Forces a strategy decision: bundle the web UI in the wheel (checkout submodules) **or** drop `f1-streamlit` from the wheel and ship Streamlit only as the R3 Docker release (`project_release_strategy`). | `release-please.yml:31-49` (no `submodules:`); released `v1.6.5` assets; `scripts/run_streamlit.py` app-path resolution; `docker-compose.yml:28-46` (the intended Streamlit delivery) | **M** |
| **PK-02** | **P1** | Release eng | **No `sync-uv-lock` job -> every release bump reds the next `--frozen` CI run until a manual re-lock.** release-please bumps `pyproject.toml`+manifest but not the root `version` in `uv.lock`; `ci.yml` installs `--frozen`, which fails on a lock/pyproject mismatch. The uncommitted `M uv.lock` today (lock self-version 1.6.1 vs pyproject 1.6.4) is this chore. Known caveat, never fixed. | `release-please.yml` (no lock step); `uv.lock:1295-1296`; `ci.yml:57,104`; `project_release_please_ci_fails` ("Separate latent issue"); `PROJECT_BOOTSTRAP.md` §7 | **S** |
| **PK-03** | **P1** | CI security | **Parent repo has almost none of the bootstrap §6 scanner stack.** No CodeQL (default-setup `not-configured`), no OSV-Scanner, no gitleaks, no `pip-audit`. For a public-bound repo the SAST + cross-ecosystem CVE + secret-history layers are absent; only Dependabot + native secret scanning cover it. (Security audit E3 owns extending scanners to the **submodule**; this owns creating them on the **parent**.) | `gh api .../code-scanning/default-setup` -> `not-configured`; no `.github/workflows/{codeql,osv-scanner,gitleaks}.yml`; no `pip-audit` step in `ci.yml`; `PROJECT_BOOTSTRAP.md` §6 | **M** |
| **PK-04** | **P1** | CI (incremental) | **No path-filter gating: docs-only / config-only / one-ecosystem PRs pay the full ML-stack install + suite.** `test` runs `uv sync --all-extras --frozen` (multi-GB cu128 torch + nvidia wheels on a cache miss) and the full suite even when a PR touches only `docs/**` or `*.md`. `dorny/paths-filter@v3` per-job gating (checkout-first, `fetch-depth:0`, per `PROJECT_BOOTSTRAP.md` §4.1) would skip `test`/`typecheck` on non-code PRs while keeping the required check green. | `ci.yml:39-119`; `PROJECT_BOOTSTRAP.md` §4.1 | **M** |
| **PK-05** | **P1** | Deps automation | **Dependabot is missing `gitsubmodule` and `npm` ecosystems.** No `gitsubmodule` entry -> the `src/telemetry` gitlink is never auto-bumped, so submodule fixes only reach the parent by a manual pointer bump. No `npm` entry despite the submodule frontend `package-lock.json` and the `docs/` React SPA (no build, but still JS deps). | `.github/dependabot.yml` (pip + github-actions only); `.gitmodules`; `src/telemetry/package-lock.json`; `PROJECT_BOOTSTRAP.md` §5 | **S** |
| **PK-06** | **P2** | Release coherence | **Three-and-a-half release trains with no coherence contract.** Parent code (release-please, 1.6.x) + submodule code (static 0.1.0, bumped only by gitlink SHA, no release automation) + HF dataset (pinned to mutable `revision="main"`) all ship together, but nothing binds a wheel tag to a submodule SHA and an HF data revision. A released `v1.6.5` wheel pulls whatever is on HF `main` that day -> non-reproducible installs. (P5 F-07 owns the HF revision-pin **mechanism**; this owns the **contract** linking wheel tag <-> submodule gitlink <-> HF revision.) | `data_cache.py:58-59` (`revision="main"`); `src/telemetry/pyproject.toml` (static 0.1.0, no release-please); gitlink `3bf3b1b`; `AUDIT_P5_DATA_ENGINEERING.md` F-07 | **M** |
| **PK-07** | **P2** | Distribution (fwd) | **`publish-wheel` has no PyPI publish path, no attestation/provenance, no artifact smoke.** Distribution is git-URL / GitHub-release only; `uv tool install f1-strat-manager` (no URL) needs a PyPI publish (Trusted Publishing / OIDC). No sigstore attestation and no post-build "does the wheel import + do the entry points resolve" check (which would have caught PK-01). | `release-please.yml:31-49`; `project_cli_distribution_plan` ("Opcional post-TFG: subir a PyPI") | **M** (forward) |
| **PK-08** | **P2** | Desktop end-state | **The thesis standalone-desktop end-state (FastAPI + React bundle) is undecided and conflicts with the current three-release plan.** No decision on record; PyInstaller/Nuitka already rejected (`project_cli_distribution_plan`: 5+ GB with torch+CUDA). Needs a realistic recommendation and blocker list (see §3). | `project_release_strategy` (3 independent releases); `project_cli_distribution_plan` ("Lo que NO va: PyInstaller"); CLAUDE.md §1 (end-state) | **L** (forward, decision + spike) |
| **PK-09** | **P2** | CI reliability | **Network-dependent tests can red the required `test` check on transient upstream errors.** The `dev` run on 2026-07-06 failed only on `test_tiktoken_encoding_roundtrip` (HTTP 503 from `openaipublic.blob.core.windows.net`); `setfit`/`fitz` import probes also skip on env drift. A required check going red on someone else's 503 is a release-flow hazard. (Test hermeticity is owned by the Testing audit; the CI-gating lens is noted here.) | CI run 28820606106 (`dev`, failed); `ci.yml:69-70`; `AUDIT_TESTING_QA.md` (hermeticity) | **S** |
| **PK-10** | **P3** | Hygiene | **Submodule CI drift + no changelog.** `src/telemetry/.github/workflows/ci.yml` PR triggers still name deleted branches `testVictor`/`testSanti`; lint is critical-select only; no CHANGELOG/release automation to make submodule bumps legible in the parent. | `src/telemetry/.github/workflows/ci.yml:6`; `project_branch_protection` (those branches deleted 2026-07-05) | **S** |
| **PK-11** | **P3** | CI (incremental) | **Parent CI push triggers miss `chore/**` and `ci/**` branch globs.** A branch named `chore/x` or `ci/x` gets CI only once a PR to `main`/`dev` is opened; the submodule CI already lists both. Minor; align the globs. Also: `test` job's `--cov` output is collected but no coverage gate/artifact exists (informational only). | `ci.yml:5`; `src/telemetry/.github/workflows/ci.yml:5` | **S** |

---

## 3. The standalone-desktop end-state (PK-08 assessment)

The thesis end-state is "a standalone desktop app bundling FastAPI + the React build." Assessed against the real dependency weight (cu128 torch + nvidia wheels ~3-5 GB, Whisper 1.5 GB checkpoint, Qdrant on-disk + BGE-M3, PySide6, the React frontend in the submodule):

| Option | Verdict | Why |
|---|---|---|
| **PyInstaller / Nuitka single binary** | **Reject** | Already rejected in `project_cli_distribution_plan`. A frozen bundle with torch+CUDA is 5+ GB, brittle across CUDA driver versions, and gains nothing over a venv the user already needs the GPU for. |
| **Electron shell** | **Reject** | Ships a second Chromium (~150 MB) for a UI the docs SPA already renders in any browser; no benefit over Tauri; heaviest option. |
| **Tauri shell over a uv-managed backend** | **Best "real desktop app" if one is truly wanted** | Tauri (system webview, ~10 MB) wraps the existing React build and spawns a **locally-managed FastAPI** (uv venv or a sidecar). The ML weight stays in the venv + lazy HF download, exactly as today. Blocker: Tauri does not bundle Python/torch; you still ship or bootstrap a uv env, so it is a nicer launcher over the same install, not a self-contained binary. |
| **Packaged uv env + `uv tool install` + lazy HF (status quo, polished)** | **Recommended default** | This is what the CLI distribution plan already targets and what the ecosystem realistically supports. The "desktop app" feeling comes from a one-command install + first-run download UX, not a monolithic binary. |

**Recommendation:** keep the surfaces as **separate releases** (`project_release_strategy` R1/R2/R3), make `uv tool install` + lazy HF the canonical "app" experience, and only if a true desktop wrapper is demanded, spike **Tauri-over-uv-backend** (webview + spawned FastAPI) rather than any freeze-to-binary path. **Blockers to record:** (a) torch cu128 wheel size + CUDA runtime dependency, no clean cross-platform GPU story; (b) Whisper checkpoint + Qdrant index are runtime downloads, not bundleable sanely; (c) the React UI lives in the submodule, so any bundler must resolve PK-01 first; (d) macOS has no CUDA path (CPU wheels, untested per DevEx DX-04).

---

## 4. Phased, chunkable plan (each phase = one future GitHub sub-issue; S/M/L)

Ordering: fix the broken artifact and the release-correctness chore first (they gate trust in every release), then the security + CI increments (independent, high value), then the forward-looking distribution and desktop work. Nothing here edits any UNTOUCHABLE file - all changes are in `.github/`, `pyproject.toml` packaging config, `release-please*.json`, `docker-compose.yml`, or new workflow files.

**Phase 1 - Fix the release wheel + decide what it carries [M]** (PK-01)
- Decide (open question OQ-1): bundle the Streamlit UI in the wheel (add `submodules: true` + `fetch-depth: 0` to the `publish-wheel` checkout, confirm `src/telemetry/**` is intended wheel content, and make the build reproducible regardless of local checkout state) **or** drop `f1-streamlit` from `[project.scripts]` and ship Streamlit only as the R3 Docker image.
- Whichever way: add a post-build artifact smoke to `publish-wheel` (install the wheel into a scratch venv, assert every declared entry point resolves - would have caught this).
- Acceptance: the released wheel's entry points all resolve on a clean venv; wheel contents are identical whether or not the builder had the submodule checked out.

**Phase 2 - Release-engineering correctness [S]** (PK-02)
- Port LexFlow's `sync-uv-lock` catch-up step into `release-please.yml` (run `uv lock` after the version bump and commit, or gate the release PR on a re-lock), so `--frozen` CI never reds on a release bump.
- Acceptance: a simulated version bump leaves `uv.lock` in sync; the next `uv sync --frozen` passes with no manual edit; the working-tree `M uv.lock` chore disappears.

**Phase 3 - Parent security-scanner stack [M]** (PK-03; coordinate with Security E3 for the submodule)
- Add `codeql.yml` (python + javascript, `security-extended`), `osv-scanner.yml` (reusable workflow, path-filtered on manifests), `gitleaks.yml`, and a `pip-audit` job in `ci.yml` (`continue-on-error: true` while baselining, promote to required after a clean week). Add the analyse jobs to `setup-github.sh` required contexts once stable.
- Acceptance: all four run green on a PR; `code-scanning/default-setup` no longer the only SAST; `setup-github.sh` documents any new required contexts.

**Phase 4 - Incremental CI tuning [M]** (PK-04, PK-11; cross-ref DevEx DX-11)
- Add `dorny/paths-filter@v3` per-job gating (checkout-first + `fetch-depth: 0`, backend filter for `test`/`typecheck`, keep `lint` + `pip-audit` always-on) so docs/config-only PRs skip the ML-stack install. Include `.github/workflows/ci.yml` in the filter globs.
- Align push-trigger branch globs to include `chore/**` and `ci/**`.
- Explicitly do **not** add pytest-xdist (suite ~51s, mostly data-gated) and do **not** shallow the submodule (small); record both as deliberate non-actions. The CPU-torch-for-CI slimming stays with DevEx DX-11.
- Acceptance: a docs-only PR runs `lint` (+ scanners) only; a `src/**` PR runs everything; required checks still post success on skipped jobs.

**Phase 5 - Dependabot ecosystem completeness [S]** (PK-05)
- Add `gitsubmodule` (weekly, `area: data`) and `npm` (submodule frontend + `docs/`, `area: deps`) update entries; keep the existing torch/transformers/core-ML ignore rules.
- Acceptance: Dependabot opens a submodule-pointer PR when the submodule advances; npm bumps appear labelled.

**Phase 6 - Release-coherence contract [M]** (PK-06; builds on P5 F-07)
- Once P5 lands HF revision pinning, document and enforce the contract: a wheel tag records the submodule gitlink SHA and the HF dataset revision it was built against (embed in release notes or a `RELEASE_MANIFEST`), and released CLI builds pin `HF_DATASET_REVISION` to that revision (dev checkouts stay on `main`).
- Acceptance: a released wheel reproduces byte-identical critical files from its pinned HF revision; the release note names the submodule SHA + HF revision.

**Phase 7 - PyPI publishing + provenance [M, forward]** (PK-07)
- Add a `publish-pypi` step to the release workflow via PyPI Trusted Publishing (OIDC, no stored token); add build provenance/attestation; the artifact smoke from Phase 1 gates the publish.
- Acceptance: `uv tool install f1-strat-manager` (no git URL) works; the PyPI page shows attestation.

**Phase 8 - Desktop end-state decision + spike [L, forward]** (PK-08)
- Record the decision (OQ-2): keep three separate releases (recommended) or commit to a desktop wrapper. If the latter, spike Tauri-over-uv-backend behind the resolved PK-01, and document the four blockers from §3.
- Acceptance: a one-page decision doc; if spiked, a Tauri shell launching the React build against a locally-spawned FastAPI on one platform.

**Phase 9 - CI reliability + hygiene [S]** (PK-09, PK-10)
- Coordinate with the Testing audit to mark network-dependent tests (`test_tiktoken_encoding_roundtrip`, external-hub import probes) so a transient upstream 503 cannot red the required `test` check (skip-on-network-error or move behind a non-gating job).
- Fix the submodule CI branch triggers (drop `testVictor`/`testSanti`); optionally add a lightweight CHANGELOG/release note to the submodule so pointer bumps are legible.
- Acceptance: a simulated upstream 503 does not fail the required check; submodule CI triggers reference only live branches.

Dependency notes: Phases 1-5 are independent and immediate. Phase 6 depends on P5 F-07. Phase 7 depends on Phase 1 (artifact smoke) + Phase 2 (clean lock). Phase 8 depends on Phase 1. Phase 9 coordinates with the Testing audit but blocks nothing.

---

## 5. Open questions (need Víctor's decision)

1. **OQ-1 - Does the CLI wheel carry the Streamlit UI, or not?** Bundle `src/telemetry` in the wheel (checkout submodules in `publish-wheel`, `f1-streamlit` works from `uv tool install`) vs drop `f1-streamlit` from the wheel and keep Streamlit as the R3 Docker-only release (`project_release_strategy`). This decides Phase 1's shape. Recommendation: if `f1-streamlit` is advertised as an entry point it must work from the released wheel; otherwise remove the entry point to stop shipping a dead command.
2. **OQ-2 - Standalone-desktop stance:** three separate releases (recommended) or a Tauri-over-uv-backend wrapper spike? Sets Phase 8's size.
3. **OQ-3 - Does `uv tool install git+...` init submodules?** DevEx OQ-1 already asks whether it honors `[tool.uv.sources]`; the sibling question here is whether it fetches the submodule at all. If not, `f1-streamlit` (and any telemetry-dependent path) is dead even from a git install, not just the wheel. Must be tested on a clean machine before Phase 1 writes the final wording.
4. **OQ-4 - PyPI publish now or post-TFG?** Trusted Publishing is cheap once the artifact smoke exists, but a multi-GB-dependency package on PyPI invites "why is install so slow" issues. Gate to a tagged milestone?
5. **OQ-5 - HF revision-pin coupling:** should the wheel tag hard-pin the HF revision (fully reproducible, manual bump per data release) or record-but-not-enforce it (installs stay fresh)? Aligns with P5 OQ-4; Phase 6 needs the answer.
6. **OQ-6 - Submodule release automation:** give `src/telemetry` its own release-please/CHANGELOG (legible bumps, second release train to maintain) or keep it gitlink-SHA-only (simpler, opaque)? Phase 9 depends on this.

---

## 6. Verification protocol (when this plan is executed)

- **Wheel integrity (Phase 1, the acceptance bar):** on a clean venv, `pip install` (or `uv tool install`) the built wheel, then assert **every** `[project.scripts]` entry point imports and resolves (`f1-strat --help`, `f1-sim --help`, `f1-arcade` menu import, `f1-streamlit` locates its app path). Confirm the wheel's `src/telemetry` file count matches the decision (0 if excluded, full if bundled) regardless of the builder's local checkout state. Compare `unzip -l` of the CI-built wheel against a local build.
- **Release-lock sync (Phase 2):** simulate a `feat:` on `main`, let release-please open the release PR, confirm `uv.lock` is re-synced in that PR and `uv sync --frozen` passes on the release commit with no manual edit.
- **Scanner stack (Phase 3):** CodeQL / OSV / gitleaks / pip-audit each post a check on a scratch PR; a planted dummy secret trips gitleaks/push-protection; a known-CVE pin trips OSV/pip-audit; `setup-github.sh` re-run reflects any new required contexts.
- **Path-filter gating (Phase 4):** a docs-only PR shows `test`/`typecheck` skipped (green) and `lint` run; a `src/**` PR runs all; a PR editing `ci.yml` re-triggers the gated jobs; required checks never stick in "expected".
- **Dependabot (Phase 5):** advance the submodule on a branch and confirm a `gitsubmodule` PR opens; confirm an npm bump appears labelled.
- **Release coherence (Phase 6):** a fresh `uv tool install` of a released tag reproduces byte-identical critical files from the pinned HF revision (reuses the P5 verification); the release note names the submodule SHA + HF revision.
- **CI reliability (Phase 9):** inject a simulated upstream 503 into the network test path and confirm the required `test` check does not go red.
- **Standing cross-check:** nothing touching the boot/data path ships without the P2 §6 re-timing probes; nothing touching sim behavior ships without the no-LLM regression diff (`python scripts/run_simulation_cli.py Sakhir HAM Mercedes --no-llm --laps 1-10`, post-#166). Both remain owned by their audits; referenced here so packaging PRs that move data resolution do not skip them.
