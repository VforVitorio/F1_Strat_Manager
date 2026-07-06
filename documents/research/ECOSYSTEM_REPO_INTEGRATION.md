# Ecosystem Repo Integration: How the Independent Repos Plug Back into the Core

**Status: architecture decision note, forward work (post-TFG). Design only, no code.**

F1 StratLab is becoming an ecosystem: the core repo (this one, the TFG system) plus
dedicated public repos (`radiogate`, `gridmind`, `box-bot`, `pitlab`) plus datasets and
models on the Hugging Face org `f1stratlab`. Each new repo needs exactly one integration
mechanism with the core, decided up front, or the ecosystem degrades into ad-hoc copies
and circular dependencies. This note fixes those decisions.

The core already runs this play once: `src/telemetry/` is the independent repo
`F1_Telemetry_Manager` mounted as a git submodule (`.gitmodules`, gitlink SHA tracked by
the parent, initialized with `git submodule update --init --recursive`). That worked
example is the template for anything that earns submodule status; everything else uses a
lighter mechanism.

---

## 1. The three mechanisms and the rule that picks one

Every ecosystem repo integrates through exactly ONE of:

1. **Git submodule** (the `src/telemetry` pattern). For code that is coupled to the
   core's runtime: the core imports it, calls it in-process, and ships it as part of a
   core surface. The parent pins a gitlink SHA; bumps are explicit commits.
2. **HF artifact / pip package**. For produced artifacts (a corpus, a LoRA, a dataset,
   feature manifests) or genuinely shared library code. The core pulls it as a versioned
   dependency: `snapshot_download(..., revision=<hash or tag>)` for HF, a pinned version
   in `pyproject.toml` for pip. No repo mounting, no gitlink.
3. **Downstream service / API**. For consumers of the core. They subscribe to the core's
   published interfaces (the SSE/WS lap stream, the FastAPI backend) and pin the contract
   version on THEIR side. The core does not know they exist.

The project's own dependency rule (from the future-vision decision, 2026-06-12):
submodule ONLY if the code is coupled to the core runtime; independent repo if it is a
service or consumer (a bot); a standalone artifact (corpus, LoRA, dataset) ships via
HF or a package, never a submodule; shared code goes to a package or HF, not a submodule.

### The hard invariant: dependency direction

**The core must NEVER depend on a downstream consumer.** Arrows only point one way:

```
HF artifacts (corpus, LoRA, datasets)      radiogate code (submodule)
              |                                     |
              v                                     v
        +----------------------- CORE -----------------------+
        |  agents, orchestrator, simulation, lap_state, CLI  |
        |  telemetry submodule (UI), pitlab (in-core Studio) |
        +----------------------------------------------------+
              |                        |
              v (SSE/WS stream)        v (FastAPI)
           box-bot                other consumers
```

Concretely: `box-bot` is a service that consumes the core's stream. If it ever became a
submodule of the core, the dependency would invert (the core repo would track and ship a
consumer of itself, and a bot outage or refactor would block core releases). That is
forbidden by construction, not by convention: the check in section 6's checklist rejects
any candidate submodule that the core does not import at runtime.

---

## 2. Decision table

| Repo / component | Mechanism | Why | Pinning | CI note |
|---|---|---|---|---|
| `F1_Telemetry_Manager` (`src/telemetry/`) | **Git submodule** (existing) | UI surface launched by core entry points (`f1-streamlit`); core-coupled runtime code | Gitlink SHA, bumped by explicit parent commit | `test` job checks out with `submodules: true`; lint/typecheck skip it |
| `radiogate` | **Independent repo + git submodule mount** for the code, **HF pinned revision** for the corpus (decision 2026-07-05) | The picaresca/trust-signal inference code is consumed in-process by the Radio Agent (and later the Rival Agent), which is runtime coupling; the corpus itself is a standalone artifact and stays on HF | Gitlink SHA for code; `revision=` hash/tag for `f1stratlab/f1-team-radio-corpus` (v1.0 / v1.x / v2.0 releases) | Mount at `src/radiogate/`; keep data OUT of the submodule so checkout stays cheap; core imports must be lazy so uninitialized clones still run |
| `gridmind` | **HF artifacts only** (`f1stratlab/f1-domain-corpus` dataset + `f1stratlab/strat-gemma-lora` model) | The core never imports gridmind's training code (Unsloth fine-tune pipeline); it only consumes the produced LoRA, served through LM Studio (or an OpenAI-compatible endpoint) via the existing provider-agnostic LLM layer | HF `revision=` hash/tag for both dataset and model; LM Studio model name in config | Zero CI impact on the core; gridmind has its own repo and CI |
| `box-bot` | **Downstream service** | Pure consumer: reads the core's SSE/WS lap stream + uses the gridmind LoRA; the core must never depend on it | box-bot pins the core's stream contract version on ITS side (API version + release tag); core versions the `lap_state`/SSE schema | Zero CI impact on the core; contract regression tests live in box-bot |
| `pitlab` | **In-core** (no new integration mechanism) | Fase 0 already puts training in `src/strategy/training/` (in-repo by decision); the Studio UI reuses the frontend stack; the tracker (ClearML or MLflow) is a pip dependency, not custom code | pip-pinned tracker in `pyproject.toml` / `uv.lock` | Normal core CI; if the UI later grows its own repo, it follows the `src/telemetry` submodule pattern exactly |
| Real-time OpenF1 WS consumer | **In-core** | It PRODUCES the same `lap_state` contract the agents already consume; it is the core runtime by definition | n/a (core code) | Normal core CI |

Two summary observations:

- Only ONE new submodule enters the core (`radiogate`). Everything else is either an HF
  pinned artifact, a downstream service, or plain in-core code. Submodules are the
  expensive mechanism (contributor friction, CI wiring, pointer-bump ceremony); the
  default answer for a new repo is "not a submodule".
- radiogate is the deliberate hybrid: the REPO is independent (own CI, own releases, own
  visibility, own README declaring F1 StratLab ecosystem membership per the branding
  rule), the CODE is mounted as a submodule because the core runtime calls it, and the
  DATA ships via HF pinned revisions. Three concerns, three channels. If the core ends
  up importing nothing from the radiogate submodule at runtime (e.g. the trust signal is
  instead published as model weights + a thin in-core scorer), the submodule loses its
  justification and should be dropped to HF-only; that is review gate Q1 below.

---

## 3. Submodule mechanics (the worked `src/telemetry` pattern)

How an independent repo is actually wired as a submodule, exactly as done today:

**Wiring.** One `.gitmodules` entry per submodule:

```
[submodule "src/radiogate"]
    path = src/radiogate
    url = https://github.com/VforVitorio/radiogate.git
```

The parent repo does not store the submodule's files; it stores a **gitlink**, a single
tree entry recording the exact commit SHA of the submodule that the parent expects. A
fresh clone sees an empty `src/radiogate/` directory until
`git submodule update --init --recursive` populates it (CONTRIBUTING.md already documents
this for `src/telemetry`).

**The two-step commit workflow** (already the standing rule in CONTRIBUTING.md's PR
checklist for `src/telemetry/*`, applied identically to radiogate):

1. Commit and push INSIDE the submodule repo first (its own branch flow, its own PRs,
   its own CI).
2. In the parent, `git add src/radiogate` stages the new gitlink SHA; commit that pointer
   bump as its own change (`chore(radiogate): bump submodule to <short-sha>`), through
   the normal feature-branch -> PR -> `dev` flow.

Never the reverse order: a parent pointer that references an unpushed submodule commit
breaks every other clone.

**Boundary rules for the mounted code:**

- The submodule carries CODE only. Corpus data, model checkpoints, and audio stay on HF
  (the submodule's own `.gitignore` enforces it). This is what keeps checkout cheap and
  avoids the large-submodule problem entirely.
- The core imports from the submodule lazily and behind a guard. The DevEx audit already
  found that a clone without `--recurse-submodules` breaks the Docker quickstart because
  `src/telemetry/` is empty; a second submodule must not widen that failure mode. Rule:
  `f1-sim`, `f1-strat`, and the test suite must run on a clone with ZERO submodules
  initialized; radiogate-dependent features degrade with a clear "run
  `git submodule update --init src/radiogate`" message. A preflight/doctor check (the
  DevEx audit's proposal) lists uninitialized submodules.
- The submodule never imports the parent. Shared contracts (e.g. the `lap_state` fields
  the trust signal needs) are passed in as plain data, or extracted to an HF-published
  schema, so radiogate stays runnable standalone.

---

## 4. Versioning and pinning strategy

The unifying principle: **every cross-repo dependency is pinned to an immutable
reference, and bumping it is an explicit, reviewable commit in the consumer.** The
mechanism differs per channel; the discipline is the same.

| Channel | Pin | Bump ritual |
|---|---|---|
| Submodule | Gitlink SHA (automatic, inherent to git) | Two-step workflow above; pointer bump is its own PR-visible commit |
| HF dataset / model | `revision=<commit hash or tag>` on `snapshot_download` / `hf_hub_download` | Edit the pinned revision in one place, commit, PR |
| pip package | Version in `pyproject.toml` + `uv.lock` | `uv lock --upgrade-package <name>`, commit lockfile |
| Service contract (box-bot side) | Core release tag + stream schema version | box-bot updates its pin when it opts into a new core release |

**The P5 gap this must close first:** the data-engineering audit (epic #242) found the
core downloads from HF with a **mutable `main` revision**, so today's "pin" is a moving
target; the same audit found the `f1stratlab` org migration decided but unexecuted. Two
consequences for this design:

1. Before any NEW HF artifact (radiogate corpus, gridmind LoRA) is consumed by the core,
   `data_cache.py`'s download path gains a `revision=` parameter and the core pins actual
   hashes/tags. Otherwise the ecosystem inherits the same reproducibility hole times N
   artifacts.
2. Pinned revisions live in ONE manifest (a small constants module or JSON next to
   `data_cache.py`: `repo_id -> revision` for every consumed HF artifact), so "what
   exact data/model does this commit of the core run against" has a single answer, the
   way `.gitmodules` + gitlink answers it for submodules and `uv.lock` answers it for
   pip. The radiogate design already assumes this shape on the producer side (corpus
   releases v1.0 / v1.x / v2.0, each a pinned HF revision; downstream training always
   references a revision hash).

Version semantics per producer: radiogate tags corpus releases on HF (dataset card
documents the diff per version); gridmind tags LoRA releases (model card records base
model, corpus revision trained on, eval numbers); the core's own releases stay with
release-please on `main`. A core release therefore freezes: its own tag + two gitlink
SHAs + the HF revision manifest + `uv.lock`. That tuple is the full reproducibility
statement.

---

## 5. CI implications

Current state (`.github/workflows/ci.yml`): three jobs (`test`, `lint`, `typecheck`);
only `test` checks out submodules (`actions/checkout@v7` with `submodules: true`);
`uv sync --frozen` everywhere; no `gitsubmodule` ecosystem in `dependabot.yml`, so
submodule bumps are manual today.

Adding radiogate as a second submodule implies:

- **Checkout**: `submodules: true` on the `test` job picks up radiogate automatically
  (it initializes all `.gitmodules` entries). `lint` and `typecheck` continue to skip
  submodules; each submodule repo lints itself in its own CI.
- **Size discipline**: both submodules are code-only, so plain checkout is fine. The
  shallow-clone lesson (`--depth 1 --filter=blob:none` + cache keyed to the gitlink SHA,
  from the LexFlow bootstrap playbook) is the documented escape hatch IF a submodule
  ever grows heavy; the real answer is to never let data into a submodule (section 3).
- **Contributors without the submodule must stay green**: the suite must pass on a clone
  where `src/radiogate/` is empty. Tests that exercise radiogate integration skip with a
  clear reason when the path is uninitialized (same posture the suite already needs for
  the data-gated tests on CI runners). The collected-count floor in `ci.yml` stays valid
  because skipped tests still collect.
- **Submodule bump automation**: add a `gitsubmodule` entry to `dependabot.yml` only
  AFTER radiogate's runtime API stabilizes. While the trust-signal interface churns,
  bumps stay manual and deliberate (a wrong auto-bump changes agent behavior silently).
  `F1_Telemetry_Manager` can adopt the same automation at the same time or stay manual.
- **Cross-repo contract checks**: the core's CI does NOT test box-bot or gridmind.
  Instead, box-bot's CI runs a contract test against a pinned core release (schema of
  the SSE/newline-JSON stream), and gridmind's CI validates its LoRA against the pinned
  corpus revision. Failures surface in the repo that owns the dependency, which is the
  repo that can fix its pin.

---

## 6. Checklist: adding a new ecosystem repo

1. **Classify it.** Does the core import its code at runtime? If NO, it is not a
   submodule, full stop. Is it a consumer of the core (bot, dashboard, notifier)? Then
   it is a downstream service and the core must gain zero references to it. Does it
   produce an artifact (dataset, model, corpus)? Then the artifact ships on the
   `f1stratlab` HF org and the core pins a revision.
2. **Apply the invariant.** Draw the dependency arrow. If any arrow points from the core
   to a consumer, the design is wrong; re-cut the boundary until arrows only point from
   consumers to the core and from the core to artifacts/submodules.
3. **Name and brand it.** Ecosystem names do not carry "f1stratlab" (`gridmind`,
   `box-bot`, `radiogate`, `pitlab`), so the README and any HF dataset/model card MUST
   state explicitly that the repo is part of the F1 StratLab ecosystem.
4. **If submodule**: mount under `src/<name>/`, add the `.gitmodules` entry, keep data
   out of it, make core imports lazy and guarded, document the two-step commit workflow
   in its README, confirm the core test suite passes with the submodule uninitialized,
   and add it to CONTRIBUTING.md's PR checklist ("commit inside the submodule first,
   then bump the pointer").
5. **If HF artifact**: create the repo under the `f1stratlab` org, tag releases, add the
   `repo_id -> revision` entry to the core's pin manifest, and record the consumed
   revision in the artifact card's provenance section.
6. **If downstream service**: give it its own repo and CI, point it at a released core
   version, and add a contract test on its side. Do not touch the core repo at all.
7. **Bootstrap the repo itself** per the standard baseline (branch protection, CI,
   Dependabot, release automation) so its releases are trustworthy enough to pin.
8. **No AI attribution anywhere** (commits, PRs, issues, releases), and any LLM
   integration uses OpenAI or LM Studio, never Anthropic. These rules apply to every
   ecosystem repo, current and future.

---

## 7. Open questions for Victor

1. **radiogate submodule review gate.** The submodule mount is justified by the core
   importing the trust-signal inference in-process. If, when the Radio Agent hook is
   actually designed, the consumption shape turns out to be "HF model weights + a thin
   in-core scorer" with no code import, do we drop the submodule and go HF-only? Propose
   deciding at radiogate R3 (picaresca layer), not before.
2. **Where does the trust-signal interface live?** Inside the radiogate submodule (core
   imports `radiogate.inference`), or as a minimal pip package published from the
   radiogate repo (core pins it in `pyproject.toml`, no submodule needed)? The package
   route weakens the case for the submodule entirely; worth an explicit call.
3. **Pin manifest shape.** Single JSON next to `data_cache.py` versus constants in the
   module: does Victor want the manifest human-editable (JSON, diff-friendly for PRs) or
   code (typed, import-checked)? Recommendation: JSON, validated at load.
4. **Dependabot for submodules.** Enable the `gitsubmodule` ecosystem for
   `F1_Telemetry_Manager` now (its API is stable) and for radiogate later, or keep all
   submodule bumps manual? Recommendation: telemetry now, radiogate after R3.
5. **box-bot contract formalization.** Is a versioned schema doc for the SSE/newline-JSON
   stream (published in the core's docs site) enough, or do we want a machine-readable
   schema (JSON Schema / Pydantic export) that box-bot's contract tests consume?
   Recommendation: export the existing Pydantic models; it is nearly free.
6. **pitlab future split.** If the Studio UI outgrows in-core (own release cadence, own
   contributors), the plan says "follow the telemetry submodule pattern". Confirm that
   the trigger is organizational (separate contributors/cadence), not size, so we do not
   split prematurely.
