# pitlab: the F1 StratLab MLOps Studio (UI, tracker decision, orchestration)

**Status: research design, future work (post-TFG). Plan only, no code, no commitments.**

This document designs the `pitlab` initiative of the F1 StratLab ecosystem (initiative 4
of 5, FUTURE.md sections 6, 10 and 11, not versioned): a local dashboard for data
engineering and model retraining without opening a notebook. It deliberately
covers only the gap that no other document owns: the Studio UI/UX, the experiment-tracker
decision, and the orchestration layer. The pipeline the Studio operates is already fully
designed in two completed audits, and this document cross-references them instead of
restating them.

Brand rule (FUTURE.md 11.1): `pitlab` does not carry "f1stratlab" in its name, so its
GitHub description, README and any HF artifacts MUST declare that it is part of the
F1 StratLab ecosystem.

---

## 1. Scope: what this document owns, and what it explicitly does not

| Topic | Owner |
|---|---|
| Stages 1-2 of the pipeline (ingestion, merge/join): race identity module, schema manifests and `f1-data verify`, the data-manager facade (`status()`, `ensure(profile)`, `ensure_race`, `ensure_radio_corpus`, `build_race_dataset`), bronze/silver/gold layering, HF Hub governance and org migration | **AUDIT_P5_DATA_ENGINEERING.md** (epic #242), especially its Phase 5, which freezes the Studio stage 1-2 contract |
| Stages 3-7 of the pipeline (feature engineering, EDA, encoding/split, NLP labeling, retrain/registry): the per-model training extraction into `src/strategy/training/` (Fase 0), the export contract, manifests v2 with `era` field, calibration and drift monitors, retrain triggers, era-refresh checklist, adaptation techniques (RevIN, year-embedding, ShifTS) | **AUDIT_2026_REG_CONCEPT_DRIFT.md** (epic #189), especially its section 6 (the `pitlab` contract) and Phases 0-4 |
| Download progress UX, full-calendar picker, cache unification | **AUDIT_P2_LOADING.md** (epic #170) |
| NLP auto-labeling itself (weak supervision, LLM-judge, corpus) | **RADIOGATE_DECEPTION_AND_AUTOLABELING.md** (the Studio only gets a QA viewport over its outputs) |
| **The experiment-tracker decision, the Studio screens and UX, the job orchestration layer, the local-first artifact flow, the pitlab roadmap** | **THIS document** |

The rule this document enforces on itself: whenever a screen "does" something, the doing
is an invocable function owned by #242 (stages 1-2) or #189 (stages 3-7). pitlab is a
thin, honest wrapper. If designing a screen ever requires inventing new pipeline logic,
that logic must be filed against the owning audit, not specified here.

### What pitlab is NOT

- Not a notebook editor or replacement viewer. `notebooks/**` stay frozen (untouchable);
  the production path is the extracted `src/strategy/training/` code.
- Not a tracker. It wraps one (section 3); it never re-implements run logging, metric
  storage, or a model registry from scratch.
- Not a race-day surface. The CLI, Arcade and the telemetry SPA are the race surfaces;
  pitlab is the maintenance workshop between races.
- Not a cloud product. It runs on a single local machine, offline by default (section 7).

---

## 2. Prerequisites (all designed elsewhere, all gating)

1. **Fase 0** (#189 F-01, FUTURE.md 6.3): `src/strategy/training/` populated with pure
   functions per model family (`build_dataset`, `train`, `evaluate`, `calibrate`,
   `export`) plus the headless `f1-train` CLI. "Sin funciones invocables no hay botones."
   pitlab writes zero code before `f1-train` reproduces the seven production artifacts.
2. **#189 Phase 1** (measurement layer): the eval harness, calibration verification and
   drift monitors. The Studio's health dashboard is a view over these, not a source.
3. **#242 Phase 2** (data-manager facade) and **Phase 5** (the frozen stage 1-2
   contract): the functions the Data and Merge screens wrap.
4. **Frontend migration sprint S1** (design system): pitlab reuses the token-mapped
   Tailwind theme and component patterns from the telemetry SPA migration
   (React 19 + Vite + TypeScript strict + TanStack Router/Query + Zustand + ECharts).
5. **#189 F-08** (era-versioned, relative-path model registry): the Registry screen
   reads and writes this artifact; the current absolute-Windows-path registry blocks
   any automation.

---

## 3. Experiment tracker decision: ClearML vs MLflow+DVC vs W&B

FUTURE.md 6.2 and #189 open question 5 left this open with three candidates. The Studio
wraps exactly one tracker; the only value pitlab builds on top is F1-specific
orchestration and panels.

### 3.1 Selection criteria (in priority order for this project)

| # | Criterion | Why it matters here |
|---|---|---|
| C1 | **Local-first / offline by default** | The project's philosophy is no mandatory external services; the whole system already runs from one machine plus HF Hub. A tracker that phones home or needs an account fails the posture |
| C2 | **Self-host footprint on Windows 11** | The deployment target is a single Windows 11 machine (uv-managed Python, CUDA-pinned torch). A docker-compose stack of databases for a single user is disproportionate ops burden |
| C3 | **Model registry with staged promotion** | #189 Phase 4 requires acceptance gates, champion/challenger comparison, promote and rollback; per-GP lineage (base model, fine-tune data, calibration provenance) |
| C4 | **Data versioning story** | Era-structured datasets and reproducible retrains need versioned data. Note: the project ALREADY has a data-versioning layer decided, HF Hub dataset revisions with per-release pinning (#242 Phase 3, open question 4). The tracker does not need to duplicate it |
| C5 | **Cost** | Zero recurring cost target; this is a personal project, not a funded team |
| C6 | **Integration effort with the actual stack** | XGBoost, LightGBM, PyTorch Lightning (the TCN), plain sklearn calibrators; a Python API clean enough that `f1-train` and the #189 5.2 monitors log automatically; a queryable REST/Python API the FastAPI service can proxy to the SPA |
| C7 | **Longevity / bus factor** | The tracker outlives any single season; prefer boring, huge-community tools |

### 3.2 Candidate assessment

**Weights & Biases.** Best-in-class UX and the least integration code, but it is
SaaS-first: the free tier lives in their cloud, full self-hosting is an enterprise
product, and offline mode is a buffering mechanism, not a posture. Fails C1 and C5 by
philosophy even where it technically works. Eliminated.

**ClearML.** The FUTURE.md 6.2 lean ("el mas cercano a MLflow nuestro"), and the
strongest all-in-one: tracking, artifacts, data management, a remote-execution agent and
queues. The costs: the self-hosted server is a docker-compose stack (MongoDB,
Elasticsearch, Redis, fileserver) that must be running for anything to log, which on a
single Windows machine means Docker Desktop, WSL2 and Elasticsearch memory tuning as a
permanent background tax (weak on C2). Its extra powers are exactly the ones this
project does not need: multi-machine queues (there is one GPU), ClearML Data (HF Hub
already owns data versioning, C4), and agent-based remote execution (the job runner in
section 6 is a subprocess on the same machine). Strong candidate, wrong size.

**MLflow + DVC.** MLflow runs fully local with zero services: a SQLite-backed store and
a filesystem artifact root under the existing data directory, `pip install mlflow` and
nothing else (best on C1, C2, C5, C7). Autologging covers XGBoost, LightGBM and PyTorch
Lightning natively (C6). The model registry with aliases (champion/challenger style)
works on the SQLite backend and gives exactly the promote/rollback primitive C3 needs.
Its two real weaknesses are orchestration (none, but section 6 builds the thin runner
anyway, and it would be built on top of ClearML too) and data versioning (none). Which
brings in DVC... and DVC is where this option should be trimmed: DVC would introduce a
second content-addressed cache and remote system in parallel with HF Hub, which #242
already established as the dataset store with revision pinning per release. Two data
versioning systems is one too many.

### 3.3 Recommendation

**Adopt MLflow, local-only (SQLite store + filesystem artifacts), WITHOUT DVC. HF Hub
remains the single data-versioning and distribution layer.** Concretely:

- `f1-train` and the #189 5.2 monitors log runs, params, metrics and artifacts to a
  local MLflow store that lives under the user data root (gitignored, never in the repo).
- Lineage instead of DVC: every run records the HF dataset revision it consumed, the
  manifest hashes (manifests v2, #189), the era/regulation-cycle tag, and the git commit
  of the training code as run parameters. Reproducing a run = pin those four.
- The MLflow model registry (aliases for champion/challenger per model family) is the
  tracker-side registry; promotion additionally writes the era-versioned
  `model_registry.json` that the agents actually consume (#189 F-08), keeping the agent
  runtime tracker-agnostic. The agents never import MLflow.
- The stock MLflow UI stays available as a free escape hatch during early phases, which
  lets pitlab build only the panels that add F1-specific value instead of re-cloning a
  generic runs table.

**Fallback trigger:** if the ecosystem ever needs multi-machine training
(cloud GPU bursts for gridmind-scale jobs) or team-grade queues, revisit ClearML then.
The `f1-train` boundary makes the tracker swappable: screens talk to the pitlab FastAPI
service, never to MLflow directly, so a tracker swap is a service-layer change.

This is a reasoned departure from FUTURE.md 6.2's ClearML lean, pending ratification
(open question 1).

---

## 4. Product shape: a separate local app on the shared stack

**Recommendation: pitlab is a SEPARATE local web app, not a section of the telemetry
SPA.** Same stack, different process and lifecycle:

- **Stack reuse (per the frontend migration plan):** React 19 + Vite + TypeScript
  strict, Tailwind mapped 1:1 to the existing `tokens.css` design system, TanStack
  Router + Query, Zustand for UI state, Apache ECharts as the single chart library.
  Calm Linear/Vercel register throughout: pitlab is all "working screens", so none of
  the expressive GSAP/Three.js budget applies. Charts and tables stay crisp, blur only
  on chrome, dark-first.
- **Why separate:** (a) different cadence and audience: the telemetry SPA is the
  race-analysis product, pitlab is the maintenance workshop, and coupling their release
  cycles couples a demo surface to an ops surface; (b) long-running GPU training jobs
  should not share a backend process with the race-analysis API; (c) the telemetry UI
  lives in the `F1_Telemetry_Manager` submodule while the training code (Fase 0) is
  core in-repo, and threading pitlab through the submodule would put its backend on the
  wrong side of that boundary.
- **Backend:** a small dedicated FastAPI service (backend stays FastAPI, per the hard
  constraint) that imports `src/strategy/training/` and the #242 data-manager facade
  directly, embeds the job runner (section 6), and proxies tracker queries. Own port,
  reverse-proxied to its SPA the same way the migration serves the telemetry app, so
  no CORS and no compose churn.
- **Where the code lives (proposal):** start in the core repo (a `src/pitlab/` service
  plus a pitlab frontend directory), because v1 is import-coupled to Fase 0 and the
  facade. Extraction into the dedicated `pitlab` repository (independent repo consumed
  as a submodule, matching the radiogate ruling) is deferred to the pending
  ecosystem-repo-integration note; FUTURE.md section 8 leaves Studio topology open.
  Open question 2.

---

## 5. The Studio screens

The seven-stage pipeline (FUTURE.md 6.4) maps to nine screens in three navigation
groups. Every screen lists what it shows, what actions it exposes, and what it wraps.
Feature lists, metrics and encodings are ALWAYS re-read from the manifests at build
time (`feature_manifest_*.json`, `tiredeg_sequence_config.json`, `encoding_maps.json`,
`tire_compounds_by_race.json`, the kmeans artifacts); nothing is hardcoded in the UI.

```
 DATA                MODELING                        OPERATIONS
 1 Data (ingest)     4 Dataset Builder (st. 3+5)     8 Evaluate
 2 Merge (st. 2)     5 Explore (st. 4)               9 Registry & Publish
 3 Radio QA (st. 6)  6 Train (st. 7a)                0 Home: Model Health
                     7 Runs (st. 7b)
```

**0. Home: Model Health (the drift dashboard).** The answer to #189's framing question
"is the system healthy for the current regulation" as a single screen. Shows: per-model
health cards against the 2025 baseline bands (pace MAE, tire-deg MAE, pit P50 MAE and
quantile coverage, the three classifier AUC-PRs), calibration status (reliability,
Brier, MC-sigma ratios), feature-drift PSI watch-list, cluster-stability flag, era
banner (which regulation cycle every deployed artifact belongs to), and the #189 5.3
trigger table with any tripped trigger highlighted and its prescribed action. Actions:
drill into a monitor's run history; jump to Train pre-filled when a trigger prescribes
a retrain. Wraps: the #189 Phase 1 monitors, read from the tracker's monitoring
experiment. pitlab renders this; it computes none of it.

**1. Data (stage 1: ingest and status).** Shows: what is on disk versus available, with
sizes and validation status per race, keyed by canonical race identity (never raw folder
strings). Actions: buttons per source exactly as FUTURE.md 6.4 specifies (year, circuit,
session type, intervals, radio, FIA PDFs, HF pull), profile-based ensure (sim, arcade,
backend, full), with real progress from the facade's stage callbacks. Wraps: the #242
Phase 2 `data_manager` facade and Phase 4 builders; session-type-aware ingestion
(FP/Q/Sprint) arrives when #189 Phase 2 lands and this screen inherits it as new
buttons, not new logic.

**2. Merge (stage 2: join preview).** Shows: a join builder over the ingested artifact
families (laps, intervals, radio, compounds), previewing output shape, per-key match
rates and percent unmatched, plus the schema-validation verdicts from `f1-data verify`.
Actions: compose a join on canonical keys, inspect mismatches, materialize the merged
frame for the Dataset Builder. Wraps: #242's identity module and schema manifests
(joins can only be trusted because identity and validation are solved there).

**3. Radio QA (stage 6: NLP labeling viewport).** Shows: transcription, sentiment and
intent outputs per clip with confidence, filterable by GP and driver. Actions: inspect,
flag suspicious items, export a review list. Explicitly a read-and-flag QA surface:
labeling automation is radiogate's design, and when its auto-labeler ships this screen
fronts its outputs. No labeling logic lives in pitlab.

**4. Dataset Builder (stages 3 and 5: features, encoding, targets, split).** The
manifest-driven workbench, containing the key panels FUTURE.md 6.4 names: the
compound-mapping editor (the 2026 relabeling surface, editing era blocks of
`tire_compounds_by_race.json` through its #242 schema), the K=4 cluster viewer with
"train with this cluster" pooling selection, the target definition form with the
outlier-threshold slider, the temporal-split viewer (train/val/test by season, leakage
boundaries visible), and the TCN sequence-config editor (windows per compound). Actions
produce a versioned dataset-build config and invoke the Fase 0 `build_dataset`
functions; edits to manifests are drafted as manifests-v2 changes with the era field,
never silent in-place mutation. Wraps: `src/strategy/training/` dataset builders plus
the #189 F-05/F-06 era-aware artifacts.

**5. Explore (stage 4: EDA, correlation, leakage).** Shows: distributions, degradation
curves, scatter and box plots, correlation heatmap with the absolute-Pearson threshold
slider, permutation/model feature importance, and the leakage audit view. Actions: the
"descartar variables facil" loop, toggle a feature off and the toggle writes a draft
manifest exclusion (reviewed, versioned), not a hidden UI state. Wraps: the EDA logic
extracted with Fase 0 (the N03-N16 analyses become invocable plot-data providers;
ECharts renders them).

**6. Train (stage 7a: launch).** The heart of the no-notebook promise. Shows: model
family picker (the seven production models), era/season pool selector, cluster-pooling
option, config summary sourced from manifests, estimated data volumes (from the #189
4.2 label-availability playbooks, so the UI itself refuses configurations the playbook
marks invalid, e.g. training the overtake classifier on FP-only data). Actions: launch
a job (which runs `f1-train` headless underneath), or launch the cheaper recalibrate
variant (Platt refit, quantile offsets, MC-sigma rescale) that #189 4.3 prescribes as
the first lever. Also hosts per-GP incremental fine-tune presets (base model + weekend
sessions) once #189 Phase 4 defines them.

**7. Runs (stage 7b: monitor).** Shows: the job queue and history (section 6 state
machine), live log tail streamed over SSE, live metric curves for the running job, and
a link-out to the underlying MLflow run. Actions: cancel a job, retry with the same
config, pin a run for comparison.

**8. Evaluate (stage 7c: judge).** Shows: the #189 Phase 1 eval-harness output for a
candidate artifact against the current champion and the 2025 baseline bands, the full
calibration report (reliability diagrams, Brier, coverage, MC-sigma ratio), and the
acceptance-gate verdict (pass/fail per gate, with the failing gate named). Actions:
approve or reject a candidate. Approval is explicit and manual: per #189 4.5, nothing
auto-promotes.

**9. Registry & Publish.** Shows: the era-versioned registry per model family (which
artifact each agent surface currently consumes, its lineage: base, train window,
metrics, calibration provenance, manifest hash), the export-contract completeness check
(the exact artifact set the agents read: model file, manifest, encoders, thresholds,
calibration JSONs, all present and era-consistent), and HF publication state. Actions:
promote an approved candidate (tracker alias flip plus registry write), roll back to
the previous champion (one click, same mechanism reversed), and publish to the HF org
`f1stratlab` with a pinned revision (section 7). Wraps: the Fase 0 `export` contract
and #189 F-08 registry.

### The core flow: retrain a model without opening a notebook

1. Home shows a tripped trigger (say, pit quantile coverage below 0.80 for 5 GPs).
2. Click through to Train, pre-filled with the model and the trigger's prescribed
   action (refit quantile offsets, or full retrain if volume allows).
3. Adjust the pool (seasons, cluster pooling) inside what the label-availability
   playbook permits; launch.
4. Watch it in Runs (live logs, live metrics); walk away, it is a queued background job.
5. When it finishes, Evaluate shows candidate vs champion vs baseline plus the gate
   verdict. Approve.
6. Registry: promote (registry JSON + tracker alias), optionally publish to HF with a
   pinned revision.
7. The surfaces (CLI, Arcade, SPA) pick up the new artifact through the normal
   `ensure_setup` pull at the pinned revision. Home goes green on the next monitor run.

Every step is a wrapped call into #189/#242-owned functions; the Studio contributes the
click path, the job runner and the guardrails (gates, era checks, no silent promotion).

---

## 6. Orchestration: jobs, logs, artifacts

The layer between "button" and `src/strategy/training/`. Kept deliberately thin.

**Execution model.** A training job is a subprocess invocation of the headless
`f1-train` CLI (or `f1-data` facade calls for ingestion jobs), never an in-process
import-and-call inside the web service. Rationale: crash isolation (a CUDA OOM kills
the job process, not the Studio), environment determinism (the same uv environment and
entry point used for a manual run, so headless and Studio runs are bit-identical),
and trivially safe cancellation (terminate the process group).

**Job state.** A small SQLite-backed job table owned by the pitlab service: queued,
running, succeeded, failed, cancelled; with config snapshot, timestamps, exit code,
tracker run id, and artifact paths. Single-GPU reality is encoded as a concurrency-one
queue (a semaphore, not a scheduler): jobs queue FIFO, data-only jobs (ingest, verify)
may run alongside because they are CPU/network bound. No cron, no DAG engine: the
per-GP cadence is manual by design, and #189's triggers prescribe actions to a human,
they do not auto-execute.

**Logs and progress.** Jobs write structured line-oriented logs; the service tails them
and streams to the SPA over SSE, the same transport pattern the sim endpoint already
uses in the backend. Progress stages (download, build, train epochs, eval) come from
the facade's progress callbacks and the training code's epoch hooks, surfaced as a
stepper plus raw log toggle.

**Tracker linkage.** The training code logs to MLflow itself (autolog plus explicit
params: HF revision, manifest hashes, era, git commit). The runner only captures the
run id and stores it on the job row. The Studio reads metrics through the service,
which queries the local MLflow store; screens never talk to MLflow directly, keeping
the tracker swappable (section 3.3).

**Artifacts.** Training outputs land where the Fase 0 export contract puts them
(the models tree the agents read), and are additionally attached to the tracker run for
lineage. Promotion (Registry screen) is the only operation that mutates what agents
consume: it flips the tracker alias and rewrites the era-versioned registry JSON
atomically, and rollback is the same write with the previous pointer. The acceptance
gate itself (metric thresholds, calibration bounds, era consistency) is computed by the
#189 eval harness; the Studio enforces "no promotion without a passing verdict" and
records who clicked and when.

**Monitors as jobs.** The per-GP monitor run (#189 5.2) is just another job type:
triggered manually after a GP (or from the Data screen after ingesting one), logging
to the monitoring experiment that Home reads.

---

## 7. Local-first posture and the HF flow

- **One machine, no cloud.** Everything runs on a single Windows 11 machine with the
  uv-managed environment and the CUDA-pinned torch build. Mandatory external surface:
  none. Optional external surface: exactly one, Hugging Face Hub, and only when pulling
  data or publishing artifacts. The tracker store, job DB, logs and artifacts all live
  under the user data root (gitignored; nothing of this enters git).
- **Pull side.** Datasets and models arrive through the existing `data_cache` machinery
  against the `f1stratlab` org (post #242 Phase 3 migration), era-structured, at pinned
  revisions per release. pitlab's Data screen is a UI over that, not a second
  downloader.
- **Push side.** Publishing a promoted artifact set to HF is an explicit Registry
  action (proposed: manual, never automatic on promotion, open question 4): upload the
  export-contract artifact set, tag the revision, update the dataset/model card with
  the era coverage note (#189 F-15). Local promotion and HF publication are decoupled
  on purpose: the machine can run ahead of the Hub.
- **No LLM in pitlab.** The Studio needs no LLM to function. If a convenience LLM
  feature ever appears (run summaries, config explanation), it uses OpenAI or LM Studio
  per the provider rule, is optional, and degrades to nothing when absent.
- **Offline degradation.** With no network: ingest of new races fails loudly (expected),
  everything else (train, evaluate, promote, monitor, explore) works fully on local
  data. This is a design requirement, not an accident.

---

## 8. Phased roadmap (each phase is a self-contained future work item)

Ordering follows the automation surface #189 section 6 prescribes: CLI first, tracker
second, UI panels third, drift dashboard and incremental jobs last. UI phases are
sequenced by value density: the vertical slice that proves the no-notebook promise
comes before breadth.

| Phase | Title | Content | Gate |
|---|---|---|---|
| S0 | **Prerequisite gate** | No pitlab code. Verify Fase 0 (`f1-train` reproduces the 7 artifacts), #242 facade, #189 Phase 1 harness exist | The other epics |
| S1 | **Tracker adoption, headless** | MLflow local store wired into `f1-train` and the monitors (autolog + lineage params: HF revision, manifest hash, era, commit). No UI; stock MLflow UI is the interim viewer | S0 |
| S2 | **Job runner + the vertical slice** | The pitlab FastAPI service (job table, queue, SSE logs) + a minimal SPA with Train and Runs screens only. Acceptance: retrain the pace model end to end without a notebook | S1 |
| S3 | **Data and merge screens** | Screens 1-2 wrapping the #242 facade, verify results and identity-keyed status; monitor-run job type + a first Home health strip | S2, #242 Ph. 2 |
| S4 | **Dataset builder + explore** | Screens 4-5, fully manifest-driven; compound-map editor, cluster pooling viewer, split viewer, TCN sequence editor, importance/exclusion loop | S3 |
| S5 | **Evaluate, registry, publish** | Screens 8-9: gate verdicts, champion/challenger, promote/rollback, era-versioned registry writes, manual HF publish; full Home drift dashboard | S4, #189 Ph. 1 |
| S6 | **2026 era workbench** | The #189 Phase 3-4 execution surface: era-refresh checklist runner, mid-season cluster re-fit workflow, per-GP incremental fine-tune presets with gates and rollback | S5, #189 Ph. 2-3 |

S2 is the moment pitlab earns its existence; if the vertical slice does not feel
better than running `f1-train` by hand plus the MLflow UI, stop and reassess breadth
(open question 3).

---

## 9. Risks and limitations (candid)

- **Fase 0 is the real bottleneck, and it is not this project.** `src/strategy/training/`
  is empty today; every screen above is a wrapper over code that does not yet exist.
  pitlab slipping is free; starting it early is the only real failure mode.
- **UI surface creep.** Nine screens is a lot of frontend for one maintainer. The
  mitigation is structural: the MLflow UI escape hatch means pitlab only ever needs to
  build panels that are F1-specific (compound editor, cluster pooling, gate verdicts,
  era health), and S2's stop-and-reassess gate is explicit.
- **Wrapper drift.** If a screen grows pipeline logic (a join fixup here, a
  threshold default there), the #189/#242 ownership boundary erodes and two sources of
  truth appear. Rule: pipeline behavior changes are PRs against the owning layer, and
  the Studio version-pins the contracts it consumes (manifests v2 schema, facade
  signatures, registry schema).
- **Manifest-driven UI depends on manifests v2 landing** (#189). Until the era field
  and schemas exist, the Dataset Builder cannot safely edit anything; S4 must not start
  early against the v1 pseudo-schema JSONs.
- **Single-machine GPU contention.** A training job and an Arcade replay or a sim run
  compete for the same GPU. The queue serializes pitlab's own jobs but cannot see
  external processes; the pragmatic posture is documentation (train between sessions),
  not enforcement.
- **Tracker bet.** MLflow-local is low risk (boring, huge community), but if the
  gridmind LoRA work later wants cloud GPUs and queues, the ClearML fallback means a
  service-layer swap. Acceptable because screens never touch the tracker directly.
- **Windows specifics.** Process-group cancellation, long path names and file locking
  behave differently on Windows; the job runner design must be validated there first,
  since it is the only deployment target.

---

## 10. Open questions

1. **Ratify the tracker: MLflow local without DVC, HF Hub as the data-versioning
   layer.** This departs from FUTURE.md 6.2's ClearML lean for footprint reasons
   (section 3.3). Confirm, or name a reason to eat the ClearML server stack.
2. **Topology: separate local app confirmed, but where does the code start?**
   Proposal: in-core (`src/pitlab/` + its own SPA) for v1 because it import-couples to
   Fase 0, extraction to the dedicated `pitlab` repo (independent, consumed as a
   submodule like the radiogate ruling) deferred to the ecosystem-repo-integration
   note. Confirm or invert (repo-first from day 1).
3. **UI breadth for v1.** Which stages need panels versus staying
   CLI + stock MLflow UI? The S2 vertical slice (Train + Runs) is committed; everything
   after is negotiable per the stop-and-reassess gate.
4. **HF publish policy.** Proposed: promotion is local, publication to `f1stratlab` is
   a separate manual action. Alternative: auto-publish on promotion (simpler mental
   model, riskier Hub hygiene).
5. **Where does Model Health live long-term?** Proposed: pitlab owns it; optionally a
   read-only mirror card in the telemetry SPA later. Alternative: health belongs in the
   main app and pitlab links to it.
6. **Which ambition does the roadmap serve** (#189 open question 9): running the 2026
   season live, or retrospective replay after the fact? Live pulls S3-S6 forward and
   hardens the weekend cadence; retrospective relaxes everything and S6 can compress.

---

## 11. Internal references

- `documents/audits/AUDIT_2026_REG_CONCEPT_DRIFT.md` (epic #189): stages 3-7 owner,
  Fase 0 contract (its section 6), monitors and triggers (its section 5), phased plan.
- `documents/audits/AUDIT_P5_DATA_ENGINEERING.md` (epic #242): stages 1-2 owner,
  identity module, schema contracts, data-manager facade, HF governance (its Phase 3),
  the frozen Studio ingestion contract (its Phase 5).
- `documents/audits/AUDIT_P2_LOADING.md` (epic #170): download progress and cache UX.
- `documents/research/RADIOGATE_DECEPTION_AND_AUTOLABELING.md`: the NLP labeling
  automation the Radio QA screen will front.
- `FUTURE.md` (repo root, not versioned): sections 6 (Studio, the 7-stage pipeline and
  panel list), 8 (repo topology rule), 10 (Fase 3 placement), 11 (naming, brand rule,
  HF org), 12.2 (the tracker question this document answers).
- Frontend migration plan (memory `project_frontend_migration_plan`, tracking issue
  F1_Telemetry_Manager#25): the stack and design system pitlab reuses.
