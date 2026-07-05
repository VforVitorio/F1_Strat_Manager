# AUDIT P5 - Data engineering & new data features

**Auditor:** Fable 5 · **Date:** 2026-07-05 · **Repo:** `F1_Strat_Manager` (read-only pass, no code changed)
**Scope:** the data layer of the CORE repo: the extraction/ingestion pipeline (`src/data_extraction/`, `src/shared/data_extraction/`, the N01-N04 notebook pipeline as-built), the `data/` on-disk layout and its hygiene, data quality/validation, the `lap_state` contract as a data product, the download + preparation flow AND its interface/UX across the three surfaces (`src/f1_strat_manager/data_cache.py` first-run HF download, Arcade, backend/SPA), the HF Hub dataset layout and its migration to the `f1stratlab` org, and NEW additive data features (telemetry-derived signals, rival data for the future Rival Agent) with sourcing feasibility.
**Hard constraints honored in every remedy:** plan only, no code; backend stays FastAPI; LLM = OpenAI / LM Studio, never Anthropic; UNTOUCHABLE (duplicate before modifying / additive entry points only): `scripts/run_simulation_cli.py`, `src/agents/` internals, `notebooks/**`, `legacy/**`. Data/models are NOT in git; they come from HF Hub.

---

## 1. Framing: the boundary with the 2026-regulation audit (read first)

The ⭐ 2026-regulation / concept-drift sub-scope of P5 was **already executed** as `documents/audits/AUDIT_2026_REG_CONCEPT_DRIFT.md` (epic #189). This audit **cross-references it and does not duplicate it**. The division of ownership:

| Topic | Owner |
|---|---|
| 2026 break analysis, era-coupled artifacts (compound maps, encoders, traversal lookups, stint constants), manifests-v2 with `era` field | **#189** (F-05, F-06, Phase 3) |
| Multi-session ingestion for retraining (FP1-FP3 / Q / Sprint / testing), FP fuel-load estimator, label-availability playbooks | **#189** (F-02, F-11, Phase 2) |
| Drift detection, calibration monitors, RevIN / year-embedding / ShifTS, retrain triggers | **#189** (F-03, F-04, F-09, Phases 1 and 4) |
| Training extraction into `src/strategy/training/` (Fase 0), the per-model export contract, the tracker decision (ClearML vs MLflow+DVC vs W&B), `pitlab` stages 3-7 | **#189** (F-01, §6) - shared prerequisite, referenced here as "Fase 0" |
| Download progress bar fix, full-calendar picker, cache-dir unification, transcripts distribution | **P2 Loading audit** (#170; issues #168, F-12, F-10/F-11, F-02 there) |
| **Race/GP identity, data validation and contracts, raw-layer reproducibility, the download/preparation flow's interface per surface, HF Hub governance and org migration, non-2026 new data features, Rival Agent data readiness, `pitlab` stages 1-2 (ingestion + merge) contract** | **THIS audit** |

Where a remedy here touches a #189 artifact (e.g. metadata versioning aligning with manifests-v2), the finding says so explicitly. Nothing in this plan requires the 2026 program to have started; everything in it also pays off before 2026.

---

## 2. Executive summary

The data layer works for the defended demo, but it works by memory, not by contract. Five structural problems, all confirmed against current code and data on disk:

1. **There is no canonical race identity.** At least five naming schemes for the same GP coexist: raw folder names (`Miami_Gardens`, underscored, and inconsistent across seasons: 2023/2024 use `Miami`), `metadata.json` friendly names (`Miami Gardens`), compound-allocation keys (`Miami`, `data/tire_compounds_by_race.json`), radio corpus slugs (`united_states_miami`, `src/f1_strat_manager/gp_slugs.py`), and official event names inside model artifacts (`Miami Grand Prix`, `data/models/pit_prediction/model_config.json` traversal lookup). Reconciliation is scattered across at least five per-surface alias tables. Consequence measured on disk: for roughly six GPs per season the flagship CLI silently loses the radio corpus and the Pirelli Cx labels, and 2023 contains the same race twice (`data/raw/2023/Spain/` and `data/raw/2023/Barcelona/`, byte-identical sizes).
2. **Nothing validates data anywhere.** `RaceReplayEngine` reads parquets with zero schema checks; FastF1's own quality flags (`IsAccurate`, `Deleted`, `DeletedReason`), present in every laps parquet, are never consulted; `metadata.json` records row counts that nothing verifies. A truncated or semantically wrong parquet flows straight into agent feature vectors.
3. **The "raw" layer is not raw and not reproducible.** N01 bakes derived features and era-coupled encodings (`CompoundID`, `TeamID`, `FuelLoad`, `drs_window`) into `data/raw/**` at extraction time, and N01 is an untouchable notebook. The only production-grade, invocable extractor in `src/` is the OpenF1 radio builder; the rest of the extraction packages are archives that the project's own READMEs disown while `CLAUDE.md` §3 still presents them as the live pipeline.
4. **The download + preparation flow has one door and two and a half surfaces without it.** Only the CLI runs first-run setup (`scripts/f1_cli.py:93-96`); Arcade assumes parquets exist and only lazily pulls radio audio; the backend 404s with no remediation path and carries its own second data-root resolver with different precedence. Meanwhile `ensure_race` downloads ~76 percent dead weight per GP (intervals + pitstops parquets have zero runtime consumers), and `scripts/download_data.py` still offers an uncurated full 31.7 GB pull.
5. **The HF Hub side is a decided-but-unexecuted migration.** Single flat dataset repo pinned to a mutable `main` revision, models living inside a dataset repo, legacy flat paths still published, org `f1stratlab` decided (FUTURE.md §11) but not created.

The flip side is a real opportunity: the data the Rival Agent (the chosen TFM) needs first, per-lap rival gap evolution, is **already downloaded for every race** (`intervals.parquet`, ~27k rows per GP) and consumed by nothing. Making it a first-class runtime input is the cheapest new-feature win in the repo.

---

## 3. Assessment

### 3.1 The extraction layer as-built (who actually produces the data)

| Component | Status | Evidence |
|---|---|---|
| `notebooks/data_engineering/N01-N04` | **The real pipeline** (race download, EDA, clustering, feature engineering). Untouchable. Race-sessions only (`ff1.get_session(year, gp_name, 'R')`, N01) | N01 cell ~292; #189 F-02 |
| `src/data_extraction/openf1/radio_dataset_builder.py` | **Production-grade** (53.8 KB): class-based, retry session, idempotent resume, slug disambiguation, smoke test, companion upload script. **This is the template** every future ingestion module should copy | its README, `scripts/build_radio_dataset.py`, `scripts/upload_radio_corpus.py` |
| `src/data_extraction/fastf1/session_extractor.py` | Reference only, Spain-2023 scoped, superseded by N01 | `src/data_extraction/README.md` "fastf1/ - reference" |
| `src/data_extraction/openf1/intervals_extractor.py` | Reference only, session_key hardcoded to 9102 (Spain 2023) | `intervals_extractor.py:36-45` |
| `src/data_extraction/legacy/`, `src/shared/data_extraction/`, `src/vision/` | Archived; `src/shared/README.md` says "not imported by any active pipeline", kept only for old notebook imports | both READMEs |

The takeaway: there is exactly one invocable, repeatable ingestion path in `src/` (radios). Adding a race (a missed 2025 GP, or any future season) means re-running notebook cells by hand. `CLAUDE.md` §3 still describes `src/shared/data_extraction/` as "fastf1/openf1 extractors, augmentation", which misleads every new contributor and every audit brief (including this one's).

### 3.2 Race identity: the five naming schemes (the load-bearing mess)

Concrete divergences on disk today:

- Folders vs seasons: `data/raw/2023/Miami/`, `data/raw/2024/Miami/`, but `data/raw/2025/Miami_Gardens/`. Cross-season joins by folder name silently drop Miami.
- Duplicate race: `data/raw/2023/Spain/` AND `data/raw/2023/Barcelona/` (same GP, both fully populated, identical file sizes). Any "all races of 2023" iteration double-counts it.
- Folder vs compounds key: folders `Mexico_City`, `Las_Vegas`, `Marina_Bay`, `Yas_Island`, `São_Paulo`, `Miami_Gardens`; compound keys `Mexico City`, `Las Vegas`, `Marina Bay`, `Yas Island`, `São Paulo`, `Miami` (`data/tire_compounds_by_race.json`, 2025 block).
- Folder vs radio slug: `resolve_gp_slug` knows the space forms (`"Mexico City"`, `"Marina Bay"`, `src/f1_strat_manager/gp_slugs.py:34-63`) but NOT the underscore folder forms the CLI documents as its input (`run_simulation_cli.py:2314` "Grand Prix folder name").
- Model artifacts key by official event name: `circuit_traversal_lookup` uses `"Miami Grand Prix"`, `"Mexico City Grand Prix"` (`data/models/pit_prediction/model_config.json`).

Reconciliation code is duplicated in at least five places, each partial: `src/simulation/__main__.py:37-43` (`_GP_FOLDER_ALIASES` + `:119` underscore-to-space), `src/arcade/config.py:301` (`"Miami": "Miami_Gardens"`, the inverse), `src/arcade/data.py:328` and `src/arcade/strategy.py:511` (ad-hoc `replace`), `src/telemetry/backend/api/v1/endpoints/strategy.py:723,757` (its own slug pair).

**Measured silent failure** (flagship CLI, six GPs per season): `f1-sim Miami_Gardens ...` (the documented folder-name input) reaches `ensure_radio_corpus(2025, "Miami_Gardens")`; `resolve_gp_slug` raises `ValueError`, which `data_cache.py:467-473` swallows by design, returning a nonexistent audio dir; the sim runs with **zero radios and no error**. The same input misses the compounds lookup (`run_simulation_cli.py:344-352` `_TIRE_ALLOC[year].get(gp_name, {})` returns `{}`), so Cx labels silently vanish. Every remedy is caller-side or in `gp_slugs.py` (editable); the untouchable CLI is not modified.

### 3.3 Data quality and validation (there is none)

- Load path: `replay_engine.py:66-73` reads `laps.parquet` / `weather.parquet` with no schema, dtype, or row-count check; `_parse_meta` (`replay_engine.py:87-99`) falls back to directory name + year 2025 silently.
- `RaceStateManager` defends with `.get()` and `pd.notna` everywhere (`race_state_manager.py:188-217`), which converts missing columns into `None` values inside `lap_state` rather than a loud failure. Good for demo resilience, bad for detecting a broken artifact.
- FastF1 quality flags are shipped and ignored: `data/raw/2025/Budapest/laps.parquet` carries `IsAccurate`, `Deleted`, `DeletedReason`, `FastF1Generated`, `IsPersonalBest`, none consulted at runtime or exposed in `lap_state` (`race_state_manager.py:188-217` field list).
- `metadata.json` per race stores `record_counts` (laps 1368, intervals 27143 for Budapest) that nothing ever re-verifies, and no pipeline/library/schema version (only `extraction_date`), so a re-extraction with a newer FastF1 (which retro-corrects data) is indistinguishable from the original.
- Weather is joined by fractional index, not timestamps (`race_state_manager.py:307-309`), acceptable for replay, wrong for a live feed; noted as a contract caveat, not a bug.

### 3.4 The raw/processed boundary is broken by design

`metadata.json` declares it: `"calculated_features": {"laps": ["CompoundID", "TeamID", "LapsSincePitStop", "FuelLoad"], "intervals": ["drs_window", ...]}`. Encodings and model features are baked into `data/raw/**` at N01 extraction time. Consequences: (a) fixing a team encoding or the fuel proxy (both flagged by #189 F-05/F-06) requires regenerating "raw" data with an untouchable notebook; (b) "raw" cannot be re-derived from FastF1/OpenF1 by any invocable code; (c) `drs_window` (a 2026-dead feature, #189 F-05) is fossilized one layer lower than anyone will look for it. The remedy is NOT to rebuild history now; it is to define the layering (bronze = provider-verbatim, silver = enriched, gold = model-ready) for the FUTURE ingestion code (Phase 4 here, feeding #189 Phase 2 and `pitlab` stage 1-2), and declare the current tree "bronze-plus-patch, schema v1, frozen".

### 3.5 Download + preparation flow and UX (per surface)

Current door count: one. Only the CLI entry points run first-run setup (`scripts/f1_cli.py:93-96`, `run_simulation_cli.py:1590-1596`).

| Surface | Data acquisition today | Gap |
|---|---|---|
| CLI (`f1-strat`/`f1-sim`) | `is_first_run()` + `ensure_setup()` + lazy `ensure_radio_corpus` per GP | Progress UX broken (issue #168, owned by P2); picker only lists local races (P2 F-12); `ensure_race` exists (`data_cache.py:413`) but no caller uses it |
| Arcade | Only `ensure_radio_corpus` (`src/arcade/strategy.py:424-437`); assumes featured parquet + race data exist; cold FastF1 fetch happens implicitly with a frozen window (P2 A1/A3) | No first-run detection, no download affordance, no "this race is not on disk, fetch it?" gate in the menu |
| Backend / Streamlit / future SPA | Nothing. `backend/utils/laps_cache.py:37` raises "Featured parquet ... not available." with no remediation hint; the SPA migration plan inherits "data must pre-exist" | No data-status endpoint, no onboarding data screen (migration open question #1 says onboarding is NEW work: design the data story into it) |

Structural duplications underneath: the backend has a SECOND resolver, `backend/core/paths.py:44-53`, honoring `F1_STRAT_DATA_ROOT` but with different precedence (no `~/.f1-strat/data` user-cache fallback, no HF awareness) than `data_cache.get_data_root()` (`data_cache.py:147-170`); a THIRD FastF1 HTTP cache copy sits at `data/f1_cache/` (25.7 MB sqlite) beside the two the P2 audit found; and `scripts/download_data.py:20-27` still documents an uncurated full snapshot (31.7 GB, no `allow_patterns`) as the way to get data, contradicting the curated 7-8 GB `ensure_setup`.

Dead weight: per race folder, `intervals.parquet` (~450-480 KB) plus `pitstops.parquet` (~20 KB) are ~76 percent of the bytes (Budapest 2025: 473.6 + 20.4 of 621 KB) and have **zero runtime consumers** (grep: only the legacy extractors reference those filenames; `replay_engine.py` loads laps + weather + metadata only). Every `ensure_race` pays it.

### 3.6 HF Hub layout and governance

Current: single flat dataset repo `VforVitorio/f1-strategy-dataset` pinned to `revision="main"` (`data_cache.py:58-59`), holding models + raw + processed + radio + RAG together. Verified consistent with the code patterns by the P2 audit (2026-07-04); the 2026-04 restructure memory is superseded. Gaps:

- **Org migration decided, not executed** (FUTURE.md §11: org `f1stratlab`, transfer the dataset, brand rule for cards). Org creation is a manual web step; the code flip is one line by design (`HF_DATASET_REPO_ID`).
- **Mutability:** `main` pinning means any Hub push silently changes what every installed CLI pulls. No release-tag / revision-pin discipline exists.
- **Models inside a dataset repo:** works, but forfeits HF model cards, model-repo tooling, and discoverability; the ecosystem plan (`gridmind`, `radiogate` artifacts) will make the mixed repo the odd one out.
- **Legacy paths still published/locally present:** loose flat parquets at `data/raw/*.parquet` (six 2023 `*_openf1_team_radio.parquet` one-offs), the `Spain`/`Barcelona` duplicate, superseded combined `laps_featured.parquet`/`laps_tiredeg.parquet` beside the per-year versions.
- **No dataset card contract:** schema, era coverage (2022-2025, per #189 F-15), per-folder naming convention, and the sentinel-race guarantee (`data_cache.py:66`) are undocumented on the Hub.

### 3.7 New data features: what exists unused, what is cheap to source

**Tier 0 - already on disk, zero download cost (highest ROI):**

| Signal | Where it sits | What it enables |
|---|---|---|
| Per-lap gap evolution, ~4 s resolution: `interval_in_seconds`, `gap_to_leader`, `drs_window`, `is_lapped` | `data/raw/<year>/<gp>/intervals.parquet` (~27k rows/GP), downloaded by `ensure_race`, consumed by nothing | Rival gap traces for the **Rival Agent** (TFM); richer Arcade/CLI gap displays; undercut-window detection from data instead of the 1.5 s constant |
| Real pit events per race | `pitstops.parquet`, same folders, unused at runtime | Ground truth for Rival Agent pit reconstruction; live pit-detection sanity checks |
| Lap quality flags: `IsAccurate`, `Deleted`, `DeletedReason`, `FastF1Generated`, `IsPersonalBest` | every `laps.parquet` | Data-quality filtering (F-02); a `personal_best` flag in `lap_state` (additive key, contract tolerates extras) |
| Sector session times (`Sector1/2/3SessionTime`), `LapsSincePitStop` | every `laps.parquet` | Mid-lap gap interpolation; stint-age features without recompute |

**Tier 1 - cheap OpenF1 additions on the proven radio-builder infrastructure** (same retry session, same `{year}/{slug}/` layout, `radio_dataset_builder.py` already implements discovery + rate limiting): `/v1/stints` (rival compound history as the pit wall would reconstruct it), `/v1/pit` (pit-lane transit times, complements N15's physical-stop model), `/v1/position` (intra-lap position changes, overtake ground truth densification). Each is a few MB per season, historical back to 2023.

**Tier 2 - heavy, defer until a consumer exists:** `/v1/car_data` and `/v1/location` (~3.7 Hz; hundreds of MB per race); FastF1 full per-lap car telemetry (Arcade already pulls what it renders). Do not ingest speculatively.

**Rival Agent readiness (TFM, chosen June 2026):** its data needs are rival stint/pit history + gap evolution + position/pressure context with ground truth from real 2024-2025 pit stops. Tier 0 + Tier 1 above cover all of it from data this repo already knows how to fetch. This audit scopes only the **data readiness pack** (a documented, validated dataset build); the modeling belongs to the TFM.

**2026-specific features** (Manual Override Mode eligibility, energy-deployment signals, post-DRS successors): owned by #189 F-05, not re-planned here.

### 3.8 `pitlab` Studio: what the core repo must expose for stages 1-2

FUTURE.md §6.4 defines the Studio's stage 1 (ingestion: "buttons per source: year, circuit, session, intervals, radio, FIA, HF") and stage 2 (merge/join with preview). The Studio never reinvents tracking (ClearML / MLflow+DVC / W&B underneath, decision deferred, #189 §6). The contract this audit imposes so those panels become thin wrappers instead of new logic:

- Every acquisition operation invocable as a pure function with progress callbacks: the data-manager façade of F-06 (`status()`, `ensure(profile)`, `ensure_race`, `ensure_radio_corpus`, plus the Phase 4 `build_race_dataset` / existing `build_radio_dataset`).
- Every artifact self-describing: schema + lineage metadata (F-02/F-11), race identity resolved through one module (F-01), so the merge panel can join by canonical key instead of guessing folder names.
- Stages 3-7 (feature engineering, EDA, encoding/split, NLP labeling, retrain/track/registry) depend on Fase 0 (`src/strategy/training/`, #189 F-01) and are out of this audit's scope.

---

## 4. Findings register (P0 - P3)

| ID | P | Finding | Evidence (anchors) | Size |
|---|---|---|---|---|
| F-01 | **P0** | **No canonical race identity.** Five naming schemes + five scattered alias tables; silent radio-corpus loss and missing Cx labels for ~6 GPs/season in the flagship CLI; `Miami` renamed to `Miami_Gardens` across seasons; duplicate `2023/Spain` + `2023/Barcelona` | `gp_slugs.py:34-63`; `data_cache.py:467-473` (swallowed ValueError); `run_simulation_cli.py:344-352, 2314`; `src/simulation/__main__.py:37-43`; `arcade/config.py:301`; `backend/.../strategy.py:723,757`; disk: `data/raw/{2023,2024,2025}` | **M** |
| F-02 | **P0** | **No validation or schema contract at any pipeline boundary**; FastF1 quality flags never consulted; metadata row counts never verified; missing columns degrade to silent `None`s in `lap_state` | `replay_engine.py:66-73, 87-99`; `race_state_manager.py:188-217`; `data/raw/2025/Budapest/laps.parquet` columns; `metadata.json` `record_counts` | **M** |
| F-03 | **P1** | **"Raw" layer contaminated and non-reproducible:** derived features + era-coupled encodings baked at N01 extraction time; no invocable code can regenerate it | `data/raw/**/metadata.json` `"calculated_features"`; N01 (untouchable); #189 F-05/F-06 consume this | **M** (as layering decision + future-code rule) |
| F-04 | **P1** | **Race ingestion is notebook-only; `src` extraction packages are archives presented as live.** No add-a-race path without notebook cells; `CLAUDE.md` §3 misdescribes `src/shared/data_extraction/` | `src/shared/README.md` ("archived"); `src/data_extraction/README.md`; cross-ref #189 F-01/F-02 for the retraining-grade version | **M** |
| F-05 | **P1** | **`ensure_race` downloads ~76 percent dead weight per GP** (intervals + pitstops, zero runtime consumers) | `data_cache.py:432` pattern `data/raw/{y}/{gp}/**`; `replay_engine.py:66-73`; Budapest sizes 473.6 + 20.4 of 621 KB | **S** |
| F-06 | **P1** | **Download/prep UX is CLI-only.** Arcade has no first-run story; backend 404s with no remediation and runs a second resolver with divergent precedence; the SPA migration inherits "data must pre-exist" | `f1_cli.py:93-96`; `arcade/strategy.py:424-437`; `backend/utils/laps_cache.py:37`; `backend/core/paths.py:44-53` vs `data_cache.py:147-170` | **M** |
| F-07 | **P1** | **HF Hub governance gap:** org migration decided but unexecuted; mutable `main` pin; models inside a dataset repo; legacy flat paths published; no dataset card contract | `data_cache.py:58-59`; FUTURE.md §11; `data/raw/*_2023_openf1_team_radio.parquet` | **M** |
| F-08 | **P2** | **Legacy extraction debt awaiting deletion + misleading docs:** `src/shared/`, `src/data_extraction/legacy/`, `src/vision/`; CLAUDE.md §3 out of date | both READMEs; `src/vision/gap_calculation.py` | **S** |
| F-09 | **P2** | **Canonical JSONs are schemaless with silent-empty fallbacks:** `tire_compounds_by_race.json` `.get(year, {})` degrades to no-Cx; keys are yet another identity coupling; `_comment/_note/_usage` pseudo-fields instead of a schema block | `src/simulation/__main__.py:46-60`; `run_simulation_cli.py:336-352`; cross-ref #189 F-06 (2026 era blocks) | **S** |
| F-10 | **P2** | **Unexploited data + cheap new features:** Tier 0 on-disk signals unused (intervals, pitstops, quality flags, sector session times); Tier 1 OpenF1 endpoints (stints/pit/position) trivially sourceable on existing infra; Rival Agent data fully covered by Tier 0+1 | §3.7; `radio_dataset_builder.py` (infra); `intervals.parquet` unused (grep) | **M** (register + wiring plan) |
| F-11 | **P2** | **No lineage/versioning in per-race metadata:** no pipeline version, no FastF1/OpenF1 lib versions, no schema version; re-extractions indistinguishable | `data/raw/2025/Budapest/metadata.json`; align with #189 §6 manifests-v2 (`era` field) | **S** |
| F-12 | **P2** | **`scripts/download_data.py` is an uncurated 31.7 GB trap** contradicting the curated 7-8 GB `ensure_setup`; still the documented "download the dataset" path in its own docstring | `scripts/download_data.py:20-27` vs `data_cache.py:90-123` | **S** |
| F-13 | **P3** | **Data-tree hygiene:** third FastF1 HTTP cache (`data/f1_cache/`, 25.7 MB); loose 2023 flat parquets at `data/raw/` root; superseded combined `laps_featured.parquet` / `laps_tiredeg.parquet` beside per-year files | `ls data/`; cross-ref P2 F-10/F-11 (cache unification owner) | **S** |
| F-14 | **P3** | **Weather joined by fractional lap index, not timestamps** - fine for replay, wrong for live; document as a contract caveat and fix in the live-feed adapter, not now | `race_state_manager.py:307-309` | **S** |
| F-15 | **P3** | **`lap_state` has no machine-readable schema** (docstrings only); SPA, Arcade stream consumers, and the future OpenF1 WS adapter would all benefit from a versioned JSON-schema artifact + additive-keys evolution rule | `race_state_manager.py:338-374`; `replay_engine.py:117-193` (`to_arcade_frame` mirrors it by hand) | **S** |

---

## 5. Phased, chunkable plan (each numbered chunk = one issue/PR; S/M/L effort)

Ordering rationale: identity before contracts (validation needs canonical keys), contracts before UX (the data manager reports against contracts), UX before the Hub migration (the migration flips one constant once the flow is unified), and new features last because they ride on all of it. Phases 0-2 pay off immediately for the current 2022-2025 system; Phases 3-5 are the forward investment.

**Phase 0 - Race identity and legacy truth (M)**
1. **[M] F-01:** grow `src/f1_strat_manager/gp_slugs.py` into a race-identity module: one table per GP with canonical key, per-season raw folder name(s), radio slug, compounds key, official event name, OpenF1 country/circuit. All five alias sites (`simulation/__main__.py`, `arcade/config.py`, `arcade/data.py`, `arcade/strategy.py`, backend `strategy.py`) consume it; `ensure_radio_corpus` resolves through it so folder-name inputs stop silently losing radios. Untouchable CLI unchanged: it already delegates to `gp_slugs` and the compounds lookup gets fixed via the JSON keys or a caller-side resolve in editable code.
2. **[S] F-01 (data side):** reconcile the tree: remove the `2023/Spain` duplicate (keep `Barcelona`), decide rename-vs-alias for `Miami`/`Miami_Gardens` (proposed: alias now, physical rename batched into the Phase 3 Hub migration as the single breaking-change window), mirror on the Hub.
3. **[S] F-08 + F-12:** verify remaining `from src.shared` notebook references, then delete or quarantine `src/shared/` + `src/data_extraction/legacy/` + `src/vision/` under `legacy/`; fix `CLAUDE.md` §3; repoint `scripts/download_data.py` at curated profiles (`ensure_setup`; full pull behind an explicit `--full`).

**Phase 1 - Data contracts and validation (M)**
4. **[M] F-02:** schema manifests per artifact family (laps/weather/intervals/pitstops/radio parquets: required columns, dtypes, nullability) + a `f1-data verify` command (or `scripts/verify_data.py`) checking schema, row counts vs metadata, and NaN rates; cheap load-time assertions in editable callers (`replay_engine`, arcade loader, backend laps_cache). Fail loud on structural breaks, warn on quality flags.
5. **[S] F-11:** metadata v2 for race folders: `schema_version`, builder + FastF1/OpenF1 versions, and the `era`/`regulation_cycle` field aligned with #189's manifests-v2 so one convention covers both.
6. **[S] F-09:** JSON schema + strict accessor for `tire_compounds_by_race.json` (unknown year/GP = actionable error through the identity module, not `{}`).
7. **[S] F-15:** publish `lap_state` as a versioned JSON schema artifact + the additive-keys evolution rule; document the weather caveat (F-14) in it.

**Phase 2 - Download and preparation UX unification (M)**
8. **[M] F-06:** additive `src/f1_strat_manager/data_manager.py` façade: `status()` (what is on disk vs available, with sizes), `ensure(profile)` (sim / arcade / backend / full), progress + stage callbacks. Consumers: CLI menu (composes with P2's #168 progress fix and F-12 full-calendar picker, not duplicating them), Arcade menu (data-presence gate + fetch-on-select using the existing banner staging), backend `GET /api/v1/data/status` + `POST /api/v1/data/ensure` for the SPA's onboarding/data screen (feeds migration open question #1).
9. **[S] F-06 (resolver):** collapse `backend/core/paths.py` onto `data_cache.get_data_root()` semantics (keep the env override and Docker fallback; add the user-cache fallback) so both resolvers cannot diverge again.
10. **[S] F-05:** scope `ensure_race` patterns to runtime needs by default (laps + weather + metadata), `--with-intervals/--with-pitstops` flags until Phase 4 item 13 makes them runtime inputs (then they return to the default).

**Phase 3 - HF Hub restructure to `f1stratlab` (M)**
11. **[M] F-07:** execute FUTURE.md §11: create the org (manual web step), transfer or re-create the dataset, flip `HF_DATASET_REPO_ID` (`data_cache.py:58`), adopt revision pinning per release (release-please tag = HF revision; `HF_DATASET_REVISION` stops being permanently `main`), write the dataset card (schema, naming convention, era coverage per #189 F-15, sentinel race, brand rule), prune legacy flat paths, execute the deferred physical renames from item 2. Decide models-split (dataset repo vs `f1stratlab` model repo) per open question 3.

**Phase 4 - Ingestion-as-code and new data features (L)**
12. **[M] F-03 + F-04:** define the bronze/silver/gold layering for all FUTURE ingestion (bronze = provider-verbatim, silver = enriched + encodings, gold = model-ready), freeze the current tree as "bronze-plus-patch v1", and build `scripts/build_race_dataset.py` on the radio builder's template (class-based, retry session, idempotent, `--years/--gps/--skip-existing`) reproducing N01's per-race output additively. This is the near-term add-a-race path; #189 Phase 2 extends the same home with FP/Q/Sprint session types.
13. **[S] F-10 (Tier 0):** wire `intervals.parquet` into a runtime gap provider behind an additive `lap_state` key (respecting the single-driver boundary: rivals' gaps are timing-screen data); expose `is_personal_best` and quality-flag filtering from laps.
14. **[M] F-10 (Tier 1):** extend the OpenF1 ingestion with `/v1/stints`, `/v1/pit`, `/v1/position` per race (schema-versioned per Phase 1, published to the Hub); explicitly NOT `/v1/car_data` / `/v1/location` until a consumer exists.
15. **[M] F-10 (Rival readiness pack):** the Rival Agent ground-truth dataset build (2024-2025 rival pit/stint/gap reconstruction from Tier 0+1 sources) as a documented, validated, Hub-published artifact with a dataset card. Modeling stays in the TFM.

**Phase 5 - `pitlab` stage 1-2 contract (S, plan-level)**
16. **[S]** Freeze the Studio ingestion contract: the Phase 2 data-manager façade + Phase 4 builders as the invocable API `pitlab` stage 1 wraps; identity module + schema manifests as what stage 2 (merge preview) joins on. Tracker decision and stages 3-7 remain with #189 §6 / FUTURE.md §12.2. Deliverable: a one-page contract doc in the repo so `pitlab` can start without re-reading this audit.

Dependency notes: 1 gates 4-6 and 8 (identity keys inside contracts and status reports); 8 gates 11 (migrate once, through one flow) and the SPA data screen; 12 gates 13-15; nothing here blocks or is blocked by the frontend migration except the shared onboarding screen (coordinate with S5). Fase 0 (`src/strategy/training/`, #189 F-01) is NOT a dependency of anything in Phases 0-3 here, by construction.

---

## 6. Open questions (need Víctor's decision)

1. **Rename vs alias for divergent raw folders** (`Miami_Gardens`, the underscore forms): physical rename is cleaner but breaks HF paths and every local cache; proposal is alias-only in Phase 0, physical rename batched into the Phase 3 Hub migration as the single breaking window. Confirm.
2. **Should `ensure_race` keep pulling intervals/pitstops by default** once Phase 4 makes them runtime inputs, or stay opt-in? Proposal: default-on after item 13 ships, opt-out flag retained.
3. **Models placement on the Hub:** keep inside `f1stratlab/f1-strategy-dataset` (one snapshot call, current behavior) vs split into a `f1stratlab` model repo (cards, discoverability, ecosystem symmetry). Splitting means two downloads in `ensure_setup`; the P2 audit's single-pass progress fix (#168) should land first either way.
4. **Revision pinning policy:** pin the HF revision per release tag (reproducible installs, manual bump per data release) vs stay on `main` (always-fresh, silently mutable). Proposal: pin in released CLI builds, `main` in dev checkouts.
5. **SPA data screen timing:** design the data-status/onboarding surface into migration sprint S1 (foundations) or defer to S5 (onboarding)? The backend endpoints (item 8) are small and could land in migration S0 (backend hardening).
6. **`src/shared/` deletion vs quarantine:** its README says deletion is safe once the last notebook import is gone; the check is cheap, the notebooks are untouchable either way. Delete, or move under `legacy/`?
7. **Rival readiness pack scope:** reconstruct ground truth for the ~5 surrounding rivals per validated GP (cheap, TFM-sized) or the full grid for the full 2024-2025 seasons (heavier, more reusable)? Proposal: full grid for the data pack (it is a dataset artifact), let the TFM subset it.

---

## 7. Verification protocol (when this plan is executed)

- **Identity (Phase 0):** parametrized test: every folder under `data/raw/{2023,2024,2025}/` resolves through the identity module to a compounds key, a radio slug, and an official event name, with zero unknowns; smoke sim on an aliased GP (`Miami_Gardens`/`Miami` 2025, `--no-llm --laps 1-5`) shows Cx labels AND a non-empty radio corpus; `2023/Spain` gone; CLI regression per the established protocol (`python scripts/run_simulation_cli.py Sakhir HAM Mercedes --no-llm --laps 1-10`, output diffed).
- **Contracts (Phase 1):** `f1-data verify` green on the full local tree; a deliberately truncated laps parquet fixture fails loud with the artifact name and the violated rule; metadata v2 present on newly built races.
- **UX (Phase 2):** scratch `F1_STRAT_DATA_ROOT` first-run walkthrough per surface: CLI (single progress-visible download, composes with #168), Arcade (menu shows data status, fetch-on-select, no frozen window), backend (`/data/status` correct before and after `ensure`); `ensure_race` byte reduction measured (expect ~4x smaller per GP until item 13).
- **Hub (Phase 3):** fresh `uv tool install` against the new org id completes setup and runs the sentinel race; pinned-revision install reproduces byte-identical critical files; old repo id documented as redirect/deprecated.
- **Features (Phase 4):** gap-provider values cross-checked against `RaceStateManager` end-of-lap gaps at lap boundaries (within timing tolerance); new OpenF1 parquets pass the Phase 1 verify; rival readiness pack validated against FastF1 pit stop counts per race (exact match or documented delta).
- **Nothing near the boot path ships without** the P2 §6 re-timing probes, and nothing touching agent inputs ships without the no-LLM regression diff above.
