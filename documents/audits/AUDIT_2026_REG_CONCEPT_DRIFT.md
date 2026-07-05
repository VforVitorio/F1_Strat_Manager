# AUDIT 2026-REG - 2026-regulation readiness / concept-drift survival plan

**Auditor:** Fable 5 · **Date:** 2026-07-05 · **Repo:** `F1_Strat_Manager` (read-only pass, no code changed)
**Scope:** everything that the 2026 F1 technical and sporting regulation reset breaks or degrades: the seven ML predictors (N06-N16), the circuit clustering (K=4), the agents / orchestrator / guardrails layer, the NLP pipeline, the RAG index, and the data pipeline that would feed retraining. Plus: the retraining data strategy (historical + FP + Qualy + Sprint, infer in race), drift detection and adaptation (RevIN, year-embedding, ShifTS), calibration maintenance, and the contract with the planned `pitlab` MLOps Studio.
**This is the ⭐ 2026-reg / concept-drift sub-scope of the P5 Data audit** (`documents/audits/AUDITS_BACKLOG.md`, P5 section).
**Hard constraints honored in every remedy:** backend stays FastAPI; LLM = OpenAI / LM Studio, never Anthropic; UNTOUCHABLE (duplicate before modifying / additive entry points only): `scripts/run_simulation_cli.py`, `src/agents/` internals, `notebooks/**`, `legacy/**`. Plan only, no code.

---

## 1. Framing (mandatory)

**This audit is NOT a current priority and is very future work, but it WILL have to be done eventually, and when it is, it must be planned with maximum rigor and care.** It is being produced now only to capture the most rigorous possible plan while top-tier model access is available. Treat this document as the definitive forward plan for surviving the 2026 F1 technical-regulation change. Nothing in it should be executed before the frontend migration (core priority #1) and the higher-priority audits; everything in it should be re-verified against the final published 2026 regulations and the actual state of the repo at execution time.

The thesis itself already commits to this work: future-work item #3 (`Docs/Memoria/capitulos/06_conclusiones.tex`, "Adaptación al reglamento técnico y deportivo de 2026") and item #4 ("Despliegue de un año completo con telemetría en vivo y reentrenamiento progresivo por Gran Premio"). The IEEE paper frames the 2026 rule change as the system's "planned obsolescence boundary" and claims adaptation is "a planned maintenance operation rather than a research contribution" (`Docs/Memoria/Paper/main.tex`, Discussion, ~lines 1053-1068). This audit converts that claim into an operational plan and, where the claim is optimistic, corrects it.

---

## 2. Executive summary

Every model in the system encodes 2022-2025 ground-effect-era physics: the seven predictors are trained 2023-2024 and tested on 2025, the K=4 circuit clustering is fitted on 2023-2024 race laps, the calibrators (Platt, quantile, MC Dropout sigmas) are frozen on 2024 validation folds, and a web of era-coupled artifacts (compound maps, team encoders, stint-capacity constants, pit-lane traversal lookups, cluster parquets pinned to 2025) is hardcoded into agent configs. The 2026 reset (roughly 50 percent electric power split without MGU-H, active aero replacing DRS, cars ~30 kg lighter and narrower, reformulated and narrower Pirelli compounds, a new circuit and a new team) invalidates feature semantics (DRS features die outright), shifts every learned distribution, and silently rots the probability calibration that the orchestrator's Monte Carlo layer depends on. Within-cycle drift is already documented in the project's own results: the negative permutation importance of TyreAge on the 2025 pit-duration holdout and the P05-P95 coverage of 0.70 against a nominal 0.90.

The retraining thesis (train on historical seasons + FP + Qualy + Sprint, infer in race, transfer by K=4 cluster, accumulate year over year, learn incrementally per GP) is sound for the regression family (pace, tire degradation) but only partially valid for the classifier family (overtake, undercut, SC, pit duration), whose labels only exist in races. Today there is zero pipeline support for it: the download pipeline only ever fetches race sessions, `src/strategy/training/` is empty, and `src/shared/data_extraction/` is legacy one-off code.

The phased path: (0) extract training code out of the notebooks and fix artifact portability; (1) build the measurement layer (eval harness, calibration verification, drift monitors) BEFORE the 2026 season starts; (2) build multi-session ingestion (FP/Q/Sprint) and label-reconstruction; (3) refresh every era-coupled artifact and re-engineer the DRS-coupled features behind versioned manifests; (4) implement adaptation (RevIN / year-embedding / ShifTS, per-GP incremental retraining, cluster pooling) through `pitlab`. Phases 0-1 have value even if 2026 adaptation never ships, because they harden the current system.

---

## 3. What 2026 changes and what breaks

### 3.1 The regulation delta (verify against final published regs at execution time)

Stated per the thesis summary (`06_conclusiones.tex`, future work #3) and the published 2026 regulations as of this audit's knowledge:

| Change | Strategic consequence |
|---|---|
| Power unit: ~50 percent electric split, MGU-H eliminated, energy-management rules | Lap-time profiles gain an energy-deployment dimension; in-lap/out-lap and straight-line speed patterns shift; "pace" is no longer a pure tire+fuel story |
| Active aero (X-mode / Z-mode) for all cars; DRS removed as an overtaking device; Manual Override Mode (electric boost for the chasing car) replaces DRS's role | `drs_window`-family features become semantically dead; overtaking dynamics change qualitatively; speed-trap deltas redistribute |
| Cars ~30 kg lighter, narrower (2000 to ~1900 mm), shorter wheelbase | Fuel-effect-per-lap constants and FuelLoad proxies change; degradation-vs-load relationship changes |
| Pirelli compounds reformulated, tires narrower (front ~25 mm, rear ~30 mm) | The entire tire-degradation stack (curves, cliffs, warm-up, stint capacities) is new physics; undercut strength (driven by warm-up delta) changes |
| 100 percent sustainable fuel, energy-flow-based limits | Fuel-burn correction models fitted on 2023-2025 races are invalid |
| Calendar: Madrid (Madring) debuts, Imola drops | New circuit has no cluster assignment, no SC rate, no undercut rate, no pit-lane traversal time |
| Grid: Cadillac joins (11th team), Sauber becomes Audi | Team encoders and label encoders contain unseen categories; some will raise at inference |

### 3.2 Per-model break analysis

Severity: **HIGH** = predictions systematically wrong or features invalid, retrain mandatory before trusting output. **MED** = degraded but usable as a weak prior with recalibration. **LOW** = mostly era-agnostic, targeted refresh only.

| Model / component | Regulation-coupled features & artifacts (evidence) | Severity | Why |
|---|---|---|---|
| **N06 pace (XGBoost delta lap time, MAE ~0.41 s)** | `Year` (2026 unseen by tree splits, collapses to "2025-like"), `CompoundID`, `TeamID`, `Cluster`, `FuelLoad`, `FuelEffect`, `Prev_SpeedST`, `mean_sector_speed` (`data/models/lap_time/xgb_laptime_delta_feature_names.json`, 25 features); reference laps and team encoding pinned to `laps_featured_2025.parquet` and clusters to `circuit_clusters_k4_2025.parquet` (`data/models/agents/pace_agent_config_v1.json` "artifacts"); `pace_agent.py:433` filters reference laps by `Year == year`, which returns empty for 2026 until a 2026 featured parquet exists | **HIGH** | Absolute pace, fuel effect, and speed-trap distributions all shift; `Year` cannot extrapolate in GBDTs; the session-median delta output breaks outright on the empty 2026 reference filter |
| **N07-N10 tire degradation (TireDegTCN + per-compound fine-tunes + MC Dropout)** | Target `FuelAdjustedDegAbsolute` depends on a fuel-correction model fitted on 2023-2025 race data (documented over-subtraction failure in N09 Step 9); 42-feature set incl. `AbsoluteCompoundID`, `CompoundHardness`, `FuelLoad`, `FuelEffect`, speed telemetry (`data/processed/tiredeg_feature_manifest.json`); per-compound window sizes from the 2023-2025 stint-length distribution, 3,220 training stints total (`data/processed/tiredeg_sequence_config.json`); per-compound MC sigmas frozen (`data/models/tire_degradation/mc_dropout_calibration.json`); `fuel_effect_per_lap: 0.055` hardcoded (`data/models/agents/tire_agent_config_v1.json` "feature_pipeline"); compound-to-C-number maps per race (`data/tire_compounds_by_race.json`, blocks for 2023/2024/2025 only) | **HIGH** (highest in the system) | Reformulated, narrower compounds = new degradation physics: curve shapes, cliff onsets, stint-length distributions (hence window sizes), fuel correction, and uncertainty calibration are all stale at once. This is exactly where N09 parked RevIN / ShifTS / TAFAS for "v5" (N09 Step 9 markdown) |
| **N11-N12 overtake (LightGBM, AUC-PR 0.5491, threshold 0.7976)** | `drs_window`, `drs_ready_gap` (semantically dead without DRS), `speed_trap_delta` (active-aero redistribution), `circuit_cluster`, `compound_x/y` (`data/models/overtake_probability/model_config.json`); Platt calibrator fitted on val-2024 scores | **HIGH** | Two features must be removed or re-mapped to Manual-Override-Mode eligibility; overtaking difficulty changes globally; threshold and calibration are era-specific. Labels (position swaps) only exist in races and Sprints, so retraining lags the season |
| **N12B causal TCN (archived negative result)** | n/a | LOW | Stays archived. Its lesson (explicit feature engineering beats raw sequences under 20k samples) applies even more in a data-poor 2026 cold start: do NOT reach for sequence models early in the new era |
| **N13-N14 safety car (LightGBM soft prior, AUC-PR 0.0723, lift 1.67x)** | `circuit_sc_rate` (historical per-circuit rate), `circuit_cluster`, `tyre_age_high_risk_count` (compound-coupled), weather + incident features mostly era-agnostic (`data/models/safety_car_probability/feature_list_v1.json`, 32 features); Platt coef/intercept fitted on val-2024 (same file, "calibration") | **MED** | Incident semantics survive; the historical priors (SC rate per circuit, base rate 0.043) go stale, and new-regulation first seasons historically spike incident rates. Already treated as a soft prior, so degradation is tolerable if recalibrated |
| **N15 pit duration (HistGBT quantiles P05/P50/P95)** | `team` LabelEncoder with the 12 teams of 2023-2025 (unseen Audi / Cadillac will raise on transform), `year`, `team_year_median` (fallback constant 2.8 s), `circuit_traversal_lookup` keyed by GP name with 24 entries and no Madrid (`data/models/pit_prediction/model_config.json`) | **MED** model / **HIGH** artifacts | The physical stop itself (crew + 4 wheels) changes least; the encoders and lookups break hard. Within-cycle drift is already proven here: TyreAge permutation importance of -0.848 s on the 2025 holdout (Paper, `main.tex` ~795-801); P05-P95 empirical coverage is 0.7047 against nominal 0.90 (same config, "eval"), i.e. the quantile calibration is ALREADY rotten before 2026 even starts |
| **N16 undercut (LightGBM, AUC-PR 0.6739, threshold 0.522)** | `pit_delta_X`, `circuit_undercut_rate`, `team_x_undercut_rate` (historical aggregates; unseen teams), `compound_delta`, tyre-life features (`data/models/pit_prediction/model_config_undercut_v1.json`); Platt on val-2024 | **HIGH** | Undercut success is driven by tire warm-up delta and pit loss, both reset by the new compounds and cars; the historical per-circuit/per-team rates are era priors that must be rebuilt |
| **K=4 circuit clustering (N03)** | Fitted on race-lap statistics: mean/std/min lap time, degradation slope, stint length/variance (N03 cells 16-21); scaler + kmeans pickles (`data/models/k_means_circuit_clustering/`); consumed as a feature by N06/N12/N14 and as threshold routing by N31 (`strategy_orchestrator_config_v1.json` "cluster_aware_thresholds", source pinned to `circuit_features_with_clusters_k4_2025.parquet`) | **HIGH** (transversal) | Every clustering input is recomputed physics; clusters cannot be re-derived until enough 2026 races exist (the thesis itself scopes this to "first half of 2026"); Madrid has no assignment at all. Everything downstream of `Cluster` inherits the staleness |
| **Agents / orchestrator N31 / guardrails** | `_STINT_CAPACITY_LAPS = {'SOFT': 18, 'MEDIUM': 30, 'HARD': 38}` (Heilmeier-era constants, `src/agents/pit_strategy_agent.py:61`); `_COMPOUND_FALLBACK` color-to-ID map (same file, ~line 58); per-cluster SC thresholds and N26 cliff thresholds incl. a hand override for Mexico City (`strategy_orchestrator_config_v1.json`); MC layer (500 samples, score = alpha*E + (1-alpha)*P10) consumes sub-agent distributions, so calibration rot propagates directly into action scores; `encoding_maps.json` team_id has no 2026 entries and already collides (`Kick Sauber: 0`, `Racing Bulls: 0`); agent configs embed absolute Windows paths (`pace_agent_config_v1.json` "model.path" = `c:\Users\victo\...`) | **HIGH** (constants + calibration flow) / structural code **LOW** | The LangGraph topology, `lap_state` contract (`src/simulation/race_state_manager.py`), MoE routing and Pydantic schemas are regulation-agnostic and survive as-is. What breaks is every embedded number. All fixes must be additive or config-side because `src/agents/` internals are untouchable |
| **RCM context resolver** | RCM classification and `sc_currently_active` propagation (`src/agents/race_situation_agent.py:1180` `_sc_active_from_rcm`, `radio_agent.py` `_classify_rcm_event`) | **LOW-MED** | Race-control message grammar is stable, but 2026 may add new message types (override-mode / energy-related directives). Extending the resolver is already thesis future-work #5; fold 2026 message types into that extension |
| **NLP pipeline (Whisper, RoBERTa sentiment, SetFit intent, BERT NER)** | Whisper and sentiment are era-agnostic. Intent taxonomy is era-agnostic but 2026 radio vocabulary shifts ("override", "energy", "harvest", no more "DRS enabled" calls). NER entities include team/driver/circuit names: Audi, Cadillac, Madring are unseen surface forms for a model already data-capped at F1 0.4151 on 399 examples (thesis limitation #5). Radio corpus is 2023-2025 (`data/processed/radio_nlp/`) | **LOW** overall | Targeted refresh, not retrain-everything: era-tag the corpus, add 2026 vocabulary examples, re-check intent boundaries. Synergy: this is exactly the `radiogate` mega-corpus initiative (FUTURE.md §5) |
| **RAG / FIA regulations** | Index built from `data/rag/documents/sporting_regs_{2023,2024,2025}.pdf`; `scripts/download_fia_pdfs.py:68` `supported_years: [2023, 2024, 2025]` | **MED** | Regulation citations (e.g. the Qatar 25-lap Pirelli mandate cited in the RCM validation case) are season-specific. The paper already prescribes the fix: point the download script at the 2026 PDFs and rebuild the index. Cheap, but mandatory before any 2026 run, otherwise the RAG agent will confidently cite 2025 rules |
| **Data pipeline (the retraining feeder)** | `notebooks/data_engineering/N01` downloads **race sessions only** (`ff1.get_session(year, gp_name, 'R')`, N01 cell 14); `src/shared/data_extraction/fastf1_extractor.py` is hardcoded to Spain 2023, `openf1_extractor.py` hardcodes `session_key 9102` (Spain 2023) and says so in its own comments, `data_augmentation.py` is a YOLO image augmenter from an unrelated project; `src/strategy/training/` is empty and `src/strategy/` is stale jupytext exports predating the production models (`src/strategy/README.md`) | **HIGH** (blocking) | The entire "FP + Qualy + Sprint" retraining regime has zero pipeline support today. The engineering home for retraining does not exist in `src/` |

### 3.3 Feature-manifest fields that become invalid or need recalibration (concrete list)

- **Dead on arrival (remove or re-map):** `drs_window`, `drs_ready_gap` (N12 manifest). Successor feature: Manual-Override-Mode eligibility (gap-based, from the 2026 sporting regs).
- **Semantics changed, keep name but re-derive:** `FuelLoad`, `FuelEffect`, `FuelAdjustedDegAbsolute` (new fuel mass and burn curve; the fuel-correction model must be refitted on 2026 races before the tire-deg target can even be computed), `speed_trap_delta`, `SpeedI1/I2/FL/ST` distributions, `mean_sector_speed`, `lap_time_pct_of_race_fastest`, `year_circuit_median`.
- **Stale priors, recompute from 2026 data as it accumulates:** `Cluster` / `circuit_cluster`, `circuit_sc_rate`, `circuit_undercut_rate`, `team_x_undercut_rate`, `team_year_median`, `team_pace_rank`.
- **Encoder / lookup breaks (hard errors, not drift):** `TeamID` (`encoding_maps.json`), N15 `team` LabelEncoder, `circuit_traversal_lookup` (Madrid missing), `tire_compounds_by_race.json` (no 2026 block), `gp_slugs` calendar entries, `Year` = 2026 unseen by all GBDT splits.
- **Calibration artifacts that must be refreshed, not just retrained:** Platt coef/intercept for SC, overtake, undercut (all fitted on val-2024); N15 quantile heads (coverage already 0.70 vs 0.90); MC Dropout per-compound sigmas; every decision threshold (0.7976 overtake, 0.522 undercut, 0.2335 SC) which is only meaningful relative to a calibrated score distribution.

---

## 4. Retraining data strategy (critical assessment of the FP + Qualy + Sprint regime)

The intended regime (FUTURE.md §2.2, thesis future-work #4): permanent, for all seasons: historical prior seasons + FP + Qualifying + Sprint of the current weekend, infer during the race; transfer by K=4 cluster pooling; accumulate year over year; incremental per-GP learning. Verdict: **directionally right, but it must be stratified per model family, because label availability differs radically.**

### 4.1 What each session actually provides (FastF1/OpenF1 availability in a 2026 weekend)

FastF1 exposes `FP1/FP2/FP3/Q/SQ/S/R` session objects with laps, weather and messages; OpenF1 provides near-real-time laps, intervals, radio, race control. Data is available within hours of each session (OpenF1 near-live, FastF1 after session end). Pre-season testing sessions are also loadable and are the only new-era data existing before round 1.

| Session | Usable for | NOT usable for | Caveats |
|---|---|---|---|
| FP1-FP3 | Pace shape, long-run degradation stints (FP2 race sims), compound behavior, track evolution | Overtake/undercut/pit/SC labels (no racing) | **Fuel load unknown** (teams run undisclosed programs); `FuelLoad = laps_remaining/total_laps` is invalid in FP; engine modes and sandbagging add noise; a fuel-load estimator (from long-run lap-time slope) is a prerequisite work item |
| Qualifying | Low-fuel pace ceiling, single-lap speed profiles, gap structure of the field | Everything strategy-labeled | Max ~10-15 laps/driver, one compound mostly |
| Sprint (~1/3 race distance, 6 weekends planned in 2026) | The ONLY pre-race race-like data: real overtakes, real degradation under racing, occasionally SC | Pit stops (no mandatory stop in Sprints) and undercuts effectively absent | Only 6 of 24 weekends; over-weighting Sprint circuits biases the pool |
| Race | All labels: laps, stints, overtake pairs, pit stops, undercut attempts, SC events | n/a | Arrives only as the season progresses |

### 4.2 Volume reality check (per weekend, ~20-car grid, against current training sizes)

| Signal | Per conventional weekend (est.) | Current training size | Implication |
|---|---|---|---|
| Race laps (fully labeled) | ~1,100 | N06: 22,106 train laps (`feature_manifest_laptime.json`) | ~10 races to reach half the current lap volume; mid-season retrain is realistic, week-1 is not |
| Degradation stints | Race ~40-60; FP2 long runs ~30-50 more if fuel-estimated | 3,220 stints, 2023-2024 (`tiredeg_sequence_config.json` per-compound n_stints) | Tire-deg TCN is retrainable earliest with FP long runs folded in; per-compound fine-tunes (N10) need a near-full season |
| Overtake pairs | ~600 per race + Sprint extras | 18,277 train pairs (`overtake .. model_config.json`) | Classifier retrain feasible around mid-season; until then cross-era model + recalibration only |
| Pit stops | ~30-40 per race | 2 seasons of stops | Quantile heads refittable after ~8-10 races; refit calibration earlier |
| SC events | 0-1 per race (base rate 0.043 per 3-lap window) | 2 seasons | Rare-event model needs a full season minimum; run 2026 with recalibrated old prior + wide uncertainty |

### 4.3 Cold start (rounds 1 to ~6 of 2026), the honest plan

1. **Before round 1:** only pre-season testing exists. No model should claim 2022-2025-level accuracy. Ship an explicit era-confidence flag in outputs (additive field, orchestrator synthesis already surfaces confidence).
2. **Old era as weak prior vs discard** (FUTURE.md open decision #6): do not decide a priori; decide per model from measured round-1-to-3 drift (Phase 1 monitors). Expectation from the break analysis: keep old data as prior for pit duration and SC (structure survives), heavily down-weight or discard for tire degradation and overtake (physics reset).
3. **Cheapest first lever = recalibration, not retraining.** Refit Platt scalers, quantile offsets and MC-sigma scaling on the first 2-3 races of 2026 (hundreds of samples suffice for 1-2 parameter calibrators) while full retrains wait for volume. This keeps the Monte Carlo layer honest even while the underlying rankers are stale.
4. **Per-weekend bootstrapping:** FP long runs feed the tire-deg and pace models (after the fuel-load estimator exists); Qualifying anchors the pace ceiling; Sprint (when present) provides the first race-like labels of the weekend.

### 4.4 K=4 cluster transfer, assessment

- The clustering inputs are race-derived statistics (N03), so 2026 clusters **cannot exist before mid-season**; the thesis already scopes re-clustering to "the first half of 2026" (future-work #3). The K=4 structure itself (archetypes) may survive even if memberships move; K should be re-validated (silhouette was a modest 0.201 even in-era, per `06_conclusiones.tex` objectives table).
- **Interim protocol (needed, currently undefined):** rounds 1 to ~12 run with 2025 cluster assignments flagged as stale priors; Madrid gets a provisional assignment by nearest circuit-geometry analogue (documented, manual); at mid-season, re-fit scaler + kmeans on 2026 races, re-emit `circuit_features_with_clusters_k4_2026.parquet`, and re-tune every cluster-keyed threshold in `strategy_orchestrator_config_v1.json` (SC thresholds, N26 cliff thresholds, plus the Mexico City hand override, which must be re-validated rather than blindly carried over).
- Cluster pooling for training (train a circuit with its cluster peers) is the right data-scarcity lever mid-season, and it is the mechanism that makes "retrain for round 13's circuit with 12 races of data" viable.

### 4.5 Year-over-year accumulation and incremental per-GP (2027+)

The thesis's Bahrain-2027 example (base = Bahrain 2026, fine-tune on the 2027 weekend's FP + Q) is the permanent regime. Requirements it imposes that do not exist today: per-GP model lineage (which base, which fine-tune data, which calibration), acceptance gates (never auto-promote a fine-tune that degrades holdout metrics), and rollback. These are precisely `pitlab` features (§6); incremental learning without them is how silent regressions ship. Overfitting risk on single-GP fine-tunes is real (one race = one circuit, one weather); the cluster pool is the regularizer.

---

## 5. Drift detection and adaptation

### 5.1 The three parked techniques (N09 Step 9 names all three; where each plugs in)

| Technique | What it buys | Where it plugs in | Cost / risk |
|---|---|---|---|
| **RevIN** (Kim et al., ICLR 2022, reversible instance normalization) | The TCN learns the SHAPE of the degradation curve independently of its absolute level; N09's own diagnostics motivate it (MAE 2.6x worse on 31+ lap stints, level-shift failure). Directly transfers curve knowledge across compound reformulations | TireDegTCN only (N09/N10 architecture, re-implemented in `src/strategy/training/`, never by editing notebooks). Highest-value single adaptation change | Small (a normalization layer pair); needs re-training to adopt; must re-derive MC-sigma calibration afterwards |
| **Year-embedding** | Season-specific offsets learned jointly, so old seasons remain usable as data instead of being discarded; replaces the numeric `Year` feature that GBDTs cannot extrapolate | TCN natively (embedding layer). For the GBDT family the equivalent is: treat year/era as a categorical with explicit handling, or train era-offset correction heads | Cold-start problem: the 2026 embedding is untrained at round 1 (initialize from 2025's vector or the mean; document the choice). For GBDTs this is a feature-engineering change, manifest-versioned |
| **ShifTS** (arXiv 2510.14814) / **TAFAS** (arXiv 2501.04970, also cited in N09) | Test-time / label-free adaptation to distribution shift, useful precisely in the cold-start window where labels lag | TireDegTCN inference path (additive wrapper around `predict_tire_degradation`, `src/strategy/inference/tire_predictor.py`) | Research-grade, not engineering-grade: gate behind an ablation on 2025 data first (simulate drift by training on 2023 only, adapting on 2024). Do not ship un-ablated adaptation into the orchestrator |

Adoption order: RevIN first (proven, cheap, addresses a documented failure), year-embedding second (data-retention lever), ShifTS/TAFAS last (only if the monitors show residual drift the first two do not absorb).

### 5.2 Monitoring signals (what to measure, per GP, from round 1 of 2026)

- **Feature drift:** PSI or KS statistic per manifest feature, per GP, against the frozen 2023-2025 training reference. Watch-list first: speed traps, deg rates, stint lengths, pit deltas.
- **Performance drift:** per-GP MAE (pace, tire-deg, pit P50) and AUC-PR on accumulating labels (overtake, undercut, SC) against the 2025 baseline bands (0.41 s, 0.71 s, 0.487 s, 0.549, 0.674, 0.072 respectively).
- **Calibration health (the silent-rot channel):** reliability diagrams + Brier score for the three classifiers (this closes thesis limitation #7, which admits calibration was never verified); empirical P05-P95 coverage for pit quantiles (already 0.70 vs 0.90 nominal on 2025, so this monitor is overdue regardless of 2026); MC-Dropout sigma-vs-realized-error ratio per compound against `mc_dropout_calibration.json`.
- **Structural drift:** cluster-assignment stability (do 2026 circuit stats still fall inside their 2025 cluster hulls); base-rate tracking for SC and overtake rates.

### 5.3 Retrain / recalibrate triggers (proposed defaults, tune at execution)

| Trigger | Action |
|---|---|
| Calibration monitor out of band on any classifier for 2 consecutive GPs | Refit that Platt calibrator on accumulated 2026 scores (cheap, days) |
| Pit quantile coverage < 0.80 over rolling 5 GPs | Refit quantile offsets, then heads when volume allows |
| Pace or tire-deg MAE > 1.5x its 2025 baseline for 3 consecutive GPs | Schedule model retrain on the 2026 pool (cluster-pooled) |
| PSI > 0.25 on 3+ watch-list features at one GP | Investigate before trusting that GP's outputs; annotate the run |
| Cluster instability (2+ circuits migrate) | Trigger the mid-season re-clustering + threshold re-tune |
| MC sigma ratio outside [0.7, 1.3] per compound | Re-derive MC-Dropout calibration JSON |

All monitors write to the experiment tracker (§6) so that "is the system healthy for the current regulation" becomes a dashboard question, which is also the gate FUTURE.md §10 imposes before the Live bot (Fase 4 before Fase 5: never publish on drifting models).

---

## 6. How it feeds `pitlab` (MLOps Studio): the contract

This audit is the requirements document for `pitlab`'s retraining surface (FUTURE.md §6). The dependency is mutual: the 2026 adaptation is the first real `pitlab` workload, and `pitlab` is how the adaptation stays repeatable every season after.

- **Phase 0 prerequisite (FUTURE.md §6.3, "the real bottleneck"):** extract training + transformations from notebooks N01-N16 into `src/strategy/training/` as pure, invocable functions. Since `notebooks/**` are untouchable, this is duplicate-and-extract: notebooks stay frozen as the historical record; the extracted modules become the single production path. Contract per model family: `build_dataset(era, sessions, cluster_pool) -> parquet`, `train(config) -> artifact`, `evaluate(artifact, holdout) -> metrics`, `calibrate(artifact, val) -> calibrator`, `export() -> the exact artifact set the agents consume` (model file + manifest + encoders + thresholds + calibration JSONs). The export contract is already implicitly defined by what `data/models/**` and the agent configs read; make it explicit and versioned.
- **Manifests as the single source of truth:** FUTURE.md §6.4 already mandates that the Studio re-reads `feature_manifest_*.json`, `tiredeg_sequence_config.json`, `encoding_maps.json`, `tire_compounds_by_race.json` and the kmeans pickles at build time, never hardcoding. This audit adds: manifests must gain an `era` / `regulation_cycle` field and a schema version, so a 2026 manifest cannot be silently consumed by an agent expecting 2022-2025 features (the DRS-feature removal makes the vectors incompatible by construction).
- **Artifact portability is broken today and blocks any automation:** `data/models/model_registry.json` maps all four cluster keys to one absolute Windows path under `c:\Users\victo\...`, and agent configs embed the same machine-specific absolute paths (`pace_agent_config_v1.json`). The registry must become relative-path, era-versioned, and carry per-artifact metadata: model, era, train window (seasons + GPs), metrics, calibration provenance, manifest hash.
- **Experiment tracking (open decision, FUTURE.md §12.2):** ClearML (self-host all-in-one) vs MLflow + DVC vs W&B. Decision criteria this plan imposes: self-hostable (project runs local-first), artifact store that tolerates GB-scale parquets or delegates to HF, and an API clean enough that the per-GP monitors (§5.2) log to it automatically. The tracker choice does not block Phase 0 or 1; it blocks Phase 4.
- **Data versioning:** the HF dataset (`VforVitorio/f1-strategy-dataset`, `src/f1_strat_manager/data_cache.py:58`, migrating to the `f1stratlab` org per FUTURE.md §11) becomes era-structured: 2026 raw/processed data lands under season-scoped paths exactly as 2023-2025 do today (`data/raw/<year>/...`), plus NEW session-type subpaths for FP/Q/Sprint parquets (today only race parquets exist). `data_cache.ensure_race` patterns extend to the new session files; per-GP laziness is preserved.
- **Automation surface, in order:** `f1-train` CLI (headless: retrain model X on era-Y pool with cluster pooling Z) -> tracker integration -> Studio UI panels (the §6.4 seven-stage pipeline: the compound-mapping editor, the cluster viewer with "train with this cluster", the target/outlier form, the temporal-split viewer, the TCN sequence-config editor are all direct consumers of this audit's artifacts) -> the drift-monitor dashboard (§5.2) -> per-GP incremental jobs with acceptance gates and rollback (§4.5).

---

## 7. Prioritized findings (P0-P3)

| ID | P | Finding | Why / risk if ignored | Size |
|---|---|---|---|---|
| F-01 | **P0** | **No production training code exists.** `src/strategy/training/` is empty; `src/strategy/` contents are stale jupytext exports that predate the production models (`src/strategy/README.md` says so explicitly); all real training lives in untouchable notebooks | Nothing else in this plan is executable: no retrain, no `pitlab`, no incremental learning. This is FUTURE.md's own Fase 0 and it gates Fases 3-4 | **L** |
| F-02 | **P0** | **No multi-session ingestion.** N01 downloads race sessions only (`get_session(..., 'R')`); `src/shared/data_extraction/` is legacy one-offs (Spain-2023 hardcodes, an unrelated YOLO augmenter); no FP/Q/Sprint path anywhere | The entire FP + Qualy + Sprint retraining regime is unimplementable; cold-start bootstrap (the only data source in early 2026) is impossible | **M-L** |
| F-03 | **P0** | **No calibration verification harness** (thesis limitation #7 admits it): Platt calibrators frozen on val-2024, pit P05-P95 coverage already 0.7047 vs 0.90 nominal on 2025, MC sigmas frozen | The orchestrator's Monte Carlo layer (score = alpha*E + (1-alpha)*P10 over the sub-agents' distributions) silently ingests rotten uncertainty; 2026 makes this catastrophic, but it is measurably wrong TODAY | **M** |
| F-04 | **P0** | **No drift monitoring / season eval harness exists** (no per-GP metric logging in any surface) | The "measure drift to decide whether to keep old data" decision (FUTURE.md §12.6) is undecidable; degradation will be discovered anecdotally, mid-demo | **M** |
| F-05 | **P1** | **DRS-coupled and fuel-coupled feature semantics break:** `drs_window`, `drs_ready_gap`, `speed_trap_delta` (N12); `FuelLoad`/`FuelEffect`/`fuel_effect_per_lap: 0.055` (N06, N07-N10, `tire_agent_config_v1.json`); fuel-correction model behind the tire-deg target | Feature vectors keep computing without error while meaning something else: the worst kind of break (silent). Needs re-engineering + manifest versioning so era mismatch is a hard error | **M** |
| F-06 | **P1** | **Era-coupled artifact web is hardcoded and 2025-pinned:** `encoding_maps.json` (no Audi/Cadillac, Kick Sauber/Racing Bulls collide at 0), N15 team LabelEncoder (raises on unseen), `tire_compounds_by_race.json` (no 2026 block), `_STINT_CAPACITY_LAPS` (`pit_strategy_agent.py:61`), `circuit_traversal_lookup` (no Madrid), agent configs pinned to `*_2025.parquet` cluster/reference artifacts, `gp_slugs` calendar | Round 1 of 2026 produces hard crashes (LabelEncoder, empty Year-filtered reference laps in `pace_agent.py:433`) and quiet nonsense (wrong compound maps). An era-refresh checklist turns this from archaeology into a procedure | **M** |
| F-07 | **P1** | **No cluster-transition protocol:** clusters are race-derived (N03), cannot be re-fitted before mid-2026; Madrid unassigned; all cluster-keyed thresholds in `strategy_orchestrator_config_v1.json` assume the 2022-2025 structure | Interim races run on undefined behavior for a transversal feature consumed by 3 models + the orchestrator routing | **M** |
| F-08 | **P1** | **Machine-absolute paths in `model_registry.json` and agent configs** (`c:\Users\victo\...`) | Blocks any automated retrain/redeploy loop and any non-Víctor machine; trivial to fix, foundational for `pitlab` | **S** |
| F-09 | **P2** | **Adaptation techniques parked, unimplemented:** RevIN / year-embedding / ShifTS-TAFAS exist only as N09 Step 9 citations; `Year` numeric feature cannot extrapolate in GBDTs | Without them, 2026 forces the discard-all-history option, the most data-poor path; with them, 2023-2025 stays usable as prior | **M-L** |
| F-10 | **P2** | **RAG regulation refresh is year-capped:** `download_fia_pdfs.py:68` supports 2023-2025 only; index built from those PDFs | The regulation agent will cite the wrong season's rules with full confidence; the paper already promises this exact refresh | **S** |
| F-11 | **P2** | **Per-model retraining playbooks do not exist** (which sessions/labels each model can legally learn from, per §4.1-4.2) | Without them, the FP/Q/Sprint regime gets applied uniformly, which is invalid for the classifier family (labels only in races/Sprints) | **M** |
| F-12 | **P2** | **NLP 2026 vocabulary/entities gap:** NER (F1 0.4151, 399 examples) has never seen Audi/Cadillac/Madring/override-mode vocabulary; RCM resolver may meet new message types | Radio and RCM signal quality degrades exactly when strategy uncertainty is highest; folds naturally into `radiogate` + thesis future-work #2 and #5 | **S-M** |
| F-13 | **P3** | **Weekend data-volume study unquantified beyond this audit's estimates** (§4.2); Sprint-calendar dependence (6 of 24 weekends) unmodeled | Retrain scheduling will be guesswork; low risk because estimates here bound it | **S** |
| F-14 | **P3** | **Arcade DRS rendering assumes DRS exists** (DRS-zone geometry, green-zone painting) for 2026 replays | Cosmetic; 2022-2025 replays unaffected | **S** |
| F-15 | **P3** | **Era labeling of published claims:** README/docs/thesis metrics are 2022-2025-era; once 2026 models ship, unlabeled metrics become misleading | Reputational, not functional; cheap to prevent with an era tag convention in docs and model cards | **S** |

---

## 8. Phased, chunkable plan (each phase = one epic / GitHub sub-issue set)

Ordering rationale: measurement before adaptation (you cannot manage drift you cannot see), data before models, artifacts before retraining. Phases 0-1 are era-independent hardening: they pay off even now, and they are the only phases worth considering before 2026 actually approaches.

**Phase 0 - Training extraction & reproducibility (L)**
Scope: create the engineering home for everything else; no behavior change to any surface.
- F-01: extract training/transform code from N01-N16 into `src/strategy/training/` as pure functions (duplicate-and-extract; notebooks stay frozen), with the explicit per-model export contract of §6.
- F-08: relative-path, era-versioned model registry + agent-config path cleanup (config-side, additive; agent internals untouched).
- Deliverable: `f1-train <model>` reproduces each of the 7 production artifacts bit-compatibly (or with documented deltas) from the HF dataset.

**Phase 1 - Measurement layer: eval harness, calibration, drift monitors (M)**
Scope: make model health observable; must exist BEFORE 2026 round 1.
- F-03: calibration verification harness (reliability + Brier for the 3 classifiers, quantile coverage for N15, MC-sigma ratio for the TCN) run against 2025 as the baseline; fix the already-broken pit coverage.
- F-04: per-GP season eval harness + the §5.2 monitor set + §5.3 trigger table, logging to the tracker (or plain parquet until the tracker is chosen).
- F-13: formalize the weekend data-volume study from real 2025 weekends (counts per session type per model).
- Deliverable: a "model health" report per GP, retro-computable over 2025.

**Phase 2 - Multi-session data pipeline (M-L)**
Scope: implement the data side of the FP + Qualy + Sprint regime.
- F-02: session-type-aware ingestion in `src/strategy/training/` (or a new `src/shared/ingestion/` superseding the legacy extractor scripts) for FP1-FP3/Q/SQ/Sprint/R + pre-season testing; era-structured HF dataset layout + `data_cache` patterns for the new session files.
- F-11: per-model retraining playbooks (label availability matrix of §4.1, encoded as config, not prose).
- Prerequisite work item surfaced by this audit: an FP fuel-load estimator (long-run slope-based), without which FP stints cannot feed the fuel-adjusted tire-deg target.
- Deliverable: a 2025 weekend fully ingested across all session types as the dress rehearsal.

**Phase 3 - Era-aware artifacts & feature re-engineering (M)**
Scope: everything that must be true on the first day of 2026, independent of retraining.
- F-06: the era-refresh checklist executed: 2026 blocks in `tire_compounds_by_race.json`, team encoders with Audi/Cadillac (fix the Sauber/Racing Bulls ID collision while at it), Madrid in traversal/slug/calendar artifacts, stint-capacity constants sourced from config instead of code where feasible (config-side or duplicated entry points; `src/agents/` internals stay untouched).
- F-05: DRS-free feature set for N12 (+ MOM eligibility successor feature), fuel-semantics revision, manifests v2 with `era` field and hard era-mismatch errors.
- F-07: the interim cluster protocol (2025-as-prior flags, Madrid provisional assignment, mid-season re-fit procedure, threshold re-tune checklist for `strategy_orchestrator_config_v1.json`).
- F-10: FIA 2026 PDFs + RAG index rebuild. F-12: NLP vocab/entity refresh plan (coordinate with `radiogate`). F-14/F-15: Arcade DRS cosmetics + era labeling of docs.
- Deliverable: the system boots and runs on a 2026 race input without hard errors, with stale-prior flags visible.

**Phase 4 - Adaptation & incremental retraining via `pitlab` (L)**
Scope: the actual 2026 model refresh, and the permanent-regime machinery.
- F-09: RevIN into the TCN (train-code side, Phase 0 home), year-embedding / era-categorical handling across the family, ShifTS/TAFAS only after a 2023-to-2024 simulated-drift ablation gates it.
- Cold-start execution per §4.3 (recalibrate-first, retrain-on-volume), cluster-pooled retrains per §4.4, mid-season re-clustering, per-GP incremental fine-tunes with acceptance gates and rollback per §4.5.
- `pitlab` integration per §6: tracker decision, monitor dashboard, `f1-train` -> UI.
- Deliverable: all seven models re-validated on accumulating 2026 holdouts, with the §5.3 triggers live; this is FUTURE.md's Fase 4 gate that unlocks the Live bot (Fase 5).

Dependency notes: Phase 1 depends only on Phase 0's registry hygiene (can even start in parallel); Phase 2 depends on Phase 0 (the ingestion code needs its home); Phase 3 is mostly independent and can be executed close to the season start; Phase 4 depends on all of 0-3. The thesis's own sequencing (re-cluster with first-half-2026 data, relabel compounds, regenerate features, monitor degradation, `06_conclusiones.tex` future-work #3) maps onto Phases 3-4 and is preserved, not contradicted.

---

## 9. Open questions & risks (need Víctor's decision or genuinely uncertain)

1. **How much usable 2026 data will exist, and when?** The §4.2 volumes are estimates from 2023-2025 shapes. Sprint count (6), calendar (Madrid in, Imola out), and testing formats must be re-verified against the final 2026 calendar at execution time.
2. **Is cross-era transfer valid at all?** Unknowable before measurement. The plan deliberately avoids betting on either answer: Phase 1 monitors produce the per-model keep-vs-discard decision (FUTURE.md §12.6) empirically by round 3-4.
3. **Old-data weighting policy per model:** if kept as a weak prior, how weighted (sample weights, year-embedding, base-model + fine-tune)? Proposed default: year-embedding for the TCN, era-categorical + refit for GBDTs; Víctor should ratify per model.
4. **Untouchable-boundary rulings:** several fixes are cleanest as tiny config-side or additive changes near `src/agents/` (stint capacities, encoder maps, manifest era checks). Same decision gate as the P2 audit's F-01: sanctioned minimal exceptions with regression runs, or strictly duplicate-and-improve. Víctor decides case by case.
5. **Tracker choice for `pitlab`** (ClearML vs MLflow+DVC vs W&B, FUTURE.md §12.2): blocks Phase 4 only. Criteria in §6.
6. **K=4 stability:** does the archetype count survive the regulation reset? Re-run the K-selection (silhouette was only 0.201 in-era), do not assume K=4.
7. **Where the FP fuel-load estimator lives** (it is a new model-ish component): inside `src/strategy/training/` as a data-prep step, or a first-class predictor with its own validation. Proposed: data-prep step first, promote if it grows.
8. **Whether 2026 sporting rules change strategy constraints themselves** (two-compound rule, mandatory stops under new tire specs, Manual Override Mode usage rules): the guardrails encode the 2022-2025 sporting regime; a sporting-regs diff pass belongs to Phase 3 and should cite the 2026 FIA documents ingested in F-10.
9. **Timing of the whole program:** this audit intentionally does not claim a start date. The only hard external deadline is the 2026 season itself if the system is meant to run on it live; if 2026 is only ever replayed retrospectively, Phases 2-4 compress and de-risk substantially (labels all exist by then). Víctor should decide which of the two ambitions (live-season vs retrospective-2026) the plan is executed against, because it changes the urgency of every phase.

---

## Verification protocol (when this plan is executed)

- Phase 0: bit-level (or documented-delta) artifact reproduction of all 7 models from `f1-train`; CLI regression per the established protocol (`python scripts/run_simulation_cli.py Sakhir HAM Mercedes --no-llm --laps 1-10`, output diffed).
- Phase 1: monitors retro-validated on 2025 (they must "detect" the already-known drift: TyreAge sign flip, pit-coverage 0.70).
- Phase 2: a full 2025 weekend ingested across all session types, row counts within 5 percent of FastF1 ground truth.
- Phase 3: a synthetic 2026 race input (2025 race relabeled with a 2026 calendar/team/compound overlay) runs end-to-end with zero hard errors and visible stale-prior flags.
- Phase 4: each adapted model beats its own cross-era baseline (old model + recalibration) on the accumulating 2026 holdout before promotion; no promotion without the acceptance gate.
