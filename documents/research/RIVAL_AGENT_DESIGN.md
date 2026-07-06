# Rival Agent: Design and Research Methodology (TFM forward design)

**Status: research design, forward plan. Plan only, no code, no commitments.**

This document is the methodology design for the **Rival Agent**, the chosen TFM
(Trabajo Fin de Master) for the MUIIA master (Master Universitario en Investigacion en
Inteligencia Artificial, AEPIA-UIMP, Especialidad 1). It is a design-ahead plan: the
master has not started yet, no coursework is mapped to concrete deliverables, and nothing
here is implemented. The goal is to leave the whole problem thought through so that when
the master begins, the work starts from a validated design instead of a blank page.

The formal proposal this document extends lives outside the repo at
`C:\Users\victo\Desktop\Documents\Master\propuesta_master.md`. This design is consistent
with it and deepens every section: the ground-truth reconstruction, the observability
treatment, the model design, the integration path, and the evaluation protocol.

Hard constraints honored throughout: design only, no code; `scripts/run_simulation_cli.py`,
`src/agents/` internals, and `notebooks/**` are untouchable (the Rival Agent is strictly
additive); LLM provider is OpenAI or LM Studio, never Anthropic.

---

## 1. Framing

### 1.1 What the Rival Agent is

F1 StratLab today is **reactive**: the orchestrator (N31) recommends strategy by looking
at our car's state (tyre cliff, pace, pit windows, radio, regulations) while treating
rivals as static scenery, a sorted list of positions and gaps. A real pit wall decides by
**anticipating**: "if we stay out one more lap, the car behind will undercut us"; "Ferrari
always reacts to a Mercedes stop within two laps"; "he is on a 24-lap-old medium, his
window opens now".

The Rival Agent is a new agent in the F1 StratLab architecture that **predicts the next
strategic move of each rival in the driver's environment** from the rival's public
timing-screen data:

- **Pit window**: probability the rival pits within the next 1 / 3 / 5 laps, and the
  distribution of the likely pit lap.
- **Compound**: the probability distribution over the compound the rival will fit next.
- **Undercut / overcut attempt**: probability the rival's stop is an attack on a specific
  car (including ours), and probability an overcut is being attempted against us.

That prediction is injected as **anticipatory context** into the orchestrator, turning
the system from reactive to anticipatory. The agent is a new node in the multi-agent
graph. It does not touch the six existing sub-agents (N25 pace, N26 tire, N27 race
situation, N28 pit strategy, N29 radio, N30 RAG) or the existing orchestrator code.

### 1.2 Research question

> Does anticipating the strategic behavior of rivals improve the quality of strategy
> recommendations compared to the current reactive system?

Operationalized as: on the Grands Prix already validated in the TFG (Hungary, Qatar,
Australia, and the documented divergence cases), does the with-Rival-Agent system agree
more often with (a) the real race outcome and (b) the real pit wall's decisions than the
without-Rival-Agent baseline, at matched decision points?

### 1.3 What already exists (verified against current code, 2026-07-06)

Every claim below was checked against the repository as it stands; file anchors are given
so the eventual implementation can re-verify.

**The single-driver boundary.** `src/simulation/race_state_manager.py` enforces the
architectural constraint: our driver gets full telemetry; every rival gets a
timing-screen-only view. `RaceStateManager.get_rival_states(lap)` emits, per rival and
per lap: `driver`, `team`, `position`, `lap_time_s`, `compound`, `tyre_life`, `stint`,
`speed_st`, `gap_to_leader_s`, `interval_to_driver_s`, `is_pitting`. This is exactly the
information a strategy engineer sees on the live timing monitor, and it is the **entire
observable input space of the Rival Agent**. The boundary is a feature of the research,
not a limitation to work around: the agent must predict rival behavior from what a real
pit wall could see.

**The lap_state contract.** `RaceStateManager.get_lap_state(lap)` returns
`{lap_number, driver, rivals, weather, session_meta}`. All seven existing agents consume
this dict. The contract tolerates additive keys (confirmed by the P5 data-engineering
audit, `documents/audits/AUDIT_P5_DATA_ENGINEERING.md`, finding F-15), which is the
channel through which richer rival gap data can flow without breaking anything.

**The "2 drivers" mode.** The existing Head-to-Head mode (`scripts/f1_cli.py` option 2,
implemented in `scripts/cli/runner.py:run_h2h`) runs the full simulation for Driver 1 and
tracks Driver 2 as a rival via the `--rival CODE` flag of `run_simulation_cli.py`. The
tracked rival's per-lap state (position, compound, tyre age, interval) is pulled from
`lap_state["rivals"]` and rendered alongside our driver. Two implications for the TFM:

1. The plumbing to follow a designated rival lap by lap already exists and is exercised
   in a shipped surface. The Rival Agent generalizes "track one rival's state" to
   "predict N rivals' next moves".
2. The mode is display-level today: the rival's data reaches the screen, not the
   decision. The Rival Agent is precisely the missing step from observation to
   anticipation.

**The orchestrator N31.** `src/agents/strategy_orchestrator.py` implements three layers:
Layer 1 MoE routing (deterministic rules pick conditional agents N28/N30), Layer 2 Monte
Carlo (500 draws from the sub-agents' distributions scoring four candidates STAY_OUT /
PIT_NOW / UNDERCUT / OVERCUT with `score = alpha * E + (1 - alpha) * P10`), Layer 3 LLM
synthesis into a frozen 14-field `StrategyRecommendation`. Three facts matter for the
integration design:

- The MC layer already consumes exactly the kind of distributions the Rival Agent will
  emit: Triangular P10/P50/P90 draws (tyre cliff, pit duration) and Bernoulli draws
  (SC within window, undercut success). The Rival Agent's output format is designed to
  slot into this sampling scheme (section 6.5).
- The undercut interaction is currently modeled with fixed constants:
  `POS_GAP_S = 1.50` seconds per position and a single N16 success probability for OUR
  undercut. The rival's possible counter-move (pitting first, covering our stop) does not
  exist in the simulation. That is the concrete hole the agent fills.
- The sub-agents are LangGraph ReAct agents (built via `create_agent` with tools wrapping
  each ML model); the orchestrator itself is a plain Python pipeline that calls their
  public entry points. "A new LangGraph node" therefore means: a new agent module in the
  same ReAct style as N25-N29, invoked by an additive orchestration entry point, plus a
  formalization of the whole graph for the multi-agent portion of the TFM (section 7).

**The data already on disk.** Verified against `data/raw/2025/Budapest/` (and the P5
audit, which verified this repo-wide):

- `laps.parquet` (1,368 rows for Budapest): full FastF1 lap table for ALL drivers,
  including `PitInTime`, `PitOutTime`, `Stint`, `Compound`, `TyreLife`, `FreshTyre`,
  `Position`, `TrackStatus`, and the quality flags `IsAccurate`, `Deleted`,
  `DeletedReason`, `FastF1Generated`.
- `intervals.parquet` (27,143 rows for Budapest, roughly 4-second resolution): OpenF1
  per-driver gap evolution with `interval_seconds`, `gap_to_leader_seconds`,
  `drs_window`, `is_lapped`, `laps_behind`. Downloaded for every race and, per the P5
  audit (finding F-10), **consumed by nothing at runtime today**. This is the highest
  value untapped signal for the Rival Agent: gap closing rates and undercut windows can
  be measured from data instead of assumed.
- `pitstops.parquet` (30 rows for Budapest): the FastF1 pit-lap rows (a filtered view of
  laps with pit events), also currently unused at runtime.
- 24 races each for 2024 and 2025 are already on disk under `data/raw/<year>/<gp>/`,
  plus 2023 for extended training if wanted.

**Models and lessons the design leans on.**

- N16 undercut model (`data/models/pit_prediction/model_config_undercut_v1.json`):
  LightGBM over 13 explicit pair features (`pos_gap`, `Lap_gap`, `tyre_life_diff`,
  `TyreLife_X/Y`, compound ids, `pit_delta_X`, `lap_race_pct`, `pos_X_before`,
  `circuit_undercut_rate`, `team_x_undercut_rate`), trained 2023-2024, tested 2025,
  AUC-PR 0.6739 (1.95x baseline), Platt-calibrated. Its dataset construction rules are
  the template for the undercut ground-truth labels (section 3.4).
- N15 pit duration model (`data/models/pit_prediction/model_config.json`): quantile
  HistGBT plus a per-circuit `circuit_traversal_lookup` (pit lane traversal seconds per
  GP). The traversal lookup is exactly the quantity needed to compute, from public data,
  whether a rival's stop would drop them into traffic (section 5).
- The N12 vs N12B lesson (project memory, confirmed in the notebooks' results): on this
  data regime (tens of thousands of rows, sparse positives), **explicit feature
  engineering beats raw sequence modeling**. N12's LightGBM on engineered pair features
  reached AUC-PR 0.5491; N12B's causal TCN on raw sequences reached roughly 0.10. This
  lesson directly shapes the model choice (section 6.2).
- N13/N14 SC model precedent: when the exact event is unpredictable, reframe to a
  within-window target (`sc_within_3_laps`) and treat the output as a calibrated soft
  prior. Rival pit prediction adopts the same posture.

### 1.4 What this TFM adds over the TFG (novelty statement)

The TFG built a reactive multi-agent recommender validated against real races. The TFM
adds, in increasing order of research weight:

1. **A reconstructed behavioral dataset** of every rival strategic move in 2024-2025
   (pit laps, stint compounds, undercut/overcut attempts and outcomes), built from
   FastF1/OpenF1 with documented validation. No such labeled dataset exists publicly.
2. **A partially observable prediction problem** formulated honestly: the agent sees
   only timing-screen data, hidden state (true degradation, team intent) is modeled as
   uncertainty, and the cost of partial observability is quantified with an oracle
   ablation (section 8.4).
3. **Opponent modeling inside a multi-agent decision system**: the predictions are not
   an offline exercise; they feed the Monte Carlo layer and the LLM synthesis of a
   working strategy system, and the effect is measured end to end by ablation on the
   same races the TFG validated.

Related work anchors to cite when writing the TFM: opponent modeling in games and
autonomous driving (trajectory/intent prediction), discrete-time survival analysis for
event timing, Heilmeier et al. (2020) for motorsport MC simulation (already cited by the
orchestrator), and the TFG's own IEEE paper in preparation (`feat/paper` branch) as the
baseline system reference.

---

## 2. Problem formulation

### 2.1 Prediction targets

For each rival `j` in scope, at each lap `L`, from information available at end of lap
`L` only, predict:

| Head | Target | Type |
|---|---|---|
| **H1: pit timing** | Will rival `j` pit within the next `k` laps? (k = 1, 3, 5) | Discrete-time hazard / binary per window |
| **H2: compound** | Compound fitted at the next stop (SOFT / MEDIUM / HARD) | Categorical, conditional on a stop happening |
| **H3: undercut attempt** | Is the (next) stop an undercut attack on the car ahead (in particular, on us)? | Binary, conditional |
| **H4: overcut posture** | Is rival `j` extending the stint to overcut a car that just pitted (in particular, us)? | Binary, conditional |

H1 is the core head and the one with the cleanest ground truth. H2 rides on H1. H3/H4
are derived interactions whose labels come from the pair-reconstruction rules in section
3.4; they are noisier by construction and are reported with that caveat.

**Framing choice: discrete-time hazard.** The natural formulation for "when will the
rival pit" is discrete-time survival analysis: each rival-lap is an at-risk observation,
the event is the pit entry, censoring occurs at race end, retirement, or red flag. The
per-lap hazard `h_j(L)` composes into any window probability
`P(pit within k) = 1 - prod_{i=1..k}(1 - h_j(L+i))`, which is exactly the composition
pattern the project already uses for multi-lap overtake probability (N12 inference:
`P = 1 - prod(1 - P_k)`). This gives one model that serves every window instead of one
binary model per k. A simpler fallback (three independent binary classifiers for k = 1,
3, 5) is kept as the ablation baseline because it is trivially trainable and the project
has shipped that shape before (N13/N14).

### 2.2 Scope: which rivals

Two scopes, used at different stages:

- **Training and dataset**: the **full grid**, all 2024-2025 races. The P5 audit's open
  question 7 already leans this way ("full grid for the data pack, let the TFM subset
  it"), and full-grid data is what makes the dataset a reusable, publishable artifact.
- **Runtime (in the sim)**: the **strategic environment** of our driver, proposed as the
  cars within one pit cycle of us: every rival whose `interval_to_driver_s` absolute
  value is below the circuit's total pit loss (physical stop plus
  `circuit_traversal_lookup`, roughly 18-28 s depending on the GP). That is typically 4-6
  cars and is the set whose moves can actually change our optimal decision. A fixed
  "5 nearest" is the simpler fallback; the pit-cycle-radius definition is preferred
  because it is strategy-grounded and self-adjusts per circuit. Final choice is open
  question Q1.

### 2.3 Per-lap cadence and information timing

Predictions are emitted once per lap, aligned with the system's decision cadence (the
orchestrator runs per lap on `lap_state`). The feature vector for lap `L` may use any
public information with timestamp up to the end of lap `L`: the rival's lap times up to
`L`, gaps up to `L`, pit events up to `L`, track status up to `L`. It must never use:

- The rival's own future rows (obvious leakage).
- Same-lap information that is only knowable at lap completion when predicting for that
  lap (the emission point is defined as end-of-lap, so end-of-lap features are legal for
  predicting laps `L+1` onward).
- Any column FastF1 backfills post-hoc in a way a live feed would not have had
  (section 4.3 lists these).

---

## 3. Ground-truth reconstruction

This is the data core of the TFM and, per the proposal, the part that gives it thesis
weight. The problem: nobody publishes "what each rival decided per lap". It has to be
reconstructed from timing artifacts, with explicit operational definitions, validation
against independent sources, and documented failure modes.

### 3.1 Sources and their roles

| Source | Artifact | Role |
|---|---|---|
| FastF1 laps | `data/raw/<year>/<gp>/laps.parquet` | Primary event truth: `PitInTime` / `PitOutTime` (in-lap / out-lap), `Stint`, `Compound`, `TyreLife`, `FreshTyre`, `Position`, `TrackStatus`, quality flags |
| FastF1 pit view | `pitstops.parquet` (same folders) | Redundant pit-lap view for cross-checking stop counts |
| OpenF1 intervals | `intervals.parquet` (same folders) | Continuous (about 4 s) gap evolution: `interval_seconds`, `gap_to_leader_seconds`, `drs_window`, `is_lapped`; needed for undercut-window measurement and closing-rate features |
| OpenF1 `/v1/stints` | Tier 1, not yet ingested | Independent stint/compound reconstruction "as the pit wall would see it"; the P5 audit (F-10 Tier 1) scopes its ingestion on the proven radio-builder infrastructure |
| OpenF1 `/v1/pit` | Tier 1, not yet ingested | Pit lane transit times per stop; validates in-lap detection and enriches pit-loss estimates |
| OpenF1 `/v1/position` | Tier 1, not yet ingested | Intra-lap position changes; densifies undercut outcome verification |
| `data/tire_compounds_by_race.json` | On disk | Pirelli Cx allocation per GP/year; prior for compound-choice modeling and for tracking each rival's remaining allocation |

The ingestion of the three Tier 1 endpoints follows the template the project already
considers production-grade: `src/data_extraction/openf1/radio_dataset_builder.py`
(class-based, retry session, idempotent resume, per-race layout). The P5 audit's Phase 4
items 14-15 already plan exactly this build plus a "Rival readiness pack"; this design
adopts those items as its M0 milestone (section 10) rather than re-planning them.

**Race identity caveat.** Cross-season joins by folder name are unsafe today (P5 finding
F-01: `Miami` vs `Miami_Gardens`, the `2023/Spain` / `2023/Barcelona` duplicate, five
naming schemes). The ground-truth build must resolve races through one identity mapping
(ideally the P5 Phase 0 identity module once it lands; a local table inside the dataset
builder otherwise). This is a stated dependency, not a new design.

### 3.2 Label: pit events (H1)

Operational definition, per rival per race:

- **In-lap**: lap `L` where `PitInTime` is non-null. The pit decision is attributed to
  lap `L` (the driver committed by entering the pit lane during lap `L`).
- **Out-lap**: lap `L+1` where `PitOutTime` is non-null. Used for out-lap pace exclusion
  in features and for validating stint arithmetic.
- **Hazard label**: `y_j(L) = 1` if rival `j`'s next in-lap is `L+1` (for the per-lap
  hazard head); window labels `y_j^k(L) = 1` if any in-lap falls in `(L, L+k]`.

Validation and edge handling:

- **Cross-check** stop counts per driver per race across three views: non-null
  `PitInTime` rows in `laps.parquet`, rows in `pitstops.parquet`, and OpenF1 `/v1/pit`
  entries. Exact match expected; any delta gets a documented resolution (the P5
  verification protocol requires "exact match or documented delta" and this design keeps
  that bar).
- **Stint arithmetic invariant**: `Stint` increments exactly at out-laps; `TyreLife`
  resets to 1 (or 0) at the out-lap and increments by 1 per lap otherwise. Violations
  flag the race for manual review (typical causes: red flag tyre changes, formation-lap
  oddities).
- **Red flags**: tyre changes under red flag are free stops with no pit lane pass;
  `Stint` increments without `PitInTime`. These are labeled as a separate event class
  (`RED_FLAG_CHANGE`) and excluded from the pit-hazard positives (no in-lane decision was
  made), while still resetting tyre-age features.
- **Retirements / DNFs**: rival-lap observations end at the last completed lap;
  censored, not negative, in the survival framing.
- **Drive-through / stop-go penalties**: pit lane passes without tyre change. Detected
  by pit lane pass with no compound/stint change (and, once `/v1/pit` is ingested,
  anomalously short or flagged stops). Labeled `PENALTY_PASS`, excluded from H1
  positives. Expected volume is small (a handful per season) but silently mislabeling
  them as strategic stops would inject exactly the wrong signal.
- **Quality flags**: laps with `Deleted == True` or `IsAccurate == False` keep their
  event labels (a pit is a pit) but their lap-time-derived features are masked (the P5
  audit notes these flags exist everywhere and are consulted nowhere; this dataset is
  the first consumer).

### 3.3 Label: stint compounds (H2)

Per stint per rival: the compound of the stint that **starts** at each pit stop, read
from the `Compound` column of the out-lap (FastF1's per-lap compound is already
stint-consistent). Mapped to the race-specific Pirelli allocation (C1-C5) through
`data/tire_compounds_by_race.json` so the model can learn in both spaces: the
race-relative space (SOFT/MEDIUM/HARD) for decision semantics, the absolute space (Cx)
for cross-race transfer.

Edge handling:

- **Wet compounds**: INTERMEDIATE / WET stints fall outside the system's dry-only
  compound enum (`_COMPOUND_VALUES` in `strategy_orchestrator.py` is
  SOFT/MEDIUM/HARD, and N16 filters to `dry_compounds`). Proposal: races with any
  wet-affected stint window are kept in the dataset with a `wet_affected` flag; the v1
  compound head trains on dry stints only, consistent with every existing model in the
  project. Whether wet races are excluded from H1 too is open question Q5 (leaning: keep
  them for H1, the pit-timing signal under rain is real and valuable, but report metrics
  split by dry/wet).
- **Independent verification**: once OpenF1 `/v1/stints` is ingested, compare its
  compound-per-stint reconstruction against FastF1's. Divergences (both sources derive
  from FIA/broadcast data but through different pipelines) get logged; agreement rate is
  itself a useful data-quality statistic for the TFM's data chapter, and doubles as an
  estimate of how noisy "observed compound" is in the live setting (section 4).

### 3.4 Label: undercut and overcut attempts and outcomes (H3/H4)

These are **pair events** and need operational definitions. The design reuses N16's
construction (its features and target already encode the project's definition of an
undercut situation) and makes the attempt/success rules explicit:

- **Undercut attempt by X on Y**: X pits on lap `L`; Y is the car directly ahead of X
  (or within `n_pos <= 2` positions) at end of lap `L-1`; the gap
  `gap(X -> Y)` at end of `L-1` is below the **undercut window** for that circuit; Y
  does not pit on lap `L` (if both pit the same lap it is a covered stop, labeled
  `COVER`, itself an interesting reactive behavior worth a label). The undercut window
  per circuit is measured from data, not fixed: total pit loss (N15 physical stop P50
  plus `circuit_traversal_lookup`) minus the typical fresh-tyre delta over the out-lap
  plus first flying lap, computed empirically from the reconstructed dataset. This
  replaces the simulator's global 1.5 s / fixed-window assumption with a data-derived,
  per-circuit quantity (the P5 audit explicitly names this as an enabled improvement).
- **Undercut success**: at the first lap `L*` where both X and Y have completed their
  stops (Y's next stop after `L`), X is ahead of Y on track (Position comparison, with
  `is_lapped` sanity checks from intervals). If Y stays out so long the comparison
  becomes strategy-divergent (Y switched to a different stop count), the pair is labeled
  `DIVERGED` and excluded from success/failure counts, mirroring how N16's
  `undercut_clean.parquet` filters ambiguous pairs.
- **Overcut attempt by Y on X**: X pits on lap `L` from within Y's undercut window; Y
  stays out at least 2 more laps and pits by `L + 6`; attempt succeeds if Y emerges
  ahead of X at `L*` as above. The 2-and-6 lap bounds are initial values to be
  sensitivity-checked in M1 (open question Q2).

**Honesty note carried into the TFM text**: attempt labels are *behavioral
reconstructions*, not declarations of intent. A stop that looks like an undercut may
have been a cliff-forced stop that happened to land in the window. The dataset therefore
stores the *situation* (in-window stop with the geometry of an undercut) plus the
*outcome*, and the TFM must phrase H3/H4 as "stop consistent with an undercut attack",
not as mind-reading. Where team radio is available (the project's radio corpus covers
our-driver-adjacent messages; the future `radiogate` corpus would cover more), radio
calls like "box to overtake" can upgrade a subset of labels to intent-confirmed; that is
an enrichment, not a dependency.

### 3.5 Dataset shape, size, and imbalance (estimates to verify in M0)

Per season on disk: 24 races, about 20 cars, 50-70 laps: roughly **28k-32k rival-lap
rows per season**, about 60k for 2024-2025 combined (plus about 30k more if 2023 is
added for training). Pit events: about 30-45 per race (Budapest 2025 has exactly 30),
so roughly **800-1,000 stops per season**, about 1,700-2,000 positives for H1 across
both seasons. Per-lap pit rate about 2.5-3%; "pit within 3 laps" positive rate about
8-10%. Undercut-attempt pairs: N16's undercut dataset and N12's 28,494 overtake pairs
suggest the pair space is in the low tens of thousands, with attempt positives in the
high hundreds.

Three design consequences:

1. **Severe imbalance** on every head: AUC-PR (not ROC) as the primary metric, class
   weighting or `scale_pos_weight` (N16 precedent: 1.63), and calibrated probabilities
   as the deliverable rather than hard labels.
2. **Small effective sample for sequence learning**: about 2,000 positive events is
   firmly inside the regime where the N12B lesson applies (explicit features beat raw
   sequences below about 20k rows of signal). This drives the model plan (section 6).
3. **SMOTE stays banned**: the project already established that synthetic oversampling
   leaks across temporal structure (N11/N12 decision); the same applies here.

### 3.6 Splits and leakage discipline

- **Temporal split, project precedent**: train on 2023-2024, test on 2025 (exactly how
  N15/N16 split). Within training, validation is the last N races of 2024, never a
  random row split: rows within a race are strongly dependent, so **all splits are by
  race**, never by row.
- **No same-race leakage**: circuit-level aggregate features (circuit undercut rate,
  median pit windows) are computed on training years only and joined as priors; they
  are never recomputed on test races.
- **No target leakage through stint features**: `TyreLife` at lap `L` is legal (it is
  derivable from observed pit events); "laps until stint end" obviously is not.
- **Regulation era**: 2023-2025 share the regulation cycle the TFG trained under; the
  2026 rules break (documented in `AUDIT_2026_REG_CONCEPT_DRIFT.md`) means the TFM
  should state clearly that models and conclusions are era-scoped to 2022-2025 style
  racing, with 2026 transfer as declared future work, not silently assumed.

### 3.7 Deliverable form

M0's output is the **Rival readiness pack** exactly as the P5 audit scopes it (Phase 4,
item 15): a documented, schema-versioned, Hub-published dataset with a dataset card
covering: the operational label definitions above, the validation results (source
cross-check tables), per-race quality flags, and the leakage rules. Publishing it makes
the TFM's data chapter reproducible and gives the ecosystem (HF org `f1stratlab`) a
citable artifact independent of the modeling results.

---

## 4. Observability limits: modeling hidden information as uncertainty

The proposal's stated risk: "the rival's true compound and degradation are hidden
information; they are modeled as uncertainty". This section makes that precise, because
"hidden" has three different grades here and conflating them would either overclaim
(pretending to see what we cannot) or underclaim (discarding data a real wall has).

### 4.1 The observability ladder

| Grade | Signals | Treatment |
|---|---|---|
| **Directly observable** (timing screen) | Position, lap times, gaps/intervals, pit entry/exit events, track status, DRS window occupancy | Used as-is; these are the columns `get_rival_states` already exposes plus `intervals.parquet` |
| **Derived-observable** (public, reconstructable live) | Tyre age (count laps since the rival's observed out-lap), stint number, current compound (broadcast tyre detection), remaining tyre allocation (race allocation minus observed used sets), pit loss for this circuit | Used, but computed **from observed events**, never read from privileged columns; current compound carries a noise model |
| **Latent** (genuinely hidden) | True degradation state (wear, cliff proximity), fuel-corrected pace potential, team strategy intent, driver instructions | Never used as features; inferred only through their observable footprint (pace deltas, stint length vs compound norms); uncertainty carried in the output distributions |

Two honest clarifications the TFM text must make:

- **Current compound is observable in the real world** (FIA tyre detection feeds the
  timing screen, and the project's own boundary already gives rivals' `compound` and
  `tyre_life` to the timing-screen view in `race_state_manager.py`). What is hidden is
  the **next** compound (a prediction target, H2), the **true degradation** of the
  current set, and whether the set was new or scuffed at fitting (`FreshTyre` exists in
  FastF1 data; live, it is inferable from allocation tracking but noisily). The design
  therefore does not pretend compound is secret; it treats it as *noisily observable*
  (live detection can lag a lap or misread) and quantifies the noise using the
  FastF1-vs-OpenF1 stint agreement rate from section 3.3.
- **`TyreLife` as a column is a replay artifact.** At inference inside the sim it is
  legal (the replay's rival state carries it, and it equals what a wall would count),
  but the agent's feature builder must compute tyre age from observed pit events, so
  the same code is correct when the input is a live feed where the column does not
  exist. This "derive, do not read" rule is the single most important implementation
  discipline for observability honesty.

### 4.2 Uncertainty representation

Three mechanisms, in increasing order of sophistication; v1 commits to the first two:

1. **Distributional outputs everywhere.** No head emits a hard label: H1 emits hazards
   and window probabilities, H2 a categorical distribution, H3/H4 calibrated
   probabilities. Calibration (section 6.4) is what makes "uncertainty" a real claim
   rather than a softmax aesthetic.
2. **Belief features with explicit staleness.** Derived-observable features carry
   companion reliability signals: laps since compound was last confirmed, whether tyre
   age is exact (pit observed) or bounded (car started on unknown-age set after red
   flag), a `wet_affected` flag. The model learns to widen its own uncertainty where
   the belief is stale, and the oracle ablation (section 8.4) verifies it does.
3. **Explicit latent-state model (stretch, optional).** A hidden Markov / state-space
   view where each rival's "strategic mode" (managing, pushing, in-window, reacting) is
   a latent variable updated per lap from observations. Scientifically attractive
   (connects to the Temporal Data course, gives interpretable posteriors), but it is a
   second modeling program on top of the supervised one. Kept as a stretch goal for the
   TFM's research-depth section, not on the critical path. Decision point at M3.

### 4.3 Replay-vs-live honesty audit

FastF1 post-processes data (retro-corrections, generated laps flagged by
`FastF1Generated`). The feature builder must be audited column by column against the
question "would the live timing feed have had this value at this timestamp?". Known
traps, to be documented in the dataset card:

- Post-hoc corrected lap times on deleted laps (mask via quality flags).
- `Position` during pit cycles is end-of-lap position; mid-lap crossovers are only in
  `/v1/position` (Tier 1).
- Weather join is by fractional lap index (P5 finding F-14), fine for replay, wrong
  live; the Rival Agent uses weather only as slow context, so this inherits the
  documented caveat rather than fixing it.

---

## 5. Feature engineering

All features are computable from the observability ladder's top two grades, per rival
`j` at end of lap `L`. Families, with source anchors:

**F1: Tyre state (derived-observable).**
- `est_tyre_age`: laps since observed out-lap (or since start).
- `compound_obs`: current compound one-hot, plus `compound_confirmed_laps` staleness.
- `age_vs_compound_norm`: est_tyre_age minus the historical median stint length for
  this compound at this circuit (computed from the reconstructed dataset; the Pirelli
  capacity constants in the pit agent are the fallback prior).
- `fresh_at_fit`: known / new / used (allocation tracking; degraded gracefully to
  unknown).
- `sets_remaining_est`: per-compound estimate from `tire_compounds_by_race.json`
  allocation minus observed usage.

**F2: Gap geometry (observable, the intervals.parquet payoff).**
- `gap_ahead_s`, `gap_behind_s` (to adjacent cars), `interval_to_our_driver_s`.
- `gap_ahead_trend_3`, `gap_ahead_trend_5`: closing rates over 3/5 laps from the
  intervals time series (this is where the 4-second resolution beats lap-boundary
  snapshots).
- `in_undercut_window_of_ahead`: boolean plus margin, using the per-circuit
  data-derived window (section 3.4).
- `pit_exit_traffic_density`: cars within +/- 3 s of the rival's projected pit exit
  (rival's `gap_to_leader_seconds` plus circuit pit loss, scanned against the field's
  gaps): the "does he have a free pit window" feature a wall computes constantly.
- `drs_train_flag`: rival stuck in a DRS train (`drs_window` from intervals sustained
  over multiple laps), a known pit-early trigger.

**F3: Pace signals (observable).**
- `pace_delta_own_5`: rival's last-lap time minus their own clean-lap median this
  stint (out-laps, in-laps, SC laps, deleted laps masked).
- `pace_slope_stint`: linear trend of lap times within the stint (public degradation
  footprint; this is the latent wear's shadow, and the honest replacement for reading
  a degradation model's internal state).
- `pace_vs_our_driver`, `pace_vs_direct_rivals`: relative deltas over recent laps.

**F4: Stint and race context.**
- `stint_number`, `lap_race_pct` (N16 precedent), `laps_remaining`.
- `stops_so_far` vs the circuit's modal stop count (1-stop vs 2-stop race shape).
- `track_status` / SC or VSC active (pit-under-SC is the single strongest pit trigger;
  `SC_PIT_BONUS = 8.0` in the MC already encodes why), and the system's own
  `sc_prob_3lap` from N27 as a soft prior (it is computed from public data, so feeding
  it to the Rival Agent respects the boundary).
- Circuit descriptors: pit loss (N15 traversal lookup), circuit cluster (the K=4
  clustering in `data/models/k_means_circuit_clustering/`), historical undercut rate
  (`circuit_undercut_rate`, already an N16 feature).

**F5: Historical tendencies (train-years priors, joined by team/circuit).**
- `team_pit_reactivity`: historical P(team covers a rival stop within 2 laps).
- `team_undercut_rate` (N16 already has `team_x_undercut_rate`).
- `team_median_first_stop_pct`: when this team typically takes its first stop, as race
  percentage, per circuit cluster.
- Driver-level versions where sample size permits (drivers move teams; team-level is
  the stable unit).

Feature count lands around 35-50 explicit features: the regime where GBDTs shine and
where every feature remains explainable to a tribunal. Each feature is tagged in the
dataset card with its observability grade and source artifact, which turns the
"observability limits" section of the TFM from prose into a table.

---

## 6. The prediction model

### 6.1 Architecture: multi-head, per-rival, shared trunk of features

One feature pipeline (section 5) feeding four heads (section 2.1). Rivals are exchangeable
(no per-driver model): identity enters only through team/driver priors (F5), which keeps
the model applicable to any grid and avoids 20 tiny per-driver datasets.

### 6.2 Model family options and the project's own evidence

| Option | For | Against | Verdict |
|---|---|---|---|
| **LightGBM per head** (hazard head as GBDT on rival-lap rows with lap-index features; N16-style pair GBDT for H3/H4) | Directly matches the project's strongest results (N12, N16); handles 35-50 heterogeneous features and missingness natively; millisecond inference for 20 rivals; interpretable (SHAP for the TFM's analysis chapter) | Sequence context must be hand-encoded (trends, slopes); no shared representation across heads | **Primary model. Build first, calibrate, freeze as the reference.** |
| **Discrete-time survival wrapper over the GBDT** (each rival-lap expanded with time-at-risk; single model outputs h(L)) | Statistically correct timing model; censoring handled; one model for all windows; strong fit for the Temporal Data course narrative | Slightly more involved dataset expansion; care with calibration across the hazard curve | **Adopted as the H1 formulation** (it is a framing of the GBDT, not a different model family) |
| **Sequence model** (GRU/TCN over the rival's last 10-15 laps of raw-ish channels, per-lap embedding into the heads) | The honest test of whether learned temporal representation beats engineered trends; natural Deep Learning course fit; could capture pattern shapes (cliff onset curvature) features miss | The N12B lesson: on about 2k positives a causal TCN collapsed (AUC-PR about 0.10 vs 0.55 for features); risk of a repeat is high and known | **Challenger only, evaluated against the frozen GBDT under identical splits.** A negative result is publishable inside the TFM as a replication of the N12B finding on a new task; a positive result is a genuine contribution. Either way it is informative, but the system integrates whichever wins. |
| Transformer / large sequence models | None at this data scale | Everything | Rejected; note the rejection and reason in the TFM |

This "GBDT first, sequence challenger second, integrate the winner" structure is the
design's answer to the tension between the project's empirical lesson and the master's
Deep Learning course: the course work is the challenger study itself, and it is honest
science regardless of which model wins.

### 6.3 Handling imbalance

`scale_pos_weight` / class weights tuned on validation AUC-PR (N16 precedent), no
synthetic oversampling (section 3.5), stratified-by-race evaluation. Report per-head
lift over base rate (the N13/N14 discipline: an AUC-PR of 0.07 was accepted there
because it was 1.67x the 0.043 baseline; the same honest framing applies).

### 6.4 Calibration

Platt scaling per head, fitted on the validation year, exactly the project's shipped
recipe (N12's calibrator, N16's `calibration: platt, fitted_on: val_2024`). Evaluation
via reliability diagrams and ECE per head, plus **decision-relevant calibration**: the
MC layer consumes these probabilities as Bernoulli parameters, so a 0.30 must mean 30%.
Calibration quality is not cosmetic here; it is the contract with Layer 2. If Platt
underfits the hazard head across the lap axis, isotonic regression is the fallback
(more flexible, needs the larger pooled validation set; decision at M2).

### 6.5 Output contract: what the orchestrator's Monte Carlo can consume

The agent's per-rival output mirrors the distribution shapes Layer 2 already samples
(Triangular via P10/P50/P90, Bernoulli via probabilities), so consumption requires no
new sampling machinery:

**Proposed `RivalIntentOutput` (one per tracked rival):**

| Field | Type | Consumed by |
|---|---|---|
| `driver` | str (FIA code) | prompt, MC bookkeeping |
| `p_pit_next1 / next3 / next5` | float [0,1], calibrated | MC Bernoulli draws; prompt |
| `pit_lap_p10 / p50 / p90` | int quantiles, conditional on stopping | MC Triangular draw of the rival's stop lap (OVERCUT scoring) |
| `compound_probs` | {SOFT, MEDIUM, HARD} simplex | prompt; future stint planning |
| `p_undercut_threat` | float, P(rival's stop is an undercut on our driver), only when we are the car ahead in their window | MC STAY_OUT penalty term; prompt |
| `p_covers_our_stop` | float, P(rival reacts to our stop within 2 laps) | MC UNDERCUT discount; prompt |
| `observability_flags` | staleness / wet / data-quality markers | prompt honesty; debugging |
| `reasoning` | str, template-generated from top features | prompt block, UI |

Plus an aggregate `RivalContext`: the list for the in-scope rivals, sorted by strategic
relevance (threat first), with a one-line grid summary (how many cars are in their pit
window this lap). The concrete Pydantic naming stays open until implementation, but the
shape above is the design commitment: **quantile triples and calibrated probabilities,
nothing the MC cannot draw from directly**.

`reasoning` is deliberately template-generated (from feature attributions), not
LLM-generated, in v1: the agent then works identically in the no-LLM path (the project
maintains a hard no-LLM mode in the CLI and programmatic guardrails; an agent whose
output depends on an LLM would break that parity). Whether the Rival Agent gets an
optional LLM synthesis layer like N25-N29 (nicer prose, tool-calling ReAct shape for
architectural symmetry) is open question Q3; the default answer is: ReAct-wrapped like
its siblings for the multi-agent narrative, but with the deterministic path as the
guaranteed spine, mirroring how the existing agents degrade.

---

## 7. Integration: an additive node feeding N31

### 7.1 The additive-only construction

Untouchables force a clean design, and the project's own rule ("duplicate before
modifying") provides the pattern:

1. **New module** (a new file under `src/agents/`, which is additive by the project's
   definition: new entry points, zero edits to existing files): the Rival Agent itself,
   built in the same shape as its siblings: a class wrapping the trained models, tools
   exposing `predict_rival_pit`, `predict_rival_compound`, `score_rival_threat`, a
   LangGraph ReAct wrapper for symmetry, and public entry points
   `run_rival_agent(lap_state)` / `run_rival_agent_from_state(lap_state, laps_df)`
   returning `RivalContext`. Its only input is `lap_state["rivals"]` plus the additive
   gap-history key (below); it physically cannot see privileged driver telemetry of
   rivals because the boundary never puts it in `lap_state`.
2. **Additive `lap_state` key** for gap history: a runtime gap provider reading
   `intervals.parquet` and exposing per-rival gap traces (the P5 audit's Phase 4 item
   13, "wire intervals.parquet into a runtime gap provider behind an additive lap_state
   key, respecting the single-driver boundary"). The Rival Agent is the first consumer
   this audit item was waiting for.
3. **New orchestration entry point** in a new module (working name: the anticipatory
   orchestrator): a duplicate-and-extend of `run_strategy_orchestrator_from_state` that
   (a) additionally invokes the Rival Agent, (b) runs the extended MC (section 7.2),
   (c) injects a RIVAL INTENT block into the Layer 3 prompt (section 7.3), and (d)
   returns the **same frozen 14-field `StrategyRecommendation`**. The existing
   orchestrator file is not edited; the baseline for the ablation is the untouched
   original entry point, which is methodologically ideal: the control arm is literally
   the shipped TFG system, byte for byte.

The six sub-agents are untouched by construction: none of them consume rival intent;
only the new orchestration layer does.

For the TFM's multi-agent chapter, the integration is also formalized abstractly: the
system graph (six specialist nodes, one rival-modeling node, one supervisor) with the
Rival Agent as an opponent-modeling node whose output changes the supervisor's decision
problem from single-agent optimization under nature-uncertainty to optimization under
strategic uncertainty. That formalization (plus the MoE routing rule for when the Rival
Agent even needs to run, e.g. skip when no rival is within a pit cycle) is coursework
material for Multi-agent Systems (102468) and thesis material regardless of
implementation details.

### 7.2 Extending the Monte Carlo layer (the mathematically explicit part)

Layer 2 currently draws, per simulation `i`: cliff (Triangular), SC (Bernoulli), pit
duration (Triangular), our undercut success (Bernoulli), and scores four candidates via
`simulate_lap_window` in position-equivalent units (`POS_GAP_S = 1.5 s/position`). The
anticipatory extension adds, for each in-scope rival `j`:

- `rivalpit_j_i ~ Bernoulli(p_pit_next_W_j)` (W = the 5-lap window the MC already uses).
- Conditional on pitting, `rivallap_j_i ~ Triangular(pit_lap_p10, p50, p90)`.

And modifies the candidate scores:

- **STAY_OUT** gains a threat term: for the rival(s) behind us within the undercut
  window, expected loss `- rivalpit_j_i * q_j * POS_GAP_S`, where `q_j` is the N16
  success probability evaluated **with roles swapped** (rival as attacker, us as
  defender): the existing calibrated model reused symmetrically, no new model needed
  for v1. This is the single most important behavioral change: staying out stops being
  free when a predicted attacker sits in our mirror.
- **UNDERCUT** gains a preemption discount: our undercut draw only pays its bonus when
  the target has not already pitted in the same window
  (`ucut_effective_i = ucut_i AND NOT (rivalpit_target_i AND rivallap_target_i <= our_stop_lap)`),
  and a cover discount via `p_covers_our_stop` (a covered undercut usually fails; the
  reconstructed dataset will quantify exactly how often, replacing this prose with a
  measured conditional probability).
- **OVERCUT** stops assuming the rival's stop timing implicitly: it conditions on the
  sampled `rivallap_j_i`, which is the quantity an overcut actually bets on.

Everything stays in the existing scoring scheme (`score = alpha * E + (1 - alpha) * P10`
over 500 draws), so risk posture, reproducibility (seeded RNG), and latency
characteristics carry over. The new draws are vectorized Bernoulli/Triangular samples,
adding microseconds per lap.

### 7.3 The prompt injection (Layer 3)

A RIVAL INTENT block is added to the orchestrator prompt, formatted exactly like the
existing sub-agent blocks (verbatim numbers so the LLM can cite them), for example one
line per in-scope rival: code, position, est. tyre age/compound, `p_pit3`,
predicted-stop quantiles, threat/cover probabilities, one-line template reasoning. The
reasoning rubric gains one clause: when a rival's `p_pit3` or threat probability
exceeds a threshold, the reasoning must name that rival and state how the decision
accounts for them, mirroring how the current rubric forces citation of tire and
situation signals.

**Schema discipline**: per the project's standing rule (StrategyRecommendation v2 stays
frozen at 14 fields; richness lives in the prompt), the recommendation schema is NOT
extended. Rival intent reaches the output through: the `reasoning` narrative, the
existing `undercut_target` field, `contingencies` (a predicted rival stop is a textbook
contingency trigger: "if VER pits within 2 laps, switch to PIT_NOW"), and
`scenario_scores` (which now reflect rival-aware MC values). Only if evaluation shows
prompt-only integration measurably loses information does a schema field (e.g.
`rival_alerts`) get proposed, following the memory's prompt-first escalation rule.

### 7.4 Runtime and surfaces

- **Latency budget**: GBDT inference for about 6 rivals is sub-millisecond; the feature
  builder is a per-lap incremental update over already-loaded frames. The rival agent
  adds no LLM call in its deterministic spine, so per-lap latency impact is negligible
  next to the existing sub-agent LLM calls.
- **Surfaces**: the CLI ablation runner is the primary TFM surface. The Arcade rival
  panel and the Streamlit/SPA views are natural consumers of `RivalContext` (the
  Head-to-Head mode finally gets a predictive column), but they are post-TFM polish,
  not evaluation infrastructure.
- **Live-feed forward compatibility**: because features are built from events, not
  replay-only columns (section 4.1), the agent is contract-compatible with the future
  OpenF1 WebSocket adapter that the `lap_state` design anticipates.

---

## 8. Evaluation and ablation

Three levels, from component to system, plus the observability study. The protocol is
fixed before training (this section is the pre-registration).

### 8.1 Level 1: predictor vs reconstructed ground truth

Split: train 2023-2024, validate late 2024, test all of 2025 (never touched during
development). Metrics per head:

- **H1 pit timing**: AUC-PR for pit-within-3 (primary, with base-rate and lift
  reported), same for within-1 and within-5; calibration (reliability curves, ECE);
  among true stops, MAE between predicted stop lap (p50) and the real one, and the
  hit rate of the [p10, p90] interval (target: about 80% empirical coverage, matching
  the interval's nominal meaning; N15's 70.5% coverage on P05-P95 shows the honest
  reporting style).
- **H2 compound**: accuracy and log-loss conditional on a stop, against the
  "most common compound for that circuit/phase" baseline; confusion by race phase.
- **H3/H4 undercut/overcut**: AUC-PR against attempt base rates; success-prediction
  checked against N16's shipped performance as a sanity anchor (the tasks overlap but
  are not identical: N16 predicts success given an attempt; H3 predicts the attempt).

Baselines to beat (all cheap, all honest):

1. Class prior (no-skill floor).
2. **Circuit-history heuristic**: pit probability from the historical stop-lap
   distribution of that circuit (train years), independent of the rival's state.
3. **Tyre-age heuristic**: pit when est. tyre age exceeds the compound's median stint
   length at that circuit; the "any wall intern could do this" baseline.
4. The GBDT itself is the baseline for the sequence challenger (section 6.2).

If the full model does not clearly beat heuristics 2-3, the TFM's conclusion changes
character (rival behavior is mostly schedule-driven, and the anticipatory value lies in
the interaction terms, not the timing model); that is a legitimate finding and the
protocol must allow it to surface rather than bury it.

### 8.2 Level 2: system ablation, with vs without the Rival Agent

The core of the research question. Design:

- **Arms**: (A) baseline, the untouched `run_strategy_orchestrator_from_state` (the
  shipped TFG system); (B) anticipatory, the new entry point with rival-aware MC and
  prompt. Identical inputs per lap (same replay stream, same radio corpus, same seeds).
- **Benchmarks**: the TFG-validated GPs (Hungary, Qatar, Australia, plus the documented
  divergence cases like the Qatar 2025 V7 SC scenario that produced the
  RCMContextResolver finding), and a held-out set of additional 2025 races never used
  in TFG validation, to guard against tuning-to-the-demo.
- **Measures per decision point** (lap or windowed decision episode):
  1. Agreement with the **real wall's decision** (did the system's action match what
     the team actually did in the window?).
  2. Agreement with the **real outcome** (when the system diverged from the wall, did
     the race outcome vindicate the system or the wall? Scored on the TFG's existing
     divergence-case methodology).
  3. **Counterfactual position delta** via the simulator's own window scoring
     (reported with the explicit caveat that judge and player share assumptions:
     supporting evidence, never primary).
  4. **Anticipation-specific probes**: the subset of laps where a rival actually pitted
     within 3 laps: did arm B's recommendations and contingencies reference the threat
     before it happened, and did arm A miss it? This is where the mechanism, not just
     the aggregate, becomes visible; case cards for the TFM's qualitative chapter.
- **LLM nondeterminism control**: temperature is already 0.0; additionally pin
  provider/model per the project rule (OpenAI or LM Studio, never Anthropic), run
  `n >= 3` repeats per arm to bound residual variance, and run a **no-LLM sub-ablation**
  (MC argmax only, both arms) that isolates the rival effect on the decision layer with
  zero LLM variance. If the effect only exists with the LLM and not in the MC scores,
  that itself is a finding about where the anticipation acts.
- **Statistics**: paired per-decision comparison across arms (same race, same lap),
  bootstrap CIs over races (races, not laps, are the independent units), and a
  pre-declared primary endpoint: agreement-with-outcome on divergence episodes.

### 8.3 Honest treatment of the references

Neither reference is ground truth of optimality, and the TFM must say so plainly: the
wall optimizes team-level objectives with private information (agreeing with it is
evidence of plausibility, not of optimality), and the real outcome is one noisy sample
from the race's stochastic process (a good decision can lose). The evaluation therefore
reports both references, never merges them into one score, and leans on the divergence
episodes, where the TFG already built the interpretive machinery, for the strongest
claims.

### 8.4 Level 3: observability and scope ablations

- **Oracle ablation (the observability price tag)**: retrain the same architecture with
  privileged features (true `TyreLife`, `FreshTyre`, actual next compound as a
  cheating upper bound for H1 conditioning). The gap between oracle and public-info
  performance **quantifies the cost of partial observability**, turning section 4 from
  a disclaimer into a measured result. This is the design's answer to "model it as
  uncertainty, do not pretend to observe it": show exactly what pretending would have
  been worth.
- **Scope ablation**: runtime rival set of 5-nearest vs pit-cycle-radius vs full grid;
  measures whether anticipating the whole grid adds anything over the strategic
  neighborhood (expectation: no, and that negative result cleanly justifies the scoped
  runtime design).
- **Feature-family ablation**: drop F2 (gap dynamics), F5 (priors), etc., to attribute
  predictive power; SHAP analysis for the interpretation chapter.

---

## 9. Master course mapping (light, forward, caveated)

**Status caveat, stated up front**: the master has not started. This mapping is an
orientation of which course could exercise which piece, taken from the formal proposal;
it is NOT a plan of record for any deliverable. Two standing gates from the project's
own notes apply before any of this is acted on: (1) confirm whether the master has
started and which pieces, if any, already exist; (2) before each course project,
confirm the course brief actually allows a self-chosen dataset/problem.

| Course | Piece of this design it could serve |
|---|---|
| Sistemas multi-agente (102468) | The Rival Agent node, the graph formalization, the anticipatory orchestration (sections 7.1-7.3). Core of the TFM. |
| Datos temporales y complejos (102472) | Rival lap-by-lap behavior as sequences: the hazard formulation, trend features, the sequence challenger, optionally the latent-state stretch (sections 2.1, 4.2, 6.2). |
| Deep Learning (102469) | The sequence challenger study vs the frozen GBDT (section 6.2). |
| Metodos supervisados (102470) | The multi-head supervised pipeline on the reconstructed ground truth: imbalance handling, calibration, evaluation (sections 6.1-6.4, 8.1). |
| Metodos no supervisados (102471) | Rival situation/profile clustering (extends the K=4 circuit clustering to behavioral profiles feeding F5). |
| Big Data (102473) | The ground-truth reconstruction pipeline at scale (section 3) and the Tier 1 ingestion. |
| Introduccion a la Investigacion (102463) | The methodological scaffolding: research question, pre-registered protocol (section 8), baselines. |
| TFM (102484) | The integration plus the end-to-end evaluation: what no single course produces. |

**CRITICAL academic caution (carried verbatim from the proposal, non-negotiable):** do
NOT double-submit the same artifact to a course and to the TFM (self-plagiarism / double
evaluation). Each course deliverable must be produced for that course (its own report,
experiments, and scope), transparently declared as extending an open project. The TFM
must contribute the **new integration, evaluation, and research** on top, not repackage
course projects. Confirm the norms of each course and of the master with coordination
(`master@aepia.org`) before relying on any of this mapping. Where a course fixes its own
dataset, use the course's dataset for the deliverable and keep the F1 variant for the
TFM.

---

## 10. Phased research roadmap

Milestone-sized phases; each produces a verifiable artifact. Dependencies on the repo's
audit backlog are named rather than duplicated. No dates: the sequencing is the
commitment, the calendar depends on the master's timeline.

**M0: Rival data readiness pack (ground truth v1).**
Execute the P5 audit's Phase 4 items 13-15 as scoped there: Tier 1 OpenF1 ingestion
(`/v1/stints`, `/v1/pit`, `/v1/position`) on the radio-builder template; the gap
provider behind an additive `lap_state` key; the reconstructed 2024-2025 full-grid
event dataset with the label rules of section 3, validated (source cross-checks, stint
invariants) and published to the Hub with a dataset card. Soft dependency: race
identity resolution (P5 Phase 0). Exit criterion: the P5 verification protocol passes
(pit counts match FastF1 exactly or with documented deltas).

**M1: Label census and behavior study.**
Exploratory analysis on the pack: stop-lap distributions per circuit/compound, undercut
attempt and success base rates, cover-reaction frequencies, the data-derived per-circuit
undercut windows, wet-race impact. Fixes the operational thresholds left open in
section 3.4 (attempt windows, overcut bounds) with sensitivity analysis. Exit: the label
definitions frozen and the evaluation base rates known.

**M2: Baseline predictors and calibration.**
Heuristic baselines, the LightGBM multi-head with the survival framing for H1, Platt
calibration, full Level 1 evaluation on 2025. Exit: a frozen, calibrated reference
model with its metrics table.

**M3: Sequence challenger and observability ablations.**
The GRU/TCN challenger under identical splits; the oracle (privileged-features)
ablation; decision on the latent-state stretch goal. Exit: the model choice for
integration, and the measured observability cost.

**M4: Additive integration.**
The Rival Agent module (deterministic spine plus ReAct wrapper), the anticipatory
orchestrator entry point (duplicate-and-extend), the MC extension of section 7.2, the
prompt block of section 7.3. Zero modification of untouchables, verified by diff. Exit:
arm B runs end to end on a validated GP with rival intent visibly reflected in
contingencies and reasoning.

**M5: System ablation.**
The full Level 2 protocol of section 8.2 on the validated GPs plus the held-out 2025
set, with repeats and the no-LLM sub-ablation; error analysis and case cards. Exit: the
research question answered with pre-registered metrics.

**M6: TFM consolidation.**
Write-up, reproducibility packaging (dataset card, model cards, seeds, exact entry
points), limitations, and the 2026-era scoping statement. Candidate spin-off: a short
paper on the dataset plus opponent-modeling results, continuing the TFG-to-IEEE-paper
path.

---

## 11. Risks and limitations

- **Label noise in intent labels (H3/H4).** Attempt labels are behavioral
  reconstructions; some "undercut attempts" are forced stops in disguise. Mitigations:
  situation-plus-outcome labeling (section 3.4), sensitivity analysis in M1, radio
  enrichment where available. Residual risk: H3/H4 metrics will be noisier than H1/H2;
  the TFM should stake its headline claims on H1/H2 and the system ablation.
- **Rival behavior may be mostly schedule-driven.** If circuit-history heuristics get
  close to the model, the incremental value of learning is small. The protocol
  surfaces this (section 8.1); the TFM narrative must be robust to it (the interaction
  modeling and the system-level effect can still carry the thesis).
- **Small positive counts bound model complexity.** About 2k stops across two seasons;
  the sequence challenger may lose (as N12B did). This is a planned-for outcome, not a
  failure mode.
- **System-level effect may be diluted.** The orchestrator has guardrails, MC noise,
  and an LLM in the loop; a good rival prediction can drown before reaching the
  decision. The no-LLM sub-ablation and the anticipation-specific probes (section 8.2)
  are designed to localize where the signal survives or dies. If the effect exists in
  MC scores but not in final recommendations, the finding is about the synthesis
  layer, and it is still a finding.
- **Evaluation references are noisy** (section 8.3). Claims are phrased against both
  references separately, with divergence episodes as the strongest evidence.
- **Era scoping.** Everything is 2022-2025 regulation racing. The 2026 rules change
  strategic behavior (documented in the 2026-reg audit); no claim transfers without
  retraining. State it; do not fight it inside the TFM.
- **Repo-side dependencies.** M0 leans on P5 audit items (identity module, Tier 1
  ingestion). If those have not landed when the TFM starts, M0 absorbs them (they are
  small and fully specified in the audit); the risk is schedule, not feasibility.
- **Academic process risk.** The course-mapping cautions of section 9 (double
  submission, dataset freedom, coordination sign-off) are process risks with a simple
  mitigation: ask first, in writing.

---

## 12. Open questions for Victor

**Q1: Runtime rival scope.** Pit-cycle-radius (recommended: strategy-grounded,
self-adjusting per circuit, typically 4-6 cars) vs fixed 5-nearest vs full grid at
runtime? (The dataset is full-grid regardless, per the P5 audit's proposal.)

**Q2: Label thresholds.** The undercut-attempt window definition (data-derived
per-circuit window as proposed?) and the overcut bounds (stay out >= 2 laps, pit by
L+6): fix them from the M1 sensitivity study, or pre-commit now for pre-registration
cleanliness?

**Q3: Does the Rival Agent get an LLM layer?** Recommended: ReAct wrapper for
architectural symmetry with N25-N29, but the deterministic ML spine is the guaranteed
path and the no-LLM mode ships first. Alternative: purely deterministic agent (cheaper,
simpler, less symmetric). This also affects LLM cost per lap (about 6 rivals must NOT
mean 6 extra LLM calls; the design assumes at most one).

**Q4: Schema discipline confirmation.** Keep `StrategyRecommendation` frozen at 14
fields and integrate rival intent via prompt, contingencies, and `undercut_target`
only (per the standing prompt-first rule), accepting that a `rival_alerts` field is
only proposed if prompt-only integration measurably fails?

**Q5: Wet races.** Keep them in H1 (pit timing) with a `wet_affected` flag and
dry-only H2, as proposed, or exclude wet-affected races entirely from v1 like the
existing dry-only models?

**Q6: 2023 in training.** Include 2023 (already on disk) for training volume at the
cost of mild era drift within the regulation cycle, or train strictly 2024, validate
2024-late, test 2025? Recommended: include 2023 (matches N15/N16 precedent of training
2023-2024).

**Q7: Timing and the master gates.** Per the standing note: before mapping any piece to
coursework or starting implementation, confirm (1) whether the master has started and
(2) whether any piece already exists from summer work, then re-open section 9 with the
actual course briefs in hand.

---

## 13. Related documents

- `C:\Users\victo\Desktop\Documents\Master\propuesta_master.md` (outside the repo): the
  formal TFM proposal this design deepens.
- `documents/audits/AUDIT_P5_DATA_ENGINEERING.md`: data readiness (F-10 Tier 0/1, Phase
  4 items 13-15, open question 7), race identity (F-01), validation contracts.
- `documents/audits/AUDIT_2026_REG_CONCEPT_DRIFT.md`: era scoping and drift program.
- `documents/audits/AUDIT_ML_AGENTS_EVAL.md`: the evaluation-infrastructure backlog the
  Level 2 protocol composes with.
- `src/simulation/race_state_manager.py`, `src/simulation/replay_engine.py`: the
  boundary and the `lap_state` contract.
- `src/agents/strategy_orchestrator.py`: the three layers, the MC scheme, the frozen
  schema (read-only reference; never modified by this work).
- `scripts/cli/runner.py` (`run_h2h`) and `run_simulation_cli.py` `--rival`: the
  existing 2-driver mode (read-only reference).
- `data/models/pit_prediction/model_config.json` and `model_config_undercut_v1.json`:
  N15/N16 features, calibration recipe, circuit traversal lookup.
- `documents/thesis/`: the TFG thesis and IEEE technical report, the deep reference for
  the baseline system's validated metrics and the divergence-case methodology.
