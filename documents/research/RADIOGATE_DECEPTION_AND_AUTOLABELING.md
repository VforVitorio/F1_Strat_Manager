# radiogate: Deception Detection and Auto-Labeling for the F1 Team Radio Corpus

**Status: research design, future work (post-TFG). Plan only, no code, no commitments.**

This document is the methodology design for the `radiogate` initiative of the F1 StratLab
ecosystem (initiative 3 of 5 in the post-TFG vision, see `FUTURE.md` sections 10 and 11,
not versioned). It covers two coupled research problems:

1. **Picaresca detection**: inferring when a driver or team is exaggerating, sandbagging,
   or misdirecting over team radio, so the strategy system can estimate real strategic
   intent instead of taking the literal words at face value.
2. **The auto-labeler**: a scalable labeling pipeline that turns every team radio clip
   OpenF1 exposes into a well-labeled corpus, published as `f1stratlab/f1-team-radio-corpus`
   on the Hugging Face Hub.

The two problems are deliberately joined: picaresca detection is only tractable because
this project already has per-lap telemetry integrated (FastF1 + OpenF1 + the strategy
models), and the corpus is only "well-labeled" if it carries the telemetry-alignment
features that make picaresca labels possible. Neither half stands alone.

Branding rule (from the ecosystem naming decision, 2026-06-12): `radiogate` does not carry
"f1stratlab" in its name, so its README and the dataset card MUST state explicitly that it
is part of the F1 StratLab ecosystem.

Constraints inherited from the project:

- LLM providers are OpenAI or LM Studio (local), or open-source models from the HF Hub.
  Never Anthropic.
- `src/agents/` internals, `scripts/run_simulation_cli.py`, and `notebooks/**` are
  untouchable. Everything proposed here is additive: new modules, new scripts, new
  dataset trees, optional new LangGraph nodes.
- This is a research design a graduate student would hand to an advisor: it is honest
  about what is hard, and every claim about existing assets is grounded in a real file.

---

## 1. Where radiogate sits

- **Ecosystem**: post-TFG, F1 StratLab becomes a multi-repo ecosystem. `radiogate` is the
  Radio NLP initiative: mega-corpus + auto-labeling + improved sentiment + picaresca
  detection. The HF artifact is `f1stratlab/f1-team-radio-corpus` under the `f1stratlab`
  org (the current `VforVitorio/f1-strategy-dataset` is planned to move there).
- **Repo topology is an open decision**: the stated preference is a submodule of the
  core repo; the standing recommendation in `project_future_vision` is an independent
  public repo for visibility (the corpus is a standalone artifact, and the ecosystem rule
  says "repo independiente si es artefacto standalone"). Decide at kickoff (open question
  Q2, section 7).
- **Academic placement**: picaresca detection was explicitly considered and discarded as
  the TFM topic (the TFM is the Rival Agent). It remains future or personal work. It maps
  naturally onto the MUIIA PLN course (102467) essay territory, and its output (a rival
  radio trust signal) is a candidate input to the Rival Agent's intent estimate, which
  creates a clean but optional dependency (Q8).
- **Roadmap placement**: FUTURE.md phases put the radio corpus in Fase 1 (corpus to HF),
  before the LoRA (Fase 2) and the 2026 adaptation (Fase 4). The picaresca layer extends
  work-item 2 of the thesis's future-work list.

---

## 2. Asset inventory: what already exists (reused, not reinvented)

| Asset | Where | State |
|---|---|---|
| OpenF1 radio ingestion (hardened) | `src/data_extraction/openf1/radio_dataset_builder.py` | Production. Resolves session_key per (year, country), pulls `/v1/team_radio` + `/v1/race_control`, maps clips to laps by interval matching on `/v1/laps` `date_start` + `lap_duration`, retry session for OpenF1 429s (5 retries, exponential backoff, honors Retry-After), multi-race country slug disambiguation (`italy_imola`, `united_states_miami`, ...) |
| Build CLI | `scripts/build_radio_dataset.py` | Production. Static per-GP builds to `data/processed/race_radios/{year}/{slug}/{radios,rcm}.parquet` + MP3s under `data/raw/radio_audio/{year}/{slug}/driver_{N}/` |
| HF publisher | `scripts/upload_radio_corpus.py` | Production. Idempotent `HfApi.upload_folder`, dedupe by content hash |
| Published corpus (2025, race sessions, strategically filtered) | HF `VforVitorio/f1-strategy-dataset` | 48 parquets (~430 KB) + 529 MP3s (~80 MB), verified on the Hub |
| Replay-time consumption | `src/nlp/radio_runner.py` | Production. `RadioPipelineRunner` (parquet to dict adapter), `WhisperTranscriber` process-local singleton, JSON transcript cache at `data/processed/radio_nlp/{year}/{slug}/transcripts.json` keyed by normalized path with model-version invalidation |
| NLP pipeline config | `data/models/nlp/pipeline_config_v1.json` | v1. Sentiment = N20 RoBERTa (`bert_sentiment_v1`), intent = N21 SetFit ModernBERT (`intent_setfit_modernbert_v1`), NER = N22 BERT-large BIO (`ner_v1/bert_bio_v1`), RCM = N23 rule-based. Latency benchmark: mean 43.7 ms, P95 45.8 ms on GPU |
| NLP notebooks | `notebooks/nlp/N17-N24, N33` | N17 labeling, N18 transcription (Whisper), N19/N20 sentiment, N21 intent, N22 NER, N23 RCM parser, N24 unified pipeline, N33 dataset builder prototype |
| Legacy src wrappers | `src/nlp/pipeline.py`, `sentiment.py`, `radio_classifier.py`, `ner.py` | Legacy jupytext exports predating N24 (old model paths, old intent model). The clean extraction of `run_pipeline` / `run_rcm_pipeline` from N24 is itself pending (`project_nlp_src_wrapper`). radiogate should treat that extraction as a prerequisite, not build on the legacy files |
| Telemetry side | FastF1 + OpenF1 extractors under `src/shared/data_extraction/`, per-lap `lap_state` contract in `src/simulation/` | Production. Driver gets full telemetry, rivals timing-only |
| Strategy models | `data/models/*` (lap time XGBoost, TireDegTCN + MC Dropout, LightGBM overtake/SC/undercut, HistGBT pit) | Production, calibrated. These become the counterfactual engines for divergence scoring (section 3.4) |

Two facts about the existing corpus that matter for radiogate:

1. **The published corpus is strategically filtered, not archival.** The builder drops
   unmapped clips, formation lap (0), race-start lap (1), and everything from the
   chequered-flag lap onward (`RACE_START_LAPS`, `DROP_LAST_LAP` in
   `radio_dataset_builder.py`). That is correct for feeding N29 during a replay, and
   wrong for a research corpus. radiogate must make the filter a consumption-time
   policy, not an ingestion-time deletion (section 4.1).
2. **NER is the weak link**: span-F1 around 0.42 trained on roughly 399 labeled examples.
   Any picaresca system that depends on claim extraction inherits this weakness, so
   fixing NER is on the critical path (section 4.5).

---

## 3. Part 1: picaresca detection

### 3.1 Reframing: divergence, not lie detection

"Is this sentence a lie?" is the wrong problem. There are no lie labels, there never will
be (nobody confesses on the record), and asking an LLM to guess sincerity from text alone
is hallucination bait: the model would pattern-match on tone and invent certainty where
none exists.

The tractable frame is **claim vs reality vs outcome divergence**:

- A radio message often contains a **claim** about a verifiable physical state ("no grip",
  "tyres are gone", "I'm saving fuel", "something is wrong with the brakes") or an
  **announced action** ("box box", "we're staying out", "push now").
- The project already has the **reality**: per-lap telemetry, stint structure, pit events,
  race control messages, and calibrated model expectations for what the car should be
  doing given tyre age, compound, and fuel-corrected pace.
- The race provides the **outcome**: what the driver actually did in the next N laps, what
  the team actually did, and how rivals reacted.

When claim, reality, and outcome line up, the message is informative at face value. When
they diverge, something interesting happened: exaggeration, sandbagging, misdirection, a
coded message, or an honest mistake. The system's output is therefore not "lie / truth"
but a **divergence measurement plus a calibrated probability of strategic misdirection**,
and those two things are kept on separate axes throughout.

This matters epistemically: divergence is objective and auditable (the lap times can be
plotted next to the transcript); intent is latent and will always be soft. Publishing hard
numbers for the first and calibrated, explicitly-uncertain numbers for the second is the
only honest way to report this work (section 3.6).

### 3.2 The label space: three axes, never collapsed

Every claim-bearing radio message gets three orthogonal annotations:

1. **Verifiability** (deterministic, from claim type): `VERIFIABLE` (maps to a telemetry
   predicate), `PARTIALLY_VERIFIABLE` (predicate exists but is confounded, e.g. fuel
   saving where fuel load is unobservable but throttle traces are not),
   `UNVERIFIABLE` (feelings, plans, references to information not observable from
   outside).
2. **Divergence** (continuous score + banded class, from the distant-supervision engine,
   section 3.4): `CONSISTENT`, `MILD_DIVERGENCE`, `STRONG_DIVERGENCE`, computed only for
   verifiable and partially verifiable claims.
3. **Misdirection intent** (probabilistic, from the model in 3.5): p(strategic
   misdirection), emitted only when divergence is at least MILD and context features
   support it. Never a hard label in the silver corpus; a soft consensus label in GOLD.

A message can be strongly divergent and innocent (the driver was wrong about his own
tyres; drivers misjudge grip constantly). Divergence is a necessary but not sufficient
signal for picaresca. Collapsing the axes would poison the corpus.

### 3.3 Taxonomy of radio picaresca

Each class below is defined with an observable signature so that it can (a) drive a
weak-supervision labeling function and (b) be recognized by human annotators with a
telemetry panel in front of them. The taxonomy is deliberately small; classes that cannot
be observed do not get labels.

| ID | Class | Definition | Observable signature (claim vs reality vs outcome) |
|---|---|---|---|
| P1 | Exaggerated degradation complaint | Driver overstates tyre/grip problems to force a pit call, justify pace, or lower expectations | Claim: intent=PROBLEM + tyre/grip entities. Reality: next-3-lap fuel-corrected pace within model expectation for that compound/age (TireDegTCN counterfactual). Outcome: no pit within k laps, or a purple/green sector shortly after the complaint |
| P2 | Sandbagging / downplaying pace | Team or driver understates true pace (classically in practice/quali radio, sometimes in race stints behind traffic) | Claim: "we're struggling", "no pace". Reality: sector times and stint pace percentile vs field contradict it. Outcome: pace materializes when it matters |
| P3 | Fake problem to misdirect rivals | Fabricated or inflated technical issue voiced on an open channel to bait a rival reaction (early pit, pushed strategy) | Claim: technical_issue entities, severe framing. Reality: no telemetry trace (speed traces, throttle/brake patterns nominal), no retirement, pace unaffected. Outcome: rival strategy response within k laps (rival pits, pushes) with no matching real problem |
| P4 | Dummy pit call | "Box, box" (or garage/pit-crew theater) with no actual stop, to trigger a rival's covering stop | Claim: pit_call entity / ORDER intent. Reality + outcome: no pit event for the driver within 2 laps (OpenF1 `/v1/pit`), optionally a rival pit event in the same window |
| P5 | Fuel/energy-saving claims | "Saving fuel", "lift and coast", "managing" used as cover for pace, or claimed but not executed | Claim: saving language. Reality: throttle traces (OpenF1 car_data ~3.7 Hz) show or do not show lift-and-coast signatures (early throttle drop before braking zones); lap-time pattern consistent or not |
| P6 | Coded / euphemistic instruction | Engineer messages whose surface form hides the real instruction ("Plan C", "Scenario 7", "strat 2", agreed code words) | Claim: low-semantic-transparency instruction (OOV codewords, no verifiable content). Outcome: a consistent action follows (pit, pace change, position swap). Not deceptive as such, but a distinct class: literal NLP reads it as noise, the action reveals meaning |
| P7 | Feigned/inflated incident severity | Brake/engine/damage complaint that conveniently disappears | Claim: severe technical issue. Reality: no degradation in speed/brake telemetry across subsequent laps, no retirement, no pit for repairs. Outcome: complaint never recurs |
| P8 | SC-fishing / condition inflation | Reporting debris, stopped cars, or track conditions in a way that invites a Safety Car or VSC review favorable to the reporter's strategy window | Claim: track_condition/incident entities. Reality: no matching `/v1/race_control` message (no investigation, no flag) within the window. Context: the reporting driver is in a pit window where an SC would gift a cheap stop |
| P0 | Honest baseline (control) | Claims that check out, complaints followed by confirming telemetry and consistent action | Required as the majority control class; everything is measured against it |

Notes on the taxonomy:

- P6 (coded instructions) is in the taxonomy because a literal NLP pipeline mis-scores it
  (it looks like INFORMATION with no content). For the strategy system, detecting "this
  is a code word, the literal meaning is not the real meaning" is as valuable as
  detecting exaggeration.
- The base rate of P3/P4/P8 (true misdirection) is low. Most radio is honest, most
  complaints are real. This is an extreme class-imbalance problem (section 6) and the
  reason weak labels are graded by divergence intensity rather than forced into binary
  deception labels.
- Famous incidents are used as case studies, not as training labels (section 3.6).

### 3.4 The distant-supervision ground-truth engine

This is the core methodological bet: **objective telemetry and subsequent outcomes are
the only scalable supervision source for picaresca**. The design:

**Step 1: claim extraction and typing.** For each transcribed clip, run the existing
pipeline (N24 config: sentiment, intent, NER) plus a new claim-typing step that maps the
message onto a small closed set of claim families: `TYRE_GRIP`, `PACE`, `FUEL_ENERGY`,
`BRAKES`, `POWER_UNIT`, `DAMAGE`, `TRAFFIC`, `WEATHER_TRACK`, `PIT_ACTION`,
`INSTRUCTION`, `NONE`. Claim typing can bootstrap from intent + entities (a PROBLEM
intent with a technical_issue entity mentioning brakes maps to `BRAKES`) with an
LLM-assisted fallback for messy phrasing (structured output, closed label set, no free
text). Each claim family carries a fixed set of verifiable predicates.

**Step 2: telemetry evidence windows.** For a claim at lap L by driver D, assemble an
evidence window: laps L-3 to L+N (N = 3 to 5, per predicate), with:

- Lap and sector times, fuel-corrected, from FastF1/OpenF1 (already extracted per-lap in
  the project's parquet trees).
- Stint context: compound, tyre age (OpenF1 `/v1/stints`), pit events (`/v1/pit`).
- Model counterfactuals: this is where radiogate has an unusual advantage. The TFG's own
  calibrated models provide the "expected reality" baseline:
  - TireDegTCN (with MC Dropout quantiles) predicts the expected degradation curve for
    that compound and tyre age: a "tyres are gone" claim is scored against the predicted
    P10-P90 band, not against a naive average.
  - The N06 lap-time model provides expected lap-time deltas, so pace claims are scored
    net of fuel burn and track evolution instead of raw seconds.
- Race control cross-reference: RCM parquet (already built per GP by the same builder)
  for flags, investigations, debris confirmations.
- Rival reaction: pit/pace events for the strategic neighbors (the drivers within a
  covering window), from the same OpenF1 endpoints.

**Step 3: divergence scoring.** For each (claim family, predicate) pair, define a signed,
z-normalized divergence score: how far reality sits from the claim-consistent region,
normalized per circuit, compound, and stint phase so that scores are comparable across
the corpus. Examples of predicate designs (final definitions are an R3 deliverable):

| Claim family | Predicate | Divergence measure |
|---|---|---|
| TYRE_GRIP ("no grip", "tyres are dead") | Post-claim pace vs deg-model expectation | Mean fuel-corrected pace delta over L+1..L+3 minus TireDegTCN-predicted delta, in units of the model's predictive std; strongly negative (faster than a dying tyre allows) = divergent. Bonus signal: any purple/personal-best sector in the window |
| PACE ("that's all I have") | Later push-lap detection under comparable conditions | Best subsequent lap in clean air, fuel-corrected, vs claimed ceiling |
| FUEL_ENERGY ("lift and coast") | Throttle lift signature | Fraction of braking zones in L..L+2 with early throttle drop (car_data at ~3.7 Hz is coarse but sufficient for lift-and-coast detection at the zone level); claim without signature = divergent |
| BRAKES / POWER_UNIT / DAMAGE | Persistence and consequence | Speed-trace and pace regression over the window + terminal events (pit for repairs, retirement); severe claim + zero trace + zero consequence = divergent |
| PIT_ACTION ("box box") | Action match | Pit event for D within 2 laps; absence = P4 candidate; rival pit within the same window strengthens the misdirection reading |
| WEATHER_TRACK ("debris turn 4") | RCM confirmation | Matching race-control message within the window; absence = divergent; SC/VSC actually deployed while the reporter is in a pit window = P8 context feature |
| INSTRUCTION (coded) | Semantic transparency + action correlation | Codeword detection (OOV strategy tokens, numbered plans) + whether a discrete action follows; this one produces a P6 flag, not a divergence score |

**Step 4: weak labels.** Divergence bands (CONSISTENT / MILD / STRONG at fixed z
thresholds, tuned on the GOLD set) become weak labels for exaggeration intensity.
Contextual features (rival reaction, pit-window position, championship stakes proxies,
repetition of the same claim across laps) become inputs for intent estimation but never
auto-generate an intent label on their own.

**Why this works here and would not work elsewhere:** the labeling signals are only as
good as the telemetry integration, the lap mapping, and the counterfactual models. This
project has all three already built and validated (the lap-mapping precision of the
builder, the calibrated deg model, the fuel-corrected pace model). radiogate is unusually
well positioned; this is the honest reason to attempt a problem this hard.

Known confounds the engine must control for (each becomes a covariate or an exclusion
rule, and each is a documented limitation in section 6):

- Track evolution and fuel burn make everyone faster; all pace comparisons are model-
  relative, never raw.
- Traffic: a driver claiming "no pace" while stuck behind a car is consistent, not
  divergent; gap/interval data (OpenF1 `/v1/intervals`) gates pace predicates to
  clean-air laps.
- Tyre warm-up: post-complaint improvement on fresh tyres is expected; predicates apply
  within a stint, never across a pit stop.
- Weather transitions: exclude windows crossing rain state changes (FastF1 weather).
- Driver error vs car problem: unobservable; folded into the honesty caveat, not modeled.

### 3.5 Modeling options and recommendation

All options consume the same inputs: the radio text (and optionally audio features), the
claim type, and the telemetry-divergence feature vector from 3.4. Provider constraint:
OpenAI or LM Studio for anything LLM; open HF models for anything fine-tuned.

**(a) Multi-task head over radio embedding + divergence features.**
A single encoder (a sentence-transformer or the ModernBERT backbone already used for
intent) produces the text embedding; concatenate the numeric divergence/context features;
small multi-task heads predict divergence band, picaresca class (taxonomy), and
misdirection probability.

- Pros: cheap at inference (fits the existing 43.7 ms pipeline budget), fully local,
  trainable on weak labels, calibratable (Platt/isotonic, consistent with how the
  project already calibrates LightGBM outputs), auditable feature importances.
- Cons: weak labels are noisy and the model can only be as good as the labeling
  functions; struggles on long-tail phrasing and multi-turn context; no reasoning trace
  to show a human.

**(b) LLM-as-judge with a strict structured rubric and telemetry evidence in-context.**
For each claim, build a prompt containing the transcript, the claim type, and a compact
telemetry evidence table (the actual numbers from 3.4), and ask for a structured verdict
(divergence band, taxonomy class, confidence, one-sentence evidence citation) under a
closed rubric. gpt-4.1-mini class models for scale, a larger model for adjudication.

- Pros: handles long-tail phrasing, multi-turn context, and subtle framing far better
  than a small classifier; produces human-readable rationales; zero training needed to
  start.
- Cons and required controls: hallucination is the central risk. Controls are
  non-negotiable: (1) the judge NEVER sees a claim without its telemetry evidence table;
  text-only judging is forbidden by design because it invites tone-based guessing;
  (2) closed output schema, no free-form verdicts; (3) mandatory abstention class when
  evidence is insufficient; (4) the rationale must quote evidence fields, and a
  post-validator rejects verdicts citing numbers not present in the prompt; (5) verdicts
  are treated as one more (strong) labeling function, never as ground truth. Cost is the
  other issue: tens of thousands of clips times multi-call judging is real money on
  OpenAI, or real wall-clock on LM Studio.

**(c) Hybrid: weak supervision trains a small calibrated classifier; LLM judge only for
hard and ambiguous cases.**
The labeling functions from 3.4 plus model votes are aggregated by a label model
(section 4.4); the aggregated probabilistic labels train option (a)'s classifier; the LLM
judge is invoked only where the classifier is uncertain, where labeling functions
disagree, or for the taxonomy classes that need contextual reading (P3, P6, P8), and its
verdicts feed back as an additional labeling function and as active-learning routing.

**Recommendation: (c), unambiguously.** Reasons:

1. **Epistemic fit.** The whole design rests on telemetry divergence being the ground
   truth engine. A hybrid keeps the deterministic, auditable signals in charge and uses
   the LLM only where determinism runs out. Option (b) alone inverts that hierarchy.
2. **Cost and scale.** The classifier handles the corpus bulk locally and instantly; the
   judge sees maybe 10 to 20 percent of claim-bearing clips. This is the only option that
   scales to full OpenF1 coverage within a hobby budget (Q6).
3. **Calibration is the product.** The downstream consumer (section 3.7) needs a
   trustworthy probability, not a verdict. Small classifiers over meaningful features
   calibrate well; raw LLM confidences do not.
4. **It matches the project's proven pattern.** The N29 radio agent already layers
   deterministic NLP stages under an optional LLM synthesis stage, with graceful
   degradation when the LLM is unavailable (the Stage 3 try/except lesson from Track A).
   radiogate should replicate that shape: the picaresca signal must exist, degraded but
   sane, with the LLM switched off.

### 3.6 Evaluation: validating what cannot be fully observed

Truth (intent) is unobservable, so the evaluation is deliberately split into what can be
measured hard and what can only be measured soft, and the reporting language keeps them
apart.

**Hard, fully defensible metrics:**

- **Claim extraction and typing**: precision/recall against the GOLD annotations;
  standard, no caveats.
- **Divergence detection**: agreement between the automatic divergence band and human
  judgment when the human sees the same telemetry panel. This is measurable because the
  human is judging the same observable evidence, not guessing intent. Report per claim
  family.
- **Proxy-label validation**: check that divergence predicts what it should. Examples:
  clips labeled STRONG divergence on TYRE_GRIP should show a lower subsequent-pit rate
  than CONSISTENT complaints; P4 candidates should show elevated rival-pit-within-window
  rates vs matched controls. These are falsifiable statements about the labels
  themselves.
- **Calibration**: reliability diagrams and ECE for the misdirection probability on
  GOLD; Platt or isotonic recalibration on a held-out fold (the project's standard
  practice for its LightGBM classifiers).

**The GOLD set (human ceiling):**

- Size: 300 to 500 clips. Composition: all high-divergence candidates from a season
  slice, an equal number of matched CONSISTENT controls (same claim family, same circuit
  type), plus a random sample for base-rate honesty.
- Annotators: 2 to 3 F1-literate annotators (one already available plus recruited; Q3). Each clip is
  presented with the transcript, the audio, AND the telemetry evidence panel; annotating
  picaresca from text alone would just reproduce the tone-guessing failure mode.
- Instrument: per-axis annotation (verifiability, divergence band given the panel,
  misdirection plausibility on a 4-point ordinal scale, taxonomy class), with written
  guidelines and a calibration round of 30 clips before real annotation.
- Agreement: Krippendorff's alpha per axis. Expectation set in advance and reported
  honestly: alpha >= 0.65 is realistic for divergence (humans looking at the same
  numbers), while alpha for misdirection intent will plausibly land in the 0.3 to 0.5
  range. If it does, that number IS a scientific finding about the task's irreducible
  subjectivity, and the paper/report must present it as such rather than hide it.
  Intent metrics are then reported against the consensus of annotators with the
  disagreement rate alongside, never as if a crisp ground truth existed.

**Case studies (qualitative anchor):** a curated set of publicly documented radio-games
incidents from the corpus era (2023 onward), verified to be present in the corpus before
inclusion. Candidate types: dummy-stop episodes, "tyres are dead" followed by fastest
laps, coded team-order negotiations, debris reports with no RCM confirmation. Each case
study shows the full pipeline output next to the known public narrative. Pre-2023
classics (for example the Abu Dhabi 2016 backing-up episode) can be used in the write-up
as conceptual illustrations but cannot be corpus case studies (no OpenF1 coverage) and
must be labeled as such. Incident list curation is an explicit deliverable (Q10) because
misremembered incidents are exactly the kind of error this document must not commit.

**Ablations that the write-up owes the reader:**

- Text-only vs divergence-features-only vs combined (does telemetry actually carry the
  signal, or is the classifier reading tone?).
- Label-model labels vs majority-vote labels (is the weak-supervision machinery earning
  its complexity?).
- With vs without LLM-judge routing (what does the expensive component add?).

**Honest reporting rule:** every published number for misdirection intent carries the
qualifier that it measures agreement with calibrated human suspicion given telemetry,
not deception in the ground-truth sense. Precision on "misdirection intent" will be soft;
the report says so in the abstract, not in a footnote.

### 3.7 Feeding the signal back into the strategy system

The consumer-side design (all additive, nothing in `src/agents/` internals changes):

- **Corpus/runner surface**: `RadioPipelineRunner` output dicts gain optional fields:
  `claim_family`, `divergence_score`, `divergence_band`, `picaresca_class`,
  `p_misdirection`, `evidence_ref`. Absent fields mean "not computed", so every existing
  consumer keeps working (same additive-contract discipline as the `RadioOutput`
  degradation contract from Track A).
- **Rival radio down-weighting**: today the system is single-driver and rival radio is
  not consumed. The natural first consumer is the future Rival Agent (the TFM): its
  intent estimate over a rival's next move should weight that rival's radio evidence by
  (1 - p_misdirection), and treat P4/P3 flags as evidence FOR the opposite of the
  literal claim (a dummy box call is information: the rival wants the tracked car to
  pit).
- **Own-pit-wall guardrail**: in the orchestrator's guardrail layer (the strategic
  guardrails already exist as a pattern), add: no strategy recommendation may cite a
  rival radio claim as primary evidence when `p_misdirection` exceeds a threshold; the
  claim can only enter as context with its trust weight attached.
- **Post-race analytics**: Streamlit gets a "radio honesty" view per race (divergence
  timelines per driver), which is also the natural QA surface for the labels themselves.
- **Deliberately out of scope**: using picaresca output to auto-trigger strategy changes.
  The signal informs and de-weights; it never drives.

---

## 4. Part 2: the auto-labeler and full-coverage corpus

### 4.1 Full OpenF1 ingestion: from 529 clips to everything

Goal: every team radio clip OpenF1 exposes, across all seasons it covers (2023 to
present; OpenF1 has no earlier data), all meetings, all session types (Practice, Quali,
Sprint formats, Race), all drivers, with archive-grade completeness accounting.

Plan (extends `RadioDatasetBuilder`, which stays the single ingestion path):

1. **Enumeration layer.** Walk `/v1/meetings?year=Y` then `/v1/sessions?meeting_key=M`
   instead of resolving one (year, country, Race) at a time. The existing
   `resolve_session` logic becomes one leaf of a full crawl. Every session (including
   pre-2025 seasons and non-race sessions) becomes a build unit with the same slug rules
   (`_MULTI_RACE_COUNTRIES` handling carries over).
2. **Archival filter policy.** The strategic filter (drop laps 0, 1, >= total_laps,
   unmapped) becomes a consumption-time flag, default ON for the replay pipeline and OFF
   for the corpus build. The corpus keeps formation-lap, cool-down, and unmapped clips
   with `lap_number = null` plus a `map_status` column (`mapped`, `formation`,
   `cooldown`, `unmapped_gap`, `no_laps_data`). Rationale: cool-down radio is where
   post-hoc honesty shows up ("we never had a brake problem, good job") and formation
   lap radio carries sandbagging; deleting them destroys picaresca evidence. Non-race
   sessions have no race laps in the same sense; for them lap mapping applies where
   `/v1/laps` provides data and `map_status` records the rest.
3. **Provenance columns** (extend `OUTPUT_SCHEMA`): `source` (openf1), `api_snapshot_date`,
   `recording_url` (already present, kept even after download as the canonical pointer),
   `audio_sha256`, `ingest_version`. Dedup key: `audio_sha256` first (the upload script
   already dedupes by content hash), `(session_key, driver_number, date)` as the logical
   key; both recorded so re-crawls are idempotent.
4. **Completeness ledger.** Per session, persist a ledger row: clips listed by the API,
   clips downloaded, download failures (URL 404s happen; today they are logged and
   skipped), clips mapped per `map_status`, clips transcribed, transcription failures.
   The ledger ships WITH the dataset (a `coverage` table), so "full coverage" is a
   verifiable claim with named holes rather than a hope. Rule: never silently drop;
   every clip the API listed appears either in the corpus or in the ledger with a
   reason.
5. **Rate-limit budget.** The existing retry session (5 retries, exponential backoff,
   Retry-After honored) is kept; the full crawl adds a global pacing budget and
   checkpointed resumability (per-session done-markers) so a multi-day crawl survives
   interruption. OpenF1 throttles aggressively (documented in the builder's constants);
   a full historical crawl is a batch job measured in days, not hours, and that is fine
   because it runs once per season plus incremental top-ups.
6. **Volume estimate (to be validated in R0 with a one-season census).** Race-only,
   post-filter, 2025 produced 529 clips; the pre-filter race estimate from the original
   ingestion plan was ~110 clips per GP. All sessions plausibly multiply race volume by
   3 to 5. Order-of-magnitude planning figure: 10,000 to 30,000 clips for 2023 to 2025,
   growing by several thousand per season. Storage: single-digit GB of MP3s, tens of MB
   of parquet. Whisper cost: at 2 to 5 s per clip on GPU, a full-history transcription
   is one to three GPU-days, cacheable forever by the existing JSON cache mechanism.

### 4.2 Transcription at scale

- **Engine**: the existing `WhisperTranscriber` (model selectable, same flag surface as
  `--whisper-model`), with the JSON cache and model-version invalidation already proven
  in Track A. No new transcription stack.
- **Quality flags per clip** (new columns): Whisper's `avg_logprob`,
  `no_speech_prob`, `compression_ratio`, clip duration, plus a derived
  `transcript_quality` band (ok / suspect / unusable) using Whisper's own standard
  thresholds. Team radio is acoustically hostile (compression, wind, engine noise,
  clipped push-to-talk boundaries); pretending WER is uniform would corrupt every
  downstream label, so quality flags gate which clips are eligible for silver labels.
- **WER measurement, not assumption**: the GOLD annotation pass (section 4.4) includes
  transcript correction for its 300 to 500 clips, which yields a measured WER on
  realistic radio audio, reported per quality band in the dataset card.
- **Language detection**: Whisper's language ID per clip, stored as `language`. Most
  radio is English but Italian, Spanish, French, and Japanese exchanges exist. v1
  policy: transcribe in-language, keep `language`, add an optional machine-translation
  column later if needed; NLP labels are only auto-applied to languages the label models
  actually support (English), everything else routes to the human loop or stays
  transcript-only.

### 4.3 Speaker attribution: driver vs engineer

OpenF1 clips frequently contain both sides of an exchange; the corpus needs per-segment
speaker roles because claims by the driver and instructions by the engineer are different
objects in the taxonomy (P1 vs P6, for instance).

Two-stage design, all open-source:

1. **Diarization**: pyannote-audio (HF, open) segments each clip into speaker turns.
   Radio clips are short (5 to 20 s) with 1 to 3 turns, a much easier regime than
   meeting diarization.
2. **Role classification per segment**: a small classifier over (a) acoustic features
   (the driver channel carries car noise, breathing, heavier compression; the pit wall
   is cleaner) and (b) text features (imperatives and data readouts skew engineer;
   first-person state reports skew driver). Bootstrap labels from the obvious cases
   (clips where text cues are unambiguous), then active-learn the rest.

Output columns: per-clip `segments` list with `(start, end, role, role_confidence)`, and
a clip-level `primary_speaker`. Honest caveat: role accuracy will be imperfect and is
itself a labeled quantity in GOLD; downstream picaresca predicates that depend on "the
driver said it" use the role confidence as a gate.

### 4.4 The auto-labeling pipeline

The pipeline that makes "well-labeled at full coverage" achievable without hand-labeling
30,000 clips. Five components:

**1. Bootstrap pre-labeling.** Run the existing trained models (N20 sentiment, N21
intent, N22 NER, N23 RCM parser via the N24 unified pipeline, after the pending clean
extraction to `src/nlp/`) over every English transcript with quality >= suspect
threshold. Store predictions WITH confidences and `nlp_model_version` (the provenance
pattern the schema already uses). These are bronze labels: nothing more.

**2. Weak supervision (Snorkel-style label model).** For each label family, write
labeling functions (LFs) that vote or abstain per clip:

- Keyword/regex LFs (box calls, tyre vocabulary, code-word patterns, question forms).
- Model-vote LFs (each existing model is one LF, its confidence its vote strength).
- Telemetry LFs: the divergence predicates from section 3.4 are labeling functions for
  the picaresca families; RCM cross-reference LFs for WEATHER_TRACK claims; pit-event
  LFs for PIT_ACTION.
- Structural LFs (speaker role, session type, lap phase: formation-lap claims skew
  sandbagging; cool-down skews honest debrief).
- LLM-judge LFs (section 3.5) on the routed subset only.

A label model (Snorkel's generative model, or its simpler successors; the library is
open-source and the corpus size is well within its comfort zone) learns LF accuracies
and correlations from their agreement structure and emits probabilistic labels. Majority
vote is kept as an ablation baseline (section 3.6). Silver label = label-model
probability above a per-family threshold, with the probability stored, never just the
argmax.

**3. Active learning loop.** Pool-based, with a human in the loop through an open-source
annotation UI (Argilla or Label Studio; decide in R2, Q3):

- Acquisition: a mix of (a) uncertainty (label-model or classifier entropy), (b)
  disagreement (query-by-committee across label model, trained classifier, and LLM
  judge where available), and (c) diversity (embedding-space clustering so batches are
  not 50 near-duplicates of "box box").
- Batch size 100 to 200 clips per round; effort estimate 30 to 60 s per clip for
  sentiment/intent/NER correction, 2 to 3 min for picaresca clips (telemetry panel
  reading); a round is an evening of work, not a month.
- After each round: retrain the affected model(s), refresh silver labels, re-rank the
  pool.
- **Stopping criterion** (explicit, so the loop does not run on vibes): stop a label
  family when EITHER the trained model's F1 on the frozen GOLD slice improves by less
  than 0.5 points for two consecutive rounds, OR the annotation budget for that family
  (Q3) is exhausted. Log the stopping state in the dataset card so corpus users know
  which families converged and which ran out of budget.

**4. Label tiers and confidence policy.** Every label in the corpus carries a tier:

- `gold`: human-annotated, doubly so for the IAA slice.
- `silver`: label-model probability above threshold, quality-gated transcript, model
  versions recorded.
- `bronze`: single-model pre-label, no agreement corroboration. Shipped because they are
  useful for weak pretraining, but clearly marked.

Consumers filter by tier; nothing pretends to be better than it is.

**5. Label families covered.** The auto-labeler applies the same machinery to ALL
families, not just the new ones: `sentiment`, `intent`, `entities` (post-fix, 4.5),
`speaker_role`, `language`, `claim_family`, `divergence` (score + band),
`picaresca_class`, `p_misdirection`, plus the raw telemetry-alignment features as
first-class dataset columns (so corpus users can build their own divergence definitions
without re-crawling telemetry).

### 4.5 Fixing NER (the weak link)

Span-F1 ~0.42 on ~399 examples is not a modeling failure so much as a data-volume and
schema problem. Plan, in order:

1. **Schema audit first.** Nine entity types over 399 examples guarantees sparse classes.
   Audit confusion structure; candidate merges (for example `situation` vs `incident`,
   `action` vs `strategy instruction`) get decided on inter-annotator confusability, not
   aesthetics. A 6-type schema that humans agree on beats a 9-type schema that nobody
   can apply consistently.
2. **LLM-assisted re-annotation.** Pre-annotate spans with an LLM under the closed schema
   (structured output, spans must be verbatim substrings, machine-validated), human
   corrects in the annotation UI. Target 3,000 to 5,000 annotated messages via the
   active-learning loop, which is 8 to 12x the current data.
3. **Model refresh.** Two candidates, evaluated head-to-head on the new GOLD spans:
   fine-tuned GLiNER (open, span-based, strong in low-data regimes, schema changes are
   cheap because labels are text prompts) vs re-trained BERT-BIO on the enlarged data.
   Acceptance bar: span-F1 >= 0.70 on held-out GOLD before NER-derived LFs are allowed
   to feed picaresca claim typing at silver tier; below that, claim typing leans on
   intent + keyword LFs and the LLM fallback instead.

### 4.6 Label quality, coverage checks, and versioning

- **Per-family QC dashboard** (build artifact, not a product): class balance per season
  and per GP, LF coverage/overlap/conflict matrices (standard Snorkel diagnostics),
  silver-vs-gold agreement per family, transcript-quality distribution of labeled vs
  unlabeled pools (to catch quality-correlated label bias).
- **Drift checks**: label distributions per season; 2026 brings a regulation change and
  plausibly different radio behavior (new energy-management vocabulary), so the card
  reports per-season distributions and the models record their training-season range.
- **Dataset versioning**: semantic versions (`v1.0` = transcripts + core NLP labels,
  `v1.x` = label refreshes, `v2.0` = picaresca layer), each version a pinned HF revision;
  downstream training always references a revision hash, mirroring how the project
  already pins `nlp_model_version` in parquet rows. A CHANGELOG section in the card maps
  versions to ingest_version + model versions + LF set versions.

### 4.7 HF dataset design: `f1stratlab/f1-team-radio-corpus`

**Layout (configs/subsets):**

| Config | Contents | One row is |
|---|---|---|
| `clips` | Provenance + coverage: session/meeting keys, year, gp slug, driver, date, lap_number + map_status, recording_url, audio_sha256, duration, quality flags, language, ledger linkage | a radio clip |
| `transcripts` | Whisper text + segments + speaker roles + model versions | a clip |
| `labels` | All label families with tier, probability, model/LF versions | a (clip, label family) pair |
| `telemetry-alignment` | Claim family, evidence-window features, divergence scores/bands, outcome features | a claim |
| `gold` | The human-annotated slice with per-annotator labels (not just consensus) and the annotation guidelines version | a (clip, annotator) pair |
| `coverage` | The completeness ledger per session | a session |

**Splits**: temporal, to prevent leakage and match how the strategy models are already
validated (train on past, test on future): train = 2023 to 2024, validation = first half
2025, test = second half 2025, with GOLD stratified across all three and 2026 reserved
as a drift-evaluation set once it exists. Additional rule: a multi-clip exchange never
straddles a split boundary (group by session).

**Dataset card** must include: the F1 StratLab ecosystem statement (branding rule), data
provenance (OpenF1, with their attribution requirements checked at publication time),
the coverage ledger summary, label-tier semantics, per-family IAA and measured WER, the
honest-reporting language for picaresca labels (section 3.6), known biases (section 6),
and the licensing posture below.

**Licensing and audio posture (candid, decision required, Q1).** Team radio audio is
FOM/F1 broadcast content; OpenF1 exposes URLs to clips on F1's own infrastructure.
Redistributing MP3s in a public dataset is the legally weakest link of the whole plan.
Options, in increasing risk:

1. **Metadata + transcripts + labels + features public; audio NOT redistributed.** The
   `clips` config carries `recording_url` and `audio_sha256`; a fetch script (shipped in
   the radiogate repo) reconstructs the audio tree locally. Transcripts are lower-risk
   derivatives with a research posture and a takedown policy. This is the recommended
   default.
2. **Gated audio**: audio files behind HF gated access with a research-use agreement.
   Middle ground; still redistribution.
3. **Full public audio**: what the current small corpus already does
   (529 MP3s, ~80 MB, on `VforVitorio/f1-strategy-dataset`). At radiogate scale
   (10,000+ clips under a branded org) this materially raises exposure; the existing
   precedent should be re-examined rather than scaled.

The recommendation is option 1, with option 2 as fallback if reproducibility pressure
demands hosted audio. Whatever the choice, the card never claims a license it does not
have: the labels and features are the project's to license (CC-BY-4.0 suggested for the
label layers), the audio is not. OpenF1's own terms (free access, attribution, and any
non-commercial constraints) must be re-verified at publication time and reflected in the
card; this document deliberately does not assert their current wording.

---

## 5. Phased roadmap (each phase is a self-contained future work item)

| Phase | Title | Contents | Exit criterion |
|---|---|---|---|
| R0 | Full-coverage ingestion and completeness ledger | Enumeration crawl (all seasons/sessions), archival filter policy, provenance + dedup columns, ledger, rate-limit budget, one-season census to validate volume estimates | Every OpenF1-listed clip for one full season is either in the tree or in the ledger with a reason; crawl is resumable |
| R1 | Transcription, quality, speakers, language | Whisper at scale over the full tree, quality flags, language ID, diarization + role classification v1 | Full-tree transcripts with quality bands; measured role-classification accuracy on a pilot GOLD slice |
| R2 | Auto-labeler v1 (core NLP families) | Clean N24 extraction to `src/nlp/` (prerequisite), bootstrap pre-labels, LF library v1, label model, annotation UI stood up, first active-learning rounds, NER schema audit + re-annotation start | Silver sentiment/intent labels with silver-vs-gold agreement measured; NER data collection running |
| R3 | Telemetry alignment and divergence engine | Claim typing, evidence windows, predicate library per claim family, divergence scoring with model counterfactuals (TireDegTCN, N06), confound controls, divergence weak labels | Divergence bands computed corpus-wide for verifiable claims; proxy-label validation checks pass or failures documented |
| R4 | Picaresca modeling and GOLD evaluation | GOLD annotation campaign (300 to 500 clips, telemetry panels, IAA), hybrid model (classifier + routed LLM judge), calibration, ablations, case studies | Calibrated p_misdirection with reliability reported; IAA published per axis; case-study set verified in-corpus |
| R5 | Corpus release and strategy feedback | HF dataset build (configs, splits, card, licensing posture executed), pinned v1/v2 revisions, additive runner fields, Rival Agent trust-weight hook + orchestrator guardrail spec | `f1stratlab/f1-team-radio-corpus` live with card; consumer-side contract documented and demoed on one replay |

Sequencing notes: R0 to R2 are independent of Part 1 and are pure corpus value (they
alone justify the initiative and unblock the gridmind LoRA corpus, Fase 2 of FUTURE.md).
R3/R4 are the research core and can slip without hurting R5's v1 release (picaresca
ships as the corpus v2 layer, Q5). The N24-to-src extraction inside R2 is the only
prerequisite touching the existing repo, and it was already planned independently.

---

## 6. Risks and limitations (candid)

1. **Truth is unobservable.** No amount of telemetry proves intent. The design measures
   divergence (hard) and infers misdirection (soft, calibrated, human-anchored). Any
   write-up that drops this distinction overclaims; the reporting rules in 3.6 exist to
   prevent that.
2. **Extreme class imbalance.** True misdirection (P3/P4/P8) is rare, plausibly under 2
   percent of claim-bearing clips. Consequences: divergence intensity is modeled as the
   primary continuous signal; precision at high confidence matters more than recall;
   evaluation slices are stratified; and reported base rates come from the random GOLD
   sample, not the enriched one.
3. **ASR errors compound.** Radio audio is compressed, noisy, and clipped; a wrong
   transcript falsifies every downstream label. Mitigations: quality gates on silver
   eligibility, measured (not assumed) WER from GOLD transcript correction, and the
   audio kept addressable so labels can always be re-derived after a better ASR pass.
4. **Broadcast selection bias.** OpenF1's team radio is the subset FOM surfaces via live
   timing, not the full radio traffic of a race weekend. The corpus inherits editorial
   selection (dramatic exchanges over routine ones), which inflates apparent
   picaresca base rates and skews sentiment. This must appear in the dataset card; no
   claim of "all F1 radio" is ever made, only "all radio OpenF1 exposes".
5. **Telemetry blind spots.** Fuel load, true tyre wear state, engine modes, and damage
   are not directly observable; predicates use proxies (pace vs model expectation,
   throttle signatures) with known confounds (traffic, track evolution, weather,
   warm-up). The confound-control list in 3.4 is a mitigation, not a cure, and each
   predicate's write-up documents its failure modes.
6. **Legal/ToS exposure on audio.** Section 4.7. The recommended posture (no audio
   redistribution) trades reproducibility convenience for safety. The existing small
   public MP3 tree is a standing decision to revisit before scaling under the
   `f1stratlab` brand.
7. **Annotator expertise and cost.** Picaresca annotation needs F1-literate humans
   reading telemetry panels; that is a scarce resource and the realistic bottleneck for
   GOLD size and active-learning throughput. The stopping criterion and tier system are
   designed so the corpus degrades gracefully with budget, but IAA on intent may simply
   come out low (that outcome is reportable, not hideable).
8. **Multilinguality and model coverage.** Auto-labels only apply where the label models
   are competent (English); other languages are transcript-only until justified.
9. **Leakage discipline.** Telemetry LFs must never leak into the evaluation of models
   that consume telemetry features (the ablations in 3.6 depend on clean separation);
   temporal splits and frozen GOLD mitigate, and the QC dashboard checks for
   quality-correlated label bias.
10. **Regulation drift (2026).** New cars, new energy management, new vocabulary; models
    trained on 2022 to 2025 radio and telemetry will drift. The corpus versioning and
    per-season distributions make drift measurable; the 2026 slice is reserved for
    exactly that.

---

## 7. Open questions

1. **Audio posture (Q1, blocking R5)**: metadata + fetch-script (recommended), gated
   audio, or full public audio? And should the existing 529 public MP3s be revisited
   when the dataset moves to the `f1stratlab` org?
2. **Repo topology (Q2, blocking kickoff)**: radiogate as a submodule of the core repo
   (the stated preference) or an independent public repo (the visibility
   recommendation)? The ecosystem rule ("standalone artifact = independent repo")
   argues for independent.
3. **Annotation resourcing (Q3, sizes R2/R4)**: who are annotators 2 and 3, what is the
   realistic hours budget, and is Argilla (or Label Studio) acceptable as the UI?
4. **v1 scope (Q4)**: all session types from day one, or race-only v1 with
   practice/quali/sprint in v1.x? Full coverage is the stated goal; race-only is a
   legitimate de-risking step.
5. **Picaresca in v1 or v2 (Q5)**: ship the corpus (R0 to R2) as v1 without picaresca
   labels and add them as the v2 layer after R4, or hold the release until both are
   ready? Recommendation: v1 first (it unblocks gridmind).
6. **LLM budget (Q6)**: OpenAI spend ceiling for the judge (gpt-4.1-mini class at 10 to
   20 percent routing), vs accepting LM Studio local throughput and a quality haircut?
7. **GOLD acceptance thresholds (Q7)**: agree in advance that divergence IAA >= 0.65 is
   the bar, and that a low intent IAA (0.3 to 0.5) is reported as a finding rather than
   triggering rework loops.
8. **Rival Agent coupling (Q8)**: does the TFM Rival Agent plan to consume
   p_misdirection as an input (which creates a timing dependency on R3/R4), or does it
   ship without it and adopt it later? (Per the standing master-planning rule: to be
   revisited against the actual state of the master when this is picked up.)
9. **NER schema (Q9)**: appetite for collapsing the 9 entity types to ~6 based on the
   confusability audit, given the annotation cost of keeping all 9 at acceptable F1?
10. **Case-study curation (Q10)**: a short working session to list remembered candidate
    radio-games incidents from 2023 to 2025, to be verified against the corpus before
    any of them is cited in write-ups.

---

## 8. Internal references

- `src/data_extraction/openf1/radio_dataset_builder.py`: ingestion, lap mapping, filters,
  retry policy, slug rules, `OUTPUT_SCHEMA` / `RCM_OUTPUT_SCHEMA`.
- `scripts/build_radio_dataset.py`, `scripts/upload_radio_corpus.py`: build + publish.
- `src/nlp/radio_runner.py`: `RadioPipelineRunner`, `WhisperTranscriber`, transcript
  cache design.
- `data/models/nlp/pipeline_config_v1.json`: current model set, labels, latency.
- `notebooks/nlp/N17-N24, N33`: the NLP lineage (labeling, transcription, sentiment,
  intent, NER, RCM parser, unified pipeline, builder prototype).
- Memory notes: `project_future_vision` (radiogate scope, naming, branding rule),
  `project_radio_ingestion_plan` (original static-build design),
  `project_track_a_radio_consumption` (runner + cache + degradation contract),
  `project_nlp_src_wrapper` (pending N24 extraction).
- Strategy-model counterfactual sources: TireDegTCN (N07-N10), lap-time model (N06),
  and the calibrated classifiers (N11-N16), under `data/models/`.
