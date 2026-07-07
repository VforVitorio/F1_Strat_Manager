# AUDIT NLP-RADIO - Team-radio NLP pipeline and Race Control Message processing

**Auditor:** Fable 5 · **Date:** 2026-07-07 · **Repo:** `F1_Strat_Manager` (read-only pass, no code changed)
**Scope:** the NLP subsystem that feeds the Radio Agent (N29): Whisper transcription and the replay-time corpus consumer (`src/nlp/radio_runner.py`), sentiment (N20 RoBERTa), intent (N21 SetFit + ModernBERT), NER (N22 BERT-large BIO), the RCM rule-based parser (N23) and the unified pipeline (N24, now living inside `src/agents/radio_agent.py`), the RCMContextResolver path (`race_situation_agent.py`), the legacy `src/nlp/` jupytext exports, and the planned `src/nlp/pipeline.py` extraction. **RCM processing is first-class in this audit** (section 5), not a footnote.
**Hard constraints honored in every remedy:** plan only, no code; UNTOUCHABLE (additive entry points only, duplicate before modifying): `src/agents/` internals, `scripts/run_simulation_cli.py`, `notebooks/**`, `legacy/**`; LLM = OpenAI / LM Studio, never Anthropic.
**Inputs read:** `src/nlp/*` (all 6 files + README), `src/agents/radio_agent.py`, `src/agents/race_situation_agent.py` (RCM override path), `src/agents/strategy_orchestrator.py` (routing + coercers), `src/agents/pit_strategy_agent.py` (SC override consumption), `src/agents/rules/nlp_rules.py`, `scripts/run_simulation_cli.py` (radio wiring, read-only), `scripts/bench_nlp_pipeline_cpu.py`, `src/data_extraction/openf1/radio_dataset_builder.py`, `data/models/nlp/pipeline_config_v1.json` + every `model_config.json` under `data/models/nlp/`, `tests/test_agents.py` + `tests/test_smoke.py` (RCM regression), `documents/audits/AUDIT_ML_AGENTS_EVAL.md` (E-12, eval package), `AUDIT_TESTING_QA.md` (T-11), `AUDIT_P2_LOADING.md` (F-01/F-02), `AUDIT_2026_REG_CONCEPT_DRIFT.md` (F-12), `documents/research/RADIOGATE_DECEPTION_AND_AUTOLABELING.md` (section 4.5), memory files `project_nlp_src_wrapper`, `project_rcm_context_resolver`, `project_n29_radio_agent`, `project_track_a_radio_consumption`.

---

## 1. Framing

The NLP / team-radio pipeline is the one code subsystem that never received a dedicated audit. The ML-eval audit (#205) covered the strategy predictors and agents and explicitly parked NLP as finding E-12 ("evaluation frozen at notebook-era", size S); the radiogate research design covers the FUTURE corpus, auto-labeler and picaresca work. Neither assessed the CURRENT pipeline's per-stage quality, its robustness to bad inputs, or the correctness of the RCM path that the Qatar-2025 fix made load-bearing. This audit does that: assessment first, then a prioritized, additive-only improvement plan.

Why it matters strategically: radio alerts (intent PROBLEM / WARNING) and the RCM-derived `sc_currently_active` flag are direct inputs to the orchestrator's Layer-1 MoE routing (`strategy_orchestrator.py:475-537`). A noisy or silently wrong NLP channel does not just degrade a dashboard panel; it changes which agents run, which distributions the Monte Carlo receives, and whether the pit decision gets the SC override that made the Qatar demo the thesis's signature moment.

**Boundaries with sibling efforts (read before converting to issues):**

- **AUDIT_ML_AGENTS_EVAL (#205)** owns the shared `src/strategy/eval/` package and the report/versioning conventions. This audit's eval harness (section 6) is a module inside that package, not a second framework. Its E-12 (alert-precision probe) is absorbed and expanded here.
- **AUDIT_TESTING_QA** owns PR gates. Its T-11 (RCM parser units + pipeline glue with stubs + one data-tier label-stability probe) is the test-side twin of this audit's harness; fixtures are shared, division of labor identical to the ML-eval audit: tests gate PRs, the harness gates claims.
- **AUDIT_P2_LOADING** owns the load-time problems: F-01 (the 3 NLP models load at module import, 30.3 s, `radio_agent.py:348-367`) and F-02 (ship `transcripts.json` in the HF dataset). This audit does not re-plan them; Phase 4 coordinates with F-02 because the transcript cache schema changes there.
- **AUDIT_2026_REG_CONCEPT_DRIFT F-12** owns the 2026 timing/trigger; section 5.4 states only what the NLP layer specifically owns.
- **radiogate** (`documents/research/RADIOGATE_DECEPTION_AND_AUTOLABELING.md` section 4.5) owns the NER retraining at corpus scale (schema merge, LLM-assisted re-annotation to 3-5k examples, GLiNER vs BERT-BIO head-to-head, acceptance span-F1 >= 0.70). This audit owns the CURRENT pipeline's mitigation of the weak NER and the acceptance bars the refresh must clear.

---

## 2. Executive summary

The transport layer is the strong part: `src/nlp/radio_runner.py` is a well-designed, well-documented replay-time bridge (atomic JSON cache keyed by normalized path, stale-model invalidation, graceful degradation on every I/O failure). The inference layer and the RCM layer are where the problems live, and they cluster into three stories:

**1. Nothing is measured.** Every stage metric is frozen at notebook-era, and where measured the numbers are weak: NER span-F1 0.4151 with four of nine entity classes effectively dead (incident 0.0, track condition 0.0, situation 0.06, strategy instruction 0.11 per `data/models/nlp/ner_v1/model_config.json`), intent weighted-F1 0.593 (`intent_setfit_modernbert_v1/model_config.json`), sentiment 87.5% accuracy on 530 messages (`src/nlp/README.md:70`). The one number that actually matters for the system, alert precision/recall on the live pipeline, has never been computed. No transcription quality is measured or even captured: Whisper's per-segment confidence signals are discarded at `radio_runner.py:166-177`, and empty transcripts flow into the models unguarded (`radio_agent.py:977`), so a hallucinated or empty string can mint a high-confidence PROBLEM alert that flips MoE routing.

**2. The RCM path has a correctness hole exactly where the Qatar fix lives.** `_sc_active_from_rcm` (`race_situation_agent.py:1180-1225`) is stateless per lap window, and the CLI feeds it only the RCMs of the current lap (`radios_for_lap(lap_num)`, exact-match filter, `radio_runner.py:319-349`; injection at `run_simulation_cli.py:1894-1897`). Race control announces SAFETY CAR DEPLOYED once; on the second and later laps of a multi-lap SC stint the window contains no SC message, so `sc_currently_active` silently drops back to False while the SC is still on track. The celebrated Qatar V7 fix holds on the deploy lap and evaporates afterwards. Additionally the parser has no branch for the standard end-of-neutralization phrasing ("SAFETY CAR IN THIS LAP" matches neither "ENDING" nor "IN THE PIT LANE" at `radio_agent.py:573-584` and classifies as DEPLOYED), DOUBLE YELLOW flags fall through to OTHER (`:586-589`), and the PENALTY routing branch in the orchestrator is dead code because no producer ever emits an `intent` of PENALTY (`strategy_orchestrator.py:524` vs `radio_agent.py:241` and `:870-877`).

**3. The production pipeline is trapped in an untouchable file, while a contradictory dead twin squats in `src/nlp/`.** The real N24 pipeline (`run_pipeline` / `run_rcm_pipeline`) lives inside `src/agents/radio_agent.py:529-622`, which is frozen. Meanwhile `src/nlp/pipeline.py` is a week-4 jupytext export with broken `../../outputs/week4/` paths and two mutually contradictory sentiment label orders inside one file (`:90` vs `:280`), and it is still imported by the legacy experta chain (`src/agents/rules/nlp_rules.py:24`). The extraction planned in memory `project_nlp_src_wrapper` never happened. Config hygiene is similar: `pipeline_config_v1.json` ships absolute `c:\Users\victo\...` paths to the HF dataset, and the sentiment checkpoint directory has no model_config at all, so the label order exists only as hardcoded tuples in code.

Plan: 5 phases. Phase 1 is a cheap truth-and-hygiene pass (including a possible live bug: SetFit `predict_proba` column-order vs the hardcoded intent tuple, which if real means PROBLEM/WARNING confidences are today reading ORDER/QUESTION's probabilities). Phase 2 builds the NLP eval module inside the shared `src/strategy/eval/` package. Phase 3 is the RCM correctness sprint: parser coverage extension plus a stateful RaceControlState tracker that makes the SC override survive the whole neutralization. Phase 4 hardens the pipeline (input gates, Whisper QA capture, calibration measurement). Phase 5 is the model refresh boundary handed to radiogate / 2026-reg with acceptance bars set here.

---

## 3. Current state inventory

### 3.1 What runs in production today

| Stage | Artifact | Where it executes | Headline quality (notebook-era, frozen) |
|---|---|---|---|
| Transcription | Whisper `turbo`, JSON cache per GP | `src/nlp/radio_runner.py` (`WhisperTranscriber`, `RadioPipelineRunner`) | Never measured (no WER reference set); quality signals discarded |
| Sentiment | RoBERTa-base, 3-class, `bert_sentiment_v1/best_roberta_sentiment_model.pt` | `radio_agent.py:408-428` (`predict_sentiment`) | 87.5% accuracy on 530 messages (README claim; no config on disk) |
| Intent | SetFit + ModernBERT, 5 labels, `intent_setfit_modernbert_v1/` | `radio_agent.py:431-442` (`predict_intent`) | weighted F1 0.593 (`model_config.json`) |
| NER | BERT-large CoNLL-03 BIO, 9 entity types, `ner_v1/bert_bio_v1/` | `radio_agent.py:445-526` (`predict_entities`) | span F1 0.4151; per-class F1: 4 of 9 classes at or near 0 |
| RCM parser | Deterministic rules (N23) | `radio_agent.py:563-622` (`_classify_rcm_event`, `run_rcm_pipeline`) | Never evaluated against the corpus; coverage matrix in 5.1 |
| Unified pipeline | `run_pipeline(text)` chaining the three models | `radio_agent.py:529-556` | Latency GPU mean 43.7 ms / P95 45.8 ms per `pipeline_config_v1.json` (memory quotes 47.8 / 59.4; divergence unresolved) |
| RCM context override | `_sc_active_from_rcm` + `sc_currently_active` propagation | `race_situation_agent.py:70-80, 1149-1173, 1180-1225` | One regression test (`tests/test_agents.py:278-293`, `tests/test_smoke.py:59-91`); stateless (finding NR-02) |
| Corpus consumer | `radios.parquet` + `rcm.parquet` per GP (2025 only), MP3s from HF | `radio_runner.py`, wired in CLI (`run_simulation_cli.py:1626-1663`) and Arcade (`src/arcade/strategy.py:427, 542`) | 48 parquets, 529 clips, 2025 season only |

Consumption chain: corpus rows -> `radios_for_lap(lap)` dicts -> `RaceState.radio_msgs` / `rcm_events` -> orchestrator coercers (`strategy_orchestrator.py:953-`) -> N29 stages 1-2 (deterministic NLP + alerts) -> Layer-1 routing (`:517-535`) and, via N27's override, `sc_currently_active` into N28's pit logic (`pit_strategy_agent.py:886, 1006-1010`) and the MC's SC-reactive scoring.

### 3.2 What sits dead or half-done

| Item | State | Evidence |
|---|---|---|
| `src/nlp/pipeline.py`, `sentiment.py`, `ner.py`, `radio_classifier.py` | Legacy week-4 jupytext exports; broken paths; kept "for reference only" | `src/nlp/README.md:61-73`; `pipeline.py:84-86` (`../../outputs/week4/models/...`) |
| Legacy import chain into the live package | `src/agents/rules/nlp_rules.py:24` imports `src.nlp.pipeline` (guarded try/except); `src/agents/strategy_agent.py:45` imports the rules engine | `test_radio_rules` (`nlp_rules.py:120-137`) would crash on the week-4 paths if invoked |
| Planned `src/nlp/pipeline.py` extraction from N24 | Never executed; the production pipeline landed inside `radio_agent.py` instead, which is now untouchable | memory `project_nlp_src_wrapper`; `src/nlp/README.md:13-16` |
| Modern-NER attempts | GLiNER zero-shot F1 0.016, GLiNER fine-tuned 0.068, NuNER 0.0 recorded in the config; numbers low enough to suggest the evaluation alignment itself was off | `ner_v1/model_config.json` "results" |
| NLP evaluation of any kind post-notebooks | Nothing reproducible; N24's latency block frozen in `pipeline_config_v1.json` | ML-eval E-12; Testing T-11 (zero NLP tests today) |

---

## 4. Findings register (P0-P3)

| ID | P | Finding | Why / risk | Size |
|---|---|---|---|---|
| **NR-01** | **P0** | **No reproducible evaluation exists for any NLP stage, and the stages measured at notebook-era are weak.** NER span-F1 0.4151 with dead classes (`ner_v1/model_config.json` per-class table), intent weighted-F1 0.593, sentiment 87.5% on n=530; alert precision (the number routing actually depends on) never computed; no regression protects against silent model-file or dependency drift (the SetFit shim at `radio_agent.py:297-302` shows the deps already moved under the pipeline once) | Radio alerts steer MoE routing (`strategy_orchestrator.py:517-525`); the system's radio channel quality is unknown and unfalsifiable. Absorbs ML-eval E-12 and pairs with Testing T-11 | **M** (harness module) |
| **NR-02** | **P0** | **`sc_currently_active` is stateless and drops mid-SC-stint.** `_sc_active_from_rcm` (`race_situation_agent.py:1180-1225`) sees only the current lap's RCMs (`radios_for_lap` exact-lap filter, `radio_runner.py:339-346`; fresh empty `rcm_events` each lap, `run_simulation_cli.py:1894-1897, 1945-1957`). SC deployment is announced once; laps 2..N of the stint carry no SC RCM, so the override releases while the SC is still out. Compounding it, the standard end-of-neutralization message ("SAFETY CAR IN THIS LAP") matches neither "ENDING" nor "IN THE PIT LANE" (`radio_agent.py:573-584`) and classifies as SAFETY_CAR_DEPLOYED, i.e. the parser has no reliable full-SC release event at all in the real message grammar | The Qatar-V7 fix (the thesis's flagship correction) generalizes only to the deploy lap. On the following SC laps N27 reverts to the future-SC prior, N28's stint safeguard re-arms, and the system can re-commit the exact STAY_OUT bias the resolver was built to kill. The pit-delta advantage of stopping under SC persists for the whole neutralization, not one lap | **M** |
| **NR-03** | **P1** | **RCM parser coverage gaps, enumerable and corpus-verifiable** (full matrix in 5.1): DOUBLE YELLOW flag falls to OTHER (`radio_agent.py:586-589` handles only exact `_FLAG_MAP` keys + `flag == "YELLOW"`); track-limit deletions, investigations ("NOTED" / "UNDER INVESTIGATION"), pit-entry/exit status, weather advisories, session start/suspend/resume, BLACK AND WHITE flag all unclassified; `event.scope == "Sector"` is case-sensitive while the corpus builder passes OpenF1 casing through and `radio_runner._rcm_row_to_dict` documents uppercase values (`radio_runner.py:640-666`), so YELLOW vs YELLOW_FLAG_SECTOR resolution depends on data source | Race-control phrasing is semi-structured and drifts by season; every unhandled family is invisible to alerts, to the resolver and to the LLM synthesis. Double yellows are precisely the highest-danger flag state and currently classify as OTHER (not in `_SAFETY_FLAGS`, no alert) | **S-M** |
| **NR-04** | **P1** | **The PENALTY routing branch is dead and RCM alerts never influence routing.** `_decide_agents_to_call` checks `alert_intents & {"PENALTY", "WARNING"}` (`strategy_orchestrator.py:524`) but (a) the intent label set has no PENALTY (`radio_agent.py:241`), (b) RCM alert dicts carry `event_type`, not `intent` (`radio_agent.py:870-877`), so their `a.get("intent","")` is always "". Consequence: TIME_PENALTY, RED_FLAG, YELLOW and collision RCMs affect routing ONLY via `sc_currently_active`; a red flag or a penalty against our car triggers no N28 re-evaluation and no N30 regulation lookup | The docstring at `:505-506` promises "Radio carries a FIA-facing alert (PENALTY...) -> regulation lookup"; production cannot deliver it. Documented behavior and actual behavior diverge in the safety-critical path | **S** |
| **NR-05** | **P1** | **No empty/degenerate-transcript gate.** Missing MP3s and Whisper failures cache `text=""` by design (`radio_runner.py:559-584`) and rows are always emitted (`:319-335` docstring); `run_radio_agent` pipes every text into the three models unconditionally (`radio_agent.py:977`). Classifier output on empty/near-empty strings is undefined behavior that can land in `alert_intents`; there is no minimum-length, no NLP-confidence threshold, and no "unusable transcript" marker in the output schema | False PROBLEM/WARNING alerts from garbage input force-activate N28+N30 (`strategy_orchestrator.py:518-519, 527-528`) and pollute the LLM synthesis prompt. The claimed "no usable text warning" behavior exists nowhere in `radio_agent.py` | **S** |
| **NR-06** | **P1** | **Whisper quality signals are discarded; transcription QA impossible.** `WhisperTranscriber.transcribe` keeps only `{text, duration_s, model}` (`radio_runner.py:166-177`); `avg_logprob`, `no_speech_prob`, `compression_ratio` per segment are dropped; failure cache entries are indistinguishable from silence (`:578-584`); `language="en"` hardcoded (`:169`); no WER/CER on any reference set has ever been computed for the F1-radio domain (noisy, jargon-heavy, clipped audio is Whisper's classic hallucination trigger) | Hallucinated transcripts are the single most likely source of false alerts, and today they are invisible: no flag reaches N29, the panel, or the eval data. Cache schema must gain quality fields (coordinate with Loading F-02, which ships these caches to HF) | **M** |
| **NR-07** | **P1** | **NER is surfaced as trustworthy signal despite 4 of 9 classes being dead** (per-class F1: B-INCIDENT 0.0, B-TRACK_CONDITION 0.0, B-SITUATION 0.06, B-STRATEGY_INSTRUCTION 0.11). Entities flow verbatim into alerts (`radio_agent.py:867`) and the synthesis prompt. The retrain is radiogate's (section 4.5: schema merge, 3-5k re-annotation, GLiNER vs BIO, bar >= 0.70); what the CURRENT pipeline owns is not pretending: suppress or mark low-reliability classes, and re-check the anomalous GLiNER-zero-shot 0.016 result (likely an evaluation-alignment bug, worth 1 hour before the paper repeats "modern NER failed") | Downstream consumers (LLM synthesis, legacy rules, dashboards) treat `incident` / `track condition` entities as real; they are statistically noise today | **S** (mitigation; retrain external) |
| **NR-08** | **P1** | **Possible live bug: intent confidence may be read from the wrong column.** `predict_intent` indexes `model.predict_proba([text])[0]` with the hardcoded tuple order INFORMATION, PROBLEM, ORDER, WARNING, QUESTION (`radio_agent.py:431-442, 241, 370`). SetFit's proba columns follow the head's `classes_`; if the sklearn head was fit on string labels, `classes_` is alphabetical (INFORMATION, ORDER, PROBLEM, QUESTION, WARNING), which swaps exactly the two alert-relevant pairs: PROBLEM<->ORDER and WARNING<->QUESTION confidences | If real: every alert confidence shown in panels, logged to JSON and given to the LLM is wrong today, and any future confidence threshold (NR-05 remedy) would be built on the wrong numbers. 5-minute check: unpickle `model_head.pkl`, inspect `classes_`; property test: argmax(predict_proba) label == predict() label on the golden set | **S** (verify first) |
| **NR-09** | **P2** | **Legacy `src/nlp/` modules are contradictory, dead and still importable from the live package.** `pipeline.py` carries two conflicting sentiment orders in one file (`:90` `["positive","negative","neutral"]` vs `:280` `['positive','neutral','negative']`), both different from production `("negative","neutral","positive")` (`radio_agent.py:242`); week-4 paths broken; imported by `src/agents/rules/nlp_rules.py:24` (whose own `test_radio_rules` would crash). The planned N24 extraction (memory `project_nlp_src_wrapper`) never landed; the production pipeline is locked inside the untouchable `radio_agent.py` | A future contributor grabbing `src/nlp/pipeline.py` (the obvious name) gets broken, contradictory code; the real pipeline has no importable home outside agent internals, which blocks the eval harness, the backend and radiogate from reusing it cleanly | **M** (extraction plan, 7.1 Phase 1/2) |
| **NR-10** | **P2** | **Config hygiene:** `pipeline_config_v1.json` ships absolute `c:\Users\victo\...` paths (`:6, :15, :26`) and is distributed via HF (`data_cache.py:80, 98`); `bert_sentiment_v1/` has no model_config.json at all (label order, base model and metric exist only in code and README); the latency block (43.7/45.8 ms) disagrees with the memory-quoted 47.8/59.4 with no provenance for either | The pipeline config is the Strategy Agent's declared source of truth; today it is non-portable and incomplete. Label-order provenance is exactly how the NR-09 class of bug is prevented | **S** |
| **NR-11** | **P2** | **Contract drift across the radio/RCM dict shapes.** `RCMEvent.racing_number` is typed `Optional[str]` (`radio_agent.py:162`) but the runner emits `Optional[int]` (`radio_runner.py:652-655`); entity labels are emitted lowercase-with-spaces ("strategy instruction", `radio_agent.py:497`) while the legacy rules engine matches uppercase-underscore keys and a dict-of-lists shape (`nlp_rules.py:42, 51`); `pipeline_config_v1.json` output_schema names `entities: list[{text,label}]` but no consumer-facing contract doc exists | Every new consumer (SPA migration #25, pit-wall #281, radiogate) will re-derive the shape from code; one contract doc + one schema test ends it | **S** |
| **NR-12** | **P2** | **Corpus coverage is 2025-only and silent.** 48 parquets / 529 clips for 2025 (Track-A memory); replaying any 2023/2024 GP degrades to no radios with a log-level warning only (`radio_runner.py:353-373`); no coverage report exists (which GPs, radios per driver, transcript-empty rate per GP) | Users and eval runs cannot distinguish "no radio activity" from "corpus missing"; the eval harness needs the coverage report anyway to normalize alert-rate metrics | **S** |
| **NR-13** | **P2** | **Alert semantics are narrower than believed and unmeasured end-to-end:** alerts = intent in {PROBLEM, WARNING} plus RCM `_SAFETY_FLAGS` only (`radio_agent.py:240, 90-99, 859-878`); the LLM `corrections` channel (`:727-747`), designed to flag NLP misclassifications, has never been evaluated for hit-rate against ground truth and is presentation-only downstream | The alert set is the Radio Agent's entire influence on strategy; its precision/recall and the corrections channel's value are unmeasured claims (feeds the E-07 N29-knockout ablation arm) | **S** (once NR-01 harness exists) |
| **NR-14** | **P3** | **Latency is fine but unconsolidated.** Per-message GPU cost ~44-48 ms; messages processed serially per lap (`radio_agent.py:977-978`); N29 runs once per lap alongside LLM calls that cost seconds, so NLP is not on the critical path; CPU numbers exist only via `scripts/bench_nlp_pipeline_cpu.py` runs, not in any versioned report; `datetime.utcnow()` deprecation at `radio_agent.py:547, 906` | Fold the CPU/GPU bench outputs into the shared eval report format (ML-eval E-14 pattern); batch inference only if the coverage report shows many-radio laps actually occur | **S** |
| **NR-15** | **P3** | **2026 vocabulary/entity/message-type gap, NLP-owned slice** (cross-ref 2026-reg F-12, do not duplicate): NER/intent training data contain no Audi/Cadillac/Madring/override-mode vocabulary; the RCM grammar may add energy-management / override-mode directives; Whisper prompt-bias for new proper nouns unexplored | Radio signal quality degrades exactly when 2026 strategy uncertainty peaks; the refresh executes under radiogate + F-12, but the acceptance bars and the harness that judges them are this audit's Phase 2 deliverable | **S** (planning only) |

---

## 5. RCM processing deep-dive (first-class)

### 5.1 Parser coverage matrix (`_classify_rcm_event`, `radio_agent.py:563-601`)

Handled today: SafetyCar category (DEPLOYED / VIRTUAL / IN THE PIT LANE / ENDING keywords), flags RED / GREEN / CLEAR / BLUE / CHEQUERED via `_FLAG_MAP`, YELLOW (+ Sector scope), DRS ENABLED/DISABLED, COLLISION/CONTACT/INCIDENT, RETIRED, PENALTY (keyword only). Everything else returns OTHER.

| RCM family (real 2023-2025 grammar) | Parser result today | Strategy relevance | Verdict |
|---|---|---|---|
| SAFETY CAR DEPLOYED / VSC DEPLOYED | SAFETY_CAR_DEPLOYED / VIRTUAL_... | pit-delta window opens | OK |
| "SAFETY CAR IN THIS LAP" (standard SC end phrasing) | **SAFETY_CAR_DEPLOYED** (no keyword hit) | restart preparation, overtaking resumes | **Wrong class; no full-SC release event exists in practice** (NR-02) |
| VIRTUAL SAFETY CAR ENDING | VIRTUAL_SAFETY_CAR_ENDING | release | OK |
| DOUBLE YELLOW flag | **OTHER** (`_FLAG_MAP` miss, `flag == "YELLOW"` miss) | highest-danger local flag; SC precursor | **Miss** (NR-03) |
| BLACK AND WHITE flag (driver warning) | OTHER | driving-standards warning, penalty precursor | Miss (low) |
| Track-limits lap-time deletions | OTHER | penalty-risk signal for us/rivals | Miss |
| "... UNDER INVESTIGATION" / "... NOTED" | OTHER (unless INCIDENT keyword present) | penalty anticipation | Miss |
| TIME PENALTY / STOP-GO decisions | TIME_PENALTY, `is_alert=False` | forced pit or post-race delta; should route N30 | Classified but inert (NR-04) |
| PIT LANE / PIT ENTRY / PIT EXIT CLOSED-OPEN | OTHER | blocks/permits PIT_NOW outright | **Miss; resolver-class item** |
| Weather advisories (LIGHT RAIN, RISK OF RAIN...) | OTHER | compound switch anticipation | Miss (feeds N26/weather someday) |
| Session start/suspend/resume, standing restart | OTHER (RED flag itself maps to RED_FLAG) | red-flag tyre-change rule, restart procedure | Partial |
| Scope=Sector yellow | YELLOW_FLAG_SECTOR only if `scope == "Sector"` exactly | severity discrimination | Case-fragile (NR-03) |

The 2025 `rcm.parquet` corpus (all GPs) makes this matrix cheaply verifiable: classify every row, report the event-type distribution and the OTHER-rate per category (Phase 2 deliverable). Season-to-season phrasing drift becomes visible the same way.

### 5.2 RCMContextResolver: from stateless override to a race-control state machine

What exists (correct but narrow): deploy/release event sets (`race_situation_agent.py:70-80`), release-beats-deploy ordering within one window (`:1223-1225`), `sc_prob_3lap` forced to 1.0 + `sc_currently_active` flag (`:1149-1173`), consumption in routing (`strategy_orchestrator.py:533-535`), pit safeguard override (`pit_strategy_agent.py:1006-1010`) and MC SC-reactive scoring. One regression test each in `tests/test_agents.py:278` and `tests/test_smoke.py:59`.

What v2 must own (the resolver becomes the single authority on race-control state):

1. **Persistence (NR-02, the P0):** a stint-level state machine, fed once per lap with that lap's classified RCMs, holding state across laps: GREEN -> SC / VSC (deploy) -> ENDING announced -> GREEN (restart). Emits `sc_currently_active` for every lap of the neutralization, plus `laps_under_sc` (the pit-window urgency decays as the queue forms) and `restart_imminent` (ending message seen, next lap is the restart).
2. **VSC vs full SC differentiation:** both currently collapse into one boolean and one MC bonus (`SC_PIT_BONUS 8.0`, `strategy_orchestrator.py:545-549`); the VSC pit delta is materially smaller and there is no queue effect. The resolver should expose `neutralization_type` so the MC constants sensitivity work (ML-eval E-04e) can split the bonus. Changing the MC constant itself stays in ML-eval's court.
3. **Release-event grammar:** classify "IN THIS LAP" as ending; treat green flag / green-light RCMs after an SC phase as the authoritative restart confirmation; define resolution for overlapping events (yellow sectors during VSC, SC deployed while VSC active: escalation wins, releases require the matching type).
4. **Red flag:** RED_FLAG is already classified and alertable, but nothing overrides strategy state. v2 owns `red_flag_active`, free-tyre-change awareness (the regulation RAG query N30 should receive), and standing-restart state. (Extension already named in thesis future-work; memory `project_rcm_context_resolver`.)
5. **Penalty applied to our car:** event exists (TIME_PENALTY), routing dead (NR-04). v2 emits `pending_penalty` so N28 can fold "serve under SC / combine with the stop" into the pit decision and N30 queries the relevant article.
6. **Pit lane closed/open:** hard veto on PIT_NOW while closed; the strongest possible override, currently invisible (5.1).

**Placement under the untouchable constraint:** the seam already exists by design: `_sc_active_from_rcm` accepts pre-classified dicts carrying `event_type` as its cheap path (`race_situation_agent.py:1186-1189` docstring). A new additive module (proposed `src/nlp/rcm_state.py`, consistent with the parser living in the NLP layer; final home is open question 2) implements the tracker; editable callers (Arcade `src/arcade/strategy.py`, the backend, the P4 CLI duplicate) construct it once per run and pass enriched, pre-classified events + state flags through `lap_state`. `radio_agent.py`, `race_situation_agent.py` and the orchestrator remain untouched: they already consume `event_type` dicts and `lap_state['sc_currently_active']` (`pit_strategy_agent.py:886`). The frozen CLI keeps its current one-lap behavior until the P4 duplicate lands; the state tracker is what the duplicate wires in.

### 5.3 How RCM feeds the models and routing (verified paths)

- **RCM vs the N13/N14 SC prior:** the hard override exists and is correctly placed: the LightGBM prior predicts FUTURE SC; `_run_core` post-processes it to 1.0 when RCM confirms deployment (`race_situation_agent.py:1149-1173`, comment states the rationale). The gap is not the override's existence but its one-lap lifetime (NR-02) and its blindness to VSC-vs-SC and to non-SC families (5.2). The prior itself is never fed RCM-derived features, which is fine (it is a soft prior by design, per N13/N14).
- **Routing:** only two RCM-derived signals reach `_decide_agents_to_call`: radio-intent alerts and `sc_currently_active`. The `event_type`-shaped alerts are structurally invisible to it (NR-04).
- **Entry points:** `run_rcm_pipeline(event)` (`radio_agent.py:604-622`) is pure and importable; the eval harness tests the parser through it and `_classify_rcm_event` directly, the same pattern `tests/test_agents.py` already uses.

### 5.4 2026 message types (NLP-owned slice of #189 F-12)

New-era candidates the RCM layer owns: override-mode / energy-deployment directives (the 2026 power-unit racing rules), revised DRS/manual-override phrasing, any new neutralization procedure wording, plus new proper nouns (Audi, Cadillac, Madring) in driver-scoped messages. Remedy shape: the Phase 2 corpus-coverage report re-run on the first 2026 GPs becomes the drift detector (OTHER-rate spike = new grammar arrived); parser + resolver extension is then a data-driven S task. Do not pre-implement guessed 2026 grammar.

---

## 6. NLP eval harness design (module inside `src/strategy/eval/`, shared with #205)

One new module family, `src/strategy/eval/nlp_eval` (name per ML-eval section 5 conventions; console entry `f1-eval nlp`), consuming the same report/versioning substrate (era tags, artifact hashes, dataset snapshot in every header). Deterministic goldens live in `tests/eval/` with Testing-audit tier markers; heavy runs write `documents/eval_reports/` markdown/CSV.

| Sub-concern | Measures | Inputs |
|---|---|---|
| `stage_metrics` | Sentiment/intent per-class P/R/F1 + confusion; NER span-F1 per class; regeneration of every `model_config` headline within tolerance | frozen checkpoints + the N17/N21/N22 labeled sets (read notebooks' data artifacts, never the notebooks) |
| `label_contracts` | predict_proba column-order property test (NR-08); label-order provenance asserted against config v2 (NR-10); output-schema validation (NR-11) | golden message bank |
| `rcm_coverage` | classify all `rcm.parquet` rows (2025, later 2026): event-type distribution, OTHER-rate per category, flag/scope casing audit, SC deploy->end sequence extraction per race | corpus parquets |
| `resolver_eval` | replay each SC/VSC-containing 2025 race through the state tracker: per-lap `sc_currently_active` trace vs FastF1 TrackStatus ground truth (the independent oracle); Qatar V7 becomes one golden among many | corpus + FastF1 track status |
| `transcript_qa` | empty-rate per GP, duration/quality-signal distributions (post NR-06 capture), WER/CER on a small hand-verified reference set (~50 clips) | transcript caches + MP3s |
| `alert_eval` | alert precision/recall on a labeled transcript set (per intent class, per RCM family); corrections-channel hit-rate (NR-13); feeds the E-07 N29-knockout arm | labeled set (open question 4) |
| `latency` | fold `bench_nlp_pipeline_cpu.py` output into the versioned report format (CPU + GPU rows) | existing bench |

Untouchable-boundary technique: everything imports the public pure functions (`run_pipeline`, `run_rcm_pipeline`, `_classify_rcm_event`, `_sc_active_from_rcm`) exactly as the existing tests do; no agent file changes.

---

## 7. Phased, chunkable plan (each phase = one GitHub sub-issue set)

Ordering rationale: cheap truth checks first (one may reveal a live bug), the harness before the RCM sprint's acceptance tests, robustness lifts after the harness can measure them, model refresh last and mostly external. Phase 3 can start in parallel with Phase 2 (the resolver design does not need the harness, only its verification does).

**Phase 1 - Truth and hygiene pass (S)**
- Verify NR-08: inspect the SetFit head's `classes_`; if swapped, file the bug issue (issue-first rule) and fix via the additive pipeline home (below), since `radio_agent.py` is frozen.
- Establish sentiment label-order provenance from the N20 artifacts once; record it.
- `pipeline_config_v2.json`: relative paths, label orders for all three models, latency provenance (resolve 43.7 vs 47.8), sentiment model_config added next to the checkpoint; keep v1 for compatibility.
- Quarantine the legacy exports: move `src/nlp/{pipeline,sentiment,ner,radio_classifier}.py` to `legacy/` (README already marks them delete-ready; `nlp_rules.py`'s guarded import degrades to `radio_nlp = None` untouched).
- Write the radio/RCM contract doc (dict shapes, label casings, alert semantics) + a schema test (NR-11).

**Phase 2 - NLP eval harness (M)**
- Build `nlp_eval` per section 6 on the shared `src/strategy/eval/` substrate (#205 Phase 1); goldens into `tests/eval/` aligned with Testing T-11 so the same fixtures serve both.
- First deliverables: stage-metrics regeneration report, RCM corpus-coverage report (the 5.1 matrix with real numbers), transcript empty-rate report, label-contract property tests green.
- Create the labeled alert-eval set (bootstrap: LLM-assisted pre-labeling with OpenAI gpt-4.1-mini + manual pass; never Anthropic) and publish alert precision/recall v1.

**Phase 3 - RCM correctness sprint (M)**
- New parser coverage (NR-03): DOUBLE YELLOW, "IN THIS LAP" ending, track-limits/investigation/pit-lane-status/weather/session families, scope-casing normalization. Home: the additive classifier superset in `src/nlp/` (duplicate-then-extend of `_classify_rcm_event`; the frozen original keeps serving `radio_agent.py` until the P4 CLI duplicate swaps callers). A parity test pins the two implementations on the shared families.
- `RaceControlStateTracker` (5.2): stateful SC/VSC lifecycle + red-flag + pending-penalty + pit-lane-status state, emitted through `lap_state` via the pre-classified `event_type` seam; wire into the editable callers (Arcade, backend) now, the CLI via the P4 duplicate.
- Fix NR-04 semantics at the same seam: RCM families map to routing-visible signals (either synthesized alert intents or new lap_state flags consumed by the editable callers), with the orchestrator untouched until its own duplicate cycle.
- Regression goldens from `resolver_eval`: full-stint `sc_currently_active` traces for every 2025 SC race; Qatar V7 stays as the canonical one.

**Phase 4 - Pipeline robustness lifts (M)**
- Input gates (NR-05): minimum-viable-text rule + per-stage confidence floor for alert eligibility, thresholds config-tunable and chosen against the Phase 2 alert-eval report, not by feel.
- Whisper QA capture (NR-06): cache schema v2 adding `avg_logprob`/`no_speech_prob`/failure-reason, lazy backfill on cache misses; coordinate with Loading F-02 before the caches ship to HF so the published schema is v2 from day one; hallucination flag propagated into the radio dict.
- NER dead-class mitigation (NR-07): suppress or mark classes below the bar; re-run the GLiNER zero-shot evaluation with corrected span alignment (1-hour check) and record the verdict for radiogate.
- Sentiment/intent calibration measurement via the shared `calibration` module (reliability/ECE) so the Phase 4 thresholds stand on measured probabilities.

**Phase 5 - Model refresh boundary + 2026 (S here; execution external)**
- Hand radiogate section 4.5 the acceptance bars and the harness: NER refresh accepted at span-F1 >= 0.70 on held-out GOLD, measured by `stage_metrics`, not by the training notebook.
- Intent/sentiment refresh criteria (weighted-F1 floor, calibration bands) recorded the same way.
- 2026 trigger (#189 F-12): re-run `rcm_coverage` + `stage_metrics` on first-2026 data; OTHER-rate and vocabulary-miss spikes generate the concrete extension issues. No speculative 2026 code before that.

Dependency spine: Phase 1 independent and immediate. Phase 2 needs #205's Phase-1 substrate (or bootstraps the shared skeleton itself if it lands first). Phase 3 parallel with 2, verified by 2. Phase 4 needs 2's reports. Phase 5 needs 2 and radiogate's own timeline.

---

## 8. Open questions (need Victor's decision)

1. **NR-08 handling if the bug is real:** the fix cannot edit `radio_agent.py`. Options: (a) ship the corrected `predict_intent` in the new `src/nlp` pipeline home and swap callers at the next duplicate cycle, documenting the known-wrong confidences until then; (b) treat it as severe enough to justify a case-by-case untouchable exception (same ruling gate as ML-eval open question 6). Decide after the 5-minute verification.
2. **Home for the RCM state tracker:** `src/nlp/rcm_state.py` (proposed: next to the parser's domain) vs `src/simulation/` (next to RaceStateManager, which the resolver docstring already anticipates as the upstream classifier). Affects import graphs for Arcade/backend/CLI-duplicate.
3. **Legacy module disposition:** move to `legacy/` (proposed) vs delete outright. Moving preserves the thesis-era artifact trail; deleting is cleaner. Either way `src/nlp/README.md` gets rewritten.
4. **Labeled alert-eval set:** target size and labeling protocol (proposal: 300-500 messages stratified by intent and GP, LLM-pre-annotated with gpt-4.1-mini, manually corrected by Victor in one sitting; doubles as a radiogate GOLD seed). Ratify.
5. **VSC-vs-SC MC differentiation:** the resolver can expose `neutralization_type` cheaply, but splitting `SC_PIT_BONUS` is an orchestrator-constants change owned by ML-eval E-04e. Sequence: expose the field now, let the sensitivity study decide the split? Ratify.
6. **Whisper model policy for QA backfill:** re-transcribing 529 clips to populate quality fields costs one long run per machine, or ships centrally via F-02. Decide whether v2 cache lands before or with the HF distribution.
7. **Does the empty-text gate suppress the row or mark it?** Suppressing keeps the LLM prompt clean; marking (`text_usable: false`) preserves the "a radio happened" signal for the panel. Proposal: mark, never silently drop. Ratify.

---

## 9. Verification protocol (when this plan is executed)

- **Phase 1:** the SetFit `classes_` verdict is written down with the pickle inspection as evidence; `f1-eval nlp` (once Phase 2 lands) or a one-off script shows argmax(predict_proba) == predict() on the golden bank; `pipeline_config_v2.json` round-trips on a clean machine with no absolute paths; importing `src.nlp` no longer exposes the contradictory legacy pipeline; the contract schema test runs in the hermetic CI tier.
- **Phase 2:** every notebook-era headline (0.4151 NER, 0.593 intent, 87.5% sentiment, 43.7 ms latency) is either reproduced within stated tolerance or the divergence is documented in the report; the RCM coverage report classifies 100% of 2025 `rcm.parquet` rows and its OTHER-rate per family matches the 5.1 matrix predictions; alert precision/recall v1 exists with n, provider and model recorded (OpenAI/LM Studio only).
- **Phase 3:** for every 2025 race with a neutralization, the tracker's per-lap `sc_currently_active` trace matches FastF1 TrackStatus for the full stint (not just the deploy lap); the Qatar V7 golden still passes; DOUBLE YELLOW and "IN THIS LAP" rows classify correctly in the parity test; a TIME_PENALTY event against the featured driver demonstrably reaches routing in an editable-caller run.
- **Phase 4:** empty-transcript rows can no longer produce alerts (battery test); cache v2 entries carry quality fields and the hallucination flag reaches the radio dict; threshold choices cite the alert-eval report; calibration report exists for both classifiers.
- **Phase 5:** the radiogate NER candidate is judged by `stage_metrics` on GOLD, and the >= 0.70 bar is enforced before any production swap; the first-2026 `rcm_coverage` re-run exists before any 2026 parser code is written.
