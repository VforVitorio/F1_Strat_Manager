# gridmind: F1-Domain LLM Fine-Tune with Unsloth (Corpus + LoRA Design)

**Status: research design, future work (post-TFG). Plan only, no code, no commitments.**

This document is the methodology design for the `gridmind` initiative of the F1 StratLab
ecosystem (initiative 1 of 5 in the post-TFG vision, see `FUTURE.md` sections 10 and 11,
not versioned). gridmind fine-tunes an LLM in the Gemma 3 (later Gemma 4) family on an
F1-domain text corpus using **Unsloth** (QLoRA), and publishes two Hugging Face artifacts
under the `f1stratlab` org:

1. **`f1stratlab/f1-domain-corpus`**: the curated F1-domain text and instruction dataset,
   with a full dataset card.
2. **`f1stratlab/strat-gemma-lora`**: the LoRA adapter (plus merged GGUF builds), with a
   full model card.

The model serves two consumers: the **strategy orchestrator N31** (its Layer 3 LLM
synthesis) and the future **X/Twitter bot `box-bot`** (initiative 2), which narrates the
live SSE stream during races. Both consumers share one hard product guardrail that shapes
every section of this design: **the model must never invent numbers or statistics.** Its
job is to reason over and cite the numbers it is given, not to recall numbers from its
weights.

Branding rule (from the ecosystem naming decision, 2026-06-12): `gridmind` does not carry
"f1stratlab" in its name, so its README, the dataset card, and the model card MUST state
explicitly that it is part of the F1 StratLab ecosystem.

Constraints inherited from the project:

- LLM providers are OpenAI or LM Studio (local), or open-source models from the HF Hub.
  Never Anthropic. gridmind itself is the open-source path: a Gemma LoRA served locally
  through LM Studio behind the same OpenAI-compatible interface the code already uses.
- Victor's chosen fine-tuning tool is **Unsloth**. This design centers on it. The only
  fallback, noted once and not developed further: if Unsloth ever blocks (for example a
  temporary gap in Gemma 4 support), plain Hugging Face TRL + PEFT QLoRA reproduces the
  same recipe at roughly 2x the VRAM and wall-clock cost.
- `src/agents/` internals, `scripts/run_simulation_cli.py`, and `notebooks/**` are
  untouchable. Everything proposed here is additive: a new independent repo, new HF
  artifacts, and configuration-level integration through the existing provider switch.
- This is a research design a graduate student would hand to an advisor: honest about
  what is hard, and every claim about existing assets is grounded in a real file.

---

## 1. Where gridmind sits

### 1.1 Ecosystem placement

Post-TFG, F1 StratLab becomes a multi-repo ecosystem: the core repo (the TFG) plus
dedicated public repos plus datasets and models on Hugging Face. gridmind is the F1 LLM
LoRA initiative. Per the ecosystem topology rule ("independent repo if it is a standalone
artifact"), gridmind should be an **independent public repo** holding the corpus-building
pipeline, the training configs, and the evaluation harness, with the heavy artifacts on
the HF Hub. It is not a submodule: the core repo consumes the model only through LM
Studio's OpenAI-compatible endpoint, so there is no code-level coupling to version.

### 1.2 Roadmap placement and dependencies

FUTURE.md phases place gridmind's inputs and outputs precisely:

- **Fase 1** (corpus to HF) produces `f1stratlab/f1-domain-corpus` and, in parallel,
  radiogate's `f1stratlab/f1-team-radio-corpus`. The radio corpus is an optional subset
  of the domain corpus (section 4.2), so gridmind's corpus v1.0 must be shippable
  without it and v1.1 can add it when radiogate Fase 1 lands.
- **Fase 2** (LoRA with Unsloth) is the training work designed here.
- **Fase 5** (the bot) consumes the model. The standing rule "do not launch the bot on
  drifting models" (Fase 4 before Fase 5) applies to the ML predictors, but gridmind has
  its own drift exposure: a model tuned on 2022-2025 regulation text will gloss 2026
  strategy with stale concepts. Section 11 treats this as a first-class risk.

### 1.3 The two consumers and what each needs

| Consumer | Task | Latency budget | Volume | Failure cost |
|---|---|---|---|---|
| N31 Layer 3 synthesis | Fill 3 narrative fields of `StrategyRecommendation` from a fully numeric prompt, as schema-constrained structured output | Seconds per strategic lap (the lap loop waits on it) | Tens of calls per simulated race | A wrong or ungrounded recommendation reaches the strategy surface; the Qatar V7 incident showed prompt-level guardrails can be violated with real consequences |
| box-bot | Turn a live SSE `lap_state` payload plus a recommendation into short public posts | Relaxed (a tweet can lag the lap by seconds) | Hundreds of generations per race weekend, race after race | A fabricated stat is published publicly under the project's name; reputational, permanent |

These pull in different directions: N31 needs maximal reasoning fidelity and structured
output discipline; box-bot needs voice, brevity, and absolute grounding at high volume.
One LoRA can serve both because the underlying skill is the same (grounded F1-domain
verbalization of provided numbers), differentiated at the prompt level. The rollout
policy (section 7.4) is deliberately asymmetric: bot first, orchestrator opt-in behind an
evaluation gate.

---

## 2. Asset inventory: what already exists (build on it, do not reinvent)

Grounding for everything downstream. All paths are in the core repo.

- **The FIA regulation corpus is already extracted and chunked.**
  `src/rag/retriever.py` serves a local Qdrant collection (`fia_regulations`, built by
  `scripts/build_rag_index.py`, embeddings `BAAI/bge-m3`, storage under
  `data/rag/qdrant_local`). The chunking pipeline that feeds Qdrant is exactly the text
  extraction gridmind needs for the regulations subset of the corpus: reuse the chunk
  source, not the vectors.
- **The N31 synthesis prompt defines the target task.**
  `src/agents/strategy_orchestrator.py`, `_build_orchestrator_prompt` (around line 741),
  assembles verbatim sub-agent numbers (pace prediction and CI, tire cliff P10/P50/P90,
  overtake and SC probabilities, pit stop duration percentiles, undercut probability and
  target, Monte Carlo scenario table) plus reasoning strings, hard strategic guardrails,
  and the N30 regulation context, then demands a reasoning paragraph that cites the cliff
  P50, the pace delta, one situation or radio signal, and a regulation article. This
  prompt is the single most important training-data template in this design: the model's
  core competence is exactly "follow this rubric with these numbers".
- **The LLM only fills narrative fields; code attaches the numbers.**
  `_get_orchestrator_llm` (line 118) wraps the model with structured output over
  `_LLMSynthesis`, which carries only the 3 fields the LLM actually writes; scenario
  scores and regulation context are attached in code afterwards (merge step around line
  1178). This is already a strong architectural guardrail: quantitative fields in the
  final `StrategyRecommendation` (14 fields) are code-sourced. The residual hallucination
  surface is the prose (misquoting a number inside `reasoning`), which is precisely what
  gridmind's guardrail training and probes target.
- **The provider switch is the integration point.**
  Every agent and the orchestrator check `F1_LLM_PROVIDER` (default `lmstudio`) and build
  a `ChatOpenAI` client pointed at LM Studio's local server with `api_key="lm-studio"`
  and a configurable model name. Serving gridmind requires zero changes to agent
  internals: load the GGUF in LM Studio, set the model name in config. Section 7 designs
  the details (structured output method, timeouts).
- **The per-layer model policy is established.** Sub-agents run gpt-4.1-mini; the
  orchestrator and chat run gpt-5.4-mini. gridmind slots into this policy as a third
  option per layer, not a wholesale replacement (section 7.4).
- **The LLM cost and latency audit (epic #261)** found the N31 prompt interleaves about
  1,300 static tokens with dynamic values and recommends a static-first restructure for
  prompt-cache friendliness. gridmind must train on the restructured (static-first)
  prompt shape so training format equals serving format, and it inherits the audit's
  timeout/retry hygiene requirements.
- **The ML and agents evaluation audit (epic #205)** designs `src/strategy/eval/` plus
  `tests/eval/` goldens, including a guardrail conformance battery and a season-scale
  replay with an ablation matrix. That battery is gridmind's acceptance gate (section 8).
- **radiogate's corpus design** (`documents/research/RADIOGATE_DECEPTION_AND_AUTOLABELING.md`)
  owns transcription quality, lap alignment, and the licensing analysis for radio-derived
  text. gridmind consumes its published dataset; it does not re-solve those problems.
- **Validated GPs from the TFG defense** (Hungary, Qatar, Australia demo cases) and the
  temporal 2023-24 train / 2025 test discipline of the ML notebooks give the natural
  eval-holdout backbone (section 4.5).
- **The environment**: Windows, uv-managed, torch pinned to cu128 via
  `[tool.uv.sources]` in `pyproject.toml` (RTX 40/50 class GPU). Training must not
  disturb this environment (section 5.6).

---

## 3. Task analysis: what the model must actually do

Before choosing data or hyperparameters, be explicit about the competences the fine-tune
must add over base Gemma, because everything else follows from them.

1. **Domain register.** Fluent use of F1 strategy vocabulary (undercut, overcut, cliff,
   delta, stint, compound offset, SC/VSC windows, box call) with correct semantics.
   Base Gemma models know casual F1; they are imprecise about strategy mechanics and
   regulation citations.
2. **Numeric grounding discipline.** Copy numbers from the prompt exactly, attribute
   them to the right signal (a cliff P50 is not a pit window; a 3-lap SC probability is
   not a 5-lap one), perform only trivially checkable arithmetic (gaps, lap differences),
   and refuse when a requested figure is absent. This is a trained behavior, not just a
   prompted one (section 6).
3. **Rubric-following synthesis.** Produce the N31 reasoning structure (action first,
   tire and pace numbers, one situational signal, regulation article when present, MC as
   confirmation, never as the sole justification) inside a JSON-schema-constrained
   response, reliably, at temperature 0.
4. **Compression for the bot.** Say the same grounded thing in a post-sized register:
   punchy, present tense, no invented color ("hammer time" is fine, "his tires are at
   47% grip" is fabrication unless the stream said so).
5. **Regulation literacy.** Recognize and correctly paraphrase the sporting-regulation
   articles that the RAG layer injects, and never cite an article number that was not in
   the provided context (article numbers are numbers too; the guardrail covers them).

Explicit non-goals: general chat ability beyond the base model, F1 trivia recall (the
guardrail actively suppresses reliance on memorized stats), multilinguality beyond what
base Gemma provides (project surfaces are English), and vision.

---

## 4. Corpus design: `f1stratlab/f1-domain-corpus`

### 4.1 Design principles

- **License-clean or owned, nothing scraped-and-hoped.** The dataset is public under the
  project's name; every subset must have a defensible provenance line in the card.
- **Two text regimes, clearly separated**: raw domain text for optional continued
  pretraining (CPT) and instruction-formatted pairs for supervised fine-tuning (SFT).
  Subsets are HF configs so consumers can load either regime alone.
- **The corpus encodes the guardrail.** Instruction targets are machine-verified to
  contain no numbers absent from their inputs (section 6.2) before they enter the
  dataset. The guardrail is a data property first and a training objective second.
- **Small and clean beats big and noisy.** Realistic total volume is tens of megabytes
  of text, not gigabytes. At that scale, one contaminated subset moves the model.

### 4.2 Source matrix

| Subset (HF config) | Source | Regime | Est. volume | License posture |
|---|---|---|---|---|
| `regulations` | FIA Sporting/Technical/Financial regulations 2022-2026, same extraction pipeline as `scripts/build_rag_index.py` | CPT + retrieval-grounded SFT | ~2-4 MB/season | FIA publishes these openly; redistribution of extracted text is customary in the community but not expressly licensed. Card must state source, extraction date, and non-affiliation; be prepared to gate or remove on request (section 9.3) |
| `race-reports` | Wikipedia race-report articles (the "Report"/"Race" sections of every GP article, 2018-2025) | CPT | ~8-15 MB | CC BY-SA 4.0, attribution list in the card; share-alike applies to this subset |
| `glossary` | Wikipedia "Glossary of Formula One" and related terminology articles, cleaned to definition pairs | CPT + SFT (definition Q/A) | <1 MB | CC BY-SA 4.0 |
| `strategy-instruct` | **The project's own generated explanations**: N31 prompt/response traces regenerated with the replay engine across non-holdout GPs, filtered by the guardrail checker | SFT (primary subset) | 5k-20k pairs | Fully owned; numbers are simulator-sourced and verifiable |
| `bot-style` | Post-sized rewrites of `strategy-instruct` items plus lap-narration pairs built from SSE `lap_state` payloads | SFT | 3k-10k pairs | Fully owned |
| `guardrail-probes-train` | Abstention and perturbation examples (section 6.2): inputs with missing stats and gold refusals, counterfactual numbers with gold echoes | SFT | 1k-3k pairs | Fully owned (synthetic) |
| `radio` (v1.1+) | radiogate's `f1stratlab/f1-team-radio-corpus`, transcript text plus labels | CPT + SFT (intent paraphrase) | per radiogate | Inherits radiogate's licensing analysis; do not duplicate it here |
| `general-mix` | A small slice (5-10% of SFT tokens) of a permissively licensed general instruction set (for example an Apache-2.0 open instruction dataset) | SFT regularizer | sized to ratio | Apache-2.0/CC-BY subset only |

Deliberately excluded, with reasons stated in the card: motorsport press (Autosport, The
Race, official formula1.com content) because it is copyrighted editorial text with no
redistribution rights; forum/Reddit content because licensing and quality are both
unclear; books and paywalled analysis for the same reason. The project's thesis and
IEEE paper text may be added as an owned `project-docs` subset if Victor wants the model
to speak the system's own architecture language, but they are small and optional.

Two source notes:

- **Teacher provenance for `strategy-instruct` and `bot-style`.** If synthesis targets
  are regenerated rather than replayed from archives, generate them with the pipeline
  under `F1_LLM_PROVIDER=lmstudio` using a strong open-weights teacher, not the OpenAI
  API. This sidesteps the OpenAI terms question about using API outputs to develop other
  models entirely, and it keeps the dataset provenance 100% open. Archived traces from
  past validated runs are usable regardless, but the card must state which teacher
  produced which slice.
- **The `general-mix` regularizer** exists to fight catastrophic forgetting and
  format-collapse on a small, narrow SFT set. It is standard practice for narrow-domain
  LoRAs; without it, a few thousand same-shaped examples can degrade the base model's
  instruction-following outside the template.

### 4.3 Curation pipeline

Per subset, in order:

1. **Extraction** with recorded tool versions (the regulations reuse the RAG pipeline's
   extraction; Wikipedia via the official dumps or API with revision IDs recorded).
2. **Cleaning**: boilerplate removal, table flattening for regulation annexes, reference
   markers stripped from wiki text, Unicode normalization.
3. **Quality filter**: language ID (English), minimum length, symbol-to-text ratio,
   and for `strategy-instruct` the guardrail checker (section 6.2) plus a
   schema-validity check on every target.
4. **Deduplication**, two levels:
   - Exact: hash on normalized text.
   - Near-duplicate: MinHash/LSH at document level within and across subsets (Wikipedia
     race reports quote regulation phrases; regulation editions repeat 90% of their text
     year over year). For regulations, keep all editions but tag them with
     `season` metadata so the 2026 refresh can filter cleanly.
5. **Decontamination against evaluation** (section 4.5), run last so nothing added later
   bypasses it.

Every record carries provenance metadata: `subset`, `source_id` (URL or file), `season`
or `gp` where applicable, `license`, `extraction_date`, `teacher` (for synthetic
targets), and `checker_version` (for guardrail-verified targets).

### 4.4 Formatting: CPT vs SFT, and the recommended mix

Three options considered:

- **CPT only** (raw text, completion loss): teaches register and regulation phrasing but
  nothing about the JSON task or the guardrail. Rejected as the sole regime.
- **SFT only**: teaches the task directly; the domain register comes along implicitly
  because the instruction inputs are saturated with domain text. Viable and simplest.
- **Short CPT then SFT** (two-stage): a light CPT pass (1 epoch over `regulations` +
  `race-reports` + `glossary`, low LR) before SFT can measurably improve regulation
  paraphrase quality. Costs one more training stage and one more ablation axis.

**Recommendation: SFT-first as the v1 baseline, with CPT-then-SFT as a single planned
ablation** (section 8.4). At this corpus size the SFT set dominates behavior; CPT earns
its stage only if the ablation shows a regulation-literacy gain without a grounding
regression. Unsloth supports both regimes on the same stack (completion-style training
and chat-template SFT with response-only loss masking).

Instruction format specifics:

- **Gemma chat template.** Gemma 3 has no separate system role in its native template;
  system-style content is folded into the first user turn. Training data must be
  serialized with the exact Gemma template (Unsloth ships it) so there is no
  train/serve mismatch through LM Studio, which uses the GGUF's embedded template.
- **Static-first prompt shape.** Adopt the LLM-cost audit's restructure: rubric,
  guardrails, and schema description first (static), race context and sub-agent numbers
  last (dynamic). Train on this shape; serve this shape. This also makes the OpenAI-path
  prompt cacheable for whoever stays on gpt-5.4-mini.
- **Loss on responses only** (Unsloth's `train_on_responses_only` mechanism) so the model
  is never trained to reproduce prompt text, only to answer it.
- **Targets are schema-shaped.** `strategy-instruct` targets are the JSON object the
  orchestrator's structured-output wrapper expects (the `_LLMSynthesis` fields), not free
  prose, so that structured-output serving is in-distribution. `bot-style` targets are
  plain short text.

### 4.5 Splits, decontamination, and the leakage policy

The failure mode to design against: the model is evaluated on GPs whose strategy
explanations (or Wikipedia race reports) it saw in training, and the evaluation
overstates grounding because the model can lean on memorized race facts that happen to
match. The policy:

- **Holdout is by Grand Prix, not by example.** Reserve an eval GP set: the TFG-validated
  demo GPs (Hungary, Qatar, Australia) plus at least 3 more GPs stratified across track
  clusters, and align with whatever season-scale replay set epic #205 fixes. For every
  holdout GP: no `strategy-instruct` pairs, no `bot-style` pairs, and the corresponding
  Wikipedia race report is excluded from `race-reports` for the same season.
- **The 2025-as-test-season discipline** of the ML notebooks extends here: prefer
  building `strategy-instruct` from 2023-2024 replays and keep 2025 GPs predominantly
  for evaluation, mirroring the predictors' temporal split so the whole system is
  evaluated on the same unseen tail.
- **Probe secrecy.** The evaluation probe set (section 8.2) ships as a separate HF config
  (`eval-probes`) that is documented as never-train; the training pipeline hard-excludes
  it by config name, and the decontamination step additionally n-gram-matches training
  targets against probe texts (13-gram overlap threshold, the standard contamination
  heuristic) in case someone regenerates similar items.
- **A decontamination manifest** (list of held-out GPs, excluded article revisions,
  n-gram filter hits) is published with each dataset version so the claim "eval is
  unseen" is auditable, which matters if gridmind results feed a paper.

### 4.6 Versioning as an HF dataset

- Semantic dataset versions with git tags on the HF repo: v1.0 (no radio), v1.1 (+radio),
  v2.0 (2026-regulation refresh, new regulation edition + post-2026 instruct pairs).
- Consumers pin the HF **commit hash**, not `main` (the P5 data-engineering audit flagged
  mutable-`main` consumption as an anti-pattern in the existing dataset; do not repeat
  it here).
- Each version's card records: subset sizes (documents, tokens), the decontamination
  manifest, checker version, and a changelog.

---

## 5. Fine-tuning with Unsloth

### 5.1 Base model choice

The family is fixed by the plan (Gemma 3 now, Gemma 4 when released and supported by
Unsloth). The open choice is size, and it is hardware-bound. Gemma 3 text sizes: 1B, 4B,
12B, 27B; the 4B+ models carry 128K context and the interleaved local/global attention
that keeps KV-cache small, which is friendly to a local server.

| Candidate | QLoRA training VRAM (Unsloth, 4-bit, ~4k seq) | GGUF Q4_K_M serving footprint | Fit |
|---|---|---|---|
| Gemma 3 4B-it | ~6-8 GB | ~2.5-3 GB + KV | Trains and serves on any RTX 40-class GPU including 8 GB cards; credible for `box-bot`; likely below the reasoning bar for replacing N31's gpt-5.4-mini |
| Gemma 3 12B-it | ~12-16 GB | ~7-8 GB + KV | The quality option; realistic N31 candidate; requires a 16 GB GPU for training (or cloud spot for the training run only) and comfortable local serving on 12 GB+ |
| Gemma 3 27B-it | ~22-24 GB | ~16-17 GB + KV | Out of scope for local training and serving on this project's hardware; not pursued |
| Gemma 3 1B | ~4 GB | ~1 GB | Too weak for rubric-following synthesis; only useful as a smoke-test target for the pipeline |

**Recommendation: train the 4B first as the pipeline-proving model and the bot model;
train the 12B as the orchestrator candidate if (and only if) the training GPU has 16 GB+
VRAM or a one-off cloud run is acceptable.** The corpus, checker, eval battery, and
serving path are identical for both; the size decision is deferred to a hardware fact
(open question Q1). Start from the instruction-tuned (`-it`) variants, not the base
checkpoints: the target tasks are instruction-shaped, the SFT set is small, and `-it`
checkpoints preserve schema-following behavior that base checkpoints would need far more
data to learn.

Gemma 4 upgrade clause: when a Gemma 4 text model ships with Unsloth support, rerun the
identical recipe (same corpus version, same configs, same eval battery) and let the
evaluation gate decide the swap. The design is family-portable by construction because
nothing below depends on Gemma-3-specific internals except the chat template, which
Unsloth abstracts.

### 5.2 Why Unsloth specifically (and its known Gemma caveats)

- QLoRA on Unsloth's optimized kernels trains roughly 2x faster with about 60-70% less
  VRAM than the stock HF stack, which is the difference between "trains on the project
  GPU" and "needs cloud" at the 12B size.
- First-class Gemma 3 support: patched chat template, response-only loss masking, and
  the documented fix for Gemma 3's float16 overflow issue (irrelevant on RTX 40/50,
  which train in bfloat16, but it indicates the maintenance depth).
- Direct GGUF export (merged weights quantized to the llama.cpp formats LM Studio loads)
  collapses the train-to-serve pipeline into one tool (section 7.1).
- Caveats to plan for: Unsloth is a fast-moving single-vendor project, so **pin the
  exact Unsloth version per training run** in the manifest; Windows-native support has
  historically trailed Linux (Triton), so the default training environment is WSL2 or a
  Linux box (section 5.6); and its notebooks change, so the repo keeps its own frozen
  config rather than tracking upstream examples.

### 5.3 QLoRA configuration (starting point, to be ablated)

| Parameter | Value | Rationale |
|---|---|---|
| Quantization | 4-bit NF4 (QLoRA), compute dtype bfloat16 | The Unsloth default path; bf16 is native on RTX 40/50 and avoids Gemma 3's fp16 issue |
| LoRA rank r | 16 | Enough capacity for register + rubric behavior on a small corpus; 8 and 32 are the ablation neighbors |
| LoRA alpha | 16 (alpha = r) | Unsloth's recommended pairing; keeps the effective scale stable if r moves |
| LoRA dropout | 0 | Unsloth's fast path; the small-data overfitting control is epochs and eval-based early stop, not dropout |
| Target modules | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj | Attention + MLP, the standard full-coverage set; embeddings and lm_head stay frozen so base vocabulary behavior (and the chat template tokens) are untouched |
| Max sequence length | 4096 | The N31 prompt plus response fits well under this (about 1.3k static + dynamic blocks); 8192 only if bot context packing wants it |
| Learning rate | 2e-4, cosine decay, warmup 3% | Standard QLoRA SFT starting point |
| Effective batch | 16 (micro-batch per VRAM, gradient accumulation to 16) | Stability on mixed-length data |
| Epochs | 1-2 over the SFT mix, eval-gated | Small corpus; more epochs mostly buy memorization, which is the exact failure mode the guardrail forbids |
| Loss masking | Responses only | Section 4.4 |
| Packing | On for CPT stage (if run), off for SFT | Packing across instruction boundaries can blur schema behavior |
| Seed | Fixed and recorded | Reproducibility (section 5.5) |
| Gradient checkpointing | Unsloth mode | The VRAM saving that makes 12B feasible on 16 GB |

The optional CPT stage (ablation): same adapter shape, LR 5e-5, 1 epoch over the raw-text
subsets, then SFT continues on the same adapter.

An optional third stage, **preference tuning (DPO)** on grounded-vs-fabricated pairs, is
designed in section 6.3 and deliberately deferred to v1.1: run it only if the SFT model's
fabrication rate on the probe battery is above the gate.

### 5.4 Training data mixture (SFT stage)

Token-weighted mix, tuned so the task subsets dominate but nothing collapses:

- `strategy-instruct`: 45-55%
- `bot-style`: 15-20%
- `guardrail-probes-train`: 10-15% (abstentions and perturbation echoes; oversampled
  relative to raw size because the behavior matters more than the token count)
- `glossary` Q/A + retrieval-grounded regulation Q/A: 10%
- `general-mix` regularizer: 5-10%

The mixture is a config file in the gridmind repo, versioned with the run.

### 5.5 Reproducibility

Adopt the project's manifest habit (the ML models ship `feature_manifest_*.json`; the
same discipline applies here). Every training run writes a **training manifest**
recording: base model HF ID and revision hash, dataset version and HF commit hash,
subset mixture weights, full hyperparameter table, Unsloth/torch/CUDA versions, seed,
hardware, wall-clock, and final loss curves. The manifest is committed to the gridmind
repo and mirrored in the model card. When pitlab (initiative 4) lands a tracker (ClearML
or MLflow), runs log there too, but the manifest file is the portable source of truth
and exists from run one, before any tracker.

### 5.6 Environment isolation (the cu128 note)

The core repo pins torch to cu128 through `[tool.uv.sources]` and that lockfile must not
be disturbed by training dependencies (Unsloth pins its own torch/triton/bitsandbytes
constellation and moves fast). Therefore:

- Training lives in the **gridmind repo with its own uv environment**, never in the core
  repo's environment. The two only meet at the HF Hub and the LM Studio endpoint.
- Default training OS is **WSL2 (Ubuntu) on the same machine**, where Unsloth and Triton
  are first-class; the cu128-class driver on the Windows host serves both worlds. If
  Unsloth's Windows-native path is green at execution time, native is acceptable, but
  WSL2 is the assumption the design makes.
- Serving needs neither environment: LM Studio consumes the exported GGUF directly on
  Windows (section 7).

---

## 6. The no-invented-numbers guardrail

This is the load-bearing section. The guardrail is not one mechanism; it is the same
invariant enforced at five layers, so that no single failure publishes a fabricated stat.

### 6.1 Threat model

Ways this model can invent numbers:

- **T1 Recall substitution**: the prompt provides a value, the model outputs a
  memorized "real world" value instead (asked about a simulated Hungary where the gap is
  2.1s, it writes 1.2s because a memorized race had 1.2s).
- **T2 Gap filling**: the prompt lacks a value the rubric mentions, and the model
  invents a plausible one rather than flagging the absence.
- **T3 Arithmetic drift**: the model derives a number (a gap difference, a lap count)
  and gets it wrong, or "derives" something underivable.
- **T4 Attribution swap**: the number is real but assigned to the wrong quantity (the
  cliff P10 quoted as the P50; the 3-lap SC probability presented as a 5-lap one; this
  family produced the class of bug the ML-eval audit flagged in the MC window mismatch).
- **T5 Citation fabrication**: a regulation article number that was not in the provided
  context (article numbers are numbers).
- **T6 Bot color**: in the loose bot register, decorative stats ("30% of races here end
  under SC") that no payload provided.

The architecture already caps the blast radius for N31 (quantitative output fields are
code-attached; the LLM writes prose), so for the orchestrator the threats live inside
the reasoning text. For box-bot, the entire output is prose, so all six threats are
fully exposed there. This asymmetry is why the bot gets the strictest serving-side
verifier (6.5) and the softest rollout (7.4).

### 6.2 Layer 1: data (the checker, and abstention training)

**The numeric grounding checker** is a deterministic tool used three times: as a dataset
filter, as an evaluation metric, and as a serving-time verifier. Its contract, described
functionally (no code by design):

- Extract every numeric span from the candidate response: integers, decimals,
  percentages, times with units (s, ms, laps, °C, kph), probability-like decimals, and
  regulation article identifiers (patterns like "Article 30.5(a)").
- Extract the same from the prompt (all sub-agent blocks, race context, MC table,
  regulation context).
- A response number is **grounded** if it matches a prompt number exactly (integers,
  article numbers) or within formatting tolerance (floats: same value up to the decimal
  precision shown in the prompt; unit-aware comparison).
- A response number is **derivable** if a depth-1 arithmetic search over prompt numbers
  reproduces it (differences, sums, lap arithmetic like "pits in 3 laps" from lap 22 and
  window 25). Derivable numbers pass but are tagged, so the eval can report the
  derived-share separately (T3 exposure).
- Everything else is a **fabrication hit**. A response with any hit fails.
- Attribution checking (T4) is heuristic at the checker level (a quoted value that
  matches P10 but is labeled P50 in surrounding text can be caught by proximity
  pattern rules for the known signal names); full T4 coverage comes from the probe set
  (6.4), not the checker.

Data-side uses:

- Every `strategy-instruct` and `bot-style` target must pass the checker before entering
  the corpus. Teacher outputs that fail are either dropped or repaired and re-checked.
  The dataset card reports the rejection rate (an honest signal of teacher quality).
- **Abstention examples** (`guardrail-probes-train`): inputs where a signal block is
  marked "not activated" (the real prompt builder emits exactly this for missing
  sub-agents) and the gold response explicitly reasons without that signal or states the
  absence, never substituting a value. These teach T2's correct behavior.
- **Perturbation-echo examples**: pairs where a prompt value is counterfactually altered
  (a Hungary replay with an implausible-but-stated 40°C track temp) and the gold response
  uses the stated value. These directly train against T1: the prompt outranks memory.

### 6.3 Layer 2: training objective

- SFT on checker-verified targets makes grounded citation the only rewarded behavior.
- The mixture oversamples abstention and perturbation examples (section 5.4) because
  behavior-shaping examples punch above their token weight.
- **Optional DPO stage** (deferred to v1.1, run only if the SFT gate fails): preference
  pairs where chosen = checker-passing response, rejected = the same response with
  fabricated or swapped numbers. Rejected samples come cheap from two sources: real
  teacher failures harvested by the checker during dataset construction, and synthetic
  corruptions (swap P10/P50 labels, perturb a cited value by 10-30%, insert a plausible
  uncited stat). This is the highest-precision anti-hallucination lever available
  without reinforcement infrastructure, and Unsloth supports DPO on the same QLoRA
  stack.

### 6.4 Layer 3: evaluation probes (specified here, gated in section 8)

Probe families, each mapping to a threat:

- **Missing-stat probes** (T2): N31-shaped prompts with one signal block removed; pass =
  no value invented for the missing signal, absence acknowledged or reasoned around.
- **Counterfactual-world probes** (T1): fictional GPs, fictional drivers, physically odd
  but stated values; pass = the stated values are used verbatim. This is the cleanest
  test that the model reads rather than recalls, because memory can offer no help.
- **Attribution probes** (T4): prompts where P10/P50/P90 (or the 3-lap vs 5-lap SC
  probabilities) are distinct and adversarially close; pass = each quoted value carries
  its correct label.
- **Citation probes** (T5): regulation context present with specific articles vs absent;
  pass = article numbers appear only when provided, and match.
- **Derivation probes** (T3): prompts inviting simple arithmetic; scored for
  correctness of the derived value.
- **No-context bot probes** (T6/T2): bot-register requests with a payload missing the
  requested figure ("what is the fastest lap?" with no fastest-lap field); pass =
  refusal or deflection, never a number.

Primary metric: **fabrication rate** = share of probe responses with at least one
checker fabrication hit. Secondary: **number-fidelity** (share of prompt-cited values
quoted exactly and attributed correctly on the attribution probes) and **abstention
accuracy** (share of missing-stat probes handled without invention). The probes ship as
the never-train `eval-probes` config (section 4.5).

### 6.5 Layer 4: decoding and serving-time enforcement

- **Temperature 0** for N31 (already the orchestrator's configuration posture) and low
  temperature with the verifier for the bot.
- **Structured output**: LM Studio enforces the response JSON schema by
  grammar-constrained decoding on GGUF models (section 7.3), which eliminates
  format-level failure and confines the guardrail problem to the content of the
  narrative fields.
- **The serving-time verifier for box-bot**: every candidate post runs through the same
  numeric grounding checker against the SSE payload + recommendation it was generated
  from; a fabrication hit blocks the post and triggers one regeneration, then a
  numberless fallback template ("Box for VER, hards, rejoining in traffic") if the retry
  also fails. The bot never publishes an unchecked number. This verifier lives in
  box-bot's repo but is specified here because it is the same checker artifact,
  published as part of gridmind so both repos consume one implementation.
- **For N31**: the checker can run in shadow mode (log-only) on the reasoning field
  during the evaluation season, feeding the #205 conformance battery; promoting it to a
  blocking regenerate-on-fail step inside the orchestrator would touch orchestrator
  code, so it is additive-only and gated on the shared-engine work (the P2b entry point)
  rather than any edit to `src/agents/` internals.

### 6.6 Layer 5: interaction with N31 and the bot (why the system tolerates residual risk)

- N31 provides every number the model should use, in the prompt, by construction
  (`_build_orchestrator_prompt` renders sub-agent outputs verbatim precisely "so the LLM
  can cite them"). gridmind's training makes the model take that offer; the code-attached
  grounding fields mean even a prose slip cannot corrupt the structured quantitative
  fields; and the #205 battery measures the residual prose slip rate.
- box-bot's contract is "cite the live stream, never invent": the stream payload is the
  only ground truth, the prompt template forbids outside stats, the LoRA is trained to
  comply, and the verifier enforces it mechanically. Four layers must fail
  simultaneously for a fabricated number to reach the public timeline.

---

## 7. Serving: LM Studio behind the existing provider layer

### 7.1 Export pipeline

Unsloth exports directly to the format LM Studio consumes:

1. Merge the LoRA into the base weights (bf16 merged checkpoint, archived to the HF
   model repo for provenance and for future re-quantization).
2. Export GGUF at a small quantization ladder: **Q4_K_M** (the default serving build),
   **Q8_0** (the quality-reference build used in evaluation to isolate quantization
   loss), and optionally Q5_K_M as the middle point. The eval battery (section 8) runs
   on the exact GGUF artifact that will serve, not on the merged fp16 checkpoint,
   because quantization can measurably move small-model behavior and the gate must test
   what ships.
3. Publish: adapter + merged weights in `f1stratlab/strat-gemma-lora`, GGUF builds
   either in the same repo or a sibling `-GGUF` repo (LM Studio's model browser
   discovers HF GGUF repos directly, which makes installation "search, click, load").

### 7.2 Integration with the provider switch (zero core-code changes)

The core repo already does everything needed:

- `F1_LLM_PROVIDER=lmstudio` (the default) routes every agent and the orchestrator to
  `ChatOpenAI` against the local server with `api_key="lm-studio"` and the configured
  model name.
- Serving gridmind = load the GGUF in LM Studio, set the layer's model name to the LM
  Studio model identifier. No agent internals change, honoring the untouchable rule.
- The per-layer policy extends naturally: model name is already a per-layer config
  value, so "N31 on gridmind, sub-agents on gpt-4.1-mini" and "everything on OpenAI,
  bot on gridmind" are both pure configuration states.

Operational requirements inherited from the audits, restated as serving requirements
rather than new design: a finite client timeout on the LM Studio path (the Security S-5
/ LLM-cost F1 one-line fix; an unbounded local call pinned a lap for about 30 minutes
once), and usage/latency logging per call so gridmind's real latency distribution is
measured from day one.

### 7.3 Structured output through LM Studio

This is the one genuinely delicate integration point, so it is designed explicitly:

- The orchestrator wraps its LLM with LangChain structured output over `_LLMSynthesis`.
  LangChain's default method for `ChatOpenAI` is tool calling; small open models served
  through llama.cpp are historically unreliable tool callers, and Gemma's tool-calling
  depends on template support.
- LM Studio's robust path is **JSON-schema response formatting** (grammar-constrained
  decoding), which guarantees schema-valid output at the decoder level regardless of the
  model's tool-calling skill.
- Therefore: (a) train the `strategy-instruct` targets as the literal JSON objects
  (section 4.4) so constrained decoding is in-distribution rather than fighting the
  model; (b) run the LM Studio path with the JSON-schema method, which on the LangChain
  side is a structured-output method option, an additive configuration concern for the
  shared-engine entry point rather than an edit to the untouchable orchestrator file;
  (c) make **schema-compliance rate under constrained decoding** an explicit eval metric
  (it should be 100% by construction; the metric exists to catch degenerate compliance,
  like empty strings in required fields, which constrained decoding cannot prevent).
- The bot path needs no structured output (free text through the same OpenAI-compatible
  endpoint) and is therefore trivially compatible.

### 7.4 Per-layer rollout policy (who actually uses gridmind)

Recommendation, in order of increasing stakes:

1. **box-bot: gridmind is the primary model.** The bot is the consumer that justifies a
   local model economically and operationally: hundreds of generations per race weekend,
   always-on during sessions, zero marginal cost, no rate limits, and the strongest
   serving-side verifier. The bot's quality bar (grounded, punchy, domain-fluent short
   text) is exactly what a 4B LoRA can meet.
2. **Streamlit/chat explanation surfaces: optional secondary.** Low stakes, human in the
   loop, useful as a live quality signal.
3. **N31 Layer 3: opt-in, gated, default OFF.** gpt-5.4-mini remains the default
   orchestrator model. gridmind becomes the N31 model only after passing the section 8
   gate (non-inferiority on the #205 battery plus superiority or parity on the
   fabrication metrics), and even then as a config choice, with the OpenAI path kept as
   the documented fallback. The honest cost-benefit from the LLM-cost audit: the
   mini-class OpenAI spend is about $1-2 per race, so gridmind's value for N31 is not
   money; it is offline capability, latency control, provider independence, and the
   research result itself. That value is real but does not justify accepting a
   reasoning regression on the strategy surface.
4. **Sub-agents (N25-N29): not a target.** Their LLM work is thin tool-wrapping where
   gpt-4.1-mini is already cheap and reliable; swapping them buys nothing and multiplies
   eval surface.

---

## 8. Evaluation: proving gridmind helps without regressing the orchestrator

### 8.1 Evaluation axes

| Axis | Question | Instruments |
|---|---|---|
| Domain fluency | Does it speak F1 strategy better than base Gemma? | Perplexity on held-out domain text (holdout race reports, unseen regulation sections); terminology-usage spot rubric |
| Synthesis quality | Are the N31 narrative fields good strategy prose? | Rubric-based LLM-as-judge (gpt-5.4-mini as judge, rubric = the prompt's own reasoning rubric: action-first, cites cliff P50 and pace delta, one situational signal, regulation article when present, MC never sole justification) + blinded human spot checks by Victor on a fixed sample |
| Grounding | Does it invent numbers? | The section 6.4 probe battery: fabrication rate, number-fidelity, abstention accuracy, citation accuracy |
| Format discipline | Does structured output hold? | Schema-compliance and degenerate-compliance rates under LM Studio constrained decoding, on the shipping GGUF |
| System non-regression | Does swapping the N31 model change decisions for the worse? | The #205 battery: guardrail conformance rate, decision agreement vs the incumbent model on the season-scale replay, golden-case suite (including the Qatar V7 RCM-SC regression case) |
| Bot quality | Are posts good and safe? | Verifier block rate in shadow runs over recorded SSE streams; human review of a fixed post sample |
| Latency | Is it fast enough per lap / per post? | P50/P95 wall-clock on the serving hardware, per quantization build |

### 8.2 Baselines and comparisons

Every metric is reported for: base Gemma (same size, same prompts, no fine-tune),
gridmind SFT, gridmind with the optional CPT stage (ablation), gridmind at Q4_K_M vs
Q8_0 (quantization delta), and the incumbent (gpt-5.4-mini) where the metric applies.
The base-Gemma column is what demonstrates the fine-tune earns its existence; the
incumbent column is what gates the N31 swap.

### 8.3 The gate (tied to epic #205)

The ML-eval audit's battery (`src/strategy/eval/` + `tests/eval/` goldens, guardrail
conformance measurement, season-scale replay with ablation matrix) is the acceptance
gate for any model change on the strategy surface, gridmind included. Concretely,
gridmind may become an N31 option only when, on the shipping GGUF:

- Fabrication rate on the probe battery: at or below the incumbent's measured rate, and
  below an absolute ceiling agreed at execution time (target posture: zero fabrication
  hits on counterfactual and missing-stat probes; these are the non-negotiable
  families).
- Guardrail conformance rate (the #205 metric that does not exist yet for the
  incumbent either; measuring the incumbent is a #205 deliverable gridmind depends on):
  no worse than the incumbent.
- Golden-case suite: all pass, including the Qatar V7 case.
- Season-scale replay: decision-agreement and outcome-score deltas within the
  non-inferiority margin defined by #205's ablation matrix.
- Judge-rubric synthesis quality: within the margin of the incumbent on the blinded
  sample.

For box-bot the gate is lighter (the verifier is the hard backstop): fabrication rate
zero after verifier, verifier block rate below an operability threshold (a bot that
blocks half its posts is not shippable), and human sign-off on voice.

### 8.4 Planned ablations (kept small deliberately)

1. SFT-only vs CPT-then-SFT (section 4.4).
2. LoRA rank 8 vs 16 vs 32 (capacity vs memorization on a small corpus).
3. With vs without `guardrail-probes-train` oversampling (does the guardrail data
   actually move the fabrication rate, or is the checker+verifier stack doing all the
   work?). This ablation is the scientifically interesting one and the one worth a
   paper paragraph.
4. 4B vs 12B (only if both are trained).
5. Q4_K_M vs Q8_0 on the full battery (quantization cost, decides the shipping build).

### 8.5 What is explicitly not claimed

The evaluation does not claim general-benchmark parity (no MMLU-style suites; the model
is a domain specialist and the `general-mix` regularizer only protects instruction
format, not general knowledge), and it does not claim the model knows F1 facts (the
guardrail actively discourages relying on them; the counterfactual probes reward
ignoring them).

---

## 9. HF artifacts: cards, licensing, branding

### 9.1 Dataset card (`f1stratlab/f1-domain-corpus`)

Must contain:

- **Ecosystem declaration** (the branding rule): first paragraph states it is part of
  the F1 StratLab ecosystem, links the core repo and the sibling artifacts
  (`f1-strategy-dataset`, `f1-team-radio-corpus`, `strat-gemma-lora`).
- Per-subset provenance: source, extraction method and date, revision IDs for wiki
  content, teacher identity for synthetic targets, checker version and rejection rate.
- **Per-subset licensing** (the card-level license is "mixed, see subsets"):
  CC BY-SA 4.0 for wiki-derived subsets with the attribution list; owned subsets under
  a permissive license of Victor's choice (CC BY 4.0 is the natural pick for data);
  the regulations subset's posture stated plainly (section 9.3); the radio subset
  deferring to radiogate's card.
- The decontamination manifest and the never-train status of `eval-probes`.
- Intended use (fine-tuning F1-domain assistants), out-of-scope use (the corpus is not
  a stats reference; simulated numbers in instruct pairs are simulator outputs, not
  historical facts, and the card must say so loudly so nobody scrapes it as a facts
  dataset).
- Known limitations: English-only, 2022-2025 regulation era (until v2.0), single-team
  perspective baked into the strategy-instruct pairs (the driver-centric lap_state
  boundary of the core system).

### 9.2 Model card (`f1stratlab/strat-gemma-lora`)

Must contain:

- **Ecosystem declaration** (same branding rule).
- Base model and revision; the statement that this is a Gemma derivative and therefore
  distributed under and subject to the **Gemma Terms of Use** (Google's license
  requires derivatives to carry the use restrictions and the prohibited-use policy
  downstream; the HF repo must include the license text and the card must link it).
  This is a hard licensing fact of choosing Gemma: the adapter and merged builds cannot
  be relicensed Apache-2.0. It is compatible with the project's open publication goals,
  but the card must be accurate about it.
- Training data: the exact dataset version (commit hash), the mixture table, the
  training manifest (section 5.5) inline or linked.
- Evaluation: the full section 8 battery results, incumbent comparisons included, and
  the fabrication-rate table given top billing (it is the model's defining property).
- **Intended use and the guardrail contract, stated as a product warning**: this model
  is designed to reason over numbers provided in its prompt and to refuse absent ones;
  it is NOT a source of F1 statistics; any deployment that asks it for facts without
  providing them is out of scope and will produce refusals (by design) or errors (out
  of contract). Downstream users get told exactly what the F1 StratLab surfaces enforce
  (verifier for the bot, structured output + code-attached numbers for the
  orchestrator).
- Serving guidance: the GGUF builds, LM Studio setup (load model, note the identifier,
  point any OpenAI-compatible client at the local server), temperature 0 for synthesis
  use, and the JSON-schema structured-output recommendation.

### 9.3 The regulations-subset licensing decision

The FIA publishes its regulations openly and the community redistributes extracted text
routinely, but there is no explicit redistribution license. Three postures, decided at
execution (open question Q3):

1. **Include, public, clearly attributed** with a takedown-on-request note (community
   norm, small legal exposure, maximal usefulness).
2. **Gated HF subset** (users click through to access; reduces exposure, keeps
   reproducibility).
3. **Exclude the text, ship the pipeline**: the corpus repo documents the extraction
   recipe (the same one `build_rag_index.py` uses) and users regenerate the subset
   locally from FIA's own PDFs. Cleanest legally, worst for one-click reproducibility.

Recommendation: posture 2 for the corpus (gated subset), because the model card can
still report training on it while the raw text is not world-scrapeable, and the
extraction recipe is published anyway for posture-3-style regeneration.

---

## 10. Phased roadmap

Phases are sequential; each ends with a checkable artifact. Sizing is deliberately in
artifacts, not dates (this is future work with no committed schedule).

- **G0. Decisions and scaffolding.** Resolve the open questions (section 12): training
  GPU VRAM, regulations-subset posture, teacher choice, radio-subset timing. Create the
  `f1stratlab` HF org (already planned in FUTURE.md 11.2) and the gridmind repo with
  the standard project bootstrap. Artifact: decided answers recorded in the repo README
  + empty HF repos reserved.
- **G1. Checker first.** Implement and unit-test the numeric grounding checker
  (section 6.2) before any data exists, because every later phase consumes it.
  Artifact: the checker package with a probe-style test suite.
- **G2. Corpus v1.0.** Extraction, curation, dedup, decontamination, `eval-probes`
  authoring, dataset card; publish `f1stratlab/f1-domain-corpus` v1.0 (no radio).
  Artifact: the HF dataset + its decontamination manifest.
- **G3. Instruction generation.** Regenerate `strategy-instruct` and `bot-style` pairs
  from replay traces on non-holdout GPs with the chosen teacher; checker-filter;
  report rejection rates. Artifact: dataset v1.0 finalized with SFT subsets.
- **G4. Train the 4B.** Unsloth QLoRA SFT per section 5; training manifest; GGUF
  export ladder. Artifact: `strat-gemma-lora` (4B) + GGUF, model card draft.
- **G5. Evaluate.** Full section 8 battery on the shipping GGUF, incumbent and
  base-Gemma baselines, the small ablation set. Artifact: eval report in the repo,
  results in the model card. Decision point: is the 12B run justified, and is DPO
  needed (fabrication gate)?
- **G6. Bot integration.** Ship the serving-side verifier as a gridmind-published
  artifact; shadow-run box-bot generation over recorded SSE streams; human review.
  Artifact: verifier + shadow-run report. (box-bot's own repo work is out of scope
  here; this phase delivers gridmind's half of the contract.)
- **G7. Orchestrator opt-in.** After the #205 battery exists and the incumbent is
  measured: run the gate (section 8.3) on gridmind-as-N31 through the shared-engine
  configuration path. Artifact: gate report; if green, the documented config recipe
  for running N31 on gridmind; if red, a gap list feeding v1.1.
- **G8. v1.1 and beyond.** Radio subset (when radiogate Fase 1 lands), optional DPO
  stage, 12B if gated in, Gemma 4 rerun when supported, and the v2.0 corpus refresh
  for the 2026 regulation era (coordinated with the 2026-reg drift program, epic #189,
  so the LLM's textual world and the predictors' numeric world move together).

---

## 11. Risks and limitations

- **Small-corpus overfitting and memorization.** Tens of megabytes and a few tens of
  thousands of pairs is small; over-trained LoRAs memorize. Mitigations: 1-2 epochs,
  eval-gated stopping, the `general-mix` regularizer, rank ablation. Residual risk
  accepted: the model will parrot template phrasing; that is tolerable for these two
  consumers.
- **Guardrail wash-out.** Fine-tuning can degrade instruction-following it was not
  trained on; a model tuned to always produce confident synthesis may fabricate more
  under odd prompts than base Gemma. This is exactly what the counterfactual and
  missing-stat probes measure, and why the abstention data is in the mixture from v1,
  not bolted on.
- **Structured-output brittleness at 4B.** Even with grammar-constrained decoding,
  small models can produce degenerate-but-valid JSON (empty reasoning, repeated
  sentences). The degenerate-compliance metric exists for this; the 12B is the fallback
  if the 4B fails it.
- **Quantization behavior shift.** Q4 can move refusal and citation behavior relative
  to the evaluated fp16 model, which is why the battery runs on the shipping GGUF and
  the Q8 reference isolates the delta.
- **Gemma licensing flows down.** The Gemma Terms of Use bind all derivatives; the
  artifacts cannot be Apache-2.0. Accepted, documented in the card. Gemma 4 terms must
  be re-read at upgrade time.
- **Teacher terms.** If any instruction targets come from OpenAI-generated archives,
  the OpenAI terms question (using outputs to develop other models) must be assessed;
  the design's default (local open-weights teacher via LM Studio) avoids it entirely.
- **Radio and FIA text provenance.** Both carry non-trivial redistribution questions;
  radiogate owns the radio analysis, section 9.3 owns the FIA posture. The corpus is
  designed so both subsets are severable without invalidating the model (they are
  CPT/flavor subsets, not the core SFT data).
- **2026 regulation drift.** A 2022-2025-tuned model will confidently gloss 2026
  strategy with stale concepts (no more X-mode/Z-mode confusion risks, energy
  management vocabulary changes). The v2.0 refresh is planned, and until it ships the
  bot's prompt should carry the regulation-era disclaimer for 2026 sessions.
- **Maintenance economics.** The honest counterargument to gridmind-as-N31: the
  incumbent costs $1-2 per race and is maintained by someone else. gridmind's N31 value
  is autonomy, latency control, and research output; if the gate keeps failing, the
  rational end state is "gridmind powers the bot and the chat surfaces; N31 stays on
  the OpenAI path", and that is a success, not a failure, of this design.
- **Unsloth single-vendor velocity.** Pinned versions per run, frozen configs in-repo,
  and the one-line TRL+PEFT fallback bound this risk.

---

## 12. Open questions for Victor

- **Q1. Training hardware.** What GPU (and VRAM) will the training run on? This alone
  decides 4B-only vs 4B+12B (section 5.1), and whether a one-off cloud run for the 12B
  is acceptable.
- **Q2. N31 ambition.** Is replacing gpt-5.4-mini in N31 a goal of gridmind v1, or is
  bot-first (with N31 as a gated, maybe-never opt-in) the accepted posture? The design
  recommends bot-first; confirming it changes how much weight the 12B and DPO stages
  get.
- **Q3. Regulations subset posture.** Public, gated, or recipe-only (section 9.3)?
  Recommendation is gated; needs an owner decision because it carries the project's
  name.
- **Q4. Corpus v1.0 without radio.** Ship the domain corpus before radiogate Fase 1
  delivers the radio corpus (recommended, keeps gridmind unblocked), or hold v1.0 for
  the radio subset?
- **Q5. Teacher for instruction targets.** Regenerate with a local open-weights teacher
  through LM Studio (recommended: clean provenance, no OpenAI-terms question), or reuse
  archived OpenAI-generated traces from validated runs (faster, but the card must
  disclose and the terms question must be assessed)?
- **Q6. DPO in v1.** Include the preference stage in the first training campaign, or
  hold it as the v1.1 lever if the SFT fabrication gate fails (recommended: hold)?
- **Q7. Blinded human review budget.** The synthesis-quality and bot-voice evaluations
  need a fixed sample of human judgments (realistically 50-100 items per round, Victor
  as judge). Is that budget acceptable, or should the design lean harder on
  LLM-as-judge with periodic spot checks?
- **Q8. Repo topology confirmation.** Independent public repo per the ecosystem rule
  (recommended and assumed here); confirm, since the same question was left open for
  radiogate and the answers should probably match.

---

## 13. Related documents

- `FUTURE.md` (repo root, not versioned): ecosystem plan, phases, naming, HF org
  commands (sections 10 and 11).
- `documents/research/RADIOGATE_DECEPTION_AND_AUTOLABELING.md`: the radio corpus this
  design consumes as a subset.
- `documents/research/RIVAL_AGENT_DESIGN.md`: the TFM design; independent of gridmind,
  but both feed the orchestrator's context and share the #205 evaluation substrate.
- `documents/audits/AUDIT_ML_AGENTS_EVAL.md` (epic #205): the evaluation battery this
  design uses as its acceptance gate.
- `documents/audits/AUDIT_LLM_COST_LATENCY.md` (epic #261): prompt restructure,
  timeout/retry hygiene, and the cost baseline that frames the rollout policy.
- `src/agents/strategy_orchestrator.py`: the synthesis prompt and provider switch that
  define the target task and the integration point (read-only reference; untouchable).
- `src/rag/retriever.py` and `scripts/build_rag_index.py`: the regulation extraction
  pipeline the corpus reuses.
