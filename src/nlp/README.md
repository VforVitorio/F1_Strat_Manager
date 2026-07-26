# src/nlp: Radio NLP modules

Two modules, both live:

1. **`radio_runner.py`**, the replay-time consumer of the static OpenF1 radio
   corpus: transcription via Whisper, plus the adapter that turns raw MP3s into
   the dict shape N29 expects.
2. **`rcm_state.py`**, the race-control state tracker that decides whether a
   neutralisation is currently active.

**The inference models are not here.** Sentiment, intent and NER are loaded and
run inside the N29 Radio Agent at `src/agents/radio_agent.py`, which is
self-contained. If you are looking for the code that classifies a radio message,
that is the file to open.

The Jupytext exports that used to sit alongside these two (`pipeline.py`,
`sentiment.py`, `ner.py`, `radio_classifier.py`) moved to
[`legacy/nlp_standalone/`](../../legacy/nlp_standalone/). They predated the
unified N24 pipeline and nothing imported them, but their presence here made this
folder look like the implementation, which is how someone ends up editing the
wrong file.

---

## Active

### `radio_runner.py`

Replay-time bridge from the static OpenF1 radio corpus to the N29 Radio Agent.

| Symbol | Type | Role |
|---|---|---|
| `RadioPipelineRunner` | dataclass | One per simulation run. Loads the per-GP `radios.parquet` + `rcm.parquet`, lazily transcribes each referenced MP3 with Whisper, exposes `radios_for_lap(lap_number)` for the CLI loop |
| `WhisperTranscriber` | class | Process-local Whisper holder. `ensure_loaded()` defers the model load until the first transcription so a fully-warm cache pays zero load cost |
| `_get_whisper(model_name)` | factory | Module-level singleton accessor. Returns the same Whisper instance across runners as long as the requested model name matches |
| `COUNTRY_SLUG_BY_GP` | dict | Re-exported from `src.f1_strat_manager.gp_slugs` for convenience, friendly GP name → on-disk corpus slug |
| `resolve_gp_slug(name)` | function | Re-exported resolver. CLI / FastF1 GP name → corpus slug |

**JSON transcript cache:** keyed by the *normalised* (forward-slash) relative
`audio_path`, persisted at
`data/processed/radio_nlp/{year}/{slug}/transcripts.json`. Each entry stamps
the Whisper model name so a `--whisper-model base` re-run cleanly invalidates
a cache that `turbo` populated. Stale-model entries are dropped on load.

**Why the runner lives here, not under `src/agents/`:** the runner does no
inference, no LangGraph, no LLM, just transcription + a thin parquet → dict
adapter. Sitting next to the legacy NLP modules keeps all the radio-NLP
plumbing in one place and lets the lazy first-run downloader
(`src/f1_strat_manager/data_cache.ensure_radio_corpus`) call `resolve_gp_slug`
without dragging the full `src.agents` package init (which loads RoBERTa /
BERT / Whisper at module-import time).

**End-to-end smoke test:**
[`notebooks/agents/N34_radio_runner_smoke.ipynb`](../../notebooks/agents/N34_radio_runner_smoke.ipynb)
,  13 cells validating cache hit/miss, per-lap distribution, transcript
sanity, and the N29 round-trip via `run_radio_agent_from_state` on
Bahrain 2025 (28 radios + 76 RCMs, lap 4 emits a PROBLEM alert).

**Wired into:** `scripts/run_simulation_cli.py` (default-on, suppressed by
`--no-real-radios`). The CLI calls `ensure_radio_corpus(year, gp_name)` once
at startup to fetch the per-GP MP3 tree from the Hub and then constructs the
runner against the local data root.

---

## Legacy (Jupytext exports)

These files are early notebook exports (week-4 NLP development) and are not
imported by the agent pipeline today. They use `../../outputs/week4/` paths
that no longer exist and are listed here only so a future cleanup pass knows
what to delete.

| File | Source notebook | What it was |
|---|---|---|
| `sentiment.py` | N20 / week-4 RoBERTa | Fine-tuning loop for `roberta-base` 3-class sentiment (87.5% test accuracy on 530 radio messages) |
| `ner.py` | N22 / week-4 BERT-large | NER training with `bert-large-conll03` BIO tagging for F1 entities |
| `radio_classifier.py` | N21 / week-4 RoBERTa | SetFit intent classifier (ORDER, INFORMATION, QUESTION, WARNING, STRATEGY, PROBLEM) |
| `pipeline.py` | N06 / week-4 model merging | Integrated sentiment + intent + NER pipeline; superseded by the N24 unified pipeline that lives inside `src/agents/radio_agent.py` |

---

## Developed in

- [`notebooks/nlp/N20_bert_sentiment.ipynb`](../../notebooks/nlp/N20_bert_sentiment.ipynb)
- [`notebooks/nlp/N21_radio_intent.ipynb`](../../notebooks/nlp/N21_radio_intent.ipynb)
- [`notebooks/nlp/N22_ner_models.ipynb`](../../notebooks/nlp/N22_ner_models.ipynb)
- [`notebooks/nlp/N23_rcm_parser.ipynb`](../../notebooks/nlp/N23_rcm_parser.ipynb)
- [`notebooks/nlp/N24_nlp_pipeline.ipynb`](../../notebooks/nlp/N24_nlp_pipeline.ipynb)
- [`notebooks/nlp/N33_radio_dataset_builder.ipynb`](../../notebooks/nlp/N33_radio_dataset_builder.ipynb)
- [`notebooks/agents/N34_radio_runner_smoke.ipynb`](../../notebooks/agents/N34_radio_runner_smoke.ipynb)
