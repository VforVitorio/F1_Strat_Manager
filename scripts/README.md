# scripts: Top-level CLI tools

Headless command-line tools for F1 StratLab. Each script is
self-contained: it imports from `src/`, walks up to the repo root, and is
runnable directly with `python scripts/<name>.py` from any working
directory.

The interactive launcher (`f1_cli.py`) and the headless simulator
(`run_simulation_cli.py`) are the two entry points the end user sees most
of the time. Everything else is one-shot infrastructure that you run once
per machine (data + RAG index + radio corpus build) or once per release
(corpus upload).

---

## Quick reference

| Script | When to run | What it does |
|---|---|---|
| [`f1_cli.py`](f1_cli.py) | Every day, interactive demo path | Arrow-key wizard around `run_simulation_cli.py`. Walks race / driver / laps / provider selection and shells out to the headless runner. UI logic lives in [`scripts/cli/`](cli/README.md) |
| [`run_simulation_cli.py`](run_simulation_cli.py) | Every day, headless / scripted runs | Lap-by-lap multi-agent simulator. Loads the race parquet, builds the runner, ticks the orchestrator per lap, and renders a Rich Live inference panel. Auto-fetches the radio corpus for the requested GP via `ensure_radio_corpus`. Flags: `--no-llm`, `--provider {openai,lmstudio}`, `--no-real-radios`, `--whisper-model NAME`, `--radio-every N`, `--laps N-M` |
| [`debug_agent.py`](debug_agent.py) | When one agent is misbehaving | Single-agent debug harness. Builds a minimal `lap_state` from CLI args and calls one of pace / tire / situation / pit / radio / rag / orchestrator in isolation. Prints the full output dataclass |
| [`download_data.py`](download_data.py) | Once per machine | Pulls a curated subset of the dataset (~7-8 GB) from `VforVitorio/f1-strategy-dataset` on HuggingFace Hub via `src/f1_strat_manager/data_cache.py`, which selects only the load-bearing models and race data needed for simulation. Can be skipped if `f1-sim` runs first, the lazy bootstrap covers the same ground per-race |
| [`download_fia_pdfs.py`](download_fia_pdfs.py) | Once per RAG rebuild | Scrapes and downloads FIA Sporting and Technical Regulation PDFs (2023-2025) into `data/rag/documents/`. Falls back to a hard-coded URL list when scraping fails |
| [`build_rag_index.py`](build_rag_index.py) | After `download_fia_pdfs.py` | One-shot ingestion: PDF → article chunks → BGE-M3 embeddings → local Qdrant collection. Idempotent (hash-based dedup) so re-running it after adding a new PDF only ingests the new content |
| [`build_radio_dataset.py`](build_radio_dataset.py) | Once per season, corpus build | Multi-GP CLI wrapper around `RadioDatasetBuilder`. Writes per-GP `radios.parquet` + `rcm.parquet` under `data/processed/race_radios/{year}/{slug}/` and downloads radio MP3s under `data/raw/radio_audio/{year}/{slug}/driver_{N}/`. Default season: 2025. Flags: `--gps`, `--years`, `--skip-audio`, `--audio-dir`, `--skip-existing` |
| [`measure_mc_tables.py`](measure_mc_tables.py) | Once per redesign: MC table measurement | Measures six empirical tables from RAW per-race parquets: `sc_window`, `neutralisation_rate`, `gap_density`, `undercut_band`, `stop_hazard`, `clean_air`. Writes `data/mc_measured_v1.json` (machine readable) plus human-readable twins under `data/eval/`. Output is deterministic; a test asserts the committed file matches a fresh run |
| [`upload_radio_corpus.py`](upload_radio_corpus.py) | Once per release, corpus publish | Pushes both radio trees (parquets + MP3s) to `VforVitorio/f1-strategy-dataset` via `HfApi.upload_folder`, preserving the on-disk layout. Pre-flight Rich panel + auth pre-check; parquets go first (fail fast); idempotent (HF Hub deduplicates by content hash). Flags: `--year`, `--dry-run`, `--skip-parquets`, `--skip-audio`, `--commit-message` |

---

## Subdirectories

### [`cli/`](cli/README.md), interactive launcher package

UI layer for `f1_cli.py`. Contains the arrow-key picker primitives, the
Rich theme, and the subprocess runner that builds the argv for
`run_simulation_cli.py`. Pure UX, no simulation logic. See
[`scripts/cli/README.md`](cli/README.md) for the full module map.

---

## Typical workflows

**First-time setup on a fresh machine:**

```bash
# Either pre-fetch everything in one go ...
python scripts/download_data.py
python scripts/download_fia_pdfs.py
python scripts/build_rag_index.py

# ... or just run the simulator and let the lazy bootstrap pull what it needs:
python scripts/f1_cli.py
```

**Daily simulation run (interactive):**

```bash
python scripts/f1_cli.py
```

**Daily simulation run (scripted, no menu):**

```bash
python scripts/run_simulation_cli.py Sakhir HAM Mercedes --provider openai --laps 1-15
```

**Debug a single agent in isolation:**

```bash
python scripts/debug_agent.py --agent tire --gp Melbourne --lap 20 --driver NOR --team McLaren
```

**Rebuild the radio corpus for a single GP after a fix:**

```bash
python scripts/build_radio_dataset.py --gps Imola --years 2025
python scripts/upload_radio_corpus.py --dry-run        # sanity check
python scripts/upload_radio_corpus.py                  # push to Hub
```
