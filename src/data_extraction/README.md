# src/data_extraction: Data Extraction Utilities

Source-keyed extractors for the project's offline raw datasets. Each
subpackage corresponds to a single upstream provider, so the active OpenF1
path is not buried under historical reference scripts and the legacy code
that predates the current pipeline can be deleted later in one cohesive
batch instead of file by file.

---

## Layout

```
src/data_extraction/
├── openf1/    ← active: OpenF1 REST extractors used by the live pipeline
├── fastf1/    ← reference: FastF1 helpers, superseded by scripts/download_data.py
└── legacy/    ← kept for history, not used by any active pipeline
```

---

## `openf1/`, active

Modules in this folder are part of the live data pipeline. They are imported
by `scripts/` and consumed by the multi-agent system.

### `radio_dataset_builder.py`

Production module that turns OpenF1 team radios + Race Control Messages into
two lap-mapped parquets per Grand Prix **plus** the matching MP3 files on
disk. Wraps the prototype validated in
[`notebooks/nlp/N33_radio_dataset_builder.ipynb`](../../notebooks/nlp/N33_radio_dataset_builder.ipynb)
and is the canonical upstream for the future N29 Radio Agent.

What it builds, per GP (under `{output_dir}/{year}/{slug}/`):
- `radios.parquet`, team radios filtered by structural rule (lap not
  in formation/race-start, lap before chequered flag), 10-column schema with
  the `audio_path` column pointing to the downloaded MP3 (relative to the
  audio root)
- `rcm.parquet`, race control messages mapped to laps via OpenF1's own
  `lap_number` when present and interval matching otherwise, 13-column
  schema (driver-specific RCMs use the targeted driver's intervals,
  track-wide RCMs fall back to the leader)
- `{audio_dir}/{year}/{slug}/driver_{N}/*.mp3`, one file per radio row,
  fetched with the same retry-enabled session as the metadata calls. Re-runs
  are idempotent: existing files are skipped, missing ones are downloaded.

Both trees use the identical `{year}/{slug}/` substructure on purpose, so
a consumer that knows the GP builds the radio parquet path and the MP3
directory path from the same fragment.

The two metadata builds share a `SessionBundle` so each GP only costs four
HTTP calls (`/sessions`, `/laps`, `/team_radio`, `/race_control`) instead of
the naive six. The same `requests.Session` is reused for both metadata and
audio fetches across the whole multi-GP loop, so 429 throttling on either
host is absorbed by the same exponential-backoff policy.

Sprint weekends: OpenF1 returns Sprint sessions with `session_type="Race"`,
so the discovery loop and `resolve_session` both filter client-side by
`session_name == "Race"`. Without that filter, China / Miami / Spa / Austin
/ São Paulo / Qatar overwrite the Sunday GP parquet with Sprint data.

Multi-race countries: a few countries host more than one Grand Prix per
season: Italy (Imola + Monza) and the United States (Miami + Austin + Las
Vegas), and the naive `country_name.lower().replace(" ", "_")` slug
collides for both. The builder fixes this by appending the OpenF1
`circuit_short_name` (lowercased) to the country slug for the countries in
the static `_MULTI_RACE_COUNTRIES = {"Italy", "United States"}` set, which
produces `italy_imola` / `italy_monza` /  `united_states_miami` /
`united_states_austin` / `united_states_las_vegas`. Single-race countries
(Bahrain, Australia, ...) are unaffected and stay on the cheap legacy slug,
so the disambiguation never reshuffles existing GPs on disk. Adding another
double-header country in a future season is a one-line change to the
constant followed by a rebuild of just the affected GPs.

What it does **not** do: no Whisper / Nemotron transcription, no
sentiment / intent / NER inference. The static build only goes as far as
the MP3 file on disk. The runtime consumer that turns those MP3s into
structured radio events lives in
[`src/nlp/radio_runner.py`](../nlp/radio_runner.py), a
`RadioPipelineRunner` that lazily transcribes per-lap slices with Whisper
and hands them to the N29 Radio Agent. The simulation CLI
(`scripts/run_simulation_cli.py`) wires this runner in automatically and
auto-fetches the audio tree for the requested GP via
`ensure_radio_corpus` in `src/f1_strat_manager/data_cache.py`, so a user
running `f1-sim Bahrain NOR McLaren` never needs to touch the builder or
download the MP3s by hand. See
[`notebooks/agents/N34_radio_runner_smoke.ipynb`](../../notebooks/agents/N34_radio_runner_smoke.ipynb)
for the end-to-end smoke test (28 radios + 76 RCMs on Bahrain 2025,
lap 4 emits a PROBLEM alert through N29).

**Publishing the corpus:** `scripts/upload_radio_corpus.py` is the
companion helper that uploads both trees (the parquets under
`data/processed/race_radios/{year}/{slug}/` and the MP3s under
`data/raw/radio_audio/{year}/{slug}/`) to the
`VforVitorio/f1-strategy-dataset` HuggingFace Dataset. It preserves the
on-disk layout, parquets upload first (fail fast), and re-runs are
idempotent because HF Hub deduplicates by content hash. CLI flags cover
`--year`, `--dry-run`, `--skip-parquets`, `--skip-audio`, and
`--commit-message`.

**Run the smoke test (single GP, isolated tmpdir):**

```bash
python -m src.data_extraction.openf1.radio_dataset_builder
```

This builds the radio + RCM tables for Bahrain 2025 in a temporary directory,
runs the audio download stage against the same tmpdir, and prints `head(10)`
for radios, RCMs, and the post-audio `audio_path` column so you can
sanity-check the schema, the filter attrition, and the MP3 layout in one run.

**Run the multi-GP build via the CLI wrapper:**

```bash
# Default — full 2025 calendar into data/processed/race_radios/ + data/raw/radio_audio/
python scripts/build_radio_dataset.py

# Subset of GPs (case-insensitive country names)
python scripts/build_radio_dataset.py --gps Bahrain Australia

# Historical seasons
python scripts/build_radio_dataset.py --years 2023 2024 2025

# Parquets only, no MP3 download (fast iteration)
python scripts/build_radio_dataset.py --skip-audio

# Custom MP3 destination
python scripts/build_radio_dataset.py --audio-dir data/raw/radio_audio

# Resume after a crash without re-downloading already-built GPs
python scripts/build_radio_dataset.py --skip-existing
```

### `intervals_extractor.py`

Reference script for pulling inter-car interval data from
`/v1/intervals`. Currently only maps the session key for the 2023 Spanish
GP. Kept here because the OpenF1 intervals shape is the source of truth
for any future undercut/DRS-window dataset, but the canonical pipeline for
gaps now lives in N11/N12.

---

## `fastf1/`, reference

### `session_extractor.py`

`extract_f1_data(year, gp, session_type)`: FastF1 session loader that pulls
laps, pit stops and weather and writes parquets under `data/raw/`. Initially
scoped to Spain 2023. **Superseded** by:

- The `notebooks/data_engineering/N01–N04` notebooks for the canonical
  feature-building pipeline
- `scripts/download_data.py` for the actual raw + processed dataset bundle
  (pulled from Hugging Face Hub)

Kept here as a reference because it documents the original FastF1 cache
strategy that the project later automated.

---

## `legacy/`, kept for history

These files are not imported by any active pipeline. They represent earlier
phases of the project (computer vision experiments, video downloading, the
pre-N33 radio dump) and are kept so the git history of the data pipeline
stays self-contained.

| File | What it was | Why it's legacy |
|---|---|---|
| `image_augmentation.py` | Albumentations pipeline for the YOLO car-team image dataset (10 class names, 250 samples per class target) | Hard-coded absolute path from the original dev machine; computer vision direction was abandoned |
| `video_downloader.py` | yt-dlp wrapper for downloading Creative Commons F1 highlight videos | Vision experiments dropped; videos now come from the OpenF1 API + FastF1 cache |
| `extract_radios.ipynb` | Original notebook radio dump (pre-N33), pulled ~110 radios per session in 2023 | Replaced by `openf1/radio_dataset_builder.py`; kept as a baseline reference for the OpenF1 pipeline's expected cardinality |

---

## Output layout

All extractors write into the project's `data/` tree, which is **never**
imported through Python, paths are relative to the working directory the
caller invokes the script from. By convention you should run the scripts
from the repo root so paths like `data/raw/...` and `data/processed/...`
resolve correctly.

```
data/
├── raw/                                     ← FastF1 / OpenF1 raw extracts
│   ├── {gp}_{year}_laps.parquet
│   ├── {gp}_{year}_pitstops.parquet
│   ├── {gp}_{year}_weather.parquet
│   ├── {gp}_{year}_openf1_intervals.parquet
│   └── radio_audio/                         ← OpenF1 radio MP3s
│       └── 2025/
│           ├── bahrain/                     ← single-race country: country slug
│           │   ├── driver_1/   *.mp3
│           │   ├── driver_44/  *.mp3
│           │   └── ...
│           ├── italy_imola/                 ← multi-race country: country_circuit
│           │   └── ...
│           ├── italy_monza/
│           │   └── ...
│           ├── united_states_miami/
│           │   └── ...
│           ├── united_states_austin/
│           │   └── ...
│           ├── united_states_las_vegas/
│           │   └── ...
│           └── ...
└── processed/
    └── race_radios/                         ← OpenF1 radio + RCM corpus
        └── 2025/
            ├── bahrain/
            │   ├── radios.parquet           ← team radios (10 cols, incl. audio_path)
            │   └── rcm.parquet              ← race control (13 cols)
            ├── italy_imola/
            │   ├── radios.parquet
            │   └── rcm.parquet
            ├── italy_monza/
            │   ├── radios.parquet
            │   └── rcm.parquet
            ├── united_states_miami/
            │   ├── radios.parquet
            │   └── rcm.parquet
            └── ...
```
