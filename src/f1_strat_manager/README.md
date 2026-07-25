# src/f1_strat_manager — CLI infrastructure package

Cross-cutting infrastructure that does not belong to any single domain
sub-package under `src/`. Hosts the first-run data bootstrap (HuggingFace
Hub downloader) and the friendly-name → on-disk-slug resolver used by the
radio corpus pipeline.

The package is intentionally **lightweight** — zero pandas, no NLP models,
no agent imports — so the bootstrap path stays cheap on first install and
the slug resolver can be imported by both the data-cache layer and the
NLP runner without circular dependencies.

---

## Files

| File | Description |
|---|---|
| [`data_cache.py`](data_cache.py) | First-run resolver + HuggingFace Hub downloader. Public API: `get_data_root`, `get_models_root`, `is_first_run`, `ensure_setup`, `ensure_race`, `ensure_radio_corpus`, `ensure_models`. Pulls a curated subset (load-bearing models + race data + radio metadata, ~7-8 GB total) from `VforVitorio/f1-strategy-dataset` via `snapshot_download` patterns the first time `f1-strat` / `f1-sim` runs without a cloned repo. Omits redundant model checkpoints that the production path never uses |
| [`gp_slugs.py`](gp_slugs.py) | `COUNTRY_SLUG_BY_GP` static dict + `resolve_gp_slug(gp_name)` function. The single source of truth that translates CLI / featured-laps friendly names (`Sakhir`, `Imola`, `Marina Bay`, ...) into the on-disk slugs the radio dataset builder writes (`bahrain`, `italy_imola`, `singapore`, ...). Multi-race countries (Italy, United States) carry a circuit suffix |
| `__init__.py` | Package docstring; no re-exports |

---

## `data_cache.py` — first-run bootstrap

Resolves `data/` and `data/models/` to project-relative paths and lazily
downloads a curated subset from `VforVitorio/f1-strategy-dataset` on the
HuggingFace Hub, pulling only the load-bearing assets needed for simulation.
Two trees are pulled differently to keep the first-run footprint small (around 7-8 GB instead of the full 31.7 GB repository):

| Tree | When it downloads | Why |
|---|---|---|
| `data/processed/race_radios/**` (radio parquets) | Eagerly via `_DEFAULT_MODEL_PATTERNS` | ~430 KB total — negligible vs the rest of the bundle |
| `data/raw/radio_audio/{year}/{slug}/**` (MP3s) | Lazily, per-GP, via `ensure_radio_corpus(year, gp_name)` | ~3 MB per race; only fetched when the user actually simulates that GP |
| `data/raw/{year}/{gp}/**` (FastF1 race dump) | Lazily, per-race, via `ensure_race(year, gp_name)` | Same per-race fetch pattern |
| Models tree (`data/models/**`) | Eagerly on `is_first_run()` | The orchestrator needs every model loaded at startup |

**Environment overrides:**

| Variable | Effect |
|---|---|
| `F1_STRAT_DATA_ROOT` | Override the data directory (default: `<repo>/data`) |
| `F1_STRAT_MODELS_ROOT` | Override the models directory (default: `<data_root>/models`) |
| `F1_STRAT_OFFLINE=1` | Skip every download, return whatever is already on disk |
| `F1_STRAT_NO_FIRST_RUN=1` | Suppress the first-run banner / progress bars |
| `HF_TOKEN` | Use a private HF token instead of the public dataset |

**Idempotence:** every `ensure_*` helper short-circuits before importing
`huggingface_hub` when its target tree is already populated, so a warm
install pays zero startup cost beyond the existence checks.

---

## `gp_slugs.py` — friendly name → slug

The CLI passes friendly GP names (`Sakhir`, `Marina Bay`, `Yas Island`,
`Imola`, …) but the radio dataset builder writes folders by lower-cased
country (`bahrain`, `singapore`, `united_arab_emirates`, `italy_imola`, …).
This module is the **only** place that translates between the two, so the
runner, the FastAPI endpoints, and the lazy first-run downloader stay in
sync.

Multi-race countries (Italy = Imola + Monza, United States = Miami + Austin
+ Las Vegas) get a circuit suffix appended to the country slug — see
`src/data_extraction/openf1/radio_dataset_builder.py` for the static
`_MULTI_RACE_COUNTRIES = {"Italy", "United States"}` set on the build side.

`resolve_gp_slug` accepts both forms (friendly name *and* canonical slug)
and falls through silently for the canonical form, so callers can pass the
output of a previous resolution back in without an error — useful for the
retry-after-partial-download path inside `ensure_radio_corpus`.

Imported by:

- [`src/nlp/radio_runner.py`](../nlp/radio_runner.py) — re-exports
  `COUNTRY_SLUG_BY_GP` and `resolve_gp_slug` so consumers can stay on the
  `radio_runner` namespace
- [`src/f1_strat_manager/data_cache.py`](data_cache.py) — calls
  `resolve_gp_slug` from inside `ensure_radio_corpus` to compute the
  per-GP audio directory path
