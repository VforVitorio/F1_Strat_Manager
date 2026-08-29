# AUDIT P2 — Loading & startup performance (cross-cutting)

**Auditor:** Fable 5 · **Date:** 2026-07-04 · **Repo:** `F1_Strat_Manager` (read-only pass, no code changed)
**Scope:** boot/startup + data loading (I/O) across the three surfaces — CLI (`f1-strat`/`f1-sim`), Arcade, Streamlit/web — plus the first-run HF download path.
**Out of scope (owned elsewhere):** per-lap runtime compute — agents' inference, the 500-sample Monte Carlo, `get_lap_state` per-lap cost → **P2b**. FastAPI endpoint design/async correctness → **P1**. Arcade rendering/60 FPS mutation → **P3**.
**Hard constraints honored in every remedy:** backend stays FastAPI; LLM = OpenAI / LM Studio, never Anthropic; UNTOUCHABLE (duplicate before modifying / additive entry points only): `scripts/run_simulation_cli.py`, `src/agents/` internals, `notebooks/**`, `legacy/**`.

---

## 0. Executive summary

The felt pain ("la carga de las cosas") decomposes into **four root causes**, all confirmed against current code and three of them **measured on this machine** during the audit:

1. **~30 s of model loading happens at Python *import* time, not at use time.** `src/agents/radio_agent.py:367` builds `CFG = RadioAgentCFG()` at module level, which loads RoBERTa-sentiment + SetFit/ModernBERT-intent + BERT-large-NER onto the GPU inside `__post_init__` (lines 348–353). `src/agents/strategy_orchestrator.py:65–81` imports every sub-agent module at its own top, so *anything* that imports the orchestrator pays it: `f1-sim --help`, every menu-spawned subprocess, the Arcade warmup thread, the backend's first strategy request. **Measured: 30.3 s** for the import chain, warm OS cache.
2. **Nothing is reused across runs.** The `f1-strat` menu spawns a **fresh subprocess per simulation** (`scripts/cli/runner.py:86`), so a second run — or the second driver of a Head-to-Head — pays the full ~40–60 s boot again. Whisper, agents, parquets: all reloaded.
3. **The Arcade session cache is stored in the most expensive possible shape.** `SessionLoader` pickles ~2.5 M `FrameData` Python objects per race (~300 MB/race, 1.9 GB across 6 races). **Measured: 8.0 s** to unpickle the warm cache — and it happens *synchronously on the UI thread*, freezing the menu window.
4. **First-run downloads are split across three uncoordinated moments with broken progress UX.** `ensure_setup` runs the full 7–8 GB `snapshot_download` **with progress bars disabled** under a spinner, then runs it a *second* time "for progress" when there is nothing left to show (`data_cache.py:390–398`). Whisper's 1.5 GB checkpoint and the HF-hub NLP backbones download separately, mid-boot, later.

The single highest-leverage cheap win found: **the Whisper transcripts are deterministic, already cached as JSON per GP, and the corpus is static — ship them in the HF dataset** and most users never load Whisper at all.

---

## 1. Measured baseline (this machine: Windows 11, CUDA GPU, warm OS file cache)

| Measurement | Value | How obtained |
|---|---|---|
| `import src.agents.radio_agent` (3 NLP models → device at import) | **30.3 s** | timed in this audit session |
| `import src.agents.{race_situation, tire, strategy_orchestrator}` after radio | +0.0 s (absorbed) | same run |
| `import transformers` alone | ~20 s | timed |
| `import torch` alone | ~6 s (importtime cold), 1.3 s re-warm | timed |
| `import xgboost, lightgbm, joblib` | ~7 s | timed |
| Arcade warm-cache unpickle (Suzuka 2025, 309 MB) | **8.0 s** — 20 drivers × 125 279 timeline frames = **2 505 580 `FrameData` objects** | timed |
| Arcade pickle cache on disk | **1.9 GB / 6 races** (290–378 MB each) | `du` |
| FastF1 HTTP cache, Arcade copy (`data/cache/fastf1`) | 1.9 GB | `du` |
| FastF1 HTTP cache, backend copy (`src/telemetry/cache`) | 292 MB (duplicate of the above, different dir) | `du` |
| `laps_featured_2025.parquet` | **2.3 MB** (repeated reads are noise — see F-15) | `du` |
| Whisper checkpoints in `~/.cache/whisper` | 3.6 GB total; `turbo` = 1.5 GB | `du` |
| Local `data/models/nlp` | 30 GB (curated HF pull is 7–8 GB; the rest is redundant checkpoints correctly excluded by `_DEFAULT_MODEL_PATTERNS`) | `du` |
| Radio audio tree (local, all GPs) | 191 MB (~3 MB/GP, lazily pulled per GP — good) | `du` |
| HF Hub layout vs code patterns | **verified consistent**: Hub has `data/models/**`, `data/raw/**` etc., matching `_DEFAULT_MODEL_PATTERNS` and `_CRITICAL_MODEL_FILES` (checked `tiredeg_modelA_v4.pt` exists at the expected path). The 85-day-old memory `project_hf_models_restructure` (models at Hub root) is **superseded** — no action needed. | HF API |

Un-measured but code-certain contributors: agent singleton materialisation (`_prewarm_agents`: XGB + TCN + 5×joblib + parquets, est. 5–15 s), Whisper `turbo` load (est. 10–20 s when transcripts are cold), eager transcription of uncached radios (~2–5 s/clip × ~30 clips on a first GP run), FastF1 cold session fetch in Arcade (minutes: race with telemetry + a second full **quali** session just for DRS zones).

---

## 2. Bottleneck → cause → remedy → expected win (per surface)

### 2.1 CLI (`f1-sim` direct + `f1-strat` menu)

Boot order today (all **serial**): provider env → first-run HF check → parquets → radio corpus (`ensure_radio_corpus` + `RadioPipelineRunner(eager_transcribe=True)` = Whisper load + transcribe) → `_prewarm_agents` → header → lap 1. (`run_simulation_cli.py:1584–1669`; **file is UNTOUCHABLE — every remedy below lands in the P4 duplicate or in editable callers.**)

| # | Bottleneck | Cause (file:line) | Target remedy | Expected win |
|---|---|---|---|---|
| C1 | **30.3 s paid before argparse** — even `--help`, even arg typos | Module-level `from src.agents.strategy_orchestrator import …` at `run_simulation_cli.py:136`; root cause is F-01 (eager `CFG` in `radio_agent.py:367`) | Fix F-01 (lazy CFG) + in the P4 duplicate, move heavy imports below argparse | `--help`/error paths: 30 s → <1 s; real runs: cost moves into the (parallelisable) warmup phase |
| C2 | **Every menu run re-pays the full boot** (~40–60 s); H2H = 2× sequential boots | `scripts/cli/runner.py:86` `subprocess.run` per simulation; process exits after each race | Warm-worker option: a duplicated `f1-sim` entry that loops the wizard *inside* one process (models load once per session, N races); menu keeps subprocess as fallback | 2nd and later runs: ~40–60 s → **~2–5 s** (data load only). Biggest UX lever for the demo loop |
| C3 | Serial boot: Whisper/transcription and agent prewarm never overlap | `run_simulation_cli.py:1626–1669` (sequential blocks) | In the P4 duplicate: run radio-corpus load and `_prewarm_agents` in 2 threads; boot = max(paths), not sum | est. 20–40 % off cold boot; ~10–15 s when transcripts uncached |
| C4 | First GP run per race: Whisper `turbo` load + ~30 clips transcription | `radio_runner.py:289–290` `eager_transcribe=True`; cache warms only after first run | **F-02: ship `transcripts.json` per GP in the HF dataset** (they already exist under `data/processed/radio_nlp/<year>/<slug>/`); `ensure_radio_corpus` pulls them with the audio | First-run-per-GP Whisper cost (~30–90 s + 1.5 GB checkpoint download) → **0 for most users**. Whisper becomes opt-in (`--retranscribe`) |
| C5 | First-run picker only offers locally-present races (fresh install = 1 sentinel GP) | `scripts/cli/pickers.py:183–206` `discover_races` scans disk only, though `ensure_race` (`data_cache.py:413`) can fetch any GP | Picker lists the full year calendar (canonical `data/tire_compounds_by_race.json`), marks non-local entries "will download", calls `ensure_race` on selection | Fresh `uv tool install` users get the whole season instead of one race; no manual HF steps |
| C6 | First-run 7–8 GB download shows a spinner, not progress; metadata pass runs twice | `data_cache.py:390–398`: first `_snapshot_download(show_progress=False)` does the **entire download silently**; the second "progress" call only re-walks Hub metadata (hundreds of HEAD requests) | Single `snapshot_download` with progress enabled; if fail-fast validation is wanted, use one cheap `repo_info()` call instead of a full silent snapshot | Real progress bar for the multi-minute phase (kills "is it hung?"); removes a full redundant metadata sweep |

### 2.2 Arcade (`f1-arcade`)

| # | Bottleneck | Cause (file:line) | Target remedy | Expected win |
|---|---|---|---|---|
| A1 | **Warm session load = 8.0 s, and the menu window freezes during it** | `views.py:397–400` forces one redraw then calls `SessionLoader().load()` synchronously on the pyglet thread (`views.py:424`); load itself is F-05 | Split: (a) background-thread the load with a progress state polled in `on_update` (backlog item, confirmed); (b) fix the cache shape (A2) | No frozen/ghost window; perceived boot from "app hung" → live progress |
| A2 | **Cache shape: 2.5 M Python dataclass objects pickled per race** (300 MB/race, 8 s load, 1.9 GB disk) | `data.py:371–404` `_resample_driver` converts numpy SoA → per-frame `FrameData` AoS *before* caching; `data.py:316–317` pickles the object graph | Cache the **numpy struct-of-arrays** (dict of arrays per driver) and construct `FrameData` views on demand at render time (20 objects/frame is trivial); bump `CACHE_VERSION` → v7 | Warm load **8 s → well under 1 s**; disk ~300 MB → ~40–80 MB/race; cold path also skips building 2.5 M objects |
| A3 | Cold path: serial per-driver telemetry extraction (Pool disabled after hangs) | `data.py:331–346` passes the **whole loaded FastF1 session** in each Pool task → Windows spawn pickles it per worker (recorded at the time as the cause of the historical hang; re-measured 2026-08-27 the session pickles in 0.14 s and `Pool(8)` completed 20 of 20 without hanging, so the cause is not established — see the comment at `src/arcade/config.py` `POOL_SIZE`); `config.py:158` `POOL_SIZE=1` | Don't parallelise by pickling the session: pre-slice per-driver laps in the parent and pass only the slices, or use a thread pool (extraction is pandas/numpy-heavy) | Cold extraction: ~20 drivers serial → parallel; est. 2–4× on the extraction phase without the hang risk |
| A4 | Cold path loads a **second full session (Quali)** just for DRS-zone geometry | `data.py:447–467` `_try_quali_reference` — full `quali.load(telemetry=True…)` per race | Cache DRS zones + ref-lap polyline per circuit as a small JSON artifact (geometry is static within a season); optionally publish to the HF dataset | Removes an entire FastF1 session download+parse from every cold race load |
| A5 | Strategy feed silent for ~30–60 s after replay starts (warmup + radio corpus, serial) | `strategy.py:316–317`: `_warmup_models()` (= F-01 import + 4 singletons) then `_load_radio_corpus()` (Whisper), sequential in the SimConnector thread | Overlap the two (2 threads); F-02 removes the Whisper leg; F-01 shrinks the import leg. Banner staging already exists (`Warming up…` / `Loading radio corpus…`) — keep | First decision on the dashboard: est. 30–60 s → 15–25 s (and → ~10 s once F-01/F-02 land) |
| A6 | Arcade bypasses `get_data_root()` — breaks any non-checkout install and forks the cache location | `config.py:150–152` (`REPO_ROOT = parents[2]`, FastF1 + arcade caches under repo), `strategy.py:488,506` (featured parquet, race dirs) — only the radio corpus routes through `get_data_root()` (`strategy.py:425–442`) | Route all four paths through `data_cache.get_data_root()` (keeps identical behavior in a checkout; fixes `uv tool install` + unifies caches) | Arcade works from a tool install; one cache tree instead of per-surface forks |
| A7 | Dashboard subprocess = fresh Python + PySide6 + pyqtgraph import boot | `app.py:331–335` spawns `python -m src.arcade.dashboard` | Accept (isolation of Qt vs pyglet event loops is deliberate and correct). Optional: spawn it *earlier* (at replay launch rather than after warmup start) so Qt boots concurrently | Dashboard visible sooner; zero risk |

### 2.3 Streamlit + backend (web surface)

Backend boot itself is clean — `main.py` imports are light, `mcp_tools.py` pulls only fastmcp/httpx, and the strategy endpoints correctly defer `src.agents` imports into the handlers (`endpoints/strategy.py:404,547,573,599,625,662,688,911`). The cost has been *moved*, not removed:

| # | Bottleneck | Cause (file:line) | Target remedy | Expected win |
|---|---|---|---|---|
| S1 | **First strategy request stalls ~40–60 s** (agents import chain + singletons + LLM client), with the SSE client waiting | Deferred imports meet F-01 on the first call; `simulator.py:47` imports the orchestrator at module level, and simulator itself is first imported inside the endpoint (`strategy.py:911`) | Env-gated startup prewarm (`F1_BACKEND_PREWARM=1` → background task in the existing `lifespan`, `backend/main.py:13–34`) + a fire-and-forget `/api/v1/strategy/warmup` endpoint the frontend calls when the Strategy page opens | First simulate click: 40–60 s → seconds (warmup ran while the user filled the form). Default stays lazy so telemetry-only deployments boot instantly |
| S2 | Backend keeps its **own** FastF1 HTTP cache, disjoint from Arcade's | `backend/services/telemetry_service.py:27–29` (`src/telemetry/cache`) vs `arcade/config.py:151` (`data/cache/fastf1`) | One shared cache dir resolved via `get_data_root()/cache/fastf1` (env-overridable) for both | Deduplicates GBs; a season cached by one surface is warm for the other |
| S3 | Second telemetry service exists with **no cache enabled** | `backend/services/telemetry/fastf1_client.py:17–20` calls `fastf1.get_session().load()` with no `enable_cache` of its own (only saved if the *other* module was imported first in the process) | Verify which endpoints import which service (P1 audit overlap); either delete the duplicate or move `enable_cache` to a single import-safe location | Removes a latent "every request re-downloads the session" failure mode |
| S4 | Chat: image → base64 → data-URI logic duplicated 3× and re-run per rerun; 172-line CSS blob re-injected every rerun; double `st.rerun()` per message | `frontend/services/chat_service.py:167–179, 227–234` (+ third site per `[[project-streamlit-refactor-backlog]]`); `components/chatbot/chat_history.py:21–192`; `pages/chat.py:65–122` | Consolidate one `_normalize_image_to_data_uri()`; inject CSS once per session (session_state flag); collapse the pending-message rerun dance | Snappier reruns on the chat page; sub-second scale each, but multiplied by every interaction |
| S5 | Selector data refetched over HTTP on rerun where not covered by `st.cache_data` (6 cached sites total; strategy-page selectors hit `/available-gps` etc. uncached client-side) | `frontend/utils/data_loaders.py` covers year/GP/session/driver; strategy + comparison pages call services directly | Extend `st.cache_data` (short TTL) to the remaining backend-selector calls; backend already RAM-caches the parquet (`backend/utils/laps_cache.py:18–32` — good) | Fewer HTTP round-trips per rerun; smoother page switches |

### 2.4 Cross-cutting / first-run (all surfaces)

| # | Bottleneck | Cause | Target remedy | Expected win |
|---|---|---|---|---|
| X1 | **F-01: eager model load at import** (the root of C1, A5, S1) | `radio_agent.py:367` `CFG = RadioAgentCFG()`; also eager-but-cheaper: `race_situation_agent.py:162` (4 joblib + 2 parquet), `tire_agent.py:319` (JSONs + parquet). Pace and pit are already lazy singletons — the pattern exists in-repo | Make radio's CFG a lazy singleton accessor (same shape as `_get_default_*_agent`). **Decision needed:** this is a ~3-line change *inside* untouchable `src/agents/` — either grant a sanctioned, minimal, test-covered exception, or apply it on the duplicated modules that the P4 refactor introduces. All consumers (arcade warmup, backend prewarm, CLI duplicate) benefit unchanged | Import chain 30.3 s → est. 8–12 s (torch/xgb/lgbm remain); model weights load once, at warmup time, overlappable and banner-visible |
| X2 | First-run downloads fragmented across 3 systems: HF snapshot (7–8 GB), Whisper checkpoint (1.5 GB, mid-boot), HF-hub NLP backbones (tokenizers/base models, mid-import) | `data_cache.ensure_setup` vs `whisper.load_model` vs `transformers` hub pulls | F-02 (ship transcripts) removes the Whisper leg for most users; fold an optional "prefetch everything" step into `ensure_setup` so the *one* first-run moment covers all three; keep per-GP laziness for audio/races | One predictable first-run wait with one progress UX, instead of surprise stalls on first sim |
| X3 | No shared prewarm façade — each surface hand-rolls its own (CLI `_prewarm_agents`, Arcade `_warmup_models`, backend none) | `run_simulation_cli.py:448–481`, `arcade/strategy.py:372–403` | **Additive** `src/f1_strat_manager/prewarm.py`: `prewarm(profile, on_stage=…)` with profiles (`sim`, `arcade`, `backend`), parallel internal loading, stage callbacks for banners; RAG included when profile wants it | One place to fix warmup ordering forever; consistent banners; backend/arcade/CLI-duplicate all call it |
| X4 | RAG retriever (Qdrant + BGE-M3, ~2 GB) is *not* prewarmed anywhere → first RAG-routed lap mid-race stalls | `rag/retriever.py:184–185` eager in ctor, `lru_cache` singleton lazy at process level (`:284`); absent from both prewarm lists | Include RAG in the prewarm profile (background, lowest priority) when RAG is enabled for the run | Removes a mid-race 10–20 s hiccup on the first SC/regulation query |

---

## 3. Findings register (P0 → P3)

| ID | P | Finding | Evidence |
|---|---|---|---|
| F-01 | **P0** | 3 NLP transformer models load onto the device at **module import**; every surface pays 30.3 s before doing anything useful | `src/agents/radio_agent.py:348–353, 367`; chain via `strategy_orchestrator.py:65–81`; measured |
| F-02 | **P0** | Whisper transcription cost is paid per user per GP although the corpus is static and the JSON caches already exist — they are simply not distributed | `src/nlp/radio_runner.py:509–557` (cache design is good); caches at `data/processed/radio_nlp/<year>/<slug>/transcripts.json`; HF dataset lacks them |
| F-03 | **P0** | `f1-strat` menu: full boot re-paid per simulation subprocess (and 2× for H2H) | `scripts/cli/runner.py:59–86, 117, 161` |
| F-04 | **P1** | First-run 7–8 GB download runs silently (progress disabled) then repeats the metadata pass | `src/f1_strat_manager/data_cache.py:390–398` |
| F-05 | **P1** | Arcade cache = pickled AoS of 2.5 M objects: 8 s warm load, 300 MB/race, 1.9 GB disk | `src/arcade/data.py:371–404, 316–317`; measured |
| F-06 | **P1** | Menu → replay load is synchronous on the UI thread (window freezes 8 s warm, minutes cold) | `src/arcade/views.py:396–400, 424` |
| F-07 | **P1** | Backend first strategy request pays the full import+warmup with no prewarm hook | `endpoints/strategy.py` deferred imports; `simulator.py:47`; no warmup in `backend/main.py` lifespan |
| F-08 | **P1** | Boot phases are serial everywhere (Whisper ∥ agents ∥ HF checks never overlap) | `run_simulation_cli.py:1584–1669` (untouchable → fix in duplicate); `arcade/strategy.py:316–317` (editable) |
| F-09 | **P1** | Arcade cold path loads a second full FastF1 session (Quali) only for DRS zones; extraction is serial after Pool was disabled for the session-pickling design flaw | `src/arcade/data.py:331–346, 447–467`; `config.py:158` |
| F-10 | **P2** | Two disjoint FastF1 HTTP caches (arcade 1.9 GB vs backend 292 MB); plus a second, cache-less FastF1 client module in the backend | `arcade/config.py:151` vs `backend/services/telemetry_service.py:27–29`; `backend/services/telemetry/fastf1_client.py:17–20` |
| F-11 | **P2** | Arcade path resolution bypasses `get_data_root()` → broken under `uv tool install`, forks cache locations | `arcade/config.py:150–152`, `arcade/strategy.py:488, 506` |
| F-12 | **P2** | Fresh-install CLI picker exposes only downloaded races (one sentinel GP) despite `ensure_race` existing | `scripts/cli/pickers.py:183–206`; `data_cache.py:413–434` |
| F-13 | **P2** | RAG (Qdrant + BGE-M3) never prewarmed → first routed lap stalls mid-race | `src/rag/retriever.py:148, 184–185, 284` |
| F-14 | **P2** | Streamlit chat: triplicated image encoding, per-rerun CSS re-injection, double-rerun message flow | `frontend/services/chat_service.py:167–179, 227–234`; `components/chatbot/chat_history.py:21–192`; `pages/chat.py:65–122`; corroborates `[[project-streamlit-refactor-backlog]]` |
| F-15 | **P3** | `laps_featured_2025.parquet` read 3× per CLI run (CLI full + PaceAgent twice, column-pruned) — file is only 2.3 MB, so this is hygiene, not latency | `run_simulation_cli.py:1614`; `src/agents/pace_agent.py:185–194, 209–212` |
| F-16 | **P3** | `race_state_manager` construction uses row-wise `.apply` + per-lap masks — fine at current sizes (<50 ms claim plausible); revisit only if live-timing scales it | `src/simulation/race_state_manager.py:44–61, 128–148` |
| F-17 | **P3** | Eager-but-cheap `CFG` at import in situation/tire agents (joblib + small parquets, ~0 s measured once torch is resident) — fold into F-01's pattern fix opportunistically, not on its own | `race_situation_agent.py:162`, `tire_agent.py:319` |

**Stale-memory correction:** `MEMORY.md` says Arcade `SessionLoader` uses `multiprocessing.Pool(processes=8)` — current code defaults to **serial** (`POOL_SIZE=1`, `config.py:158`) after the Pool hung on Windows spawn; and `project_hf_models_restructure` (Hub `models/` at root) is superseded — the Hub is nested under `data/` and matches the code. Update memory when this audit is filed.

---

## 4. Shared caching & prewarm strategy (the target architecture)

**One resolver, one cache tree.** `data_cache.get_data_root()` is already the designed single source of truth — finish the job: Arcade's four hardcoded paths (F-11) and the backend FastF1 cache (F-10) route through it. Resulting layout: `<data_root>/cache/fastf1` (shared HTTP cache), `<data_root>/cache/arcade` (session caches), `<data_root>/processed/radio_nlp` (transcripts), `<data_root>/rag` (Qdrant). Everything env-overridable via the existing `F1_STRAT_DATA_ROOT`.

**Distribute derived artifacts instead of recomputing them.** The pattern that already works (per-GP radio audio, lazily pulled) generalizes:
- transcripts.json per GP → HF dataset (F-02);
- DRS-zone + ref-lap geometry JSON per circuit → HF dataset or repo (A4);
- (optional, later) prebuilt Arcade SoA session caches for the flagship demo races.

**One prewarm façade (additive).** New `src/f1_strat_manager/prewarm.py` — profiles `sim` / `arcade` / `backend`, parallel loading internally (agents ∥ Whisper-if-needed ∥ RAG-if-enabled), stage callbacks so each surface renders its own banner (Rich status, arcade dashboard banner, backend log). CLI duplicate, Arcade `_warmup_models`, and the backend prewarm flag all delegate to it. This is the mechanism that makes F-08 fixable once instead of three times.

**Warm-process reuse.** The menu (`f1-strat`) is the only surface that throws the warm process away (F-03). A duplicated in-process wizard loop ("run another race?" without exiting Python) converts the 30–60 s per-run tax into a once-per-session cost — this composes with everything above and is the single biggest repeat-use win.

### Lazy vs eager policy (decision table)

| Asset | `--help` / menus | CLI sim boot | Arcade replay | Arcade+strategy | Backend boot | Backend 1st strategy call | Streamlit boot |
|---|---|---|---|---|---|---|---|
| torch/transformers import | never | warmup phase | never | warmup thread | never (unless prewarm flag) | yes (or prewarmed) | never |
| NLP radio models (F-01) | never | warmup, parallel | never | warmup thread | prewarm flag | yes | never |
| XGB/TCN/LightGBM singletons | never | warmup, parallel | never | warmup thread | prewarm flag | yes | never |
| Whisper | never | **only if transcripts missing** (post F-02) | never | same as CLI | never | only voice endpoints | never |
| RAG (Qdrant+BGE-M3) | never | background, low prio | never | background | prewarm flag (low prio) | first use | never |
| Race parquets / session data | never | eager (small) | eager (cache-first) | eager | RAM-cached on first request (`laps_cache.py`) | cached | via backend |
| HF snapshot (models) | check only | first-run only | first-run only | first-run only | deploy-time | — | — |
| Per-GP race + radio + transcripts | never | lazy per GP (`ensure_race`/`ensure_radio_corpus`) — keep | lazy per GP | lazy per GP | lazy | lazy | lazy |

### Cold-start budgets (acceptance targets, measured on this machine's class of hardware)

| Scenario | Today (est./measured) | Budget after Phase 1–2 | Budget after Phase 3 |
|---|---|---|---|
| `f1-sim --help`, arg errors | ~30 s | — (needs P4 duplicate) | **< 1 s** |
| `f1-sim`, warm data + warm transcripts → lap 1 | ~45–60 s | ~25–35 s (parallel warmup, no Whisper) | **≤ 20 s** |
| `f1-strat` menu, **2nd+ run** in a session | ~40–60 s | ~40–60 s (unchanged) | **≤ 5 s** (warm worker) |
| Arcade menu → replay visible (cached race) | ~8–10 s, window frozen | **≤ 2 s, window responsive** | ≤ 2 s |
| Arcade first strategy decision on dashboard | ~30–60 s | ~15–25 s | ≤ 15 s |
| Arcade cold race (never fetched) | minutes, frozen window | minutes, responsive + progress; no quali session | 2–4× faster extraction |
| Streamlit: click Simulate on Strategy page (backend cold) | 40–60 s stall on first call | **≤ 5 s** (warmup fired at page open) | ≤ 5 s |
| Fresh install first run | 7–8 GB silent spinner + later surprise Whisper/HF pulls | one progress-visible download; **no silent phase > 5 s** | same |

---

## 5. Phased, chunkable plan (each chunk = one issue/PR; S/M/L effort)

**Phase 1 — quick wins, no untouchable files, no behavior risk (1 sprint)**
1. **[S] F-04:** `ensure_setup` single-pass download with visible progress (`data_cache.py` is editable). Optionally swap fail-fast probe to `repo_info()`.
2. **[S] F-02:** upload existing `transcripts.json` per GP to the HF dataset; add the pattern to `ensure_radio_corpus` + `_DEFAULT_MODEL_PATTERNS`. Whisper becomes the fallback, not the default cost.
3. **[S] F-10/F-11:** route Arcade + backend caches/paths through `get_data_root()`; one shared FastF1 cache dir. Verify the duplicate backend FastF1 client (F-10b) and remove/fix cache enablement placement.
4. **[S] F-07:** backend `F1_BACKEND_PREWARM` flag in `lifespan` + `/strategy/warmup` endpoint; Streamlit Strategy page fires it on open.
5. **[S] F-12:** full-calendar race picker with lazy `ensure_race` download-on-select.

**Phase 2 — structural loading fixes (1–2 sprints)**
6. **[M] F-05 + F-06:** Arcade SoA numpy cache (`CACHE_VERSION` v7) + background-threaded `SessionLoader.load()` with progress states in `MenuView`.
7. **[M] F-01 (decision gate):** lazy `RadioAgentCFG` accessor. Requires either a sanctioned minimal edit inside `src/agents/` (with a regression run: `f1-sim Sakhir HAM Mercedes --no-llm --laps 1-10` diffed against the original) or landing it on the P4 duplicated modules. Fold F-17 in opportunistically.
8. **[S] F-08 (arcade half):** overlap `_warmup_models` ∥ `_load_radio_corpus` in `SimConnector`; spawn the dashboard subprocess earlier (A7).
9. **[M] X-03:** additive `prewarm.py` façade with profiles + stage callbacks; wire Arcade + backend to it. Include RAG (F-13).

**Phase 3 — CLI duplicate & cold-path work (post-P4 kickoff, 2+ sprints; coordinates with the P4 audit)**
10. **[L] C1/C2/C3:** duplicated CLI (`run_simulation_cli_v2.py` or `f1-sim` v2 entry): argparse before imports, parallel staged boot via `prewarm.py`, in-process wizard loop for warm re-runs. Original stays untouched as the PMV.
11. **[M] F-09:** Arcade cold-path extraction parallelism (pre-sliced per-driver inputs) + DRS/ref-lap geometry JSON cache per circuit.
12. **[M] F-14 + S5 batch:** Streamlit rerun hygiene (image-encoding helper, CSS once, cache_data on selector services) — coordinate with the P0 frontend migration so effort isn't spent on pages being replaced.

Dependency notes: (2) unlocks most of C4/A5-Whisper without touching any protected file. (7) is the only item requiring an untouchable-rule decision — everything else is editable surface. (9) is the keystone for keeping the three surfaces' boot behavior converged long-term; (10) consumes it.

---

## 6. Verification protocol (per fix)

- Re-run the three timed probes from §1 (agents import chain; arcade unpickle; library imports) and record in the PR description — the budgets in §4 are the acceptance bars.
- CLI regression (any change near boot): `python scripts/run_simulation_cli.py Sakhir HAM Mercedes --no-llm --laps 1-10` output diffed vs. original (per `[[project-cli-refactor-backlog]]` protocol); LLM smoke run for provider-path changes.
- Arcade cache change: warm load timing + one full replay smoke (driver + rival + strategy mode) + `CACHE_VERSION` bump so stale pickles self-invalidate.
- First-run flow: test in a scratch `F1_STRAT_DATA_ROOT` with `HF_HUB` reachable, confirm single progress-visible download and a runnable sentinel race.
