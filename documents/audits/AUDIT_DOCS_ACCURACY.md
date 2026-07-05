# AUDIT — Documentation accuracy (metrics + structural claims)

**Auditor:** Fable 5 · **Date:** 2026-07-05 · **Repo:** `F1_Strat_Manager` (read-only pass, no docs or code changed)
**Scope:** factual accuracy of every published claim in `README.md`, `INSTALL.md`, `ARCHITECTURE.md`, `CONTRIBUTING.md`, `ROADMAP.md`, the docs site (`docs/pages/*.md`, `docs/app/*.js`) and the project `CLAUDE.md`, cross-checked against (a) the AUTHORITATIVE final figures in `documents/thesis/F1StratLab_TFG_thesis.pdf` (ch. 5-6, esp. Tabla 6.1 p. 113) and `documents/thesis/F1StratLab_IEEE_technical_report.pdf` (consolidated results tables), and (b) the actual code (argparse signatures, routers, workflows, pyproject).
**Out of scope:** editing the thesis or IEEE PDFs themselves; the `site/` directory (gitignored MkDocs-era build); `docs/pages/changelog.md` entries that are verbatim CHANGELOG history (annotation policy is an open question, not a finding).
**Rule applied throughout:** where the notebook-era number and the thesis/IEEE final number differ, the thesis/IEEE final is authoritative (per `project_ieee_paper` memory and the thesis's own statement on p. 98 that only re-runnable figures are reported).

---

## 0. Executive summary

The docs site (`docs/pages/`) is in good shape after the June audit sprints: its hero framing ("seven ML models, six LangGraph sub-agents and one orchestrator") matches the thesis, and its changelog already anchors the pace model to the final 0.4104 s figure. The rot is concentrated in three places:

1. **`ROADMAP.md` (and the docs roadmap page) still publish two superseded metrics as achievements.** Lap-time MAE **0.392 s** appears 4 times in ROADMAP.md and once in `docs/pages/roadmap.md`; the thesis final (Tabla 6.1, p. 113) and the IEEE report both say **0.4104 s / 0.410 s**. Sentiment accuracy **87.5%** appears 3 times in ROADMAP.md and twice in `docs/pages/roadmap.md`; the thesis (p. 98) explicitly rules that figure out ("el encabezado del cuaderno N20 anuncia un 87,5% ... el classification_report devuelve 0,84; se reporta esta segunda cifra como autoritativa") and the IEEE report publishes **0.84**. ROADMAP.md also still marks tire degradation as "pending formal evaluation" when the thesis/IEEE publish final numbers (global MAE 0.7078 s, C2 fine-tune 0.5501 s).
2. **Three copy-paste commands in root docs are broken.** README and INSTALL show `f1-sim VER Melbourne "Red Bull Racing"` (positional order is `gp_name driver team`, so VER is parsed as the GP), INSTALL's verification command uses a nonexistent `--lap-range 1 1` flag (real flag: `--laps 1-1`), and CONTRIBUTING's data-bootstrap one-liner imports `f1_strat_manager.data_cache` (the installed package root is `src.`, so the import fails).
3. **`ARCHITECTURE.md` links to a docs tree that no longer exists** (`docs/architecture.md`, `docs/arcade/*.md`, `docs/diagrams/*.drawio`, etc. - 11 dead relative links; the pages live under `docs/pages/` and the diagrams under `documents/dev_docs/diagrams/`).

**One suspected inconsistency is RESOLVED as a non-issue:** the "NLP latency 47.8 ms vs 43.7 ms" contradiction does not exist. No document says 43.7 ms; the string comes from **243.7 ms**, which is the RAG retrieval P95 (thesis p. 101, IEEE report). The published NLP pipeline figures are mean 47.8 ms / P95 59.4 ms (IEEE report table; thesis quotes the P95), and `docs/pages/changelog.md` matches them. The only genuine latency divergence is `docs/pages/thesis.md` reporting a **regenerated** mean of 42.1 ms from `data/eval/nlp_pipeline_cpu.md` (42.069 ms measured), which is a different, later measurement run, self-labelled as regenerated; it needs a reconciliation note, not a correction.

Counts also drift: README calls the system "seven specialised agents coordinated by an orchestrator" and ROADMAP says "Seven specialised sub-agents (N25-N30)" and "eight ML predictive models"; the thesis and IEEE report are unambiguous (**six** specialised sub-agents + one supervisor orchestrator N31; IEEE: "a family of six supervised predictors"; thesis objective: "familia de siete modelos predictivos"; docs count seven by including circuit clustering).

---

## 1. Authoritative baseline (thesis + IEEE report, extracted this audit)

| Metric | Final published value | Where extracted |
|---|---|---|
| Lap-time delta MAE (2025 holdout) | **0.4104 s** (IEEE rounds to 0.410; R² 0.9947; persistence baseline 0.408) | Thesis Tabla 6.1 p. 113; IEEE report results table |
| Tire degradation TCN | global MAE **0.7078 s** (R² 0.605); C2 fine-tune **0.5501 s** | Thesis p. 113; IEEE (0.708 / 0.550) |
| Overtake (N12 LightGBM) | AUC-PR **0.5491**, AUC-ROC 0.8758 (IEEE: 0.549 / 0.876, base rate 0.076) | Thesis p. 113; IEEE |
| Safety car (N14 LightGBM) | AUC-PR **0.0723** vs baseline 0.0432, lift **1.67x** | Thesis p. 113/115; IEEE (0.072 / 0.043) |
| Pit duration (N15 HistGBT) | P50 MAE **0.487 s** vs baseline 0.555 s, coverage 70.5%, pinball P05/P95 0.038/0.110 | Thesis pp. 96, 113; IEEE |
| Undercut (N16 LightGBM) | AUC-ROC **0.7708**, AUC-PR **0.6739** (baseline 0.345), lift 1.95x | Thesis p. 113; IEEE (0.771 / 0.674) |
| Sentiment (RoBERTa) | accuracy **0.84**, macro F1 0.75 (87.5% notebook header explicitly ruled non-authoritative) | Thesis p. 98 + Tabla 6.1; IEEE |
| Intent (SetFit + ModernBERT) | accuracy 0.61, macro F1 **0.5338** | Thesis p. 113; IEEE (0.53) |
| NER (BERT-large BIO) | token F1 **0.4151** | Thesis p. 113; IEEE (0.415) |
| NLP pipeline latency (GPU) | mean **47.8 ms**, P95 **59.4 ms** | IEEE report table + prose; thesis p. 74 (P95) |
| RAG retrieval | 2,279 chunks; Content P@5 0.800; MRR 0.235; **P95 243.7 ms** | Thesis pp. 101, 113; IEEE |
| Radio corpus | **530** hand-labelled messages (sentiment split 371/79/80); intent subset 529 (370/79/80); +28 msgs / 76 RCMs Bahrain sub-corpus | Thesis pp. 97-98; IEEE |
| System shape | **six** specialised sub-agents (Pace, Tire, Race Situation, Pit Strategy, Radio, RAG) + orchestrator N31; "familia de siete modelos predictivos" (thesis) / "family of six supervised predictors" (IEEE) | Thesis pp. 18, 113; IEEE report |
| Clustering | K=4, silhouette 0.201, fitted 2023-2024, applied to 2025 without refit | Thesis p. 113 |

Regenerated local eval artifacts (`data/eval/`, produced by N33/N30B, the source for `docs/pages/thesis.md`) differ slightly and are a separate measurement lineage: pace MAE 0.410 (matches), NLP mean 42.069 ms / P95 44.246 ms (`data/eval/nlp_pipeline_cpu.md`), Whisper mean 233.9 ms.

---

## 2. Inconsistency register (metric claims)

| # | Claim | Where it appears (doc side) | Current value | Authoritative value | Source of truth |
|---|---|---|---|---|---|
| M-01 | Lap-time MAE | `ROADMAP.md:100`, `ROADMAP.md:117`, `ROADMAP.md:583`, `ROADMAP.md:612`; `docs/pages/roadmap.md:267` | 0.392 s | **0.4104 s** | Thesis Tabla 6.1 p. 113; IEEE table (0.410). Already correct: `docs/pages/changelog.md:86` (0.4104 anchor), `docs/pages/thesis.md:47` (0.410), `docs/app/home.js:6` (0.41 s) |
| M-02 | RoBERTa sentiment accuracy | `ROADMAP.md:251`, `ROADMAP.md:280`, `ROADMAP.md:586`; `docs/pages/roadmap.md:301`, `docs/pages/roadmap.md:303` | 87.5% | **0.84** (84%) | Thesis p. 98 (declares 0.84 the only reproducible figure); IEEE table. The 87.5% is the N20 notebook header the thesis explicitly overrides |
| M-03 | Tire degradation evaluation status | `ROADMAP.md:110` ("pending formal evaluation"), `ROADMAP.md:613` ("Pending formal holdout evaluation") | pending / no number | **Evaluated: global MAE 0.7078 s (R² 0.605), C2 fine-tune 0.5501 s**; R² > 0.85 target not met, reframed | Thesis p. 113; IEEE (0.708 / 0.550) |
| M-04 | NLP pipeline mean latency | `docs/pages/thesis.md:49` (42.1 ms) vs `docs/pages/changelog.md:206` (47.8 ms) | two lineages coexist | Published = **47.8 ms mean / 59.4 ms P95** (IEEE); regenerated = 42.1/44.2 (`data/eval/nlp_pipeline_cpu.md`) | IEEE report. Not an error (thesis.md self-labels as regenerated) but needs a one-line reconciliation note so readers do not flag it as a contradiction |
| M-05 | "47.8 vs 43.7 ms" suspected contradiction | (audit brief / memory) | 43.7 ms alleged | **Non-issue.** 43.7 only exists inside **243.7 ms** = RAG retrieval P95 | Thesis p. 101 (Tabla 5.12), IEEE report. Record in memory so it is not re-hunted |
| M-06 | Sub-agent count | `README.md:45` ("seven specialised agents coordinated by an orchestrator"); `ROADMAP.md:341` ("Seven specialised sub-agents (N25-N30)"); `docs/pages/simulation.md:5,24` ("all seven strategy agents" / "all 7 agents") | seven sub-agents | **Six sub-agents + one orchestrator N31** ("seven agents" is only valid as the TOTAL incl. N31) | Thesis pp. 18, 113 ("seis sub-agentes especializados"); IEEE ("Six specialised sub-agents"); `src/agents/` = 6 sub-agent modules + `strategy_orchestrator.py`. Already correct: `README.md:5` tagline, `docs/pages/multi-agent.md:3`, `docs/pages/home.md:3`, `docs/pages/architecture.md:3` |
| M-07 | ML model count | `ROADMAP.md:11` ("eight ML predictive models") | eight | **seven** (canonical docs phrasing: 6 supervised predictors + circuit clustering; IEEE says "six supervised predictors" - see OQ-1) | Thesis p. 18 ("familia de siete modelos predictivos"); `docs/app/home.js:103` enumerates the seven |
| M-08 | Overtake production threshold | `ROADMAP.md:143` (0.80) vs `docs/pages/thesis.md:11` (0.7976) | rounding split | 0.7976 exact, 0.80 as rounded display is acceptable | N12 step 5; no fix required, note only |
| M-09 | NER F1 | `ROADMAP.md:263` (0.42) | 0.42 | 0.4151 (0.42 is fair rounding) | Thesis p. 113; IEEE (0.415). No fix required |
| M-10 | NLP latency budget | `ROADMAP.md:274` (target < 500 ms) vs thesis p. 74 (RNF-01 ceiling **100 ms**) vs IEEE ("500 ms operational budget") | 500 vs 100 ms | ambiguous between the two authoritative sources | See OQ-2; no repo change until resolved |
| M-11 | Radio corpus size | `ROADMAP.md:319` ("529 MP3s + 48 parquets") vs thesis "530 mensajes" | 529 vs 530 | Both correct: 530 = labelled text corpus (sentiment), 529 = intent subset AND the HF audio count is a separate inventory | Thesis pp. 97-98. No fix; optionally disambiguate wording |

Metrics verified CONSISTENT everywhere they appear (no action): overtake 0.5491/0.8758, SC 0.0723 + lift 1.67x, pit 0.487 vs 0.555, undercut 0.6739/0.7708 + threshold 0.522, RAG 2,279 chunks + scores 0.62-0.76, MC Dropout 50 passes, MC 500 samples x 4 candidates, 14-field `StrategyRecommendation`, 28,494 overtake pairs, K=4 clustering.

---

## 3. Findings register (P0 -> P3)

Priorities: **P0** = broken commands users copy-paste, or wrong published research figures in live claim sections. **P1** = wrong structural claims (counts, dead links, defaults, phantom API surface). **P2** = stale-but-harmless drift (renames, descoped features, versions). **P3** = polish/completeness.

| ID | P | Finding | Doc-side anchor | Truth-side anchor |
|---|---|---|---|---|
| F-01 | **P0** | `f1-sim VER Melbourne "Red Bull Racing" --year 2025` has the positional args in the wrong order (parses VER as the GP); correct order is `gp_name driver team` | `README.md:53`; `INSTALL.md:132` | `scripts/run_simulation_cli.py:2314-2316` (positionals: gp_name, driver, team). Correct examples exist at `INSTALL.md:44` and `CLAUDE.md` §5 |
| F-02 | **P0** | Verification command uses nonexistent flag `--lap-range 1 1`; the real flag is `--laps` with a range string (`--laps 1-1`) | `INSTALL.md:132` | `scripts/run_simulation_cli.py:2339-2343` |
| F-03 | **P0** | Data-bootstrap one-liner imports `f1_strat_manager.data_cache`, which is not an importable root; the installed packages are `src*`/`scripts*`, so the command raises ModuleNotFoundError. Correct: `from src.f1_strat_manager.data_cache import ensure_setup` | `CONTRIBUTING.md:34` | `pyproject.toml:140-142` (`include = ["src*", "scripts*"]`); `src/f1_strat_manager/data_cache.py:348` (signature with `show_progress` confirmed) |
| F-04 | **P0** | Lap-time MAE 0.392 s published as the achieved result in live claim sections (v0.7 goals, success metrics, milestone table) | `ROADMAP.md:100,117,583,612`; `docs/pages/roadmap.md:267` | Thesis Tabla 6.1 p. 113 (0.4104 s); IEEE table (0.410) |
| F-05 | **P0** | Sentiment 87.5% published as the achieved result; the thesis explicitly rules this figure non-authoritative in favour of 0.84 | `ROADMAP.md:251,280,586`; `docs/pages/roadmap.md:301,303` | Thesis p. 98 + Tabla 6.1; IEEE (0.84 / macro F1 0.75) |
| F-06 | **P1** | Tire-degradation evaluation marked "pending" although the thesis/IEEE publish final holdout numbers (and the R² > 0.85 target was missed and reframed) | `ROADMAP.md:110,613` | Thesis p. 113 (0.7078 global / 0.5501 C2); IEEE (0.708 / 0.550, R² 0.605) |
| F-07 | **P1** | "Seven specialised agents coordinated by an orchestrator" / "Seven specialised sub-agents (N25-N30)" - N25-N30 is six sub-agents; seven is only the total including N31 | `README.md:45`; `ROADMAP.md:341`; `docs/pages/simulation.md:5,24` | Thesis pp. 18, 113; IEEE report; `src/agents/` (6 sub-agent modules + orchestrator) |
| F-08 | **P1** | "Eight ML predictive models" - canonical count is seven (docs) / six supervised predictors (IEEE) | `ROADMAP.md:11` | Thesis p. 18; IEEE report; `docs/app/home.js:103` |
| F-09 | **P1** | ARCHITECTURE.md carries 11 dead relative links to a pre-React docs tree: `docs/architecture.md`, `docs/arcade/strategy-pipeline.md`, `docs/arcade/dashboard.md`, `docs/agents-api-reference.md`, `docs/backend-api.md`, `docs/streamlit-frontend.md`, `docs/simulation/overview.md`, `docs/diagrams/{strategy_pipeline_flow,arcade_3window_architecture,tcp_broadcast_dataflow,data_pipeline}.drawio`, `docs/diagrams/` | `ARCHITECTURE.md:24,32-33,47-49,61-63,69-73` | Real paths: `docs/pages/{architecture,arcade-strategy-pipeline,arcade-dashboard,agents-api,backend-api,streamlit,simulation}.md`; diagrams at `documents/dev_docs/diagrams/*.drawio` (`docs/diagrams/` does not exist; `site/` is gitignored) |
| F-10 | **P1** | Same class of dead link: INSTALL points at `docs/arcade-quick-start.md`, CONTRIBUTING at `docs/agents-api-reference.md` | `INSTALL.md:73`; `CONTRIBUTING.md:126` | `docs/pages/arcade-quick-start.md`; `docs/pages/agents-api.md` |
| F-11 | **P1** | LLM provider defaults contradict each other and the code. INSTALL says "Arcade and CLI paths use OpenAI gpt-4.1-mini by default"; the CLI default is **lmstudio** and gpt-4.1-mini is the sub-agent model (orchestrator uses gpt-5.4-mini). multi-agent.md and getting-started say "default is LM Studio" with no Arcade exception; the Arcade default is **openai** | `INSTALL.md:13-15`; `docs/pages/multi-agent.md:184`; `docs/pages/getting-started.md:75` | `scripts/run_simulation_cli.py:2350-2353` (`--provider` default `lmstudio`); `src/arcade/main.py:57` (default `openai`); thesis p. 80 (gpt-5.4-mini orchestrator / gpt-4.1-mini sub-agents) |
| F-12 | **P1** | INSTALL "Data bootstrap" says the HF first-run download "would be downloaded ... deferred past the first release" - it shipped in v0.9 and is invoked automatically by the CLI entry points | `INSTALL.md:118-122` | `src/f1_strat_manager/data_cache.py:348-360` (`ensure_setup`, "Invoked by the CLI entry points"); `ROADMAP.md:324` ("Lazy first-run data download ✅") |
| F-13 | **P1** | Backend API docs list an `auth` router at `endpoints/auth.py`; the file no longer exists and main.py registers six routers, not seven | `docs/pages/backend-api.md:13` | `src/telemetry/backend/main.py:48-63`; `src/telemetry/backend/api/v1/endpoints/` (no auth.py, only a stale .pyc) |
| F-14 | **P2** | Old repo slug `VforVitorio/F1_Strat_Manager` in live install/clone commands (works only via GitHub redirect); canonical is `VforVitorio/F1-StratLab` | `INSTALL.md:30,58,81`; `CONTRIBUTING.md:24`; `ROADMAP.md:325`; `docs/pages/setup.md:15` | `git remote -v` (origin = F1-StratLab); README badges/URLs already correct |
| F-15 | **P2** | Python version stated as "3.10 or 3.11" / "3.10 / 3.11" while the pin (quoted in the same sentence) allows 3.12 | `INSTALL.md:10`; `README.md:82` | `pyproject.toml:11` (`requires-python = ">=3.10,<3.13"`); CI typechecks on 3.12; `CLAUDE.md` §2 says 3.10-3.12 |
| F-16 | **P2** | "uv tool install drops two global binaries" - it installs four entry points | `INSTALL.md:34-41` | `pyproject.toml:113-117` (f1-strat, f1-sim, f1-arcade, f1-streamlit); `docs/pages/getting-started.md:21-28` already says four |
| F-17 | **P2** | Simulation docs still present Kafka as the committed live path ("will replace the iterator in v0.14+", "Future - Kafka integration (v0.14)"); Kafka was descoped in v0.12 and the planned live path is the OpenF1 WebSocket | `docs/pages/simulation.md:7,178-180` | `ROADMAP.md:35,509` (descope note), `ROADMAP.md:643` (v1.8.0 OpenF1 WebSocket) |
| F-18 | **P2** | Docker claims drift: getting-started says compose "boots the FastAPI backend, the Streamlit frontend and the Qdrant store"; ROADMAP's R3 bullet still lists "Qdrant + Kafka + LM Studio sidecar". The compose file has exactly two services (backend, frontend); Qdrant is an embedded on-disk client, LM Studio is reached via host.docker.internal | `docs/pages/getting-started.md:54`; `ROADMAP.md:513` | `docker-compose.yml:1-55`; `docs/pages/setup.md:95-98` (correct two-service description) |
| F-19 | **P2** | Docs roadmap page still describes the docs site as "React + Babel"; Babel was dropped in #136/PR #157 | `docs/pages/roadmap.md:436` | `docs/index.html:136` ("no JSX, no Babel, no build step"); memory `project_docs_audit_plan` |
| F-20 | **P2** | setup.md instructs a fully manual HF download + hand-placement under `data/` without mentioning the automatic `ensure_setup()` first-run path (and uses the old repo slug, F-14) | `docs/pages/setup.md:26-49` | `src/f1_strat_manager/data_cache.py:348` |
| F-21 | **P2** | thesis.md (docs) presents regenerated eval numbers (NLP mean 42.1 ms) on a page titled "as referenced in chapter 5 of the TFG thesis"; the published chapter figure is 47.8 ms mean / 59.4 P95. Needs an explicit two-lineage note ("published vs regenerated") rather than a silent divergence | `docs/pages/thesis.md:3,49` | IEEE report (47.8/59.4); `data/eval/nlp_pipeline_cpu.md` (42.069/44.246) |
| F-22 | **P3** | CONTRIBUTING cites `actions/labeler@v5` (actual: v6) and "three templates" (actual: four - bug report, feature request, data issue, epic) | `CONTRIBUTING.md:162,224-226` | `.github/workflows/labeler.yml:19`; `.github/ISSUE_TEMPLATE/` (4 templates + config) |
| F-23 | **P3** | ci-cd.md drift: "Three workflows" (actual: five - ci, docs, release-please, labeler, auto-update-prs), push triggers omit `test` and `docs/**`, and release-please pinned at @v4 (actual @v5) | `docs/pages/ci-cd.md:35,39,51` | `.github/workflows/` (5 files); `ci.yml:5`; `release-please.yml:19` |
| F-24 | **P3** | backend-api.md strategy table omits three live GET endpoints: `/radio-available-gps`, `/radio-laps`, `/radio-transcript` (completeness, nothing wrong listed) | `docs/pages/backend-api.md:110-131` | `src/telemetry/backend/api/v1/endpoints/strategy.py:745,763,823` |
| F-25 | **P3** | architecture.md (docs) names the 14-field orchestrator output `StrategyState`; the Pydantic contract is `StrategyRecommendation` (`StrategyState` is the Arcade snapshot dataclass). Naming conflation, risk of confusion in the paper era | `docs/pages/architecture.md:58-60` | `src/agents/strategy_orchestrator.py` (StrategyRecommendation, 14 fields); `src/arcade/` (StrategyState.snapshot_dict) |
| F-26 | **P3** | README project-layout line says `docs/` contains "draw.io diagrams"; the .drawio sources live under `documents/dev_docs/diagrams/` | `README.md:94` | Glob: no `docs/**/*.drawio`; `documents/dev_docs/diagrams/*.drawio` |

**Verified accurate (spot-checked, no action):** project `CLAUDE.md` factual claims (Python 3.10-3.12, command examples, CI jobs, agent/orchestrator description); CI description in CONTRIBUTING (jobs, `--frozen`, mypy scope, cache strategy); `docker-compose.yml` vs INSTALL's Streamlit section (ports 8000/8501, mounts); `f1-arcade` example flags vs `src/arcade/main.py` argparse; `~/.f1-strat/` first-run location (`data_cache.py:168`); `uv tool uninstall f1-strat-manager` (pyproject name); backend router map minus auth; strategy endpoint list vs `strategy.py` decorators; agents-api entry-point table (`run_*_from_state` adapters exist across `src/agents/`); nav.js page registry vs `docs/pages/*.md`; README badges and repo URLs; docs home/getting-started/multi-agent/architecture count framing; MkDocs references fully purged from live docs pages (#156).

---

## 4. Phased fix plan (each phase = one future GitHub sub-issue)

All phases are docs-only PRs (`docs:` commits). No code changes anywhere. Suggested order = listed order; phases 1-3 are the ones that matter before any paper/award submission links back to the repo.

### Phase 1 — Fix broken copy-paste commands (P0) — **S**
- `README.md:53` and `INSTALL.md:132`: swap to `f1-sim Melbourne VER "Red Bull Racing" --year 2025`.
- `INSTALL.md:132`: replace `--lap-range 1 1` with `--laps 1-1` (and keep `--no-llm`).
- `CONTRIBUTING.md:34`: `python -c "from src.f1_strat_manager.data_cache import ensure_setup; ensure_setup(show_progress=True)"`.
- Acceptance: every shell command in README/INSTALL/CONTRIBUTING runs (or at least parses args) against the current wheel.

### Phase 2 — Reconcile published metrics to thesis/IEEE finals (P0/P1) — **M**
- ROADMAP.md: 0.392 -> 0.4104 s at lines 100, 117, 583, 612 (keep the "target < 0.5 s met" framing, it still holds).
- ROADMAP.md: 87.5% -> 0.84 accuracy (macro F1 0.75) at lines 251, 280, 586, with a one-line note mirroring the thesis rationale (notebook header vs reproducible classification_report).
- ROADMAP.md:110 + 613: replace "pending formal evaluation" with the final figures (global TCN MAE 0.7078 s, R² 0.605; C2 fine-tune 0.5501 s) and mark the R² > 0.85 target as not met / reframed.
- `docs/pages/roadmap.md`: same three metrics (lines 267, 301, 303).
- Decide and apply the changelog policy from OQ-3 (recommended: leave `docs/pages/changelog.md` verbatim as release history; it already carries the 0.4104 anchor at line 86).
- Acceptance: `grep -rn "0\.392\|87\.5" README.md ROADMAP.md docs/pages/roadmap.md` returns nothing (changelog exempt per policy).

### Phase 3 — Normalize agent/model counts (P1) — **S**
- `README.md:45`: "six specialised agents coordinated by an orchestrator" (or "six sub-agents + one orchestrator").
- `ROADMAP.md:341`: "Six specialised sub-agents (N25-N30) coordinate under a Supervisor Orchestrator (N31)".
- `ROADMAP.md:11`: "eight ML predictive models" -> "seven ML models" (pending OQ-1 wording).
- `docs/pages/simulation.md:5,24`: "the six sub-agents (plus the N31 orchestrator)".
- Adopt the canonical sentence used by docs home/getting-started everywhere: "seven ML models, six LangGraph sub-agents and one strategy orchestrator".

### Phase 4 — Repair dead docs links in root markdown (P1) — **S/M**
- ARCHITECTURE.md: retarget the 11 links to `docs/pages/*.md` (or the deployed `https://docs.f1stratlab.com/#/<slug>` URLs, pick one convention) and the four .drawio links to `documents/dev_docs/diagrams/`.
- `INSTALL.md:73` -> `docs/pages/arcade-quick-start.md`; `CONTRIBUTING.md:126` -> `docs/pages/agents-api.md`.
- Acceptance: a link-checker pass over root .md files reports zero broken relative links.

### Phase 5 — One truth table for LLM provider defaults (P1) — **S**
- Write the per-surface defaults once and reuse: CLI `f1-sim` defaults to **lmstudio** (`--provider openai` to switch); Arcade defaults to **openai**; backend/agents read `F1_LLM_PROVIDER` (default lmstudio); sub-agents = gpt-4.1-mini, orchestrator = gpt-5.4-mini.
- Fix `INSTALL.md:13-15`, `docs/pages/multi-agent.md:184`, `docs/pages/getting-started.md:75` (setup.md:59 is already right for the backend path).

### Phase 6 — Backend/CI reference sync (P1/P2) — **M**
- `docs/pages/backend-api.md`: drop the `auth` router row (F-13); add the three `/radio-*` endpoints (F-24).
- `docs/pages/ci-cd.md`: five workflows, add `test` + `docs/**` to the trigger list, release-please @v5 (F-23).
- `CONTRIBUTING.md`: labeler@v6, four issue templates (F-22).

### Phase 7 — Freshness sweep on install/runtime claims (P2) — **M**
- Repo slug normalization to `F1-StratLab` (F-14: INSTALL x3, CONTRIBUTING, ROADMAP, setup.md).
- Python "3.10-3.12" wording (F-15: INSTALL, README).
- "four console entry points" in INSTALL (F-16).
- INSTALL data-bootstrap section rewritten around `ensure_setup()` automatic first-run + `~/.f1-strat/` (F-12); setup.md gains the automatic path as the primary flow (F-20).
- Kafka -> OpenF1 WebSocket in `docs/pages/simulation.md` (F-17); Docker service list corrected in getting-started + ROADMAP R3 (F-18); "React + Babel" -> "React (no build step)" in docs roadmap (F-19).

### Phase 8 — Two-lineage metrics note + naming polish (P2/P3) — **S**
- `docs/pages/thesis.md`: add a short "published vs regenerated" note (published: mean 47.8 ms / P95 59.4 ms, IEEE report; regenerated on current hardware: 42.1 / 44.2 ms from `data/eval/`), so the page cannot be read as contradicting the paper (F-21).
- `docs/pages/architecture.md`: rename the output contract to `StrategyRecommendation` (F-25).
- `README.md:94`: point diagrams at `documents/dev_docs/diagrams/` (F-26).
- Update project memory: record M-05 (43.7 = 243.7 RAG P95, non-issue) so the "47.8 vs 43.7" ghost is laid to rest.

---

## 5. Open questions (resolve with the author before/while executing)

- **OQ-1 — What exactly are the "seven models"?** Docs enumerate 6 supervised predictors + circuit clustering (`docs/app/home.js:103`); the IEEE report says "a family of six supervised predictors"; the thesis objective says "familia de siete modelos predictivos" listing six target variables (p. 18) with clustering named separately (p. 113), and p. 50 even says "los seis modelos del sistema". Recommendation: standardize on "seven ML models (six supervised predictors + circuit clustering)" in repo docs, which keeps every current headline true; confirm the phrasing that the IEEE paper will use so repo and paper match.
- **OQ-2 — NLP latency budget: 100 ms or 500 ms?** Thesis p. 74 cites RNF-01 with a 100 ms ceiling; the IEEE report and ROADMAP use a 500 ms operational budget. Both hold (59.4 << 100 << 500), but the repo should quote one. Which is the RNF as finally written in ch. 3?
- **OQ-3 — Historical-entry policy.** `docs/pages/changelog.md` (verbatim CHANGELOG) and the v0.7/v0.8.2 release narratives contain the notebook-era 0.392/87.5/47.8 figures as release history. Recommendation: leave CHANGELOG verbatim, fix ROADMAP + docs roadmap page (they read as current claims). Confirm.
- **OQ-4 — `docs/pages/thesis.md` charter.** Is the page meant to track the latest regenerated artifacts (then keep 42.1 ms and label the lineage, per Phase 8) or to mirror the publication (then pin 47.8/59.4 and drop the auto-regeneration claim)? Phase 8 assumes the former.
- **OQ-5 — Thesis-internal slips (out of repo scope, relevant for the paper/AEPIA derivations):** the thesis annex p. 141 says "siete sub-agentes" once, and the bibliography annotation p. 119 still carries MAE 0.392 s for the XGBoost citation. The IEEE report already uses 0.410 everywhere. Worth a note in the `feat/paper` branch backlog.
- **OQ-6 — Wheel asset naming.** `docs/pages/getting-started.md:18` builds the wheel URL as `f1_strat_manager-<version>-py3-none-any.whl` for every release; release-please's `publish-wheel` job should guarantee this, but verify the v1.6.2 release actually carries that asset before treating the command as gospel.
