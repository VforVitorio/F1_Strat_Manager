# AUDIT - RAG layer (FIA regulation retrieval: index, retriever, N30 consumption)

> **Scope:** the retrieval-augmented layer over the FIA Sporting Regulations: `src/rag/retriever.py` (RagRetriever, `query_rag_tool`, RagConfig), `scripts/build_rag_index.py` (PDF to Qdrant ingestion), `scripts/download_fia_pdfs.py` (FIA scraper + known-URLs fallback), the on-disk Qdrant index (`data/rag/qdrant_local/`, `BAAI/bge-m3` embeddings), and how N30 (`src/agents/rag_agent.py`) and the orchestrator N31 (`src/agents/strategy_orchestrator.py`) consume it. Last code subsystem without a dedicated audit.
>
> **Inputs read:** `src/rag/{__init__,retriever}.py`, `scripts/build_rag_index.py`, `scripts/download_fia_pdfs.py`, `src/agents/rag_agent.py:100-230` (read-only, untouchable), `src/agents/strategy_orchestrator.py:717-760`, `src/f1_strat_manager/data_cache.py` (get_data_root + HF snapshot patterns), `data/rag/` contents, memory `project_rag_src_plan`, `reference_n31_bibliography`; cross-referenced (not duplicated): `AUDIT_2026_REG_CONCEPT_DRIFT.md` F-10, `AUDIT_ML_AGENTS_EVAL.md` E-11/R-9 (#205), `AUDIT_SECURITY.md` S-9/D1 (#223), `AUDIT_DEVEX.md` DX-05 (#251).
>
> **Constraint:** plan only. `src/agents/` internals are untouchable (additive entry points only); every change below lands in `src/rag/`, `scripts/`, or a new shared eval package.

---

## 1. Executive summary

The RAG layer is small, clean, and well-documented at the function level: one retriever class, one build script, one downloader, a single Qdrant collection (`fia_regulations`, bge-m3, 1024-dim, cosine), and a LangGraph `@tool` wrapper consumed by N30. The code style is the best in the repo. The problems are all systemic, not local:

1. **Season correctness is enforced nowhere at query time.** The index mixes 2023/2024/2025 chunks in one collection; `RagRetriever.query()` (`src/rag/retriever.py:208-255`) has no `year` or `doc_type` filter, even though the `RegulationChunk` docstring promises callers can filter by both (`retriever.py:106-113`). The orchestrator never passes the race's season into `_build_rag_question` (`strategy_orchestrator.py:717-738`), and the N30 system prompt hardcodes "Always prefer the most recent regulation year (2025)" (`rag_agent.py`, `_SYSTEM_PROMPT`). A 2023 replay can be answered with 2025 rules, and vice versa when a 2023 chunk simply scores higher. This is the query-time half of the 2026-reg audit's F-10 (which covers the ingest half: `download_fia_pdfs.py:68` caps `supported_years` at 2023-2025).
2. **Citation grounding is structurally loose.** `run_rag_agent` (`rag_agent.py:175-210`) lets the ReAct agent retrieve with its own rewritten queries, then re-queries the retriever with the *original* question to populate `RegulationContext.chunks/articles`. The chunks attached as evidence are not necessarily the passages the LLM actually read, so `ctx.articles` and the article numbers inside `ctx.answer` can diverge silently. There is also no similarity floor: `query_rag_tool` returns the top-5 whatever their scores, so an off-topic question still feeds five weak passages to a model instructed to cite articles.
3. **The index is unversioned and staleness is undetectable.** No manifest records which PDF issue, embedding model, or chunk parameters built the index. `health_check()` (`retriever.py:257-276`) reports a vector count but not year coverage or model stamp. Nothing ever triggers a reindex: `download_link` skips any existing file (`download_fia_pdfs.py:406-410`) and its docstring references a `--force` flag that does not exist in `main()` (only `--years`/`--dry-run`), so an FIA erratum requires a manual delete that nobody is prompted to do.
4. **Retrieval quality has one one-shot, unwired measurement.** N30B (15 Spanish queries, P@k/MRR, manual ground truth) exists as a notebook only, and the three canned production questions the orchestrator actually asks are not in it (ML-eval E-11). The minimal fix is a small RAG suite inside the shared `src/strategy/eval/` package planned by #205, not a parallel harness.
5. **The build is broken on a fresh env and the docstrings have drifted.** `pypdf` is imported (`build_rag_index.py:34`) but not declared (DevEx DX-05, P1 there; cross-referenced, not re-owned). `extract_text_from_pdf`'s docstring says "using PyMuPDF" (`build_rag_index.py:201`), `ensure_collection`'s says the embeddings come from "all-MiniLM-L6-v2" (`build_rag_index.py:376`); both are relics of earlier model choices and will mislead the next maintainer.

Good news worth stating: idempotent hash-based incremental indexing works (`chunk_hash` + `get_existing_hashes`), the collection-existence check fails loudly with an actionable message (`retriever.py:187-192`), the `lru_cache` singleton correctly avoids Qdrant's local-mode double-open lock (`retriever.py:284-297`), the downloader has a sane scraper + known-URLs fallback design, and only Sporting Regulations are indexed by deliberate, documented choice (`download_fia_pdfs.py:92-96`).

---

## 2. How the layer hangs together (for orientation)

```
download_fia_pdfs.py            build_rag_index.py                 retriever.py
FIA site scrape + known URLs -> sporting_regs_<year>.pdf ->        RagRetriever.query()
(years capped 2023-2025, F-10)  512-char windows, 64 overlap,      top_k=5, cosine, NO filters
                                regex article/section tags,   ->   query_rag_tool (@tool, string out)
                                sha256 dedup, upsert to                 |
                                data/rag/qdrant_local              rag_agent.py (N30, ReAct, 1 tool)
                                                                        |
                                                   strategy_orchestrator.py (N31): conditional
                                                   activation (SC / pit / radio PENALTY-WARNING),
                                                   _build_rag_question -> regulation_context field
```

Consumers: N30's `run_rag_agent` / `run_rag_agent_from_state`; N31 attaches the answer string as `StrategyRecommendation.regulation_context` (`strategy_orchestrator.py:399-400,465`); the chat surface exposes it as the `query_regulations` MCP tool (Security audit S-2/S-9 territory). Data distribution: `data/rag/**` is in the HF snapshot patterns as an optional artefact (`data_cache.py:120-122`), so a prebuilt index can ship from the Hub with no version pin.

---

## 3. Findings register

| ID | Prio | Finding | Why it matters / size |
|---|---|---|---|
| **RAG-01** | **P1** | **No season scoping end to end.** One collection mixes years 2023-2025; `query()` exposes no `year`/`doc_type` filter (`retriever.py:208-243`) despite `RegulationChunk` docstrings promising both (`retriever.py:106-113`); `_build_rag_question` (`strategy_orchestrator.py:717-738`) never mentions the race season; N30's prompt hardcodes "prefer 2025" and "2023-2025" (`rag_agent.py` `_SYSTEM_PROMPT`). The Qatar demo's Article 36.3 citation is season-correct by luck of scoring, not by construction. Query-time complement of F-10 (`AUDIT_2026_REG_CONCEPT_DRIFT.md:178`). | Wrong-season rule cited with full confidence in replays and, post-2026, guaranteed drift. Fix is additive: filter param + payload index + caller wiring. **M** |
| **RAG-02** | **P1** | **Evidence and answer can diverge.** `run_rag_agent` re-queries with the original question after the agent answered from its own (possibly rewritten) tool queries (`rag_agent.py:175-210`, the docstring documents the double retrieval); `ctx.chunks`/`ctx.articles` are therefore not guaranteed to be what the LLM read. No similarity threshold anywhere: `query_rag_tool` (`retriever.py:317-349`) formats top-5 regardless of score; "No relevant passages" is returned only for an empty hit list, never for a low-quality one. Nothing checks that articles cited in `answer` appear in the retrieved set (hallucinated-article risk, ML-eval R-9). | Citations are the product here (they reach the UI and the paper verbatim). Faithfulness must be checkable, then checked. Indirect-injection side of the same surface is owned by Security S-9/D1 (#223); not duplicated here. **M** |
| **RAG-03** | **P1** | **Fresh-env index build fails: `pypdf` undeclared.** `build_rag_index.py:34` imports it; pyproject declares only `qdrant-client`, `sentence-transformers`, `bs4` (pyproject.toml:43,74,108). **Owned by DevEx DX-05 (#251); tracked here only as a blocking dependency of every phase below.** | ModuleNotFoundError on `uv sync` + run. **S** (lands via #251) |
| **RAG-04** | **P2** | **Index has no version, manifest, or staleness signal.** Nothing records source PDF hashes/issue titles, embedding model, dim, chunk params, or build date; `health_check()` omits year coverage and model stamp; retriever docstrings warn twice that a model mismatch "produces meaningless similarity scores" (`retriever.py:42-45,172-174`) yet nothing enforces the match. Reindex is purely manual; `download_link` skips existing files and its docstring's `--force` option does not exist (`download_fia_pdfs.py:389-410` vs `main():510-528`), so FIA errata are silently never picked up. HF-shipped index (`data_cache.py:120-122`) has no pin, so a Hub-side rebuild can silently change local behaviour (ties to the ecosystem HF pin-manifest design). | Stale or mismatched index is undetectable today; this blocks the 2026 refresh from being verifiable. **M** |
| **RAG-05** | **P2** | **Retrieval quality is unmeasured on the questions production actually asks.** N30B: 15 Spanish queries, P@k/MRR, 3 configs, notebook-only, never re-runnable in CI (ML-eval E-11). The 3 canned orchestrator questions (SC pit rules, compound-change restrictions, mandatory dry-race compounds) are not covered verbatim; no wrong-year-rate metric exists at all. | Eval must live in the shared `src/strategy/eval/` package (#205), which does not exist yet (`src/strategy/` has only `training/`, `inference/`); design in §4 Phase 3. **S-M** |
| **RAG-06** | **P2** | **Chunking is character-windowed and article-blind.** 512-char windows / 64 overlap (`build_rag_index.py:75-76,322-365`) cut articles mid-clause; `extract_article_reference` tags each chunk with the *first* regex match (`build_rag_index.py:272-287`), so a chunk whose overlap head carries the tail of Article 47 gets labelled 47 while its body is Article 48 (systematic off-by-one citations). The stated rationale, "fits inside BGE-M3's 512-token limit" (`build_rag_index.py:61-63`), is wrong: bge-m3 accepts 8192 tokens, and 512 chars is only ~100-170 tokens, so the model's context is 98 percent unused. | Mislabelled citations poison RAG-02's faithfulness check from below; larger, article-aligned chunks are cheap to try once the eval (Phase 3) can arbitrate. **M** |
| **RAG-07** | **P2** | **Build/runtime path split-brain.** The retriever resolves `data/rag/` through `get_data_root()` (env override `F1_STRAT_DATA_ROOT`, or `~/.f1-strat/data/` in the `uv tool install` flow; `retriever.py:60-75`), but the builder and downloader are hardwired repo-relative (`build_rag_index.py:79-90`, `download_fia_pdfs.py:73-86`) and ignore the override. In an installed-tool env, `build_rag_index.py` writes an index the retriever will never open. | Silent "collection not found" for exactly the users who followed the docs; one shared path helper fixes all three files. **S** |
| **RAG-08** | **P3** | **Qdrant local mode is single-process.** The embedded client holds a file lock; the `lru_cache` singleton (`retriever.py:284-297`) protects one process only, so backend + CLI + Streamlit running simultaneously against the same `qdrant_local/` raise `AlreadyLocked` for the latecomers. Undocumented in README/INSTALL. | Confusing failure the day two surfaces run at once; document now, consider a served Qdrant only if it ever actually bites. **S** |
| **RAG-09** | **P3** | **Docstring drift + minor ingest nits.** (a) "using PyMuPDF" (`build_rag_index.py:201`) vs actual `pypdf`; (b) "all-MiniLM-L6-v2" (`build_rag_index.py:376`) vs bge-m3; (c) `RegulationChunk` promises doc_type/year filtering that does not exist (RAG-01); (d) within-batch duplicate hashes are not deduped (`get_existing_hashes` covers only pre-existing points, `build_rag_index.py:565-569`), so the same passage in two PDFs indexed in one run creates two points; (e) sequential point IDs from `points_count` (`build_rag_index.py:583`) collide if points are ever deleted individually. | Cheap truth-restoring fixes; (d)/(e) matter only when the corpus grows. **S** |

No P0: nothing crashes the shipped flows or leaks data (the injection-path P1s live in the Security audit). The two P1s are silent-correctness risks on a headline feature.

---

## 4. Phased plan (each phase = one future sub-issue)

**Phase 1 - Truth and build integrity (S).**
Verify #251/DX-05 landed `pypdf` (else this phase carries it); fix the three drifted docstrings (RAG-09 a-c); delete or implement the phantom `--force` in `download_fia_pdfs.py` (prefer implement: re-download replaces the file); add within-batch hash dedup. Acceptance: `python scripts/build_rag_index.py --help` works on a fresh `uv sync`; no docstring names a component the code does not use.

**Phase 2 - Index manifest + season scoping (M).** *(RAG-01, RAG-04, RAG-07)*
Builder writes `data/rag/index_manifest.json` (source PDF sha256 + FIA issue title, embedding model + dim, chunk params, years indexed, build timestamp); retriever validates model/collection against it at init and `health_check()` reports year coverage + manifest hash. All three files resolve paths through one shared helper honouring `get_data_root()`. Add optional `year`/`doc_type` filter params to `RagRetriever.query()` and `query_rag_tool` (Qdrant payload filter, additive signature); wire the season from `lap_state.session_meta` into the RAG question path **additively** (new entry point or param default; `src/agents/` internals stay byte-identical, so the prompt's "prefer 2025" is superseded by filtered retrieval rather than edited). Include the manifest in the HF pin-manifest scheme. Acceptance: querying with `year=2023` never returns a 2025 chunk; retriever refuses (loud warning or raise) on model mismatch.

**Phase 3 - RAG eval inside the shared #205 package (M).**
No parallel harness: a `rag/` module in `src/strategy/eval/` when #205 scaffolds it (or the first resident if RAG lands first, built to #205's dataset/report conventions). Port N30B's 15-query ground truth (translate to English), extend to ~30 queries covering the 3 production question shapes x seasons and 5-8 known year-differing rules. Metrics: P@k, MRR, **wrong-year rate**, and **citation-match rate** (articles in answer that appear in retrieved set; needs Phase 4's capture for the agent-level variant, retriever-level runs immediately). Opt-in pytest marker, one JSON+markdown report. Acceptance: one command re-runs the benchmark; baseline numbers recorded before Phase 5 changes anything.

**Phase 4 - Grounding and citation faithfulness (M).** *(RAG-02)*
Additive N30 entry point that extracts the agent's *actual* tool calls/results from the LangGraph message history so `RegulationContext.chunks` = what the LLM read (drop the second retrieval); add a configurable similarity floor in `query_rag_tool` returning the explicit "no relevant passages" string below it; post-hoc citation check (cited articles as subset of retrieved articles) flagging violations on the `RegulationContext`. Security D1 (#223) owns delimiting retrieved text as untrusted data; this phase only verifies the wrapper cooperates. Acceptance: eval's citation-match rate computed on real agent traces; a below-threshold query yields the refusal string, not five weak chunks.

**Phase 5 - 2026 refresh + chunking experiment (M-L).** *(F-10 execution + RAG-06)*
Uncap `supported_years`, add 2026 PDFs + known URLs, rebuild with `--force-rebuild`, publish index + manifest to HF pinned. Then, gated by Phase 3 numbers: article-aware chunking (split on `_ARTICLE_RE` boundaries, larger windows given bge-m3's 8192-token capacity) as a benchmarked A/B, adopted only if P@k/citation metrics improve. Acceptance: 2026 queries answered from 2026 chunks; chunking change justified by the eval, not vibes.

Order rationale: 1 unblocks everything; 2 kills the silent wrong-season class before eval measures it as noise; 3 must exist before 4/5 so improvements are provable; 5 last because it is the only phase whose value depends on the FIA's calendar.

---

## 5. Open questions

1. **Season default for chat:** the orchestrator knows the replay season, but the chat `query_regulations` tool has no race context. Default to latest indexed year, or require an explicit year in the tool schema?
2. **Technical Regulations:** deliberately excluded (`download_fia_pdfs.py:92-96`) yet half-supported everywhere (filename regex, title patterns, doc_type payloads). Keep the latent support or strip it?
3. **Similarity floor value:** bge-m3 cosine scores on this corpus cluster high; pick the threshold from the Phase 3 score distributions, not a priori. Who signs it off?
4. **HF index vs local build:** should the Hub-shipped prebuilt index be the *only* supported path for end users (build script demoted to maintainer tool), simplifying RAG-07?
5. **Ingestion trust policy** (Security #223 Q5): formally state that only operator-vetted FIA PDFs enter `data/rag/documents/`, in `src/rag/README.md`?

---

## 6. Verification protocol (how we will know it worked)

- **RAG-01/Phase 2:** eval query set with year-discriminating ground truth (rules that changed 2023 to 2025); wrong-year rate = 0 with the filter on, measured nonzero baseline with it off.
- **RAG-04/Phase 2:** delete the manifest or swap the model name; retriever init fails loudly with an actionable message. `health_check()` output includes `years` and `manifest_hash`.
- **RAG-05/Phase 3:** benchmark re-run twice gives identical numbers (deterministic); report lands in the #205 report location.
- **RAG-02/Phase 4:** trace-level test: agent answer citing an article absent from its retrieved set raises the faithfulness flag; injected-chunk fixture behaviour is asserted by Security's S-9 test (cross-check only).
- **RAG-06/Phase 5:** A/B table (current window vs article-aware) with P@k/MRR/citation-match on the same query set; adoption decision recorded in the PR.
- **RAG-07:** `F1_STRAT_DATA_ROOT=<tmp>` set for both build and query in one test; index built and found in the same directory.

---

*Audit date: 2026-07-07. Plan-only; no code changed. Cross-references: F-10 (`AUDIT_2026_REG_CONCEPT_DRIFT.md`), E-11/R-9 + #205 (`AUDIT_ML_AGENTS_EVAL.md`), S-2/S-9/D1 + #223 (`AUDIT_SECURITY.md`), DX-05 + #251 (`AUDIT_DEVEX.md`).*
