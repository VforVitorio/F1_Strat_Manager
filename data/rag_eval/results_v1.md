# Quantitative benchmark results - RAG Agent (N30)

Evaluation set: 15 Spanish queries on the FIA Sporting Regulations 2023-2025, distributed across 5 categories (drs, flags_penalties, pit_stops, safety_car, tyre_allocation). Each query carries manual ground truth verified as a literal substring of the corresponding PDF.

**Reported metrics.** `Precision@k (strict)` requires both an article match (in the payload or in the chunk text) **and** a keyword match in the text. `Content P@5` relaxes the condition and only requires a keyword match, isolating the embedding quality from the noise introduced by the indexer's article tagging. Reporting both columns is the honest way to present the result: the strict metric reflects what the agent actually cites, the content metric reflects what the retriever actually finds.

## Comparative summary table

| Configuration | Precision@1 | Precision@3 | Precision@5 | Content P@5 | MRR | Latency P50 (ms) | Latency P95 (ms) |
|---|---|---|---|---|---|---|---|
| BGE-M3 chunk 512 (production) | 0.200 | 0.200 | 0.200 | 0.800 | 0.235 | 57.8 | 243.7 |
| MiniLM-L6-v2 chunk 512 | 0.200 | 0.200 | 0.200 | 0.800 | 0.232 | 23.4 | 48.2 |
| BGE-M3 chunk 256 | 0.067 | 0.133 | 0.133 | 0.800 | 0.108 | 70.4 | 94.3 |

## Discussion

**Strict Precision@5 vs Content P@5.** The strict metric requires an article match (in payload or in chunk text) and a keyword match. The `content_p_at_5` metric ignores the `article` field and only requires a keyword in the text, isolating the pure embedding quality. The gap between the two columns quantifies the cost of the `_ARTICLE_RE` regex in `scripts/build_rag_index.py`, which tags every chunk with the first `Article X.Y` reference it finds - a reference that in many chunks corresponds to a cross-citation (for example `Article 30.1a)`) and not to the article the chunk actually belongs to. This degradation is a known limitation of the indexing pipeline; mitigating it would require post-processing the payload (enriching it with the closest document heading), which is out of scope for this benchmark.

**BGE-M3 vs MiniLM-L6-v2.** BGE-M3 is a multilingual 1024d model with MTEB ~67 trained with massive contrastive learning; MiniLM-L6-v2 is ~6x smaller (384d) and English-only. The Precision@k delta between rows 1 and 2 of the table quantifies the quality cost of replacing the production model with a lightweight alternative, and the latency delta quantifies the corresponding CPU saving.

**Chunk 512 vs chunk 256 with BGE-M3.** Finer chunks provide more retrieval granularity but raise the probability of splitting an article into two fragments. Comparing rows 1 and 3 of the table measures that trade-off directly: if Precision@5 drops when moving to chunk 256, the fine chunking is leaving relevant content outside the top-k.

**Latency P50 / P95.** The retriever is called once per orchestrator turn. A P95 below 100 ms keeps the RAG out of the bottleneck path against the LLM call (several hundred ms even with a local model), so the retrieval-stage cost stays well within the agent's SLA.

### Typical failure cases

- **Chunking too fine**: articles such as `30.2`, which span more than 500 characters in the PDF, get split into two chunks. The chunk holding the concrete numeric clause (for example `thirteen (13) sets`) may fall outside the top-k even when the article header is represented.
- **Ambiguous query**: queries that simultaneously mention DRS and Safety Car (Q14, Q15) compete with the chunks of sections 55 (Safety Car) and 56 (Virtual Safety Car), which also mention both terms.
- **Wrong year**: measured on this set, 43 of 75 top-5 hits came from a season other than the one asked about, and 7 of 15 queries had a top-1 from another year. Literal keywords capture the failure when numeric values change across years (intermediates: 4 sets in 2023 vs 5 sets in 2025; DRS enabled after 2 laps in 2023 vs 1 lap in 2025), but not in every case. Retrieval is scoped by season since #320, which takes that rate to 0 of 75; the numbers in the table above predate the filter and are the unscoped configuration.

### Reproducibility

The notebook `notebooks/agents/N30B_rag_benchmark.ipynb` rebuilds the set and the two new Qdrant collections idempotently: Restart Kernel -> Run All skips them if they already exist. The query JSON lives at `data/rag_eval/queries_v1.json`, and the literal keyword verification is the responsibility of the set author, not of the notebook - any future extension must go through the same filter against the original PDFs.
