# src/rag: FIA Regulation Retrieval

**Status: Active**, imported by N30 and N31.

Provides runtime retrieval-augmented generation (RAG) over FIA regulation PDFs.
The Qdrant index must be built once with `scripts/build_rag_index.py` before any query.

---

## Public API

| Symbol | Type | Description |
|---|---|---|
| `RagConfig` | dataclass | Centralised config: collection name, embedding model, top-k, derived paths |
| `CFG` | `RagConfig` | Module-level singleton config; edit this to change defaults |
| `RegulationChunk` | dataclass | Single retrieved passage with `text`, `article`, `doc_type`, `year`, `score`, `section_title` |
| `RagRetriever` | class | Holds Qdrant client + sentence encoder; call `.query()` per request |
| `get_retriever()` | function | Returns the process-level `RagRetriever` singleton (lazy init, loads model once) |
| `query_rag_tool` | `@tool` | LangGraph-compatible tool wrapper; returns formatted string for the LLM. Takes the season from the RunnableConfig key `configurable.season`, never as a tool argument, so the model cannot choose which rulebook it reads |

### `RagRetriever` methods

- `__init__(qdrant_path, collection_name, embedding_model, top_k)`, loads encoder (~1-2 s); raises `RuntimeError` if collection missing
- `query(question, top_k=None, year=None, doc_type=None) -> list[RegulationChunk]`, cosine similarity search, ordered by descending score. `year` restricts the search to one season's rulebook; a season the index does not hold falls back to an unscoped search with one warning rather than returning nothing
- `health_check() -> dict`, returns `{collection, vector_count, embedding_model, qdrant_path}` for diagnostics

---

## Usage

```python
from src.rag.retriever import query_rag_tool, get_retriever

# As a LangGraph tool (N31 Orchestrator)
result_str = query_rag_tool.invoke({"question": "pit lane speed limit"})

# Direct retrieval (N30 RAG agent, diagnostics)
retriever = get_retriever()
chunks = retriever.query("safety car restart procedure", top_k=10)

# Scoped to one season. The same article is renumbered and reworded between
# rulebooks, so an unscoped query mixes them: 43 of 75 top-5 hits on the tracked
# gold set come from a season other than the one asked about.
chunks = retriever.query("safety car restart procedure", year=2025)
for c in chunks:
    print(c.article, c.score, c.text[:80])

# Startup health check
print(retriever.health_check())
```

---

## Key dependencies

- `qdrant-client`, local on-disk vector store at `data/rag/qdrant_local/`
- `sentence-transformers`, embedding model `BAAI/bge-m3` (1024-dim, ~8 GB VRAM)
- `langchain-core`: `@tool` decorator for LangGraph integration

---

## Pre-requisites

The Qdrant collection must exist before calling `get_retriever()`:

```bash
python scripts/build_rag_index.py
```

FIA PDFs are downloaded by `scripts/download_fia_pdfs.py` into `data/rag/documents/`.

---

## Developed in

[`notebooks/agents/N30_rag_agent.ipynb`](../../notebooks/agents/N30_rag_agent.ipynb)
