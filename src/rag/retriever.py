"""
Run-time retrieval module for FIA regulation lookup.

Initialise ``RagRetriever`` once per process and call ``.query()`` on each
LLM tool invocation. The Qdrant collection must be populated first by
running ``scripts/build_rag_index.py``.

Public interface::

    from src.rag.retriever import RagRetriever, RegulationChunk, query_rag_tool
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool

# QdrantClient and SentenceTransformer are imported inside RagRetriever.__init__.
# sentence_transformers alone costs 7.3 s at import and pulls torch and
# transformers with it, and this module is reached from the orchestrator on every
# run, so `f1-sim --help` and any --no-llm lap paid for a vector store neither
# one opens. get_retriever() is already a lazy singleton (see its docstring), so
# the cost now lands on the first regulation question instead of on startup.

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class RagConfig:
    """Centralised configuration for the RAG retriever.

    Grouping all tunable parameters here means changing the embedding model,
    collection name, or storage path requires editing exactly one place rather
    than hunting for scattered constants across the module.

    Attributes:
        collection_name: Name of the Qdrant collection that holds the FIA
                         regulation vectors. Must match the name used in
                         ``build_rag_index.py``: a mismatch means queries
                         silently hit an empty or wrong collection.
        embedding_model: Sentence-transformers model identifier. Must be the
                         same model used at index build time. Mixing models
                         produces meaningless similarity scores because the
                         vector spaces are incompatible.
        top_k:           Default number of chunks returned per query. Five is
                         enough context for most strategy questions; increase
                         to 10 for multi-article topics like safety car + pit lane.
    """

    collection_name: str = "fia_regulations"
    embedding_model: str = "BAAI/bge-m3"  # 1024-dim, MTEB ~67, fits in 8 GB VRAM
    top_k: int = 5

    def __post_init__(self) -> None:
        # Derived from this file's location so the module works regardless of
        # the caller's working directory.
        self._repo_root = Path(__file__).resolve().parent.parent.parent

    @property
    def rag_dir(self) -> Path:
        """Root directory for all RAG artefacts under ``data/rag/``.

        Routes through :func:`src.f1_strat_manager.data_cache.get_data_root`
        when the helper is importable so the Qdrant collection is found under
        ``~/.f1-strat/data/rag/`` in the ``uv tool install`` flow; otherwise
        falls back to the repo-relative path for dev checkouts that do not
        have the helper on ``sys.path`` yet.
        """
        try:
            from src.f1_strat_manager.data_cache import get_data_root

            return get_data_root() / "rag"
        except (ImportError, OSError):
            # ImportError: helper not on sys.path (uv tool install layout).
            # OSError: get_data_root()/_find_repo_root() only ever touch the
            # local filesystem (env var, path resolve, mkdir) — read in full,
            # neither does network I/O, so OSError is the only realistic
            # runtime failure besides the import itself.
            return self._repo_root / "data" / "rag"

    @property
    def qdrant_path(self) -> Path:
        """On-disk Qdrant storage directory, created by ``build_rag_index.py``."""
        return self.rag_dir / "qdrant_local"


CFG = RagConfig()

# ---------------------------------------------------------------------------
# Data transfer object
# ---------------------------------------------------------------------------


@dataclass
class RegulationChunk:
    """A single regulation passage returned by a retrieval query.

    This is the atomic unit of information the RAG agent passes back to the
    orchestrator. Keeping article reference, document type and year alongside
    the raw text means downstream agents (e.g. N28 Pit Strategy) can filter
    by regulatory domain without having to re-parse the text themselves.

    Attributes:
        text:          The regulation passage itself, the verbatim paragraph
                       extracted from the FIA document after chunking.
        article:       The article or section identifier found inside the chunk
                       (e.g. ``"Article 48.3"``). Used by the LLM to cite the
                       source precisely and by callers to filter by article range.
                       Empty string when no reference could be extracted.
        doc_type:      Which FIA document the chunk comes from: distinguishes
                       sporting rules (race procedures, penalties) from technical
                       rules (equipment, pit stop mechanics). Callers can restrict
                       queries to a specific document type when the context is clear.
        year:          The regulation year the chunk belongs to. FIA rules change
                       annually, so a 2023 safety-car article may differ from 2025;
                       this field lets the agent pick the version that matches the
                       current race season.
        score:         Cosine similarity between the query embedding and this chunk,
                       in [0, 1]. Higher means more relevant. Exposed so callers can
                       apply a minimum threshold or rerank results if needed.
        section_title: The section heading from the source document when available.
                       Provides additional context about where in the regulations
                       the chunk sits without needing to read the full article.
    """

    text: str
    article: str
    doc_type: str
    year: int
    score: float
    section_title: str = field(default="")

    def __repr__(self) -> str:
        preview = self.text[:80].replace("\n", " ")
        return (
            f"RegulationChunk("
            f"article={self.article!r}, "
            f"score={self.score:.3f}, "
            f"text={preview!r}...)"
        )


# ---------------------------------------------------------------------------
# Retriever
# ---------------------------------------------------------------------------


class RagRetriever:
    """Holds an initialised Qdrant client and sentence encoder, and answers
    natural-language queries against the FIA regulation collection.

    Both the Qdrant client and the embedding model are loaded eagerly on
    construction so that the first call to ``query()`` has no cold-start
    penalty. Instantiate once at module level (or via ``get_retriever()``)
    and reuse across all agent calls: re-creating the encoder on every query
    would add ~2-3 s of overhead per call.
    """

    def __init__(
        self,
        qdrant_path: Path | str,
        collection_name: str,
        embedding_model: str,
        top_k: int = 5,
    ) -> None:
        """Initialise the retriever and verify the Qdrant collection exists.

        Args:
            qdrant_path:     Path to the on-disk Qdrant storage directory created
                             by ``build_rag_index.py``. Accepts both ``Path`` and
                             plain string so callers can pass either without casting.
            collection_name: Name of the Qdrant collection to query. Must match
                             the name used during indexing; a mismatch raises a
                             ``RuntimeError`` with an actionable message rather than
                             a cryptic Qdrant error.
            embedding_model: Sentence-transformers model identifier used to encode
                             queries. Must be the same model used during indexing:
                             mixing models produces meaningless similarity scores.
            top_k:           Default number of chunks to return per query. Can be
                             overridden per call in ``query()`` when a broader or
                             narrower context window is needed.
        """
        from qdrant_client import QdrantClient
        from sentence_transformers import SentenceTransformer

        self._qdrant_path = Path(qdrant_path)
        self._collection_name = collection_name
        self._embedding_model = embedding_model
        self._top_k = top_k
        # Scopes already reported as absent from the index, so the warning in
        # _warn_unindexed_scope fires once per scope instead of once per lap.
        self._unscoped_warned: set[tuple[int | None, str | None]] = set()

        self._client = QdrantClient(path=str(self._qdrant_path))
        self._encoder = SentenceTransformer(embedding_model)

        existing = {c.name for c in self._client.get_collections().collections}
        if collection_name not in existing:
            raise RuntimeError(
                f"Qdrant collection '{collection_name}' not found in {qdrant_path}. "
                "Run `python scripts/build_rag_index.py` to build the index first."
            )

    def _encode(self, text: str) -> list[float]:
        """Encode a single text string into a normalised embedding vector.

        Normalisation (L2) is applied so that dot product and cosine similarity
        are equivalent in Qdrant, which is the convention used during indexing.
        Returns a plain Python list because that is what ``QdrantClient.search``
        expects. Passing a numpy array causes a silent type error in some versions.

        Args:
            text: The string to embed. Typically a user query, but can also be
                  used to embed individual chunks during a reranking pass.
        """
        return self._encoder.encode(text, normalize_embeddings=True).tolist()

    def _build_scope_filter(self, year: int | None, doc_type: str | None):
        """Build the Qdrant payload filter that restricts a search to one season.

        The embedding carries no year signal: three near-identical rulebooks differ
        by a few numbers, so a bi-encoder ranks the 2023 and the 2025 wording of the
        same article almost equally. Measured on the tracked gold set, an unfiltered
        top-5 puts 43 of 75 hits in a season other than the one asked about, which is
        not distinguishable from drawing by chunk share (z = -1.3 against a 64.6%
        baseline). Scoping therefore happens on the payload, where the year is exact,
        rather than in the question text, where naming the season moves the mix by a
        few hits and sometimes the wrong way.

        Args:
            year:     Season to restrict to, coerced to ``int``. ``None`` leaves the
                      search unscoped, which is what the notebooks and both README
                      examples rely on.
            doc_type: Document family to restrict to (``"sporting_regs"``). ``None``
                      leaves it unscoped. Only one family is indexed today, so this
                      discriminates nothing until a technical rulebook is added.

        Returns:
            A ``Filter`` matching every condition given, or ``None`` when neither
            argument was, which callers read as "do not filter".

        Raises:
            ValueError: If ``year`` cannot be coerced to an int. The season arrives
                from ``lap_state["year"]``, so a value that is not a year is a wiring
                fault and stays loud rather than silently returning nothing.
        """
        # Imported here rather than at module level because
        # tests/agents/test_agent_import_cost.py forbids qdrant_client in a fresh
        # agent import, and unlike most of the RAG suite that test runs on CI.
        from qdrant_client.models import Condition, FieldCondition, Filter, MatchValue

        # Annotated with qdrant's own Condition union rather than
        # list[FieldCondition]: a list is invariant, so the narrower element type
        # does not satisfy Filter(must=...) and mypy rejects it.
        conditions: list[Condition] = []
        if year is not None:
            conditions.append(FieldCondition(key="year", match=MatchValue(value=int(year))))
        if doc_type is not None:
            conditions.append(FieldCondition(key="doc_type", match=MatchValue(value=str(doc_type))))

        if not conditions:
            return None
        return Filter(must=conditions)

    def _search(self, vector: list[float], limit: int, scope: Any) -> list[Any]:
        """Run one vector search, optionally restricted to a payload scope.

        Args:
            vector: The encoded question.
            limit:  How many hits to ask Qdrant for.
            scope:  A ``Filter`` from ``_build_scope_filter``, or ``None`` for an
                    unscoped search.

        Returns:
            The raw Qdrant hits, each still carrying its payload and its score.
        """
        response = self._client.query_points(
            collection_name=self._collection_name,
            query=vector,
            limit=limit,
            with_payload=True,
            query_filter=scope,
        )
        return list(response.points)

    def _warn_unindexed_scope(self, year: int | None, doc_type: str | None) -> None:
        """Say once that a scope matched nothing, then let the caller run unscoped.

        A filtered search comes back empty only when no point carries that payload
        value, so an empty filtered result means the scope is absent from the index
        rather than that the question has no answer. Repeating the warning would emit
        it on every routed lap, which is how a configuration problem comes to look
        like flaky data.

        Args:
            year:     The season that matched nothing, named in the message.
            doc_type: The document family that matched nothing, named in the message.
        """
        scope_key = (year, doc_type)
        if scope_key in self._unscoped_warned:
            return

        self._unscoped_warned.add(scope_key)
        logger.warning(
            "The regulation index holds nothing for season=%s doc_type=%s, so this "
            "query and every later one like it run UNSCOPED across all indexed "
            "seasons. Rebuild the index with that document to scope it. Logged once, "
            "not per lap.",
            year,
            doc_type,
        )

    def query(
        self,
        question: str,
        top_k: int | None = None,
        year: int | None = None,
        doc_type: str | None = None,
    ) -> list[RegulationChunk]:
        """Retrieve the most relevant regulation chunks for a natural-language question.

        Encodes the question, searches the Qdrant collection by cosine similarity, and
        maps each hit back to a ``RegulationChunk`` with its source metadata. The
        payload fields (``article``, ``doc_type``, ``year``, ``section_title``) are
        stored verbatim from indexing time, so they are available even when the
        original PDFs are not present at query time.

        When ``year`` is given the search is restricted to that season's rulebook,
        because the same article carries different numbers in different years: the 2024
        chunk of the dry-tyre allocation reads "twelve (12) sets" where 2025 reads
        thirteen. A season the index does not hold falls back to an unscoped search
        with one warning rather than returning nothing, because the regulation block is
        an enrichment and an empty result reaches the agent as "the regulation does not
        cover this case", which is false when the case is covered in another year.

        Args:
            question: The natural-language query to answer. Can be a full sentence
                      ("What are the pit lane speed limits?") or a short phrase
                      ("safety car restart procedure"): the embedding handles both
                      equally well.
            top_k:    Number of chunks to return. When ``None``, falls back to the
                      instance default set at construction time. Pass a larger value
                      (e.g. 10) when the question spans multiple regulation articles
                      and the LLM needs broader context.
            year:     Season whose rulebook to search. ``None`` searches every indexed
                      season, the behaviour every caller had before season scoping.
            doc_type: Document family to search. ``None`` searches all of them.

        Returns:
            List of ``RegulationChunk`` objects ordered by descending cosine
            similarity, every one of them from ``year`` when that season is indexed.
            Empty list only when the collection itself holds no vectors.
        """
        k = top_k if top_k is not None else self._top_k
        vector = self._encode(question)
        scope = self._build_scope_filter(year, doc_type)

        hits = self._search(vector, k, scope)
        if scope is not None and not hits:
            self._warn_unindexed_scope(year, doc_type)
            hits = self._search(vector, k, None)

        return [
            RegulationChunk(
                text=hit.payload.get("text", ""),
                article=" ".join(hit.payload.get("article", "").split()),
                doc_type=hit.payload.get("doc_type", "unknown"),
                year=hit.payload.get("year", 0),
                score=round(float(hit.score), 4),
                section_title=hit.payload.get("section_title", ""),
            )
            for hit in hits
        ]

    def health_check(self) -> dict[str, Any]:
        """Return a summary of the collection's current state for diagnostics.

        Useful at notebook startup to confirm the index was built correctly before
        running agent demos. Reports the number of indexed vectors, the embedding
        model in use, and the Qdrant storage path so misconfigurations are caught
        early rather than at query time.

        Returns:
            Dictionary with keys ``collection``, ``vector_count``, ``embedding_model``,
            and ``qdrant_path``. ``vector_count`` is 0 if the collection is empty,
            meaning indexing started but failed partway through.
        """
        info = self._client.get_collection(self._collection_name)
        return {
            "collection": self._collection_name,
            "vector_count": info.points_count,
            "embedding_model": self._embedding_model,
            "qdrant_path": str(self._qdrant_path),
        }


# ---------------------------------------------------------------------------
# Module-level singleton + LangGraph tool wrapper
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def get_retriever(
    qdrant_path: Path | str | None = None,
    collection_name: str | None = None,
    embedding_model: str | None = None,
    top_k: int | None = None,
) -> RagRetriever:
    """Return the process-level singleton ``RagRetriever``, creating it on first call.

    Wrapped in ``functools.lru_cache`` so the embedded Qdrant client is
    instantiated exactly once per process. A second ``QdrantClient(path=...)``
    on the same storage directory fails (local mode holds an exclusive file
    lock), which would break any caller (N31 orchestrator, the chat tool
    ``query_rag_tool``) that reaches into the retriever more than once.

    **What it actually raises is a bare ``RuntimeError``**, not portalocker's
    ``AlreadyLocked``: ``qdrant_client/local/qdrant_local.py`` catches the lock
    exception itself and re-raises ``"Storage folder ... is already accessed by
    another instance of Qdrant client."``. This sentence used to name the
    portalocker type, and #827's first guard was written from it rather than
    from an executed collision — so the guard caught an exception that never
    occurs and three races of a measurement run died silently. If you need to
    detect the lock, match that message (see
    ``strategy_orchestrator._is_store_locked``), and confirm it against a real
    two-client collision, not against this docstring.

    Args:
        qdrant_path:     Path to the on-disk Qdrant storage. Defaults to
                         ``CFG.qdrant_path`` (``data/rag/qdrant_local/``).
        collection_name: Qdrant collection to query. Defaults to
                         ``CFG.collection_name``.
        embedding_model: Sentence-transformers model to load. Must match the model
                         used when the index was built.
        top_k:           Default number of chunks returned per query. Defaults to
                         ``CFG.top_k``.
    """
    return RagRetriever(
        qdrant_path=qdrant_path or CFG.qdrant_path,
        collection_name=collection_name or CFG.collection_name,
        embedding_model=embedding_model or CFG.embedding_model,
        top_k=top_k or CFG.top_k,
    )


# `config` carries the active season and is deliberately NOT documented in the
# tool's Args: block. LangChain builds the schema the LLM fills in from the typed
# arguments and their docstring entries, and a `year` argument there would put the
# choice of season in the model's hands, which is the failure #320 exists to close
# (driving the real agent graph with a stub model, it invented 2019). RunnableConfig
# is injected by LangChain instead, so it stays out of the schema, and an unconfigured
# `query_rag_tool.invoke({"question": ...})` still runs unscoped, which is what the
# notebook and both README examples do.
#
# The annotation has to stay exactly `RunnableConfig`, with no default and no
# `| None`. This module runs under `from __future__ import annotations`, so
# LangChain resolves the string and injects only on an exact match: written as
# `RunnableConfig | None = None` the parameter is treated as an ordinary argument,
# lands in the schema as `['config', 'question']`, and the season silently never
# arrives. Measured both ways before this line was written.
@tool
def query_rag_tool(question: str, config: RunnableConfig) -> str:
    """Search the FIA regulation index and return the most relevant passages.

    This is the LangGraph-compatible wrapper around ``RagRetriever.query()``.
    The output is a plain string rather than a list of ``RegulationChunk`` objects
    because the LLM receives tool results as text: structured formatting here
    (article reference on its own line, score in brackets) makes it easy for the
    model to cite specific articles in its final answer.

    Each result block follows the pattern:
        [rank] doc_type YEAR — Article X.Y  (score)
        <regulation text>

    Args:
        question: Natural-language question about FIA regulations. Works for
                  procedural queries ("what happens when a safety car is deployed"),
                  rule lookups ("pit lane speed limit"), and sanction checks
                  ("penalty for causing a collision").
    """
    season = (config or {}).get("configurable", {}).get("season")

    retriever = get_retriever()
    chunks = retriever.query(question, year=season)

    if not chunks:
        return "No relevant regulation passages found for this query."

    blocks = []
    for i, chunk in enumerate(chunks, start=1):
        article_ref = f" — {chunk.article}" if chunk.article else ""
        header = f"[{i}] {chunk.doc_type} {chunk.year}{article_ref}  (score: {chunk.score:.3f})"
        blocks.append(f"{header}\n{chunk.text}")

    return "\n\n".join(blocks)
