"""Guards for #320: retrieval is scoped to one season, and both call sites get it.

Everything here runs on CI. That is the point of the file: the RAG behaviour tests
that need the Hugging Face dataset skip on CI, and before #320 nothing that runs
there asserted anything about the season of a retrieved chunk, so a filter that
silently returned nothing for every query would have passed the whole suite.

The store tests build a three-point collection under ``tmp_path`` with hand-written
vectors, so they need neither the real index nor the embedding model. The wiring
tests drive the real ``@tool`` object and the real ``run_rag_agent``, with only the
LLM and the retriever replaced, because what they guard is which arguments travel,
not what the model answers.
"""

from __future__ import annotations

import logging

import pytest

from src.rag.retriever import RagRetriever, query_rag_tool

COLLECTION = "regs_under_test"
VECTOR = [1.0, 0.0]


@pytest.fixture
def store(tmp_path):
    """A three-point collection, one chunk per season, all on the same vector.

    Identical vectors are deliberate: with nothing to separate the seasons by
    similarity, any same-season result is the filter working rather than the
    ranking happening to agree with it.
    """
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, PointStruct, VectorParams

    client = QdrantClient(path=str(tmp_path / "qdrant"))
    client.create_collection(
        collection_name=COLLECTION,
        vectors_config=VectorParams(size=2, distance=Distance.COSINE),
    )
    client.upsert(
        collection_name=COLLECTION,
        points=[
            PointStruct(
                id=i,
                vector=VECTOR,
                payload={
                    "text": f"rule text for {year}",
                    "article": "Article 30.5",
                    "doc_type": "sporting_regs",
                    "year": year,
                    "section_title": "",
                },
            )
            for i, year in enumerate((2023, 2024, 2025))
        ],
    )
    yield client
    client.close()


@pytest.fixture
def retriever(store):
    """A ``RagRetriever`` bound to the test store, with the encoder left out.

    Built with ``object.__new__`` rather than through ``__init__`` because the
    constructor eagerly loads sentence-transformers, which costs 7.3 s, pulls torch,
    and has nothing to do with what these tests check. The alternative, a test-only
    constructor on the production class, is the thing not to do.
    """
    r = object.__new__(RagRetriever)
    r._client = store
    r._collection_name = COLLECTION
    r._top_k = 5
    r._unscoped_warned = set()
    r._encode = lambda text: VECTOR
    return r


# ---------------------------------------------------------------------------
# The filter, against a real Qdrant store
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("season", [2023, 2024, 2025])
def test_a_scoped_query_returns_only_that_season(retriever, season):
    """The defect itself: without the filter all three seasons come back ranked."""
    chunks = retriever.query("any question", year=season)

    assert [c.year for c in chunks] == [season], (
        f"asked for {season}, got {[c.year for c in chunks]}"
    )


def test_an_unscoped_query_still_sees_every_season(retriever):
    """The default has to stay unscoped, because the notebooks and both READMEs use it."""
    chunks = retriever.query("any question")

    assert sorted(c.year for c in chunks) == [2023, 2024, 2025]


def test_a_season_the_index_does_not_hold_falls_back_instead_of_returning_nothing(
    retriever, caplog
):
    """An empty result is worse than the bias it replaces.

    Qdrant answers a filter it cannot match with an empty list and no error, and the
    agent's prompt turns an empty retrieval into "The regulation does not cover this
    case". That sentence is false when the case is covered in a season the index does
    hold, so a miss degrades to an unscoped search and says so once.
    """
    with caplog.at_level(logging.WARNING, logger="src.rag.retriever"):
        chunks = retriever.query("any question", year=2022)

    assert sorted(c.year for c in chunks) == [2023, 2024, 2025]
    assert sum("holds nothing for season=2022" in r.message for r in caplog.records) == 1


def test_the_fallback_warning_is_not_repeated_per_lap(retriever, caplog):
    """Sixty identical warnings is how a configuration problem looks like flaky data."""
    with caplog.at_level(logging.WARNING, logger="src.rag.retriever"):
        for _ in range(5):
            retriever.query("any question", year=2022)

    assert sum("holds nothing for season=2022" in r.message for r in caplog.records) == 1


def test_a_season_given_as_a_string_still_scopes(retriever):
    """The payload year is an int, and MatchValue does not coerce.

    A string reaching the filter matches no point at all, which the fallback above
    would then paper over as an unscoped search. Coercion happens in
    ``_build_scope_filter`` so a year arriving from an HTTP body or a tool argument
    scopes rather than quietly widening.
    """
    chunks = retriever.query("any question", year="2024")

    assert [c.year for c in chunks] == [2024]


# ---------------------------------------------------------------------------
# The wiring: who is allowed to choose the season
# ---------------------------------------------------------------------------


def test_the_season_is_not_in_the_schema_the_model_fills_in():
    """The season must never be the model's to pick.

    A ``year`` argument on the tool lands in the JSON schema the LLM writes, and a
    model asked to choose the year of a regulation will choose one: driving the real
    agent graph with a stub, it invented 2019. The season therefore arrives as
    RunnableConfig, which LangChain injects and keeps out of the schema.
    """
    fields = sorted(query_rag_tool.args_schema.model_json_schema()["properties"])

    assert fields == ["question"], f"the model can see {fields}"


def test_the_tool_passes_the_configured_season_to_the_retriever(monkeypatch):
    """RunnableConfig only helps if the tool actually reads it."""
    import src.rag.retriever as retriever_module

    seen = {}

    class _Recording:
        def query(self, question, top_k=None, year=None, doc_type=None):
            seen["year"] = year
            return []

    monkeypatch.setattr(retriever_module, "get_retriever", _Recording)

    query_rag_tool.invoke({"question": "q"}, config={"configurable": {"season": 2023}})
    assert seen["year"] == 2023

    query_rag_tool.invoke({"question": "q"})
    assert seen["year"] is None


def test_both_retrieval_sites_receive_the_same_season(monkeypatch):
    """The twin. ``run_rag_agent`` retrieves twice and both calls have to be scoped.

    One call feeds the LLM through the tool, the other populates ``ctx.chunks`` and
    ``ctx.articles``, which is what the orchestrator prints as citations. Scoping
    only the first would have the model reading one season while the recommendation
    cites another, and no other test in the suite compares them.
    """
    import src.agents.rag_agent as rag_agent_module
    import src.rag.retriever as retriever_module

    tool_seasons: list = []
    direct_seasons: list = []

    class _Recording:
        def query(self, question, top_k=None, year=None, doc_type=None):
            direct_seasons.append(year)
            return []

    class _FakeAgent:
        """Stands in for the compiled graph, and calls the real tool the way it does."""

        def invoke(self, payload, config=None):
            query_rag_tool.invoke({"question": "q"}, config=config)
            tool_seasons.append((config or {}).get("configurable", {}).get("season"))
            return {"messages": [type("M", (), {"content": "answer"})()]}

    monkeypatch.setattr(retriever_module, "get_retriever", _Recording)
    monkeypatch.setattr(rag_agent_module, "get_retriever", _Recording)
    monkeypatch.setattr(rag_agent_module, "get_rag_react_agent", _FakeAgent)

    rag_agent_module.run_rag_agent("q", year=2024)

    assert tool_seasons == [2024], "the agent's own retrieval was not scoped"
    assert direct_seasons == [2024, 2024], f"the two retrieval sites disagree: {direct_seasons}"
