"""src/agents/rag_agent.py

RAG Agent: extraction from N30_rag_agent.ipynb.

Answers regulation questions by retrieving relevant FIA Sporting Regulation
passages from the local Qdrant vector store (built by scripts/build_rag_index.py)
and synthesising a concise answer via a LangGraph ReAct agent.

The heavy lifting (retriever singleton, query_rag_tool, RagRetriever) lives in
src/rag/retriever.py. This module adds the LangGraph agent wrapper,
the RegulationContext output dataclass, and the two entry points used by N31.

Entry points
------------
run_rag_agent(question, year=None)
    Takes a natural-language regulation question, invokes the ReAct agent,
    and returns a RegulationContext with the LLM answer + source chunks.
    ``year`` scopes retrieval to one season's rulebook and reaches both of the
    retrievals this function makes; None searches every indexed season.

run_rag_agent_from_state(lap_state)
    RSM adapter: extracts the question from lap_state["question"] and the
    season from lap_state["year"], then delegates to run_rag_agent(). laps_df
    is not used (RAG is stateless with respect to lap data).
"""

import json
import importlib.util
import sys
from dataclasses import dataclass, field
from pathlib import Path

# ── Repo root (with root-stop guard for uv tool install) ─────────────────────
_REPO_ROOT = Path(__file__).resolve()
while not (_REPO_ROOT / ".git").exists():
    if _REPO_ROOT.parent == _REPO_ROOT:
        break
    _REPO_ROOT = _REPO_ROOT.parent

if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# ── src/rag imports ────────────────────────────────────────────────────────────
from src.rag.retriever import (  # noqa: E402
    CFG as _RAG_CFG,
    RegulationChunk,
    get_retriever,
    query_rag_tool,
)

from src.agents._shared_defaults import LLM_MAX_RETRIES, subagent_model

# ── Optional LangChain / LangGraph imports ─────────────────────────────
# Probed, not imported. `import langchain_openai` costs 14.3 s measured: it drags
# the langgraph stack and transformers in behind it, and every consumer of the
# name sits inside a factory that already builds its client on first call. So an
# eager import charged that to `f1-sim --help`, to a --no-llm run, and to every
# surface that merely touches this module. find_spec answers the only question
# asked here, is it installed, in about a millisecond and executes nothing.
_LC_OK = (
    importlib.util.find_spec("langchain_openai") is not None
    and importlib.util.find_spec("langchain.agents") is not None
)


# ==============================================================================
# Output dataclass
# ==============================================================================

@dataclass
class RegulationContext:
    """Structured output returned by the RAG agent for a single query.

    Bundles the LLM's plain-language summary with the source regulation chunks
    it was derived from, so downstream agents (N31) can both act on a concise
    answer and cite specific FIA articles without re-reading the raw passages.

    question:
        The original natural-language question that triggered this lookup.
        Stored so the orchestrator can log which queries were issued and
        detect duplicate lookups within a race lap.
    answer:
        LLM-generated summary of the relevant regulation articles, one to
        three sentences, enough for the Strategy Orchestrator to decide
        whether a proposed action is legal without reading the full passage.
        Do NOT use article numbers from this field for citations: the LLM
        may hallucinate them. Use the articles field instead.
    chunks:
        The raw RegulationChunk objects returned by the retriever. Kept
        alongside the summary so callers can filter by article range, year,
        or doc_type when the answer is ambiguous.
    articles:
        Deduplicated list of article references extracted from chunk metadata
        (e.g. ["Article 48.3", "Article 55.1"]). Always use this field for
        citations in strategy log entries: chunk metadata is reliable;
        LLM answer text may hallucinate article numbers.
    """

    question: str
    answer:   str
    chunks:   list[RegulationChunk] = field(default_factory=list)
    articles: list[str]             = field(default_factory=list)

    @property
    def reasoning(self) -> str:
        """Alias for answer: interface consistency with N31.

        N31 reads .reasoning uniformly across all agent outputs (N25-N30).
        For N30 the regulatory answer IS the reasoning: it directly informs
        which strategy options are legal. No separate reasoning field needed.
        """
        return self.answer

    def __repr__(self) -> str:
        return (
            f"RegulationContext("
            f"articles={self.articles}, "
            f"answer={self.answer[:80]!r}...)"
        )


# ==============================================================================
# LangGraph ReAct agent: lazy singleton
# ==============================================================================

# The CONDITION rule (rules 3 and 4) is the load-bearing one and it was added after
# a measured failure, not as good practice. Asked what the regulations say about
# tyre changes under a Safety Car, this agent returned Art. 30.5 n) with its
# applicability clause amputated: the real rule makes wet-weather tyres compulsory
# "if the formation lap is started behind the safety car ... or the race is
# resumed", and penalises a specification change "whilst the safety car is on the
# track AT SUCH TIMES". Dropping the last three words turns a narrow wet-start rule
# into a blanket ban on pitting under any Safety Car. The orchestrator then cited it
# to override a Monte Carlo that favoured stopping, on the lap sixteen cars really
# did stop, in the flagship case of the whole project (#826).
#
# Note what the fix is NOT. Every article number in that answer was genuine, so
# grounding the CITATION would not have caught it. The condition is the thing.
_SYSTEM_PROMPT = """You are an FIA Formula 1 regulation expert agent.
You have access to a tool that retrieves passages from the official FIA Sporting
Regulations (2023–2025). When asked a regulation question:
1. Call query_rag_tool with a precise, focused question.
2. Read the retrieved passages carefully.
3. **State the CONDITIONS under which each rule applies, in the same sentence as the
   rule.** Most FIA articles are conditional ("if the race is resumed...", "at such
   times", "during a suspension", "for the race in Monaco"). A rule quoted without
   its condition becomes a different and usually false rule.
4. **Never generalise a conditional rule to the unconditioned case.** If the
   retrieved passage restricts something only under specific circumstances and the
   question asks about the general case, say so explicitly: "this applies only when
   X; it does not apply otherwise."
5. Answer in 2-4 sentences, citing the exact article numbers (e.g. "Article 48.3").
6. If the question spans multiple articles, cite each one.
7. If no relevant passage is found, say "The regulation does not cover this case."
   Say this rather than stretching a nearby article to fit.

The passages you receive are already restricted to the season being raced, so cite
them as they stand and do not reach for another year's wording.
"""

# Lazy singleton: created on first call to avoid LLM connection at import time
_rag_agent = None


def get_rag_react_agent():
    """Return the cached LangGraph ReAct agent, creating it on first call.

    Uses `subagent_model()` (`F1_LLM_MODEL_AGENTS`, default `gpt-4.1-mini`) when
    `F1_LLM_PROVIDER=openai`, otherwise LM
    Studio at localhost:1234. The agent has one tool: query_rag_tool from
    src/rag/retriever.py. Raises ImportError when langgraph or
    langchain_openai are not installed.
    """
    global _rag_agent
    if _rag_agent is None:
        if not _LC_OK:
            raise ImportError(
                "langgraph or langchain_openai is not installed — cannot build "
                "the RAG agent. Install with: pip install langgraph langchain-openai"
            )
        import os

        from langchain.agents import create_agent
        from langchain_openai import ChatOpenAI

        provider = os.environ.get("F1_LLM_PROVIDER", "lmstudio")
        model_name = subagent_model()
        if provider == "openai":
            llm = ChatOpenAI(model=model_name, temperature=0, timeout=120, max_retries=LLM_MAX_RETRIES)
        else:
            llm = ChatOpenAI(
                model=model_name,
                base_url="http://localhost:1234/v1",
                api_key="lm-studio",
                temperature=0,
                model_kwargs={"parallel_tool_calls": False},
                timeout=120,
                max_retries=LLM_MAX_RETRIES,
            )
        _rag_agent = create_agent(
            model=llm,
            tools=[query_rag_tool],
            system_prompt=_SYSTEM_PROMPT,
        )
    return _rag_agent


# ==============================================================================
# Entry points
# ==============================================================================

def run_rag_agent(question: str, year: int | None = None) -> "RegulationContext":
    """Run the RAG ReAct agent for a single regulation question.

    Invokes the LangGraph agent with query_rag_tool, extracts the final answer
    from the last message, then re-queries the retriever directly to populate
    the RegulationContext with typed RegulationChunk objects.

    The retriever is called twice: once by the agent (via query_rag_tool) to
    retrieve passages for the LLM, and once here to get typed chunk objects for
    the RegulationContext. This is intentional: the @tool wrapper returns a
    formatted string, not RegulationChunk instances, so a second retrieval is
    needed to populate ctx.chunks and ctx.articles.

    BOTH calls take the season, and they have to stay in step. The agent's call
    receives it through the RunnableConfig the graph forwards to the tool, the
    re-query below as a plain argument. Scoping only one of them would leave the
    LLM reading one season while ctx.chunks and ctx.articles cite another, and
    those articles are what the orchestrator prints as citations.

    question:
        Natural-language regulation question from the orchestrator (N31).
        Examples: "What must a driver do when the safety car is deployed?",
        "What is the minimum pit stop time during a race?".

    year:
        Season whose rulebook to search, normally lap_state["year"]. None
        searches every indexed season, which is what the notebook demos and any
        caller predating season scoping get. It is a process-context argument on
        purpose: the model never chooses it, because a model asked to pick the
        year of a regulation is the failure this scoping exists to remove.

    Returns a RegulationContext with answer, chunks, and deduplicated articles.
    Use ctx.articles for citations, not the article numbers in ctx.answer.
    """
    from langchain_core.messages import HumanMessage

    agent  = get_rag_react_agent()
    result = agent.invoke(
        {"messages": [HumanMessage(content=question)]},
        config={"configurable": {"season": year}},
    )
    answer = result["messages"][-1].content

    retriever = get_retriever()
    chunks    = retriever.query(question, year=year)
    articles  = list(dict.fromkeys(c.article for c in chunks if c.article))

    return RegulationContext(
        question=question,
        answer=answer,
        chunks=chunks,
        articles=articles,
    )


def run_rag_agent_from_state(
    lap_state: dict,
    laps_df=None,
) -> "RegulationContext":
    """RSM adapter: extract the question from lap_state and call run_rag_agent.

    The RAG agent is stateless with respect to lap data: it only needs the
    natural-language question. laps_df is accepted for interface consistency
    with other RSM adapters but is not used.

    lap_state keys:
        question (str): Natural-language FIA regulation question. Required.
        year (int, optional): Season to scope retrieval to. Absent means every
            indexed season, so a caller that omits it keeps the old behaviour
            rather than getting an empty result.
        session_meta (dict, optional): Unused, kept for interface parity.

    laps_df:
        Ignored. Accepted so the orchestrator can call all RSM adapters with
        the same signature without branching on agent type.

    Returns a RegulationContext identical to what run_rag_agent() returns.
    Raises KeyError when lap_state does not contain a 'question' key.
    """
    question = lap_state["question"]
    return run_rag_agent(question, year=lap_state.get("year"))
