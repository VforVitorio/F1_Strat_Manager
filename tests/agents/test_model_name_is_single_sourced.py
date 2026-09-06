"""No agent decides its own LLM model, and no config field says it does (#264).

Eight literals in six modules, and the six disagreed about where the policy lived:
tire, race_situation and pit read a ``get_react_agent`` PARAMETER default, radio and
the orchestrator read a dataclass field, rag passed the string inline once per
provider branch. Three of those modules ALSO carried a ``model_name`` field on their
config dataclass, documented as the knob to turn and read by nothing, so editing the
obvious place changed nothing at runtime.

That last shape is why this file has two halves. The static half stops a literal
coming back, the way ``test_llm_retry_budget_is_single_sourced.py`` does for the
retry budget. It cannot see a field that is declared, documented and never read, so
the runtime half builds each client against a recorder and asserts the model that
goes out is the one the shared resolver returns.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent.parent
AGENTS = ROOT / "src" / "agents"

# Named rather than globbed, so deleting a module's client is visible here instead of
# silently shrinking the guard. Same list as the retry-budget guard.
_LLM_MODULES = (
    "strategy_orchestrator.py",
    "pit_strategy_agent.py",
    "race_situation_agent.py",
    "tire_agent.py",
    "radio_agent.py",
    "rag_agent.py",
)

_HAS_WEIGHTS = (ROOT / "data" / "models" / "tire_degradation").is_dir()


def _model_values(source: str) -> list[ast.expr]:
    """Every ``model=<expr>`` passed to a ``ChatOpenAI(...)`` call in the module.

    Scoped to ChatOpenAI on purpose: ``create_agent(model=llm, ...)`` also takes a
    ``model`` keyword, and it receives the built client rather than an identifier.
    """
    return [
        kw.value
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "ChatOpenAI"
        for kw in node.keywords
        if kw.arg == "model"
    ]


def test_the_two_layer_resolvers_exist_and_differ():
    """Both are importable, both return a non-empty id, and they are not the same one."""
    from src.agents._shared_defaults import orchestrator_model, subagent_model

    assert subagent_model() and orchestrator_model()
    assert subagent_model() != orchestrator_model(), (
        "the two layers were split because Layer 3 writes the synthesis and the "
        "sub-agents fill a small structured output; collapsing them is a decision"
    )


def test_the_resolvers_read_their_environment_variable(monkeypatch: pytest.MonkeyPatch):
    """Set after import and it still lands, which is why these are functions."""
    from src.agents._shared_defaults import orchestrator_model, subagent_model

    monkeypatch.setenv("F1_LLM_MODEL_AGENTS", "sentinel-agents")
    monkeypatch.setenv("F1_LLM_MODEL_ORCHESTRATOR", "sentinel-orchestrator")

    assert subagent_model() == "sentinel-agents"
    assert orchestrator_model() == "sentinel-orchestrator"


@pytest.mark.parametrize("module", _LLM_MODULES)
def test_the_module_never_hardcodes_its_model(module: str):
    """Both provider branches pass a name; neither restates a model id."""
    values = _model_values((AGENTS / module).read_text(encoding="utf-8"))

    assert len(values) == 2, (
        f"{module} builds ChatOpenAI {len(values)} times, expected 2 (the openai "
        f"branch and the lmstudio branch). A new client needs the resolver too; a "
        f"removed one needs this count updated."
    )
    for value in values:
        assert isinstance(value, ast.Name), (
            f"{module}:{value.lineno} passes a literal model id to ChatOpenAI. Resolve "
            f"it through subagent_model() / orchestrator_model() in "
            f"src.agents._shared_defaults, so the policy moves in one place."
        )


@pytest.mark.data
@pytest.mark.skipif(not _HAS_WEIGHTS, reason="agent imports need the HF weights")
def test_every_layer_sends_the_model_the_resolver_returns(monkeypatch: pytest.MonkeyPatch):
    """The DECLARED model is the SERVED model, for all six clients.

    The static half above cannot catch the defect this issue was filed for: three
    config dataclasses carried a ``model_name`` field that was documented as the knob
    and read by nothing, so the value a reader would edit and the value that reached
    the provider were different strings. Recording what ChatOpenAI is actually
    constructed with is the only check that sees that.

    Hermetic despite the marker: ChatOpenAI and create_agent are replaced, so nothing
    leaves the process. The marker is here because importing the agents pulls the tire
    agent's routing config, which needs the weights.
    """
    import langchain.agents
    import langchain_openai

    sent: list[str] = []

    class _Recorder:
        def __init__(self, **kwargs):
            sent.append(kwargs.get("model"))

        def with_structured_output(self, *args, **kwargs):
            return self

        def bind_tools(self, *args, **kwargs):
            return self

    monkeypatch.setattr(langchain_openai, "ChatOpenAI", _Recorder)
    monkeypatch.setattr(langchain.agents, "create_agent", lambda *a, **k: object())
    monkeypatch.setenv("F1_LLM_PROVIDER", "lmstudio")
    monkeypatch.setenv("F1_LLM_MODEL_AGENTS", "sentinel-agents")
    monkeypatch.setenv("F1_LLM_MODEL_ORCHESTRATOR", "sentinel-orchestrator")

    from src.agents import radio_agent, rag_agent, strategy_orchestrator
    from src.agents.pit_strategy_agent import PitStrategyAgent
    from src.agents.race_situation_agent import RaceSituationAgent
    from src.agents.tire_agent import TireAgent

    def _uninitialised(cls):
        """An instance with only what ``get_react_agent`` reads, so no weights load."""
        agent = object.__new__(cls)
        agent._react_agent = None
        agent._tools = []
        return agent

    for name, build in (
        ("tire", lambda: _uninitialised(TireAgent).get_react_agent()),
        ("race_situation", lambda: _uninitialised(RaceSituationAgent).get_react_agent()),
        ("pit", lambda: _uninitialised(PitStrategyAgent).get_react_agent()),
        (
            "radio",
            lambda: (setattr(radio_agent, "_structured_llm", None), radio_agent._get_radio_llm()),
        ),
        ("rag", lambda: (setattr(rag_agent, "_rag_agent", None), rag_agent.get_rag_react_agent())),
    ):
        sent.clear()
        build()
        assert sent and set(sent) == {"sentinel-agents"}, (
            f"{name} sent {sorted(set(sent))}, not the sub-agent resolver's value. A "
            f"model_name that is declared but never read is exactly the #264 defect."
        )

    sent.clear()
    monkeypatch.setattr(strategy_orchestrator, "_orchestrator_llm", None)
    strategy_orchestrator._get_orchestrator_llm()
    assert set(sent) == {"sentinel-orchestrator"}, (
        f"the orchestrator sent {sorted(set(sent))}, not the Layer 3 resolver's value"
    )
