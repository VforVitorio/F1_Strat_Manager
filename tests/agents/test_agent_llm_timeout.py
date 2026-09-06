"""Agent-side provider-kwarg contract (LLM-cost L-1 / #263, plus the temperature drop).

The sub-agents construct their own ``ChatOpenAI`` clients; without a finite
timeout a stalled provider (a hung LM Studio, a dead socket) can pin a lap for
the SDK's full ~30-min retry budget. Every construction now passes
``timeout=`` + ``max_retries=``. These hermetic tests pin both halves of the
contract - the kwargs are still honored by langchain-openai, and no agent
construction silently loses the timeout - without importing the agents (which
would load the model configs) or hitting the network.

The second half of the file covers the kwarg that is NOT honored. langchain-openai
accepts ``temperature`` for gpt-4.1-mini and silently discards it for the gpt-5.x
family, nulling the attribute instead of raising, so the orchestrator has been
sampling at the provider default while its config said 0.0. These tests live here
rather than in a file of their own because the subject is identical - what the
client library does with the kwargs we pass - and a second file on the same subject
is how the two copies drift apart.

Everything here stays source-level or uses a dummy api key, so it runs on a CI
runner with no weights and no key. That matters more than usual for the
temperature canary: the whole premise of
``documents/audits/AUDIT_ORCHESTRATOR_MEMORY.md`` is that the parameter is dropped,
and if a library upgrade starts honoring it, this file is what tells us.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

_AGENTS_DIR = Path(__file__).parent.parent.parent / "src" / "agents"
_ORCHESTRATOR = _AGENTS_DIR / "strategy_orchestrator.py"
_AGENT_FILES = [
    "pace_agent.py",
    "tire_agent.py",
    "pit_strategy_agent.py",
    "race_situation_agent.py",
    "radio_agent.py",
    "rag_agent.py",
    "strategy_orchestrator.py",
]


def test_chat_openai_still_accepts_timeout_and_retries():
    """langchain-openai honors the two kwargs the agents rely on (lazy, no network)."""
    ChatOpenAI = pytest.importorskip("langchain_openai").ChatOpenAI
    client = ChatOpenAI(model="gpt-4.1-mini", api_key="test-key", timeout=120, max_retries=1)
    assert client.request_timeout == 120.0
    assert client.max_retries == 1


@pytest.mark.parametrize("filename", _AGENT_FILES)
def test_every_agent_chatopenai_has_a_timeout(filename: str):
    """Each ``ChatOpenAI(`` construction in the agent carries a ``timeout=`` kwarg.

    Source-level guard so a future edit that drops the timeout on any of the 12
    construction sites fails here instead of shipping a hang-prone agent. Two per
    module, one per provider branch. ``pace_agent.py`` has none since #778/#780 took
    its LLM out, so it iterates nothing; it stays in the list to cover a future one.
    """
    source = (_AGENTS_DIR / filename).read_text(encoding="utf-8")
    # For each construction, scan from ``ChatOpenAI(`` to its balanced close paren.
    for match in re.finditer(r"ChatOpenAI\(", source):
        start = match.end()
        depth = 1
        i = start
        while i < len(source) and depth > 0:
            if source[i] == "(":
                depth += 1
            elif source[i] == ")":
                depth -= 1
            i += 1
        construction = source[match.start() : i]
        assert "timeout=" in construction, f"{filename}: a ChatOpenAI(...) has no timeout="


# ── The temperature the orchestrator asks for and does not get ────────────────


def _orchestrator_cfg_defaults() -> dict[str, object]:
    """Read OrchestratorCFG's field defaults without importing the orchestrator.

    Importing ``src.agents.strategy_orchestrator`` pulls in the tire agent, which
    reads its routing config at import time, so it is unimportable without the HF
    weights. Parsing the dataclass keeps these tests running on a CI runner that
    has none - and it reads the SAME values the module will, instead of restating
    them here where they would quietly go stale.
    """
    tree = ast.parse(_ORCHESTRATOR.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "OrchestratorCFG":
            fields = {}
            for stmt in node.body:
                if isinstance(stmt, ast.AnnAssign) and stmt.value is not None:
                    fields[stmt.target.id] = ast.literal_eval(stmt.value)
            return fields
    raise AssertionError("OrchestratorCFG not found in strategy_orchestrator.py")


def _served_orchestrator_model() -> str:
    """The model ``_get_orchestrator_llm`` will actually send, resolved its way.

    ``OrchestratorCFG.model_name`` defaults to ``None`` since #264, meaning "the
    layer default", so reading the field alone now yields ``None`` rather than a
    model. This mirrors ``strategy_orchestrator.py``'s own
    ``CFG.model_name or orchestrator_model()``, which keeps the canary below aimed
    at the model the orchestrator really builds with instead of at a literal that
    no longer lives in the dataclass.

    Returns:
        The field's default when it holds one, otherwise ``orchestrator_model()``.
        Importing ``_shared_defaults`` is safe here where importing the
        orchestrator is not: it is a leaf module with no heavy imports, which
        ``tests/agents/test_no_shadowed_shared_defaults.py`` already relies on.
    """
    from src.agents._shared_defaults import orchestrator_model

    declared = _orchestrator_cfg_defaults()["model_name"]
    return declared or orchestrator_model()


def test_the_orchestrator_model_still_discards_temperature():
    """Canary: the audit's central premise is that this kwarg does not survive.

    Not a bug being asserted as correct - a fact being pinned. If a langchain-openai
    or provider upgrade starts honoring temperature for this model, Layer 3 becomes
    materially more deterministic overnight and every number in
    AUDIT_ORCHESTRATOR_MEMORY.md (the 36/41 confidence disagreements between two
    identical passes, the coin flip on the one decision lap) is measured on a
    configuration that no longer exists.

    When this fails: re-run scripts/prompt_ab, then update the audit and the
    OrchestratorCFG docstring. Do not just flip the assertion.
    """
    ChatOpenAI = pytest.importorskip("langchain_openai").ChatOpenAI
    model = _served_orchestrator_model()

    dropped = ChatOpenAI(model=model, api_key="test-key", temperature=0.0)
    kept = ChatOpenAI(model="gpt-4.1-mini", api_key="test-key", temperature=0.0)

    assert kept.temperature == 0.0, (
        "gpt-4.1-mini stopped honoring temperature; the sub-agents relied on it"
    )
    assert dropped.temperature is None, (
        f"{model} now KEEPS temperature - this is good news and it "
        "invalidates the audit's measurements; see this test's docstring"
    )


def test_the_orchestrator_warns_when_its_temperature_is_discarded():
    """``_get_orchestrator_llm`` must call the check, not construct and hope.

    Source-level (AST) rather than behavioural, for the import cost above. The
    failure this guards is silent by construction: the client returns a working
    object either way, so nothing downstream can notice the setting evaporated.
    """
    tree = ast.parse(_ORCHESTRATOR.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "_get_orchestrator_llm":
            called = {
                n.func.id
                for n in ast.walk(node)
                if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
            }
            assert "_temperature_was_dropped" in called, (
                "_get_orchestrator_llm no longer checks whether the client kept "
                "CFG.temperature, so Layer 3 can silently go back to sampling"
            )
            return
    raise AssertionError("_get_orchestrator_llm not found in strategy_orchestrator.py")


def test_the_config_docstring_does_not_promise_determinism():
    """The docstring claimed the opposite of what happens, for every shipped model.

    It read "temperature=0.0 ensures deterministic structured output from Layer 3
    LLM" while the parameter was being discarded. A docstring that asserts a
    guarantee the code does not provide is worse than no docstring: it is why
    nobody checked for years.
    """
    tree = ast.parse(_ORCHESTRATOR.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "OrchestratorCFG":
            doc = (ast.get_docstring(node) or "").lower()
            assert "ensures deterministic" not in doc, (
                "OrchestratorCFG's docstring promises determinism again; the client "
                "discards temperature for the gpt-5.x family"
            )
            assert "requested, not guaranteed" in doc, (
                "OrchestratorCFG's docstring must say temperature is requested, not "
                "guaranteed, so the next reader does not trust it"
            )
            return
    raise AssertionError("OrchestratorCFG not found in strategy_orchestrator.py")
