"""Agent-side provider-timeout contract (LLM-cost L-1 / #263).

The sub-agents construct their own ``ChatOpenAI`` clients; without a finite
timeout a stalled provider (a hung LM Studio, a dead socket) can pin a lap for
the SDK's full ~30-min retry budget. Every construction now passes
``timeout=`` + ``max_retries=``. These hermetic tests pin both halves of the
contract - the kwargs are still honored by langchain-openai, and no agent
construction silently loses the timeout - without importing the agents (which
would load the model configs) or hitting the network.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

_AGENTS_DIR = Path(__file__).parent.parent / "src" / "agents"
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

    Source-level guard so a future edit that drops the timeout on any of the 14
    construction sites fails here instead of shipping a hang-prone agent.
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
