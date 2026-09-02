"""No agent restates the LLM retry budget as a literal (#1153).

A 429 the API declared recoverable in 148 ms cost a whole lap of an LLM run,
because ``max_retries=1`` is two attempts and a TPM window drains over a minute.
The number itself was the smaller half of the problem: it was written out
**twelve times across six modules**, once per provider branch, so raising it in
one place and missing the other eleven was the likely outcome.

This asserts the shape rather than the value. ``_shared_defaults`` is where the
repo already collapses this kind of drift (see ``DEFAULT_TOTAL_LAPS``), and
``test_no_shadowed_shared_defaults.py`` then stops an agent redefining the name.
What is left to guard is the literal coming back.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent.parent
AGENTS = ROOT / "src" / "agents"

# Every module that builds an LLM client. Named rather than globbed, so deleting a
# module's client is visible here instead of silently shrinking the guard.
_LLM_MODULES = (
    "strategy_orchestrator.py",
    "pit_strategy_agent.py",
    "race_situation_agent.py",
    "tire_agent.py",
    "radio_agent.py",
    "rag_agent.py",
)


def _max_retries_values(source: str) -> list[ast.expr]:
    """Every ``max_retries=<expr>`` passed to a call in the module."""
    return [
        kw.value
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Call)
        for kw in node.keywords
        if kw.arg == "max_retries"
    ]


def test_the_shared_budget_exists_and_beats_a_single_retry():
    """The constant is importable, and it is worth more than the 1 it replaced."""
    from src.agents._shared_defaults import LLM_MAX_RETRIES

    assert isinstance(LLM_MAX_RETRIES, int)
    assert LLM_MAX_RETRIES > 1, (
        "one retry is two attempts, which is what let a mid-burst 429 cost a lap"
    )


@pytest.mark.parametrize("module", _LLM_MODULES)
def test_the_module_builds_its_client_with_the_shared_budget(module: str):
    """Both provider branches read the constant; neither restates a number."""
    path = AGENTS / module
    values = _max_retries_values(path.read_text(encoding="utf-8"))

    assert len(values) == 2, (
        f"{module} passes max_retries {len(values)} times, expected 2 "
        f"(the openai branch and the lmstudio branch). A new client needs the "
        f"constant too; a removed one needs this count updated."
    )
    for value in values:
        assert isinstance(value, ast.Name) and value.id == "LLM_MAX_RETRIES", (
            f"{module}:{value.lineno} passes a literal to max_retries. Import "
            f"LLM_MAX_RETRIES from src.agents._shared_defaults instead, so raising "
            f"the budget moves every call site at once."
        )
