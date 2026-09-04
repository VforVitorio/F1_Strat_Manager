"""The INSTALL.md provider table is checked against the code, not against other docs.

`INSTALL.md` carries one table saying what each surface falls back to when
`F1_LLM_PROVIDER` is unset. Before it existed, that claim was spread across seven
documents and disagreed with itself: `INSTALL.md` said the CLI flag defaulted to
`lmstudio` months after #805 changed it to `None`, and `arcade-quick-start.md` told
readers to pass a flag the arcade has never read (#201, #216).

A guard comparing one document against another would have passed on all of that. This
one reads the source file each row names and asserts the constant it actually holds, so
flipping a fallback without editing the table turns it red.

The `.env`-precedence and override columns are prose, not constants, and are not
asserted here; what is pinned is the column a code edit can silently invalidate. The
`--provider` flag's own `default=None` is guarded separately, in
`tests/cli/test_provider_precedence.py`, and is not restated here.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
_INSTALL = ROOT / "INSTALL.md"

# Surface label in the table's first column -> (source file, the literal that file must
# contain for the documented fallback to be true, the fallback the table has to state).
_ROWS: dict[str, tuple[str, str, str]] = {
    "`f1-sim`": (
        "src/agents/strategy_orchestrator.py",
        'os.environ.get("F1_LLM_PROVIDER", "lmstudio")',
        "lmstudio",
    ),
    "`f1-arcade`, `f1-pitwall`": (
        "src/arcade/app.py",
        'os.environ.get("F1_LLM_PROVIDER") or "openai"',
        "openai",
    ),
    "`f1-webapp` chat tab": (
        "src/telemetry/backend/services/chatbot/llm_service.py",
        'os.getenv("F1_LLM_PROVIDER", os.getenv("LLM_PROVIDER", "lmstudio"))',
        "lmstudio",
    ),
    "backend `POST /simulate`": (
        "src/telemetry/backend/api/v1/endpoints/strategy.py",
        'provider: str = Field("lmstudio"',
        "lmstudio",
    ),
}


def _fallback_cell(surface: str) -> str:
    """Return the "Provider when nothing is set" cell for one table row.

    Args:
        surface: the row's first cell, verbatim, as it appears in `INSTALL.md`.

    Returns:
        The third cell of that row, stripped. Fails the test rather than returning a
        sentinel when the row is gone, because a missing row is the table being
        rewritten and the guard needing to be rewritten with it.
    """
    table = _INSTALL.read_text(encoding="utf-8")
    pattern = rf"^\|\s*{re.escape(surface)}\s*\|[^|]*\|([^|]*)\|"
    match = re.search(pattern, table, re.MULTILINE)
    assert match is not None, (
        f"INSTALL.md has no provider-table row for {surface}. The table moved or was "
        f"renamed, so this guard no longer describes it."
    )
    return match.group(1).strip()


@pytest.mark.parametrize("surface", sorted(_ROWS))
def test_the_documented_fallback_is_the_one_in_the_code(surface: str) -> None:
    """Each row's fallback must be the constant its own source file holds."""
    source_path, literal, documented = _ROWS[surface]
    source = ROOT / source_path

    if not source.exists():
        pytest.skip(f"{source_path} is absent (submodule not initialised)")

    assert literal in source.read_text(encoding="utf-8"), (
        f"{source_path} no longer contains {literal!r}, so INSTALL.md's claim that "
        f"{surface} falls back to {documented!r} is no longer readable from the code. "
        f"Re-derive the row before editing this guard."
    )
    assert documented in _fallback_cell(surface), (
        f"INSTALL.md says {surface} falls back to {_fallback_cell(surface)!r}, but "
        f"{source_path} resolves to {documented!r}."
    )
