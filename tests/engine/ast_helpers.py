"""Static inspection of what one function passes to another.

Shared by the two anti-drift guards in this directory, which ask opposite
questions of the same call sites: ``test_engine_threads_every_argument`` asserts
the engine passes everything the orchestrator does, and
``test_memory_scope_is_deliberate`` asserts one specific argument goes only one
way. A second copy of this parser is exactly the shape of defect both files exist
to catch, so it lives here instead.

Static rather than runtime because the alternative is calling the layer functions
for real, which needs the model weights, an LLM client and a race state. The
question is about the call site, not the result.
"""

from __future__ import annotations

import ast
import inspect


def kwargs_passed_by(func, callee: str) -> set[str]:
    """The keyword names ``func`` passes to ``callee``.

    Returns an empty set when ``func`` never calls ``callee``, which callers must
    distinguish from "calls it with no keywords" themselves — assert the call
    exists first, or an empty set reads as a passing test for a function that was
    renamed out from under it.
    """
    tree = ast.parse(inspect.getsource(func).lstrip())
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == callee
        ):
            return {kw.arg for kw in node.keywords if kw.arg}
    return set()
