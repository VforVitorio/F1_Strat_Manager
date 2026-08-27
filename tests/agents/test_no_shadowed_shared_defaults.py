"""No agent module may redefine a name ``_shared_defaults`` already exports (#1088).

``race_state_builder`` defined ``DEFAULT_AIR_TEMP_C = 25.0`` and
``DEFAULT_TRACK_TEMP_C = 35.0`` three lines below an import that pulled
``DEFAULT_TOTAL_LAPS`` out of ``_shared_defaults``, where the same two names
hold 24.6 and 34.7. Both pairs were live on the same path.

**The name collision is the defect, not the two values.** #789 existed to
collapse five fallback pairs into one and walked past this one, because a
duplicate under a DIFFERENT name is visible at a glance while a duplicate under
the SAME name reads as the import above it. So this asserts the shape that hid
it rather than the numbers that differed: whether the two pairs should hold one
value is a measurement question with its own before and after, and a test that
demanded equality would have forced that decision as a drive-by.

Parsed with ``ast``, never grepped. A grep for the constant names finds every
import, every call site and every mention in a comment, and finds no
assignment it was not already told to look for.

Companion to ``test_weather_defaults_single_source.py``, which covers the other
half: a module inventing a numeric fallback under no name at all.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent.parent
_AGENTS = ROOT / "src" / "agents"
_SHARED = _AGENTS / "_shared_defaults.py"


def _module_level_assignments(path: Path) -> dict[str, ast.expr]:
    """Every ``NAME = ...`` and ``NAME: T = ...`` bound at module scope.

    Module scope only. A name bound inside a function or a class is scoped by
    construction and cannot be mistaken for the import at the top of the file,
    which is the confusion this guards.

    **It descends into ``if`` and ``try`` bodies, which is not a detail.** A
    name bound in a ``try/except ImportError`` fallback or under
    ``if TYPE_CHECKING:`` still binds at module scope and still shadows the
    import, and those are precisely where a value someone did not want reviewed
    ends up. Reading only ``tree.body`` misses them, and the first version of
    this guard did.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    bound: dict[str, ast.expr] = {}

    def visit(body: list[ast.stmt]) -> None:
        for node in body:
            if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                if node.value is not None:
                    bound[node.target.id] = node.value
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        bound[target.id] = node.value
            # Not FunctionDef or ClassDef: those open a new scope. These do not.
            elif isinstance(node, ast.If):
                visit(node.body)
                visit(node.orelse)
            elif isinstance(node, ast.Try):
                visit(node.body)
                visit(node.orelse)
                visit(node.finalbody)
                for handler in node.handlers:
                    visit(handler.body)

    visit(tree.body)
    return bound


def _shared_constant_names() -> set[str]:
    """The public constants ``_shared_defaults`` exports, by their real names."""
    names = {n for n in _module_level_assignments(_SHARED) if n.isupper()}
    assert "DEFAULT_AIR_TEMP_C" in names, "the module this guard is about stopped exporting it"
    return names


_AGENT_MODULES = sorted(
    p for p in _AGENTS.glob("*.py") if p.name not in {"_shared_defaults.py", "__init__.py"}
)


def test_the_shared_module_exports_the_names_this_guard_reads() -> None:
    """Anchors the guard to something real, so it cannot pass on an empty set.

    A guard whose comparison set is empty is green about nothing, which is the
    failure mode this repo has a written lesson about.
    """
    assert len(_shared_constant_names()) >= 3


@pytest.mark.parametrize("module", _AGENT_MODULES, ids=lambda p: p.name)
def test_no_agent_module_shadows_a_shared_default(module: Path) -> None:
    """A module-level name that also lives in ``_shared_defaults`` is a shadow.

    Against the pre-#1088 tree this fails on ``race_state_builder.py`` naming
    ``DEFAULT_AIR_TEMP_C`` and ``DEFAULT_TRACK_TEMP_C``. It stays green on the
    fix because those became ``RACE_STATE_DEFAULT_*``: a second pair is allowed
    to exist and to hold different numbers, it is just not allowed to wear the
    shared module's name while doing it.
    """
    shared = _shared_constant_names()
    shadowed = sorted(name for name in _module_level_assignments(module) if name in shared)
    assert shadowed == [], (
        f"{module.name} redefines {shadowed} at module level, and _shared_defaults already "
        f"exports those names with different values. Import them, or give this module's own "
        f"constants a distinct prefix so the divergence is visible at the definition."
    )


def test_the_second_pair_still_exists_under_its_own_name() -> None:
    """The rename kept a real second pair; it did not quietly unify the values.

    Which of 24.6/25.0 and 34.7/35.0 is right reaches a served prompt, so it is
    a measurement, and #1088 deliberately did not decide it. This fails if a
    later edit collapses the pair by importing the shared one instead, which
    would be that decision made silently.
    """
    from src.agents._shared_defaults import DEFAULT_AIR_TEMP_C, DEFAULT_TRACK_TEMP_C
    from src.agents.race_state_builder import (
        RACE_STATE_DEFAULT_AIR_TEMP_C,
        RACE_STATE_DEFAULT_TRACK_TEMP_C,
    )

    assert (RACE_STATE_DEFAULT_AIR_TEMP_C, RACE_STATE_DEFAULT_TRACK_TEMP_C) != (
        DEFAULT_AIR_TEMP_C,
        DEFAULT_TRACK_TEMP_C,
    ), "the two pairs were unified without the before/after that decision owes"
