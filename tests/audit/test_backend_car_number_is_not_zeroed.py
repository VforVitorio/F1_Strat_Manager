"""Guards gate finding F-A2: a missing car number must not become 0 in the backend.

#831's argument is that `DriverNumber = 0` is not an absent value but a **findable**
one: N06 trains on car numbers 1-81, so a 0 sorts below every real number and sends
each `DriverNumber` split down its left branch. That PR fixed the replay path and
left both backend lap-state producers doing `int(_safe(row.get("DriverNumber", 0)))`
— the same defect, one layer away from where the argument was made, which is this
repository's most repeated shape.

WHY THIS TEST LIVES IN THE PARENT REPO
--------------------------------------
The code it guards is in the `src/telemetry` submodule, which has **no test runner
of its own** (only vendored library suites under `.venv`). So a test placed there
would never execute. The parent's CI is the only thing that runs, which makes this
the only place the guard can sit.

It asserts on the PARSED SOURCE rather than on a string match, per this repo's own
"grep is not an audit" lesson: a comment mentioning `DriverNumber` must not satisfy
it, and a real assignment must not slip past it because the formatting changed.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

_BACKEND_ENDPOINT = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "telemetry"
    / "backend"
    / "api"
    / "v1"
    / "endpoints"
    / "strategy.py"
)

# The helper both producers must route through. It returns None for an absent
# reading and an int otherwise -- the int matters because the first producer falls
# back to the RAW per-race parquet, where DriverNumber is dtype=object holding
# strings, while the featured frame holds int32.
_REQUIRED_HELPER = "_int_or_none"


def _driver_number_values(tree: ast.AST) -> list[ast.AST]:
    """Every value assigned to a `"driver_number"` key in a dict literal."""
    found: list[ast.AST] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Dict):
            continue
        for key, value in zip(node.keys, node.values):
            is_the_key = isinstance(key, ast.Constant) and key.value == "driver_number"
            if is_the_key:
                found.append(value)
    return found


@pytest.mark.skipif(
    not _BACKEND_ENDPOINT.exists(),
    reason="src/telemetry submodule not initialised",
)
def test_neither_lap_state_producer_defaults_the_car_number_to_zero():
    """Both producers route through the None-preserving helper, not a 0 default."""
    tree = ast.parse(_BACKEND_ENDPOINT.read_text(encoding="utf-8"))
    values = _driver_number_values(tree)

    # Two lap-state producers plus one response row that legitimately carries a
    # known number. Fewer means a producer was renamed and this guard went blind.
    assert len(values) >= 2, f"expected at least 2 driver_number assignments, got {len(values)}"

    routed_through_helper = [
        v
        for v in values
        if isinstance(v, ast.Call)
        and isinstance(v.func, ast.Name)
        and v.func.id == _REQUIRED_HELPER
    ]
    assert len(routed_through_helper) == 2, (
        f"expected exactly 2 producers using {_REQUIRED_HELPER}(), found "
        f"{len(routed_through_helper)}; a missing car number is being coerced to a "
        f"value the model can find (gate finding F-A2 on #831)"
    )

    # The specific regression: the helper called with a literal 0 fallback, or the
    # old `_safe(..., 0)` / `_s(..., 0)` shape, defeats the whole point.
    for call in routed_through_helper:
        for arg in call.args:
            if isinstance(arg, ast.Call) and arg.args:
                literals = [a.value for a in arg.args if isinstance(a, ast.Constant)]
                assert 0 not in literals, (
                    "a 0 default was reintroduced inside the car-number lookup; "
                    "absent must stay absent and reach the model as NaN"
                )
