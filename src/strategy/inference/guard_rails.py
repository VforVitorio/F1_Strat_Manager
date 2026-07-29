"""The deterministic pit guard rails, in a module that imports nothing heavy.

Canonical re-host of the backend's ``apply_guard_rails``. Identical rule set to
``src/telemetry/backend/services/simulation/guard_rails.py`` and the CLI's inline
copy (``run_simulation_cli.py``). Re-hosted rather than imported because the
parent must never depend on the submodule; the backend copy retires when P1
migrates, the CLI copy when the P4 duplicate lands (3 -> 1).

WHY THIS IS ITS OWN MODULE (#708):
These rules used to live in ``no_llm.py``, which imports the agent stack and
therefore loads model weights at import time. That made "what is the minimum
stint?" a question you could not ask without ``data/models/`` on disk — it broke
``f1-eval`` on any install without the weights, and it broke CI. A policy
constant should not cost a LightGBM load, so the rules live here and ``no_llm``
imports them. Anything that needs to MIRROR a rail must import it from here and,
better still, call ``apply_guard_rails`` rather than re-derive its boundaries:
the eval tier initially retyped ``remaining < 3`` against the rail's
``remaining <= 3`` and shipped a test that encoded the off-by-one.
"""

from __future__ import annotations

_PIT_ACTIONS = frozenset({"PIT_NOW", "UNDERCUT", "OVERCUT", "REACTIVE_SC"})
_NO_PIT_BEFORE_LAP = 5
_NO_PIT_LAST_N_LAPS = 3
_CLIFF_P10_SAFE = 2
_MIN_STINT_LAPS = {"SOFT": 8, "MEDIUM": 12, "HARD": 15}
_DEFAULT_MIN_STINT = 10


def apply_guard_rails(
    action: str,
    lap: int,
    total_laps: int,
    compound: str,
    tyre_life: int,
    cliff_p10: float = 99.0,
) -> tuple[str, str | None]:
    """Override *action* with STAY_OUT when a hard strategic constraint fires.

    Rules: no pit before lap 5; no pit in the last 3 laps unless the cliff is
    imminent (cliff_p10 < 2); minimum stint SOFT 8 / MEDIUM 12 / HARD 15. Returns
    ``(action, reason)`` with ``reason=None`` when no rail fired.
    """
    if action not in _PIT_ACTIONS:
        return action, None

    remaining_laps = total_laps - lap

    if lap < _NO_PIT_BEFORE_LAP:
        return "STAY_OUT", f"guard-rail: pit window not open (lap < {_NO_PIT_BEFORE_LAP})"

    if remaining_laps <= _NO_PIT_LAST_N_LAPS and cliff_p10 >= _CLIFF_P10_SAFE:
        return "STAY_OUT", f"guard-rail: too late to pit (<={_NO_PIT_LAST_N_LAPS} laps left)"

    min_life = _MIN_STINT_LAPS.get(compound, _DEFAULT_MIN_STINT)
    if tyre_life < min_life:
        return (
            "STAY_OUT",
            f"guard-rail: minimum stint not reached ({compound} {tyre_life}/{min_life} laps)",
        )

    return action, None
