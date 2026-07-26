"""Canonical team names, so a mid-window rebrand does not silently become another team.

F1 teams change name inside the 2023-2025 window this project trains and tests on.
FastF1 reports whatever the season used, but every frozen model artefact carries the
label-encoder classes from ITS training years, so a 2025 name can be absent from a
2024-fitted encoder. Resolving that mismatch is a one-line lookup, and it lived in one
line inside ``pit_strategy_agent`` while the eval harness never got it: 20 of 252 rows
in the 2025 pit holdout were being encoded as ``Alfa Romeo``, a different team (#629).

This module exists so there is one lookup rather than two that drift. It holds no
logic beyond the mapping and the resolver, and nothing here imports an agent, so the
eval layer can use it without pulling the agent stack in behind it.

--- WHERE TO CHANGE IF A TEAM RENAMES AGAIN ---
Add the new FastF1 name here mapping to whatever the trained encoders already know.
If instead you RETRAIN, the encoder learns the new name and the alias becomes
redundant, but leaving it costs nothing and keeps older artefacts loadable.
"""

from __future__ import annotations

# FastF1's current name -> the name the frozen encoders were fitted on.
# `Racing Bulls` is the 2025 entrant; the 2024 encoders know it as `RB`, and the 2023
# ones as `AlphaTauri`. Only the 2025 case is mapped because that is the one the
# holdout hits; a 2023-fitted artefact loaded against 2025 data would need the other.
TEAM_ALIASES: dict[str, str] = {"Racing Bulls": "RB"}


def canonical_team(name: str) -> str:
    """The name a frozen encoder is likely to know, or the input unchanged.

    Deliberately total rather than raising on an unknown name: the caller decides
    what an unrecognised team means, because that answer differs between an agent
    (degrade and carry on) and an eval harness (say so loudly, the metric is at
    stake). Returning the input unchanged keeps that decision at the call site.
    """
    return TEAM_ALIASES.get(name, name)
