"""Narrowing a season laps frame to one race.

A LEAF module, and that is its whole reason to exist. This function used to live in
``engine.py``, which imports every agent at module level — and ``radio_agent`` builds its
three transformer models AT IMPORT. So the backend importing one small helper from there
cost 16.7 s and pulled RoBERTa, the intent head, the NER model, the RAG agent and the
orchestrator into the RAM and VRAM of a worker that may never serve a radio request. That
is exactly what ``src/agents/__init__`` was made lazy to prevent, reintroduced one layer up.

Its real dependencies are pandas, a logger and the GP-name resolver, so it belongs here and
``engine`` re-exports it for the callers and tests that already import it by its old path.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import pandas as pd

from src.f1_strat_manager.gp_slugs import resolve_gp_key

if TYPE_CHECKING:  # pragma: no cover - the annotation only; importing it is the whole cost
    from src.agents.strategy_orchestrator import RaceState

logger = logging.getLogger(__name__)


def _scope_laps_to_gp(
    laps_df: pd.DataFrame,
    lap_state: dict[str, Any] | None,
    race_state: RaceState | None = None,
) -> pd.DataFrame:
    """Narrow a season-wide laps frame to the Grand Prix being analysed (#429).

    Every caller loads ``laps_featured_<year>.parquet``, which holds the WHOLE season,
    and hands it straight to the agents. Their lookups (``_get_lap_row``,
    ``_get_position_map``, ``_get_undercut_candidates``, ``_get_driver_stint``, the SC
    feature builder) filter by Driver and LapNumber but never by GP, so each silently
    resolved to whichever race sorted first or last in the file: measured while analysing
    Lusail, the lap-7 position map came from Zandvoort and the driver's lap row from
    Barcelona. Every one of those lookups wants the single race, so scoping once here
    fixes them all without touching the agents.

    The GP name comes from ``lap_state['session_meta']['gp_name']``, which
    ``RaceStateManager`` emits in the same keyspace the parquet's ``GP_Name`` uses
    (verified: 'Lusail'). When ``lap_state`` is ``None`` (the ``_build_default_lap_state``
    path, #465), there is no ``session_meta`` yet to read a GP from — so, given a
    ``race_state``, this derives the GP the same way ``_build_default_lap_state`` will
    (the (driver, lap) row match) and scopes on THAT. This lets scoping happen BEFORE the
    default lap_state is built instead of after: the previous order scoped on a still-None
    ``lap_state`` (a no-op) and only built the default from the still-unscoped, season-wide
    frame, so a GP's grid could be decided from another race's data.

    Falls back to the full frame, loudly, when the name does not resolve — handing the
    agents an EMPTY frame would be worse than the bug this fixes, and a warning is how we
    find out the keyspaces have drifted apart (see #448).
    """
    # `or {}` on the inner read as well, not just the two-arg get: a producer that emits
    # `"session_meta": null` is a present key holding None, so the default never fires and
    # the chained `.get` raised AttributeError — a 500 where the honest answer is the same
    # loud fallback an unknown GP already takes. Same trap as dict.get/Series.get/getattr,
    # which this project has now hit in all four forms.
    session_meta = (lap_state or {}).get("session_meta") or {}
    gp_name = session_meta.get("gp_name") if isinstance(session_meta, dict) else None
    if gp_name is not None and not isinstance(gp_name, str):
        gp_name = str(gp_name)

    if (
        not gp_name
        and race_state is not None
        and laps_df is not None
        and "GP_Name" in laps_df.columns
    ):
        driver_rows = laps_df[laps_df["Driver"] == race_state.driver]
        lap_row = driver_rows[driver_rows["LapNumber"] == race_state.lap]
        if not lap_row.empty:
            gp_name = str(lap_row["GP_Name"].iloc[0])

    if not gp_name or laps_df is None or "GP_Name" not in laps_df.columns:
        if laps_df is not None:
            logger.warning(
                "Could not resolve a GP to scope laps by (gp_name=%r) — falling back to "
                "the unscoped frame; agent lookups may resolve to the wrong race (#429/#465)",
                gp_name,
            )
        return laps_df

    # Resolve the spelling first: the replay path scopes with the metadata name while the
    # frame is keyed by the parquet slug, so 2025 Miami matched nothing and took the
    # fallback below — the whole race ran on the UNSCOPED season frame, which is the very
    # regression the fallback's warning names (PR3_GP_KEYSPACE_SWEEP.md).
    stored_name = resolve_gp_key(set(laps_df["GP_Name"].dropna().astype(str)), gp_name)
    scoped = laps_df[laps_df["GP_Name"] == stored_name]
    if scoped.empty:
        logger.warning(
            "GP %r not found in the laps frame (%d rows, %d GPs); falling back to the "
            "unscoped season frame — agent lookups may resolve to the wrong race (#429/#448)",
            gp_name, len(laps_df), laps_df["GP_Name"].nunique(),
        )
        return laps_df
    return scoped
