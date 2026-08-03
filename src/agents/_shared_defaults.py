"""Fallbacks shared by the agent modules when session_meta or weather arrives incomplete.

A leaf module — no other agent internals, no heavy imports — so pulling these in never
drags in model weights (same reasoning as ``tire_parsing.py``).
"""

from __future__ import annotations

from typing import Any, Mapping

# RaceStateManager.get_session_meta() always supplies total_laps (CLAUDE.md section 6,
# "lap_state is the single contract"; race_state_manager.py's own get_session_meta sets
# it unconditionally). This constant only guards a hand-built session_meta -- a test
# fixture, or a partially populated state -- from resolving a missing key to a
# different number at every call site. 57 is the median/mode race length across the
# 2023-2025 dataset (71 races) -- NOT "2022-2025": there is no 2022 season anywhere in
# data/ (CLAUDE.md section 1). Previously restated as a bare literal in
# pit_strategy_agent.py (x3), race_situation_agent.py (x2) and tire_agent.py (x1);
# consolidated here so a future change updates every caller at once instead of drifting
# one site at a time.
DEFAULT_TOTAL_LAPS: int = 57


def reading_or_default(source: Mapping[str, Any], key: str, default: float) -> float:
    """Read a numeric reading that may be ABSENT or PRESENT-AND-``None``.

    ``dict.get(key, default)`` only fires its default when the KEY is missing. Our
    producers deliberately report an unmeasured reading as the key present with a
    ``None`` value -- ``_safe_none`` in the telemetry backend and
    ``race_state_manager.get_weather_state`` both do this on purpose, so that an absent
    measurement never becomes a searchable sentinel (#465). The two conventions meet
    badly: ``wx.get('air_temp', 28.0)`` returns ``None``, and the ``float()`` one layer
    down raises ``TypeError``.

    That is not hypothetical. Every 2025 laps parquet ships without weather columns, so
    the backend's producer emits ``None`` for all four readings on every 2025 lap. It
    crashed ``/recommend`` with a 422 for the whole default season (#788) via
    ``race_situation_agent``, and it silently moved ``tire_agent``'s cliff estimate 2.3
    laps in the optimistic direction -- the dangerous one, since it delays the pit call.

    The bug shape is a twin that never got the fix: ``pace_agent`` had guarded this read
    with an inline conditional and a comment describing the exact crash, while the
    identical reads in ``tire_agent`` and ``race_situation_agent`` had not. Hence one
    named function rather than a fourth inline copy.

    ``default`` stays a per-caller argument on purpose: the agents disagree on the
    fallback temperatures (pace uses 25/35, tire and race_situation use 28/38) and
    reconciling those numbers is a modelling decision tracked in #789, not something to
    smuggle in behind a crash fix.
    """
    value = source.get(key)
    if value is None:
        return default
    return value
