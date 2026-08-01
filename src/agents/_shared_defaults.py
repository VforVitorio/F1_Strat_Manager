"""Fallback constants shared by the agent modules when session_meta arrives incomplete.

A leaf module — no other agent internals, no heavy imports — so pulling in this one
constant never drags in model weights (same reasoning as ``tire_parsing.py``).
"""

from __future__ import annotations

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
