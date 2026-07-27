"""Contract tests for the shared inference engine's public dispatch.

Covers the ``run_lap`` profile contract that CLI/Arcade/backend consumers rely on:
the two delivered profiles, the reserved ``fast`` profile (a pointing error so a
three-valued switch never falls through silently), and an unknown-profile guard.

Data-tier: importing the engine pulls ``src.agents.strategy_orchestrator``, whose
sub-agent modules read model configs at import time, so this carries the same
``_skip_no_models`` guard as the other agent-touching tests. The full
engine-vs-orchestrator byte-parity test (the anti-drift guard) lands with the
no-llm profile in Phase 1.2 (it additionally needs the FakeOpenAI stub on :1234).
"""

from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent.parent
_HAS_MODELS = (ROOT / "data" / "models" / "tire_degradation" / "routing_config.json").exists()
_skip_no_models = pytest.mark.skipif(
    not _HAS_MODELS,
    reason="data/models/ not present (CI runner without model weights)",
)


@_skip_no_models
def test_profiles_are_rich_and_no_llm():
    from src.strategy.inference.engine import PROFILES

    assert PROFILES == ("rich", "no-llm")


@_skip_no_models
def test_fast_profile_is_reserved_with_a_pointing_error():
    """``fast`` is Phase 2 — it must raise, not silently fall through to a default."""
    from src.strategy.inference.engine import run_lap

    with pytest.raises(ValueError, match="fast"):
        run_lap(None, None, profile="fast")  # raises before touching race_state/laps_df


@_skip_no_models
def test_unknown_profile_raises():
    from src.strategy.inference.engine import run_lap

    with pytest.raises(ValueError):
        run_lap(None, None, profile="does-not-exist")
