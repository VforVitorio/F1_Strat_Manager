"""No-LLM profile tests for the shared engine — the executable #166 fix.

These prove the deterministic ``run_lap(profile="no-llm")`` path:
  * constructs ZERO LLM clients (a bomb on every agent module's ``ChatOpenAI``
    ``__init__`` fires if any client is built) — so ``--no-llm`` can never crash on
    the old 3-tuple, never pays retry backoff, and never silently becomes LLM mode;
  * runs every lap without raising, over the whole 9-lap mini_race fixture;
  * respects the Safety-Car RCM override (``sc_currently_active`` -> routing forces
    N28/N30) while still never running the LLM-backed N28/N30 (``pit_out`` stays None);
  * is deterministic.

Data-tier (`_skip_no_models`): importing the engine pulls the agent modules, which
read model configs at import. Runs locally + on the data tier; skips on bare CI.
Hermetic apart from model weights — the ``ChatOpenAI`` bomb, not a live server, is
what proves "zero clients", so no port 1234 / FakeOpenAI stub is needed here.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).parent.parent.parent
_HAS_MODELS = (ROOT / "data" / "models" / "tire_degradation" / "routing_config.json").exists()
_skip_no_models = pytest.mark.skipif(
    not _HAS_MODELS,
    reason="data/models/ not present (CI runner without model weights)",
)

FIXTURE = ROOT / "tests" / "fixtures" / "mini_race.parquet"
_ACTIONS = {"STAY_OUT", "PIT_NOW", "UNDERCUT", "OVERCUT", "ALERT"}
_AGENT_MODULES = (
    "src.agents.tire_agent",
    "src.agents.race_situation_agent",
    "src.agents.radio_agent",
    "src.agents.pit_strategy_agent",
    "src.agents.strategy_orchestrator",
)


class _LLMClientBomb:
    """Explodes if any no-llm code path constructs an LLM client."""

    def __init__(self, *args, **kwargs):
        raise AssertionError("LLM client constructed in the no-llm profile")


@pytest.fixture
def no_llm_clients(monkeypatch):
    """Bomb ``ChatOpenAI`` in every agent module — the zero-clients guarantee."""
    import importlib

    for module_path in _AGENT_MODULES:
        module = importlib.import_module(module_path)
        if hasattr(module, "ChatOpenAI"):
            monkeypatch.setattr(module, "ChatOpenAI", _LLMClientBomb)


def _race_state(df: pd.DataFrame, lap: int, rcm_events=None):
    from src.agents.strategy_orchestrator import RaceState

    row = df[(df["Driver"] == "VER") & (df["LapNumber"] == lap)].iloc[0]
    return RaceState(
        driver="VER",
        lap=lap,
        total_laps=13,
        position=int(row["Position"]),
        compound=str(row["Compound"]),
        tyre_life=int(row["TyreLife"]),
        gap_ahead_s=1.5,
        pace_delta_s=0.2,
        air_temp=28.0,
        track_temp=35.0,
        rcm_events=rcm_events or [],
    )


@pytest.fixture(scope="module")
def rsm_and_df():
    from src.simulation.race_state_manager import RaceStateManager

    df = pd.read_parquet(FIXTURE)
    team = df.loc[df["Driver"] == "VER", "Team"].iloc[0]
    rsm = RaceStateManager(df, driver_code="VER", team=team, gp_name="Lusail", year=2025)
    return rsm, df


@_skip_no_models
def test_no_llm_sweep_produces_a_decision_every_lap_with_zero_clients(no_llm_clients, rsm_and_df):
    """The #166 fix: every fixture lap returns a valid recommendation, no LLM client."""
    from src.strategy.inference.engine import run_lap

    rsm, df = rsm_and_df
    for lap in range(5, 14):
        rec, outs, timings = run_lap(
            _race_state(df, lap), df, rsm.get_lap_state(lap), profile="no-llm"
        )
        assert rec.action in _ACTIONS, f"lap {lap}: bad action {rec.action}"
        assert rec.reasoning.startswith("[no-llm"), f"lap {lap}: {rec.reasoning[:40]}"
        assert set(rec.scenario_scores) == {"STAY_OUT", "PIT_NOW", "UNDERCUT", "OVERCUT"}
        for scores in rec.scenario_scores.values():
            # The projection path adds `eligible` and `target`; the legacy path
            # emits the four numbers alone. Both are valid, so assert the numeric
            # core is always present rather than pinning an exact key set — the
            # exact-set form failed the moment the engine started threading
            # rivals, which is a schema change the surfaces tolerate by design.
            assert {"E", "P10", "P90", "score"} <= set(scores)
            if scores.get("eligible") is False:
                assert scores["score"] is None, "an unoffered candidate must carry no score"
            else:
                assert scores["score"] is not None
        assert outs["pit_out"] is None  # N28 is LLM-backed, never run in no-llm
        assert "total" in timings


@_skip_no_models
def test_no_llm_safety_car_lap_routes_n28_n30_without_running_them(no_llm_clients, rsm_and_df):
    """A deployed SC (RCM) forces sc_currently_active + N28/N30 routing, pit still None."""
    from src.strategy.inference.engine import run_lap

    rsm, df = rsm_and_df
    sc_rcm = [{"message": "SAFETY CAR DEPLOYED", "category": "SafetyCar", "flag": "", "lap": 8}]
    rec, outs, _ = run_lap(
        _race_state(df, 8, rcm_events=sc_rcm), df, rsm.get_lap_state(8), profile="no-llm"
    )
    assert outs["situation_out"].sc_currently_active is True
    assert {"N28", "N30"} <= set(outs["active"])
    assert outs["pit_out"] is None  # routed but never executed (LLM-backed)


@_skip_no_models
def test_no_llm_is_deterministic(no_llm_clients, rsm_and_df):
    from src.strategy.inference.engine import run_lap

    rsm, df = rsm_and_df
    first = run_lap(_race_state(df, 6), df, rsm.get_lap_state(6), profile="no-llm")[0]
    second = run_lap(_race_state(df, 6), df, rsm.get_lap_state(6), profile="no-llm")[0]
    assert first.model_dump() == second.model_dump()
