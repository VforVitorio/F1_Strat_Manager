"""Hardening tests for the Tire Agent — #476 (unvalidated LLM driver input) and
#477 (Rainfall hardcode + negative degradation-rate regex).

No LLM calls: `_parse_tool_outputs` is a pure string parser, and
`_validate_driver_on_track` / the tool functions built by `_build_tools()`
return before any TCN inference or ReAct invocation when the guard fires.
Nothing here builds a react agent or talks to an LLM endpoint.

#476 — `predict_tire_deg_tool` and `estimate_laps_to_cliff_tool` accepted any
free-text driver code from the LLM. `laps_df` carries the WHOLE race's
history, so a driver who crashed early still has rows the stint filter in
`_get_driver_stint` can happily build a "stint" from — it never checked
whether the driver was still racing at the lap the agent is currently
analysing. Austin 2024 is the exact repro named in the issue: HAM's last
real row is LapNumber=2 (confirmed against the raw parquet below), the race
runs to lap 56, and asking about him at a later lap used to return a
confident P50 instead of an error.

#477 — `_parse_tool_outputs`'s regexes used a bare `[\\d.]+` digit class,
which cannot match a leading minus. A negative degradation rate is real and
expected per the tire agent's own system prompt ("a negative degradation
rate means the driver is improving pace... this is real, not an error"), so
the parse miss silently fell through to the 0.0 default instead of raising.
"""

from __future__ import annotations

from itertools import islice
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).parent.parent.parent
RACE_DIR = ROOT / "data" / "raw" / "2024" / "Austin"
_HAS_MODELS = (ROOT / "data" / "models" / "tire_degradation" / "routing_config.json").exists()
_HAS_RACE_DATA = (RACE_DIR / "laps.parquet").exists()

# TireAgent() loads the TCN bundles at __init__ time, and the fixture below
# needs the real Austin 2024 raw parquet to derive a genuine rivals list.
# Both come from Hugging Face, not git, so a CI runner without data/ skips
# this whole module rather than failing on missing fixtures.
pytestmark = pytest.mark.skipif(
    not (_HAS_MODELS and _HAS_RACE_DATA),
    reason="needs data/models/tire_degradation/ and data/raw/2024/Austin/ (data/ comes from HF, not git)",
)


class _FakeToolMessage:
    """Minimal stand-in for a LangChain ToolMessage — only `.content` is read."""

    def __init__(self, content: str) -> None:
        self.content = content


# ---------------------------------------------------------------------------
# #477 — negative degradation rate must parse, not silently drop
# ---------------------------------------------------------------------------


def test_parse_tool_outputs_matches_a_negative_degradation_rate():
    """The regex must accept a leading minus, not just `[\\d.]+`."""
    from src.agents.tire_agent import _parse_tool_outputs

    message = _FakeToolMessage(
        "Driver NOR | Compound C2 | TyreLife 12\n"
        "Cumulative degradation: -1.200 s | Degradation rate: -0.05 s/lap"
    )

    parsed = _parse_tool_outputs([message])

    assert parsed["deg_rate"] == -0.05


# ---------------------------------------------------------------------------
# #476 — the tools must refuse a driver who is not on track
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def tire_agent_at_austin_lap_30():
    """A TireAgent wired exactly as run_from_state() wires it, Austin 2024 lap 30.

    HAM crashed on lap 2 of this 56-lap race — his last raw-data row is
    LapNumber=2 (real reference data, not a synthetic fixture). NOR ran the
    full distance, so he is a legitimate "still on track" control case.
    """
    from src.agents.tire_agent import TireAgent
    from src.simulation.replay_engine import RaceReplayEngine

    replay = RaceReplayEngine(RACE_DIR, driver_code="NOR", team="McLaren", interval_seconds=0.0)
    lap_state = next(islice(replay.replay(), 29, 30))  # 0-indexed 29 -> lap 30

    agent = TireAgent()
    agent.laps_df = pd.read_parquet(RACE_DIR / "laps.parquet")
    agent.session_meta = {"current_lap": 30, "total_laps": 56}
    agent._live_drivers = {"NOR"} | {r["driver"] for r in lap_state["rivals"] if r.get("driver")}
    return agent


def test_ham_is_absent_from_the_lap_30_rivals(tire_agent_at_austin_lap_30):
    """The premise: HAM crashed on lap 2, so he cannot still be racing at lap 30."""
    live = tire_agent_at_austin_lap_30._live_drivers
    assert "HAM" not in live
    assert "NOR" in live


@pytest.mark.parametrize("tool_name", ["predict_tire_deg_tool", "estimate_laps_to_cliff_tool"])
def test_tool_refuses_a_driver_not_on_track(tire_agent_at_austin_lap_30, tool_name):
    """Asking either LLM-facing tool about HAM at lap 30 must error, not compute.

    Before #476 this returned a confident-looking prediction (e.g.
    'P50: 21867.1') built from HAM's stale pre-crash laps instead of refusing.
    """
    agent = tire_agent_at_austin_lap_30
    tool = next(t for t in agent._tools if t.name == tool_name)

    result = tool.invoke({"driver": "HAM", "compound_id": "C2", "tyre_life": 10})

    assert result.startswith("error:")
    assert "HAM" in result
    assert "NOR" in result  # the valid-drivers list should still name a real car


def test_a_live_driver_is_not_refused_by_the_guard(tire_agent_at_austin_lap_30):
    """The guard must not buy safety by refusing everyone — NOR must pass it."""
    assert tire_agent_at_austin_lap_30._validate_driver_on_track("NOR") is None
