"""#1155: `no_llm` reaches `run_lap`'s profile, all the way from the request.

`no_llm` was carried through two dataclasses and a constructor call and dropped
at the last hop: `SimConnector._step_once` (`strategy.py:485`) called
`run_strategy_pipeline` without it, and `run_strategy_pipeline`
(`strategy_pipeline.py:33-38`) had no parameter to receive it and hardcoded
`profile="rich"` (`:64`). Setting the flag on the arcade's request changed
nothing, and a unit test of either end would have passed throughout: the
request held the value, and `run_lap` already accepted a `profile` keyword.
The defect lived in the hop between them, so the guard has to sit there too.

`run_lap` is the only thing patched below. `_step_once`, `run_strategy_pipeline`
and `_build_decision` all run for real, so a "fix" that reconnects the wire at
the wrong hop (patching the middle back to a hardcoded profile, say) still
fails this file, not just the line it touched.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest

pytest.importorskip("arcade", reason="the arcade replay is an optional surface")

from src.arcade.strategy import SimConnector, SimulateRequestDTO, StrategyState  # noqa: E402


def _connector(no_llm: bool) -> SimConnector:
    """A `SimConnector` whose request carries `no_llm`, nothing else exercised."""
    request = SimulateRequestDTO(
        year=2025, gp="Lusail", driver="NOR", team="McLaren", no_llm=no_llm
    )
    return SimConnector(request=request, state=StrategyState())


def _fake_race_state() -> SimpleNamespace:
    """Just enough fields for the real, unpatched `_build_decision` to read back."""
    return SimpleNamespace(lap=1, compound="MEDIUM", tyre_life=5, position=3, gap_ahead_s=1.2)


def _fake_run_lap(captured: dict[str, Any]):
    """A `run_lap` stand-in that records `profile` and returns a minimal, valid triple.

    Signature mirrors the real `run_lap` exactly (`race_state, laps_df, lap_state,
    *, profile, return_agent_outputs, memory`) so a caller that drifts its call
    shape fails here instead of at the real boundary.
    """

    def run_lap(
        race_state, laps_df, lap_state, *, profile="rich", return_agent_outputs=True, memory=None
    ):
        captured["profile"] = profile
        captured["memory"] = memory
        rec = SimpleNamespace(
            action="STAY_OUT",
            confidence=0.5,
            reasoning="",
            scenario_scores={},
            pace_mode=None,
            risk_posture=None,
            pit_lap_target=None,
            compound_next=None,
            undercut_target=None,
            contingencies=None,
            key_risks=None,
        )
        return rec, {}, {"total": 0.0}

    return run_lap


@pytest.mark.parametrize("no_llm, expected_profile", [(True, "no-llm"), (False, "rich")])
def test_the_request_reaches_run_lap_as_the_matching_profile(monkeypatch, no_llm, expected_profile):
    """`no_llm=True/False` on the request must reach `run_lap` as `profile="no-llm"`/`"rich"`.

    Drives the real `_step_once` end to end. `_build_race_state` is stubbed
    because race-state construction is not the wire under test here (it has
    its own coverage elsewhere); `run_lap` is stubbed because calling it for
    real needs model weights and, on the rich profile, an LLM client. Every
    hop between the two — `_step_once`, `run_strategy_pipeline`,
    `_build_decision` — is the real production code.
    """
    import src.arcade.strategy_pipeline as pipeline_module

    captured: dict[str, Any] = {}
    monkeypatch.setattr(pipeline_module, "run_lap", _fake_run_lap(captured))

    connector = _connector(no_llm)
    monkeypatch.setattr(
        connector, "_build_race_state", lambda lap_state, prev_lap_time: _fake_race_state()
    )

    connector._step_once(pd.DataFrame(), {"lap_number": 1, "driver": {"lap_time_s": 90.0}}, 90.0)

    assert captured.get("profile") == expected_profile, (
        f"no_llm={no_llm} on the request reached run_lap as "
        f"profile={captured.get('profile')!r}, expected {expected_profile!r} — "
        f"the flag is dropped somewhere between the request and run_lap"
    )


def test_run_strategy_pipeline_defaults_to_the_rich_profile(monkeypatch):
    """A caller that omits `no_llm` entirely must keep the pre-#1155 behaviour.

    `run_strategy_pipeline`'s docstring promises backward compatibility: `memory`
    and `no_llm` are both optional. This pins the second half of that promise —
    the dashboard formatters the docstring names call this positionally without
    ever knowing `no_llm` exists, and they must keep getting `"rich"`.
    """
    from src.arcade.strategy_pipeline import run_strategy_pipeline

    captured: dict[str, Any] = {}
    monkeypatch.setattr("src.arcade.strategy_pipeline.run_lap", _fake_run_lap(captured))

    run_strategy_pipeline(_fake_race_state(), pd.DataFrame(), {"lap_number": 1})

    assert captured["profile"] == "rich"
