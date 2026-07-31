"""#766 — the previous-lap sentinel had two values and two substitution rules.

`pace_agent` guards this key with `d.get('prev_lap_time') or 90.0` and carries a
fifteen-line comment explaining why the two-arg `dict.get` form is wrong for it:
`RaceStateManager` emits the key PRESENT with a `None` value whenever no surviving
predecessor exists, and the two-arg form substitutes only for an absent KEY.

`strategy_orchestrator`'s dict path carried `lap_state.get("prev_lap_time", 92.0)`.
Both defects at once: the wrong form, so an honest `None` passed straight through,
and a different number for the same quantity. `_predict` reads the value into
`prev + delta` with no NaN branch, so the `None` reached a subtraction and the
exported public entry point raised `TypeError` on the first lap of any stint.

Found by gate G3's twin sweep, which is the point of that sweep: the fix landed on
one producer and its twin one module over never got it.
"""

from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent.parent
_HAS_MODELS = (ROOT / "data" / "models" / "lap_time").is_dir()

pytestmark = pytest.mark.skipif(
    not _HAS_MODELS, reason="importing the agents loads model artefacts (HF, not git)"
)


def _lap_state(*, prev_lap_time):
    """The keys `_run_always_on_agents` documents as required, plus the one under test."""
    return {
        "driver_number": 4,
        "stint": 2,
        "team": "McLaren",
        "year": 2025,
        "gp_name": "Lusail",
        "prev_lap_time": prev_lap_time,
    }


def test_the_sentinel_has_exactly_one_definition():
    """Both modules must read the same object, not two equal literals.

    Equal-but-separate would pass an equality check today and drift tomorrow, which
    is what happened: 90.0 against 92.0 for the same missing value.
    """
    from src.agents import pace_agent, strategy_orchestrator

    assert strategy_orchestrator.MISSING_PREV_LAP_TIME_S is pace_agent.MISSING_PREV_LAP_TIME_S


def test_an_honest_none_is_substituted_and_never_forwarded(monkeypatch):
    """The lap state says "there is no previous lap" the only way it can.

    `None` is the correct value for the first lap of a stint, so the fix is not to
    stop emitting it. It is to substitute at the boundary that cannot represent it,
    with the same rule and the same number as the twin entry point.
    """
    from src.agents import strategy_orchestrator as so

    captured = {}

    def _spy(**kwargs):
        captured.update(kwargs)
        raise _Stop

    class _Stop(Exception):
        pass

    monkeypatch.setattr(so, "run_pace_agent", _spy)

    lap_state = _lap_state(prev_lap_time=None)
    race_state = so.RaceState(
        driver="NOR",
        lap=30,
        total_laps=57,
        position=3,
        compound="MEDIUM",
        tyre_life=5,
        gap_ahead_s=2.0,
        pace_delta_s=0.0,
        risk_tolerance=0.5,
        air_temp=25.0,
        track_temp=35.0,
    )

    with pytest.raises(_Stop):
        so._run_always_on_agents(race_state, lap_state)

    assert captured["prev_lap_time"] == so.MISSING_PREV_LAP_TIME_S
    assert captured["prev_lap_time"] is not None


def test_a_real_previous_lap_is_passed_through_untouched(monkeypatch):
    """The guard above must not swallow a genuine reading.

    Without this the substitution could be unconditional and still pass the other
    test, which would replace every anchor in the dict path with the placeholder.
    """
    from src.agents import strategy_orchestrator as so

    captured = {}

    class _Stop(Exception):
        pass

    def _spy(**kwargs):
        captured.update(kwargs)
        raise _Stop

    monkeypatch.setattr(so, "run_pace_agent", _spy)

    race_state = so.RaceState(
        driver="NOR",
        lap=30,
        total_laps=57,
        position=3,
        compound="MEDIUM",
        tyre_life=5,
        gap_ahead_s=2.0,
        pace_delta_s=0.0,
        risk_tolerance=0.5,
        air_temp=25.0,
        track_temp=35.0,
    )
    lap_state = _lap_state(prev_lap_time=85.304)

    with pytest.raises(_Stop):
        so._run_always_on_agents(race_state, lap_state)

    assert captured["prev_lap_time"] == 85.304
