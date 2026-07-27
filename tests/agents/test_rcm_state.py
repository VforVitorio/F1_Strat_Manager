"""Hermetic tests for the stateful Safety-Car tracker (NR-02 / NR-03, #305).

Before #305, ``sc_currently_active`` was stateless: it fired only on the deploy
lap and dropped on the laps in between (which carry no fresh RCM message), so the
STAY_OUT-under-SC override re-armed mid-stint. These pin the persistence across
laps, the corrected classification of the FIA end message, and the safety valve.

The tracker-logic tests feed **pre-classified** event dicts so they stay hermetic
(no ``radio_agent`` import → no NLP model load); the one classification test is
guarded because importing ``radio_agent`` pulls model configs absent on CI.
"""

from __future__ import annotations

import pytest

from src.nlp.rcm_state import _MAX_SC_LAPS, RaceControlStateTracker

# Pre-classified RCM event dicts (the cheap path: RaceStateManager/agents already
# accept `event_type` dicts, so the tracker uses them verbatim without parsing).
DEPLOY = {"event_type": "SAFETY_CAR_DEPLOYED"}
ENDING = {"event_type": "SAFETY_CAR_ENDING"}
VSC_DEPLOY = {"event_type": "VIRTUAL_SAFETY_CAR_DEPLOYED"}
VSC_END = {"event_type": "VIRTUAL_SAFETY_CAR_ENDING"}


def test_sc_persists_between_deploy_and_release():
    """The core NR-02 fix: SC stays active on the laps with no fresh message.

    Replays the Qatar 2025 window: DEPLOYED at lap 7, "IN THIS LAP" at lap 10.
    """
    t = RaceControlStateTracker()

    t.ingest(7, [DEPLOY])
    assert t.sc_active and t.deployed_lap == 7
    assert not t.should_inject(7)  # deploy lap already carries the real message

    t.ingest(8, [])  # no fresh message -> must persist
    assert t.sc_active and t.should_inject(8)
    t.ingest(9, [])
    assert t.sc_active and t.should_inject(9)

    t.ingest(10, [ENDING])
    assert not t.sc_active
    assert not t.should_inject(10)

    t.ingest(11, [])  # stays clear afterwards
    assert not t.sc_active


def test_synthetic_event_is_message_shaped():
    """Message-shaped so it survives _to_rcm_event coercion (which drops event_type)."""
    t = RaceControlStateTracker()
    t.ingest(7, [DEPLOY])
    ev = t.synthetic_event()
    assert ev["message"] == "SAFETY CAR DEPLOYED"
    assert ev["category"] == "SafetyCar"


def test_synthetic_event_survives_engine_coercion_and_fires_override():
    """The end-to-end link the tracker-only tests missed.

    Both engine profiles coerce ``race_state.rcm_events`` via
    ``strategy_orchestrator._to_rcm_event``, which builds an RCMEvent from
    message/category and drops an ``event_type`` dict. This asserts the injected
    event still makes the SC override fire *after* that coercion. Deferred +
    guarded import (heavy agent stack, absent on CI).
    """
    try:
        from src.agents.race_situation_agent import _sc_active_from_rcm
        from src.agents.strategy_orchestrator import _to_rcm_event
    except Exception as exc:  # noqa: BLE001 - agent stack not importable on CI -> skip
        pytest.skip(f"agent stack not importable: {exc}")

    t = RaceControlStateTracker()
    t.ingest(7, [DEPLOY])
    t.ingest(8, [])  # persisted SC lap with no fresh message
    assert t.should_inject(8)
    coerced = [_to_rcm_event(t.synthetic_event())]
    assert _sc_active_from_rcm(coerced) is True


def test_vsc_tracked_separately():
    t = RaceControlStateTracker()
    t.ingest(3, [VSC_DEPLOY])
    assert t.sc_active and t.sc_kind == "VSC"
    assert t.synthetic_event()["message"] == "VIRTUAL SAFETY CAR DEPLOYED"
    t.ingest(4, [VSC_END])
    assert not t.sc_active


def test_release_wins_over_deploy_in_same_window():
    """A deploy + ending in one lap's events releases (matches the classifier)."""
    t = RaceControlStateTracker()
    t.ingest(5, [DEPLOY, ENDING])
    assert not t.sc_active


def test_safety_valve_clears_a_missed_release():
    """A deploy with no release must not pin SC for the whole race."""
    t = RaceControlStateTracker()
    t.ingest(5, [DEPLOY])
    for lap in range(6, 5 + _MAX_SC_LAPS):
        t.ingest(lap, [])
        assert t.sc_active  # still within the generous cap
    t.ingest(5 + _MAX_SC_LAPS, [])
    assert not t.sc_active  # cap reached -> auto-cleared


def test_safety_car_in_this_lap_classifies_as_ending():
    """radio_agent must classify the FIA end message as ending, not deploy (NR-03).

    This is what lets the tracker's release actually fire on real corpus data -
    "SAFETY CAR IN THIS LAP" used to fall through to SAFETY_CAR_DEPLOYED. The
    import is deferred + guarded here: radio_agent loads NLP model configs that
    are absent on a data-less CI runner, so skip there rather than fail.
    """
    try:
        from src.agents.radio_agent import RCMEvent, _classify_rcm_event
    except Exception as exc:  # noqa: BLE001 - missing model configs on CI -> skip
        pytest.skip(f"radio_agent (NLP model configs) not importable: {exc}")

    end = RCMEvent(
        message="SAFETY CAR IN THIS LAP",
        flag="",
        category="SafetyCar",
        lap=10,
        racing_number=None,
        scope="",
    )
    assert _classify_rcm_event(end) == "SAFETY_CAR_ENDING"
    dep = RCMEvent(
        message="SAFETY CAR DEPLOYED",
        flag="",
        category="SafetyCar",
        lap=7,
        racing_number=None,
        scope="",
    )
    assert _classify_rcm_event(dep) == "SAFETY_CAR_DEPLOYED"
