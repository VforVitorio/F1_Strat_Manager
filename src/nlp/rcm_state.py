"""Stateful Race Control tracker — fixes the Safety-Car override drop (NR-02, #305).

``sc_currently_active`` was derived per lap from ONLY that lap's Race Control
Messages. A real "SAFETY CAR DEPLOYED" message is a one-shot FIA announcement (a
single row at the deploy lap); it is never repeated on laps 8, 9, 10 of the same
neutralisation, while the per-lap simulation loop rebuilds ``RaceState`` from
scratch each lap with ``rcm_events=[]``. So a stateless classifier saw an empty
window after the deploy lap and reported the Safety Car GONE mid-stint — exactly
when the STAY_OUT-under-SC pit override matters most.

:class:`RaceControlStateTracker` persists the SC/VSC state across laps: it goes
active on a deploy event and stays active until a release event (or a safety-valve
cap), so the per-lap loop can re-assert an active Safety Car to the agents on the
laps that carry no fresh message.
"""

from __future__ import annotations

from dataclasses import dataclass

# Event-type strings emitted by ``radio_agent._classify_rcm_event``. Mirrored
# here (not imported from ``race_situation_agent``) to keep this tracker free of
# the agent/model import weight; kept in sync with that module's frozensets.
_SC_ACTIVE_EVENT_TYPES: frozenset[str] = frozenset(
    {"SAFETY_CAR_DEPLOYED", "VIRTUAL_SAFETY_CAR_DEPLOYED"}
)
_SC_RELEASE_EVENT_TYPES: frozenset[str] = frozenset(
    {"SAFETY_CAR_ENDING", "SAFETY_CAR_IN_PIT_LANE", "VIRTUAL_SAFETY_CAR_ENDING"}
)

# A Safety Car that never receives a release message auto-clears after this many
# laps, so a missed or mis-parsed end-message cannot pin the override for the
# whole race — a worse failure than the drop this class fixes. Real full-SC
# periods run ~3-6 laps; the ceiling is deliberately generous.
# ponytail: fixed cap; swap for FastF1 TrackStatus if a longer neutralisation ever matters.
_MAX_SC_LAPS: int = 8


def _event_types(rcm_events: list | None) -> list[str]:
    """Classify a lap's RCM events into canonical event-type strings.

    Mirrors ``race_situation_agent._sc_active_from_rcm``'s dispatch:
    pre-classified dicts (already carrying ``event_type``) pass through;
    ``RCMEvent`` instances and raw FastF1-shaped dicts are classified via
    ``radio_agent`` — imported lazily to avoid the agents import loop.
    """
    if not rcm_events:
        return []

    out: list[str] = []
    # Import radio_agent lazily and ONLY when a raw event actually needs
    # classifying: a caller passing pre-classified `event_type` dicts (the CLI's
    # synthetic events, the hermetic tests) must never pull in the heavy
    # agent/model stack, which fails at import on a data-less CI runner.
    radio_agent = None
    for ev in rcm_events:
        if isinstance(ev, dict) and "event_type" in ev:
            out.append(str(ev["event_type"]))
            continue

        if radio_agent is None:
            from src.agents import radio_agent

        if isinstance(ev, radio_agent.RCMEvent):
            out.append(radio_agent._classify_rcm_event(ev))
        elif isinstance(ev, dict):
            out.append(
                radio_agent._classify_rcm_event(
                    radio_agent.RCMEvent(
                        message=str(ev.get("message", "")),
                        flag=str(ev.get("flag", "") or ""),
                        category=str(ev.get("category", "")),
                        lap=int(ev.get("lap", 0) or 0),
                        racing_number=ev.get("racing_number") or ev.get("RacingNumber"),
                        scope=str(ev.get("scope", "") or ""),
                    )
                )
            )
    return out


@dataclass
class RaceControlStateTracker:
    """Persists Safety-Car / VSC state across the laps of one race run.

    Feed it every lap's RCM events in lap order via :meth:`ingest`; read
    :attr:`sc_active` for whether a neutralisation is in force right now. When it
    is active but a given lap carried no fresh deploy message,
    :meth:`should_inject` is True and :meth:`synthetic_event` supplies a
    pre-classified event to re-assert it to the agents.

    Invariants:
        - A release event clears the state (release wins over a deploy in the
          same window, matching the stateless classifier).
        - State persists across laps with no SC signal, bounded by
          :data:`_MAX_SC_LAPS` so a missed end-message cannot pin it forever.
    """

    sc_active: bool = False
    sc_kind: str = ""  # "SC" | "VSC" | ""
    deployed_lap: int | None = None
    last_seen_lap: int | None = None

    def ingest(self, lap: int, rcm_events: list | None) -> None:
        """Update SC state from this lap's RCM events. Call once per lap, in order.

        Resolution order matches the stateless classifier so a deploy+release in
        the same window releases: a release event clears; else a deploy event
        sets/refreshes the state; else the prior state persists, bounded by the
        safety cap.
        """
        self.last_seen_lap = lap
        types = _event_types(rcm_events)

        if any(t in _SC_RELEASE_EVENT_TYPES for t in types):
            self._clear()
            return

        deploy = next((t for t in types if t in _SC_ACTIVE_EVENT_TYPES), None)
        if deploy is not None:
            self.sc_active = True
            self.sc_kind = "VSC" if deploy.startswith("VIRTUAL") else "SC"
            self.deployed_lap = lap
            return

        # No fresh signal: persist, but do not pin forever if the end-message was
        # missed or mis-parsed.
        if (
            self.sc_active
            and self.deployed_lap is not None
            and lap - self.deployed_lap >= _MAX_SC_LAPS
        ):
            self._clear()

    def should_inject(self, lap: int) -> bool:
        """True when SC is active but this lap carried no fresh deploy message.

        On such laps the per-lap loop must re-assert the neutralisation to the
        agents (which only ever see this lap's RCM window) via
        :meth:`synthetic_event`. On the deploy lap itself the real message is
        present, so no injection is needed.
        """
        return self.sc_active and self.deployed_lap != lap

    def synthetic_event(self) -> dict[str, object]:
        """A synthetic RCM row re-asserting the active neutralisation.

        Message-shaped (NOT an ``event_type`` dict) on purpose: both engine
        profiles coerce ``race_state.rcm_events`` through
        ``strategy_orchestrator._to_rcm_event``, which builds an ``RCMEvent`` from
        the ``message`` / ``category`` keys and **drops** ``event_type``. A message
        row survives that coercion and re-classifies to (V)SAFETY_CAR_DEPLOYED, so
        the SC override fires on the persisted laps with zero edits in
        ``src/agents``. Note: like the real deploy message, this reaches the radio
        agent as a Safety-Car alert each persisted lap - deliberate, an SC in force
        every lap is exactly what a pit wall keeps reacting to.
        """
        msg = "VIRTUAL SAFETY CAR DEPLOYED" if self.sc_kind == "VSC" else "SAFETY CAR DEPLOYED"
        return {
            "message": msg,
            "category": "SafetyCar",
            "flag": "",
            "lap": self.last_seen_lap or 0,
            "scope": "Track",
        }

    def _clear(self) -> None:
        self.sc_active = False
        self.sc_kind = ""
        self.deployed_lap = None
