"""Single canonical builder of ``RaceState`` from the ``lap_state`` contract (#784).

Why this module exists: the CLI (``scripts/run_simulation_cli.py``), the Arcade
(``src/arcade/strategy.py``) and the telemetry backend
(``backend/utils/race_state_builder.py``) each carried their own copy of the same
lap_state -> RaceState mapping, and the copies disagreed on values the models
actually receive (track_temp 40.0 vs 35.0, compound "UNKNOWN" vs "MEDIUM",
tyre_life 0 vs 1, three different sources for the lap number).

Be precise about which of those divergences is REACHABLE, because an earlier
draft of this docstring was not: on the replay path (CLI and Arcade, both via
RaceReplayEngine) the temperature defaults are effectively dead with the shipped
data. All 71 race directories carry a readable weather.parquet with zero NaN
AirTemp/TrackTemp rows, so the keys are always present and always non-None, and
40.0-vs-35.0 never actually reached a model there. The value below is chosen on
the measured-median argument alone, not on a live-divergence one. What IS live is
the present-but-None case this builder now handles: see DEFAULT_TRACK_TEMP_C.

The drift that actually bit this codebase was LOGIC drift, not just
literals: the #750 pace-delta axis, the #465 dead position default, the #633 gap
zero-conflation. One implementation is the fix; a parity test was rejected because
it cannot run where the drift originates. Full per-field decision record: issue
#784 and ``documents/audits/DESIGN_race_state_single_contract.md``.

Leaf-module constraint (HARD): ``RaceState`` is imported LAZILY inside
``build_race_state`` because it lives in ``strategy_orchestrator``, which drags
LangChain/LangGraph and every sub-agent's model artefacts at import time.
Importing THIS module must stay cheap - the same discipline
``_shared_defaults.py`` documents - and a test asserts that importing it does not
pull ``langchain`` into ``sys.modules``. Do not add a top-level
``strategy_orchestrator`` import here.

``radio_msgs`` / ``rcm_events`` are PARAMETERS, never built here: the three
surfaces have three different sources (the CLI's OpenF1 corpus plus its
``--radio-every`` synthetic generator with a precedence rule, the Arcade's
``RadioPipelineRunner`` instance state, the backend's request payloads). Building
them here would drag Whisper and per-surface policy into a shared leaf module.

Known coupling, recorded on purpose: the CLI mutates the built object's
``radio_msgs`` / ``rcm_events`` lists AFTER construction (its main loop owns the
corpus-suppresses-synthetic rule). If ``RaceState`` is ever frozen, that
post-construction mutation breaks; the fix then is for the CLI to pass the lists
as parameters here instead.

--- WHERE TO CHANGE IF THE CONTRACT CHANGES ---
Consumers of this builder: ``scripts/run_simulation_cli.py::_build_race_state``
(pure delegation), ``src/arcade/strategy.py::_build_race_state`` (delegation plus
arcade-side radio/RCM sourcing), and the telemetry backend's
``backend/utils/race_state_builder.py`` (a re-export shim, #786; its call sites are
``simulator.py``, ``endpoints/strategy.py`` and ``mcp_tools.py``, none of which
needed changing because the shim preserves the public name).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

from src.agents._shared_defaults import DEFAULT_TOTAL_LAPS
from src.agents.position_projection import GAP_UNKNOWN_FALLBACK_S

logger = logging.getLogger(__name__)

# Canonical defaults. Each literal below was measured on the shipped 2023-2025
# parquets (70 races) before being chosen; the numbers are in the #784 decision
# table and in documents/audits/DESIGN_race_state_single_contract.md (F10).

# Zero rows carry the literal "UNKNOWN" in any season, so it can never collide
# with a real reading. The strings that DO appear when FastF1 has no compound are
# ""/"nan"/"None": race_state_manager.py:352 emits `str(r.get("Compound", ""))`,
# so a stored NaN reaches this builder as the STRING "nan" with the key present,
# where a dict.get default never fires (the Series.get lesson, CLAUDE.md
# section 11). `normalise_compound` folds all of them into this one honest marker.
UNKNOWN_COMPOUND = "UNKNOWN"
_MISSING_COMPOUND_MARKERS = frozenset({"", "nan", "none"})

# TyreLife == 0 occurs ZERO times in 2023/2024/2025 (season minimums 2.0/2.0/1.0),
# so 0 is a non-colliding sentinel. The old arcade/backend default of 1 collides
# with real fresh-tyre laps, which is the #428 bug shape: a default the code can
# also legitimately find.
UNKNOWN_TYRE_LIFE = 0

# Dataset medians (2023+2024 weather columns; the 2025 featured parquet carries
# none): air median 24.1 C, track median exactly 35.0 C. The CLI's old 40.0 track
# default corresponded to nothing measured, which is the whole reason to pick a
# canonical value here.
#
# Where these actually fire, measured rather than assumed: NOT on the replay path
# (all 71 race dirs ship a readable weather.parquet with zero NaN temperature
# rows, so get_weather_state always emits real readings). They fire on the
# BACKEND's producer path, where the 2025 laps parquets carry no weather columns
# at all and the producer honours that as an explicit None per key. That case
# used to reach `float(None)` and raise TypeError; treating present-but-None as
# missing is what makes these constants load-bearing.
DEFAULT_AIR_TEMP_C = 25.0
DEFAULT_TRACK_TEMP_C = 35.0


def _targeting_against_rival(
    lap_state: Dict[str, Any],
    rival: str,
    fallback_gap_s: float,
    fallback_pace_s: float,
) -> Tuple[float, float]:
    """Gap and pace delta of our driver measured against a chosen ``rival``.

    Returns ``(gap_ahead_s, pace_delta_s)`` framed around the rival the user
    picked in the Strategy tab, so the recommendation reasons about the duel the
    user actually asked for instead of about whichever car happens to sit one
    position ahead (#431). Both values land on the RaceState, which feeds N27's
    overtake scoring inputs and the orchestrator's synthesis prompt.

    gap_ahead_s is the absolute on-track interval to the rival, read from the
    rival's ``interval_to_driver_s`` (rival elapsed time minus ours: the sign
    encodes ahead/behind, the magnitude is the gap). This is the same
    driver-relative interval RaceStateManager already emits per rival, mirrored
    onto the /lap-state rivals so both callers carry it.

    pace_delta_s is our last lap time minus the rival's, matching N27's
    convention (negative = we are faster).

    Falls back to the caller-supplied values when the rival is absent from this
    lap (e.g. it crashed out and the liveness filter dropped it, #428/#430) or
    when a lap time is missing, so a stale selection degrades to current
    behaviour rather than to a fabricated zero.
    """
    rivals = lap_state.get("rivals", []) or []
    match = next((r for r in rivals if r.get("driver") == rival), None)
    if match is None:
        return fallback_gap_s, fallback_pace_s

    interval = match.get("interval_to_driver_s")
    gap_ahead_s = abs(float(interval)) if interval is not None else fallback_gap_s

    driver_lap_s = lap_state.get("driver", {}).get("lap_time_s")
    rival_lap_s = match.get("lap_time_s")
    if driver_lap_s and rival_lap_s:
        pace_delta_s = float(driver_lap_s) - float(rival_lap_s)
    else:
        pace_delta_s = fallback_pace_s

    return round(gap_ahead_s, 3), round(pace_delta_s, 3)


def _car_ahead(lap_state: Dict[str, Any], our_position: int) -> Optional[Dict[str, Any]]:
    """The rival sitting exactly one position ahead of us, or None when we lead.

    Presence in the ``rivals`` list is the liveness signal: a retired car simply
    has no row this lap (#428/#430). No age threshold, no fabricated positions.
    """
    rivals = lap_state.get("rivals", []) or []
    match = next((r for r in rivals if r.get("position") == our_position - 1), None)
    return match


def _gap_to_car_ahead(car_ahead: Optional[Dict[str, Any]]) -> float:
    """Absolute interval to the car ahead, without conflating two zeros.

    No car ahead means we lead, and 0.0 is honest there. A car ahead whose
    ``interval_to_driver_s`` was never measured is NOT a zero gap: 0.0 reads as
    side by side, which the orchestrator's clean-air band and N27's sub-1.0s DRS
    window both act on (#633). It degrades to ``GAP_UNKNOWN_FALLBACK_S`` instead.

    Be honest about what that fallback still is: 2.0 is fabricated, and a real
    2.0 s gap is common, so it does not satisfy the rule that a default must
    never be a value the code can also legitimately find. It is less harmful
    than 0.0, not correct. The real fix is ``RaceState.gap_ahead_s`` becoming
    ``float | None``, which ``RivalState`` in position_projection.py already is
    and whose consumers already guard with ``is not None``. That is a Pydantic
    contract change, tracked under #628.
    """
    if car_ahead is None:
        return 0.0
    interval = car_ahead.get("interval_to_driver_s")
    if interval is None:
        return GAP_UNKNOWN_FALLBACK_S
    return abs(float(interval))


def _pace_delta_vs_car_ahead(
    driver_state: Dict[str, Any],
    car_ahead: Optional[Dict[str, Any]],
) -> float:
    """Our lap time minus the car ahead's SAME-lap time, negative = we are faster.

    pace_delta_s is contractually RIVAL-relative (this driver's lap time minus
    the car directly ahead's SAME lap) per race_situation_agent.py:292/904, the
    schema N27 itself computes. The formula this replaced compared the current
    lap against our OWN previous lap instead, a same-car same-driver quantity
    that reported roughly -20 s of phantom "pace gain" on the lap after a pit
    stop, a green lap read against our own out-lap. 0.0 when the car ahead or
    either lap time is unknown is the schema's documented neutral, not a guess
    in either direction (#750).
    """
    our_lap_time = driver_state.get("lap_time_s") or 0.0
    ahead_lap_time = car_ahead.get("lap_time_s") if car_ahead is not None else None
    if ahead_lap_time is not None and our_lap_time:
        return float(our_lap_time) - float(ahead_lap_time)
    return 0.0


def normalise_compound(raw: Any) -> str:
    """Fold every spelling of "no compound" into the one honest marker.

    Matching is case-insensitive so pandas' "nan"/"NaN" spellings and Python's
    "None" all land on ``UNKNOWN_COMPOUND`` (see the marker-set comment above for
    why these strings exist at all). A real reading passes through untouched; the
    producers already emit canonical casing, so no re-casing happens here.
    """
    if raw is None:
        return UNKNOWN_COMPOUND
    text = str(raw).strip()
    if text.lower() in _MISSING_COMPOUND_MARKERS:
        return UNKNOWN_COMPOUND
    return text


def _weather_reading(weather: Dict[str, Any], key: str, default: float) -> float:
    """Read one weather float, treating a present-but-None value as missing.

    ``get_weather_state`` can emit a weather key whose VALUE is None (a NaN
    weather row, race_state_manager.py:478-479); ``weather.get(key, default)``
    passes that None straight into ``float()`` and crashes, which is exactly what
    all three pre-#784 builders did. The ``is not None`` read is the same pattern
    pace_agent's own from_state adapter uses for these fields.
    """
    value = weather.get(key)
    reading = float(value) if value is not None else default
    return reading


def _resolve_lap(lap_state: Dict[str, Any], driver_state: Dict[str, Any]) -> int:
    """Lap number from the top-level key, then the driver dict, then 1 loudly.

    RaceStateManager emits both copies, so every real path takes the first read;
    the fallbacks only breathe on hand-built lap_states (HTTP clients of
    /recommend, test fixtures). The pre-#784 CLI read ONLY the driver dict and
    crashed on states the other two surfaces accepted.
    """
    lap = lap_state.get("lap_number")
    if lap is None:
        lap = driver_state.get("lap_number")
    if lap is None:
        logger.warning(
            "lap_state carries no lap_number (top-level or driver dict); defaulting to lap 1"
        )
        return 1
    return int(lap)


def _resolve_total_laps(session_meta: Dict[str, Any]) -> int:
    """total_laps from session_meta, else ``DEFAULT_TOTAL_LAPS`` loudly.

    Every internal producer supplies total_laps unconditionally; the one
    reachable miss is arbitrary client JSON at /recommend. The downstream agents
    already fall back to the same ``DEFAULT_TOTAL_LAPS``, so a builder stricter
    than its own consumers would buy a crash, not correctness. The warning keeps
    the gap visible without turning a sloppy-but-working API call into a 500.
    """
    total = session_meta.get("total_laps")
    if total is None:
        logger.warning(
            "session_meta carries no total_laps; falling back to DEFAULT_TOTAL_LAPS=%d",
            DEFAULT_TOTAL_LAPS,
        )
        return DEFAULT_TOTAL_LAPS
    return int(total)


def build_race_state(
    lap_state: Dict[str, Any],
    *,
    driver: Optional[str] = None,
    gap_ahead_s: Optional[float] = None,
    pace_delta_s: Optional[float] = None,
    risk_tolerance: float = 0.5,
    radio_msgs: Optional[List[dict]] = None,
    rcm_events: Optional[List[dict]] = None,
    rival: Optional[str] = None,
):
    """Construct the canonical ``RaceState`` from a raw ``lap_state`` dict.

    The single lap_state -> RaceState mapping shared by the CLI, the Arcade and
    the telemetry backend (#784). The defaults are the per-field canonical
    literals decided in that issue; the module constants above carry each
    measurement.

    ``None`` means "compute it here" for ``gap_ahead_s`` / ``pace_delta_s``. When
    a caller supplies a value it is used unchanged - the /recommend endpoint
    forwards the client's ``gap_ahead_s`` and mcp_tools substitutes its own
    fallback, and both stay byte-compatible. When left as None, the builder
    resolves the positional car ahead once and derives both values from it
    (absorbing the backend simulator's ``_compute_gap_ahead`` copy).

    ``driver`` is the CLI's explicit code override; when None the driver dict's
    own code is used, then "UNK".

    ``rival`` (#431): when set, gap and pace are recomputed against that specific
    car rather than the positional car ahead, with the already-resolved values as
    fallbacks, so a stale selection degrades to positional behaviour rather than
    to a fabricated zero.

    Raises:
        ValueError: when the driver's position is None. A fabricated position is
            exactly the value the car-ahead lookup searches by
            (``position == our_pos - 1``), so an unknown position and a
            genuinely-last car would silently resolve to the same rival - the
            #428 bug shape. All three surfaces converged on failing loud here
            (the CLI under #628, the Arcade and backend under #465); every
            caller's per-lap guard skips these laps first, and its try/except
            turns a breach of that invariant into a surfaced per-lap error, not
            a crashed run.
    """
    from src.agents.strategy_orchestrator import RaceState

    driver_state = lap_state.get("driver", {}) or {}
    weather = lap_state.get("weather", {}) or {}
    session_meta = lap_state.get("session_meta", {}) or {}

    position = driver_state.get("position")
    if position is None:
        raise ValueError(
            "build_race_state: driver position is None for this lap (an incomplete "
            "or out-lap the caller's guard should have skipped); refusing to "
            "fabricate a searchable position (#628, #465)"
        )

    needs_car_ahead = gap_ahead_s is None or pace_delta_s is None
    car_ahead = _car_ahead(lap_state, position) if needs_car_ahead else None
    if gap_ahead_s is None:
        gap_ahead_s = _gap_to_car_ahead(car_ahead)
    if pace_delta_s is None:
        pace_delta_s = _pace_delta_vs_car_ahead(driver_state, car_ahead)

    if rival:
        gap_ahead_s, pace_delta_s = _targeting_against_rival(
            lap_state,
            rival,
            float(gap_ahead_s),
            float(pace_delta_s),
        )

    tyre_life = driver_state.get("tyre_life")
    race_state = RaceState(
        driver=driver or driver_state.get("driver") or "UNK",
        lap=_resolve_lap(lap_state, driver_state),
        total_laps=_resolve_total_laps(session_meta),
        position=position,
        compound=normalise_compound(driver_state.get("compound")),
        tyre_life=tyre_life if tyre_life is not None else UNKNOWN_TYRE_LIFE,
        gap_ahead_s=float(gap_ahead_s),
        pace_delta_s=float(pace_delta_s),
        air_temp=_weather_reading(weather, "air_temp", DEFAULT_AIR_TEMP_C),
        track_temp=_weather_reading(weather, "track_temp", DEFAULT_TRACK_TEMP_C),
        rainfall=bool(weather.get("rainfall", False)),
        radio_msgs=radio_msgs if radio_msgs is not None else [],
        rcm_events=rcm_events if rcm_events is not None else [],
        risk_tolerance=float(risk_tolerance),
    )
    return race_state
