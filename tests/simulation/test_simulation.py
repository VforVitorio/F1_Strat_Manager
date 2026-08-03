"""
Tests for the simulation service and its SSE endpoint.

Two kinds of coverage live here:

1. Unit test against ``simulate_race`` directly — confirms the generator emits
   the documented frame order (``start`` first, at least one ``lap``, one
   ``summary`` last) when driven with a ``SimConfig(no_llm=True)`` config on a
   small lap window. The no-LLM path is chosen deliberately so that LM Studio
   / OpenAI do not have to be running in CI: sub-agent stubs kick in whenever
   an ``APIConnectionError`` surfaces, keeping the run deterministic.

2. Integration test against the ``/api/v1/strategy/simulate`` endpoint using
   ``fastapi.testclient.TestClient.stream`` — the exact pattern the manual
   smoke test used. The mini-app only mounts ``strategy.router`` so we avoid
   pulling ``backend.main`` (which imports FastMCP, Supabase, etc.) and the
   test stays hermetic.

Both tests skip cleanly when either the featured parquet or the race
directory for ``2025/Melbourne`` is missing, so contributors without the
full data set still get a green suite.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from tests.conftest import skip_no_tire_models as _skip_no_models

ROOT = Path(__file__).parent.parent.parent

# Guard: simulation service needs laps_featured_YYYY.parquet + raw race dir.
_PARQUET = ROOT / "data" / "processed" / "laps_featured_2025.parquet"
_RACE_DIR = ROOT / "data" / "raw" / "2025" / "Melbourne"
_HAS_DATA = _PARQUET.exists() and _RACE_DIR.exists()
_skip_no_data = pytest.mark.skipif(
    not _HAS_DATA,
    reason="Melbourne 2025 parquet + race dir required for simulation tests",
)

# Guard: the telemetry backend must be importable — the simulation service
# lives inside ``src/telemetry/backend`` and we need ``backend.*`` on sys.path
# to import it without starting the full FastAPI app.
_BACKEND_ROOT = ROOT / "src" / "telemetry"
_HAS_BACKEND = (_BACKEND_ROOT / "backend").is_dir()
_skip_no_backend = pytest.mark.skipif(
    not _HAS_BACKEND,
    reason="src/telemetry/backend not present in this checkout",
)

# Importing `simulator` is not free: it pulls the agent modules, which read their
# routing config at IMPORT time. So a test that needs nothing but the module still
# needs the weights on disk, and `_skip_no_backend` alone is not enough — that
# combination is what made the first version of the payload tests fail on CI while
# passing locally.


def _ensure_backend_on_path() -> None:
    """Insert ``src/telemetry`` at the front of ``sys.path``.

    The simulation service and the strategy router both import via ``backend.*``
    absolute paths; this mirrors what ``conftest.py`` inside the submodule does
    for its own suite. Safe to call multiple times — we guard against duplicate
    insertions so pytest's module discovery stays stable across tests.
    """
    path_str = str(_BACKEND_ROOT)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)


# ---------------------------------------------------------------------------
# Unit — simulate_race generator contract
# ---------------------------------------------------------------------------


@_skip_no_data
@_skip_no_backend
def test_simulate_race_emits_start_lap_summary():
    """Generator must yield ``start`` first, at least one ``lap``, then ``summary``.

    Uses a 3-lap window (laps 5..7) on Melbourne 2025 with ``no_llm=True`` so
    no LLM backend is required. Validates the exact ordering contract the
    SSE endpoint relies on: consumers (Arcade, curl probes) cannot lock the
    layout until ``start`` arrives, and cannot finalise stats until
    ``summary`` arrives. The happy path must emit ZERO ``error`` frames (this
    guards the ``--no-llm`` class of bug, #166) and close with the summary.
    """
    _ensure_backend_on_path()
    from backend.services.simulation import SimConfig, simulate_race

    config = SimConfig(
        year=2025,
        gp="Melbourne",
        driver="NOR",
        team="McLaren",
        lap_range=(5, 7),
        no_llm=True,
        interval_s=0.0,
    )

    events = list(simulate_race(config))

    assert events, "simulate_race yielded nothing"
    assert events[0]["type"] == "start", f"first event should be start, got {events[0]['type']}"
    assert events[-1]["type"] == "summary", (
        f"last event should be summary, got {events[-1]['type']}"
    )

    lap_events = [e for e in events if e["type"] == "lap"]
    assert len(lap_events) >= 1, "expected at least one lap event in 3-lap window"

    # The no-LLM happy path must not degrade to error frames. Tightened from the
    # old "errors tolerated" contract: a per-lap crash here is exactly the
    # --no-llm class of bug (#166) this test now guards against.
    error_events = [e for e in events if e["type"] == "error"]
    assert not error_events, (
        f"no-LLM happy path emitted {len(error_events)} error frame(s) "
        f"(--no-llm class, #166): {error_events[:2]}"
    )

    # Spot-check the LapDecision schema on the first lap payload.
    first_lap = lap_events[0]["data"]
    assert "lap_number" in first_lap
    assert "action" in first_lap
    assert "scenario_scores" in first_lap
    assert isinstance(first_lap["scenario_scores"], dict)

    # The memory fields must be on the wire even on this branch, so the webapp can
    # rely on their presence rather than probing for them. no-llm builds no prompt,
    # so there is no block: the values are the explicit "no memory here" pair, not
    # missing keys.
    assert "memory_block" in first_lap, (
        "LapDecision must always carry memory_block; the webapp reads it to explain "
        "a changed recommendation and cannot branch on a key that sometimes exists"
    )
    assert first_lap["memory_block"] is None, "the no-llm branch builds no prompt to hold a block"
    assert first_lap["plan_changed"] is False


# ---------------------------------------------------------------------------
# Integration — /api/v1/strategy/simulate SSE endpoint
# ---------------------------------------------------------------------------


@_skip_no_data
@_skip_no_backend
def test_simulate_endpoint_streams_sse_frames():
    """POST /api/v1/strategy/simulate must return an SSE stream with 4+ frames.

    Mounts only ``strategy.router`` on a bare FastAPI app so the test does not
    pull ``backend.main`` (which imports FastMCP, Supabase, voice stack, etc.).
    This mirrors the pattern used by the manual smoke test and keeps the
    integration cost low enough for CI.

    We assert:
      * HTTP 200 on the streaming response,
      * at least 4 SSE ``data:`` frames (start + >=2 laps + summary),
      * the first ``data:`` frame parses as JSON and declares ``type=start``.
    """
    import json

    _ensure_backend_on_path()
    from backend.api.v1.endpoints import strategy
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    app = FastAPI()
    app.include_router(strategy.router, prefix="/api/v1")

    payload = {
        "year": 2025,
        "gp": "Melbourne",
        "driver": "NOR",
        "team": "McLaren",
        "lap_range": [5, 7],
        "no_llm": True,
        "interval_s": 0.0,
    }

    with TestClient(app) as client:
        with client.stream("POST", "/api/v1/strategy/simulate", json=payload) as response:
            assert response.status_code == 200, response.text
            assert "text/event-stream" in response.headers.get("content-type", "")

            data_frames: list[str] = []
            for line in response.iter_lines():
                if line and line.startswith("data:"):
                    data_frames.append(line[len("data:") :].strip())

    assert len(data_frames) >= 4, (
        f"expected >=4 SSE data frames (start + laps + summary), got {len(data_frames)}"
    )

    first = json.loads(data_frames[0])
    assert first.get("type") == "start"
    last = json.loads(data_frames[-1])
    assert last.get("type") == "summary"


# ---------------------------------------------------------------------------
# Unit — SimulateRequest Pydantic validation
# ---------------------------------------------------------------------------


@_skip_no_backend
def test_simulate_request_schema_defaults():
    """SimulateRequest must default to ``no_llm=False`` with lmstudio provider.

    Protects the public API contract documented in ``project_sim_sse_endpoint_plan.md``:
    callers posting the minimum payload (year + gp + driver + team) must get a
    well-formed request with the expected defaults — ``risk_tolerance=0.5``,
    ``provider="lmstudio"``, ``interval_s=0.0``. Changing any default here is a
    breaking change for Arcade and the manual curl probes.
    """
    _ensure_backend_on_path()
    from backend.api.v1.endpoints.strategy import SimulateRequest

    req = SimulateRequest(year=2025, gp="Melbourne", driver="NOR", team="McLaren")
    assert req.no_llm is False
    assert req.provider == "lmstudio"
    assert req.risk_tolerance == 0.5
    assert req.interval_s == 0.0
    assert req.lap_range is None


@_skip_no_backend
def test_simulate_request_rejects_invalid_provider():
    """Provider must match the ``^(lmstudio|openai)$`` pattern.

    The orchestrator reads ``F1_LLM_PROVIDER`` to pick the LLM client; allowing
    arbitrary strings here would silently fall back to the default and make
    provider bugs hard to diagnose downstream. Pydantic's pattern validator is
    our first line of defence — this test pins the behaviour.
    """
    _ensure_backend_on_path()
    from backend.api.v1.endpoints.strategy import SimulateRequest
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        SimulateRequest(
            year=2025, gp="Melbourne", driver="NOR", team="McLaren", provider="anthropic"
        )


@_skip_no_backend
@_skip_no_models
def test_the_memory_block_reaches_the_lap_payload():
    """The block the orchestrator was shown must survive into the wire format.

    This is the whole point of #694: the memory layer changes decisions and leaves
    no trace in `reasoning`, and asking the model to narrate it was measured and
    made the decisions worse. So the explanation has to be the deterministic INPUT,
    which means it has to reach the webapp intact.

    Exercised through `_parse_lap_decision` rather than a live `rich` simulation
    because a real run needs an LLM provider; what is under test here is the
    transport, and the layer below it was verified on a real `f1-sim` run.
    """
    _ensure_backend_on_path()
    from types import SimpleNamespace

    from backend.services.simulation.simulator import _parse_lap_decision

    block = "DECISION MEMORY (your own previous calls this race):\n  Last call: STAY_OUT.\n"
    race_state = SimpleNamespace(
        lap=42, compound="MEDIUM", tyre_life=20, position=3, gap_ahead_s=1.8
    )
    result = SimpleNamespace(
        action="PIT_NOW",
        confidence=0.9,
        reasoning="stub",
        scenario_scores={},
        pace_mode=None,
        risk_posture=None,
        pit_lap_target=42,
        compound_next="HARD",
        undercut_target=None,
    )

    decision = _parse_lap_decision(
        result, race_state, {}, 90.0, memory_block=block, plan_changed=True
    )
    payload = decision.model_dump()

    assert payload["memory_block"] == block
    assert payload["plan_changed"] is True


@_skip_no_backend
@_skip_no_models
def test_a_lap_with_no_memory_still_carries_both_fields():
    """Absent memory is an explicit pair of values, never missing keys.

    The webapp has to render "no history for this call" rather than branch on a
    key that sometimes exists. `/recommend` and the MCP tool are memoryless by
    design, so that state is permanent, not transitional.
    """
    _ensure_backend_on_path()
    from types import SimpleNamespace

    from backend.services.simulation.simulator import _parse_lap_decision

    race_state = SimpleNamespace(lap=1, compound="SOFT", tyre_life=1, position=5, gap_ahead_s=0.0)
    result = SimpleNamespace(
        action="STAY_OUT",
        confidence=0.5,
        reasoning="stub",
        scenario_scores={},
        pace_mode=None,
        risk_posture=None,
        pit_lap_target=None,
        compound_next=None,
        undercut_target=None,
    )

    payload = _parse_lap_decision(result, race_state, {}, None).model_dump()

    assert payload["memory_block"] is None
    assert payload["plan_changed"] is False
