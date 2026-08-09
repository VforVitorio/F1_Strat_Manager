"""#784 — the canonical RaceState builder replaces three drifted per-surface copies.

Two tiers, split by what each test needs to import. The pure-helper tests and the
leaf-module guard import only ``src.agents.race_state_builder`` (cheap by
contract) and run everywhere, including CI without ``data/models`` and without the
``src/telemetry`` submodule. The tests that CALL ``build_race_state`` construct a
real ``RaceState``, whose lazy import pulls in ``strategy_orchestrator`` and
therefore model artefacts, so they carry the same models-present guard as the
sibling agent tests.

Every canonical literal asserted here is asserted against the module's own
constant, never a restated number, so a deliberate future change to a default
fails exactly one definition away from this file.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from src.agents._shared_defaults import DEFAULT_TOTAL_LAPS
from src.agents.position_projection import GAP_UNKNOWN_FALLBACK_S
from src.agents.race_state_builder import (
    DEFAULT_AIR_TEMP_C,
    DEFAULT_TRACK_TEMP_C,
    UNKNOWN_COMPOUND,
    UNKNOWN_TYRE_LIFE,
    build_race_state,
    normalise_compound,
)

ROOT = Path(__file__).parent.parent.parent
_HAS_MODELS = (ROOT / "data" / "models" / "lap_time").is_dir()

needs_models = pytest.mark.skipif(
    not _HAS_MODELS, reason="building a RaceState imports the agents, which load model artefacts"
)


def _lap_state(**overrides):
    """A minimal-but-complete lap_state shaped like RaceStateManager emits it.

    Our driver is P3; VER sits one position ahead with a measured interval and a
    same-lap time, so both the gap and the #750 pace delta have known expected
    values (gap 1.8, pace 92.5 - 92.1 = +0.4). Overrides replace TOP-LEVEL keys.
    """
    state = {
        "lap_number": 30,
        "driver": {
            "driver": "NOR",
            "lap_number": 30,
            "position": 3,
            "compound": "MEDIUM",
            "tyre_life": 8,
            "lap_time_s": 92.5,
        },
        "rivals": [
            {"driver": "VER", "position": 2, "interval_to_driver_s": -1.8, "lap_time_s": 92.1},
            {"driver": "LEC", "position": 4, "interval_to_driver_s": 2.4, "lap_time_s": 93.0},
        ],
        "weather": {"air_temp": 22.0, "track_temp": 31.5, "rainfall": False},
        "session_meta": {"total_laps": 57},
    }
    state.update(overrides)
    return state


# ---------------------------------------------------------------------------
# Tier 1 — pure helpers and the leaf guarantee (no models, no submodule)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("raw", ["", "nan", "None", "NaN", "NONE", "  ", None])
def test_missing_compound_spellings_normalise_to_unknown(raw):
    """Every spelling of "no compound" folds into the one honest marker.

    race_state_manager.py emits `str(r.get("Compound", ""))`, so a stored NaN
    reaches the builder as the STRING "nan" with the key present — a dict.get
    default never fires on it. This normalisation is the substantive half of the
    "UNKNOWN" decision; without it the canonical default is a choice about an
    almost-unreachable branch.
    """
    assert normalise_compound(raw) == UNKNOWN_COMPOUND


@pytest.mark.parametrize("real", ["SOFT", "MEDIUM", "HARD", "INTERMEDIATE", "WET"])
def test_real_compound_passes_through_untouched(real):
    assert normalise_compound(real) == real


def test_module_import_is_a_leaf_and_never_drags_langchain():
    """The guard that keeps every surface's boot cheap.

    ``build_race_state`` imports ``RaceState`` lazily because
    ``strategy_orchestrator`` drags LangChain/LangGraph and model artefacts. A
    future editor hoisting that import to module level would slow the CLI, the
    arcade and the backend at once — this subprocess check (a fresh interpreter,
    so no module already cached by other tests can mask the leak) fails the
    moment that happens.
    """
    probe = (
        "import sys; import src.agents.race_state_builder; "
        "leaked = sorted(m for m in sys.modules "
        "if m.startswith('langchain') or m.startswith('langgraph')); "
        "assert not leaked, f'leaf module dragged: {leaked}'"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


# ---------------------------------------------------------------------------
# Tier 2 — the canonical field behaviour, end to end (needs model artefacts)
# ---------------------------------------------------------------------------


@needs_models
def test_canonical_defaults_fire_when_keys_are_absent(caplog):
    """Each approved default fires on a lap_state that omits its key."""
    state = _lap_state(
        driver={"position": 1, "lap_time_s": 92.5},
        rivals=[],
        weather={},
        session_meta={},
    )
    del state["lap_number"]

    with caplog.at_level("WARNING", logger="src.agents.race_state_builder"):
        rs = build_race_state(state)

    assert rs.driver == "UNK"
    assert rs.lap == 1
    assert rs.total_laps == DEFAULT_TOTAL_LAPS
    assert rs.compound == UNKNOWN_COMPOUND
    assert rs.tyre_life == UNKNOWN_TYRE_LIFE
    assert rs.air_temp == DEFAULT_AIR_TEMP_C
    assert rs.track_temp == DEFAULT_TRACK_TEMP_C
    assert rs.rainfall is False
    assert rs.radio_msgs == []
    assert rs.rcm_events == []
    assert rs.risk_tolerance == 0.5
    # Both silent-fallback holes must be loud, not quiet (#784 decision table).
    warnings = " ".join(r.getMessage() for r in caplog.records)
    assert "total_laps" in warnings
    assert "lap_number" in warnings


@needs_models
def test_explicit_driver_param_wins_over_the_driver_dict():
    rs = build_race_state(_lap_state(), driver="PIA")
    assert rs.driver == "PIA"


@needs_models
def test_lap_falls_back_to_the_driver_dict_before_defaulting():
    """The middle rung of the lap ladder: top-level absent, driver dict present."""
    state = _lap_state()
    del state["lap_number"]
    rs = build_race_state(state)
    assert rs.lap == 30


@needs_models
def test_compound_nan_string_is_normalised_end_to_end():
    """The live RSM path: key PRESENT with the literal string "nan"."""
    state = _lap_state()
    state["driver"]["compound"] = "nan"
    rs = build_race_state(state)
    assert rs.compound == UNKNOWN_COMPOUND


@needs_models
def test_present_but_none_weather_lands_on_defaults_without_crashing():
    """A NaN weather row emits keys whose VALUE is None (race_state_manager.py).

    All three pre-#784 builders crashed here, though not by the same mechanism: the
    arcade and backend copies wrapped the read in float(), so they raised TypeError,
    while the CLI passed the None straight to Pydantic, which rejected it; the canonical
    builder treats present-but-None as missing, identical behaviour on every lap
    that worked before.
    """
    state = _lap_state(weather={"air_temp": None, "track_temp": None, "rainfall": None})
    rs = build_race_state(state)
    assert rs.air_temp == DEFAULT_AIR_TEMP_C
    assert rs.track_temp == DEFAULT_TRACK_TEMP_C
    assert rs.rainfall is False


@needs_models
def test_position_none_raises_value_error():
    """The one converged fail-loud guard: never fabricate a searchable position."""
    state = _lap_state()
    state["driver"]["position"] = None
    with pytest.raises(ValueError, match="position is None"):
        build_race_state(state)


@needs_models
def test_gap_is_none_when_there_is_no_car_ahead():
    """P1 with no rival at position 0: leading, and the gap is an ABSENCE.

    This test used to assert `== 0.0` and its docstring called that
    "honest" (#878). It was the green test defending the defect: 0.0 is
    also what two cars side by side measure, and a falsy `or` downstream
    turned it into a fabricated 2.0 rival for the race leader.
    """
    state = _lap_state()
    state["driver"]["position"] = 1
    rs = build_race_state(state)
    assert rs.gap_ahead_s is None


@needs_models
def test_gap_falls_back_when_the_interval_is_unmeasured():
    """A car ahead with interval None is NOT a zero gap (#633)."""
    state = _lap_state(
        rivals=[{"driver": "VER", "position": 2, "interval_to_driver_s": None}],
    )
    rs = build_race_state(state)
    assert rs.gap_ahead_s == GAP_UNKNOWN_FALLBACK_S


@needs_models
def test_gap_is_the_absolute_measured_interval():
    rs = build_race_state(_lap_state())
    assert rs.gap_ahead_s == pytest.approx(1.8)


@needs_models
def test_pace_delta_is_rival_relative_same_lap():
    """The #750 contract: our lap time minus the car ahead's SAME-lap time."""
    rs = build_race_state(_lap_state())
    assert rs.pace_delta_s == pytest.approx(92.5 - 92.1)


@needs_models
def test_pace_delta_is_neutral_zero_when_the_ahead_lap_time_is_unknown():
    state = _lap_state(
        rivals=[{"driver": "VER", "position": 2, "interval_to_driver_s": -1.8}],
    )
    rs = build_race_state(state)
    assert rs.pace_delta_s == 0.0


@needs_models
def test_pace_delta_is_neutral_zero_when_our_lap_time_is_unknown():
    state = _lap_state()
    state["driver"]["lap_time_s"] = None
    rs = build_race_state(state)
    assert rs.pace_delta_s == 0.0


@needs_models
def test_explicit_gap_and_pace_override_the_internal_computation():
    """None-means-compute is what keeps the backend call sites byte-compatible.

    /recommend forwards the client's gap_ahead_s and mcp_tools substitutes its
    own fallback; both must land unchanged even when rivals are present and the
    internal computation would produce different numbers (1.8 / +0.4 here).
    """
    rs = build_race_state(_lap_state(), gap_ahead_s=7.7, pace_delta_s=-0.3)
    assert rs.gap_ahead_s == 7.7
    assert rs.pace_delta_s == -0.3


@needs_models
def test_rival_targeting_recomputes_both_values():
    """#431: gap and pace framed around the chosen rival, not the car ahead."""
    rs = build_race_state(_lap_state(), rival="LEC")
    assert rs.gap_ahead_s == pytest.approx(2.4)
    assert rs.pace_delta_s == pytest.approx(92.5 - 93.0)


@needs_models
def test_rival_absent_from_the_lap_falls_back_to_positional_values():
    """A stale rival selection degrades to positional behaviour, never to zero."""
    rs = build_race_state(_lap_state(), rival="XXX")
    assert rs.gap_ahead_s == pytest.approx(1.8)
    assert rs.pace_delta_s == pytest.approx(92.5 - 92.1)


@needs_models
def test_radio_and_rcm_parameters_land_on_the_state():
    """The parameter shape is the surface-neutral one (F3 in the design gate)."""
    radios = [{"lap": 30, "message": "box box"}]
    rcms = [{"lap": 30, "category": "SafetyCar"}]
    rs = build_race_state(_lap_state(), radio_msgs=radios, rcm_events=rcms)
    assert rs.radio_msgs == radios
    assert rs.rcm_events == rcms


# --- No car ahead is an ABSENCE, not a zero (#878) ---------------------------
#
# These sit in the pure-helper tier on purpose. The integration tier needs the
# real `RaceState`, whose lazy import reaches `race_situation_agent` and a
# parquet the curated download does not ship - so those tests SKIP on a clean
# checkout and in CI alike. A defect worth 2,315 laps of the served season
# cannot be guarded by a test that never runs, so the semantics are asserted
# here, where they execute everywhere.


def test_no_car_ahead_is_none_and_a_measured_zero_survives():
    """The three claims the old 0.0 collapsed into one number.

    `_gap_to_car_ahead` returned 0.0 for "nobody ahead" and called it honest.
    It is not: 0.0 is also what two cars side by side measure - the served
    2025 season has four such laps - and a falsy `or` downstream turned the
    leader's 0.0 into a fabricated 2.0 rival on 1,262 laps. Absent, measured
    and unmeasured are three different claims and must look different.
    """
    from src.agents.race_state_builder import _gap_to_car_ahead

    assert _gap_to_car_ahead(None) is None, "nobody ahead is an absence"
    assert _gap_to_car_ahead({"driver": "VER", "interval_to_driver_s": 0.0}) == 0.0, (
        "a measured zero is a measurement and must survive"
    )
    assert _gap_to_car_ahead({"driver": "VER", "interval_to_driver_s": -1.42}) == 1.42
    assert _gap_to_car_ahead({"driver": "VER"}) == GAP_UNKNOWN_FALLBACK_S, (
        "a car that IS there with no interval is a different claim, unchanged"
    )


def test_rival_targeting_survives_a_none_positional_fallback():
    """The coercion twin that nearly shipped inside this fix.

    With no car ahead the positional fallback is now None, and the old body
    did `round(gap_ahead_s, 3)` unconditionally on the way out - `round(None)`
    raises. The call site coerced with `float(gap_ahead_s)`, which raises too.
    Both had to move; the design gate's own first draft moved one.
    """
    from src.agents.race_state_builder import _targeting_against_rival

    lap_state = {
        "driver": {"lap_time_s": 81.4},
        "rivals": [
            {"driver": "VER", "position": 2, "interval_to_driver_s": -1.42, "lap_time_s": 81.6},
            {"driver": "RUS", "position": 4, "lap_time_s": 81.9},
        ],
    }

    assert _targeting_against_rival(lap_state, "VER", None, 0.0)[0] == 1.42, "measured wins"
    assert _targeting_against_rival(lap_state, "RUS", None, 0.0)[0] is None, "no interval, no gap"
    assert _targeting_against_rival(lap_state, "HAM", None, 0.0)[0] is None, "rival not on track"
    assert _targeting_against_rival(lap_state, "HAM", 2.0, 0.0)[0] == 2.0, "a float still passes"


def test_the_prompt_says_leading_instead_of_printing_a_gap_that_is_not_there():
    """What the LLM actually reads, which is the only live consumer.

    Exactly three places in the repo read `race_state.gap_ahead_s`: this
    prompt line and two wire boundaries. No model consumes it - N27 computes
    its own pair gap from laps_df - so this string IS the behaviour, and it
    used to say "Gap ahead: 0.00s" to a driver leading the race.

    Extracted with `ast` rather than imported: the module reaches
    `race_situation_agent` at import time and a parquet the curated download
    omits, which is exactly how this surface stayed untested.
    """
    import ast

    source = (
        Path(__file__).resolve().parents[2] / "src" / "agents" / "strategy_orchestrator.py"
    ).read_text(encoding="utf-8")
    node = next(
        n
        for n in ast.parse(source).body
        if isinstance(n, ast.FunctionDef) and n.name == "_gap_ahead_context_line"
    )
    node.args.args[0].annotation = None
    node.returns = None
    namespace: dict = {}
    exec(
        compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), "<t>", "exec"),
        namespace,
    )
    render = namespace["_gap_ahead_context_line"]

    class _State:
        def __init__(self, gap, position):
            self.gap_ahead_s, self.position = gap, position

    assert render(_State(1.42, 4)) == "Gap ahead: 1.42s"
    assert render(_State(0.0, 7)) == "Gap ahead: 0.00s", "a measured zero is still printed"
    assert "LEADING" in render(_State(None, 1)), "the leader is told they are leading"
    assert "0.00" not in render(_State(None, 1)) and "2.00" not in render(_State(None, 1))
    assert "no car classified" in render(_State(None, 5)), "P5 with a gap-less pos-4 car"
