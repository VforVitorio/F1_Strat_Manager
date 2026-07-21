"""Regression tests for the #465 scope-before-build + dead-position-default fixes.

F7 (``src/strategy/inference/engine.py``): ``run_lap``'s ``_build_default_lap_state``
fallback (``lap_state=None``) used to run against the still season-wide frame,
because ``_scope_laps_to_gp`` had no ``gp_name`` to scope by until AFTER the default
was built. A ``race_state``-driven fallback in ``_scope_laps_to_gp`` now derives the
GP from the (driver, lap) row match BEFORE the default lap_state is built, so
``session_meta['gp_name']`` and the frame handed to every agent both refer to the
SAME race instead of two unrelated ones.

F6 (``src/strategy/inference/no_llm.py``, ``src/arcade/strategy.py``): three call
sites used to default a missing ``position`` to a fixed number (20 / 99 / 10) that
the SAME code then searched for (``position - 1``) — a sentinel that can collide
with a real rival's position (the #428 bug shape). All three now propagate the
unknown position instead of inventing one.

No LLM client is constructed by any test here. The scoping test asserts against the
real featured parquet (``data/processed/laps_featured_2025.parquet``); the position
tests use hand-built dicts, per the epic's no-LLM-assertion doctrine.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import pytest

ROOT = Path(__file__).parent.parent.parent
_HAS_MODELS = (ROOT / "data" / "models" / "tire_degradation" / "routing_config.json").exists()
_skip_no_models = pytest.mark.skipif(
    not _HAS_MODELS,
    reason="data/models/ not present (CI runner without model weights)",
)

_PARQUET = ROOT / "data" / "processed" / "laps_featured_2025.parquet"
_skip_no_parquet = pytest.mark.skipif(
    not _PARQUET.exists(),
    reason="laps_featured_2025.parquet not present (data/ not downloaded)",
)


def _pick_target_and_decoy_gp(df: pd.DataFrame) -> tuple[pd.Series, str, str]:
    """Find a (driver, lap) row whose GP is genuinely NOT the frame's first GP.

    Returns ``(candidate_row, target_gp, decoy_gp)``. ``decoy_gp`` is whatever GP
    sorts first in ``df`` — what the pre-#465 code picked via ``laps_df.iloc[0]``,
    regardless of which driver/lap was actually requested. ``target_gp`` is a
    DIFFERENT GP holding a clean (no missing Position/Compound/TyreLife) row for
    some driver+lap — the race the fix must resolve to instead. Skips the test
    (rather than failing) when the parquet only has one GP, since the bug this
    guards cannot be reproduced without a second one.
    """
    decoy_gp = str(df["GP_Name"].iloc[0])
    required_cols = ["Position", "Compound", "TyreLife"]
    for gp_name in df["GP_Name"].unique():
        if gp_name == decoy_gp:
            continue
        clean = df[df["GP_Name"] == gp_name].dropna(subset=required_cols)
        if not clean.empty:
            return clean.iloc[0], str(gp_name), decoy_gp
    pytest.skip("no second GP with a clean row found in the featured parquet")


@_skip_no_models
@_skip_no_parquet
def test_scope_laps_to_gp_resolves_requested_gp_before_default_build():
    """F7: scoping (via ``race_state``) must pick the driver's OWN GP, not whichever
    GP happens to sort first in a multi-GP frame — and the default lap_state built
    from the scoped frame must carry that same GP in ``session_meta``.

    Builds a synthetic ``combined`` frame from two REAL GP slices (decoy rows
    first, target rows second) so the "wrong GP picked first" scenario is
    deterministic instead of hoping the raw parquet happens to exhibit it for
    some guessed driver/lap.
    """
    from src.agents.strategy_orchestrator import RaceState
    from src.strategy.inference.engine import _build_default_lap_state, _scope_laps_to_gp

    df = pd.read_parquet(_PARQUET)
    candidate, target_gp, decoy_gp = _pick_target_and_decoy_gp(df)
    driver = str(candidate["Driver"])
    lap = int(candidate["LapNumber"])

    # Decoy rows: real rows from a DIFFERENT GP, with any accidental (driver, lap)
    # match stripped out so the only match for (driver, lap) in `combined` is the
    # target row below — otherwise the test would not discriminate between the
    # two GPs (both would look like valid "first matches").
    decoy_rows = df[
        (df["GP_Name"] == decoy_gp) & ~((df["Driver"] == driver) & (df["LapNumber"] == lap))
    ]
    target_rows = df[df["GP_Name"] == target_gp]
    # Decoy rows come FIRST, mirroring a season-wide frame where an unrelated GP
    # sorts ahead of the driver's actual race.
    combined = pd.concat([decoy_rows, target_rows], ignore_index=True)
    assert str(combined["GP_Name"].iloc[0]) == decoy_gp  # sanity: decoy really is first

    race_state = RaceState(
        driver=driver,
        lap=lap,
        total_laps=int(target_rows["LapNumber"].max()),
        position=int(candidate["Position"]),
        compound=str(candidate["Compound"]),
        tyre_life=int(candidate["TyreLife"]),
        gap_ahead_s=1.0,
        pace_delta_s=0.0,
        air_temp=25.0,
        track_temp=35.0,
    )

    scoped = _scope_laps_to_gp(combined, None, race_state)
    assert scoped["GP_Name"].nunique() == 1
    assert scoped["GP_Name"].iloc[0] == target_gp, (
        f"scoping picked {scoped['GP_Name'].iloc[0]!r}, expected the driver's own "
        f"GP {target_gp!r} (decoy was {decoy_gp!r})"
    )

    lap_state = _build_default_lap_state(race_state, scoped)
    assert lap_state["session_meta"]["gp_name"] == target_gp

    # Confirm this is a real fix, not a tautology: building the default straight
    # from the UNSCOPED frame (the pre-#465 order) reproduces the bug — gp_name
    # comes from the decoy GP's first row, unrelated to the driver's actual race.
    buggy_lap_state = _build_default_lap_state(race_state, combined)
    assert buggy_lap_state["session_meta"]["gp_name"] == decoy_gp
    assert buggy_lap_state["session_meta"]["gp_name"] != target_gp


@_skip_no_models
def test_situation_no_llm_skips_rival_lookup_when_position_is_none(monkeypatch):
    """F6: a missing ``position`` (RSM's ``None`` convention for an incomplete lap,
    #428) must not default to 20 — a rival sitting at position 19 would otherwise
    be wrongly picked as "the car ahead" purely because ``20 - 1 == 19``.

    Stubs ``run_from_state`` so the test never needs a real ``laps_df`` or GPU/CPU
    model inference — it only inspects which tools ``_situation_no_llm`` queued
    into the null tool-runner BEFORE ``run_from_state`` would have executed them.
    """
    from src.strategy.inference import no_llm

    agent = no_llm._get_situation_agent()
    captured: dict[str, Any] = {}

    def _fake_run_from_state(lap_state, laps_df):
        captured["tool_names"] = {tool.name for tool, _ in agent._react_agent._tool_calls}
        return None

    monkeypatch.setattr(agent, "run_from_state", _fake_run_from_state)

    sit_lap_state = {
        "session_meta": {"driver": "VER", "gp_name": "Lusail", "year": 2025},
        "driver": {"position": None, "compound": "MEDIUM", "tyre_life": 5},
        "lap_number": 10,
        "rivals": [{"driver": "HAM", "position": 19}],
    }

    no_llm._situation_no_llm(sit_lap_state, laps_df=None)

    assert "predict_sc_tool" in captured["tool_names"]
    assert "predict_overtake_tool" not in captured["tool_names"], (
        "position=None must skip the rival lookup, not resolve it via a position=20 default"
    )


@_skip_no_models
def test_build_race_state_fails_loudly_instead_of_defaulting_position():
    """F6 (arcade): ``_build_race_state`` must not fabricate a searchable P10/P99
    car when ``position`` is ``None``.

    The DNF/incomplete-lap guard (``_lap_skip_reason``) is supposed to skip this
    lap before ``_build_race_state`` ever runs; reaching this method with
    ``position=None`` means that invariant broke, and #465 makes it fail loudly
    (caught by the driver loop's ``except Exception``, surfaced as ``state.error``)
    instead of silently building a fake P10 car.
    """
    from src.arcade.strategy import SimConnector, SimulateRequestDTO, StrategyState

    request = SimulateRequestDTO(year=2025, gp="Lusail", driver="VER", team="Red Bull Racing")
    connector = SimConnector(request, StrategyState())

    lap_state = {
        "driver": {"position": None, "lap_time_s": 90.0, "compound": "MEDIUM", "tyre_life": 5},
        "weather": {},
        "session_meta": {"total_laps": 57},
        "rivals": [],
        "lap_number": 10,
    }

    with pytest.raises(ValueError, match="position"):
        connector._build_race_state(lap_state, prev_lap_time=89.0)
