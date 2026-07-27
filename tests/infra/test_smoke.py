"""Smoke tests — verify CI infrastructure and project structure."""

from pathlib import Path

ROOT = Path(__file__).parent.parent.parent


def test_src_structure():
    """src/ directory exists with expected sub-packages."""
    assert (ROOT / "src").is_dir()
    assert (ROOT / "src" / "rag").is_dir()
    assert (ROOT / "src" / "nlp").is_dir()
    assert (ROOT / "src" / "agents").is_dir()
    assert (ROOT / "src" / "simulation").is_dir()


def test_notebooks_structure():
    """Agent notebooks directory exists."""
    assert (ROOT / "notebooks" / "agents").is_dir()


def test_pyproject_exists():
    """pyproject.toml is present at repo root."""
    assert (ROOT / "pyproject.toml").is_file()


def test_simulation_imports():
    """RaceStateManager and RaceReplayEngine are importable."""
    pytest = __import__("pytest")
    pd = pytest.importorskip("pandas", reason="pandas not installed in this environment")  # noqa: F841
    from src.simulation.race_state_manager import RaceStateManager  # noqa: F401
    from src.simulation.replay_engine import RaceReplayEngine  # noqa: F401


def test_race_state_manager_melbourne():
    """RaceStateManager produces a valid lap_state from the Melbourne 2025 parquet."""
    import pytest

    pd = pytest.importorskip("pandas", reason="pandas not installed in this environment")
    from src.simulation.race_state_manager import RaceStateManager

    laps_path = ROOT / "data" / "raw" / "2025" / "Melbourne" / "laps.parquet"
    if not laps_path.exists():
        pytest.skip("Melbourne 2025 parquet not available in this environment")

    laps = pd.read_parquet(laps_path)
    rsm = RaceStateManager(laps, "NOR", "McLaren", gp_name="Melbourne", year=2025)

    assert rsm.total_laps == 57
    state = rsm.get_lap_state(20)

    assert state["lap_number"] == 20
    assert state["driver"]["driver"] == "NOR"
    assert state["driver"]["position"] == 1
    assert len(state["rivals"]) > 0
    assert "gp_name" in state["session_meta"]


def test_qatar_2025_v7_pia_sc_override():
    """Catar 2025 V7 (PIA, McLaren) — RCM override flips sc_prob_3lap to 1.0.

    Reproduces the McLaren strategic miss from the real race (a deployed SC at V7
    that the LightGBM model predicted with low probability).  The override should
    flag sc_currently_active=True and elevate threat_level to HIGH regardless of
    what the model returned for raw sc_prob.
    """
    import pytest

    pd = pytest.importorskip("pandas", reason="pandas not installed")
    pytest.importorskip("langchain_openai", reason="needs LLM stack")

    laps_path = ROOT / "data" / "raw" / "2025" / "Lusail" / "laps.parquet"
    rcm_path = ROOT / "data" / "raw" / "2025" / "Lusail" / "rcm.parquet"
    if not (laps_path.exists() and rcm_path.exists()):
        pytest.skip("Qatar 2025 parquets not available in this environment")

    from src.agents.race_situation_agent import run_race_situation_agent_from_state
    from src.agents.strategy_orchestrator import _to_rcm_event
    from src.simulation.race_state_manager import RaceStateManager

    laps = pd.read_parquet(laps_path)
    rcms = pd.read_parquet(rcm_path)
    sc_lap7 = rcms[(rcms["lap_number"] == 7) & (rcms["category"] == "SafetyCar")].to_dict("records")
    assert sc_lap7, "fixture must contain a SafetyCar RCM at lap 7"

    rsm = RaceStateManager(laps, "PIA", "McLaren", gp_name="Lusail", year=2025)
    lap_state = rsm.get_lap_state(7)
    lap_state["rcm_events"] = [_to_rcm_event(e) for e in sc_lap7]

    out = run_race_situation_agent_from_state(lap_state, laps)
    assert out.sc_currently_active is True
    assert out.sc_prob_3lap == 1.0
    assert out.threat_level == "HIGH"
