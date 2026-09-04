"""Real-data checks on ``RaceStateManager`` over the season parquets.

Both tests read real parquets and skip when one is absent, so a clone without
``data/`` stays green.

Contents:
- ``_data_root``: the anchor both tests resolve their paths against.
- ``_skip_unless_present``: the skip guard, which names the file that is missing.
- ``test_race_state_manager_melbourne``: the lap_state golden check on Melbourne 2025.
- ``test_qatar_2025_v7_pia_sc_override``: the RCM safety-car override regression.
"""

from pathlib import Path

import pytest

# Both tests read parquets that ship from Hugging Face rather than git, which is
# what the marker records. Applied at module level so the two cannot drift apart.
pytestmark = pytest.mark.data


def _data_root() -> Path:
    """The tree the production code reads, which ``F1_STRAT_DATA_ROOT`` can move.

    Returns:
        The resolved data directory, so a probe here and a read in the code under
        test always name the same tree.

    Anchoring on the repo instead lets a probe HIT while the code MISSES, which
    turns a skip into a hard failure. Called inside a test rather than at module
    scope, so collection never runs this function's mkdir.
    """
    from src.f1_strat_manager.data_cache import get_data_root

    return get_data_root()


def _skip_unless_present(*paths: Path) -> None:
    """Skip on the first missing path, naming that path.

    Args:
        paths: Files the caller reads, in the order they are read.

    The guard this replaces and-ed two paths into a single message asserting that
    neither was available. One of the two resolved, so the message was false and a
    path that matched nothing on any machine stayed invisible for months (#1172).
    A skip reason that does not name the file cannot be told from a real absence.
    """
    for path in paths:
        if not path.exists():
            pytest.skip(f"not on disk: {path}")


def test_race_state_manager_melbourne():
    """RaceStateManager produces a valid lap_state from the Melbourne 2025 parquet."""
    pd = pytest.importorskip("pandas", reason="pandas not installed in this environment")
    from src.simulation.race_state_manager import RaceStateManager

    laps_path = _data_root() / "raw" / "2025" / "Melbourne" / "laps.parquet"
    _skip_unless_present(laps_path)

    laps = pd.read_parquet(laps_path)
    rsm = RaceStateManager(laps, "NOR", "McLaren", gp_name="Melbourne", year=2025)

    assert rsm.total_laps == 57
    state = rsm.get_lap_state(20)

    assert state["lap_number"] == 20
    assert state["driver"]["driver"] == "NOR"
    assert state["driver"]["position"] == 1
    assert len(state["rivals"]) > 0
    assert "gp_name" in state["session_meta"]


def test_qatar_2025_v7_pia_sc_override(monkeypatch):
    """Catar 2025 V7 (PIA, McLaren) — RCM override flips sc_prob_3lap to 1.0.

    Reproduces the McLaren strategic miss from the real race (a deployed SC at V7
    that the LightGBM model predicted with low probability).  The override should
    flag sc_currently_active=True and elevate threat_level to HIGH regardless of
    what the model returned for raw sc_prob.

    Importing the agent builds the module-level ``CFG``, which loads both LightGBM
    models, both calibrators and two processed parquets, so those are probed too:
    without them the import raises where a skip is the honest answer.

    The ReAct round trip is stubbed and nothing else is. ``_run_core`` computes the
    override before it constructs the output, and ``threat_level`` is derived inside
    that constructor, so the three assertions below read the forced values and never
    the model's. What still runs for real is everything they depend on: the Lusail
    parquet through ``RaceStateManager``, the Qatar RCM row through
    ``_to_rcm_event``, the event classifier, ``_neutralization_from_rcm``, the
    override arithmetic and the band. Left live, the test instead needs a backend on
    localhost:1234, fails in 45 s on any machine without one, and would be the first
    test here to reach a real LLM client, so a shell exporting F1_LLM_PROVIDER=openai
    would spend credits on every run of the suite.
    """
    pd = pytest.importorskip("pandas", reason="pandas not installed")
    pytest.importorskip("langchain_openai", reason="needs LLM stack")

    from src.f1_strat_manager.gp_slugs import resolve_gp_slug

    root = _data_root()
    laps_path = root / "raw" / "2025" / "Lusail" / "laps.parquet"
    # The corpus is keyed by country slug, so "Lusail" resolves to "qatar" through
    # the same helper radio_runner builds this path with. Writing "qatar" here would
    # be a second copy of that mapping.
    rcm_path = (
        root / "processed" / "race_radios" / "2025" / resolve_gp_slug("Lusail") / "rcm.parquet"
    )
    _skip_unless_present(
        laps_path,
        rcm_path,
        root / "models" / "overtake_probability" / "model_config.json",
        root / "models" / "safety_car_probability" / "feature_list_v1.json",
        root / "processed" / "circuit_clustering" / "circuit_clusters_k4.parquet",
        root / "processed" / "sc_labeled" / "sc_labeled_2023_2025.parquet",
    )

    from langchain_core.messages import AIMessage

    from src.agents.race_situation_agent import (
        RaceSituationAgent,
        run_race_situation_agent_from_state,
    )
    from src.agents.strategy_orchestrator import _to_rcm_event
    from src.simulation.race_state_manager import RaceStateManager

    class _StubReactAgent:
        """A compiled graph that answers without calling a model.

        Returns:
            The one shape ``_run_core`` reads back: a ``messages`` list whose last
            entry carries ``.content``. Holding no ToolMessage is deliberate, since
            ``_parse_tool_outputs`` then returns its no-answer defaults
            (``overtake_prob=None``, ``sc_prob_3lap=0.0``) and the override has to
            supply every value the assertions check.
        """

        def invoke(self, _payload: dict) -> dict:
            return {"messages": [AIMessage(content="stubbed: no tool calls")]}

    monkeypatch.setattr(
        RaceSituationAgent, "get_react_agent", lambda self, *args, **kwargs: _StubReactAgent()
    )

    laps = pd.read_parquet(laps_path)
    rcms = pd.read_parquet(rcm_path)
    sc_lap7 = rcms[(rcms["lap_number"] == 7) & (rcms["category"] == "SafetyCar")].to_dict("records")
    assert sc_lap7, "fixture must contain a SafetyCar RCM at lap 7"

    rsm = RaceStateManager(laps, "PIA", "McLaren", gp_name="Lusail", year=2025)
    lap_state = rsm.get_lap_state(7)
    # The corpus column is `lap_number`; _to_rcm_event reads `lap`, so a raw row
    # would build an RCMEvent carrying lap 0. The production producers already map
    # it (src/nlp/radio_runner.py:640,668), and this is the only caller handing
    # _to_rcm_event a row straight off the parquet.
    lap_state["rcm_events"] = [_to_rcm_event({**e, "lap": int(e["lap_number"])}) for e in sc_lap7]

    out = run_race_situation_agent_from_state(lap_state, laps)
    assert out.sc_currently_active is True
    assert out.sc_prob_3lap == 1.0
    assert out.threat_level == "HIGH"
