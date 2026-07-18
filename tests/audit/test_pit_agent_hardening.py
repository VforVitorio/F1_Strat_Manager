"""Hardening tests for the Pit Strategy Agent — #432, #450, #465 (F6), #476.

No LLM calls, no ReAct invocation, no orchestrator run. Every test exercises a pure
function or a private builder method directly:

#432 — `_parse_tool_outputs` latched onto the FIRST `P(undercut_success)=` match and
never tracked which driver it belonged to, so `undercut_target` was assembled from the
POSITIONAL `rival_ahead` candidate instead of the rival N16 actually scored highest.
`score_undercut_tool` now embeds `candidate=<driver_y>` in its return string so the
parser can track the argmax (driver, prob) pair across every rival scored in one run.

#476 — `predict_pit_duration_tool`, `score_undercut_tool` and `recommend_compound_tool`
all accept free-text driver codes from the LLM with no guard against naming a car that
is not (or no longer) on track. Applies the same `_live_drivers` refusal shape
`score_undercut_tool` already used. `predict_pit_duration_tool`'s `under_sc` argument is
also LLM free text; it is now cross-checked against `self.sc_currently_active`, the
RCM-confirmed ground truth set for the duration of one real `_run_core` invocation.

#450 — `_build_pit_duration_features` fed N15 two FROZEN constants: `tight_pit_box`
was hardcoded 0 and `team_year_median` was always the 2.8s fallback, regardless of
which circuit, team or year was actually asked about. Both are now real lookups:
`tight_pit_box` checks the (slug-resolved) circuit against N15's trained Monaco/
Singapore/Hungary set, and `team_year_median` is aggregated from the raw per-GP
pitstops.parquet files at `PitAgentCFG.__post_init__` time.

#465 (F6) — `run_from_state`'s `d.get('position', 20)` used a SEARCHABLE sentinel: a
missing position silently became P20, and `driver_pos - 1 == 19` could then match a
REAL P19 rival, inventing a rival-ahead relationship for a car whose position was
actually unknown. The fix propagates `None` instead and skips the rival lookup.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).parent.parent.parent
_HAS_MODELS = (ROOT / "data" / "models" / "pit_prediction" / "model_config.json").exists()
_HAS_UNDERCUT_DATA = (
    ROOT / "data" / "processed" / "undercut_labeled" / "undercut_clean.parquet"
).exists()
_HAS_RAW_PITSTOPS = any((ROOT / "data" / "raw").glob("*/*/pitstops.parquet"))

# PitAgentCFG.__post_init__ loads the N15/N16 model bundles, the undercut training
# parquet, and now scans data/raw/<year>/<GP>/pitstops.parquet too. All of that comes
# from Hugging Face Hub on first run, not from git, so a CI runner without data/ skips
# this whole module rather than failing on missing fixtures.
pytestmark = pytest.mark.skipif(
    not (_HAS_MODELS and _HAS_UNDERCUT_DATA and _HAS_RAW_PITSTOPS),
    reason=(
        "needs data/models/pit_prediction/, data/processed/undercut_labeled/, and "
        "data/raw/<year>/<GP>/pitstops.parquet (data/ comes from HF, not git)"
    ),
)


class _FakeToolMessage:
    """Minimal stand-in for a LangChain ToolMessage — only `.content` is read."""

    def __init__(self, content: str) -> None:
        self.content = content


def _one_row_laps_df(driver: str, lap_number: int) -> pd.DataFrame:
    """A single-row laps_df just complete enough for _build_pit_duration_features."""
    return pd.DataFrame(
        [
            {
                "Driver": driver,
                "LapNumber": lap_number,
                "TyreLife": 10.0,
                "Compound": "MEDIUM",
            }
        ]
    )


@pytest.fixture(scope="module")
def pit_agent():
    """A real PitStrategyAgent — real N15/N16 models, real aggregated lookups.

    Module-scoped: constructing it loads several joblib model bundles and now also
    scans every raw pitstops.parquet, so tests share one instance and only mutate the
    per-call state (laps_df / session_meta / _live_drivers) each test needs.
    """
    from src.agents.pit_strategy_agent import PitStrategyAgent

    return PitStrategyAgent()


# ---------------------------------------------------------------------------
# #432 — undercut_target must be the argmax-scored driver, not the first one seen
# ---------------------------------------------------------------------------


def test_parse_tool_outputs_tracks_the_argmax_undercut_driver():
    """Two rivals scored in one run: the HIGHER-probability driver must win.

    Before #432, only the FIRST P(undercut_success) match was kept and the driver it
    belonged to was never recorded at all — undercut_target came from the positional
    rival_ahead candidate built for the prompt, which can name either rival depending
    on race state, independent of which one N16 actually favoured.
    """
    from src.agents.pit_strategy_agent import _parse_tool_outputs

    messages = [
        _FakeToolMessage(
            "candidate=SAI | P(undercut_success)=0.310 | threshold=0.522 | "
            "pos_gap=2 | tyre_life_diff=+3 laps | verdict=NO"
        ),
        _FakeToolMessage(
            "candidate=HAM | P(undercut_success)=0.680 | threshold=0.522 | "
            "pos_gap=1 | tyre_life_diff=-2 laps | verdict=YES"
        ),
    ]

    parsed = _parse_tool_outputs(messages)

    assert parsed["undercut_target"] == "HAM"
    assert parsed["undercut_prob"] == pytest.approx(0.680)


def test_parse_tool_outputs_argmax_is_order_independent():
    """The SAME two candidates in reverse order must still pick the higher probability.

    Guards against a regression that accidentally keeps "first seen" semantics instead
    of a true argmax (e.g. `>` vs never updating after the first match).
    """
    from src.agents.pit_strategy_agent import _parse_tool_outputs

    messages = [
        _FakeToolMessage(
            "candidate=HAM | P(undercut_success)=0.680 | threshold=0.522 | "
            "pos_gap=1 | tyre_life_diff=-2 laps | verdict=YES"
        ),
        _FakeToolMessage(
            "candidate=SAI | P(undercut_success)=0.310 | threshold=0.522 | "
            "pos_gap=2 | tyre_life_diff=+3 laps | verdict=NO"
        ),
    ]

    parsed = _parse_tool_outputs(messages)

    assert parsed["undercut_target"] == "HAM"


def test_validate_undercut_target_drops_a_car_not_on_track(pit_agent):
    """Assembly must re-check liveness independently of score_undercut_tool's own guard.

    The two guards must not depend on each other to be correct (#432): even if a
    driver code slipped through parsing (a stale message format, a test double), the
    final PitStrategyOutput.undercut_target must not name a car that is not racing.
    """
    pit_agent._live_drivers = {"NOR", "HAM"}
    assert pit_agent._validate_undercut_target("HUL") is None
    assert pit_agent._validate_undercut_target("HAM") == "HAM"
    # Unknown liveness (None) must not reject everything either.
    pit_agent._live_drivers = None
    assert pit_agent._validate_undercut_target("HUL") == "HUL"


# ---------------------------------------------------------------------------
# #476 — the LLM-facing tools must refuse a driver who is not on track
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "tool_name,extra_kwargs",
    [
        (
            "predict_pit_duration_tool",
            {"lap_number": 30, "compound": "MEDIUM", "compound_change": True, "under_sc": False},
        ),
        (
            "recommend_compound_tool",
            {"lap_number": 30, "current_compound": "MEDIUM"},
        ),
    ],
)
def test_tool_refuses_a_driver_not_on_track(pit_agent, tool_name, extra_kwargs):
    """Asking either LLM-facing tool about a car absent from the roster must refuse.

    Before #476 this would have gone straight into the feature builder and either
    crashed on missing lap data or, worse, quietly answered from stale data.
    """
    pit_agent.session_meta = {
        "gp_name": "Budapest",
        "year": 2024,
        "total_laps": 70,
        "team_lookup": {},
    }
    pit_agent._live_drivers = {"NOR", "HAM"}

    tool = next(t for t in pit_agent._tools if t.name == tool_name)
    result = tool.invoke({"driver": "ZZZ", **extra_kwargs})

    assert "REFUSED" in result
    assert "ZZZ" in result
    assert "NOR" in result or "HAM" in result  # the valid-drivers list names a real car


def test_a_live_driver_is_not_refused_by_the_guard(pit_agent):
    """The guard must not buy safety by refusing everyone — a real car must pass it."""
    pit_agent.laps_df = _one_row_laps_df("NOR", 30)
    pit_agent.session_meta = {
        "gp_name": "Budapest",
        "year": 2024,
        "total_laps": 70,
        "team_lookup": {"NOR": "McLaren"},
    }
    pit_agent._live_drivers = {"NOR", "HAM"}

    tool = next(t for t in pit_agent._tools if t.name == "predict_pit_duration_tool")
    result = tool.invoke(
        {
            "driver": "NOR",
            "lap_number": 30,
            "compound": "MEDIUM",
            "compound_change": False,
            "under_sc": False,
        }
    )

    assert "REFUSED" not in result
    assert "physical_stop" in result


def test_predict_pit_duration_tool_trusts_the_rcm_over_the_llm(pit_agent, monkeypatch):
    """An LLM-supplied under_sc that contradicts the confirmed RCM state is overridden.

    sc_currently_active is only ever set by _run_core, for the duration of one real
    invocation; setting it directly here stands in for that without needing an LLM
    run. _build_pit_duration_features is stubbed to CAPTURE the under_sc value the
    tool actually forwards, isolating the cross-check from the quantile models'
    sensitivity to that one feature (which is not itself under test here).
    """
    pit_agent.session_meta = {
        "gp_name": "Budapest",
        "year": 2024,
        "total_laps": 70,
        "team_lookup": {"NOR": "McLaren"},
    }
    pit_agent._live_drivers = {"NOR"}
    pit_agent.sc_currently_active = True  # RCM: the SC IS out right now

    captured: dict = {}

    def fake_build_features(driver, lap_number, compound, compound_change, under_sc):
        captured["under_sc"] = under_sc
        return pd.DataFrame([{col: 0 for col in pit_agent.cfg.pit_features}])

    monkeypatch.setattr(pit_agent, "_build_pit_duration_features", fake_build_features)

    tool = next(t for t in pit_agent._tools if t.name == "predict_pit_duration_tool")
    tool.invoke(
        # The LLM claims no SC (under_sc=False); the RCM's True must win instead.
        {
            "driver": "NOR",
            "lap_number": 30,
            "compound": "MEDIUM",
            "compound_change": False,
            "under_sc": False,
        }
    )

    assert captured["under_sc"] is True, "the RCM-confirmed value must override the LLM's guess"

    del pit_agent.sc_currently_active  # avoid leaking state into later tests


# ---------------------------------------------------------------------------
# #450 — tight_pit_box and team_year_median must be real lookups, not frozen constants
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "gp_name,expected",
    [
        ("Monaco", 1),
        ("Monaco Grand Prix", 1),  # the FastF1 full-event-name keyspace must resolve too
        ("Silverstone", 0),
    ],
)
def test_tight_pit_box_matches_n15s_trained_circuit_set(pit_agent, gp_name, expected):
    """Monaco (either keyspace) must flag tight_pit_box=1; Silverstone must not.

    N15 was trained with tight_pit_box = GP_Name.isin({"Monaco Grand Prix",
    "Singapore Grand Prix", "Hungarian Grand Prix"}) (N15_pit_duration.ipynb cell 4).
    Before #450 this feature was hardcoded to 0 for every circuit, every call.
    """
    pit_agent.laps_df = _one_row_laps_df("NOR", 20)
    pit_agent.session_meta = {
        "gp_name": gp_name,
        "year": 2024,
        "team_lookup": {"NOR": "McLaren"},
    }

    feat_df = pit_agent._build_pit_duration_features("NOR", 20, "MEDIUM", False, False)

    assert feat_df["tight_pit_box"].iloc[0] == expected


def test_team_year_median_is_aggregated_from_real_pit_data(pit_agent):
    """A known (team, year) combo must differ from the 2.8s constant fallback.

    Before #450, team_year_median was ALWAYS cfg.team_year_median_fallback (2.8s),
    for every team and every year — the aggregation in _load_team_year_medians must
    actually vary the feature with the input.
    """
    cfg = pit_agent.cfg
    assert cfg.team_year_median, "aggregation must find at least one real (team, year) combo"

    ferrari_2024 = cfg.team_year_median.get(("Ferrari", 2024))
    assert ferrari_2024 is not None, "Ferrari raced in 2024; raw pit data must cover it"
    assert ferrari_2024 != pytest.approx(cfg.team_year_median_fallback)

    # The read-time helper must actually surface the aggregated value...
    assert cfg.team_year_median_for("Ferrari", 2024) == pytest.approx(ferrari_2024)
    # ...and still fall back cleanly for a combo with no raw data.
    assert cfg.team_year_median_for("NoSuchTeam", 1999) == pytest.approx(
        cfg.team_year_median_fallback
    )


# ---------------------------------------------------------------------------
# #465 (F6) — a missing position must propagate as None, not the searchable P20
# ---------------------------------------------------------------------------


def _stub_run_core(monkeypatch, captured: dict) -> None:
    """Replace PitStrategyAgent._run_core with a recorder, so run_from_state can be
    exercised end-to-end up to (and including) the rival_ahead computation without
    ever reaching the LangGraph ReAct agent or an LLM endpoint.

    Args:
        monkeypatch: The pytest monkeypatch fixture.
        captured: Dict the fake _run_core writes its `rival` argument into.
    """
    from src.agents.pit_strategy_agent import PitStrategyAgent, PitStrategyOutput

    def fake_run_core(
        self,
        driver,
        lap_number,
        compound,
        rival,
        sc_prob,
        laps_cliff,
        sc_currently_active=False,
        vsc_active=False,
    ):
        captured["rival"] = rival
        return PitStrategyOutput(
            action="STAY_OUT",
            recommended_lap=None,
            compound_recommendation="MEDIUM",
            stop_duration_p05=0.0,
            stop_duration_p50=0.0,
            stop_duration_p95=0.0,
            undercut_prob=None,
            undercut_target=None,
            sc_reactive=False,
            reasoning="",
        )

    monkeypatch.setattr(PitStrategyAgent, "_run_core", fake_run_core)


def test_run_from_state_rival_ahead_is_none_when_position_is_missing(pit_agent, monkeypatch):
    """A driver with no known position must not resolve rival_ahead via a fake P20.

    Before the fix, `d.get('position', 20)` handed a REAL P19 rival a rival-ahead
    relationship it never earned, purely because 20 - 1 happens to equal 19.
    """
    captured: dict = {}
    _stub_run_core(monkeypatch, captured)

    laps_df = pd.DataFrame([{"Driver": "NOR", "LapNumber": 30}])
    lap_state = {
        "driver": {"compound": "MEDIUM"},  # no 'position' key at all
        "session_meta": {
            "driver": "NOR",
            "gp_name": "Budapest",
            "total_laps": 70,
            "year": 2024,
            "team": "McLaren",
        },
        "lap_number": 30,
        # A real rival at P19 — this must NOT be picked as "the car ahead" just
        # because the dead default made driver_pos - 1 equal 19.
        "rivals": [{"driver": "HAM", "position": 19, "team": "Mercedes"}],
    }

    pit_agent.run_from_state(lap_state, laps_df)

    assert captured["rival"] is None, (
        "a missing position must not invent a rival-ahead via the old fake P20 default"
    )


def test_run_from_state_rival_ahead_resolves_when_position_is_known(pit_agent, monkeypatch):
    """Regression guard: a KNOWN position must still resolve the real rival ahead.

    The #465/F6 fix must not overcorrect into never finding a rival at all.
    """
    captured: dict = {}
    _stub_run_core(monkeypatch, captured)

    laps_df = pd.DataFrame([{"Driver": "NOR", "LapNumber": 30}])
    lap_state = {
        "driver": {"compound": "MEDIUM", "position": 5},
        "session_meta": {
            "driver": "NOR",
            "gp_name": "Budapest",
            "total_laps": 70,
            "year": 2024,
            "team": "McLaren",
        },
        "lap_number": 30,
        "rivals": [{"driver": "HAM", "position": 4, "team": "Mercedes"}],
    }

    pit_agent.run_from_state(lap_state, laps_df)

    assert captured["rival"] == "HAM"
