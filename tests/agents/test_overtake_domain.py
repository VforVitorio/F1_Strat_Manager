"""N27 declines to score a battle the overtake model never saw, instead of guessing.

N11's pair builder drops every pair more than 2.5 s apart before labelling — "not an active
battle" (`.nb_py/N11_overtake_eda.py:233-235`) — so the model has no labelled example out
there. `predict_overtake_tool` had range and roster guards but no gap guard, and LightGBM
extrapolated happily.

Measured over all 24 races of 2025 with N11's own pairing rule: **8,816 of 20,449
position-adjacent pairs (43.1%) sit outside the domain**, median gap 2.06 s, p90 9.11 s. The
earlier audit reported 41.9% over a different denominator (25,215 pairs, a differently
filtered frame); the two agree on the magnitude and this file re-derives the rate rather
than quoting either.

The probability does not reach the Monte Carlo — it reaches `threat_level`, the N31 prompt
and the dashboards. That is why an invented number is worse here than a noisy one: it is not
averaged, it is argued from.

This file also covers the two siblings fixed alongside it: the rolling-3 window, which paired
the two cars' laps by array position (diverges on 10.78% of pairs), and the unknown-circuit
sentinel, which was 0 — a real cluster.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).parent.parent.parent
_PAIRS = ROOT / "data" / "processed" / "overtake_labeled" / "overtake_pairs_2023_2025.parquet"
_FEATURED = ROOT / "data" / "processed" / "laps_featured_2025.parquet"
_HAS_MODELS = (ROOT / "data" / "models" / "tire_degradation" / "routing_config.json").exists()

# Importing the module builds RaceSituationConfig, which reads data/models/.
pytestmark = pytest.mark.skipif(
    not _HAS_MODELS, reason="importing race_situation_agent reads data/models/ (HF, not git)"
)


# --- the domain itself, re-derived from the labelled artefact -----------------


@pytest.mark.data
@pytest.mark.skipif(not _PAIRS.exists(), reason="labelled overtake pairs absent")
def test_the_declared_domain_is_the_one_the_labelled_pairs_span():
    """The constant is a claim about the training set; this checks it against the set.

    Asserting only `max <= 2.5` would also pass on a training set that happened to stop at
    1.0, which would make the gate too permissive without anything noticing, so the bound
    is asserted from BOTH sides.
    """
    from src.agents.race_situation_agent import _TRAINED_MAX_GAP_S

    gaps = pd.read_parquet(_PAIRS, columns=["gap_ahead_s"])["gap_ahead_s"]
    assert gaps.max() == pytest.approx(_TRAINED_MAX_GAP_S, abs=1e-6), (
        f"the labelled pairs reach {gaps.max()}, the declared domain is {_TRAINED_MAX_GAP_S}"
    )
    assert (gaps > _TRAINED_MAX_GAP_S).sum() == 0


@pytest.mark.data
@pytest.mark.skipif(not _PAIRS.exists(), reason="labelled overtake pairs absent")
def test_the_unknown_circuit_code_is_not_a_trained_cluster():
    """0 was the old default and it is a REAL cluster; the sentinel must not be one."""
    from src.agents.race_situation_agent import _UNKNOWN_CIRCUIT_CLUSTER

    trained = set(pd.read_parquet(_PAIRS, columns=["circuit_cluster"])["circuit_cluster"].unique())
    assert _UNKNOWN_CIRCUIT_CLUSTER not in trained, (
        f"the unknown code {_UNKNOWN_CIRCUIT_CLUSTER} collides with a trained cluster"
    )


@pytest.mark.data
@pytest.mark.skipif(not _FEATURED.exists(), reason="featured parquet absent")
def test_the_gate_is_not_vacuous_on_real_races():
    """A gate that never fires proves nothing. This pins that it fires on four laps in ten.

    Uses N11's own pairing rule — adjacent by Position, gap as |Time_x - Time_y| — so the
    share measured here is the share of real questions the tool now declines.
    """
    from src.agents.race_situation_agent import _TRAINED_MAX_GAP_S
    from src.f1_strat_manager.laps_augment import augment_featured_laps

    laps = augment_featured_laps(pd.read_parquet(_FEATURED), 2025)
    laps = laps[laps["Year"] == 2025].copy()
    if "Time_s" not in laps.columns:
        laps["Time_s"] = pd.to_timedelta(laps["Time"]).dt.total_seconds()

    gaps: list[float] = []
    for _gp, race in laps.groupby("GP_Name"):
        for _lap, grp in race.groupby("LapNumber"):
            usable = grp.dropna(subset=["Position", "LapTime_s", "Time_s"])
            usable = usable[usable["Position"] > 0]
            by_pos = dict(zip(usable["Position"], usable["Time_s"]))
            for pos, t_x in by_pos.items():
                t_y = by_pos.get(pos - 1)
                if t_y is not None:
                    gaps.append(abs(t_x - t_y))

    assert gaps, "no adjacent pairs found: this would hold vacuously"
    outside = sum(1 for g in gaps if g > _TRAINED_MAX_GAP_S)
    share = outside / len(gaps)
    assert 0.30 < share < 0.55, (
        f"{share:.1%} of adjacent pairs are out of domain; the fix was measured at 43.1% "
        "and a large move means the pairing rule or the artefact changed"
    )


# --- the parser: unknown must not become 0.0 ----------------------------------


def test_an_out_of_domain_answer_parses_as_unknown_not_as_zero():
    """The whole point. 0.0 is what the regulation asserts under a Safety Car."""
    from src.agents.race_situation_agent import _parse_tool_outputs

    class _Msg:
        content = (
            "P(overtake) = UNKNOWN (gap 9.11s is beyond N11's trained 2.5s domain; "
            "no labelled example exists out here) | gap=9.11s | "
            "pace_delta=0.400s/lap | DRS: inactive"
        )

    parsed = _parse_tool_outputs([_Msg()])
    assert parsed["overtake_prob"] is None
    # The other two are measurements, not model output, so they survive the decline.
    assert parsed["gap_ahead_s"] == pytest.approx(9.11)
    assert parsed["pace_delta_s"] == pytest.approx(0.400)


def test_an_in_domain_answer_still_parses_to_its_number():
    from src.agents.race_situation_agent import _parse_tool_outputs

    class _Msg:
        content = "P(overtake) = 0.412 | gap=1.20s | pace_delta=-0.300s/lap | DRS: active"

    parsed = _parse_tool_outputs([_Msg()])
    assert parsed["overtake_prob"] == pytest.approx(0.412)


def test_a_tool_that_never_ran_leaves_the_probability_unknown():
    """Previously this defaulted to 0.0, which reads as "no chance" rather than "no answer"."""
    from src.agents.race_situation_agent import _parse_tool_outputs

    assert _parse_tool_outputs([])["overtake_prob"] is None


# --- the consumers ------------------------------------------------------------


def test_an_unknown_probability_cannot_raise_the_threat_level():
    """It is not evidence, so it must not band as if it were. And it must not crash."""
    from src.agents.race_situation_agent import RaceSituationOutput

    unknown = RaceSituationOutput(overtake_prob=None, sc_prob_3lap=0.0)
    assert unknown.threat_level == "LOW"


def test_an_unknown_probability_does_not_suppress_the_other_signals():
    """The SC terms and the live-neutralisation flag are evaluated exactly as before.

    Without this, a fix that simply returned LOW whenever the overtake value is missing
    would pass the test above while silencing the safety-car path on 43% of laps.
    """
    from src.agents.race_situation_agent import CFG, RaceSituationOutput

    neutralised = RaceSituationOutput(
        overtake_prob=None, sc_prob_3lap=0.0, sc_currently_active=True
    )
    assert neutralised.threat_level == "HIGH"

    sc_says_so = RaceSituationOutput(overtake_prob=None, sc_prob_3lap=CFG.high_sc)
    assert sc_says_so.threat_level == "HIGH"

    medium = RaceSituationOutput(overtake_prob=None, sc_prob_3lap=CFG.medium_sc)
    assert medium.threat_level == "MEDIUM"


def test_a_known_probability_bands_exactly_as_it_did():
    """The in-domain path must be untouched by all of this."""
    from src.agents.race_situation_agent import CFG, RaceSituationOutput

    assert (
        RaceSituationOutput(overtake_prob=CFG.high_overtake, sc_prob_3lap=0.0).threat_level
        == "HIGH"
    )
    assert (
        RaceSituationOutput(overtake_prob=CFG.medium_overtake, sc_prob_3lap=0.0).threat_level
        == "MEDIUM"
    )
    assert RaceSituationOutput(overtake_prob=0.0, sc_prob_3lap=0.0).threat_level == "LOW"


def test_the_prose_renderer_never_formats_none_as_a_number():
    """`f"{None:.2f}"` raises, and this value is rendered into the RCM override note."""
    from src.agents.race_situation_agent import _OUT_OF_DOMAIN_MARKER, _fmt_prob

    assert _fmt_prob(None) == _OUT_OF_DOMAIN_MARKER
    assert _fmt_prob(0.4123) == "0.41"


def test_the_dashboard_shows_an_absence_rather_than_zero_percent():
    """ "overtake 0%" and "we cannot say" would prompt opposite calls on the pit wall."""
    from src.arcade.dashboard.agent_formatters import format_situation

    _headline, _colour, body, _status = format_situation(
        {"overtake_prob": None, "sc_prob_3lap": 0.1, "threat_level": "LOW"}
    )
    overtake_line = next(text for text, _c in body if text.startswith("overtake"))
    assert "0%" not in overtake_line
    assert "—" in overtake_line


# --- the rolling window -------------------------------------------------------


def test_the_rolling_window_pairs_by_lap_number_not_by_array_position():
    """The sibling defect: one car missing a lap shifted every element of the subtraction.

    X has laps 8, 9, 10; Y is missing lap 9 (a pit or safety-car lap the featured frame
    drops). Positionally, X's lap 9 paired with Y's lap 10. By LapNumber only laps 8 and 10
    are shared, and N12's feature is the mean of the PAIR's own per-lap pace deltas.
    """
    from src.agents.race_situation_agent import _pair_rolling_features

    def _lap(driver: str, lap: int, lap_time_s: float, elapsed_s: float) -> dict:
        return {
            "Driver": driver,
            "LapNumber": lap,
            "LapTime": pd.Timedelta(seconds=lap_time_s),
            "Time": pd.Timedelta(seconds=elapsed_s),
        }

    laps_recent = pd.DataFrame(
        [
            _lap("X", 8, 91.0, 800.0),
            _lap("X", 9, 92.0, 892.0),
            _lap("X", 10, 90.0, 982.0),
            _lap("Y", 8, 90.0, 799.0),
            _lap("Y", 10, 89.0, 979.0),
        ]
    )

    rolling, trend = _pair_rolling_features(
        laps_recent,
        driver_x="X",
        driver_y="Y",
        lap_number=10,
        gap_ahead_s=3.0,
        pace_delta_s=1.0,
    )

    # Shared laps are 8 and 10: deltas +1.0 and +1.0 -> mean 1.0.
    assert rolling == pytest.approx(1.0)
    # The positional version would have used X8-Y8 and X9-Y10 = +1.0 and +3.0 -> 2.0.
    assert rolling != pytest.approx(2.0)
    # gap_trend is the pair's own gap series: (982-979) - (800-799) = 2.0.
    assert trend == pytest.approx(2.0)


def test_a_pair_with_no_shared_history_falls_back_to_this_lap():
    """`min_periods=1` in N12: the first lap of a pair's series still yields a value."""
    from src.agents.race_situation_agent import _pair_rolling_features

    empty = pd.DataFrame(columns=["Driver", "LapNumber", "LapTime", "Time"])
    rolling, trend = _pair_rolling_features(
        empty, driver_x="X", driver_y="Y", lap_number=5, gap_ahead_s=1.0, pace_delta_s=0.25
    )
    assert rolling == pytest.approx(0.25)
    assert trend == 0.0
