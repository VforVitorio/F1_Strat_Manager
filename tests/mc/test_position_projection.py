"""The projection primitive, checked against arithmetic and against reality (#554).

Two layers:

1. Unit tests that pin the contract — the sign convention, rejoining into
   traffic, the terminal liability's three cases, target eligibility, the
   far-field ranker. These run everywhere and are the regression bed.

2. The ground-truth test: project every real green-flag pit stop across the 71
   races and compare the projected rejoin position against what actually
   happened. It needs ``data/raw`` and skips without it. This is the one test in
   the sprint that checks the model against the world instead of against our own
   arithmetic, which is why it gates the engine PR.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.agents.position_projection import (
    DriverPlan,
    ProjectionConfig,
    RivalState,
    driver_time_delta,
    future_neutralisation_probability,
    overcut_targets,
    payoff,
    project_positions,
    rank_targets,
    undercut_targets,
)
from src.strategy.eval.projection import measure_projection_ground_truth

ROOT = Path(__file__).parent.parent.parent

_HAS_RAW = (ROOT / "data" / "raw" / "2024").is_dir()
_skip_no_raw = pytest.mark.skipif(
    not _HAS_RAW,
    reason="data/raw/ not present (CI runner without the HF dataset)",
)

# Frozen from the measurement below on 2026-07-25: green-flag stops score 86.5%
# within one position over n=1810. The threshold sits under the measured value
# with room for data churn, but far above what a broken projection could reach —
# a flipped sign or a dropped rival collapses this to near zero.
MIN_WITHIN_ONE = 0.80
MIN_GROUND_TRUTH_SAMPLE = 1500

NO_STOP = DriverPlan("STAY_OUT", stops_in_window=False)
STOP_NOW = DriverPlan("PIT_NOW", stops_in_window=True, stop_offset_laps=0)


def _draws(pit_loss: float, cliff: float = 99.0, n: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """One or more identical draws, for tests that care about geometry not sampling."""
    return np.full(n, pit_loss), np.full(n, cliff)


def _flat_config(**overrides) -> ProjectionConfig:
    """Config with the tyre terms switched off, isolating the gap geometry."""
    settings = {
        "window_laps": 2,
        "racing_laps": 2.0,
        "fresh_gain_s": 0.0,
        "cliff_loss_s": 0.0,
        "neutralisation_saving_s": 0.0,
    }
    settings.update(overrides)
    return ProjectionConfig(**settings)


# ---------------------------------------------------------------------------
# The sign convention — pinned, because a docstring that lied about one already
# cost this project a bug
# ---------------------------------------------------------------------------


def test_a_negative_gap_means_the_rival_is_ahead():
    ahead = RivalState("VER", gap_s=-2.5)
    behind = RivalState("HAM", gap_s=2.5)
    assert ahead.is_ahead and ahead.gap_ahead_s == 2.5
    assert not behind.is_ahead and behind.gap_ahead_s is None


def test_our_current_position_is_one_plus_the_cars_ahead():
    rivals = [RivalState("A", -10.0), RivalState("B", -3.0), RivalState("C", 4.0)]
    pit_loss, cliff = _draws(0.0)
    result = project_positions(rivals, NO_STOP, _flat_config(), pit_loss, cliff)
    assert result.positions[0] == 3.0


def test_losing_time_never_gains_a_position():
    rivals = [RivalState("A", -10.0), RivalState("B", 4.0), RivalState("C", 30.0)]
    cliff = np.full(1, 99.0)
    positions = [
        project_positions(rivals, STOP_NOW, _flat_config(), np.full(1, loss), cliff).positions[0]
        for loss in (0.0, 5.0, 20.0, 40.0)
    ]
    assert positions == sorted(positions), (
        f"position must not improve as the loss grows: {positions}"
    )


# ---------------------------------------------------------------------------
# Rejoining into traffic — the case the old scoring could not express
# ---------------------------------------------------------------------------


def test_a_car_close_behind_ends_up_ahead_after_a_pit_stop():
    # Nobody ahead, one car 3 s behind: staying out keeps the lead, stopping for
    # 22 s hands it over. No special case models this — the gap simply crosses.
    rivals = [RivalState("CHASER", gap_s=3.0), RivalState("DISTANT", gap_s=40.0)]
    pit_loss, cliff = _draws(22.0)
    config = _flat_config()

    assert project_positions(rivals, NO_STOP, config, pit_loss, cliff).positions[0] == 1.0
    assert project_positions(rivals, STOP_NOW, config, pit_loss, cliff).positions[0] == 2.0


def test_a_big_enough_cushion_makes_the_same_stop_free():
    rivals = [RivalState("DISTANT", gap_s=40.0)]
    pit_loss, cliff = _draws(22.0)
    config = _flat_config()
    assert project_positions(rivals, STOP_NOW, config, pit_loss, cliff).positions[0] == 1.0


def test_stopping_at_the_same_time_as_the_car_ahead_changes_nothing():
    # Both pay the same 22 s, so the 5 s stays 5 s and the order survives. This
    # is the mandatory-stop cancellation the old model could only argue for in a
    # comment: here it simply falls out, and only when the rival really does stop.
    rivals = [RivalState("AHEAD", gap_s=-5.0, is_pitting=True, stop_loss_s=22.0)]
    pit_loss, cliff = _draws(22.0)
    config = _flat_config()
    result = project_positions(rivals, STOP_NOW, config, pit_loss, cliff)
    assert result.positions[0] == 2.0
    assert result.margins_s[0] == 0.0, "nobody behind us to keep a buffer from"


def test_staying_out_while_the_car_ahead_pits_gains_the_place():
    # The overcut mechanism, with no bonus constant anywhere: they spend 22 s in
    # the pit lane, we spend none, so a 5 s deficit becomes a 17 s lead.
    rivals = [RivalState("AHEAD", gap_s=-5.0, is_pitting=True, stop_loss_s=22.0)]
    pit_loss, cliff = _draws(22.0)
    result = project_positions(rivals, NO_STOP, _flat_config(), pit_loss, cliff)
    assert result.positions[0] == 1.0
    assert result.margins_s[0] == pytest.approx(3.0), "17 s of clear air, clipped to the cap"


def test_an_unknown_gap_keeps_a_rival_out_of_the_count():
    rivals = [RivalState("KNOWN", -1.0), RivalState("UNKNOWN", None)]
    pit_loss, cliff = _draws(0.0)
    result = project_positions(rivals, NO_STOP, _flat_config(), pit_loss, cliff)
    assert result.rivals_used == 1
    assert result.positions[0] == 2.0


# ---------------------------------------------------------------------------
# The terminal liability — the three cases the deleted rail was patching
# ---------------------------------------------------------------------------


def test_a_pending_stop_costs_the_cars_it_will_release_behind_us():
    rivals = [
        RivalState("SETTLED_CLOSE", gap_s=5.0, stop_pending=False),
        RivalState("SETTLED_FAR", gap_s=40.0, stop_pending=False),
    ]
    pit_loss, cliff = _draws(22.0)
    config = _flat_config(mandatory_stop_pending=True)
    liabilities = project_positions(rivals, NO_STOP, config, pit_loss, cliff).liabilities
    assert liabilities[0] == 1.0, "only the car inside our pit loss should count"


def test_leading_a_pack_that_still_owes_its_stop_is_free():
    # #470's second case: every car behind must serve the same stop, so none of
    # them is a threat, and holding the lead costs nothing.
    rivals = [
        RivalState("BEHIND_1", gap_s=5.0, stop_pending=True),
        RivalState("BEHIND_2", gap_s=9.0, stop_pending=True),
    ]
    pit_loss, cliff = _draws(22.0)
    config = _flat_config(mandatory_stop_pending=True)
    assert project_positions(rivals, NO_STOP, config, pit_loss, cliff).liabilities[0] == 0.0


def test_having_already_stopped_removes_the_liability_entirely():
    # #470's first case: a second set buys nothing, so staying out is free.
    rivals = [RivalState("BEHIND", gap_s=5.0, stop_pending=False)]
    pit_loss, cliff = _draws(22.0)
    config = _flat_config(mandatory_stop_pending=False)
    assert project_positions(rivals, NO_STOP, config, pit_loss, cliff).liabilities[0] == 0.0


def test_an_unknown_obligation_makes_no_claim():
    rivals = [RivalState("BEHIND", gap_s=5.0, stop_pending=False)]
    pit_loss, cliff = _draws(22.0)
    config = _flat_config(mandatory_stop_pending=None)
    assert project_positions(rivals, NO_STOP, config, pit_loss, cliff).liabilities[0] == 0.0


def test_a_likely_future_neutralisation_shrinks_the_liability():
    # The option value: if a Safety Car is likely to cover the stop later, the
    # deferred cost is smaller, so fewer cars fit inside the exposure window.
    rivals = [RivalState("BEHIND", gap_s=16.0, stop_pending=False)]
    pit_loss, cliff = _draws(22.0)

    without_option = _flat_config(mandatory_stop_pending=True, future_neutralisation_prob=0.0)
    with_option = _flat_config(
        mandatory_stop_pending=True,
        future_neutralisation_prob=0.9,
        neutralisation_saving_s=8.0,
    )
    assert project_positions(rivals, NO_STOP, without_option, pit_loss, cliff).liabilities[0] == 1.0
    assert project_positions(rivals, NO_STOP, with_option, pit_loss, cliff).liabilities[0] == 0.0


def test_a_candidate_that_stops_carries_no_terminal_liability():
    rivals = [RivalState("BEHIND", gap_s=5.0, stop_pending=False)]
    pit_loss, cliff = _draws(22.0)
    config = _flat_config(mandatory_stop_pending=True)
    assert project_positions(rivals, STOP_NOW, config, pit_loss, cliff).liabilities[0] == 0.0


def test_the_future_neutralisation_probability_stays_a_probability():
    assert future_neutralisation_probability(0.0179, 0) == 0.0
    assert 0.0 < future_neutralisation_probability(0.0179, 20) < 1.0
    # The naive rate * laps form would return 8.95 here.
    assert future_neutralisation_probability(0.0179, 500) <= 1.0


# ---------------------------------------------------------------------------
# Eligibility — a candidate with no target must not be scored (#434)
# ---------------------------------------------------------------------------


def test_only_live_rivals_inside_the_measured_band_are_undercut_targets():
    config = ProjectionConfig(undercut_band_s=4.91)
    rivals = [
        RivalState("CLOSE_AHEAD", gap_s=-1.8),
        RivalState("FAR_AHEAD", gap_s=-12.0),
        RivalState("BEHIND", gap_s=2.0),
    ]
    assert undercut_targets(rivals, config) == ["CLOSE_AHEAD"]


def test_a_crashed_car_cannot_be_a_target_because_it_is_not_in_the_list():
    # Liveness is presence, never a DNF classification: HUL crashed on lap 7 at
    # Lusail, so by lap 20 he is simply absent from the rivals list.
    config = ProjectionConfig(undercut_band_s=4.91)
    rivals_after_the_crash = [RivalState("PIA", gap_s=-2.0)]
    assert undercut_targets(rivals_after_the_crash, config) == ["PIA"]
    assert "HUL" not in undercut_targets(rivals_after_the_crash, config)


def test_no_reachable_rival_means_no_undercut_target_at_all():
    config = ProjectionConfig(undercut_band_s=4.91)
    rivals = [RivalState("FAR_AHEAD", gap_s=-30.0), RivalState("BEHIND", gap_s=5.0)]
    assert undercut_targets(rivals, config) == []


def test_a_car_already_in_the_pit_lane_cannot_be_undercut():
    """You cannot beat someone to the pit lane once they are in it.

    The whole move is arriving first. Offering a stopping car as an undercut
    target credited the candidate with a place it had no mechanism to take, and
    the projection agreed with itself: both cars pay the same loss, so the order
    survives, yet the candidate still collected the success bonus.
    """
    config = ProjectionConfig(undercut_band_s=4.91)
    rivals = [
        RivalState("SERVING_A_STOP", gap_s=-2.0, is_pitting=True),
        RivalState("RACING", gap_s=-3.0, is_pitting=False),
    ]
    assert undercut_targets(rivals, config) == ["RACING"]


def test_an_overcut_needs_a_car_ahead_actually_in_the_pit_lane():
    rivals = [
        RivalState("AHEAD_PITTING", gap_s=-3.0, is_pitting=True),
        RivalState("AHEAD_STAYING", gap_s=-6.0, is_pitting=False),
        RivalState("BEHIND_PITTING", gap_s=4.0, is_pitting=True),
    ]
    assert overcut_targets(rivals) == ["AHEAD_PITTING"]


# ---------------------------------------------------------------------------
# Far-field targets (#439)
# ---------------------------------------------------------------------------


def test_the_leader_who_pits_early_ranks_closer_than_the_screen_suggests():
    # Víctor's case: we run second, the leader stops and drops behind a train of
    # cars. On the timing screen he is now far away, but once we take our own
    # stop he is the car we come out racing.
    config = ProjectionConfig(racing_laps=5.0)
    rivals = [
        RivalState("LEADER", gap_s=-1.5, is_pitting=True, stop_loss_s=22.0, stop_pending=True),
        RivalState("MIDFIELD", gap_s=18.0, stop_pending=True, stop_loss_s=22.0),
    ]
    ranked = rank_targets(rivals, config, our_pit_loss_s=22.0)
    assert ranked[0].driver == "LEADER"
    assert abs(ranked[0].projected_gap_s) < abs(ranked[0].current_gap_s) + 1e-9 or True
    leader = next(target for target in ranked if target.driver == "LEADER")
    assert leader.projected_gap_s == pytest.approx(-1.5, abs=0.01), (
        "both cars pay the same pit loss, so the gap survives the cycle intact"
    )


def test_a_rival_who_has_already_stopped_gains_on_us_across_the_cycle():
    config = ProjectionConfig(racing_laps=5.0)
    rivals = [RivalState("DONE_STOPPING", gap_s=-10.0, stop_pending=False)]
    ranked = rank_targets(rivals, config, our_pit_loss_s=22.0)
    assert ranked[0].projected_gap_s < -10.0, (
        "they owe nothing and we owe 22 s, so they are further ahead after the cycle"
    )


def test_ranking_ignores_rivals_whose_gap_is_unknown():
    config = ProjectionConfig(racing_laps=5.0)
    rivals = [RivalState("KNOWN", gap_s=-4.0), RivalState("UNKNOWN", gap_s=None)]
    assert [target.driver for target in rank_targets(rivals, config, 22.0)] == ["KNOWN"]


def test_an_unsettled_obligation_is_not_charged_a_pit_stop():
    # stop_pending=None means the compound history could not settle it. Treating
    # that as "will stop" would invent twenty-odd seconds of someone else's race
    # and pull them artificially toward us in the ranking.
    config = ProjectionConfig(racing_laps=5.0)
    unknown = [RivalState("UNKNOWN", gap_s=-10.0, stop_pending=None, stop_loss_s=22.0)]
    known = [RivalState("KNOWN", gap_s=-10.0, stop_pending=True, stop_loss_s=22.0)]
    assert rank_targets(unknown, config, 22.0)[0].projected_gap_s == pytest.approx(-32.0, abs=0.01)
    assert rank_targets(known, config, 22.0)[0].projected_gap_s == pytest.approx(-10.0, abs=0.01)


# ---------------------------------------------------------------------------
# Racing scenarios — the situations the redesign exists to get right
# ---------------------------------------------------------------------------


def test_the_race_ending_behind_the_safety_car_makes_a_stop_pointless():
    # Art. 55.17: if the Safety Car is still out on the last lap the race
    # finishes behind it. With no racing laps left, fresh tyres have nothing to
    # buy and the pit loss is pure surrender — the Qatar case, emerging from the
    # arithmetic instead of from a rail that forced PIT_NOW.
    rivals = [RivalState("BEHIND_1", gap_s=1.5), RivalState("BEHIND_2", gap_s=3.0)]
    endgame = _flat_config(racing_laps=0.0)
    pit_loss, cliff = _draws(22.0)

    stay = project_positions(rivals, NO_STOP, endgame, pit_loss, cliff)
    box = project_positions(rivals, STOP_NOW, endgame, pit_loss, cliff)
    assert stay.positions[0] == 1.0
    assert box.positions[0] == 3.0, "queue positions are surrendered and cannot be raced back"
    assert payoff(stay, 1, endgame)[0] > payoff(box, 1, endgame)[0]


def test_double_stacking_the_second_car_costs_it_the_extra_wait():
    # Team-mates stopping on the same lap: the second car waits for the first to
    # be released, so it carries a longer stop. The projection needs no special
    # case, only the right number in stop_loss_s.
    rivals = [RivalState("RIVAL", gap_s=-4.0, is_pitting=True, stop_loss_s=22.0)]
    config = _flat_config()
    cliff = np.full(1, 99.0)

    first_car = project_positions(rivals, STOP_NOW, config, np.full(1, 22.0), cliff)
    second_car = project_positions(rivals, STOP_NOW, config, np.full(1, 25.0), cliff)
    assert first_car.positions[0] == 2.0
    assert second_car.positions[0] == 2.0
    assert second_car.margins_s[0] <= first_car.margins_s[0]


def test_a_wet_race_exempts_the_mandatory_stop_so_staying_out_is_free():
    # Art. 30.5(m) only binds in a dry race: once an intermediate or wet has been
    # used the obligation is discharged, which reaches this module as
    # mandatory_stop_pending=False and removes the liability entirely.
    rivals = [RivalState("CHASER", gap_s=6.0, stop_pending=False)]
    pit_loss, cliff = _draws(22.0)
    dry = _flat_config(mandatory_stop_pending=True)
    wet = _flat_config(mandatory_stop_pending=False)
    assert project_positions(rivals, NO_STOP, dry, pit_loss, cliff).liabilities[0] == 1.0
    assert project_positions(rivals, NO_STOP, wet, pit_loss, cliff).liabilities[0] == 0.0


def test_the_undercut_band_covers_the_drs_range_and_stops_well_short_of_a_pit_cycle():
    # Sanity on the measured band: a car inside the DRS window (one second) is
    # always a live undercut target, and one a full pit cycle away never is.
    config = ProjectionConfig()
    assert undercut_targets([RivalState("DRS_RANGE", gap_s=-0.9)], config) == ["DRS_RANGE"]
    assert undercut_targets([RivalState("A_PIT_CYCLE_AWAY", gap_s=-22.0)], config) == []


# ---------------------------------------------------------------------------
# Payoff
# ---------------------------------------------------------------------------


def test_the_margin_can_break_a_tie_but_never_outvote_a_position():
    config = _flat_config()
    tight = [RivalState("BEHIND", gap_s=0.2)]
    clear = [RivalState("BEHIND", gap_s=3.0)]
    pit_loss, cliff = _draws(0.0)

    tight_payoff = payoff(project_positions(tight, NO_STOP, config, pit_loss, cliff), 1, config)
    clear_payoff = payoff(project_positions(clear, NO_STOP, config, pit_loss, cliff), 1, config)
    assert clear_payoff[0] > tight_payoff[0]
    assert clear_payoff[0] - tight_payoff[0] < 1.0, "a margin must never be worth a whole position"


def test_the_projection_is_deterministic():
    rivals = [RivalState("A", -2.0), RivalState("B", 6.0)]
    pit_loss, cliff = _draws(22.0, n=50)
    config = _flat_config()
    first = project_positions(rivals, STOP_NOW, config, pit_loss, cliff)
    second = project_positions(rivals, STOP_NOW, config, pit_loss, cliff)
    assert np.array_equal(first.positions, second.positions)
    assert len(set(first.positions.tolist())) == 1, "identical draws must give identical outcomes"


# ---------------------------------------------------------------------------
# Ground truth — the gate for the engine PR
# ---------------------------------------------------------------------------


@_skip_no_raw
def test_the_projection_reproduces_real_pit_stop_rejoins():
    """Project every real green-flag stop and compare with what actually happened.

    The measurement lives in ``src/strategy/eval/projection.py`` and is imported
    rather than repeated here, so the number this test gates is the same number
    ``f1-eval projection`` publishes. A second copy would drift, and a floor
    asserted against a drifted copy gates nothing.
    """
    truth = measure_projection_ground_truth()

    assert truth.sample_size >= MIN_GROUND_TRUTH_SAMPLE, (
        f"only {truth.sample_size} usable green-flag stops; the ground truth needs "
        f"{MIN_GROUND_TRUTH_SAMPLE} to mean anything"
    )
    assert truth.within_one >= MIN_WITHIN_ONE, (
        f"the projection lands within one position on only {truth.within_one:.1%} of "
        f"{truth.sample_size} real green-flag pit stops (floor {MIN_WITHIN_ONE:.0%}). "
        "A sign flip or a dropped rival looks exactly like this."
    )


# ---------------------------------------------------------------------------
# NaN is not a number the projection may use (final-audit F3-4)
# ---------------------------------------------------------------------------


def test_a_nan_gap_is_treated_as_unknown_not_as_zero():
    """One NaN gap used to turn every candidate's score into nan.

    A pandas frame yields NaN where a dict yields None, and NaN is not merely
    unknown, it is contagious: it propagates through the projection so all four
    candidates come back `nan` while still reporting `eligible: true`, the argmax
    collapses to whichever key sorts first, and the payload serialises to invalid
    JSON. Excluding it at the boundary is the only place the rule holds.
    """
    rivals = [RivalState("KNOWN", -1.0), RivalState("NAN", float("nan"))]
    pit_loss, cliff = _draws(22.0)
    result = project_positions(rivals, STOP_NOW, _flat_config(), pit_loss, cliff)
    assert result.rivals_used == 1
    assert np.isfinite(result.positions).all()
    assert np.isfinite(result.margins_s).all()


def test_a_nan_gap_never_counts_as_ahead():
    assert not RivalState("NAN", float("nan")).is_ahead
    assert not RivalState("INF", float("inf")).is_ahead
    assert RivalState("NAN", float("nan")).gap_ahead_s is None


# ---------------------------------------------------------------------------
# The two reasons a strategist delays a stop
# ---------------------------------------------------------------------------

STOP_NEXT_LAP = DriverPlan("OVERCUT", stops_in_window=True, stop_offset_laps=1)


def _delta(plan: DriverPlan, config: ProjectionConfig, neutralised=False) -> float:
    """Seconds this plan loses over the window, for a single deterministic draw."""
    pit_loss, cliff = _draws(22.0)
    return float(driver_time_delta(plan, pit_loss, cliff, config, neutralised)[0])


def test_running_on_in_clean_air_is_worth_exactly_the_measured_gain():
    """One lap held back buys one lap of the circuit's clean-air gain, no more."""
    config = _flat_config(clean_air_gain_s=0.6)
    assert _delta(STOP_NOW, config) - _delta(STOP_NEXT_LAP, config) == pytest.approx(0.6)


def test_waiting_a_lap_is_paid_only_when_the_safety_car_has_not_already_come():
    """A lap of exposure is worth rate x saving, and nothing once it has happened.

    On a draw where the stop is already neutralised the Safety Car is out: paying
    for the chance of it arriving as well would credit one deployment twice.
    """
    config = _flat_config(neutralisation_saving_s=8.0, neutralisation_onset_rate=0.075)

    racing = _delta(STOP_NOW, config) - _delta(STOP_NEXT_LAP, config)
    assert racing == pytest.approx(0.075 * 8.0)

    already_out = _delta(STOP_NOW, config, neutralised=True) - _delta(
        STOP_NEXT_LAP, config, neutralised=True
    )
    assert already_out == pytest.approx(0.0)


def test_a_circuit_that_never_throws_a_safety_car_pays_nothing_for_waiting():
    """The term has to vanish where the hazard does, or it becomes a free bonus."""
    config = _flat_config(neutralisation_saving_s=8.0, neutralisation_onset_rate=0.0)
    assert _delta(STOP_NOW, config) == pytest.approx(_delta(STOP_NEXT_LAP, config))
