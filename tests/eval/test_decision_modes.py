"""Tests for the decision-agreement tier (#708).

Two layers, deliberately split the same way ``test_position_projection.py`` splits:

1. Pure tests that pin the contract, which stops the guard rails make
   unanswerable, how an agreement is aggregated, when coverage is untrustworthy,
   and that the report keeps naming its scope. These run everywhere.

2. One data-tier test that refuses to believe a sample it has not counted. It
   needs ``data/raw`` and skips without it.

The report's honesty is itself under test here.
``test_render_names_every_race_and_keeps_season_scope_apart_from_coverage``
exists because the single most damaging edit anyone could make to this module is
deleting the sentence that says which races the figures come from. That sentence
used to say the sample was six races; since #1143 it says the whole 2025 season,
and the test follows the scope rather than the wording.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.strategy.eval.decision_modes import (
    DECISION_WINDOW_LAPS,
    MIN_SCORED_SHARE,
    SAMPLED_RACES,
    DecisionAgreement,
    StopVerdict,
    _asks_to_stop,
    _pit_decision_lap,
    _render_table,
    _replay_span,
    coverage_verdict,
    guard_rail_block,
    lap_inputs,
)
from src.strategy.inference.guard_rails import _NO_PIT_BEFORE_LAP

ROOT = Path(__file__).parent.parent.parent
_HAS_RAW = (ROOT / "data" / "raw" / "2024").is_dir()


def _agreement(
    offsets, guard_railed=0, no_call=0, races=6, no_data=0, no_boundary=0
) -> DecisionAgreement:
    return DecisionAgreement(
        offsets=np.array(offsets, dtype=int),
        guard_railed=guard_railed,
        no_call=no_call,
        races=races,
        no_data=no_data,
        no_boundary=no_boundary,
    )


# --- the import surface ----------------------------------------------------


def test_importing_the_module_does_not_load_the_agent_stack():
    """Importing this report must not require model weights on disk.

    ``no_llm`` pulls in the agent stack, which loads LightGBM weights at import
    time. Importing the guard rails from there turned every `f1-eval` subcommand
    into something that could not even be COLLECTED without `data/models/`, and it
    is what turned CI red on the first push of this tier.
    """
    import subprocess
    import sys

    probe = (
        "import sys; import src.strategy.eval.decision_modes; "
        "loaded = [m for m in sys.modules if m.startswith('src.agents')]; "
        "print(','.join(sorted(loaded)))"
    )
    out = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        cwd=ROOT,
        check=True,
    )
    assert out.stdout.strip() == "", f"agent modules imported eagerly: {out.stdout.strip()}"


# --- guard rails: which stops can never be agreed with ---------------------


@pytest.mark.parametrize(
    ("lap", "total", "compound", "tyre_life", "expected"),
    [
        (3, 57, "MEDIUM", 20, "opening_laps"),
        (56, 57, "MEDIUM", 20, "closing_laps"),
        (30, 57, "SOFT", 1, "min_stint"),
        (30, 57, None, 5, "min_stint"),
        (30, 57, "SOFT", 20, None),
        (30, 57, "MEDIUM", None, None),
    ],
)
def test_guard_rail_block_names_the_rule(lap, total, compound, tyre_life, expected):
    """Each rail is named, so an exclusion can be counted instead of vanishing."""
    assert guard_rail_block(lap, total, compound, tyre_life) == expected


def test_block_agrees_with_the_rail_on_every_lap_of_a_race():
    """The bucketer must agree with the rail itself, lap by lap, never re-derive it.

    The first version of ``guard_rail_block`` retyped the closing-laps boundary as
    ``remaining < 3`` against the rail's ``remaining <= 3``, and the test that was
    supposed to catch it re-derived the same boundary and so agreed with the bug.
    Asserting equivalence with ``apply_guard_rails`` is the only formulation that
    cannot drift, because there is nothing left to retype.
    """
    from src.strategy.inference.guard_rails import apply_guard_rails

    total = 57
    for lap in range(1, total + 1):
        for compound, tyre_life in (("SOFT", 20), ("MEDIUM", 4), ("HARD", 30)):
            rail_action, _reason = apply_guard_rails("PIT_NOW", lap, total, compound, tyre_life)
            blocked = guard_rail_block(lap, total, compound, tyre_life)
            assert (rail_action == "STAY_OUT") == (blocked is not None), (
                f"disagreement at lap {lap} on {compound}/{tyre_life}"
            )


def test_every_bucket_is_reachable_through_the_real_rail():
    """Each named bucket is produced by a real rail firing, not by a stale string.

    The bucketer keys on a fragment of the rail's reason. If a message is ever
    reworded this fails here instead of silently raising in the middle of a
    twenty-minute measurement run.
    """
    assert guard_rail_block(2, 57, "HARD", 30) == "opening_laps"
    assert guard_rail_block(55, 57, "HARD", 30) == "closing_laps"
    assert guard_rail_block(30, 57, "HARD", 2) == "min_stint"


def test_closing_rail_includes_the_boundary_lap():
    """Exactly three laps remaining is blocked: the rail is `<=`, not `<`."""
    assert guard_rail_block(54, 57, "HARD", 30) == "closing_laps"


def test_unknown_tyre_life_still_evaluates_the_lap_based_rails():
    """A missing tyre life suspends only the stint rail, never the lap ones."""
    assert guard_rail_block(2, 57, "SOFT", None) == "opening_laps"
    assert guard_rail_block(30, 57, "SOFT", None) is None


# --- picking the lap the stack would have chosen ---------------------------
#
# These used to test `_first_pit_lap`, which returned the earliest pit action in
# the window. #752 replaced it: that reported the window's LEFT EDGE for any
# stack already committed when the window opened, so the reported error moved
# with the window width instead of with the model. The tests below pin the
# transition semantics, and the width-invariance property they were missing is
# the last one in this group.


def test_the_decision_lap_is_the_transition_not_the_earliest_call():
    """The earliest pit call is NOT the answer when nothing declined before it.

    Shaped after a real occupant of `no_boundary_in_window` (PIA, 2025 Monza):
    committed when the window opened, withdrew mid-window, then re-committed. The
    retired helper returned 27, the window's own left edge, because it reported the
    first pit ACTION. The transition is at 30, where the stack actually changes its
    mind, and lap 26 is evaluated as a pit call so the earlier asks are ruled out by
    being committed rather than by being unwitnessed.
    """
    actions = {
        26: "PIT_NOW",
        27: "PIT_NOW",
        28: "PIT_NOW",
        29: "STAY_OUT",
        30: "PIT_NOW",
    }
    assert _pit_decision_lap(actions, 27, 32) == 30


def test_no_call_at_all_returns_none():
    """No call in the window is a result, and it must not read as lap zero."""
    assert _pit_decision_lap({lap: "STAY_OUT" for lap in range(27, 33)}, 27, 32) is None


def test_a_call_outside_the_window_is_not_agreement_with_this_stop():
    assert _pit_decision_lap({35: "PIT_NOW"}, 27, 32) is None


def test_a_stack_already_committed_has_no_decision_lap():
    """THE #752 CASE. Pitting on every lap offered is not a choice of lap.

    Lap 26 is the evaluated predecessor and it already says PIT, so no transition
    exists anywhere in [27, 32]. The old helper returned 27 — the window's edge —
    and called it the model's chosen lap.
    """
    actions = {lap: "PIT_NOW" for lap in range(26, 33)}
    assert _pit_decision_lap(actions, 27, 32) is None
    assert _asks_to_stop(actions, 27, 32) is True


def test_an_unevaluated_predecessor_cannot_witness_a_transition():
    """Absent is not the same as stay-out: lap 26 was never asked.

    Conservative on purpose. Treating a missing predecessor as a stay-out would
    reintroduce the edge report through the back door on the first lap of every
    window, which is why the caller evaluates one lap before it.
    """
    assert _pit_decision_lap({27: "PIT_NOW", 28: "PIT_NOW"}, 27, 32) is None


def test_the_decision_lap_does_not_move_when_only_the_window_widens():
    """The property that was missing, and the whole point of #752.

    Same actions, three window widths. A stack whose transition lies strictly inside
    all three must report the SAME lap, because the transition is a property of the
    model and the window is a property of the harness.

    This is the achievable half of the property, not width-invariance in general: a
    wider window is a strictly larger observation, so it can legitimately reveal a
    transition a narrower one could not reach. What it must never do is report its
    own edge, and that is what this pins.

    This fixture does NOT discriminate the retired helper, which returns 30 here too
    (laps 20-29 are explicit stay-outs, so its first pit action is also 30). The
    tests above are the ones that kill it; keeping that straight matters, because a
    docstring naming the wrong mechanism is worse than no docstring at all.
    """
    actions = {lap: "STAY_OUT" for lap in range(20, 30)}
    actions.update({lap: "PIT_NOW" for lap in range(30, 38)})

    assert _pit_decision_lap(actions, 27, 32) == 30
    assert _pit_decision_lap(actions, 25, 35) == 30
    assert _pit_decision_lap(actions, 22, 37) == 30


def test_the_replay_span_opens_before_the_first_scoreable_lap():
    """The one lap that a Fable gate found protected by a comment and nothing else.

    Reverting it left all 34 tests green while flipping a real stop out of `scored`
    and moving the published mean signed error by half its value. The property, not
    the formula: the predecessor of the earliest lap this run can score must itself
    be inside the replayed span, or that lap is unjudgeable by construction.
    """
    for stop_laps, total_laps in ([(32,), 53], [(18, 41), 60], [(9,), 50]):
        low, high = _replay_span(list(stop_laps), total_laps)
        earliest_scoreable = max(1, min(stop_laps) - DECISION_WINDOW_LAPS)
        assert low <= earliest_scoreable - 1 or earliest_scoreable == 1
        assert high >= max(stop_laps) + DECISION_WINDOW_LAPS or high == total_laps


def test_the_replay_span_cannot_ask_for_laps_that_do_not_exist():
    """Lap 1 has no predecessor and the race has no lap after the last one."""
    low, high = _replay_span([2], 50)
    assert low == 1

    low, high = _replay_span([48], 50)
    assert high == 50


def test_a_transition_at_the_opening_rail_is_the_rail_releasing_not_a_decision():
    """The retired defect's shape at a different boundary, found by gate G1.

    Actions are recorded AFTER the guard rails, and the opening rail forces stay-out
    below `_NO_PIT_BEFORE_LAP`. So a stack that wants to box from lap 1 is recorded as
    stay-outs then a pit, and the "transition" lands on the rail boundary for every
    such stop: a constant of the harness wearing a timing estimate, which is exactly
    what #752 retired at `window_low`.
    """
    railed = {1: "STAY_OUT", 2: "STAY_OUT", 3: "STAY_OUT", 4: "STAY_OUT"}
    railed.update({lap: "PIT_NOW" for lap in range(_NO_PIT_BEFORE_LAP, 12)})

    assert _pit_decision_lap(railed, 1, 11) is None
    assert _asks_to_stop(railed, 1, 11) is True


def test_a_transition_after_the_rail_releases_is_still_a_decision():
    """The guard above must not swallow a genuine mid-race change of mind.

    Without this, rejecting the rail boundary could be widened into rejecting every
    early lap, and the metric would go quiet on exactly the stops it is meant to
    grade.
    """
    actions = {lap: "STAY_OUT" for lap in range(1, _NO_PIT_BEFORE_LAP + 2)}
    actions.update({lap: "PIT_NOW" for lap in range(_NO_PIT_BEFORE_LAP + 2, 14)})

    assert _pit_decision_lap(actions, 1, 13) == _NO_PIT_BEFORE_LAP + 2


def test_the_new_bucket_counts_toward_eligible_so_the_share_is_not_inflated():
    """#752's denominator trap: those stops WERE looked at.

    Leaving `no_boundary` out of `eligible` would shrink the denominator by
    exactly the stops the old code used to score wrongly, and the coverage share
    would jump for a reason that is pure bookkeeping.
    """
    agreement = _agreement([0, -1, 1], guard_railed=2, no_call=5, no_data=1, no_boundary=9)

    assert agreement.eligible == 3 + 2 + 5 + 1 + 9
    assert agreement.scored_share == pytest.approx(3 / 20)


def test_asks_to_stop_separates_declining_from_being_already_committed():
    """The two findings the old bucketing merged, and they are opposites."""
    declined = {lap: "STAY_OUT" for lap in range(27, 33)}
    committed = {lap: "PIT_NOW" for lap in range(26, 33)}

    assert _asks_to_stop(declined, 27, 32) is False
    assert _asks_to_stop(committed, 27, 32) is True
    # Both are unscored, and for opposite reasons.
    assert _pit_decision_lap(declined, 27, 32) is None
    assert _pit_decision_lap(committed, 27, 32) is None


# --- aggregation -----------------------------------------------------------


def test_agreement_reports_signed_bias_not_just_magnitude():
    """A stack that always stops two laps early must not look unbiased."""
    agreement = _agreement([-2, -2, -2, -2])
    assert agreement.mean_signed_error == -2.0
    assert agreement.mean_absolute_error == 2.0
    assert agreement.exact == 0.0


def test_agreement_tolerance_bands_are_nested():
    offsets = [0, 1, -1, 2, 4]
    agreement = _agreement(offsets)
    assert agreement.exact == pytest.approx(0.2)
    assert agreement.within_one == pytest.approx(0.6)
    assert agreement.within_two == pytest.approx(0.8)


def test_retired_cars_are_counted_apart_from_declined_calls():
    """`no_data` and `no_call_in_window` must never be merged.

    A car that had already retired gave the stack nothing to evaluate; a car the
    stack looked at and declined to stop is a finding. Folding the first into the
    second charges a retirement to the model as a missed call — the same shape as
    the sentinel bugs this repo has paid for before.
    """
    agreement = _agreement([0, 1], no_call=3, no_data=5)
    assert agreement.eligible == 10
    assert agreement.no_call == 3
    assert agreement.no_data == 5


def test_no_data_counts_against_coverage():
    """Stops the tier could not look at still shrink the share it can vouch for."""
    assert coverage_verdict(_agreement([0] * 5, no_data=5)) == "masked"


def test_empty_agreement_reports_zero_rather_than_dividing_by_zero():
    """An empty sample is a reporting state, not a crash and not a perfect score."""
    agreement = _agreement([])
    assert agreement.sample_size == 0
    assert agreement.within_one == 0.0
    assert agreement.scored_share == 0.0


# --- the coverage guard ----------------------------------------------------


def test_coverage_ok_when_most_stops_were_scored():
    assert coverage_verdict(_agreement([0] * 9, guard_railed=1)) == "ok"


def test_coverage_masked_when_the_unscored_buckets_dominate():
    """The adapted compensation guard: a headline drawn from a third of the sample
    is a headline about whatever survived, and the survivors are not random."""
    agreement = _agreement([0] * 3, guard_railed=4, no_call=3)
    assert agreement.scored_share < MIN_SCORED_SHARE
    assert coverage_verdict(agreement) == "masked"


def test_coverage_unavailable_when_nothing_was_eligible():
    assert coverage_verdict(_agreement([])) == "unavailable"


# --- which laps are evaluable at all ---------------------------------------


def _lap_state(lap, position=4, tyre_life=12):
    return {
        "driver": {
            "lap_number": lap,
            "position": position,
            "compound": "MEDIUM",
            "tyre_life": tyre_life,
            "gap_ahead_s": 2.0,
        },
        "weather": {"air_temp": 25.0, "track_temp": 35.0},
        "session_meta": {"total_laps": 57},
    }


def test_laps_without_a_position_are_skipped_not_defaulted():
    """A None position skips the lap; it must never become a number.

    This crashed the first real measurement run. The state manager returns None on
    purpose because a sentinel position has already collided with a real one here,
    so the fix is to skip the lap, not to invent a plausible place.
    """
    assert lap_inputs(_lap_state(10, position=None)) is None
    assert lap_inputs(_lap_state(11)) == 11


def test_retired_cars_yield_nothing_to_evaluate():
    """An empty driver dict is how a retirement shows up for the rest of the race."""
    assert lap_inputs({"driver": {}}) is None
    assert lap_inputs({}) is None


def test_lap_inputs_decides_answerability_and_nothing_else():
    """It returns a lap number, never field values (#829).

    It used to return a dict that the caller fed into a hand-built ``RaceState``,
    and two of those fields were constants: a ``gap_ahead_s`` read from a key the
    driver dict does not carry, so 2.0 on every lap of every race, and a
    ``pace_delta_s`` of 0.0. The values now come from the canonical
    ``build_race_state``. This asserts the shape so nothing can drift back into
    inventing them here.
    """
    result = lap_inputs(_lap_state(11, tyre_life=0))
    assert isinstance(result, int)
    assert result == 11


def test_tyre_life_is_the_builders_contract_now():
    """The guard this module used to carry lives where the value is produced.

    ``build_race_state`` maps an unknown tyre life to ``UNKNOWN_TYRE_LIFE = 0``,
    measured to occur zero times across 2023-2025, so it cannot collide with a
    real reading. This module's old fallback was ``10``, which is a perfectly
    ordinary tyre age. The contract is asserted in
    ``tests/agents/test_race_state_builder.py``; duplicating it here is how two
    copies of one rule start to drift.
    """
    from src.agents.race_state_builder import UNKNOWN_TYRE_LIFE

    assert UNKNOWN_TYRE_LIFE == 0


# --- the report's honesty --------------------------------------------------


def test_render_names_every_race_and_keeps_season_scope_apart_from_coverage():
    """The scope caveats are part of the artifact, not decoration.

    The sample is now the whole 2025 season, so the old "stratified subset, not full
    coverage" wording is gone. The distinction it protected is not: covering every race
    of a season is a different claim from the coverage verdict, which says how many of
    those stops the metric could locate a decision inside. A reader who conflates the
    two reads `masked` as "we only looked at some races".
    """
    body = _render_table(
        _agreement([0, 1], guard_railed=1),
        [StopVerdict(2025, "Lusail", "NOR", 30, 30, 0, "scored")],
        "ok",
    )
    assert "not a stratified subset" in body
    assert "not** the same thing as the coverage verdict" in body
    assert "no-llm" in body
    for _year, race in SAMPLED_RACES:
        assert race in body


def test_render_without_data_says_so_instead_of_printing_zeros():
    body = _render_table(None, [], "unavailable")
    assert "Not measured" in body
    assert "0.0%" not in body


# --- the one test that checks against the world ----------------------------


@pytest.mark.data
@pytest.mark.skipif(not _HAS_RAW, reason="data/raw absent (CI runner without the dataset)")
def test_measured_sample_is_non_empty_before_any_figure_is_believed():
    """Guard against a green run that quietly graded nothing.

    A tier iterating a DISCOVERED set can pass every assertion about the empty
    set. This asserts the set exists first; the repo has shipped that bug before.
    """
    from src.strategy.eval.decision_modes import measure_decision_agreement

    agreement, verdicts = measure_decision_agreement(races=((2025, "Lusail"),))
    assert verdicts, "no real stops enumerated: the sample is empty, not accurate"
    assert agreement.eligible == len(verdicts)
    assert agreement.races == 1
    assert all(v.offset_laps is None for v in verdicts if v.bucket != "scored")
