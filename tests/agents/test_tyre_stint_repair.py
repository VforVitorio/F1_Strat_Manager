"""#790 — the tyre-stint repair must fix broken races and leave healthy ones untouched.

The second half of that sentence is the load-bearing one. A rebuilt ``TyreLife`` that is
subtly wrong is worse than the NaN it replaces, because a NaN is visible in every
downstream check and a plausible integer is not. So the first test here is not that the
repair works — it is that the repair does nothing at all to data that was already right,
including the two shapes that LOOK like corruption and are not.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.f1_strat_manager.tyre_stint_repair import (
    find_misplaced_boundaries,
    repair_tyre_stints,
)


def _laps(rows):
    """Build a per-driver laps frame from (lap, stint, compound, tyre_life, pit_in)."""
    return pd.DataFrame(
        [
            {
                "Driver": "NOR",
                "LapNumber": float(lap),
                "Stint": stint,
                "Compound": compound,
                "TyreLife": tyre_life,
                "PitInTime": pd.Timedelta(seconds=1) if pit_in else pd.NaT,
                "Position": 1.0,
            }
            for lap, stint, compound, tyre_life, pit_in in rows
        ]
    )


HEALTHY = _laps([
    (1, 1.0, "MEDIUM", 1.0, False),
    (2, 1.0, "MEDIUM", 2.0, False),
    (3, 1.0, "MEDIUM", 3.0, True),    # pits here
    (4, 2.0, "HARD", 1.0, False),     # new stint starts on the out-lap: correct
    (5, 2.0, "HARD", 2.0, False),
])

# Monaco 2025 RUS's shape: a stop-and-go serves no tyres, so the stint correctly does not
# advance, and a LATER real stop is what changes the compound.
STOP_AND_GO_THEN_REAL_STOP = _laps([
    (1, 2.0, "MEDIUM", 5.0, False),
    (2, 2.0, "MEDIUM", 6.0, True),    # penalty served: no new set
    (3, 2.0, "MEDIUM", 7.0, False),
    (4, 2.0, "MEDIUM", 8.0, True),    # the real stop
    (5, 3.0, "HARD", 1.0, False),
])

# A pit entry with no tyre change anywhere after it (a served penalty near race end).
PENALTY_ONLY = _laps([
    (1, 1.0, "HARD", 4.0, False),
    (2, 1.0, "HARD", 5.0, True),
    (3, 1.0, "HARD", 6.0, False),
])

# Miami 2025 NOR's shape: one stop on lap 3, but the metadata only starts stint 2 on
# lap 6, so TyreLife counts through the stop and the new set's age is understated.
MIAMI_SHAPE = _laps([
    (1, 1.0, "MEDIUM", 1.0, False),
    (2, 1.0, "MEDIUM", 2.0, False),
    (3, 1.0, "MEDIUM", 3.0, True),    # real stop
    (4, 1.0, "MEDIUM", 4.0, False),   # impossible: counting across the stop
    (5, 1.0, "MEDIUM", 5.0, False),
    (6, 2.0, "HARD", 1.0, False),     # boundary lands 3 laps late
    (7, 2.0, "HARD", 2.0, False),
])


@pytest.mark.parametrize(
    "frame, shape",
    [
        (HEALTHY, "a normal stop whose stint advances on the out-lap"),
        (STOP_AND_GO_THEN_REAL_STOP, "a served stop-and-go followed by a real stop"),
        (PENALTY_ONLY, "a pit entry with no tyre change at all"),
    ],
)
def test_correct_data_is_returned_untouched(frame, shape):
    """The safeguard: no healthy shape may be 'repaired'.

    The two penalty shapes are here because they are the ones that superficially look
    broken — the stint does not advance across a pit entry — and acting on that single
    signal would rewrite a right answer into a wrong one.
    """
    repaired, report = repair_tyre_stints(frame)
    pd.testing.assert_frame_equal(repaired, frame)
    assert not report.changed_anything
    assert report.boundaries_corrected == 0


def test_a_served_penalty_is_not_a_misplaced_boundary():
    assert find_misplaced_boundaries(STOP_AND_GO_THEN_REAL_STOP) == []
    assert find_misplaced_boundaries(PENALTY_ONLY) == []


def test_the_miami_shape_is_detected():
    assert find_misplaced_boundaries(MIAMI_SHAPE) == [(3, 6)]


def test_the_rebuilt_age_counts_from_the_out_lap_not_from_the_feed():
    repaired, report = repair_tyre_stints(MIAMI_SHAPE)
    by_lap = repaired.set_index("LapNumber")

    # The out-lap is the new set's first lap.
    assert by_lap.loc[4.0, "TyreLife"] == 1.0
    assert by_lap.loc[5.0, "TyreLife"] == 2.0
    # And the laps the feed mislabelled now continue that count instead of restarting.
    assert by_lap.loc[6.0, "TyreLife"] == 3.0
    assert by_lap.loc[7.0, "TyreLife"] == 4.0

    assert report.boundaries_corrected == 1
    assert report.changed_anything


def test_the_impossible_count_across_the_stop_is_gone():
    """Before: 3 -> 4 across a pit stop. After: the age must drop, not rise."""
    before = MIAMI_SHAPE.set_index("LapNumber")["TyreLife"]
    assert before.loc[4.0] > before.loc[3.0]

    after = repair_tyre_stints(MIAMI_SHAPE)[0].set_index("LapNumber")["TyreLife"]
    assert after.loc[4.0] < after.loc[3.0]


def test_the_corrected_laps_take_the_new_compound():
    repaired = repair_tyre_stints(MIAMI_SHAPE)[0].set_index("LapNumber")
    assert repaired.loc[4.0, "Compound"] == "HARD"
    assert repaired.loc[4.0, "Stint"] == 2.0
    # The laps genuinely on the old set are left alone.
    assert repaired.loc[3.0, "Compound"] == "MEDIUM"
    assert repaired.loc[3.0, "Stint"] == 1.0


def test_a_missing_first_stint_stays_unknown_on_the_FIRING_path():
    """The repair must never invent a starting tyre age, even while it repairs elsewhere.

    A car may start a stint on a used set and the offset varies per driver, so a first
    stint with no metadata stays NaN. The fixture is deliberately one where the repair
    DOES fire (a misplaced boundary at the stop): an earlier version of this test used a
    frame the function never touched at all, so it passed without ever reaching the code
    it claimed to pin — the project's recorded wrong-reason-green scar.
    """
    unknown_then_repairable = _laps([
        (1, None, "nan", None, False),
        (2, None, "nan", None, False),
        (3, 1.0, "MEDIUM", 1.0, False),   # feed recovers mid-stint: age counts from zero
        (4, 1.0, "MEDIUM", 2.0, True),    # real stop, on a now-known compound
        (5, 1.0, "MEDIUM", 3.0, False),   # mislabelled: already the new set
        (6, 2.0, "HARD", 1.0, False),     # boundary lands late
    ])
    repaired, report = repair_tyre_stints(unknown_then_repairable)
    by_lap = repaired.set_index("LapNumber")

    assert report.changed_anything, "the fixture must exercise the repairing path"
    # Laps on the race-start set: their age was never published, so it stays unknown -
    # including lap 3, where the feed offered a plausible 1.0 that is provably wrong.
    assert pd.isna(by_lap.loc[1.0, "TyreLife"])
    assert pd.isna(by_lap.loc[3.0, "TyreLife"])
    assert pd.isna(by_lap.loc[4.0, "TyreLife"])
    # The post-stop laps ARE recoverable and get counted from the out-lap.
    assert by_lap.loc[5.0, "TyreLife"] == 1.0
    assert by_lap.loc[6.0, "TyreLife"] == 2.0


def test_a_stop_made_while_the_feed_is_dark_is_left_unknown_rather_than_guessed():
    """The conservative edge: no real compound before the stop means no confirmed change.

    A pit entry alone does not prove a tyre change (a served penalty is one too), and with
    the pre-stop compound lost there is nothing left to confirm it with. So the fabricated
    ages are nulled and nothing is rebuilt, rather than a stint being invented from a
    stop that might have been a penalty. Miami 2025 BOR/STR/HAD have exactly this shape.
    """
    dark_at_the_stop = _laps([
        (1, None, "nan", None, False),
        (2, None, "nan", None, False),
        (3, None, "nan", None, True),     # stop while the feed is dark
        (4, None, "nan", None, False),
        (5, 1.0, "HARD", 1.0, False),     # metadata reappears
        (6, 1.0, "HARD", 2.0, False),
    ])
    repaired, report = repair_tyre_stints(dark_at_the_stop)

    assert report.boundaries_corrected == 0, "nothing confirms a tyre change here"
    assert repaired["TyreLife"].isna().all(), "plausible-but-unprovable ages become unknown"
    assert report.fabricated_ages_nulled == 2


# The feed going dark stringifies a missing compound rather than emitting NaN. Treating
# that as "a different compound" would let data loss masquerade as a tyre change, and
# writing it back would put a data-loss marker over a real compound (Montréal 2023 TSU).
FEED_DIED_MID_STINT = _laps([
    (1, 2.0, "HARD", 30.0, False),
    (2, 2.0, "HARD", 31.0, True),
    (3, 3.0, "None", None, False),
    (4, 3.0, "None", None, False),
])


def test_a_data_loss_sentinel_is_not_a_compound_change():
    assert find_misplaced_boundaries(FEED_DIED_MID_STINT) == []


def test_a_real_compound_is_never_overwritten_by_a_sentinel():
    repaired, _ = repair_tyre_stints(FEED_DIED_MID_STINT)
    assert repaired.set_index("LapNumber").loc[2.0, "Compound"] == "HARD"


# The feed invents a stint mid-race: metadata reappears on a lap that is NOT a pit
# out-lap, so its ages count from the wrong zero. This driver never pits, which is why a
# boundary correction alone cannot reach him (Miami 2025 GAS/HUL/BEA).
FABRICATED_NO_STOP = _laps([
    (1, None, "nan", None, False),
    (2, None, "nan", None, False),
    (3, 1.0, "MEDIUM", 1.0, False),   # a 3-lap-old set reported as 1 lap old
    (4, 1.0, "MEDIUM", 2.0, False),
])


def test_fabricated_ages_are_nulled_even_without_a_pit_stop():
    """The invisible half: a plausible integer that is provably too small becomes NaN."""
    repaired, report = repair_tyre_stints(FABRICATED_NO_STOP)
    ages = repaired.set_index("LapNumber")["TyreLife"]
    assert ages.isna().all()
    assert report.fabricated_ages_nulled == 2


def test_a_used_set_keeps_its_prior_usage():
    """FreshTyre is read, not assumed: a used set does not restart at 1.

    Measured across every healthy stint transition in 2024/2025, a fresh set starts at
    exactly 1.0 (1156/1156) while used sets start between 2 and 16. Hardcoding 1 would
    silently invent for stints 2+ the very offset the module refuses to invent for
    stint 1.
    """
    used_set = _laps([
        (1, 1.0, "MEDIUM", 1.0, False),
        (2, 1.0, "MEDIUM", 2.0, True),    # real stop
        (3, 1.0, "MEDIUM", 3.0, False),   # mislabelled: already the new set
        (4, 2.0, "HARD", 4.0, False),     # feed says the set arrived with 3 laps on it
    ])
    repaired, _ = repair_tyre_stints(used_set)
    by_lap = repaired.set_index("LapNumber")
    # Prior usage is 3 (published 4 minus the 1 it would carry if fresh), so the out-lap
    # is 4 rather than 1 and the count continues from there.
    assert by_lap.loc[3.0, "TyreLife"] == 4.0
    assert by_lap.loc[4.0, "TyreLife"] == 5.0


def test_a_nan_driver_row_is_not_annihilated():
    """groupby drops NaN keys and reindex re-inserts them as all-NaN, destroying columns
    this repair has no business touching. No shipped race carries one, but this runs on
    every future download."""
    frame = _laps([(1, 1.0, "MEDIUM", 1.0, False), (2, 1.0, "MEDIUM", 2.0, False)])
    frame.loc[0, "Driver"] = None
    repaired, _ = repair_tyre_stints(frame)
    assert repaired.iloc[0]["LapNumber"] == 1.0
    assert repaired.iloc[0]["Position"] == 1.0


def test_a_frame_without_the_required_columns_degrades_quietly():
    bare = pd.DataFrame({"Driver": ["NOR"], "LapNumber": [1.0]})
    repaired, report = repair_tyre_stints(bare)
    pd.testing.assert_frame_equal(repaired, bare)
    assert not report.changed_anything


def test_a_boundary_with_no_published_age_rebuilds_nothing():
    """No anchor means no rebuild — counting from 1 would invent a fresh set.

    The boundary lap here carries a real compound but a NaN TyreLife, so there is nothing
    to subtract the prior usage from. An earlier version returned 0.0 in that case and
    filled the laps with fabricated fresh-set ages, contradicting the module's own rule.
    """
    no_anchor = _laps([
        (1, 1.0, "MEDIUM", 1.0, False),
        (2, 1.0, "MEDIUM", 2.0, True),    # real stop
        (3, 1.0, "MEDIUM", 3.0, False),
        (4, 2.0, "HARD", None, False),    # boundary, but no published age
    ])
    repaired, report = repair_tyre_stints(no_anchor)
    assert report.boundaries_corrected == 0
    pd.testing.assert_frame_equal(repaired, no_anchor)


def test_a_boundary_with_no_stint_id_does_not_half_apply():
    """Either the whole correction lands, or none of it does.

    Rewriting Compound and then bailing on a missing Stint would return a MUTATED frame
    while reporting no change — and the caller uses that report to decide whether to
    patch the featured frame, so raw and featured would end up disagreeing.
    """
    no_stint = _laps([
        (1, 1.0, "MEDIUM", 1.0, False),
        (2, 1.0, "MEDIUM", 2.0, True),
        (3, 1.0, "MEDIUM", 3.0, False),
        (4, None, "HARD", 4.0, False),    # boundary compound is real, Stint is not
    ])
    repaired, report = repair_tyre_stints(no_stint)
    assert not report.changed_anything
    # Reported no change, so there must BE no change.
    pd.testing.assert_frame_equal(repaired, no_stint)
