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


HEALTHY = _laps(
    [
        (1, 1.0, "MEDIUM", 1.0, False),
        (2, 1.0, "MEDIUM", 2.0, False),
        (3, 1.0, "MEDIUM", 3.0, True),  # pits here
        (4, 2.0, "HARD", 1.0, False),  # new stint starts on the out-lap: correct
        (5, 2.0, "HARD", 2.0, False),
    ]
)

# Monaco 2025 RUS's shape: a stop-and-go serves no tyres, so the stint correctly does not
# advance, and a LATER real stop is what changes the compound.
STOP_AND_GO_THEN_REAL_STOP = _laps(
    [
        (1, 2.0, "MEDIUM", 5.0, False),
        (2, 2.0, "MEDIUM", 6.0, True),  # penalty served: no new set
        (3, 2.0, "MEDIUM", 7.0, False),
        (4, 2.0, "MEDIUM", 8.0, True),  # the real stop
        (5, 3.0, "HARD", 1.0, False),
    ]
)

# A pit entry with no tyre change anywhere after it (a served penalty near race end).
PENALTY_ONLY = _laps(
    [
        (1, 1.0, "HARD", 4.0, False),
        (2, 1.0, "HARD", 5.0, True),
        (3, 1.0, "HARD", 6.0, False),
    ]
)

# Miami 2025 NOR's shape: one stop on lap 3, but the metadata only starts stint 2 on
# lap 6, so TyreLife counts through the stop and the new set's age is understated.
MIAMI_SHAPE = _laps(
    [
        (1, 1.0, "MEDIUM", 1.0, False),
        (2, 1.0, "MEDIUM", 2.0, False),
        (3, 1.0, "MEDIUM", 3.0, True),  # real stop
        (4, 1.0, "MEDIUM", 4.0, False),  # impossible: counting across the stop
        (5, 1.0, "MEDIUM", 5.0, False),
        (6, 2.0, "HARD", 1.0, False),  # boundary lands 3 laps late
        (7, 2.0, "HARD", 2.0, False),
    ]
)


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
    unknown_then_repairable = _laps(
        [
            (1, None, "nan", None, False),
            (2, None, "nan", None, False),
            (3, 1.0, "MEDIUM", 1.0, False),  # feed recovers mid-stint: age counts from zero
            (4, 1.0, "MEDIUM", 2.0, True),  # real stop, on a now-known compound
            (5, 1.0, "MEDIUM", 3.0, False),  # mislabelled: already the new set
            (6, 2.0, "HARD", 1.0, False),  # boundary lands late
        ]
    )
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
    dark_at_the_stop = _laps(
        [
            (1, None, "nan", None, False),
            (2, None, "nan", None, False),
            (3, None, "nan", None, True),  # stop while the feed is dark
            (4, None, "nan", None, False),
            (5, 1.0, "HARD", 1.0, False),  # metadata reappears
            (6, 1.0, "HARD", 2.0, False),
        ]
    )
    repaired, report = repair_tyre_stints(dark_at_the_stop)

    assert report.boundaries_corrected == 0, "nothing confirms a tyre change here"
    assert repaired["TyreLife"].isna().all(), "plausible-but-unprovable ages become unknown"
    assert report.fabricated_ages_nulled == 2


# The feed going dark stringifies a missing compound rather than emitting NaN. Treating
# that as "a different compound" would let data loss masquerade as a tyre change, and
# writing it back would put a data-loss marker over a real compound (Montréal 2023 TSU).
FEED_DIED_MID_STINT = _laps(
    [
        (1, 2.0, "HARD", 30.0, False),
        (2, 2.0, "HARD", 31.0, True),
        (3, 3.0, "None", None, False),
        (4, 3.0, "None", None, False),
    ]
)


def test_a_data_loss_sentinel_is_not_a_compound_change():
    assert find_misplaced_boundaries(FEED_DIED_MID_STINT) == []


def test_a_real_compound_is_never_overwritten_by_a_sentinel():
    repaired, _ = repair_tyre_stints(FEED_DIED_MID_STINT)
    assert repaired.set_index("LapNumber").loc[2.0, "Compound"] == "HARD"


# The feed invents a stint mid-race: metadata reappears on a lap that is NOT a pit
# out-lap, so its ages count from the wrong zero. This driver never pits, which is why a
# boundary correction alone cannot reach him (Miami 2025 GAS/HUL/BEA).
FABRICATED_NO_STOP = _laps(
    [
        (1, None, "nan", None, False),
        (2, None, "nan", None, False),
        (3, 1.0, "MEDIUM", 1.0, False),  # a 3-lap-old set reported as 1 lap old
        (4, 1.0, "MEDIUM", 2.0, False),
    ]
)


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
    used_set = _laps(
        [
            (1, 1.0, "MEDIUM", 1.0, False),
            (2, 1.0, "MEDIUM", 2.0, True),  # real stop
            (3, 1.0, "MEDIUM", 3.0, False),  # mislabelled: already the new set
            (4, 2.0, "HARD", 4.0, False),  # feed says the set arrived with 3 laps on it
        ]
    )
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
    no_anchor = _laps(
        [
            (1, 1.0, "MEDIUM", 1.0, False),
            (2, 1.0, "MEDIUM", 2.0, True),  # real stop
            (3, 1.0, "MEDIUM", 3.0, False),
            (4, 2.0, "HARD", None, False),  # boundary, but no published age
        ]
    )
    repaired, report = repair_tyre_stints(no_anchor)
    assert report.boundaries_corrected == 0
    pd.testing.assert_frame_equal(repaired, no_anchor)


def test_a_boundary_with_no_stint_id_does_not_half_apply():
    """Either the whole correction lands, or none of it does.

    Rewriting Compound and then bailing on a missing Stint would return a MUTATED frame
    while reporting no change — and the caller uses that report to decide whether to
    patch the featured frame, so raw and featured would end up disagreeing.
    """
    no_stint = _laps(
        [
            (1, 1.0, "MEDIUM", 1.0, False),
            (2, 1.0, "MEDIUM", 2.0, True),
            (3, 1.0, "MEDIUM", 3.0, False),
            (4, None, "HARD", 4.0, False),  # boundary compound is real, Stint is not
        ]
    )
    repaired, report = repair_tyre_stints(no_stint)
    assert not report.changed_anything
    # Reported no change, so there must BE no change.
    pd.testing.assert_frame_equal(repaired, no_stint)


# --- #988: an age republished on a set that never came off the car ---------------

# Melbourne 2025 STR's shape: the safety car leads the field through the pit lane, the
# feed opens a stint on each pass, and the age resets to 1 with INTERMEDIATE still on the
# car. It then counts on from that wrong zero for the whole run.
REPUBLISHED_ON_TRANSIT = _laps(
    [
        (1, 1.0, "INTERMEDIATE", 1.0, False),
        (2, 1.0, "INTERMEDIATE", 2.0, True),  # transit under the safety car
        (3, 2.0, "INTERMEDIATE", 1.0, True),  # the age resets: no set came off
        (4, 3.0, "INTERMEDIATE", 1.0, True),
        (5, 4.0, "INTERMEDIATE", 1.0, False),
        (6, 4.0, "INTERMEDIATE", 2.0, False),  # counts on from the wrong zero
        (7, 4.0, "INTERMEDIATE", 3.0, True),  # the real stop
        (8, 5.0, "HARD", 1.0, False),  # the new set's own age is fine
        (9, 5.0, "HARD", 2.0, False),
    ]
)


def test_a_republished_age_is_nulled_rather_than_believed():
    """The effect, not the mechanism: no lap of that run keeps a number.

    Nulling only the reset lap would leave laps 4 to 7 reading 1, 1, 2, 3 on a set that
    has done four, five, six and seven laps - short by three, and invisible.
    """
    repaired, report = repair_tyre_stints(REPUBLISHED_ON_TRANSIT)
    ages = dict(zip(repaired["LapNumber"], repaired["TyreLife"]))
    assert ages[1.0] == 1.0 and ages[2.0] == 2.0, "the honest prefix survives"
    assert all(pd.isna(ages[lap]) for lap in (3.0, 4.0, 5.0, 6.0, 7.0))
    assert report.republished_ages_nulled == 5
    assert report.drivers_touched == ["NOR"]


def test_the_new_set_after_the_stop_keeps_its_own_age():
    """The run ends at the compound change, so the repair cannot swallow the next stint."""
    repaired, _ = repair_tyre_stints(REPUBLISHED_ON_TRANSIT)
    after = repaired[repaired["LapNumber"] >= 8.0]
    assert list(after["TyreLife"]) == [1.0, 2.0]


def test_a_data_loss_sentinel_is_not_read_as_the_same_set():
    """`HARD -> 'None'` is the feed dying, and its age drop must not mark a run.

    Without the `is_real_compound` pair this would null every lap from the moment the
    compound column went blank, which is data loss repairing itself into more data loss.
    """
    dark = _laps(
        [
            (1, 1.0, "HARD", 5.0, False),
            (2, 1.0, "None", 1.0, False),
            (3, 1.0, "None", 2.0, False),
        ]
    )
    repaired, report = repair_tyre_stints(dark)
    assert report.republished_ages_nulled == 0
    pd.testing.assert_frame_equal(repaired, dark)


def test_a_used_same_compound_refit_is_left_alone():
    """The shape that MUST NOT fire, and the reason the rule reads the landing value.

    A same-compound stop is the modal dry strategy of this regulation, and a plain "the
    age dropped" test fires on every one of them: 435 across the shipped 2023-2025
    featured laps, in 53 grands prix, all with `Stint` advancing. Nulling those would
    take 9.7 to 15.2 % of a season's ages, and N15 reads a NaN age as a fresh set, so the
    loss would come back as a number further from the truth than the feed's.

    Requiring the age to land on exactly 1 removes all 435 of them. **That is all it shows.**
    It does NOT show that a real refit never lands on 1: the featured frames drop the
    out-lap, which is the lap a fresh set reads 1 on, so their first visible row reads 2 by
    construction. On raw frames a fresh same-compound refit landing on 1 is ordinary, which
    is what `_entered_the_pits_twice_running` exists for. This fixture is the featured-frame
    shape: HARD 11 laps old replaced by a HARD that has done 2.
    """
    refit = _laps(
        [
            (1, 1.0, "HARD", 10.0, False),
            (2, 1.0, "HARD", 11.0, True),  # a real stop onto another, used, HARD
            (3, 2.0, "HARD", 2.0, False),
            (4, 2.0, "HARD", 3.0, False),
        ]
    )
    repaired, report = repair_tyre_stints(refit)
    assert report.republished_ages_nulled == 0
    pd.testing.assert_frame_equal(repaired, refit)


def test_a_fresh_same_compound_refit_at_a_lone_stop_is_left_alone():
    """The shape the landing-value test alone could NOT tell from the artefact.

    A car that fits a FRESH set of the compound it is already running publishes exactly
    what a republished age publishes: age 1, compound unchanged. The featured parquets
    cannot say how often that happens, because they drop the out-lap and so hold almost
    no age-1 rows at all. What separates it is the second condition: this car made ONE pit
    entry, and a car making a stop is not in the pit lane on two consecutive laps.
    """
    refit = _laps(
        [
            (1, 1.0, "HARD", 10.0, False),
            (2, 1.0, "HARD", 11.0, True),  # a lone entry: a real stop
            (3, 2.0, "HARD", 1.0, False),
            (4, 2.0, "HARD", 2.0, False),
        ]
    )
    repaired, report = repair_tyre_stints(refit)
    assert report.republished_ages_nulled == 0
    pd.testing.assert_frame_equal(repaired, refit)


def test_the_residual_is_a_fresh_refit_made_beside_another_pit_entry():
    """The cost of the rule, asserted rather than left to be discovered.

    Both conditions have to coincide: a fresh set of the compound already on the car, AND
    the car in the pit lane on two consecutive laps. That is what a driver who serves a
    penalty and then pits for real on the next lap looks like, and its age would be nulled.
    The base rate is not measured, because a curated install carries one raw race; it is
    written down as a residual rather than quantified.
    """
    refit = _laps(
        [
            (1, 1.0, "HARD", 10.0, True),  # a penalty served
            (2, 1.0, "HARD", 11.0, True),  # and a real stop on the very next lap
            (3, 2.0, "HARD", 1.0, False),
            (4, 2.0, "HARD", 2.0, False),
        ]
    )
    repaired, report = repair_tyre_stints(refit)
    assert report.republished_ages_nulled == 2
    assert list(repaired["TyreLife"])[:2] == [10.0, 11.0]
    assert repaired["TyreLife"].iloc[2:].isna().all()


def test_the_real_race_the_artefact_was_found_on_comes_back_repaired():
    """Melbourne 2025, the only race a curated install carries. The effect, on real rows.

    The synthetic fixtures above pin the rule; this pins what the rule does to the frame
    that ships. Before #988 the repair reported `changed_anything is False` on this
    race - 0 of 927 rows - while five cars carried an age short by up to four laps for
    the whole of their opening stint.
    """
    from src.f1_strat_manager.data_cache import get_data_root

    frame_path = get_data_root() / "raw" / "2025" / "Melbourne" / "laps.parquet"
    if not frame_path.exists():
        pytest.skip("2025/Melbourne is not in this install's curated data set")
    laps = pd.read_parquet(frame_path)
    assert len(laps) == 927, "the frame this measurement was taken on"

    repaired, report = repair_tyre_stints(laps)
    assert sorted(report.drivers_touched) == ["ALB", "BEA", "LAW", "OCO", "STR"]
    assert report.republished_ages_nulled == 162

    # The transit that corrupts each, and the last lap of the run it poisons.
    for code, drop_lap, end_lap in (
        ("STR", 3, 33),
        ("ALB", 3, 33),
        ("LAW", 4, 33),
        ("OCO", 5, 39),
        ("BEA", 5, 39),
    ):
        rows = repaired[repaired["Driver"] == code].set_index("LapNumber")["TyreLife"]
        assert rows.loc[float(drop_lap - 1)] == float(drop_lap - 1), f"{code} keeps its prefix"
        poisoned = rows.loc[float(drop_lap) : float(end_lap)]
        assert len(poisoned) == end_lap - drop_lap + 1, f"{code}'s run was not empty"
        assert poisoned.isna().all(), f"{code} laps {drop_lap}-{end_lap}"

    # The twelve runners who counted through the same three transits are untouched, and
    # NOR's age is still exactly his lap number to the moment he really stops.
    nor = repaired[repaired["Driver"] == "NOR"].set_index("LapNumber")["TyreLife"]
    assert [nor.loc[float(lap)] for lap in range(1, 35)] == [float(lap) for lap in range(1, 35)]


def test_the_rule_fires_on_nothing_across_the_shipped_seasons():
    """The population the repair actually runs over, not the one race it was found on.

    `augment_featured_laps` calls this repair for EVERY grand prix in the featured frame,
    so the number that matters is not what it does to Melbourne but what it does to the
    other seventy races. An earlier version of the rule triggered on any age drop with the
    compound unchanged, which is what a same-compound pit stop looks like: 435 of them
    across 2023-2025, in 53 grands prix, and it would have nulled 9.7 to 15.2 % of every
    season's ages.

    This asserts the non-firing half on the shipped featured parquets. The count of drops
    examined is pinned first: a scan that found nothing would otherwise pass while proving
    nothing, which is the failure mode this repo has already shipped once.

    **What it does NOT prove, stated so nobody reads more into it.** The featured parquet
    drops the out-lap, so it holds almost no rows with `TyreLife == 1.0` at all: 14 in the
    whole of 2025 against 52 in Melbourne's raw frame alone. A landing-value rule therefore
    could not fire here whatever it did. The population this guard rules out is the one the
    PLAIN drop rule would have hit, and that is worth pinning because that rule was written
    and nearly shipped. The rule that actually ships is pinned against the RAW frame by
    `test_the_real_race_the_artefact_was_found_on_comes_back_repaired` and
    `test_no_genuine_stop_on_the_real_race_is_touched`, and this test SKIPS on CI, which
    carries no data at all.
    """
    from src.f1_strat_manager.data_cache import get_data_root
    from src.f1_strat_manager.tyre_stint_repair import _age_restarted_on_an_unchanged_set

    root = get_data_root()
    examined = 0
    restarted: list[tuple[str, str, int]] = []
    for year in (2023, 2024, 2025):
        path = root / "processed" / f"laps_featured_{year}.parquet"
        if not path.exists():
            continue
        frame = pd.read_parquet(path)
        for (gp, driver), group in frame.groupby(["GP_Name", "Driver"], sort=False):
            ordered = group.sort_values("LapNumber").reset_index(drop=True)
            ages = ordered["TyreLife"]
            for position in range(1, len(ordered)):
                previous, current = ages.iloc[position - 1], ages.iloc[position]
                if pd.isna(previous) or pd.isna(current) or current >= previous:
                    continue
                examined += 1
                if _age_restarted_on_an_unchanged_set(ordered, position):
                    restarted.append((str(gp), str(driver), int(ordered["LapNumber"][position])))
    if not examined:
        pytest.skip("the featured parquets are not in this install's curated data set")
    assert examined > 1000, f"only {examined} age drops examined across three seasons"
    assert not restarted, f"the rule fires on real stops: {restarted[:10]}"


def test_no_genuine_stop_on_the_real_race_is_touched():
    """The other half of the real-frame check: the 24 stops that must survive.

    `test_the_real_race_the_artefact_was_found_on_comes_back_repaired` asserts the five
    runs are nulled. This asserts the complement on the same frame, which is the half a
    too-wide rule would fail: Melbourne 2025 has 29 pit passages that end with an age of 1,
    five of them the artefact and twenty-four genuine compound changes, and no genuine one
    may lose its age.

    Both counts are pinned before anything is compared, because a frame that yielded no
    passages at all would make every assertion below vacuous.
    """
    from src.f1_strat_manager.data_cache import get_data_root

    frame_path = get_data_root() / "raw" / "2025" / "Melbourne" / "laps.parquet"
    if not frame_path.exists():
        pytest.skip("2025/Melbourne is not in this install's curated data set")
    laps = pd.read_parquet(frame_path)
    repaired, _report = repair_tyre_stints(laps)

    artefacts, genuine = [], []
    for driver, group in laps.groupby("Driver", sort=False):
        ordered = group.sort_values("LapNumber").reset_index(drop=True)
        for position in range(1, len(ordered)):
            previous = ordered["TyreLife"].iloc[position - 1]
            current = ordered["TyreLife"].iloc[position]
            if pd.isna(previous) or pd.isna(current) or current != 1.0 or previous <= 1.0:
                continue
            lap = int(ordered["LapNumber"].iloc[position])
            same_compound = (
                ordered["Compound"].iloc[position - 1] == ordered["Compound"].iloc[position]
            )
            (artefacts if same_compound else genuine).append((str(driver), lap))
    assert len(artefacts) == 5, f"expected the five known artefacts, got {artefacts}"
    assert len(genuine) == 24, f"expected 24 genuine stops landing on 1, got {len(genuine)}"

    for driver, lap in genuine:
        rows = repaired[repaired["Driver"] == driver].set_index("LapNumber")["TyreLife"]
        assert rows.loc[float(lap)] == 1.0, f"{driver} lost a real stop's age on lap {lap}"


def test_a_used_same_compound_refit_beside_another_entry_is_left_alone():
    """The fixture that pins the LANDING condition, which the run condition cannot.

    Both halves of the trigger need a shape that isolates them. This is the one for the
    landing value: a real stop onto a USED set of the same compound, made on a lap adjacent
    to another pit entry, so the run condition is satisfied and only the landing value keeps
    the repair away. Dropping `current_age == 1.0` back to a plain drop test nulls this.

    Without it the whole landing condition is pinned only by a data-gated test that skips on
    CI, which carries no data at all.
    """
    refit = _laps(
        [
            (1, 1.0, "HARD", 10.0, True),  # a served penalty
            (2, 1.0, "HARD", 11.0, True),  # a real stop on the very next lap
            (3, 2.0, "HARD", 4.0, False),  # onto a USED HARD: lands on 4, not 1
            (4, 2.0, "HARD", 5.0, False),
        ]
    )
    repaired, report = repair_tyre_stints(refit)
    assert report.republished_ages_nulled == 0
    pd.testing.assert_frame_equal(repaired, refit)


def test_an_age_already_reading_one_is_not_a_restart():
    """`previous_age > 1.0` isolated, on a shape that really happens.

    A car that pits on lap 1 and again on lap 2 reads age 1 on both, and the entries are
    adjacent, so every other condition is satisfied. Las Vegas 2025 BOR is exactly that pair.
    Nothing restarted: the age was already 1, and dropping the `previous_age > 1.0` term
    would null a run on a car whose record is not contradictory at all.
    """
    twice = _laps(
        [
            (1, 1.0, "MEDIUM", 1.0, True),
            (2, 2.0, "MEDIUM", 1.0, True),
            (3, 3.0, "MEDIUM", 1.0, False),
            (4, 3.0, "MEDIUM", 2.0, False),
        ]
    )
    repaired, report = repair_tyre_stints(twice)
    assert report.republished_ages_nulled == 0
    pd.testing.assert_frame_equal(repaired, twice)
