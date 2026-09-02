"""#782 — the weather restore must reproduce N04, not improve on it.

The models were trained on whatever N04's merge produced. A "better" alignment here would
feed 2025 weather on a different basis than the training data — a silent distribution
shift wearing a fix's clothes. So the load-bearing test is not that weather appears; it is
that the same method reproduces the seasons where the published truth still exists.

Two checks, and they see different things. The pace holdout in `tests/eval/` reproduces
its published MAE of 0.4104 to within 0.0007, which proves the VALUES are right at the
distribution level -- all-NaN weather moves that MAE by 0.26. It does NOT see the
alignment: a wrong backward join moves it by only 0.0003 and still passes. The last test
in this file closes that, by comparing against N04's own published 2025 output.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.f1_strat_manager.weather_restore import (
    WEATHER_COLUMNS,
    normalise_rainfall,
    weather_for_race,
)

ROOT = Path(__file__).parent.parent.parent


def _weather(samples):
    """A weather.parquet shape: (session seconds, air, track, humidity, rainfall)."""
    return pd.DataFrame(
        [
            {
                "Time": pd.Timedelta(seconds=t),
                "AirTemp": air,
                "TrackTemp": track,
                "Humidity": hum,
                "Rainfall": rain,
            }
            for t, air, track, hum, rain in samples
        ]
    )


def _laps(times):
    """A raw laps shape: one row per lap, carrying the session-elapsed Time."""
    return pd.DataFrame(
        {
            "Driver": ["NOR"] * len(times),
            "LapNumber": [float(i + 1) for i in range(len(times))],
            "Time": [pd.Timedelta(seconds=t) for t in times],
        }
    )


def test_each_lap_takes_its_nearest_sample_not_the_preceding_one():
    """N04 joins with direction='nearest'. A backward join would shift every reading.

    The lap at 100 s sits between samples at 60 s and 120 s and is closer to the later
    one, so 'nearest' gives 30.0 where 'backward' would give 20.0.
    """
    weather = _weather([(0, 10.0, 20.0, 50.0, False), (120, 15.0, 30.0, 60.0, False)])
    laps = _laps([100])
    aligned = weather_for_race(laps, weather)
    assert aligned["TrackTemp"].iloc[0] == 30.0
    assert aligned["AirTemp"].iloc[0] == 15.0


def test_laps_keep_their_own_index_after_the_time_sort():
    """merge_asof requires both sides sorted and does not preserve the index.

    Reattaching it wrong would scramble weather across drivers — every lap would get
    somebody else's reading, silently and plausibly.
    """
    weather = _weather([(0, 10.0, 20.0, 50.0, False), (100, 99.0, 88.0, 77.0, False)])
    laps = _laps([100, 0])  # deliberately out of time order
    aligned = weather_for_race(laps, weather)
    assert aligned.index.equals(laps.index)
    assert aligned["TrackTemp"].iloc[0] == 88.0  # lap 1 is at t=100
    assert aligned["TrackTemp"].iloc[1] == 20.0  # lap 2 is at t=0


def test_a_race_with_no_weather_samples_yields_all_unknown():
    aligned = weather_for_race(_laps([0, 60]), _weather([]))
    assert aligned[list(WEATHER_COLUMNS)].isna().all().all()


def test_a_lap_with_no_session_time_stays_unknown():
    laps = _laps([0, 60])
    laps.loc[1, "Time"] = pd.NaT
    aligned = weather_for_race(laps, _weather([(0, 10.0, 20.0, 50.0, False)]))
    assert aligned["TrackTemp"].iloc[0] == 20.0
    assert pd.isna(aligned["TrackTemp"].iloc[1])


def test_rainfall_becomes_N04s_integer_flag():
    """N04's closing step: absent rainfall means dry, and the flag is an int not a bool."""
    frame = pd.DataFrame({"Rainfall": [True, False, None]})
    result = normalise_rainfall(frame)
    assert result["Rainfall"].tolist() == [1, 0, 0]
    assert result["Rainfall"].dtype.kind == "i"


def test_rainfall_is_filled_per_season_not_per_race():
    """A frame without the column is returned untouched rather than gaining a fake one.

    The fill runs once over the season, as N04 does it: filling per race would give a
    race with no weather parquet a confident 0 (dry) instead of an honest gap.
    """
    frame = pd.DataFrame({"AirTemp": [20.0]})
    pd.testing.assert_frame_equal(normalise_rainfall(frame), frame)


# ---------------------------------------------------------------------------
# The real safeguard: N04's own 2025 output is on disk, so the restore is
# checkable against ground truth rather than against its own reasoning.
# ---------------------------------------------------------------------------

_COMBINED = ROOT / "data" / "processed" / "laps_featured.parquet"
_PER_YEAR_2025 = ROOT / "data" / "processed" / "laps_featured_2025.parquet"

needs_artefacts = pytest.mark.skipif(
    not (_COMBINED.exists() and _PER_YEAR_2025.exists()),
    reason="featured laps artefacts absent (CI runner without data)",
)


@needs_artefacts
def test_the_restore_reproduces_N04s_own_2025_output_exactly():
    """The check the module's docstring promises, against real published values.

    The artefact now carries the four weather columns natively, which is what the
    regeneration restored — so the ground truth is the file itself, and the restore is run
    against a copy with those columns REMOVED.

    That stripping is the whole test. An adversarial gate found this had gone vacuous the
    moment the artefact gained the columns: `augment_featured_laps` declines when any of
    the four is already present, so it was comparing the file with itself and passed even
    with the restore poisoned. The suite's only alignment-versus-truth check had died
    silently, which is precisely the failure it exists to catch, one level up.

    And it is the test the pace-MAE reproduction CANNOT replace: an adversarial gate showed
    a wrong `direction='backward'` join changes 7,014 TrackTemp cells and still reproduces
    the published MAE to within 0.0003. The MAE sees the distribution, not the alignment.
    This sees the alignment.
    """
    from src.f1_strat_manager.laps_augment import augment_featured_laps

    truth = pd.read_parquet(_PER_YEAR_2025)
    assert not [c for c in WEATHER_COLUMNS if c not in truth.columns], (
        "the artefact carries no weather, so there is nothing to compare the restore "
        "against and this test would hold vacuously"
    )

    # The guard above names the WRONG precondition. What the restore reads is the RAW
    # tree, one directory per GP, and the curated download ships a single race - so on a
    # clean install the restore correctly returns NaN for the other 23 and this test
    # reported "22,197 of 22,760 laps disagree with N04", a number about missing files
    # and not about alignment. Scope to the races whose raw laps are actually here.
    available = sorted(
        {path.name for path in (ROOT / "data" / "raw" / "2025").glob("*") if path.is_dir()}
        & set(truth["GP_Name"].dropna().unique())
    )
    if not available:
        pytest.skip("no 2025 raw race directories present; the restore has nothing to read")

    stripped = truth.drop(columns=list(WEATHER_COLUMNS))
    restored = augment_featured_laps(stripped, 2025)
    restored = restored[restored["GP_Name"].isin(available)]
    truth = truth[truth["GP_Name"].isin(available)]

    keys = ["GP_Name", "Driver", "LapNumber"]
    joined = restored[[*keys, *WEATHER_COLUMNS]].merge(
        truth[[*keys, *WEATHER_COLUMNS]], on=keys, suffixes=("_mine", "_n04")
    )
    assert len(joined) == len(restored), "every restored lap must find its published twin"
    # Vacuity is the failure this test exists to catch (see the docstring): a restore that
    # produced nothing at all would otherwise compare an empty frame and pass.
    assert joined["TrackTemp_mine"].notna().any(), (
        f"the restore returned no weather for {available} even though their raw laps are "
        "present; comparing this would hold vacuously"
    )

    for column in WEATHER_COLUMNS:
        mine, published = joined[f"{column}_mine"], joined[f"{column}_n04"]
        both_absent = mine.isna() & published.isna()
        identical = (mine.astype("float64") - published.astype("float64")).abs() < 1e-9
        mismatched = int((~(both_absent | identical)).sum())
        assert mismatched == 0, f"{column}: {mismatched} of {len(joined)} laps disagree with N04"


@needs_artefacts
def test_a_partial_weather_set_is_declined_rather_than_merged_into_suffix_columns():
    """A frame carrying SOME of the four must not be half-restored.

    With an `all(...)` guard the per-race slice carries all four names and the left-merge
    collides with the ones already present: `TrackTemp` is replaced by a
    `TrackTemp_x`/`TrackTemp_y` pair that every consumer selecting the plain name then
    fails to find. Declining is the safe answer, and it is what the `any(...)` guard does.
    """
    from src.f1_strat_manager.laps_augment import augment_featured_laps

    # DROPPING three of the four, not assigning one. This test used to build its partial
    # frame with `.assign(AirTemp=20.0)` on an artefact that carried none of them, which
    # made it partial by accident of the era: once the regenerated artefact natively carries
    # all four, that same line produces a COMPLETE frame and the assertion below fails
    # against a file that is more correct than before. Constructing the partial state from
    # whatever the artefact happens to hold keeps the test about the guard rather than
    # about the vintage of the data underneath it.
    partial = pd.read_parquet(_PER_YEAR_2025)
    for column in ("TrackTemp", "Humidity", "Rainfall"):
        if column in partial.columns:
            partial = partial.drop(columns=[column])
    if "AirTemp" not in partial.columns:
        partial = partial.assign(AirTemp=20.0)
    partial["AirTemp"] = 20.0  # one of four present, three missing

    result = augment_featured_laps(partial, 2025)

    assert not [c for c in result.columns if c.endswith(("_x", "_y"))]
    assert "TrackTemp" not in result.columns, "declined, not half-merged"
    assert (result["AirTemp"] == 20.0).all(), "the column it did carry is untouched"
