"""N06 is served the columns it was trained on: CompoundID's scale and the previous trap.

Two inputs of the pace agent were a different quantity from the trained column.

`CompoundID` was read from the manifest's `categorical_encoding.Compound` block, which is
0-based. The model's 39 `features_in` include `CompoundID` and not `Compound`, so what it
ate is N01's 1-based column straight from the parquet. Every lap arrived one class low, and
because N01 encodes an unreported compound as 0, a SOFT lap was served as the code that
means "no compound reading".

`Prev_SpeedST` was served `speed_st` — THIS lap's trap — where N04 builds every `Prev_*`
column as one grouped shift within the stint. That is the defect #435 fixed for
`Prev_LapTime` and left in place for its sibling, in the same call.

Measured before the fix, per race: 100% of laps had the wrong CompoundID; the trap differed
on 27 of 27 Lusail laps (mean 6.67 km/h, max 20.0). The prediction itself moved on 16% of
Lusail laps and 7% of Miami's, max 2.89 s and 3.24 s.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).parent.parent.parent
_PROCESSED = ROOT / "data" / "processed"
_FEATURED = _PROCESSED / "laps_featured_2025.parquet"
_RACE = ROOT / "data" / "raw" / "2025" / "Lusail"

# The same artefact pair `test_pace_circuit_speed.py` gates on, so the two files skip and
# run together rather than disagreeing about what "the pace model is present" means.
_HAS_MODEL = (ROOT / "data" / "models" / "lap_time" / "xgb_laptime_delta_final.json").exists() and (
    _PROCESSED / "laps_featured.parquet"
).exists()


# --- the compound scale, re-derived from the artefact -------------------------


@pytest.mark.data
@pytest.mark.skipif(not _FEATURED.exists(), reason="featured parquet absent")
def test_the_declared_compound_codes_are_the_ones_the_parquet_stores():
    """The constant is a claim about the trained column; this re-derives it.

    Deliberately checked against the parquet rather than against N01's source: the parquet
    is what N06 read, so if the two ever disagree it is the parquet that decides.
    """
    from src.agents.pace_agent import _N01_COMPOUND_ID

    pairs = (
        pd.read_parquet(_FEATURED, columns=["Compound", "CompoundID"]).dropna().drop_duplicates()
    )
    # Code 0 is the unknown class and its Compound cell is whatever FastF1 failed to
    # report — the parquet holds both the string 'nan' and the string 'None' against it.
    # Those are not names to check; the named compounds are the ones with a code.
    named = pairs[pairs["CompoundID"] != 0]
    stored = {str(c): int(i) for c, i in zip(named["Compound"], named["CompoundID"])}
    assert stored, "no named compounds in the artefact: this would hold vacuously"

    for compound, code in stored.items():
        assert _N01_COMPOUND_ID[compound] == code, (
            f"{compound}: declared {_N01_COMPOUND_ID[compound]}, artefact stores {code}"
        )


@pytest.mark.data
@pytest.mark.skipif(not _FEATURED.exists(), reason="featured parquet absent")
def test_the_unknown_code_is_a_class_the_model_saw():
    """0 is not a spare slot. N01 does `.fillna(0)`, so unreported compounds trained as 0.

    This is what makes the off-by-one worse than a shift: it collides the SOFT class with
    the absent-reading class instead of merely relabelling it.
    """
    from src.agents.pace_agent import _COMPOUND_ID_UNKNOWN, _N01_COMPOUND_ID

    codes = set(
        pd.read_parquet(_FEATURED, columns=["CompoundID"])["CompoundID"].dropna().astype(int)
    )
    assert _COMPOUND_ID_UNKNOWN in codes, "the artefact never stores the unknown code"
    assert _COMPOUND_ID_UNKNOWN not in _N01_COMPOUND_ID.values(), (
        "the unknown code collides with a named compound"
    )


@pytest.mark.data
@pytest.mark.skipif(not _HAS_MODEL or not _FEATURED.exists(), reason="pace model absent")
def test_an_unrecognised_compound_is_served_as_unknown_not_as_soft():
    """The old default was 1, which is SOFT on the trained scale — a specific tyre."""
    from src.agents.pace_agent import _COMPOUND_ID_UNKNOWN, PaceAgent

    agent = PaceAgent()
    code, _team, _cluster = agent._encode_categorical("NOT_A_COMPOUND", "McLaren", "Lusail")
    assert code == _COMPOUND_ID_UNKNOWN


# --- both inputs, against the trained columns, over a whole race ---------------


@pytest.fixture(scope="module")
def served_vs_trained():
    """Every lap of Lusail 2025 replayed, paired with the featured parquet's own row.

    A whole race rather than a probe row: a hand-built row cannot show that a wrong input
    only moves the prediction in the regions where the trees split on it, and it was a
    single probe that nearly had an earlier fix in this file recorded as cosmetic.
    """
    from src.f1_strat_manager.laps_augment import augment_featured_laps
    from src.simulation.replay_engine import RaceReplayEngine

    featured = augment_featured_laps(pd.read_parquet(_FEATURED), 2025)
    trained = featured[(featured["GP_Name"] == "Lusail") & (featured["Driver"] == "NOR")].set_index(
        "LapNumber"
    )

    replay = RaceReplayEngine(_RACE, driver_code="NOR", team="McLaren", interval_seconds=0.0)
    rows = []
    for lap_state in replay.replay():
        lap = lap_state["driver"].get("lap_number")
        if lap in trained.index:
            rows.append((lap, lap_state["driver"], trained.loc[lap]))
    assert rows, "no laps matched: the pairing is wrong, not the code under test"
    return rows


@pytest.mark.data
@pytest.mark.skipif(
    not _HAS_MODEL or not _FEATURED.exists() or not _RACE.exists(), reason="model or data absent"
)
def test_the_served_compound_id_is_the_trained_one_on_every_lap(served_vs_trained):
    from src.agents.pace_agent import PaceAgent

    agent = PaceAgent()
    mismatches = [
        f"lap {lap}: served {agent._encode_categorical(d.get('compound') or '', '', '')[0]}, "
        f"trained {int(row['CompoundID'])}"
        for lap, d, row in served_vs_trained
        if pd.notna(row.get("CompoundID"))
        and agent._encode_categorical(d.get("compound") or "", "", "")[0] != int(row["CompoundID"])
    ]
    assert mismatches == [], f"{len(mismatches)} laps served the wrong class: {mismatches[:5]}"


@pytest.mark.data
@pytest.mark.skipif(not _FEATURED.exists() or not _RACE.exists(), reason="data absent")
def test_the_producer_emits_the_previous_trap_not_this_lap_s(served_vs_trained):
    """Value AND absence: the stint's first lap has no predecessor and must say so.

    Asserting only the populated laps would pass on a producer that invented a number for
    the openers, which is exactly what the `or 300.0` it replaces did — and 300 km/h sits
    inside the trained range, so the invention was unfalsifiable from the value alone.
    """
    wrong_value, invented = [], []
    for lap, d, row in served_vs_trained:
        served, trained = d.get("prev_speed_st"), row.get("Prev_SpeedST")
        if pd.notna(trained):
            if served is None or abs(float(served) - float(trained)) > 1e-6:
                wrong_value.append(f"lap {lap}: served {served}, trained {trained}")
        elif served is not None:
            invented.append(f"lap {lap}: served {served} where training had none")

    assert wrong_value == [], f"{len(wrong_value)} laps differ: {wrong_value[:5]}"
    assert invented == [], f"{len(invented)} laps invented a reading: {invented[:5]}"


@pytest.mark.data
@pytest.mark.skipif(not _FEATURED.exists() or not _RACE.exists(), reason="data absent")
def test_the_previous_trap_is_not_simply_the_current_one(served_vs_trained):
    """Guards against a producer that satisfies the test above by aliasing the columns.

    If `prev_speed_st` were `speed_st` again, the assertions above would still hold on any
    lap where the two happen to coincide, so this asserts the two genuinely differ.
    """
    differing = sum(
        1
        for _lap, d, _row in served_vs_trained
        if d.get("prev_speed_st") is not None
        and d.get("speed_st") is not None
        and abs(float(d["prev_speed_st"]) - float(d["speed_st"])) > 1e-6
    )
    assert differing > 0, "prev_speed_st never differs from speed_st: it is the same column"
