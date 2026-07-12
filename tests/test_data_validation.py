"""Hermetic tests for the load-boundary data validation (F-02, #244).

Pins that a missing required column fails loudly at ingestion (naming the
artifact + column) instead of surfacing as a silent ``None`` in ``lap_state``,
and that the FastF1 quality flags only warn.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.simulation.data_validation import (
    REQUIRED_LAPS_COLUMNS,
    DataValidationError,
    validate_laps_df,
    warn_low_quality_laps,
)


def _valid_laps() -> pd.DataFrame:
    """A minimal 2-driver x 2-lap frame carrying every required column."""
    return pd.DataFrame(
        {
            "Driver": ["NOR", "VER", "NOR", "VER"],
            "LapNumber": [1, 1, 2, 2],
            "LapTime": [90.1, 90.3, 89.9, 90.0],
            "Time": [90.1, 90.3, 180.0, 180.3],
            "Position": [1, 2, 1, 2],
            "Compound": ["SOFT", "SOFT", "MEDIUM", "MEDIUM"],
            "TyreLife": [1, 1, 2, 2],
            "TrackStatus": ["1", "1", "1", "1"],
        }
    )


def test_valid_laps_pass():
    validate_laps_df(_valid_laps(), source="test")  # must not raise


@pytest.mark.parametrize("col", REQUIRED_LAPS_COLUMNS)
def test_missing_column_raises_and_names_it(col):
    df = _valid_laps().drop(columns=[col])
    with pytest.raises(DataValidationError) as exc:
        validate_laps_df(df, source="Melbourne 2025 laps.parquet")
    assert col in str(exc.value)
    assert "Melbourne 2025 laps.parquet" in str(exc.value)


def test_empty_laps_raises():
    with pytest.raises(DataValidationError):
        validate_laps_df(_valid_laps().iloc[0:0], source="test")


def test_quality_flags_warn(capsys):
    df = _valid_laps()
    df["IsAccurate"] = [True, False, True, True]
    df["Deleted"] = [False, True, False, False]
    warn_low_quality_laps(df, source="race")
    err = capsys.readouterr().err
    assert "IsAccurate=False" in err
    assert "Deleted" in err


def test_quality_flags_silent_when_clean(capsys):
    df = _valid_laps()
    df["IsAccurate"] = [True, True, True, True]
    df["Deleted"] = [False, False, False, False]
    warn_low_quality_laps(df, source="race")
    assert capsys.readouterr().err == ""


def test_rsm_rejects_missing_required_column():
    """RaceStateManager enforces the schema from any construction path."""
    from src.simulation.race_state_manager import RaceStateManager

    df = _valid_laps().drop(columns=["Compound"])
    with pytest.raises(DataValidationError):
        RaceStateManager(laps_df=df, driver_code="NOR", team="McLaren", gp_name="Test", year=2025)


def test_valid_frame_constructs_race_state_manager():
    """A frame that passes validation actually builds a RaceStateManager.

    Guards the gap Fable found: the required-column list must be a SUPERSET of
    what RSM indexes unconditionally (Time/TrackStatus included), so a "valid"
    frame never passes validation only to crash in construction.
    """
    from src.simulation.race_state_manager import RaceStateManager

    rsm = RaceStateManager(
        laps_df=_valid_laps(), driver_code="NOR", team="McLaren", gp_name="Test", year=2025
    )
    assert rsm.total_laps == 2


def test_all_nan_lapnumber_raises():
    """A present-but-all-NaN LapNumber is caught, not left to crash int(max())."""
    import numpy as np

    df = _valid_laps()
    df["LapNumber"] = np.nan
    with pytest.raises(DataValidationError):
        validate_laps_df(df, source="test")


def test_quality_flags_never_raise_on_bad_dtype():
    """A hostile quality-flag dtype (strings) must be swallowed, never raised."""
    df = _valid_laps()
    df["IsAccurate"] = ["True", "False", "True", "False"]  # strings, not bools
    warn_low_quality_laps(df, source="race")  # must return without raising
