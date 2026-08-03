"""Restore the weather columns the 2025 featured parquet was published without (#782).

`laps_featured_2023/2024.parquet` carry 53 columns including `AirTemp`, `TrackTemp`,
`Humidity` and `Rainfall`. `laps_featured_2025.parquet` carries 48 and none of them, which
breaks two golden reproduction tests (`test_pace_mae_reproduces_from_featured_laps`,
`test_reproduction_matches_overtake_auc_pr`) with a `KeyError` inside
`src/strategy/eval/reproduce.py`'s own feature rebuild.

**The data is not lost.** Every one of the 71 shipped race directories has a readable
`weather.parquet` with all four readings and zero NaN temperature rows -- 2025 included.
What is missing is only the merge into the featured frame, so this restores it at load
time rather than asking anyone to regenerate a published artefact.

--- WHERE TO CHANGE IF THE ARTEFACT CHANGES ---
Called from `laps_augment.augment_featured_laps`, for the reasons that module documents:
the featured parquet is republished from Hugging Face and pulled by
`scripts/download_data.py`, so a locally patched file would be silently reverted, and its
only producer is a read-only notebook (N04).

## This reproduces N04, it does not improve on N04

The models were trained on whatever N04 produced for 2023 and 2024, so a "better" join
here would feed 2025 weather on a different basis than the training data -- a silent
distribution shift dressed up as a fix. N04 Step 5 states its method and this mirrors it
exactly: per race, `pd.merge_asof(direction="nearest")` on the session `Time` timedelta,
with `Rainfall` filled to 0 and cast to an int flag.

N04's own 2025 output is on disk to check against: `laps_featured.parquet`, the combined
2023-2025 artefact, carries all four columns for 2025. Only the per-year split dropped
them. So the safeguard is a direct comparison against ground truth rather than against
this module's own reasoning, and it is committed --
`tests/agents/test_weather_restore.py::test_the_restore_reproduces_N04s_own_2025_output_exactly`
asserts all 22,760 laps match.

That test is not interchangeable with the pace-MAE reproduction, and an adversarial gate
proved why: a WRONG `direction="backward"` join changes 7,014 TrackTemp cells and still
reproduces the published MAE to within 0.0003. The MAE sees the values' distribution; only
the comparison above sees the alignment.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# The four N04 Step 5 features, in N04's own order.
WEATHER_COLUMNS = ("AirTemp", "TrackTemp", "Humidity", "Rainfall")

# The session-elapsed column both sides join on. It is a timedelta in the raw parquets.
_TIME = "Time"


def weather_for_race(raw_laps: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
    """Per-lap weather for one race, aligned exactly as N04 aligned it.

    Args:
        raw_laps: That race's raw laps frame; only its ``Time`` column is read.
        weather: That race's ``weather.parquet``.

    Returns:
        A frame indexed like ``raw_laps`` carrying the four weather columns. Rows whose
        lap has no session time keep NaN, which is what N04 leaves them as.
    """
    columns = [c for c in WEATHER_COLUMNS if c in weather.columns]
    empty = pd.DataFrame(
        {c: pd.Series(index=raw_laps.index, dtype="float64") for c in WEATHER_COLUMNS}
    )
    if not columns or _TIME not in weather.columns or _TIME not in raw_laps.columns:
        return empty

    samples = weather[[_TIME, *columns]].dropna(subset=[_TIME]).sort_values(_TIME)
    laps = raw_laps[[_TIME]].dropna(subset=[_TIME]).sort_values(_TIME)
    if samples.empty or laps.empty:
        return empty

    # merge_asof demands both sides be sorted on the key, hence the sorts above; the
    # index is reattached afterwards because merge_asof does not preserve it.
    merged = pd.merge_asof(laps, samples, on=_TIME, direction="nearest")
    merged.index = laps.index

    # Each column is reindexed in its own dtype rather than written into a pre-made
    # float64 frame: `Rainfall` arrives as bool, and assigning it into a float column
    # is the deprecated incompatible-dtype set that pandas now warns about.
    aligned = {}
    for column in WEATHER_COLUMNS:
        if column in columns:
            aligned[column] = merged[column].reindex(raw_laps.index)
        else:
            aligned[column] = pd.Series(index=raw_laps.index, dtype="float64")
    return pd.DataFrame(aligned)


def normalise_rainfall(featured: pd.DataFrame) -> pd.DataFrame:
    """Apply N04's closing step: absent rainfall means dry, and the flag is an int.

    Season-level rather than per-race because that is where N04 does it, and the placement
    is not neutral: a race whose weather parquet is missing keeps NaN temperatures (an
    honest gap) but still ends up with `Rainfall == 0`, a confident "dry" nobody measured.
    That asymmetry is N04's, inherited deliberately so the column matches what the models
    were trained on rather than being quietly improved here.
    """
    if "Rainfall" not in featured.columns:
        return featured
    result = featured.copy()
    result["Rainfall"] = result["Rainfall"].fillna(0).astype(int)
    return result


def read_race_weather(race_dir: Path) -> pd.DataFrame | None:
    """That race's ``weather.parquet``, or None when it is absent or unreadable."""
    path = race_dir / "weather.parquet"
    if not path.exists():
        return None
    try:
        return pd.read_parquet(path)
    except (OSError, ValueError) as exc:
        # Same degrade-quietly contract the raw-laps merge upstream uses: a race whose
        # weather cannot be read keeps NaN readings rather than failing the whole load.
        logger.warning("%s: cannot read weather parquet (%s); continuing without it", path, exc)
        return None
