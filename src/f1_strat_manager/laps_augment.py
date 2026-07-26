"""Restore onto the featured laps frame the columns its producer drops.

**Every consumer of `laps_featured_<year>.parquet` must call `augment_featured_laps`.**
Reading the parquet directly is a bug, and it has been one three times now.

N04 drops `Time` (session elapsed) with a comment saying it is "already converted to
`*_s`". It is not: there is no `Time_s` in any published featured parquet. But N11 trains
its overtake gap as `abs(row_x["Time_s"] - row_y["Time_s"])`, so without it the agent
falls back to a single lap's LapTime delta, and two cars 20 s apart lapping within 0.5 s
of each other read as "in the DRS window".

Measured on Lusail 2024, 629 position-adjacent pairs:

    mean gap fed to N12: 0.453 s   |  truth: 3.113 s
    "in DRS window":     91.1%     |  truth: 20.5%
    worst pair:  BOT vs LAW L27 -> model told 0.130 s, real gap 21.066 s

This lives in the PARENT package on purpose. It used to live in the telemetry backend's
loader, which meant the backend was fixed and every other consumer was not: the CLI
(`f1-sim`, the TFG's PMV) read the parquet straight from disk and shipped the degraded
gap on 100% of calls. The data layer belongs to whoever owns the data, and the submodule
consumes it: not the other way round.

`PitInTime` is deliberately NOT restored, though raw has it: a car only carries one on
the lap it actually pits, and those are precisely the laps N04's `IsAccurate` filter
drops. Measured: 44/1067 raw Lusail laps have one and 0% survive into featured, so a
merge could only ever restore nulls. Deriving `active_pitstop_count` from `Stint` /
`TyreLife` resets is the real answer; faking it with a structurally-empty column is not.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, Dict, Optional

import pandas as pd

logger = logging.getLogger(__name__)

# Raw column -> the name the models were trained on.
RAW_COLUMNS_TO_RESTORE: Dict[str, str] = {
    "Time": "Time_s",  # timedelta -> seconds, matching the trained feature
    "TrackStatus": "TrackStatus",
}

_JOIN_KEYS = ["GP_Name", "Driver", "LapNumber"]

# Inverted from gp_slugs.FOLDER_ALIASES (keyed folder -> friendly) so the one place that
# owns circuit renames stays the one place. Reversing at import keeps this from becoming
# a second, drifting copy.
try:
    from src.f1_strat_manager.gp_slugs import FOLDER_ALIASES as _FOLDER_ALIASES

    _FRIENDLY_TO_FOLDER: Dict[str, str] = {v: k for k, v in _FOLDER_ALIASES.items()}
except ImportError as exc:  # pragma: no cover
    # gp_slugs failed to import, or FOLDER_ALIASES was renamed/removed there
    # (a `from X import NAME` raises ImportError for both). FOLDER_ALIASES is
    # a plain dict literal in gp_slugs.py, so the dict-comprehension line
    # below cannot itself raise anything else. This falls back to a single
    # stale entry, so it is worth a warning rather than a fully silent degrade.
    logger.warning(
        "Could not load FOLDER_ALIASES from gp_slugs (%s); falling back to a "
        "single hardcoded Miami mapping. _raw_race_dir may misresolve other "
        "renamed circuits.",
        exc,
    )
    _FRIENDLY_TO_FOLDER = {"Miami": "Miami_Gardens"}


def _default_data_root() -> Path:
    """The repo's data root, resolved the way the rest of the parent package does."""
    from src.f1_strat_manager.data_cache import get_data_root

    return get_data_root()


def _raw_race_dir(data_root: Path, year: int, gp_name: str) -> Path:
    """Raw per-race directory for a featured `GP_Name`.

    Three forms have to be tried, and the third is not optional: the raw dirs mostly
    match `GP_Name` exactly, the underscore variant covers the space-vs-underscore forms
    FastF1 emits (`Marina Bay` -> `Marina_Bay`), and a small number of circuits were
    simply renamed on disk (`Miami` -> `Miami_Gardens`). Skipping the rename cost Miami
    its whole augmentation on the first pass, 3.8% of the season silently unfixed, which
    is exactly the silent-miss class this work is about.
    """
    base = data_root / "raw" / str(year)
    for candidate in (
        base / gp_name,
        base / gp_name.replace(" ", "_"),
        base / _FRIENDLY_TO_FOLDER.get(gp_name, gp_name),
    ):
        if candidate.exists():
            return candidate
    return base / gp_name


def augment_featured_laps(
    df: pd.DataFrame,
    year: int,
    data_root: Optional[Path] = None,
    root_resolver: Optional[Callable[[], Path]] = None,
) -> pd.DataFrame:
    """Merge `Time_s` / `TrackStatus` onto a featured laps frame, from the raw parquets.

    --- WHERE TO CHANGE IF THE ARTEFACT CHANGES ---
    This runs at LOAD time on purpose, not as a rewrite of the parquet. The featured
    parquet is published on Hugging Face and pulled by `scripts/download_data.py`, so a
    locally-patched file would be silently reverted by the next download; and its only
    producer is a read-only notebook (N04). Merging here is immune to both: nothing to
    re-upload, no divergence from the published dataset, no fork of the pipeline.

    The join is safe: `(GP_Name, Driver, LapNumber)` is unique in the featured frame and
    every featured row is a subset of raw. A race whose raw parquet is absent is skipped
    with a warning rather than failing: the agents' existing fallbacks then behave as
    they did before, which is the old status quo rather than a new failure.

    Args:
        df: A featured laps frame, as read from `laps_featured_<year>.parquet`.
        year: The season, used to locate the raw per-race directories.
        data_root: Data root override. The telemetry backend passes its own resolver's
            answer, since it honours `$F1_STRAT_DATA_ROOT` and a different `.git` walk.
        root_resolver: Callable form of the same, resolved lazily.

    Returns:
        The frame with `Time_s` and `TrackStatus` merged on, or unchanged when the raw
        parquets are unavailable.
    """
    if "GP_Name" not in df.columns:
        return df
    if "Time_s" in df.columns:
        return df  # already augmented; do not merge twice

    root = data_root or (root_resolver() if root_resolver else _default_data_root())

    frames = []
    missing: list[str] = []
    for gp_name in df["GP_Name"].dropna().unique():
        path = _raw_race_dir(root, year, str(gp_name)) / "laps.parquet"
        if not path.exists():
            missing.append(str(gp_name))
            continue
        raw = pd.read_parquet(path)
        if not set(RAW_COLUMNS_TO_RESTORE).issubset(raw.columns):
            missing.append(str(gp_name))
            continue
        slice_ = raw[["Driver", "LapNumber", *RAW_COLUMNS_TO_RESTORE]].copy()
        slice_["GP_Name"] = gp_name
        # `Time` is a session-elapsed timedelta; the models were trained on seconds.
        slice_["Time"] = pd.to_timedelta(slice_["Time"]).dt.total_seconds()
        frames.append(slice_.rename(columns=RAW_COLUMNS_TO_RESTORE))

    if missing:
        logger.warning(
            "Raw laps unavailable for %d GP(s) in %d: %s: their laps keep the "
            "degraded behaviour (the overtake gap falls back to a lap-time delta)",
            len(missing),
            year,
            sorted(missing),
        )
    if not frames:
        return df

    augmented = df.merge(pd.concat(frames, ignore_index=True), on=_JOIN_KEYS, how="left")
    restored = int(augmented["Time_s"].notna().sum())
    logger.info(
        "Restored Time_s/TrackStatus onto %d/%d laps of %d from the raw parquets",
        restored,
        len(augmented),
        year,
    )
    return augmented
