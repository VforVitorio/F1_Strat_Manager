"""Regenerate ``mini_race.parquet`` — a tiny, deterministic race slice for tests.

Slices the raw FastF1 laps parquet of the 2025 Qatar GP (Lusail) down to a
9-lap window over 6 drivers that straddles the Safety Car (TrackStatus "4" on
laps 7-10). This is the schema ``RaceStateManager`` consumes natively (raw laps:
``LapTime`` as timedelta, ``TrackStatus``, ``PitInTime`` ...), and the SC window
lets the fixture exercise the SC-override path the thesis case study (Qatar V7)
depends on.

The source (``data/raw/2025/Lusail/laps.parquet``) is a Hugging Face asset, not in
git; the sliced artifact IS committed (< 150 KB, the tests/fixtures carve-out).
Regenerate with:

    python tests/fixtures/generate_mini_race.py

Deterministic: fixed GP, lap window, driver set and column order, so re-running
yields a byte-stable file unless the upstream data changes.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

# --- fixture definition (the knobs that make this slice reproducible) --------
SOURCE_GP = "Lusail"  # 2025 Qatar GP — has the Safety Car case study
YEAR = 2025
LAP_LOW, LAP_HIGH = 5, 13  # 9 laps, brackets the SC (laps 7-10)
DRIVERS = ["VER", "GAS", "ANT", "ALO", "LEC", "STR"]  # 6 cars, full window coverage


def _repo_root() -> Path:
    root = Path(__file__).resolve()
    while not (root / ".git").exists():
        if root.parent == root:
            raise RuntimeError("repo root (.git) not found above this file")
        root = root.parent
    return root


def build_mini_race() -> pd.DataFrame:
    """Return the sliced, driver-filtered, lap-windowed race DataFrame."""
    source = _repo_root() / "data" / "raw" / str(YEAR) / SOURCE_GP / "laps.parquet"
    if not source.exists():
        raise FileNotFoundError(
            f"Source laps parquet not found: {source}. "
            "Pull the Hugging Face data assets first (see CLAUDE.md §0.3)."
        )
    df = pd.read_parquet(source)

    in_window = df["LapNumber"].between(LAP_LOW, LAP_HIGH)
    is_selected_driver = df["Driver"].isin(DRIVERS)
    mini = df[in_window & is_selected_driver].copy()

    # Stable ordering so the parquet is byte-reproducible across regenerations.
    mini = mini.sort_values(["LapNumber", "Driver"]).reset_index(drop=True)
    return mini


def main() -> None:
    mini = build_mini_race()
    out = Path(__file__).parent / "mini_race.parquet"
    mini.to_parquet(out, index=False)
    size_kb = out.stat().st_size / 1024
    print(
        f"wrote {out.name}: {len(mini)} rows, {mini['Driver'].nunique()} drivers, "
        f"laps {LAP_LOW}-{LAP_HIGH}, {size_kb:.1f} KB"
    )
    if size_kb > 150:
        raise SystemExit(f"mini_race.parquet is {size_kb:.1f} KB (> 150 KB budget)")


if __name__ == "__main__":
    main()
