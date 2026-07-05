"""Cover-first CLI smoke for the ``--no-llm`` path (#166, Testing audit #180).

The ``--no-llm`` mode of the PMV (``f1-sim``) has been broken since 2026-05-09
(a 3-tuple return the CLI consumer never adopted). This subprocess smoke is the
*executable* debt for that bug: it is expected to fail (``xfail``) until the P2b
shared engine lands and the CLI no-LLM path is fixed. It runs only where the
Melbourne 2025 data is present (``data`` tier), so CI stays green.

``scripts/run_simulation_cli.py`` is UNTOUCHABLE — this test only drives it as a
black-box subprocess, never imports or edits it.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
_PARQUET = ROOT / "data" / "processed" / "laps_featured_2025.parquet"
_RACE_DIR = ROOT / "data" / "raw" / "2025" / "Melbourne"
_HAS_DATA = _PARQUET.exists() and _RACE_DIR.exists()


@pytest.mark.data
@pytest.mark.skipif(not _HAS_DATA, reason="Melbourne 2025 parquet + race dir required")
@pytest.mark.xfail(
    reason="--no-llm broken since 2026-05-09 (#166); fixed with the P2b shared engine",
    strict=False,
)
def test_cli_no_llm_smoke():
    """`f1-sim Melbourne NOR McLaren --no-llm --laps 5-7` must exit 0 with no [ERROR]."""
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_simulation_cli.py",
            "Melbourne",
            "NOR",
            "McLaren",
            "--no-llm",
            "--no-real-radios",
            "--laps",
            "5-7",
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=600,
    )
    combined = proc.stdout + proc.stderr
    assert proc.returncode == 0, f"exit {proc.returncode}\n{combined[-2000:]}"
    assert "[ERROR]" not in combined, "no-LLM run logged [ERROR]"
