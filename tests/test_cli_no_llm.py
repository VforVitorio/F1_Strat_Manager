"""Cover-first CLI smoke for the ``--no-llm`` path (#166, Testing audit #180).

The ``--no-llm`` mode of the PMV (``f1-sim``) was broken from 2026-05-09 to #236
(a 3-tuple return the CLI consumer never adopted). #236 wired the CLI to the P2b
shared engine (``run_lap(profile="no-llm")``), which fixes the crash by
construction. This subprocess smoke is the executable regression net: it must now
exit 0 with no ``[ERROR]`` row. It runs only where the Melbourne 2025 data is
present (``data`` tier), so CI stays green.
"""

from __future__ import annotations

import os
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
        # Both halves of this are needed, and only one of them was here.
        #
        # PYTHONUTF8 tells the CHILD to emit UTF-8: capture_output pipes stdout,
        # so on Windows the Rich header's non-ASCII glyphs would otherwise hit
        # the cp1252 codec and crash for reasons unrelated to this test.
        #
        # encoding tells the PARENT how to read it back. Without it, text=True
        # decodes using the parent's locale, which on Windows is cp1252, and the
        # decode raises inside subprocess. The visible symptom is not an error:
        # returncode comes back 0 and proc.stdout comes back None, so the test
        # died on `None + str` while the run underneath had actually succeeded.
        encoding="utf-8",
        errors="replace",
        env={**os.environ, "PYTHONUTF8": "1"},
    )
    assert proc.stdout is not None, "the CLI produced no capturable stdout"
    combined = proc.stdout + proc.stderr
    assert proc.returncode == 0, f"exit {proc.returncode}\n{combined[-2000:]}"
    assert "[ERROR]" not in combined, "no-LLM run logged [ERROR]"
