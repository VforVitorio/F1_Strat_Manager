"""#794 — the tracked files under `data/` must still be on disk.

On 2026-08-02, 36 of them vanished from a working tree. Nothing had committed the
deletion, so `git restore data/` recovered everything and nothing was lost — but it was
**silent**, and a `git add -A` at the wrong moment would have committed the loss of the
measured tables that back published numbers (`mc_measured_v1.json`, the threshold sweeps,
the pace baselines).

The cause was never determined; #794 records what was ruled out. This test does not
prevent the deletion. It turns a silent one into a red build, which is the difference
between finding out and not.

Why it is worth having despite not fixing anything: `data/.gitignore` covers `/raw`,
`/processed` and `/models`, so the files it does NOT cover sit in the one directory every
download path treats as its own — including `huggingface_hub`, which
`data_cache.py:285-290` hands the repository root as a managed directory. Git owns these
files; the tooling believes it does.

The list is derived from git rather than hardcoded, so adding or removing a tracked data
file needs no edit here and this can never drift into asserting about a stale set.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent.parent


def _tracked_data_files() -> list[str]:
    """Paths git tracks under `data/`, straight from the index."""
    result = subprocess.run(
        ["git", "ls-files", "data/"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return []
    return [line for line in result.stdout.splitlines() if line.strip()]


@pytest.mark.skipif(not (ROOT / ".git").exists(), reason="not a git checkout")
def test_every_tracked_data_file_is_still_on_disk():
    """A tracked file missing from the worktree is a deletion nobody asked for."""
    tracked = _tracked_data_files()
    if not tracked:
        pytest.skip("git ls-files returned nothing (shallow or non-git environment)")

    missing = [path for path in tracked if not (ROOT / path).exists()]

    assert not missing, (
        f"{len(missing)} of {len(tracked)} tracked files under data/ are missing from the "
        f"working tree. Nothing committed their deletion, so `git restore data/` recovers "
        f"them. See #794 for what was ruled out. Missing: {sorted(missing)[:8]}"
    )
