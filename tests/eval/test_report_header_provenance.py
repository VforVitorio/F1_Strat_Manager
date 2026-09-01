"""Every committed report's provenance header is what the renderer emits today.

The header is generated, so it is the one part of a report nobody should edit by
hand. It was edited by hand: the writing pass that removed em dashes from the
documentation tree reached into ``documents/eval_reports/`` and rewrote the
renderer's ``artifacts`` placeholder in four reports without touching the
renderer. The next regeneration put the em dash back in a fifth, so the tree held
two spellings of the same empty value and neither traced to the code.

Two tests close that loop from both ends: the renderer's placeholder is asserted
directly, and every committed report is asserted to carry a placeholder the
renderer can produce.

A third test guards a separate defect in the same header (#1152):
``harness_sha`` used to pin the parent commit with no marker saying the working
tree that generated the report was never committed at all, so a report could
name a commit that cannot produce the bytes it stamps. It also pins the SHAPE
of the value, not only its dirty/clean state: a probe repo with no reachable
tag cannot tell a bare ``git describe --always --dirty`` apart from the
correct ``--exclude='*'`` form, since both fall back to the same short hash
when there is nothing to describe from, so the probe repo carries a tag on
purpose.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

import src.strategy.eval.report as report
from src.strategy.eval.report import ReportHeader, _render_md

ROOT = Path(__file__).parent.parent.parent
REPORTS = sorted((ROOT / "documents" / "eval_reports").glob("*.md"))

# What an empty artifact dict renders as. An em dash here is a prose defect in
# generated output as well as a mismatch, so the value is asserted literally
# rather than by "is not empty".
EMPTY_ARTIFACTS = "none"


def _artifacts_line(text: str) -> str:
    """The `- artifacts: ...` line of a report, or "" when the report has none."""
    match = re.search(r"^- artifacts: (.*)$", text, flags=re.MULTILINE)
    return match.group(1) if match else ""


def test_renderer_spells_an_empty_artifact_set_as_none():
    """The placeholder is `none`, so no generated report carries an em dash."""
    header = ReportHeader(
        harness_sha="deadbeef",
        dataset="probe",
        seed_policy="deterministic",
        llm="none",
        artifacts={},
    )
    rendered = _artifacts_line(_render_md("probe", header, "| a |\n|---|\n"))
    assert rendered == EMPTY_ARTIFACTS


@pytest.mark.parametrize("report", REPORTS, ids=lambda p: p.stem)
def test_committed_report_artifacts_line_matches_the_renderer(report: Path):
    """A committed report either lists hashed artifacts or uses the placeholder.

    Anything else means the line was written by something other than
    :func:`_render_md`, which is how the two spellings appeared.
    """
    line = _artifacts_line(report.read_text(encoding="utf-8"))
    assert line, f"{report.name} has no artifacts line"
    if line == EMPTY_ARTIFACTS:
        return
    # A non-empty set is `name=`hash`` pairs joined by ", ".
    for entry in line.split(", "):
        assert re.fullmatch(r"[\w.\-]+=`[0-9a-f]+`", entry), (
            f"{report.name} artifacts entry {entry!r} is not a rendered name=`hash` pair"
        )


def _run_git(repo: Path, *args: str) -> None:
    """Run a git command against `repo`, raising if it fails."""
    subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True, text=True)


# A short git SHA: hex digits only, git's usual 7-40 char abbreviation range.
# A tag-relative `describe` output (`legacy-2026-07-13-1098-g8b6cb305`) fails
# this on both counts, letters outside a-f and a length no abbreviation reaches.
_SHORT_SHA = re.compile(r"^[0-9a-f]{4,40}$")


def test_harness_sha_marks_a_dirty_working_tree(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """`_harness_sha()` is a short sha, dirty-suffixed only when the tree is.

    Exercises a real, disposable git repository under `tmp_path` rather than
    mocking `subprocess`, with a tag on its first commit so the repo behaves
    like this one (which carries `legacy-2026-07-13` on an ancestor of HEAD)
    rather than a bare scratch repo, where a tagless `describe` looks
    identical whether or not `--exclude` is passed.

    Two things are pinned, not one. The dirty marker guards the fix from
    #1152: reverting to `git rev-parse --short HEAD` carries no `-dirty`
    suffix and always names the last commit, even when the tree that
    produced the report was never committed. The shape guards the fix to
    that fix: dropping `--exclude='*'` lets `describe` walk back to the
    reachable tag and stamp a long, tag-relative description in its place,
    which still ends in `-dirty` on schedule and would pass a check that
    only looked at the suffix.
    """
    repo = tmp_path / "probe"
    repo.mkdir()
    _run_git(repo, "init", "-q")
    _run_git(repo, "config", "user.email", "probe@test.local")
    _run_git(repo, "config", "user.name", "probe")
    tracked = repo / "tracked.txt"
    tracked.write_text("v1", encoding="utf-8")
    _run_git(repo, "add", "tracked.txt")
    _run_git(repo, "commit", "-q", "-m", "init")
    # Annotated, not lightweight: `describe` ignores lightweight tags by default.
    _run_git(repo, "tag", "-a", "unrelated-legacy-tag", "-m", "probe tag")

    monkeypatch.setattr(report, "_find_repo_root", lambda: repo)

    clean_sha = report._harness_sha()
    assert not clean_sha.endswith("-dirty"), f"clean tree stamped as dirty: {clean_sha!r}"
    assert _SHORT_SHA.fullmatch(clean_sha), (
        f"clean sha {clean_sha!r} is not a bare short sha, the reachable tag leaked in"
    )

    tracked.write_text("v2", encoding="utf-8")  # uncommitted change to a tracked file
    dirty_sha = report._harness_sha()
    assert dirty_sha == f"{clean_sha}-dirty", (
        f"dirty sha {dirty_sha!r} is not the clean sha plus -dirty"
    )
