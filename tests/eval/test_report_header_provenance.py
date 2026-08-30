"""Every committed report's provenance header is what the renderer emits today.

The header is generated, so it is the one part of a report nobody should edit by
hand. It was edited by hand: the writing pass that removed em dashes from the
documentation tree reached into ``documents/eval_reports/`` and rewrote the
renderer's ``artifacts`` placeholder in four reports without touching the
renderer. The next regeneration put the em dash back in a fifth, so the tree held
two spellings of the same empty value and neither traced to the code.

These two tests close that loop from both ends: the renderer's placeholder is
asserted directly, and every committed report is asserted to carry a placeholder
the renderer can produce.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

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
