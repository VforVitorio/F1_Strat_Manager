"""The docs site's changelog page must not fall behind the repo's own.

It did, badly: the public page advertised **1.10.5** while the repo had shipped
**2.5.1** — a whole major version of releases missing from the only public
record of them. Nobody forgot on purpose. It was a hand-refreshed copy of a
file that release-please rewrites on every merge to `main`, and there was no
step carrying one across to the other.

`scripts/sync_docs_changelog.mjs` generates the page now, and the docs
workflow runs it on every deploy. These are the guards that make the rot
visible in the repo too, rather than only on the site.

They assert the EFFECT — what the page SAYS — rather than diffing it against
the generator's output. A byte-diff would only prove the generator is
deterministic, which is not the property that was broken.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CANONICAL = REPO_ROOT / "CHANGELOG.md"
PAGE = REPO_ROOT / "docs" / "pages" / "changelog.md"

_ISO_HEADING = re.compile(r"^#{2,3} .*\((\d{4})-(\d{2})-(\d{2})\)\s*$", re.MULTILINE)
_RELEASE = re.compile(r"^#{2,3} \[(\d+\.\d+\.\d+)\]", re.MULTILINE)


def _newest(text: str) -> str:
    found = _RELEASE.search(text)
    assert found, "no release heading at all - has the format changed?"
    return found.group(1)


def test_the_page_carries_the_newest_release_the_repo_has_shipped():
    """The whole defect, in one assertion.

    Pinned against `CHANGELOG.md` rather than a hardcoded version: a literal
    would need editing on every release, which is the same manual step that
    failed in the first place.
    """
    newest_shipped = _newest(CANONICAL.read_text(encoding="utf-8"))
    newest_published = _newest(PAGE.read_text(encoding="utf-8"))

    assert newest_published == newest_shipped, (
        f"the docs page stops at {newest_published} but the repo has shipped "
        f"{newest_shipped}. Run: node scripts/sync_docs_changelog.mjs"
    )


def test_every_release_in_the_repo_reaches_the_page():
    """Newest-only would pass on a page that lost its middle.

    A generator bug, or a bad hand-edit, can drop releases anywhere in the
    file while leaving the top intact — and the top is the only thing anyone
    checks by eye.
    """
    shipped = set(_RELEASE.findall(CANONICAL.read_text(encoding="utf-8")))
    published = set(_RELEASE.findall(PAGE.read_text(encoding="utf-8")))

    assert shipped, "the canonical changelog has no releases; this test is checking nothing"
    assert not shipped - published, f"missing from the docs page: {sorted(shipped - published)}"


def test_release_dates_are_shown_day_month_year():
    """Víctor's call, 2026-08-10: day-month-year, not the ISO the tool emits.

    Asserted as the ABSENCE of ISO headings plus the presence of day-first
    ones, because a page with no dates at all would satisfy either half on
    its own.
    """
    page = PAGE.read_text(encoding="utf-8")

    iso = _ISO_HEADING.findall(page)
    assert not iso, f"{len(iso)} release headings still carry an ISO date, first {iso[0]}"

    # Counted against the CANONICAL file, not against a number chosen here.
    # The first version of this asserted `>= 50` and failed at 47 — because ten
    # of the 57 releases are the pre-1.2.0 ones seeded retroactively from the
    # GitHub Releases history and carry no date in their heading at all. A
    # floor picked by eye tests the guess; this tests the transformation.
    dated_upstream = len(_ISO_HEADING.findall(CANONICAL.read_text(encoding="utf-8")))
    day_first = re.findall(r"^#{2,3} .*\((\d{2})-(\d{2})-(\d{4})\)\s*$", page, re.MULTILINE)
    assert dated_upstream > 0, "the canonical file has no dated headings; this checks nothing"
    assert len(day_first) == dated_upstream, (
        f"{dated_upstream} dated headings upstream but {len(day_first)} on the page"
    )
    # A day-month-year date has a day that can exceed 12; a month never can.
    # If the two components were swapped, nothing here would ever be > 12.
    assert any(int(day) > 12 for day, _month, _year in day_first), (
        "no day above 12 anywhere - the day and month look swapped"
    )
    assert all(1 <= int(month) <= 12 for _day, month, _year in day_first), (
        "a month outside 1-12 - the day and month are definitely swapped"
    )


def test_the_page_says_it_is_generated():
    """The old header told the reader it was a "manually-refreshed mirror",
    which was an instruction nobody followed and a promise it could not keep.
    A wrong instruction in a header is how the next person edits the wrong
    file."""
    header = PAGE.read_text(encoding="utf-8")[:600]

    assert "generated" in header.lower(), "the page no longer says it is generated"
    assert "sync_docs_changelog" in header, "the header does not name the generator"
    assert "manually-refreshed" not in header, "the stale instruction came back"
