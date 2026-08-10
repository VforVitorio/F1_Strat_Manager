"""Nothing on the docs site may point at a page that is not there.

Written when the changelog page was removed. Deleting a page touches four
places — `docs/app/nav.js`, the markdown that links to it, `docs/llms.txt`,
and the file itself — and missing one of them leaves a nav entry that loads
nothing or a link that 404s. Neither fails any build: the site is a SPA that
fetches markdown at runtime, so a dangling entry is a blank pane and a
dangling link is a dead click, both silent.

These read the real files rather than a list maintained here, so a page added
tomorrow is covered without anyone remembering this file exists.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DOCS = REPO_ROOT / "docs"
NAV = DOCS / "app" / "nav.js"
PAGES = DOCS / "pages"

_NAV_ENTRY = re.compile(
    r"\{\s*slug:\s*\"(?P<slug>[^\"]+)\".*?file:\s*\"(?P<file>[^\"]+)\"",
    re.DOTALL,
)
# `[text](#/slug)` and `[text](#/slug#heading)`, which is how every in-app link
# is written. The heading half is not checked: an anchor that moves is a
# nuisance, a page that is gone is a dead end.
_INTERNAL_LINK = re.compile(r"\]\(#/([a-z0-9-]+)")


def _nav_entries() -> list[tuple[str, str]]:
    entries = _NAV_ENTRY.findall(NAV.read_text(encoding="utf-8"))
    assert len(entries) >= 15, f"only {len(entries)} nav entries parsed - has the shape changed?"
    return entries


def test_every_nav_entry_points_at_a_page_that_exists():
    """A nav entry whose file is gone renders an empty pane, silently."""
    missing = [(slug, file) for slug, file in _nav_entries() if not (DOCS / file).is_file()]
    assert not missing, f"nav entries with no page: {missing}"


def test_every_page_is_reachable_from_the_nav():
    """The other direction: a page nobody can navigate to is dead weight that
    still gets prerendered, sitemapped and fed to crawlers."""
    listed = {(DOCS / file).resolve() for _slug, file in _nav_entries()}
    orphans = sorted(p.name for p in PAGES.glob("*.md") if p.resolve() not in listed)
    assert not orphans, f"pages not in the nav: {orphans}"


def test_no_internal_link_points_at_a_slug_that_does_not_exist():
    """The failure removing a page actually causes, if you miss a caller."""
    slugs = {slug for slug, _file in _nav_entries()}
    # Not a page, but a real route the app handles.
    slugs.add("graph")

    dangling: list[str] = []
    for page in sorted(PAGES.glob("*.md")):
        for target in _INTERNAL_LINK.findall(page.read_text(encoding="utf-8")):
            if target not in slugs:
                dangling.append(f"{page.name} -> #/{target}")
    assert not dangling, f"internal links with no destination: {dangling}"


def test_llms_txt_only_advertises_pages_that_exist():
    """It is the file AI crawlers read as the site's index, so a stale entry
    there is a promise made to something that will follow it."""
    advertised = re.findall(
        r"https://docs\.f1stratlab\.com/pages/([a-z0-9-]+\.md)",
        (DOCS / "llms.txt").read_text(encoding="utf-8"),
    )
    assert advertised, "llms.txt lists no pages at all - has the format changed?"
    missing = [name for name in advertised if not (PAGES / name).is_file()]
    assert not missing, f"llms.txt advertises pages that are gone: {missing}"
