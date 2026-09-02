"""PITWALL's windows ARE the Vite bundle, so a wheel without it installs a dead command.

Measured on 2.5.1 before this existed: the wheel carried
`src/pitwall/ui/package-lock.json`, `package.json` and `tsconfig.json` — three
build-tool manifests — and **zero** files of the bundle, although `dist/` was
sitting on disk at the time. The exact inversion of what is wanted, and nothing
said so: `f1-pitwall` would have started, found no `dist/`, printed its build
hint and exited 1.

Two properties are checked here, and they are not the same kind of check:

1. **`dist/` is internally consistent** — every asset a page references exists,
   and no asset exists that no page references. That second half is the one that
   found a real defect: `emptyOutDir: true` is set in `vite.config.ts` and does
   NOT empty the directory on this platform, so builds accumulated. Measured at
   the time: 83 assets and 11 MB against the 6 and 1.4 MB a hand-cleaned build
   produces — seven copies of a 1.3 MB chunk, all but one dead, all of them
   about to ship.
2. **`pyproject.toml` still asks for the bundle.** This one is a MECHANISM check
   and is labelled as such rather than dressed up: the effect — building a wheel
   and reading it back — is a ~40 s subprocess with a build backend and a
   network-capable isolated env, which does not belong in the unit suite. The
   effect is verified by hand at release time, and the procedure is in
   `CONTRIBUTING.md`: `npm run build` FIRST, then `uv build`, then install the
   wheel into a clean venv and open the windows.
"""

from __future__ import annotations

import re
from pathlib import Path

try:  # 3.11+
    import tomllib
except ModuleNotFoundError:  # 3.10, which this project still supports
    import tomli as tomllib

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
UI = REPO_ROOT / "src" / "pitwall" / "ui"
DIST = UI / "dist"
PAGES = ("data.html", "agents.html")


@pytest.fixture(scope="module")
def built_pages() -> dict[str, str]:
    """The two built pages, or a skip naming the command that makes them.

    `dist/` is gitignored build output, so a fresh clone and every CI job that
    has not run the UI build simply does not have it. Skipping is honest there;
    what must never happen is a wheel built in that state.
    """
    if not DIST.is_dir():
        pytest.skip("src/pitwall/ui/dist is absent - run `npm run build` in src/pitwall/ui")
    missing = [name for name in PAGES if not (DIST / name).is_file()]
    if missing:
        pytest.fail(f"dist/ exists but is missing {missing} - the build did not finish")
    return {name: (DIST / name).read_text(encoding="utf-8") for name in PAGES}


def _referenced(pages: dict[str, str]) -> set[str]:
    """Every `assets/...` path the two pages point at."""
    found: set[str] = set()
    for html in pages.values():
        found.update(re.findall(r"assets/[A-Za-z0-9_.\-]+", html))
    return found


def test_every_asset_a_page_references_is_on_disk(built_pages):
    """A missing chunk renders a blank window with nothing in any log."""
    absent = sorted(ref for ref in _referenced(built_pages) if not (DIST / ref).is_file())

    assert not absent, f"the pages reference assets that are not in dist/: {absent}"


def test_no_asset_survives_that_no_page_references(built_pages):
    """The check that caught the accumulation, and the only one that could.

    An orphan hurts nobody at runtime - the pages name the chunks they load -
    which is exactly why it went unnoticed until a wheel was weighed. Every
    orphan ships, and one of them is 1.3 MB.
    """
    referenced = _referenced(built_pages)
    on_disk = {f"assets/{path.name}" for path in (DIST / "assets").iterdir() if path.is_file()}
    orphans = sorted(on_disk - referenced)

    assert not orphans, (
        f"{len(orphans)} asset(s) in dist/assets are referenced by neither page, and would ship: "
        f"{orphans[:6]}{' ...' if len(orphans) > 6 else ''}. "
        "`npm run build` removes dist/ before building for this reason."
    )


def test_the_wheel_is_still_told_to_carry_the_bundle():
    """MECHANISM, deliberately, and the docstring above says why.

    Frozen the way `test_pitwall_tokens.py` freezes its hex list: the value is
    not that this line is correct, it is that removing it fails here instead of
    silently shipping a wheel whose `f1-pitwall` cannot start.
    """
    config = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    setuptools = config["tool"]["setuptools"]

    assert "ui/dist/**" in setuptools["package-data"].get("src.pitwall", []), (
        "src/pitwall/ui/dist is no longer packaged - `f1-pitwall` would install with no windows"
    )
    assert "f1-pitwall" in config["project"]["scripts"], "the f1-pitwall entry point disappeared"
    # The manifests that DID ship while the bundle did not.
    assert "*.json" in setuptools["exclude-package-data"].get("src.pitwall.ui", []), (
        "the UI's build-tool JSONs would be packaged again"
    )
