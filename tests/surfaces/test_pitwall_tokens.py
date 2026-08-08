"""Design-token drift across every copy in the repo.

The webapp's `tokens.css` is the canonical palette. PITWALL copies it rather
than extracting a publishable design package, which is the cheap answer and
the right one - but a copy with no guard is how palettes drift, and in this
repo that is not a hypothetical:

> P3 finding A16 warned about exactly this. A gate then measured that the
> drift had ALREADY happened: `src/arcade/config.py`'s Python palette
> disagrees with the webapp on **every semantic colour**, and its own
> comment cites a file that no longer exists.

So this test covers all four copies, not just the pair PITWALL introduces.
A test that guarded only the new pair would leave the broken one uncovered,
which is this repo's most-repeated defect committed inside the fix for it.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
WEBAPP_TOKENS = REPO_ROOT / "src" / "telemetry" / "webapp" / "src" / "styles" / "tokens.css"
PITWALL_TOKENS = REPO_ROOT / "src" / "pitwall" / "ui" / "src" / "styles" / "tokens.css"

# The submodule is not checked out in every working tree; CI checks it out.
pytestmark = pytest.mark.skipif(
    not WEBAPP_TOKENS.is_file(),
    reason="src/telemetry submodule not checked out - nothing to compare against",
)


def _dark_tokens(css: str) -> dict[str, str]:
    """Hex tokens from the default (dark) block only.

    The file declares light as an override under `:root[data-theme='light']`,
    so reading the whole file would take whichever value appears first and
    silently compare the arcade's dark palette against light-theme hexes.
    """
    dark_block = css.split("[data-theme='light']")[0]
    return {
        name: value.lower()
        for name, value in re.findall(r"--([\w-]+):\s*(#[0-9a-fA-F]{3,8})", dark_block)
    }


def _rgb_to_hex(rgb: tuple[int, int, int]) -> str:
    return "#{:02x}{:02x}{:02x}".format(*rgb)


# --- The pair PITWALL introduces: must be identical, always -----------------


def test_the_pitwall_copy_is_byte_identical_to_the_webapp_source():
    """The one hard guarantee. Re-copy the file; do not hand-edit it.

    If this fails and the change was intended, the intended change belongs
    in the webapp first, because that is the canonical source.
    """
    assert PITWALL_TOKENS.read_bytes() == WEBAPP_TOKENS.read_bytes(), (
        f"cp {WEBAPP_TOKENS.relative_to(REPO_ROOT)} {PITWALL_TOKENS.relative_to(REPO_ROOT)}"
    )


def test_the_canonical_source_actually_defines_the_tokens_the_ui_uses():
    """A guard that asserted about an empty set would pass on a moved file."""
    tokens = _dark_tokens(WEBAPP_TOKENS.read_text(encoding="utf-8"))

    assert len(tokens) >= 20, f"only {len(tokens)} hex tokens found - has the file moved?"
    for required in ("bg-1", "bg-2", "purple-600", "success", "warning", "danger"):
        assert required in tokens, f"--{required} is gone from the canonical palette"


# --- The Python copies: one must mirror the other, and both have drifted ----


def test_the_two_python_palettes_still_mirror_each_other():
    """`dashboard/theme.py` is a copy of `config.py` and says so.

    It disappears when `src/arcade/dashboard/` is deleted in sprint 7. Until
    then a silent divergence between them would put the arcade HUD and the
    Qt dashboard on different colours with nothing to catch it.
    """
    # Skip on the module that actually fails, not on PySide6: PySide6 itself
    # imports fine on a headless runner and `theme.py`'s Qt submodule is what
    # needs libEGL. `exc_type` is explicit because pytest 9.1 stops swallowing
    # ImportError by default.
    theme = pytest.importorskip(
        "src.arcade.dashboard.theme",
        reason="the Qt dashboard is an optional surface and needs a display stack",
        exc_type=ImportError,
    )
    from src.arcade import config

    shared = [
        name
        for name in dir(config)
        if name.isupper() and isinstance(getattr(config, name), tuple) and hasattr(theme, name)
    ]

    assert len(shared) >= 8, "the palette names moved; this test is checking nothing"
    for name in shared:
        assert getattr(theme, name) == getattr(config, name), f"{name} drifted between the copies"


# The Python palette predates the webapp's current tokens and has NOT been
# migrated: doing so would restyle the pyglet track and HUD, which is a visual
# decision and not this sprint's. Freezing the divergence is what makes it
# monitored: a new Python colour, or a change to either side, fails here and
# forces the decision rather than deepening the drift silently.
KNOWN_PYTHON_DRIFT = {
    "BG_COLOR": ("#121127", "bg-1"),
    "CONTENT_BG": ("#181633", "bg-2"),
    "ACCENT": ("#a78bfa", "purple-600"),
    "SUCCESS": ("#10b981", "success"),
    "WARNING": ("#f59e0b", "warning"),
    "DANGER": ("#ef4444", "danger"),
}


def test_the_python_palettes_known_drift_has_not_moved():
    """Every one of these differs from the canonical palette, on purpose for now.

    This is not a test that the drift is correct. It is a test that the
    drift is EXACTLY the enumerated set, so nobody widens it by accident and
    nobody reads A16 as closed. Unifying them is a visual change and its own
    piece of work.
    """
    from src.arcade import config

    canonical = _dark_tokens(WEBAPP_TOKENS.read_text(encoding="utf-8"))

    still_drifted = []
    for name, (expected_hex, token) in KNOWN_PYTHON_DRIFT.items():
        actual_hex = _rgb_to_hex(getattr(config, name))
        assert actual_hex == expected_hex, (
            f"{name} changed to {actual_hex}; if that was deliberate, decide whether it "
            f"should now match --{token} ({canonical.get(token)}) and update this map"
        )
        if actual_hex != canonical.get(token):
            still_drifted.append(name)

    assert set(still_drifted) == set(KNOWN_PYTHON_DRIFT), (
        "a Python colour now matches the canonical palette - remove it from "
        "KNOWN_PYTHON_DRIFT rather than leaving a stale exception"
    )
