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
    """`palette.py` is a copy of `config.py`'s tuples and says so.

    The two run in different processes, so the duplication is deliberate;
    a silent divergence would put the arcade HUD, the Qt dashboard and
    PITWALL's AGENTS window on different colours with nothing to catch it.

    **This used to be skipped in CI.** The copy lived in
    `dashboard/theme.py`, importing it needed libEGL on a headless runner,
    and `importorskip` therefore skipped the only guard the pair had. The
    Qt-free split is what made it run.
    """
    from src.arcade import config, palette

    shared = [
        name
        for name in dir(config)
        if name.isupper() and isinstance(getattr(config, name), tuple) and hasattr(palette, name)
    ]

    assert len(shared) >= 8, "the palette names moved; this test is checking nothing"
    for name in shared:
        assert getattr(palette, name) == getattr(config, name), f"{name} drifted between the copies"


def test_the_qt_theme_still_serves_the_same_tuples_it_re_exports():
    """`dashboard/theme.py` keeps every name its widgets import.

    The split moved the values out and left re-exports behind, so a widget
    doing `from ...theme import ACCENT` must still get the same object. If
    Qt is unavailable this skips — but the pair above is guarded either
    way now, which is the point of the split.
    """
    theme = pytest.importorskip(
        "src.arcade.dashboard.theme",
        reason="the Qt dashboard is an optional surface and needs a display stack",
        exc_type=ImportError,
    )
    from src.arcade import palette

    for name in ("ACCENT", "SUCCESS", "WARNING", "DANGER", "TEXT_PRIMARY", "MONO_FONT_STACK"):
        assert getattr(theme, name) is getattr(palette, name), (
            f"{name} is no longer the same object"
        )
    assert theme.compound_pill_html is palette.compound_pill_html
    assert theme.flag_chip_html is palette.flag_chip_html


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


# --- The two copies the AGENTS window introduced ----------------------------

AGENTS_CSS = REPO_ROOT / "src" / "pitwall" / "ui" / "src" / "styles" / "agents.css"
AGENTS_WINDOW = (
    REPO_ROOT / "src" / "pitwall" / "ui" / "src" / "features" / "agents" / "AgentsWindow.tsx"
)

# The `--qt-*` custom properties, and what each one copies from `palette.py`.
QT_CSS_TOKENS = {
    "qt-bg": "BG_COLOR",
    "qt-panel": "CONTENT_BG",
    "qt-elevated": "SECONDARY_BG",
    "qt-border": "BORDER_COLOR",
    "qt-fg-1": "TEXT_PRIMARY",
    "qt-fg-2": "TEXT_SECONDARY",
    "qt-fg-3": "TEXT_TERTIARY",
    "qt-accent": "ACCENT",
}


def test_the_agents_css_palette_has_not_drifted_from_palette_py():
    """Copy number three, and the first one a stylesheet could break silently.

    The AGENTS window renders the arcade palette rather than `tokens.css`,
    because it is a 1:1 port of a Qt window and the two palettes
    deliberately differ. That means the hexes are duplicated into CSS,
    and a duplicate with no guard is how this repo's most frequent defect
    starts.
    """
    from src.arcade import palette

    declared = dict(re.findall(r"--([\w-]+):\s*(#[0-9a-fA-F]{6})", AGENTS_CSS.read_text("utf-8")))

    assert set(QT_CSS_TOKENS) <= set(declared), (
        f"a --qt-* token was renamed or removed: {sorted(set(QT_CSS_TOKENS) - set(declared))}"
    )
    for token, name in QT_CSS_TOKENS.items():
        assert declared[token].lower() == _rgb_to_hex(getattr(palette, name)), (
            f"--{token} is {declared[token]}, but palette.{name} is "
            f"{_rgb_to_hex(getattr(palette, name))}"
        )


# The boot state's colours, SLOT BY SLOT. Membership in the palette is not
# enough: swapping the badge from ACCENT to DANGER keeps every hex inside
# the palette and still boots the window in the wrong colour. Measured on a
# mutated source before this map existed — the membership version passed.
BOOT_SLOTS = {
    "action_colour": "ACCENT",
    "confidence_colour": "TEXT_TERTIARY",
    "pace_colour": "TEXT_TERTIARY",
    "risk_colour": "TEXT_TERTIARY",
    "connection_colour": "WARNING",
    "bar_colour": "TEXT_SECONDARY",
    "label_colour": "TEXT_SECONDARY",
    "score_colour": "TEXT_SECONDARY",
    "glyph_colour": "TEXT_TERTIARY",
    "headline_colour": "TEXT_PRIMARY",
    "actual_colour": "INFO",
    "pred_colour": "ACCENT",
    "band_colour": "ACCENT",
    "trend_colour": "TEXT_PRIMARY",
    "cliff_colour": "WARNING",
    "boundary_colour": "TEXT_TERTIARY",
}

# Copy number five, which had no detector at all: the ECharts axis styling
# repeats the pens `pace_chart.py` builds from the same two names.
ECHART_MODULE = (
    REPO_ROOT / "src" / "pitwall" / "ui" / "src" / "features" / "agents" / "useEChart.ts"
)
ECHART_NAMES = ("TEXT_SECONDARY", "BORDER_COLOR")


def test_the_boot_state_colours_are_in_the_right_slots():
    """Copy number four: the state the window paints before the first tick.

    It cannot come from the host — there is no host answer yet — so the
    colours are literals in TSX, and each one copies a specific palette
    name. Asserting only that a hex is SOMEWHERE in the palette lets a
    wrong-but-known colour through, which is the failure mode a membership
    test always has.
    """
    from src.arcade import palette

    source = AGENTS_WINDOW.read_text("utf-8")
    found = dict(re.findall(r'(\w*_?colour)"?:\s*"(#[0-9a-fA-F]{6})"', source))

    assert set(BOOT_SLOTS) <= set(found), (
        f"a boot colour slot disappeared: {sorted(set(BOOT_SLOTS) - set(found))}"
    )
    for slot, name in BOOT_SLOTS.items():
        assert found[slot] == _rgb_to_hex(getattr(palette, name)), (
            f"{slot} is {found[slot]}, but it copies palette.{name} "
            f"({_rgb_to_hex(getattr(palette, name))})"
        )


def test_the_chart_axis_palette_has_not_drifted_either():
    """The fifth copy, and nothing referenced this file before.

    `useEChart.ts` repeats the axis pens `pace_chart.py` builds from
    BORDER_COLOR and TEXT_SECONDARY. Same duplication as the other four,
    and it had no guard at all.
    """
    from src.arcade import palette

    literals = set(re.findall(r'"(#[0-9a-fA-F]{6})"', ECHART_MODULE.read_text("utf-8")))
    expected = {_rgb_to_hex(getattr(palette, name)) for name in ECHART_NAMES}

    assert literals, "the chart theme stopped carrying colours"
    assert literals == expected, (
        f"the chart axis palette drifted: {sorted(literals)} against {sorted(expected)}"
    )
