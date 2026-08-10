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


def _without_comments(css: str) -> str:
    """CSS with its `/* ... */` blocks removed.

    The raw-hex freezes below scan a whole stylesheet, and a comment that
    NAMES a colour as evidence is not a colour the stylesheet uses. One did:
    a note explaining that VER's #0600ef was invisible on a dark panel failed
    the guard that exists to catch an unguarded declaration. Scanning prose
    is a false positive today and a reason to delete the evidence tomorrow.
    """
    return re.sub(r"/\*.*?\*/", "", css, flags=re.DOTALL)


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


# --- The copies the two PITWALL windows introduced --------------------------

UI_SRC = REPO_ROOT / "src" / "pitwall" / "ui" / "src"
# Sprint 4 moved the `--qt-*` block out of `agents.css`: the DATA window needs
# the same tokens, and pasting them into a second stylesheet would have been
# copy number six. This guard follows the declarations, it does not follow the
# AGENTS window.
QT_BASE_CSS = UI_SRC / "styles" / "qt-base.css"
AGENTS_CSS = UI_SRC / "styles" / "agents.css"
DATA_CSS = UI_SRC / "styles" / "data.css"
AGENTS_WINDOW = UI_SRC / "features" / "agents" / "AgentsWindow.tsx"
OWN_CAR_TRACES = UI_SRC / "features" / "data" / "OwnCarTraces.tsx"

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


def test_the_qt_base_palette_has_not_drifted_from_palette_py():
    """Copy number three, and the first one a stylesheet could break silently.

    Both PITWALL windows render the arcade palette rather than `tokens.css`,
    because they are 1:1 ports of Qt windows and the two palettes
    deliberately differ. That means the hexes are duplicated into CSS,
    and a duplicate with no guard is how this repo's most frequent defect
    starts.
    """
    from src.arcade import palette

    declared = dict(re.findall(r"--([\w-]+):\s*(#[0-9a-fA-F]{6})", QT_BASE_CSS.read_text("utf-8")))

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
# repeats the pens `pace_chart.py` and `telemetry_panel.py` build from the
# same names. Sprint 4 collapsed it from four hard-coded axis blocks to one
# `valueAxis()` helper both windows call, so the counts fell with it - one
# site per colour instead of one per axis - and TEXT_TERTIARY joined for
# band 4's shared cursor.
ECHART_MODULE = REPO_ROOT / "src" / "pitwall" / "ui" / "src" / "lib" / "chart.ts"
ECHART_SITES = (("TEXT_SECONDARY", 1), ("BORDER_COLOR", 1), ("TEXT_TERTIARY", 1))

# Copy number six: band 4's four traces. Each metric draws in a DIFFERENT
# palette colour (`telemetry_panel.py` passes INFO, DANGER and SUCCESS to
# `_make_chart`), so this is a slot map for the same reason `BOOT_SLOTS` is -
# swapping brake from DANGER to WARNING keeps every hex inside the palette
# and is still the wrong chart in the wrong colour.
TRACE_SLOTS = {
    "delta_main": "INFO",
    "speed_main": "INFO",
    "brake_main": "DANGER",
    "throttle_main": "SUCCESS",
    "rival": "WARNING",
}


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

    `lib/chart.ts` repeats the axis pens `pace_chart.py` and
    `telemetry_panel.py` build from BORDER_COLOR and TEXT_SECONDARY, plus
    TEXT_TERTIARY for band 4's cursor. Same duplication as the other four,
    and it had no guard at all.

    **Counted per site, not as a set.** Set equality passes when ONE of
    the `#d1d5db` sites becomes `rgba(209,213,219,0.4)`: the hex is
    still present elsewhere, so the set is unchanged and a real render
    drift goes green. Measured on a mutated copy before this counted.
    """
    from collections import Counter

    from src.arcade import palette

    source = ECHART_MODULE.read_text("utf-8")
    found = Counter(hex_.lower() for hex_ in re.findall(r'"(#[0-9a-fA-F]{6})"', source))
    expected = Counter({_rgb_to_hex(getattr(palette, name)): count for name, count in ECHART_SITES})

    assert found, "the chart theme stopped carrying colours"
    assert found == expected, f"the chart axis palette drifted: {found} against {expected}"

    # The splitLine pen is rgba, so the hex regex above cannot see it at all.
    # It is value-paired with pyqtgraph's grid alpha.
    assert source.count("rgba(255,255,255,0.06)") == 1, "the grid alpha moved"


def test_the_trace_colours_are_in_the_right_slots():
    """Copy number six: band 4's four traces, one palette name each.

    `telemetry_panel.py` hands `_make_chart` a DIFFERENT main colour per
    metric - INFO for delta and speed, DANGER for brake, SUCCESS for
    throttle, WARNING for the rival on all four. A membership test cannot
    see brake and throttle swapping places; both hexes stay in the palette
    and the window is simply wrong.
    """
    from src.arcade import palette

    source = OWN_CAR_TRACES.read_text("utf-8")
    found = dict(re.findall(r"(\w+):\s*\"(#[0-9a-fA-F]{6})\"", source))

    assert set(TRACE_SLOTS) <= set(found), (
        f"a trace colour slot disappeared: {sorted(set(TRACE_SLOTS) - set(found))}"
    )
    for slot, name in TRACE_SLOTS.items():
        assert found[slot] == _rgb_to_hex(getattr(palette, name)), (
            f"{slot} is {found[slot]}, but it copies palette.{name} "
            f"({_rgb_to_hex(getattr(palette, name))})"
        )


def test_the_data_stylesheets_raw_hexes_are_guarded_too():
    """The DATA window's own pair, the same blind spot `agents.css` had.

    The two driver chips take their colour from the chip's own constructor
    argument in Qt (`_driver_chip(code, INFO)` / `(code, WARNING)`), so in
    CSS they are literals rather than `--qt-*` tokens - and a literal is
    invisible to the declaration-reading guard above.
    """
    from src.arcade import palette

    raw = sorted(
        {
            hex_.lower()
            for hex_ in re.findall(
                r"(#[0-9a-fA-F]{6})", _without_comments(DATA_CSS.read_text("utf-8"))
            )
        }
    )

    assert raw == ["#3b82f6", "#f59e0b"], (
        f"a new raw hex entered the DATA stylesheet: {raw}. Either use a --qt-* token or add it "
        "here with the palette name it copies."
    )
    assert raw[0] == _rgb_to_hex(palette.INFO), "the main driver chip copies INFO"
    assert raw[1] == _rgb_to_hex(palette.WARNING), "the rival driver chip copies WARNING"


def test_the_two_raw_hexes_in_the_stylesheet_are_guarded_too():
    """The pair the `--qt-*` guard structurally cannot see.

    `test_the_agents_css_palette_has_not_drifted_from_palette_py` reads
    custom-property declarations, so a literal written straight into a
    rule is invisible to it. #876's wording -- "useEChart.ts stops being
    the ONE palette copy with no detector" -- was literally true and
    stepped around these two.
    """
    from src.arcade import palette

    css = _without_comments(AGENTS_CSS.read_text("utf-8"))
    declared = set(re.findall(r"--[\w-]+:\s*(#[0-9a-fA-F]{6})", css))
    raw = sorted({hex_.lower() for hex_ in re.findall(r"(#[0-9a-fA-F]{6})", css)} - declared)

    assert raw == ["#282834", "#ef4444"], (
        f"a new raw hex entered the stylesheet: {raw}. Either use a --qt-* token or add it here "
        "with the palette name it copies."
    )
    assert "#ef4444" == _rgb_to_hex(palette.DANGER), "the guardrail rule copies DANGER"
    # #282834 is the empty half of the confidence/scenario bar. It is NOT in
    # palette.py: `orchestrator_card.py` writes it straight into its Qt
    # stylesheet too, so freezing it here is what makes the pair monitored.
    assert css.count("#282834") == 1, "the empty-bar shade is used in exactly one rule"
