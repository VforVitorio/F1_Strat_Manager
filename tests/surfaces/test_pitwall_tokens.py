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
TRACE_STACK = UI_SRC / "features" / "data" / "TraceStack.tsx"

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
    # TEXT_TERTIARY, not WARNING. The boot literal is a placeholder for the
    # polled pair, and "Connecting..." is an ABSENCE rather than a state: the
    # DATA strip made that argument in its own stylesheet while this window
    # painted the same socket amber, so one socket wore two colours on two
    # windows open side by side.
    "connection_colour": "TEXT_TERTIARY",
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
    # Sprint 8's addition. The map is a SUBSET assertion, so a new boot colour
    # it does not name is invisible to it - which is why every slot that exists
    # has to be listed here rather than only the interesting ones.
    #
    # `action_text_colour` sat here too, mapped to BG_COLOR, until the decision
    # band stopped painting the action as a FILL. Text on `--qt-panel` needs no
    # chosen ink, so the field has no consumer and no boot literal; what
    # replaces the guarantee is the contrast assertion in
    # `test_pitwall_agents_view.py`, which now runs the action colour against
    # the panel over the whole of `_ACTION_STYLE` plus its fallback.
    "cursor_colour": "TEXT_TERTIARY",
}

# Copy number five, which had no detector at all: the ECharts axis styling
# repeats the pens `pace_chart.py` and `telemetry_panel.py` build from the
# same names. Sprint 4 collapsed it from four hard-coded axis blocks to one
# `valueAxis()` helper both windows call, so the counts fell with it - one
# site per colour instead of one per axis - and TEXT_TERTIARY joined for
# band 4's shared cursor.
ECHART_MODULE = REPO_ROOT / "src" / "pitwall" / "ui" / "src" / "lib" / "chart.ts"
ECHART_SITES = (("TEXT_SECONDARY", 1), ("BORDER_COLOR", 1), ("TEXT_TERTIARY", 1))

# Copy number six: band 4's SIX lanes. Each channel draws in a DIFFERENT palette
# colour, so this is a slot map for the same reason `BOOT_SLOTS` is - swapping brake
# from DANGER to WARNING keeps every hex inside the palette and is still the wrong
# lane in the wrong colour.
#
# The names moved with the panel: the 2x2's four `<metric>_main` constants became one
# lane table in `TraceStack`, whose colours are the four palette names it uses plus
# the two the new lanes add - ACCENT for gear (its own channel, a staircase) and INFO
# reused for DRS (a bit, in the thinnest lane).
# No RIVAL slot any more (#1070). Band 4's rival used to draw in a fixed
# palette.WARNING on all six lanes, inherited from the Qt panel, and it now takes
# the pinned driver's own colour off `driver_colors`. There is no constant left
# to pin, and the property that replaced it - the served colour equals the pinned
# car's - is not a palette question at all: it is asserted through the built
# bundle in `smoke-data.mjs`, because a palette-membership check can never fail
# on it.
TRACE_SLOTS = {
    "INFO": "INFO",
    "SUCCESS": "SUCCESS",
    "DANGER": "DANGER",
    "ACCENT": "ACCENT",
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

    # Same blind spot, second occupant: the neutralised-lap band the race trace
    # shades. Its RGB is WARNING and only the alpha is chosen here, so the guard
    # pins the triplet to the palette and lets the alpha be what it is.
    bands = re.findall(r"rgba\((\d+),\s*(\d+),\s*(\d+),\s*([\d.]+)\)", source)
    neutralised = [triplet for triplet in bands if triplet[:3] != ("255", "255", "255")]
    assert len(neutralised) == 1, f"expected one neutralised band colour, found {neutralised}"
    assert tuple(int(value) for value in neutralised[0][:3]) == palette.WARNING, (
        "the neutralised-lap band copies palette.WARNING; the pace grid's rail uses the same "
        "hue in a different channel and the two must not drift apart"
    )


def test_the_trace_colours_are_in_the_right_slots():
    """Copy number six: band 4's six lanes, one palette name each.

    A DIFFERENT colour per channel - INFO for speed and the delta baseline, SUCCESS
    for throttle, DANGER for brake, ACCENT for gear. A membership test cannot see
    brake and throttle swapping places; both hexes stay in the palette and the
    window is simply wrong.

    The rival is NOT in this map. It used to be, as WARNING on every lane, and
    #1070 replaced that constant with the pinned driver's own colour.

    Read from `TraceStack`, which is where the lane table lives now. The file this
    used to read (`OwnCarTraces`) kept four `<metric>_main` constants for the 2x2;
    the stack holds one named constant per palette colour and the lane table points
    at them, so the slot map is by NAME rather than by metric.
    """
    from src.arcade import palette

    source = TRACE_STACK.read_text("utf-8")
    found = dict(re.findall(r"^const (\w+) = \"(#[0-9a-fA-F]{6})\";", source, re.M))

    assert set(TRACE_SLOTS) <= set(found), (
        f"a lane colour slot disappeared: {sorted(set(TRACE_SLOTS) - set(found))}"
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

    assert raw == ["#10b981", "#3b82f6", "#a78bfa", "#ef4444", "#f59e0b"], (
        f"a new raw hex entered the DATA stylesheet: {raw}. Either use a --qt-* token or add it "
        "here with the palette name it copies."
    )
    assert raw[0] == _rgb_to_hex(palette.SUCCESS), (
        "band 1's Connected chip and the tower's personal-best sector copy SUCCESS"
    )
    assert raw[1] == _rgb_to_hex(palette.INFO), "the main driver chip copies INFO"
    assert raw[2] == _rgb_to_hex(palette.ACCENT), (
        "a session-best sector is purple, and purple here is the arcade's ACCENT"
    )
    assert raw[3] == _rgb_to_hex(palette.DANGER), "band 1's Disconnected chip copies DANGER"
    assert raw[4] == _rgb_to_hex(palette.WARNING), (
        "band 1's PROVISIONAL chip, a slower-than-own-best sector, the neutralised-lap "
        "rail and the radio's SC / flag category chips copy WARNING"
    )


def test_the_radio_category_chips_are_pinned_PER_SITE_not_as_a_set():
    """The slot, not the membership - the failure this file's own docstrings warn about.

    **An adversarial gate refuted the set-based guard above for these two sites.** The
    category chips introduced `.radio-cat.is-sc` / `.is-flag` on WARNING and `.is-clear`
    on SUCCESS, and both hexes were already in the stylesheet's set - so swapping the
    clear chip to amber, or the safety-car chip to green, leaves
    `test_the_data_stylesheets_raw_hexes_are_guarded_too` green while the panel says the
    opposite of what happened. A CLEAR flag painted amber reads as a new warning.

    So this reads the rules themselves: each selector, and the exact value it declares.
    The pairing is the claim - amber for anything that changes how the track is being
    driven, green for the all-clear - and it is the pairing a swap breaks.
    """
    from src.arcade import palette

    css = _without_comments(DATA_CSS.read_text("utf-8"))
    declared = {}
    # Selector LISTS, not one selector per rule: `.is-sc` and `.is-flag` deliberately
    # share a declaration because they carry the same weight, and a regex that stops at
    # the first class silently drops the second - which is how the first version of this
    # guard reported three chips where the stylesheet has four.
    #
    # **And `background-color` as well as `background`, with the LAST rule winning**, which
    # is what the cascade does. Reading only `background:` let a second rule appended later
    # in the file repaint the all-clear chip amber with every assertion below still green -
    # a gate proved it by executing exactly that. Rules are walked in file order and the
    # dict is overwritten, so this guard now agrees with the browser about which one wins.
    for selectors, body in re.findall(r"([^{}]+)\{([^{}]*)\}", css):
        matches = re.findall(r"background(?:-color)?:\s*([^;]+);", body)
        if not matches:
            continue
        for chip in re.findall(r"\.radio-cat\.(is-[\w-]+)", selectors):
            declared[chip] = matches[-1].strip().lower()

    assert set(declared) == {"is-sc", "is-flag", "is-clear", "is-drs"}, (
        f"a category chip was added or renamed without pinning its colour: {sorted(declared)}"
    )
    warning = _rgb_to_hex(palette.WARNING)
    assert declared["is-sc"] == warning, "a safety car changes how the track is driven"
    assert declared["is-flag"] == warning, "so does a flag, and it wears the same weight"
    assert declared["is-clear"] == _rgb_to_hex(palette.SUCCESS), (
        "the all-clear is the one green chip; amber here would read as a new warning"
    )
    assert declared["is-drs"] == "var(--qt-fg-3)", (
        "the DRS note is informational and takes the tertiary token, not a hue"
    )


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

    assert raw == ["#282834", "#ef4444", "#f59e0b"], (
        f"a new raw hex entered the stylesheet: {raw}. Either use a --qt-* token or add it here "
        "with the palette name it copies."
    )
    # `.chip.is-frozen`, the dead-producer chip. It was the guardrail rule too
    # until #974 deleted a line no producer could fill, and the hex survived
    # that deletion because the frozen chip is its second copy site.
    assert "#ef4444" == _rgb_to_hex(palette.DANGER), "the frozen chip copies DANGER"
    assert "#f59e0b" == _rgb_to_hex(palette.WARNING), (
        "the `was <call>` dot copies WARNING - a CHANGE worth noticing, deliberately not the "
        "DANGER one line below it, which means a fault"
    )
    # #282834 is the empty half of the confidence/scenario bar. It is NOT in
    # palette.py: `orchestrator_card.py` writes it straight into its Qt
    # stylesheet too, so freezing it here is what makes the pair monitored.
    assert css.count("#282834") == 1, "the empty-bar shade is used in exactly one rule"


# --- The sweep, because the list above is a list -----------------------------


def test_no_pitwall_source_file_carries_an_unregistered_hex():
    """Every guard above names its files. This one FINDS them.

    That is the difference between a check that cannot miss and one that misses
    by construction. `QT_BASE_CSS`, `AGENTS_CSS`, `DATA_CSS`, `AGENTS_WINDOW`
    and `TRACE_STACK` are explicit paths, so a colour written into any file
    that is not one of them is invisible to the palette guards - and a sprint
    that adds `PlanTimeline.tsx`, `WhyPanel.tsx`, `Tooltip.tsx` and
    `agents_view/timeline.py` adds four such files at once.

    Discovery, then. Walk both PITWALL source trees, collect every literal hex,
    and require each one to be a colour `palette.py` actually defines. The
    palette is the register; a hex that is in it is a copy the other guards can
    reason about, and a hex that is not is somebody's eyedropper.

    Not a substitute for the slot maps above, which assert that a copy sits in
    the RIGHT slot. This one only asserts that no copy is off the books.
    """
    from src.arcade import palette

    known = {
        _rgb_to_hex(value).lower()
        for name, value in vars(palette).items()
        if name.isupper() and isinstance(value, tuple) and len(value) == 3
    }
    # Compound and flag colours are palette data too, held in dicts rather than
    # as module constants.
    for mapping in (palette._COMPOUND_COLOUR_BY_LABEL, palette._FLAG_BG_BY_INTENT):
        known.update(_rgb_to_hex(value).lower() for value in mapping.values())
    # The empty half of the confidence and scenario bars. It is NOT in
    # `palette.py` - `orchestrator_card.py` wrote it straight into its own Qt
    # stylesheet too - and the guard above freezes it to exactly one rule.
    known.add("#282834")
    # The Qt reasoning highlighter's five rule colours, ported verbatim from
    # `reasoning_tabs.py` and living in `agents_view/reasoning.py`. They are a
    # palette copy the named guards above never covered - this sweep is what
    # found them - so they are pinned here rather than waved through.
    #
    # **Nothing renders them since #1020.** The tabs they coloured are gone;
    # `build_reasoning` still splits six bodies into per-character runs on every
    # tick and the segments' colours reach no element. Tracked in #1026.
    known.update({"#f472b6", "#d946ef", "#facc15", "#22d3ee"})

    # Explicit globs, not one recursive root. `src/pitwall` contains `ui/dist`
    # and `ui/node_modules`, and sweeping from the top pulled the BUILT bundle
    # in - every hex in the repo, reported against a file nobody edits.
    pitwall = REPO_ROOT / "src" / "pitwall"
    sources = [
        # The top level: `agent_formatters.py` and `host.py` live here, not in
        # `agents_view`, and the first version of this sweep did not look.
        *pitwall.glob("*.py"),
        *(pitwall / "agents_view").rglob("*.py"),
        *UI_SRC.joinpath("features").rglob("*.tsx"),
        *UI_SRC.joinpath("features").rglob("*.ts"),
        *UI_SRC.joinpath("lib").rglob("*.ts"),
        *UI_SRC.joinpath("styles").rglob("*.css"),
        # And the harness stubs, which spell colours BY DESIGN - they are what
        # the smokes compare rendered pixels against, so a stub that drifts from
        # the palette makes a green check meaningless.
        *(pitwall / "ui" / "scripts").glob("*.mjs"),
    ]
    # `tokens.css` is the WEB palette and it is deliberately a different one:
    # both PITWALL windows render in the Qt palette, and CLAUDE.md says so in as
    # many words. It has its own guard in this file -
    # `test_the_pitwall_copy_is_byte_identical_to_the_webapp_source` - which is
    # a stronger claim than membership of `palette.py`.
    skip = {UI_SRC / "styles" / "tokens.css"}
    # **Six hex digits, and NOTHING after them.** The first version of this
    # line ended in a word boundary and the escape did not survive being
    # written: the compiled pattern was `#[0-9a-fA-F]{6}` followed by a literal
    # BACKSPACE, so it matched nothing, in any file, ever. The sweep visited 52
    # files, found zero hexes and passed - a guard about the empty set, written
    # to close the class of guards about the empty set. What caught it was
    # planting a hex and watching this stay green.
    hex_literal = re.compile("#[0-9a-fA-F]{6}")

    # **The pattern, before the files.** Its first version ended in an escape
    # that did not survive being written, and it matched nothing anywhere -
    # which looked exactly like a clean repo. One probe is the difference
    # between a sweep and a green light.
    assert hex_literal.findall("a #123abc b") == ["#123abc"], (
        f"the hex pattern matches nothing: {hex_literal.pattern!r}"
    )

    visited = 0
    offenders: list[str] = []
    for path in sorted(set(sources)):
        if path in skip:
            continue
        visited += 1
        source = _without_comments(path.read_text("utf-8"))
        for found in hex_literal.findall(source):
            if found.lower() not in known:
                offenders.append(f"{path.relative_to(REPO_ROOT)}: {found}")

    # The enumeration first. A glob that resolves to nothing - a moved
    # directory, a wrong `parents[]` - passes silently, and this repo has
    # already shipped one census that found zero files and said so cheerfully.
    assert visited >= 55, f"the sweep only visited {visited} files; the roots are wrong"
    assert not offenders, (
        "a colour that palette.py does not define is written into PITWALL source: "
        + "; ".join(offenders)
    )
