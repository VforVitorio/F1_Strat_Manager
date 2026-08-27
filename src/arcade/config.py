"""Constants and theme palette for the Arcade race replay.

Centralises every magic number used across `data.py`, `track.py`, `overlays.py`
and `app.py` so the visual design can be tuned from one place. Values are
ported from the Tom Shaw f1-race-replay reference, with TFG-specific
overrides flagged inline.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Final

from src.f1_strat_manager.data_cache import get_data_root

logger = logging.getLogger(__name__)

# --- Playback & timing ----------------------------------------------------
FPS: Final[int] = 25
DT: Final[float] = 1.0 / FPS
PLAYBACK_SPEEDS: Final[tuple[float, ...]] = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0)
DEFAULT_SPEED_IDX: Final[int] = PLAYBACK_SPEEDS.index(1.0)
SEEK_RATE_MULTIPLIER: Final[float] = 3.0

# --- Window geometry ------------------------------------------------------
SCREEN_WIDTH: Final[int] = 1280
SCREEN_HEIGHT: Final[int] = 720
WINDOW_TITLE: Final[str] = "F1 StratLab - Race Replay"

# --- Viewport margins (reserve UI space before fitting track) -------------
MARGIN_LEFT: Final[int] = 340
MARGIN_RIGHT: Final[int] = 260
MARGIN_BOTTOM: Final[int] = 90
MARGIN_TOP: Final[int] = 20
TRACK_PADDING: Final[float] = 0.05

# --- Weather panel --------------------------------------------------------
WEATHER_LEFT: Final[int] = 20
WEATHER_TOP_OFFSET: Final[int] = 90
WEATHER_WIDTH: Final[int] = 280
WEATHER_ROW_GAP: Final[int] = 22
WEATHER_ICON_SIZE: Final[int] = 16

# --- Driver info panel ----------------------------------------------------
DRIVER_BOX_WIDTH: Final[int] = 300
DRIVER_BOX_HEIGHT: Final[int] = 145
DRIVER_BOX_GAP: Final[int] = 32
DRIVER_HEADER_HEIGHT: Final[int] = 28
DRIVER_ROW_GAP: Final[int] = 19

# --- Leaderboard ----------------------------------------------------------
LEADERBOARD_WIDTH: Final[int] = 240
LEADERBOARD_RIGHT_MARGIN: Final[int] = 260
LEADERBOARD_ROW_HEIGHT: Final[int] = 28
LEADERBOARD_N_SLOTS: Final[int] = 22

# --- Progress bar ---------------------------------------------------------
PROGRESS_BAR_BOTTOM: Final[int] = 30
PROGRESS_BAR_HEIGHT: Final[int] = 24

# --- Controls legend ------------------------------------------------------
LEGEND_X: Final[int] = 20
LEGEND_BOTTOM: Final[int] = 60

# --- Theme palette (RGB tuples) ------------------------------------------
# This is the canonical Python palette: `palette.py` mirrors it, guarded by
# `tests/surfaces/test_pitwall_tokens.py`. The webapp's `tokens.css` is a
# separate, later palette that deliberately disagrees with these hexes on
# every semantic colour (see `KNOWN_PYTHON_DRIFT` in that test file).
# Duplicated as literals here (not imported) to keep src/arcade/
# dependency-free from the backend package.
BG_COLOR: Final[tuple[int, int, int]] = (18, 17, 39)  # #121127 PRIMARY_BG
CONTENT_BG: Final[tuple[int, int, int]] = (24, 22, 51)  # #181633 CONTENT_BG (panels)
SECONDARY_BG: Final[tuple[int, int, int]] = (30, 27, 75)  # #1e1b4b SECONDARY_BG
BORDER_COLOR: Final[tuple[int, int, int]] = (45, 45, 58)  # #2d2d3a BORDER
TEXT_PRIMARY: Final[tuple[int, int, int]] = (255, 255, 255)  # #ffffff
TEXT_SECONDARY: Final[tuple[int, int, int]] = (209, 213, 219)  # #d1d5db
TEXT_TERTIARY: Final[tuple[int, int, int]] = (156, 163, 175)  # #9ca3af
ACCENT: Final[tuple[int, int, int]] = (167, 139, 250)  # #a78bfa purple
SUCCESS: Final[tuple[int, int, int]] = (16, 185, 129)  # #10b981 emerald
WARNING: Final[tuple[int, int, int]] = (245, 158, 11)  # #f59e0b amber
DANGER: Final[tuple[int, int, int]] = (239, 68, 68)  # #ef4444 red
INFO: Final[tuple[int, int, int]] = (59, 130, 246)  # #3b82f6 blue

# --- Typography (arcade.Text font_name accepts a fallback tuple) ---------
FONT_BODY: Final[tuple[str, ...]] = ("Inter", "Segoe UI", "Arial")
FONT_TITLE: Final[tuple[str, ...]] = ("Exo 2", "Inter", "Segoe UI", "Arial")

# --- Track rendering ------------------------------------------------------
TRACK_EDGE_COLOR: Final[tuple[int, int, int]] = (150, 150, 150)
TRACK_EDGE_WIDTH: Final[int] = 4
TRACK_FILL_COLOR: Final[tuple[int, int, int]] = (40, 40, 44)
DRS_COLOR: Final[tuple[int, int, int]] = (0, 220, 0)
DRS_WIDTH: Final[int] = 5
FINISH_CHEQUER_SEGMENTS: Final[int] = 20
FINISH_CHEQUER_WIDTH: Final[int] = 6
TRACK_WIDTH_WORLD: Final[float] = 200.0
TRACK_INTERP_REF: Final[int] = 4000
TRACK_INTERP_EDGE: Final[int] = 2000

# --- Tyre compounds (FastF1 int codes) -----------------------------------
COMPOUND_COLORS: Final[dict[int, tuple[int, int, int]]] = {
    0: (230, 50, 50),  # SOFT
    1: (230, 200, 50),  # MEDIUM
    2: (230, 230, 230),  # HARD
    3: (60, 200, 60),  # INTERMEDIATE
    4: (60, 130, 230),  # WET
}
COMPOUND_LETTERS: Final[dict[int, str]] = {
    0: "S",
    1: "M",
    2: "H",
    3: "I",
    4: "W",
}
COMPOUND_NAMES: Final[dict[int, str]] = {
    0: "SOFT",
    1: "MEDIUM",
    2: "HARD",
    3: "INTER",
    4: "WET",
}

# --- Car rendering --------------------------------------------------------
CAR_RADIUS: Final[float] = 7.0
CAR_BORDER_WIDTH: Final[float] = 2.0
CAR_BORDER_COLOR: Final[tuple[int, int, int]] = (255, 255, 255)
CAR_LABEL_FONT_SIZE: Final[int] = 11

# --- Background cars (all 20 dots when "show all" toggle is on) ----------
# Rendered smaller and less saturated than the featured main/rival dots
# so the eye still tracks the selected driver(s) while having full field
# context. Toggled with the ``A`` key at runtime (see ControlsLegend).
CAR_BG_RADIUS: Final[float] = 3.8
CAR_BG_ALPHA: Final[int] = 170

# --- Progress bar flag colors --------------------------------------------
FLAG_COLORS: Final[dict[str, tuple[int, int, int]]] = {
    "yellow_flag": WARNING,
    "red_flag": DANGER,
    "safety_car": WARNING,
    "vsc": (245, 158, 11),
    "dnf": DANGER,
    "progress_fill": ACCENT,
    "lap_marker": BORDER_COLOR,
    "background": CONTENT_BG,
    "playhead": TEXT_PRIMARY,
}

# --- Paths ----------------------------------------------------------------
# Resolved through data_cache rather than from __file__, so the arcade lands in
# the same directory as every other surface and honours $F1_STRAT_DATA_ROOT
# (docker-compose sets it to /app/data). The FastF1 cache in particular was
# fragmented: this surface wrote 3.8 GB under data/cache/fastf1 while the
# backend kept its own 274 MB copy of the same sessions, so whichever one
# touched a race first paid the full parse cost and the other paid it again.
REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[2]
FASTF1_CACHE_DIR: Final[Path] = get_data_root() / "cache" / "fastf1"
ARCADE_CACHE_DIR: Final[Path] = get_data_root() / "cache" / "arcade"
# v11: the all-NaN pedal channel fix, which runs at BUILD time. A v10 cache
# built before it has the wrong multiplier baked in and the fix cannot reach
# it. That is the whole reason: `FrameData.rel_dist` was not affected, since
# it is still derived and pickled per frame. What #863 removed was the raw
# `RelativeDistance` extraction in the worker intermediate, which nothing
# had consumed since v10 and which left the cached bytes identical.
# **The obvious guard cannot enforce this, and one tried.** `CACHE_VERSION != "v12"` is true
# forever from the commit that writes it and can never see the NEXT change that forgets to
# bump. A golden test keyed to the version COULD catch part of it (pin a small rebuilt
# session's frames against a fixture and require the version to move when they do), and is
# not written because rebuilding a session costs 254 s. Until it is, the obligation lives
# here: if the bytes a rebuild would produce differ from the bytes on disk, this string moves
# in the SAME commit, or the fix reaches nobody who already has a pickle.
# v15: a NaN Rainfall sample now pickles as None instead of "WET" (#1087). The bytes differ only
# for a session that HAS a NaN weather row, which is exactly the session the fix exists for, so
# leaving the version at v14 would have delivered the fix to everyone except the affected.
CACHE_VERSION: Final[str] = "v15"  # + the shared lap-boundary sample sorts stably (#1069)

# --- Multiprocessing pool -------------------------------------------------
# Serial by default: Windows spawn + pickling a loaded session across 8
# workers has hung in cold-cache runs. Flip to >1 once FastF1 is warm.
POOL_SIZE: Final[int] = 1


# --- Telemetry stream (arcade -> dashboard process) ----------------------
# Version of the broadcast payload's shape. Bump it whenever a key is
# renamed, removed, or changes meaning; adding a key an old consumer can
# ignore does not need a bump. A consumer that reads a version it does not
# know should say so rather than silently render a field it guessed at.
#
# 2 (#1048): the telemetry block went from two role-keyed spans,
# `{main: [...], rival: [...]}`, to one span per driver under
# `{drivers: {CODE: [...]}}`. `rewound` and `dropped` stay where they were:
# they describe the tick, not a car. Read the pair a v1 consumer wanted as
# `telemetry.drivers[arcade.driver_main]` and `[arcade.driver_rival]`.
STREAM_SCHEMA_VERSION: Final[int] = 2
STREAM_HOST: Final[str] = os.environ.get("F1_STREAM_HOST", "127.0.0.1")
STREAM_PORT: Final[int] = int(os.environ.get("F1_STREAM_PORT", "9998"))
# Broadcast every N arcade frames. At 60 FPS on_update, N=6 gives ~10 Hz,
# smooth enough for the live charts without saturating localhost.
STREAM_BROADCAST_EVERY_N_FRAMES: Final[int] = 6
# Ceiling on how many telemetry samples one tick may carry. Smooth playback
# cannot reach it: the widest span the clock produces is a ~0.1 s broadcast
# interval at 8x with the 3x seek multiplier, about 60 frames.
#
# What DOES reach it is a click on the progress bar, which sets the frame
# index directly and can jump tens of thousands of frames; a process stall is
# the rarer second case. Either way the frames in between never go out, so the
# tick publishes `dropped` alongside the span - a forward jump is otherwise
# invisible to a consumer, which sees a contiguous `seq` and a forward clock.
#
# The cap is not free: a bigger message fills a stalled subscriber's socket
# buffer in fewer broadcasts. Sends run on stream.py's own sender thread and
# never on the pyglet one (measured: 300 broadcasts against a client that
# never reads cost the caller 0.98 ms at worst), so the replay cannot freeze;
# the per-client send timeout is what prunes a subscriber that stops reading.
# The cap bounds what one tick can carry, the timeout bounds what one client
# can cost.
STREAM_MAX_SPAN_FRAMES: Final[int] = 250
# Cap how many LapDecision entries the broadcast history tail keeps.
STREAM_HISTORY_TAIL: Final[int] = 30

# --- Menu view ------------------------------------------------------------
MENU_TITLE: Final[str] = "F1 STRATLAB"
MENU_ROW_HEIGHT: Final[int] = 40
# Half the space between the label column's right edge and the value column's
# left edge. Both columns are anchored off the window's centre axis, so this is
# what separates them.
MENU_GUTTER: Final[int] = 20
# Breathing room the focus fill adds beyond the form's own content. The accent
# rule under the focused row takes the content extent with no padding, so the
# fill reads as a band around the text and the rule as an underline of it.
MENU_FOCUS_PAD: Final[int] = 24
MENU_LABEL_FONT: Final[int] = 13
MENU_VALUE_FONT: Final[int] = 15
MENU_HINT_FONT: Final[int] = 11
MENU_TITLE_FONT: Final[int] = 32
MENU_SUBTITLE_FONT: Final[int] = 13
MENU_STATUS_FONT: Final[int] = 13
# Every length above is what the menu draws at SCREEN_HEIGHT, and the view
# multiplies all of them by the window's height over that. Without it a taller
# window bought nothing but two larger empty gaps: at 1920x1080 the form was the
# same 280 px it is at 720, stranded between a title alone at the top and a hint
# alone at the bottom.
#
# The bounds are what keeps the scaling useful at both ends. Below the floor the
# type stops being readable, which is the opposite of the point; above the
# ceiling the type would keep growing past any useful size. The ceiling sits at
# 2.0 so that a maximised window on a 1440p display is still inside the range;
# above it the window keeps growing and the form does not, which is the same
# void this scaling exists to remove, deliberately traded for a legibility cap.
MENU_SCALE_MIN: Final[float] = 0.85
MENU_SCALE_MAX: Final[float] = 2.0
# Extra pitch inserted where one group of rows ends and the next begins. The
# seven options are three different kinds of decision (which race, which cars,
# whether the agent pipeline runs) and used to render as one undifferentiated
# list.
MENU_GROUP_GAP: Final[int] = 22
# How much larger an emphasised row draws than its siblings. One row carries it:
# the strategy toggle, which decides whether the multi-agent layer runs at all
# and so is the only choice on the form that changes what the replay IS.
MENU_EMPHASIS: Final[float] = 1.3
# Distances from the window's own edges, at scale 1.0.
MENU_TITLE_TOP: Final[int] = 80
MENU_SUBTITLE_TOP: Final[int] = 112
MENU_HINT_BOTTOM: Final[int] = 60
MENU_STATUS_BOTTOM: Final[int] = 120
STRATEGY_REQUIRED_YEAR: Final[int] = 2025

# --- 2025 grid: driver code -> team --------------------------------------
# Mirrors `data/processed/laps_featured_2025.parquet` (unique Driver/Team
# pairs). Consumed by MenuView to auto-fill the team field when the user
# types a driver code (the same UX as the CLI, where team is derived from
# the driver argument). Mid-season moves (TSU Racing Bulls -> Red Bull, LAW the
# opposite) resolved to each driver's end-of-season team.
DRIVER_TO_TEAM_2025: Final[dict[str, str]] = {
    "VER": "Red Bull Racing",
    "TSU": "Red Bull Racing",
    "NOR": "McLaren",
    "PIA": "McLaren",
    "LEC": "Ferrari",
    "HAM": "Ferrari",
    "RUS": "Mercedes",
    "ANT": "Mercedes",
    "ALO": "Aston Martin",
    "STR": "Aston Martin",
    "ALB": "Williams",
    "SAI": "Williams",
    "GAS": "Alpine",
    "DOO": "Alpine",
    "COL": "Alpine",
    "HUL": "Kick Sauber",
    "BOR": "Kick Sauber",
    "BEA": "Haas F1 Team",
    "OCO": "Haas F1 Team",
    "LAW": "Racing Bulls",
    "HAD": "Racing Bulls",
}

# --- Grand Prix names (round -> short label) -----------------------------
GP_NAMES: Final[dict[int, str]] = {
    1: "Bahrain",
    2: "SaudiArabia",
    3: "Australia",
    4: "Japan",
    5: "China",
    6: "Miami",
    7: "Monaco",
    8: "Canada",
    9: "Spain",
    10: "Austria",
    11: "Britain",
    12: "Hungary",
    13: "Belgium",
    14: "Netherlands",
    15: "Italy",
    16: "Singapore",
    17: "Mexico",
    18: "Brazil",
    19: "LasVegas",
    20: "AbuDhabi",
    21: "Qatar",
    22: "USA",
    23: "Monza",
}

# --- GP name → on-disk folder (FastF1 Location) --------------------------
# The CLI / backend store per-race FastF1 data under ``data/raw/<year>/<loc>/``
# where ``<loc>`` is the circuit Location FastF1 emits (``Sakhir`` for Bahrain,
# ``Melbourne`` for Australia, ...). The menu / CLI in arcade currently uses
# the country-ish labels in ``GP_NAMES`` for display; this mapping translates
# them to the disk name for the local strategy pipeline so it can find the
# race directory. Pass-through entries are included so a user who already
# types a Location (e.g. ``--gp Melbourne`` from the CLI shortcut) does not
# trip the lookup.
# --- Canonical per-year calendar --- data/tire_compounds_by_race.json ---
# Memory rule: ``data/tire_compounds_by_race.json`` is THE canonical source
# for per-year GP metadata (see MEMORY.md → feedback_check_data_folder). The
# arcade used to carry a hand-maintained ``GP_NAMES`` mapping that drifted
# from the active season (``GP_NAMES[3] == "Australia"`` but 2025 round 3 is
# Suzuka); ``get_gp_names(year)`` below reads the JSON and returns an
# ``{round: Location}`` dict for the requested year, so menu/viewer/strategy
# paths always resolve the right race without another hardcoded table.

_GP_NAMES_JSON_PATH: Final[Path] = (
    Path(__file__).resolve().parents[2] / "data" / "tire_compounds_by_race.json"
)
_gp_names_cache: dict[int, dict[int, str]] = {}


def get_gp_names(year: int) -> dict[int, str]:
    """Return ``{round_number: Location}`` for ``year`` (1-indexed rounds).

    Reads the canonical ``data/tire_compounds_by_race.json`` and assumes
    the insertion order of the keys matches the calendar order (the
    builder writes them in round order, verified for 2023/2024/2025).
    Falls back to the hardcoded ``GP_NAMES`` table (2024 layout) when the
    JSON is missing or the year is absent, so the arcade still boots
    without the data artifact.
    """
    if year in _gp_names_cache:
        return _gp_names_cache[year]
    try:
        with open(_GP_NAMES_JSON_PATH, encoding="utf-8") as fh:
            raw = json.load(fh)
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        logger.warning("GP calendar JSON unreadable (%s) — using hardcoded fallback", exc)
        return GP_NAMES
    year_block = raw.get(str(year))
    if not isinstance(year_block, dict):
        logger.warning(
            "No calendar for %d in %s — using hardcoded fallback", year, _GP_NAMES_JSON_PATH.name
        )
        return GP_NAMES
    mapping = {i + 1: name for i, name in enumerate(year_block.keys()) if not name.startswith("_")}
    _gp_names_cache[year] = mapping
    return mapping


GP_TO_LOCATION: Final[dict[str, str]] = {
    "Bahrain": "Sakhir",
    "SaudiArabia": "Jeddah",
    "Australia": "Melbourne",
    "Japan": "Suzuka",
    "China": "Shanghai",
    "Miami": "Miami_Gardens",
    "Monaco": "Monaco",
    "Canada": "Montréal",
    "Spain": "Barcelona",
    "Austria": "Spielberg",
    "Britain": "Silverstone",
    "Hungary": "Budapest",
    "Belgium": "Spa-Francorchamps",
    "Netherlands": "Zandvoort",
    "Italy": "Monza",
    "Singapore": "Marina_Bay",
    "Mexico": "Mexico_City",
    "Brazil": "São_Paulo",
    "LasVegas": "Las_Vegas",
    "AbuDhabi": "Yas_Island",
    "Qatar": "Lusail",
    "USA": "Austin",
    "Monza": "Monza",
    "Imola": "Imola",
}

# DRS codes that mean the wing is OPEN, and the one home for that fact.
#
# From FastF1's own channel documentation (`fastf1/_api.py`, `car_data`'s docstring):
# 0 and 1 are Off, **8 is "Detected, Eligible once in Activation Zone"**, and
# **10 / 12 / 14 are all On** with an undocumented distinction between them. So the
# open set is the three On codes, and 8 is deliberately NOT in it: a car that may use
# DRS is not a car with an open wing.
#
# **⚠️ The comment that stood here through `track.py` and this file said "Value 10
# (eligible) covers the short stretch between the detection line and the moment the
# wing actually opens".** Both halves were wrong against the source above - 10 is On,
# and the eligible code is the 8 this set EXCLUDES - and the error is not cosmetic: it
# told the next reader the set already contains a not-yet-open code, which is an
# invitation to add 8 "for consistency". `tests/surfaces/test_arcade_telemetry_span.py`
# exists to refuse exactly that, because publishing 8 as open draws an open wing on a
# closed one.
#
# **It lives here rather than in `track.py` because five places in this repo decode it,
# and a sixth cannot.** In Python: the track overlay's zones (`track.py`), the driver
# telemetry box's label and colour (`overlays.py`, two sites), the wire's decoded
# `drs_open` (`app.py::_frame_to_telemetry`), and the FIA-doc audit script
# (`scripts/verify_drs_zones.py`). `OwnCarTraces` refused to fork the set into
# TypeScript, correctly, and the way to honour that refusal is to decode it once here.
#
# **The sixth is out of reach and worth naming rather than hiding.** The telemetry
# webapp's `channels.ts::binarizeDrs` tests `value >= 10`; it is TypeScript behind a git
# submodule boundary, so it cannot import this and has to be kept in step by hand. Note
# `>= 10` is not equivalent to this set - it admits 11 and 13, which FastF1 never emits
# and the arcade's resampler manufactures (#1002).
#
# Two claims about this constant have already been wrong, both caught by adversarial
# gates: the commit that created it said "two subsystems" while leaving `overlays.py`'s
# pair literal, and the sentence replacing that said "FOUR call sites" while a fifth sat
# in `scripts/` with a `>= 10` threshold the AST census could not see. The census now
# looks for the threshold form too, and asserts it visited files at all.
DRS_OPEN_CODES: Final[frozenset[int]] = frozenset({10, 12, 14})

# The code that means "you may open it in the next zone", which is not "it is open".
#
# Named because two call sites compared against a bare `8`, and because it is the
# boundary the open set is defined AGAINST: the guard that keeps 8 out of
# `DRS_OPEN_CODES` reads better when both sides have names.
DRS_ELIGIBLE_CODE: Final[int] = 8
