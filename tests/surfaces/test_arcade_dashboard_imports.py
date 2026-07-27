"""
Import-smoke tests for ``src/arcade/dashboard/``.

The PySide6 dashboard was added in Phase 3.5 Proceso B as a 13-file package
spawned as a subprocess from the arcade. None of the behavioural surfaces
(charts, reasoning panels, telemetry buffers) are easy to exercise without
a running QApplication + TCP stream + live RaceReplayEngine, so this suite
only verifies that every module parses and imports cleanly when PySide6 is
available. A broken import here surfaces regressions in cross-module
dependencies (theme → cards → window) before the arcade launcher tries to
spawn the subprocess at race time.

The entire file is skipped when PySide6 is not installed (CI runners, headless
builds), which keeps the rest of the suite green on minimal environments.
Run with:
    pytest tests/test_arcade_dashboard_imports.py -v
"""

from __future__ import annotations

import importlib

import pytest

# Skip the whole module when PySide6 is not present — the dashboard is an
# optional frontend, not a hard dependency of the core TFG pipeline.
pytest.importorskip(
    "PySide6",
    reason="PySide6 not installed — arcade dashboard is an optional UI layer",
)
pytest.importorskip(
    "pyqtgraph",
    reason="pyqtgraph not installed — dashboard charts depend on it",
)


# ---------------------------------------------------------------------------
# Per-module import smoke — every file under src/arcade/dashboard/ must parse
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "module_path",
    [
        "src.arcade.dashboard.theme",
        "src.arcade.dashboard.agent_formatters",
        "src.arcade.dashboard.stream_client",
        "src.arcade.dashboard.scenario_bars",
        "src.arcade.dashboard.pace_chart",
        "src.arcade.dashboard.tire_chart",
        "src.arcade.dashboard.reasoning_tabs",
        "src.arcade.dashboard.agent_card",
        "src.arcade.dashboard.orchestrator_card",
        "src.arcade.dashboard.telemetry_panel",
        "src.arcade.dashboard.telemetry_window",
        "src.arcade.dashboard.window",
    ],
)
def test_dashboard_module_importable(module_path):
    """Each dashboard module must import without raising.

    A cold import exercises module-level pyqtgraph/QtWidgets symbols, so any
    ``from PySide6.QtFoo import Bar`` that has shifted between releases will
    surface as an ImportError here and block the run.
    """
    module = importlib.import_module(module_path)
    assert module is not None


# ---------------------------------------------------------------------------
# Theme entry points — the one module that is safe to touch without QApplication
# ---------------------------------------------------------------------------


def test_theme_exposes_palette_helpers():
    """theme.py must expose the palette + compound/flag HTML helpers the
    other cards consume. These names are the fragile surface — renaming any
    of them ripples through agent_card / orchestrator_card / reasoning_tabs.
    """
    from src.arcade.dashboard import theme

    assert hasattr(theme, "apply_dark_palette")
    assert callable(theme.apply_dark_palette)
    # Compound pill + flag chip HTML helpers are load-bearing for Qt rich
    # text labels (Tire card, Radio card, orchestrator plan strip).
    assert hasattr(theme, "compound_pill_html")
    assert hasattr(theme, "flag_chip_html")
    assert callable(theme.compound_pill_html)
    assert callable(theme.flag_chip_html)


def test_theme_palette_constants_are_rgb_triples():
    """Palette constants must stay RGB triples so QColor unpacking works."""
    from src.arcade.dashboard import theme

    for name in ("BG_COLOR", "CONTENT_BG", "ACCENT", "TEXT_PRIMARY"):
        value = getattr(theme, name)
        assert isinstance(value, tuple), f"{name} should be a tuple"
        assert len(value) == 3, f"{name} should have 3 channels, got {len(value)}"
        assert all(0 <= c <= 255 for c in value), f"{name} channels out of range"
