"""Pace chart: predicted lap time vs actual, with P10/P90 CI band.

Embedded inside the Pace N25 AgentCard (the chart slot reserved in C5).
pyqtgraph is used instead of matplotlib because PlotDataItem/FillBetweenItem
redraws at 10 Hz without saturating the Qt event loop.

Data model (fed by ``MainWindow._pace_history``):

    {lap_number: {
        "actual":  float | None,   # lap_time_s from history_tail / latest
        "pred":    float | None,   # per_agent.pace.lap_time_pred
        "ci_p10":  float | None,
        "ci_p90":  float | None,
    }}

``update`` rebuilds the three plot items each lap (window size ≤30, so
the cost is negligible). Missing values are simply skipped: laps before
the dashboard connected will only have ``actual`` and the predicted line
/ CI band start from the first lap we saw the per-agent payload for.
"""

from __future__ import annotations

from typing import Any

import pyqtgraph as pg
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor

from src.arcade.dashboard.theme import (
    ACCENT,
    BG_COLOR,
    BORDER_COLOR,
    INFO,
    TEXT_SECONDARY,
    qcolor,
)


class PaceChart(pg.PlotWidget):
    """PlotWidget with three items: actual (solid blue), predicted
    (dashed purple), CI band (filled purple, alpha ~0.2)."""

    def __init__(self) -> None:
        super().__init__()
        self.setBackground(qcolor(BG_COLOR))
        self.setMinimumHeight(140)

        axis_pen = pg.mkPen(QColor(*BORDER_COLOR), width=1)
        for side in ("left", "bottom"):
            axis = self.getAxis(side)
            axis.setPen(axis_pen)
            axis.setTextPen(QColor(*TEXT_SECONDARY))
            axis.setStyle(tickFont=None)
        self.setLabel("left", "Lap time (s)", color="#d1d5db")
        self.setLabel("bottom", "Lap", color="#d1d5db")
        self.getPlotItem().showGrid(x=True, y=True, alpha=0.15)
        self.getPlotItem().enableAutoRange("y", True)

        # Pre-allocated plot items so ``update`` only pushes setData calls.
        self._pred = pg.PlotDataItem(
            pen=pg.mkPen(QColor(*ACCENT), width=2, style=Qt.DashLine),
            name="predicted",
        )
        self._actual = pg.PlotDataItem(
            pen=pg.mkPen(QColor(*INFO), width=2),
            name="actual",
        )
        self._p10 = pg.PlotDataItem(pen=pg.mkPen(QColor(*ACCENT), width=0))
        self._p90 = pg.PlotDataItem(pen=pg.mkPen(QColor(*ACCENT), width=0))
        band_color = QColor(*ACCENT)
        band_color.setAlpha(50)
        self._band = pg.FillBetweenItem(self._p10, self._p90, brush=band_color)
        self.addItem(self._band)
        self.addItem(self._p10)
        self.addItem(self._p90)
        self.addItem(self._pred)
        self.addItem(self._actual)

    def update_from(self, history: dict[int, dict[str, Any]]) -> None:
        """Rebuild the three items from the window's pace history dict.

        Lap times outside the plausible F1 range (30-200 s) are dropped
        so an occasional stub value cannot blow the Y-axis autoscale and
        flatten the real series to a hairline. The whole 30-lap window
        is rebuilt per update, cheap at these sizes (<1 ms)."""
        if not history:
            self._clear()
            return
        laps = sorted(history.keys())
        actual_x: list[float] = []
        actual_y: list[float] = []
        pred_x: list[float] = []
        pred_y: list[float] = []
        band_x: list[float] = []
        p10_y: list[float] = []
        p90_y: list[float] = []
        for lap in laps:
            row = history[lap]
            a = _sane_lap_time(row.get("actual"))
            p = _sane_lap_time(row.get("pred"))
            lo = _sane_lap_time(row.get("ci_p10"))
            hi = _sane_lap_time(row.get("ci_p90"))
            if a is not None:
                actual_x.append(float(lap))
                actual_y.append(a)
            if p is not None:
                pred_x.append(float(lap))
                pred_y.append(p)
            if lo is not None and hi is not None:
                band_x.append(float(lap))
                p10_y.append(lo)
                p90_y.append(hi)

        self._actual.setData(actual_x, actual_y)
        self._pred.setData(pred_x, pred_y)
        self._p10.setData(band_x, p10_y)
        self._p90.setData(band_x, p90_y)

    def _clear(self) -> None:
        self._actual.setData([], [])
        self._pred.setData([], [])
        self._p10.setData([], [])
        self._p90.setData([], [])


def _sane_lap_time(value: Any) -> float | None:
    """Accept only lap-time values plausible for an F1 race (30-200 s).

    Filters out ``None`` and the occasional TCN stub that can return a
    value orders of magnitude off on the first two laps. The lower bound
    30 s is well under the fastest modern F1 lap; 200 s (over 3 minutes)
    is loose enough to accommodate wet Monaco and VSC laps."""
    if value is None:
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    if 30.0 <= v <= 200.0:
        return v
    return None
