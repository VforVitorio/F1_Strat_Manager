"""Tire chart: lap time over laps with per-stint colour and cliff projection.

Embedded inside the Tire N26 ``AgentCard`` (chart slot reserved in C5).
The previous version plotted ``tyre_life`` (laps on the current set) on
the Y axis, which conveyed stint progression but never the *cost* of
running on those tyres. This rewrite mirrors the ``PaceChart`` model so
the user reads degradation the same way an engineer does on a pit-wall:

- **X axis** is the lap number, identical to ``PaceChart`` so the two
  cards line up vertically and the eye does not have to recalibrate.
- **Y axis** is the lap time in seconds: every climb on the curve is
  literal degradation, every drop is a fresh tyre.
- **One ``PlotDataItem`` per stint**, coloured with the Pirelli palette
  for that compound (red / yellow / white / green / blue). pyqtgraph
  draws no line between consecutive stints because each is its own item,
  so the compound change appears as a visual break instead of a
  misleading line that walks through the in/out laps.
- **A dashed white overlay** carries a 3-lap centred rolling mean of
  the actual lap times. It mirrors the dashed-line idiom from
  ``PaceChart`` (predicted vs actual) and gives the strategist the
  smoothed degradation trend without losing the per-lap colour story.
- **A translucent vertical band** between ``current_lap + p10`` and
  ``current_lap + p90`` from the latest ``TireOutput`` shades the lap
  range in which the cliff is expected. The single dashed marker at
  ``current_lap + p50`` collapses the previous three-line view into a
  cleaner "where is the cliff" annotation.

Data model (fed by ``MainWindow._tire_history``):

    [{lap: int, lap_time_s: float | None,
      tyre_life: float | None, compound: str | None}, ...]

The widget keeps no domain state of its own: the per-stint segment list
is rebuilt on every ``update_from`` call. At the dashboard's 30-lap
window the cost is well under a millisecond.
"""

from __future__ import annotations

from typing import Any, NamedTuple

import pyqtgraph as pg
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor

from src.arcade.dashboard.theme import (
    BG_COLOR,
    BORDER_COLOR,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
    TEXT_TERTIARY,
    WARNING,
    compound_color,
    qcolor,
)


class _Stint(NamedTuple):
    """A contiguous run of laps on a single compound.

    Holds the X (lap number) and Y (lap time in seconds) sequences for
    one ``PlotDataItem`` so the segmentation pass can be expressed as a
    flat for-loop without nesting list-builders inline. ``compound`` is
    the canonical upper-cased label used by ``compound_color`` to pick
    the Pirelli RGB.
    """

    compound: str
    xs: list[float]
    ys: list[float]


class TireChart(pg.PlotWidget):
    """PlotWidget that pairs per-stint lap-time segments with a cliff band.

    The widget is intentionally a thin renderer over ``MainWindow``'s
    ``_tire_history`` dict: every render rebuilds segments and the
    cliff annotations from scratch so a stale stint cannot survive a
    pit stop or a mid-stream reconnect. Pre-allocated items (the cliff
    band edges, the median marker, the dashed smoothing overlay) keep
    the per-update cost to ``setData`` and ``setValue`` calls; the
    per-stint ``PlotDataItem``s are added and removed on the fly because
    their count depends on how many compound changes the window sees.
    """

    # Cliff projections above this horizon are TCN early-stint noise —
    # hide the band and marker rather than stretch the X-axis to 50k laps.
    _CLIFF_MAX_SANE: float = 100.0

    def __init__(self) -> None:
        super().__init__()
        self.setBackground(qcolor(BG_COLOR))
        self.setMinimumHeight(140)

        axis_pen = pg.mkPen(QColor(*BORDER_COLOR), width=1)
        for side in ("left", "bottom"):
            axis = self.getAxis(side)
            axis.setPen(axis_pen)
            axis.setTextPen(QColor(*TEXT_SECONDARY))
        self.setLabel("left", "Lap time (s)", color="#d1d5db")
        self.setLabel("bottom", "Lap", color="#d1d5db")
        self.getPlotItem().showGrid(x=True, y=True, alpha=0.15)
        self.getPlotItem().enableAutoRange("y", True)

        # Per-stint segments and stint-boundary markers — rebuilt every
        # update because the count is data-driven.
        self._segments: list[pg.PlotDataItem] = []
        self._stint_markers: list[pg.InfiniteLine] = []

        # Cliff band — vertical translucent rectangle between p10 and p90
        # forward-projected laps. ``FillBetweenItem`` needs two
        # ``PlotDataItem`` edges, each of which is a 2-point vertical
        # segment whose Y endpoints we refresh from the visible Y range
        # so the band always covers the current chart area.
        edge_pen = pg.mkPen(QColor(*WARNING), width=0)
        self._cliff_lo = pg.PlotDataItem(pen=edge_pen)
        self._cliff_hi = pg.PlotDataItem(pen=edge_pen)
        band_color = QColor(*WARNING)
        band_color.setAlpha(40)
        self._cliff_band = pg.FillBetweenItem(
            self._cliff_lo,
            self._cliff_hi,
            brush=band_color,
        )
        self.addItem(self._cliff_band)
        self.addItem(self._cliff_lo)
        self.addItem(self._cliff_hi)

        # Median cliff marker — single dashed vertical line at p50.
        median_pen = pg.mkPen(QColor(*WARNING), width=2, style=Qt.DashLine)
        self._cliff_p50 = pg.InfiniteLine(pos=0.0, angle=90, pen=median_pen)
        self._cliff_p50.setVisible(False)
        self.addItem(self._cliff_p50)

        # Smoothed lap-time overlay — dashed translucent white line on
        # top of the per-stint segments, conveying the trend the engineer
        # reasons about without obscuring the raw per-lap colour story.
        trend_color = QColor(TEXT_PRIMARY[0], TEXT_PRIMARY[1], TEXT_PRIMARY[2], 150)
        self._actual_dashed = pg.PlotDataItem(
            pen=pg.mkPen(trend_color, width=1, style=Qt.DashLine),
            name="trend",
        )
        self.addItem(self._actual_dashed)

    # --- Public API ---------------------------------------------------

    def update_from(
        self,
        history: list[dict[str, Any]],
        current_lap: int | None,
        tire_out: dict[str, Any] | None,
    ) -> None:
        """Rebuild segments, smoothing overlay and cliff annotations.

        ``history`` is a chronological list of per-lap snapshots, one
        dict per lap with the ``lap``, ``lap_time_s`` and ``compound``
        keys at minimum. ``current_lap`` anchors the cliff band and
        median marker; ``tire_out`` carries the percentile projections
        from the latest ``TireOutput``. Missing or implausibly large
        values hide the band and marker without clearing the segments
        so the curve stays readable while the TCN warms up.
        """
        self._clear_segments()
        self._clear_stint_markers()

        stints = _build_stints(history)
        for stint in stints:
            colour = QColor(*compound_color(stint.compound))
            item = pg.PlotDataItem(
                stint.xs,
                stint.ys,
                pen=pg.mkPen(colour, width=2),
                symbol="o",
                symbolSize=4,
                symbolBrush=colour,
                symbolPen=pg.mkPen(colour, width=0),
                name=stint.compound,
            )
            self.addItem(item)
            self._segments.append(item)

        # Stint-boundary markers — one thin dashed vertical line per
        # compound change, kept faint because the colour break itself
        # already communicates the change.
        boundary_pen = pg.mkPen(
            QColor(TEXT_TERTIARY[0], TEXT_TERTIARY[1], TEXT_TERTIARY[2], 80),
            width=1,
            style=Qt.DashLine,
        )
        for stint in stints[1:]:
            if not stint.xs:
                continue
            line = pg.InfiniteLine(pos=stint.xs[0], angle=90, pen=boundary_pen)
            self.addItem(line)
            self._stint_markers.append(line)

        # Smoothed overlay — flatten every stint's lap times in order
        # and apply a 3-lap centred mean. Smoothing across stint
        # boundaries is acceptable because the dashed line is a trend
        # cue, not a per-lap claim; the per-stint solid lines own the
        # truth axis.
        all_xs: list[float] = []
        all_ys: list[float] = []
        for stint in stints:
            all_xs.extend(stint.xs)
            all_ys.extend(stint.ys)
        if all_ys:
            self._actual_dashed.setData(all_xs, _rolling_mean(all_ys, window=3))
        else:
            self._actual_dashed.setData([], [])

        # Cliff annotations.
        self._refresh_cliff(history, current_lap, tire_out)

        # X range — clamp to the actual lap window plus a few laps of
        # forward headroom (or the full sane cliff horizon when the
        # band is visible) so a bad p90 cannot stretch the view.
        self._anchor_x_range(history, current_lap, tire_out)

    # --- Cliff annotations -------------------------------------------

    def _refresh_cliff(
        self,
        history: list[dict[str, Any]],
        current_lap: int | None,
        tire_out: dict[str, Any] | None,
    ) -> None:
        """Position the cliff band edges and the median marker.

        Reads ``laps_to_cliff_p10/p50/p90`` from ``tire_out`` and
        renders the band only when both p10 and p90 are present and
        within the sane horizon. The band edges are drawn from the
        bottom to the top of the visible Y range so the rectangle
        spans the chart area regardless of autoscale.
        """
        if current_lap is None or not tire_out:
            self._hide_cliff()
            return

        p10 = _sane_cliff(tire_out.get("laps_to_cliff_p10"), self._CLIFF_MAX_SANE)
        p50 = _sane_cliff(tire_out.get("laps_to_cliff_p50"), self._CLIFF_MAX_SANE)
        p90 = _sane_cliff(tire_out.get("laps_to_cliff_p90"), self._CLIFF_MAX_SANE)
        cur = float(current_lap)

        ys = [_sane_lap_time(row.get("lap_time_s")) for row in history]
        ys = [y for y in ys if y is not None]
        if ys:
            y_lo = min(ys) - 1.0
            y_hi = max(ys) + 5.0
        else:
            # No history yet — fall back to a generous window so the
            # band still renders if the cliff is the only signal.
            y_lo, y_hi = 60.0, 130.0

        if p10 is not None and p90 is not None:
            x_p10 = cur + p10
            x_p90 = cur + p90
            self._cliff_lo.setData([x_p10, x_p10], [y_lo, y_hi])
            self._cliff_hi.setData([x_p90, x_p90], [y_lo, y_hi])
        else:
            self._cliff_lo.setData([], [])
            self._cliff_hi.setData([], [])

        if p50 is not None:
            self._cliff_p50.setValue(cur + p50)
            self._cliff_p50.setVisible(True)
        else:
            self._cliff_p50.setVisible(False)

    def _hide_cliff(self) -> None:
        self._cliff_lo.setData([], [])
        self._cliff_hi.setData([], [])
        self._cliff_p50.setVisible(False)

    # --- X axis -------------------------------------------------------

    def _anchor_x_range(
        self,
        history: list[dict[str, Any]],
        current_lap: int | None,
        tire_out: dict[str, Any] | None,
    ) -> None:
        """Clamp the X axis so a bad cliff value cannot blow the view.

        When the cliff band is visible the right edge extends to the
        sane cliff horizon so the band has room; otherwise the range
        tracks history plus three laps of headroom for the next tick.
        """
        xs: list[float] = [float(row.get("lap", 0)) for row in history]
        if current_lap is not None:
            xs.append(float(current_lap))
        if not xs:
            self.setXRange(0, 1, padding=0.05)
            return
        x_min = min(xs) - 0.5
        x_max = max(xs)
        cliff_visible = (
            tire_out is not None
            and current_lap is not None
            and _sane_cliff(tire_out.get("laps_to_cliff_p90"), self._CLIFF_MAX_SANE) is not None
        )
        if cliff_visible:
            x_max += self._CLIFF_MAX_SANE
        else:
            x_max += 3
        self.setXRange(x_min, x_max, padding=0.02)

    # --- Item lifecycle ----------------------------------------------

    def _clear_segments(self) -> None:
        for item in self._segments:
            self.removeItem(item)
        self._segments.clear()

    def _clear_stint_markers(self) -> None:
        for line in self._stint_markers:
            self.removeItem(line)
        self._stint_markers.clear()


# --- Module-level helpers ------------------------------------------------


def _sane_lap_time(value: Any) -> float | None:
    """Accept only lap-time values plausible for an F1 race (30-200 s).

    Filters out ``None`` and the occasional pipeline stub that can
    return a value orders of magnitude off on the first laps. Mirrors
    the helper in ``pace_chart`` rather than importing it so the two
    chart widgets stay independently substitutable.
    """
    if value is None:
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    if 30.0 <= v <= 200.0:
        return v
    return None


def _sane_cliff(value: Any, ceiling: float) -> float | None:
    """Return the cliff-projection value when it is strictly positive and
    below ``ceiling``; otherwise ``None``. Used to suppress TCN warm-up
    noise (negative or absurdly large projections) without injecting
    placeholder values into the chart.
    """
    if value is None:
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    if 0.0 < v <= ceiling:
        return v
    return None


def _rolling_mean(ys: list[float], window: int = 3) -> list[float]:
    """3-point centred rolling mean over a list of lap times.

    The window is small on purpose: heavier smoothing (5 or more)
    visibly lags the underlying trend in the 25-30-lap windows the
    dashboard renders, defeating the point of the overlay. ``min_periods=1``
    semantics: edges are averaged over whatever points exist so the line
    starts at lap 1 instead of lap ``window // 2``.
    """
    if not ys:
        return []
    n = len(ys)
    half = window // 2
    out: list[float] = []
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        chunk = ys[lo:hi]
        out.append(sum(chunk) / len(chunk))
    return out


def _build_stints(history: list[dict[str, Any]]) -> list[_Stint]:
    """Group a chronological history list into per-compound stints.

    A stint ends and a new one begins as soon as the compound label
    changes between two consecutive observed laps. Rows with no usable
    lap time are skipped so a missing measurement does not split a
    stint into two artificial halves; rows with no compound inherit
    the previous compound (so the chart never paints a "blank" segment
    when the broadcast briefly drops the field).
    """
    stints: list[_Stint] = []
    current: _Stint | None = None
    last_compound: str | None = None
    for row in history:
        y = _sane_lap_time(row.get("lap_time_s"))
        if y is None:
            continue
        try:
            x = float(row.get("lap", 0))
        except (TypeError, ValueError):
            continue
        compound = str(row.get("compound") or "").upper().strip() or last_compound or "MEDIUM"
        if current is None or compound != current.compound:
            current = _Stint(compound=compound, xs=[], ys=[])
            stints.append(current)
        current.xs.append(x)
        current.ys.append(y)
        last_compound = compound
    return stints
