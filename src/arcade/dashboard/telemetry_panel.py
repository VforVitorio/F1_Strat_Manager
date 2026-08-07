"""Live telemetry panel: 2×2 grid of pyqtgraph charts with fixed axes.

Layout mirrors the Streamlit circuit-comparison page:

    ┌────────────────────┬────────────────────┐
    │ Delta Time         │ Speed              │
    │ (2-driver only)    │ (main + rival)     │
    ├────────────────────┼────────────────────┤
    │ Brake Pressure     │ Throttle           │
    │ (main + rival)     │ (main + rival)     │
    └────────────────────┴────────────────────┘

Axes are locked so only the lines move between updates: a moving
viewport on every broadcast is visually noisy and masks where on the
lap the car actually is. X is fixed to ``[0, circuit_length]`` and Y
is per-metric:

- Speed: 0-360 km/h (Monza's straight tops out around 357).
- Brake / Throttle: -5 to 105 % with tiny padding so traces at 0 and
  100 do not kiss the frame.
- Delta: ±3 s, generous for one lap, clipped when the series wanders.

Each chart carries a title label above the plot and a mini colour
legend (``MAIN · VER`` · ``RIVAL · LEC``) so the user never has to
guess which trace is which. Per-lap buffers clear on lap change so a
chart always shows the ongoing lap only.
"""

from __future__ import annotations

import bisect
from typing import Any

import pyqtgraph as pg
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from src.arcade.dashboard.theme import (
    BG_COLOR,
    BORDER_COLOR,
    DANGER,
    INFO,
    SUCCESS,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
    TEXT_TERTIARY,
    WARNING,
    hex_str,
    qcolor,
)

# --- Sample buffer ------------------------------------------------------

Bucket = dict[int, tuple[float, float, float, float]]
# Tuple layout: (t, speed, throttle, brake)
_BUCKET_T, _BUCKET_S, _BUCKET_TH, _BUCKET_BR = 0, 1, 2, 3

# --- Fixed Y ranges per chart ------------------------------------------
_SPEED_Y_RANGE: tuple[float, float] = (0.0, 360.0)  # Monza peak ~357 km/h
_BRAKE_Y_RANGE: tuple[float, float] = (-5.0, 105.0)
_THROTTLE_Y_RANGE: tuple[float, float] = (-5.0, 105.0)
_DELTA_Y_RANGE: tuple[float, float] = (-3.0, 3.0)

_DEFAULT_X_RANGE: tuple[float, float] = (0.0, 5500.0)  # fallback until broadcast


class TelemetryPanel(QFrame):
    """2×2 pyqtgraph grid + top header with lap + driver indicators."""

    def __init__(self) -> None:
        super().__init__()
        self.setProperty("card", True)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        root = QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(8)

        # --- Top header: lap counter + driver chips --------------------
        header = QWidget()
        header_lay = QHBoxLayout(header)
        header_lay.setContentsMargins(2, 0, 2, 0)
        header_lay.setSpacing(10)
        self._lap_label = QLabel("LAP —")
        self._lap_label.setStyleSheet(
            f"color: {hex_str(TEXT_PRIMARY)}; font-size: 13px; "
            "font-weight: 800; letter-spacing: 1px;"
        )
        self._main_chip = _driver_chip("—", INFO)
        self._vs_label = QLabel("vs")
        self._vs_label.setStyleSheet(f"color: {hex_str(TEXT_TERTIARY)}; font-size: 11px;")
        self._rival_chip = _driver_chip("—", WARNING)
        self._vs_label.hide()
        self._rival_chip.hide()
        header_lay.addWidget(self._lap_label)
        header_lay.addSpacing(10)
        header_lay.addWidget(self._main_chip)
        header_lay.addWidget(self._vs_label)
        header_lay.addWidget(self._rival_chip)
        header_lay.addStretch()
        root.addWidget(header)

        # --- 2×2 grid of charts ---------------------------------------
        grid = QGridLayout()
        grid.setSpacing(10)
        # Delta chart: F1-broadcast convention — main driver is the flat
        # y=0 reference line (rendered in main colour), rival is the
        # wavy trace showing how the gap to main evolves along the lap.
        # Positive y means rival is SLOWER (behind main's pace at that
        # distance point); negative means rival is faster (ahead).
        (
            delta_wrapper,
            self._delta_plot,
            self._delta_main,  # unused — main is represented by zero line
            self._delta_rival,  # the wavy trace (rival − main)
            self._delta_main_legend,
            self._delta_rival_legend,
        ) = self._make_chart(
            "Δ Time (s)",
            "(rival − main)",
            INFO,
            WARNING,
            _DELTA_Y_RANGE,
        )
        # Main is the zero reference; drop the separate "main" line and
        # replace it with a solid horizontal line at y=0 coloured as the
        # main driver. Rival is visible by default on this chart.
        self._delta_plot.removeItem(self._delta_main)
        self._delta_main = None
        self._delta_main_reference = pg.InfiniteLine(
            pos=0,
            angle=0,
            pen=pg.mkPen(QColor(*INFO), width=2),
        )
        self._delta_plot.addItem(self._delta_main_reference)
        self._delta_rival.setVisible(True)
        self._delta_rival_legend.show()
        self._delta_placeholder = QLabel("single-driver mode")
        self._delta_placeholder.setAlignment(Qt.AlignCenter)
        self._delta_placeholder.setStyleSheet(
            f"color: {hex_str(TEXT_TERTIARY)}; font-style: italic; font-size: 12px;"
        )
        self._delta_placeholder.hide()
        # Stack the placeholder on top of the plot via the wrapper's layout.
        delta_wrapper.layout().addWidget(self._delta_placeholder)

        (
            speed_wrapper,
            self._speed_plot,
            self._speed_main,
            self._speed_rival,
            self._speed_main_legend,
            self._speed_rival_legend,
        ) = self._make_chart(
            "Speed",
            "km/h",
            INFO,
            WARNING,
            _SPEED_Y_RANGE,
        )
        (
            brake_wrapper,
            self._brake_plot,
            self._brake_main,
            self._brake_rival,
            self._brake_main_legend,
            self._brake_rival_legend,
        ) = self._make_chart(
            "Brake Pressure",
            "%",
            DANGER,
            WARNING,
            _BRAKE_Y_RANGE,
        )
        (
            throttle_wrapper,
            self._throttle_plot,
            self._throttle_main,
            self._throttle_rival,
            self._throttle_main_legend,
            self._throttle_rival_legend,
        ) = self._make_chart(
            "Throttle",
            "%",
            SUCCESS,
            WARNING,
            _THROTTLE_Y_RANGE,
        )

        grid.addWidget(delta_wrapper, 0, 0)
        grid.addWidget(speed_wrapper, 0, 1)
        grid.addWidget(brake_wrapper, 1, 0)
        grid.addWidget(throttle_wrapper, 1, 1)
        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 1)
        grid.setRowStretch(0, 1)
        grid.setRowStretch(1, 1)
        root.addLayout(grid, 1)

        # State
        self._main_buffer: Bucket = {}
        self._rival_buffer: Bucket = {}
        self._current_lap: int | None = None
        self._x_range_set: bool = False
        # Default X range until a broadcast tells us the real circuit length.
        self._apply_x_range(*_DEFAULT_X_RANGE)

    # --- Public update slot -----------------------------------------

    def update_from(self, data: dict[str, Any] | None) -> None:
        """Accept the full broadcast dict: we need ``arcade.telemetry`` and
        ``arcade.circuit_length_m`` from the same payload."""
        if not data:
            self._reset()
            return
        arcade = data.get("arcade") or {}
        telemetry = arcade.get("telemetry") or {}
        # Both are SPANS: every sample the replay clock crossed since the
        # previous tick, oldest first. Empty means "no new samples this
        # tick" (paused, or a backwards seek), never "no driver" — treating
        # an empty span as absence would blank the charts ten times a
        # second while the user holds pause.
        main_span = telemetry.get("main") or []
        rival_span = telemetry.get("rival") or []

        # Lock the X axis the first time we know the circuit length.
        if not self._x_range_set:
            length = arcade.get("circuit_length_m")
            if length and float(length) > 100:
                self._apply_x_range(0.0, float(length))
                self._x_range_set = True

        # Header
        driver_main = arcade.get("driver_main") or "—"
        driver_rival = arcade.get("driver_rival")
        lap = arcade.get("lap")
        # A driver whose telemetry never places the car produces four empty
        # charts. Saying so beats a populated header over a blank grid, which
        # reads as a broken panel rather than as absent data.
        main_car = (arcade.get("drivers") or {}).get(driver_main) or {}
        blind = main_car.get("has_position") is False
        lap_text = f"LAP {lap}" if lap else "LAP —"
        self._lap_label.setText(f"{lap_text}  ·  NO POSITION DATA" if blind else lap_text)
        self._main_chip.set_code(driver_main)
        self._update_legend_codes(driver_main, driver_rival)
        if driver_rival:
            self._rival_chip.set_code(driver_rival)
            self._vs_label.show()
            self._rival_chip.show()
        else:
            self._vs_label.hide()
            self._rival_chip.hide()

        # A backwards seek invalidates everything held: the buffers are
        # keyed on distance-within-lap, so they contain samples for track
        # the car has not re-driven yet and nothing else would evict them.
        if telemetry.get("rewound"):
            self._reset_buffers()

        # The span can cross a lap boundary at high playback speed, so the
        # lap-change clear is applied per sample rather than once per tick.
        for sample in main_span:
            lap_n = int(sample.get("lap") or 0)
            if lap_n != self._current_lap:
                self._current_lap = lap_n
                self._main_buffer.clear()
                self._rival_buffer.clear()
            self._append(self._main_buffer, sample)

        # Only accumulate rival samples that match the main driver's
        # current lap. When the two drivers are on different laps (one
        # pitted, one lapped, one half a track ahead) the buffer would
        # otherwise mix samples whose ``t`` values come from different
        # laps, and the delta interpolation produces ~4-6 s spikes at
        # the point where the older-lap samples sort next to the newer
        # ones.
        for sample in rival_span:
            if int(sample.get("lap") or 0) == self._current_lap:
                self._append(self._rival_buffer, sample)

        if telemetry.get("dropped"):
            # A forward seek the span could not cover: the frames between
            # here and the last tick were never sent, so appending would
            # splice two unrelated parts of the race into one trace.
            self._reset_buffers()
        self._refresh_speed_brake_throttle(has_rival=bool(driver_rival))
        # Two-driver mode is a property of the session, not of whether this
        # particular tick happened to carry a rival sample — an empty span
        # while paused must not collapse the delta chart to its placeholder.
        self._refresh_delta(has_rival=bool(driver_rival))

    # --- Buffer ingestion -------------------------------------------

    @staticmethod
    def _append(buffer: Bucket, sample: dict[str, Any]) -> None:
        dist = sample.get("dist")
        if dist is None:
            return
        try:
            key = int(float(dist))
        except (TypeError, ValueError):
            return
        buffer[key] = (
            float(sample.get("t", 0.0) or 0.0),
            float(sample.get("speed", 0.0) or 0.0),
            float(sample.get("throttle", 0.0) or 0.0),
            float(sample.get("brake", 0.0) or 0.0),
        )

    # --- Chart refresh ----------------------------------------------

    def _refresh_speed_brake_throttle(self, has_rival: bool) -> None:
        main_xs, main_rows = self._sorted(self._main_buffer)
        rival_xs, rival_rows = self._sorted(self._rival_buffer)

        self._speed_main.setData(main_xs, [r[_BUCKET_S] for r in main_rows])
        self._brake_main.setData(main_xs, [r[_BUCKET_BR] for r in main_rows])
        self._throttle_main.setData(main_xs, [r[_BUCKET_TH] for r in main_rows])

        # Visibility follows the SESSION's rival, not this tick's buffer.
        # Keyed on the buffer, the three traces and their legends vanished
        # for the whole of a rewind hold and every lap change, which reads
        # as single-driver mode rather than as an empty moment. Same fix
        # `_refresh_delta` already got; this was its twin.
        if has_rival:
            self._speed_rival.setData(rival_xs, [r[_BUCKET_S] for r in rival_rows])
            self._brake_rival.setData(rival_xs, [r[_BUCKET_BR] for r in rival_rows])
            self._throttle_rival.setData(rival_xs, [r[_BUCKET_TH] for r in rival_rows])
            for line, legend in (
                (self._speed_rival, self._speed_rival_legend),
                (self._brake_rival, self._brake_rival_legend),
                (self._throttle_rival, self._throttle_rival_legend),
            ):
                line.setVisible(True)
                legend.show()
        else:
            for line, legend in (
                (self._speed_rival, self._speed_rival_legend),
                (self._brake_rival, self._brake_rival_legend),
                (self._throttle_rival, self._throttle_rival_legend),
            ):
                line.setVisible(False)
                legend.hide()

    def _refresh_delta(self, has_rival: bool) -> None:
        if not has_rival:
            self._delta_plot.hide()
            self._delta_placeholder.show()
            return
        self._delta_placeholder.hide()
        self._delta_plot.show()

        main_xs, main_rows = self._sorted(self._main_buffer)
        rival_xs, rival_rows = self._sorted(self._rival_buffer)
        if len(main_xs) < 2 or len(rival_xs) < 2:
            self._delta_rival.setData([], [])
            return
        rival_t_series = [r[_BUCKET_T] for r in rival_rows]
        deltas: list[float] = []
        xs_out: list[float] = []
        # Main is the reference line at y=0 — plot ``t_rival − t_main``
        # so a positive trace means rival is slower at that distance
        # (behind main's pace) and a negative trace means rival is
        # faster (ahead of main). Matches the F1-broadcast convention.
        for i, x in enumerate(main_xs):
            interp = _lerp_sorted(rival_xs, rival_t_series, x)
            if interp is None:
                continue
            deltas.append(interp - main_rows[i][_BUCKET_T])
            xs_out.append(x)
        self._delta_rival.setData(xs_out, deltas)

    # --- Axis control -----------------------------------------------

    def _apply_x_range(self, lo: float, hi: float) -> None:
        for plot in (
            self._delta_plot,
            self._speed_plot,
            self._brake_plot,
            self._throttle_plot,
        ):
            plot.setXRange(lo, hi, padding=0)

    # --- Chart factory ----------------------------------------------

    @staticmethod
    def _make_chart(
        title: str,
        subtitle: str,
        main_color: tuple[int, int, int],
        rival_color: tuple[int, int, int],
        y_range: tuple[float, float],
    ) -> tuple[
        QWidget,
        pg.PlotWidget,
        pg.PlotDataItem,
        pg.PlotDataItem,
        QLabel,
        QLabel,
    ]:
        """Build a chart wrapper: title + legend row on top, plot below.

        Returns the wrapper widget plus the plot/line/legend handles so
        the caller can keep references for updates."""
        wrapper = QWidget()
        wlay = QVBoxLayout(wrapper)
        wlay.setContentsMargins(0, 0, 0, 0)
        wlay.setSpacing(4)

        # Title row: "Speed   km/h               MAIN · VER  RIVAL · LEC"
        title_row = QHBoxLayout()
        title_row.setContentsMargins(4, 0, 4, 0)
        title_row.setSpacing(6)
        title_lbl = QLabel(title)
        title_lbl.setStyleSheet(
            f"color: {hex_str(TEXT_PRIMARY)}; font-size: 12px; font-weight: 700;"
        )
        subtitle_lbl = QLabel(subtitle)
        subtitle_lbl.setStyleSheet(f"color: {hex_str(TEXT_TERTIARY)}; font-size: 10px;")
        main_legend = _legend_chip("MAIN", "—", main_color)
        rival_legend = _legend_chip("RIVAL", "—", rival_color)
        rival_legend.hide()
        title_row.addWidget(title_lbl)
        title_row.addWidget(subtitle_lbl)
        title_row.addStretch()
        title_row.addWidget(main_legend)
        title_row.addWidget(rival_legend)
        wlay.addLayout(title_row)

        plot = pg.PlotWidget()
        plot.setBackground(qcolor(BG_COLOR))
        axis_pen = pg.mkPen(QColor(*BORDER_COLOR), width=1)
        for side in ("left", "bottom"):
            axis = plot.getAxis(side)
            axis.setPen(axis_pen)
            axis.setTextPen(QColor(*TEXT_SECONDARY))
        plot.setLabel("bottom", "Distance (m)", color="#d1d5db")
        # Subtler grid than the default (alpha 0.5 → 0.12) so the traces
        # dominate the read without losing orientation against the ticks.
        plot.getPlotItem().showGrid(x=True, y=True, alpha=0.12)
        # Lock both axes — X gets set once from broadcast, Y is fixed per metric.
        plot.getPlotItem().enableAutoRange(False)
        plot.setYRange(y_range[0], y_range[1], padding=0)
        plot.setMouseEnabled(x=False, y=False)  # no accidental pan/zoom
        plot.hideButtons()
        # The delta chart adds its own y=0 reference line in the main
        # driver's colour after this factory returns — no auto zero line
        # here to avoid stacking two lines on top of each other.
        wlay.addWidget(plot, 1)

        main_line = pg.PlotDataItem(pen=pg.mkPen(QColor(*main_color), width=2))
        rival_line = pg.PlotDataItem(pen=pg.mkPen(QColor(*rival_color), width=2, style=Qt.DashLine))
        plot.addItem(main_line)
        plot.addItem(rival_line)
        rival_line.setVisible(False)
        return wrapper, plot, main_line, rival_line, main_legend, rival_legend

    # --- Helpers ----------------------------------------------------

    def _update_legend_codes(self, main: str, rival: str | None) -> None:
        for lbl in (
            self._speed_main_legend,
            self._brake_main_legend,
            self._throttle_main_legend,
            self._delta_main_legend,
        ):
            lbl.set_code(main)
        if rival:
            for lbl in (
                self._speed_rival_legend,
                self._brake_rival_legend,
                self._throttle_rival_legend,
                self._delta_rival_legend,
            ):
                lbl.set_code(rival)

    def _reset_buffers(self) -> None:
        self._main_buffer.clear()
        self._rival_buffer.clear()
        self._current_lap = None

    def _reset(self) -> None:
        self._reset_buffers()
        for line in (
            self._speed_main,
            self._brake_main,
            self._throttle_main,
            self._delta_rival,
        ):
            line.setData([], [])
        for line, legend in (
            (self._speed_rival, self._speed_rival_legend),
            (self._brake_rival, self._brake_rival_legend),
            (self._throttle_rival, self._throttle_rival_legend),
        ):
            line.setVisible(False)
            legend.hide()

    @staticmethod
    def _sorted(buffer: Bucket) -> tuple[list[float], list[tuple[float, float, float, float]]]:
        if not buffer:
            return [], []
        xs = sorted(buffer.keys())
        rows = [buffer[x] for x in xs]
        return [float(x) for x in xs], rows


# --- Custom chip widgets -------------------------------------------------


class _DriverChip(QLabel):
    """Small rounded label that holds a driver code and colour."""

    def __init__(self, code: str, colour: tuple[int, int, int]) -> None:
        super().__init__(code)
        self._colour = colour
        self._apply_style()

    def set_code(self, code: str) -> None:
        self.setText(code or "—")
        self._apply_style()

    def _apply_style(self) -> None:
        self.setStyleSheet(
            f"color: {hex_str(self._colour)}; font-size: 12px; "
            "font-weight: 800; letter-spacing: 1px; "
            f"border: 1px solid {hex_str(self._colour)}; "
            "border-radius: 8px; padding: 2px 8px;"
        )


def _driver_chip(code: str, colour: tuple[int, int, int]) -> _DriverChip:
    return _DriverChip(code, colour)


class _LegendChip(QLabel):
    """``MAIN · VER`` small chip used in chart title rows."""

    def __init__(self, role: str, code: str, colour: tuple[int, int, int]) -> None:
        super().__init__()
        self._role = role
        self._colour = colour
        self.set_code(code)

    def set_code(self, code: str) -> None:
        self.setText(f"{self._role} · {code or '—'}")
        self.setStyleSheet(
            f"color: {hex_str(self._colour)}; font-size: 10px; "
            "font-weight: 700; letter-spacing: 0.5px;"
        )


def _legend_chip(role: str, code: str, colour: tuple[int, int, int]) -> _LegendChip:
    return _LegendChip(role, code, colour)


# --- Interpolation -------------------------------------------------------


def _lerp_sorted(xs: list[float], ys: list[float], x: float) -> float | None:
    """Linear interpolation of ``ys`` at ``x`` given sorted ``xs``.

    Returns ``None`` when ``x`` is outside the known range so the delta
    chart does not plot extrapolated values where the rival has not
    reached yet (or passed into a later sector)."""
    if not xs or x < xs[0] or x > xs[-1]:
        return None
    idx = bisect.bisect_left(xs, x)
    if idx < len(xs) and xs[idx] == x:
        return ys[idx]
    if idx == 0:
        return ys[0]
    x0, x1 = xs[idx - 1], xs[idx]
    y0, y1 = ys[idx - 1], ys[idx]
    if x1 == x0:
        return y0
    return y0 + (y1 - y0) * (x - x0) / (x1 - x0)
