"""The two embedded charts' series, computed where the Qt widgets compute them.

`pace_chart.py` and `tire_chart.py` together are about 250 lines and only
part of that is drawing. The rest is judgement about what may be plotted:
a sanity window on lap times, per-stint segmentation, a centred rolling
mean, and a cliff band whose geometry is keyed off the CURRENT lap. All
of it is ported here so the client only places marks.

The one thing that does NOT come across is pyqtgraph's autoscale. ECharts
does its own, and the X clamp below is the part that mattered: without it
a bad p90 stretched the axis to the cliff horizon and flattened the real
series to a hairline.
"""

from __future__ import annotations

from typing import Any, NamedTuple

from src.arcade.palette import (
    ACCENT,
    INFO,
    TEXT_PRIMARY,
    TEXT_TERTIARY,
    WARNING,
    compound_color,
    hex_str,
)

# Anything outside this is a pipeline stub, not a lap. The lower bound is
# well under the fastest modern F1 lap; 200 s is loose enough for a wet
# Monaco or a Safety Car lap.
SANE_LAP_TIME_S: tuple[float, float] = (30.0, 200.0)
# Cliff projections above this horizon are TCN early-stint noise: the MC
# Dropout samples have no history to converge on and return tens of
# thousands of laps.
CLIFF_MAX_SANE_LAPS: float = 100.0

ACTUAL_COLOUR = hex_str(INFO)
PRED_COLOUR = hex_str(ACCENT)
BAND_COLOUR = hex_str(ACCENT)
TREND_COLOUR = hex_str(TEXT_PRIMARY)
CLIFF_COLOUR = hex_str(WARNING)
# TEXT_TERTIARY at the alpha 80/255 the Qt pen uses.
BOUNDARY_COLOUR = hex_str(TEXT_TERTIARY)
# The current lap, on both charts. The same TEXT_TERTIARY band 4 uses for its
# shared cursor and for the same reason: it marks where the car is NOW, and
# anything brighter competes with the series it annotates. SOLID, because a
# dashed vertical already means a compound boundary on the tyre chart.
CURSOR_COLOUR = hex_str(TEXT_TERTIARY)
BOUNDARY_OPACITY = 80 / 255


class _Stint(NamedTuple):
    """A contiguous run of laps on one compound, and its Pirelli colour."""

    compound: str
    colour: str
    points: list[list[float]]


def _sane_lap_time(value: Any) -> float | None:
    if value is None:
        return None
    try:
        seconds = float(value)
    except (TypeError, ValueError):
        return None
    low, high = SANE_LAP_TIME_S
    return seconds if low <= seconds <= high else None


def _sane_cliff(value: Any) -> float | None:
    """Strictly positive and inside the horizon, else unknown.

    `None` rather than a placeholder: a suppressed band is a band that is
    not drawn, and a zero would be a lap number the chart could plot.
    """
    if value is None:
        return None
    try:
        laps = float(value)
    except (TypeError, ValueError):
        return None
    return laps if 0.0 < laps <= CLIFF_MAX_SANE_LAPS else None


def build_pace_series(
    history: dict[int, dict[str, Any]],
    x_range: list[float] | None = None,
    current_lap: int | None = None,
) -> dict[str, Any]:
    """Actual, predicted and the P10-P90 band, each skipping what it lacks.

    Three independent series rather than one table with holes: laps before
    the window connected carry only an actual, and the predicted line has
    to start where the per-agent payload first arrived rather than draw a
    line through zero.
    """
    actual: list[list[float]] = []
    pred: list[list[float]] = []
    band: list[list[float]] = []
    predicted_laps: list[int] = []
    for lap in sorted(history):
        row = history[lap]
        observed = _sane_lap_time(row.get("actual"))
        predicted = _sane_lap_time(row.get("pred"))
        low = _sane_lap_time(row.get("ci_p10"))
        high = _sane_lap_time(row.get("ci_p90"))
        if observed is not None:
            actual.append([float(lap), observed])
        if predicted is not None:
            pred.append([float(lap), predicted])
            predicted_laps.append(lap)
        if low is not None and high is not None:
            band.append([float(lap), low, high])
    return {
        "actual": actual,
        "pred": pred,
        "band": band,
        "actual_colour": ACTUAL_COLOUR,
        "pred_colour": PRED_COLOUR,
        "band_colour": BAND_COLOUR,
        # The SAME lap axis the tyre chart draws, and the same current-lap
        # mark. They sit side by side measuring the same quantity, and they
        # used to disagree: this one autoranged to 12-24 while its neighbour
        # locked 12.5-35, so "now" sat at 95 % of one plot and 47 % of the
        # other and neither said which column it was.
        "x_range": x_range,
        "current_lap": None if current_lap is None else float(current_lap),
        "cursor_colour": CURSOR_COLOUR,
        # The newest lap that carries a PREDICTION, which is not the newest lap
        # on the chart. A tick with no `per_agent` block still delivers the
        # actual lap time, so the solid line keeps advancing while the dashed
        # one and its band stop where the last prediction was - and nothing on
        # the chart said so. The renderer dims the two stale series and prints
        # the lap.
        #
        # **Only the PACE chart gets this.** The tyre chart's own stale element
        # is the cliff band, which is recomputed from `per_agent.tire` every
        # tick and simply disappears when there is none; its trend is a rolling
        # mean of OBSERVED lap times and keeps advancing, so dimming it would
        # claim staleness about live data.
        "prediction_lap": float(max(predicted_laps)) if predicted_laps else None,
    }


def _build_stints(rows: list[dict[str, Any]]) -> list[_Stint]:
    """Group chronological rows into per-compound stints.

    A stint ends the moment the compound label changes between two
    observed laps. Rows with no usable lap time are skipped rather than
    ending the stint, so a missing measurement does not split one run into
    two artificial halves; a row with no compound inherits the previous
    one, so a briefly dropped field does not paint a blank segment.
    """
    stints: list[_Stint] = []
    current: _Stint | None = None
    last_compound: str | None = None
    for row in rows:
        seconds = _sane_lap_time(row.get("lap_time_s"))
        if seconds is None:
            continue
        try:
            lap = float(row.get("lap", 0))
        except (TypeError, ValueError):
            continue
        compound = str(row.get("compound") or "").upper().strip() or last_compound or "MEDIUM"
        if current is None or compound != current.compound:
            current = _Stint(compound, hex_str(compound_color(compound)), [])
            stints.append(current)
        current.points.append([lap, seconds])
        last_compound = compound
    return stints


def _rolling_mean(values: list[float], window: int = 3) -> list[float]:
    """Centred rolling mean with `min_periods=1` edges.

    Three points on purpose: heavier smoothing visibly lags the trend over
    the 25-30 lap window this chart shows, which defeats the overlay. The
    edges average over whatever exists so the line starts at the first lap
    rather than at `window // 2`.
    """
    if not values:
        return []
    half = window // 2
    out: list[float] = []
    for index in range(len(values)):
        chunk = values[max(0, index - half) : min(len(values), index + half + 1)]
        out.append(sum(chunk) / len(chunk))
    return out


def build_tire_series(
    rows: list[dict[str, Any]], current_lap: int | None, tire_out: dict[str, Any] | None
) -> dict[str, Any]:
    """Per-stint segments, the smoothing overlay, and the cliff annotations.

    The compound change is a BREAK, not a colour change on one line: each
    stint is its own series, so nothing is drawn through the in-lap and
    the out-lap, which are neither the same length as each other nor as a
    racing lap.
    """
    stints = _build_stints(rows)
    flat = [point for stint in stints for point in stint.points]
    smoothed = _rolling_mean([point[1] for point in flat])
    trend = [[point[0], value] for point, value in zip(flat, smoothed)]

    cliff = _build_cliff(current_lap, tire_out)
    return {
        "stints": [
            {"compound": stint.compound, "colour": stint.colour, "points": stint.points}
            for stint in stints
        ],
        # One faint dashed vertical per compound change, kept faint because
        # the colour break already says it (`tire_chart.py:184-195`). The
        # first stint has no boundary in front of it.
        "boundaries": [stint.points[0][0] for stint in stints[1:] if stint.points],
        "boundary_colour": BOUNDARY_COLOUR,
        "trend": trend,
        "trend_colour": TREND_COLOUR,
        "cliff": cliff,
        "cliff_colour": CLIFF_COLOUR,
        "boundary_opacity": BOUNDARY_OPACITY,
        "x_range": _x_range(flat, current_lap, cliff),
        "y_range": _y_range(flat + trend),
        "current_lap": None if current_lap is None else float(current_lap),
        "cursor_colour": CURSOR_COLOUR,
    }


def _build_cliff(current_lap: int | None, tire_out: dict[str, Any] | None) -> dict[str, Any] | None:
    """The band at `[lap + p10, lap + p90]` and the marker at `lap + p50`.

    Keyed off the CURRENT lap, because the percentiles are laps REMAINING.
    Suppressed whole when the projection is outside the sane horizon: an
    unreadable band is worse than none, and the TCN emits those on the
    first laps of every stint.
    """
    if current_lap is None or not tire_out:
        return None
    p10 = _sane_cliff(tire_out.get("laps_to_cliff_p10"))
    p50 = _sane_cliff(tire_out.get("laps_to_cliff_p50"))
    p90 = _sane_cliff(tire_out.get("laps_to_cliff_p90"))
    if p10 is None and p50 is None and p90 is None:
        return None
    lap = float(current_lap)
    return {
        "lo": None if p10 is None else lap + p10,
        "hi": None if p90 is None else lap + p90,
        "p50": None if p50 is None else lap + p50,
    }


# How much air to leave above and below a bounded lap-time axis, in seconds.
# Wide enough that a flat stint does not read as a ruler, narrow enough that
# a real 0.3 s/lap degradation still has slope.
_Y_PAD_S = 2.5


def _y_range(points: list[list[float]]) -> list[float] | None:
    """Bound the lap-time axis to the laps actually plotted.

    **The twin of `_x_range`, which has existed all along.** That one
    clamps the lap axis so "a bad p90 cannot flatten the series to a
    hairline"; nothing did the same for the value axis, and the value
    axis is where the flattening was actually happening. Autoranged
    against a stint band that reaches 140 s, the tyre trace at 81.2 s
    occupied FOUR PIXELS of a 150 px plot, so the 0.031 s/lap degradation
    the panel exists to show was sub-pixel. Any real in-lap - about +20 s,
    and inside the 30-200 s sanity window - reproduces it.

    **Bound over everything DRAWN, which is the stint points and the trend
    together.** It used to read `trend or flat`, two mistakes in three
    words: the `or` is dead, since the trend is non-empty exactly when the
    points are, and the trend is the SMOOTHED series - a 22 s in-lap
    averages down inside it, so the raw point the stints really plot fell
    outside the bound and clipped off the top of its own chart.

    A genuine pit lap does widen the axis to about 25 s, and the trend then
    reads flat again. That is honest - the plotted data really does span
    that - and it is a different question from the one this fixes, which
    was an axis derived from a cliff band the chart never draws on it. Whether
    an in-lap belongs in a degradation trace at all is filed separately.

    Returns `None` when there is nothing to bound, which leaves the axis
    autoranged rather than inventing a window around no data.
    """
    values = [point[1] for point in points if point[1] is not None]
    if not values:
        return None
    low, high = min(values), max(values)
    pad = max(_Y_PAD_S, (high - low) * 0.15)
    return [round(low - pad, 2), round(high + pad, 2)]


def _x_range(
    points: list[list[float]], current_lap: int | None, cliff: dict[str, Any] | None
) -> list[float]:
    """Clamp the lap axis so a bad p90 cannot flatten the series to a hairline.

    ⚠ **The one deliberate deviation from the Qt chart in this sprint.**
    `tire_chart.py::_anchor_x_range` extends the axis by the whole sane
    cliff horizon whenever the band is visible - `x_max += 100` - so on
    lap 23 with a cliff six laps out the axis runs to 123 and the stint
    occupies about eight per cent of the width. That is not a bad-value
    guard doing its job, it fires on every normal cliff. Here the axis
    stops three laps past whichever is further, the last observed lap or
    the band's own upper edge, which bounds it for the same reason
    without the unreadable result. Flagged rather than smuggled: it is a
    visible difference and the exit gate should see it named.
    """
    laps = [point[0] for point in points]
    if current_lap is not None:
        laps.append(float(current_lap))
    if not laps:
        return [0.0, 1.0]
    high = max(laps)
    if cliff and cliff.get("hi") is not None:
        high = max(high, cliff["hi"])
    return [min(laps) - 0.5, high + 3.0]
