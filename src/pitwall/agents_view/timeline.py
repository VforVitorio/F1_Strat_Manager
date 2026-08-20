"""The decision band's PLAN module: the race on one lap axis.

Three lanes over lap 1 to the flag - the tyre cliff, the stints run and the
stint planned, and the marks for now and for the scheduled stop. It is the one
place on this window that shows the whole race rather than a rolling window,
which is why it reads `LapHistory.compound_runs` and not the 40-lap chart
stores.

**Positions leave here as FRACTIONS of the race, not as pixels.** The renderer
places spans by per cent, so the arithmetic that decides where lap 24 sits is
in the same language as everything else the view computes, and a test can read
it without a browser.

Nothing here invents a compound, a lap or a length. A stint whose compound the
producer never reported renders in a neutral colour and says so; a race with no
`total_laps` renders an empty track rather than dividing by zero.
"""

from __future__ import annotations

from typing import Any

from src.arcade.palette import TEXT_TERTIARY, compound_color, hex_str


def _fraction(lap: float, total_laps: int) -> float:
    """Where a lap sits along the track, in [0, 1].

    Lap 1 is the start of the track and `total_laps` its end, so the divisor is
    `total_laps - 1` rather than `total_laps`: with 57 laps, lap 57 has to land
    at 1.0 and not at 0.982.
    """
    span = max(1, total_laps - 1)
    return min(1.0, max(0.0, (lap - 1) / span))


def _segment(lo: int, hi: int, compound: str | None, total_laps: int, planned: bool) -> dict:
    """One bar on the stint lane, as a left/width pair in per cent.

    `hi` is the last lap OF the stint, so the bar has to run to the end of that
    lap rather than to its start - otherwise a one-lap stint has zero width and
    every stint reads one lap short.
    """
    start = _fraction(lo, total_laps)
    end = _fraction(hi + 1, total_laps)
    known = bool(compound)
    return {
        "lo": lo,
        "hi": hi,
        "compound": compound if known else None,
        # A stint nobody reported a compound for is TEXT_TERTIARY, the same
        # colour this window uses everywhere for "not reported" - never the
        # neutral-looking MEDIUM yellow a fallback would paint.
        "colour": hex_str(compound_color(compound)) if known else hex_str(TEXT_TERTIARY),
        "planned": planned,
        "left_pct": round(start * 100, 2),
        "width_pct": round(max(0.0, end - start) * 100, 2),
    }


def build_plan_timeline(
    runs: list[dict[str, Any]],
    latest: dict[str, Any] | None,
    arcade: dict[str, Any],
    current_lap: int | None,
    cliff: dict[str, Any] | None,
    cliff_colour: str,
    caption: str,
) -> dict[str, Any]:
    """The whole PLAN module, ready to place.

    Args:
        runs: `LapHistory.compound_runs()` - the stints actually observed.
        latest: this lap's decision, for `pit_lap_target` and `compound_next`.
        arcade: the tick's arcade block, for `total_laps`.
        current_lap: where the car is now, marked on the axis.
        cliff: the tyre chart's own cliff band, so one fact is drawn at two
            zoom levels rather than computed twice.
        cliff_colour: and its colour, from the same place, so the two lanes
            cannot drift onto two different ambers.
        caption: the orchestrator's plan line, verbatim, including its
            empty-state branches.
    """
    try:
        total_laps = int(arcade.get("total_laps") or 0)
    except (TypeError, ValueError):
        total_laps = 0

    # Before the first tick the arcade has not said how long the race is. An
    # empty track is the honest rendering; a track drawn against a guessed
    # length would place every mark somewhere wrong.
    if total_laps < 2:
        return {
            "total_laps": 0,
            "first_known_lap": None,
            "segments": [],
            "pit_lap": None,
            "pit_pct": None,
            "cliff": None,
            "current_lap": None,
            "current_pct": None,
            "caption": caption,
        }

    segments = [
        _segment(run["lo"], run["hi"], run.get("compound"), total_laps, planned=False)
        for run in runs
    ]

    # The planned stint: from the scheduled stop to the flag, hollow. Only when
    # the producer named BOTH a lap and a compound - a stop with no compound is
    # a plan the window cannot draw, and inventing one is how a hollow bar
    # becomes a claim.
    pit_lap = latest.get("pit_lap_target") if latest else None
    compound_next = latest.get("compound_next") if latest else None
    if isinstance(pit_lap, int) and compound_next:
        segments.append(_segment(pit_lap, total_laps, str(compound_next), total_laps, planned=True))

    band = None
    if cliff and cliff.get("lo") is not None and cliff.get("hi") is not None:
        lo, hi = float(cliff["lo"]), float(cliff["hi"])
        band = {
            "lo": lo,
            "hi": hi,
            "colour": cliff_colour,
            "left_pct": round(_fraction(lo, total_laps) * 100, 2),
            "width_pct": round(
                max(0.0, _fraction(hi, total_laps) - _fraction(lo, total_laps)) * 100, 2
            ),
        }

    return {
        "total_laps": total_laps,
        # Laps before this one are blank TRACK, not a grey fill: a window opened
        # mid-race never saw them, and drawing something there would be a claim
        # about a stint nobody reported.
        "first_known_lap": segments[0]["lo"] if runs else None,
        "segments": segments,
        "pit_lap": pit_lap if isinstance(pit_lap, int) else None,
        "pit_pct": round(_fraction(pit_lap, total_laps) * 100, 2)
        if isinstance(pit_lap, int)
        else None,
        "cliff": band,
        "current_lap": current_lap if isinstance(current_lap, int) else None,
        "current_pct": round(_fraction(current_lap, total_laps) * 100, 2)
        if isinstance(current_lap, int)
        else None,
        "caption": caption,
    }
