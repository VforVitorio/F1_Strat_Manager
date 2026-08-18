"""Decoding FastF1's ``TrackStatus`` digits, in the one place that owns the rule.

**Moved out of ``overlays.py`` so a consumer can read it without importing the
arcade GUI library.** The functions themselves are unchanged and ``overlays``
re-exports them, so every existing caller is untouched; what changed is that
``src/pitwall/session_data.py`` can now decode the status of each LAP while
building the bulk payload. Before this, the only decoded form on the wire was
the arcade-level one for the lap on screen, and the race-pace grid had no way to
know a lap was neutralised without a second copy of the priority order and the
four labels in TypeScript - which is the defect ``driver_colors`` rides on the
wire to prevent.

--- WHERE TO CHANGE IF THE STATUS CODES CHANGE ---
``src/arcade/app.py`` publishes ``track_status_label`` for the lap on screen and
``src/pitwall/session_data.py`` publishes ``neutralised`` per lap row. Both read
this module; neither has its own table.
"""

from __future__ import annotations

from src.arcade.config import SUCCESS

__all__ = ["neutralised_label", "track_status_banner", "track_status_label"]

# The statuses under which the field is NOT racing freely, so a per-lap pace
# ranking over them ranks the queue rather than the pace. A single yellow is
# deliberately absent: it is sector-local, and cars away from it are racing.
_NOT_RACING = frozenset({"RED FLAG", "SAFETY CAR", "VSC"})


def track_status_banner(code: str) -> tuple[str, tuple[int, int, int]] | None:
    """Map a FastF1 multi-digit ``TrackStatus`` to (label, RGB), or None if clear.

    Priority red > SC > VSC > yellow > clear, matching how race control
    announces concurrent events: a red flag wins even if a yellow was
    already out in another sector.

    Module level rather than a method because several surfaces read it. It used
    to be ``RaceEventsPanel._status_for``, and a second consumer arriving in
    TypeScript would have forked the priority order and the four labels
    across two languages - the defect ``driver_colors`` rides on the wire to
    prevent.
    """
    if not code:
        return None
    digits = set(code)
    if "5" in digits:
        return ("RED FLAG", (239, 68, 68))
    if "4" in digits:
        return ("SAFETY CAR", (255, 140, 0))
    if "6" in digits or "7" in digits:
        return ("VSC", (245, 158, 11))
    if "2" in digits:
        return ("YELLOW FLAG", (250, 204, 21))
    return None


def track_status_label(code: str) -> tuple[str, tuple[int, int, int]] | None:
    """The banner, but with the clear case named instead of hidden.

    The arcade's pill HIDES itself on a clear track, so it never had to tell
    "clear" apart from "the loader has no entry for this lap" - both are the
    absence of a pill. A timing strip always shows a track status, so the
    two have to separate: ``""`` is unknown and returns None, ``"1"`` is
    green and says so. Conflating them would put a confident GREEN on a lap
    whose status nobody knows, which is the sentinel class this repo keeps
    paying for.
    """
    if not code:
        return None
    return track_status_banner(code) or ("GREEN", SUCCESS)


def neutralised_label(code: str | None) -> str | None:
    """The label when the field was NOT racing freely on this lap, else None.

    This is what the race-pace grid needs and what no client should derive: it
    ranks every timed lap into thirds and paints the thirds green / grey / amber,
    and under a safety car those thirds are the accordion's queue order, not
    pace. Measured on Melbourne 2025, 22 of 57 laps carry a safety-car digit and
    **213 of the 776 cells the grid ranks (27.4 %)** sit on one, with lap times
    running 86.4-148.2 s (median 131.6) against a green median of 91.9.

    A single yellow returns None on purpose. It is sector-local: the cars that
    are not in that sector are racing, and marking the whole lap would spend the
    marker on 43 more rows of this one race without a claim behind it.

    ``None`` is also the answer for an unknown status (``""`` or absent), because
    "nobody recorded what the track was doing" is not "the track was clear" -
    and it must not paint a marker either way.
    """
    decoded = track_status_banner(code or "")
    if decoded is None:
        return None
    return decoded[0] if decoded[0] in _NOT_RACING else None
