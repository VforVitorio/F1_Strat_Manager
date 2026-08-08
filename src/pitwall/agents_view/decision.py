"""The left column's top half: the N31 decision and the scenario scores.

Transcribed from `orchestrator_card.py` and `scenario_bars.py`, including
their colour maps and their three-branch plan line, and calling
`classify_action` rather than repeating its table - a hand-copied twin of
that table lived in `theme.py` until 2026-08-01 and drifted, which is the
mechanism #620 fixed once already.

`ScenarioBars.update_from` is the one thing here that is an algorithm
rather than a lookup: the four Monte Carlo scores are position-equivalent
gains against the STAY_OUT baseline, so they are signed and frequently
all negative. Shifting by the minimum before scaling is what keeps the
bar widths valid, and getting the shift, the scale or the tie-break wrong
changes which scenario reads as the winner.
"""

from __future__ import annotations

from typing import Any

from src.arcade.palette import (
    ACCENT,
    DANGER,
    SUCCESS,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
    TEXT_TERTIARY,
    WARNING,
    compound_pill_html,
    hex_str,
)
from src.arcade.strategy import classify_action

_PACE_COLOURS: dict[str, tuple[int, int, int]] = {
    "PUSH": DANGER,
    "NEUTRAL": TEXT_SECONDARY,
    "MANAGE": WARNING,
    "LIFT_AND_COAST": WARNING,
}

_RISK_COLOURS: dict[str, tuple[int, int, int]] = {
    "AGGRESSIVE": DANGER,
    "BALANCED": TEXT_SECONDARY,
    "NEUTRAL": TEXT_SECONDARY,
    "CONSERVATIVE": WARNING,
    "DEFENSIVE": WARNING,
}

SCENARIO_KEYS: tuple[str, ...] = ("STAY_OUT", "PIT_NOW", "UNDERCUT", "OVERCUT")
SCENARIO_LABELS: dict[str, str] = {
    "STAY_OUT": "STAY",
    "PIT_NOW": "PIT",
    "UNDERCUT": "UCUT",
    "OVERCUT": "OCUT",
}


def _confidence_colour(confidence: float) -> tuple[int, int, int]:
    """The three-tier traffic light `orchestrator_card.py` paints."""
    if confidence >= 0.66:
        return SUCCESS
    if confidence >= 0.33:
        return WARNING
    return DANGER


def _plan_line(latest: dict[str, Any], action: str) -> str:
    """`Pit: L24 · Next: <pill> · UCUT: RUS`, or the two empty-state branches.

    On STAY_OUT with no tactical plan the orchestrator leaves all three
    fields blank deliberately, and three "--" chips read as noise rather
    than as "nothing scheduled". That branch is copy, not a formatting
    accident.
    """
    pit_target = latest.get("pit_lap_target")
    compound_next = latest.get("compound_next")
    undercut_target = latest.get("undercut_target")
    if not any((pit_target, compound_next, undercut_target)):
        if action.upper() == "STAY_OUT":
            return "stint continues · no pit window yet"
        return "Pit plan pending"
    bits = [
        f"Pit: L{pit_target}" if pit_target else "Pit: —",
        f"Next: {compound_pill_html(compound_next)}" if compound_next else "Next: —",
        f"UCUT: {undercut_target}" if undercut_target else "UCUT: —",
    ]
    return " · ".join(bits)


def build_orchestrator(latest: dict[str, Any] | None) -> dict[str, Any]:
    """The action badge, the confidence bar, the two chips and the plan."""
    if not latest:
        return {
            "action": "--",
            "action_colour": hex_str(TEXT_TERTIARY),
            "confidence": None,
            "confidence_label": "Confidence: --",
            "confidence_colour": hex_str(TEXT_TERTIARY),
            "pace": "Pace: --",
            "pace_colour": hex_str(TEXT_TERTIARY),
            "risk": "Risk: --",
            "risk_colour": hex_str(TEXT_TERTIARY),
            "plan": "Pit: -- · Next: -- · UCUT: --",
            "guardrail": "",
        }

    action = str(latest.get("action") or "--")
    confidence = float(latest.get("confidence") or 0.0)
    pace_mode = latest.get("pace_mode")
    risk_posture = latest.get("risk_posture")
    guardrail = latest.get("guardrail_reason")
    badge_colour, badge_label = classify_action(action)

    return {
        "action": badge_label,
        "action_colour": hex_str(badge_colour),
        "confidence": confidence,
        "confidence_label": f"Confidence: {confidence * 100:.0f}%",
        "confidence_colour": hex_str(_confidence_colour(confidence)),
        "pace": f"Pace: {pace_mode or '--'}",
        "pace_colour": hex_str(_PACE_COLOURS.get(str(pace_mode or "").upper(), TEXT_TERTIARY)),
        "risk": f"Risk: {risk_posture or '--'}",
        "risk_colour": hex_str(_RISK_COLOURS.get(str(risk_posture or "").upper(), TEXT_TERTIARY)),
        "plan": _plan_line(latest, action),
        "guardrail": f"⚠ Guardrail: {guardrail}" if guardrail else "",
    }


def build_scenarios(scores: dict[str, Any] | None) -> list[dict[str, Any]]:
    """Four rows: label, bar fill in [0, 1], signed score, winner flag.

    The fill is min-max normalised across whichever of the four keys are
    present, so the winner always reaches full width and the worst draws
    an empty bar whatever the signs. An absent key draws nothing and
    prints `--`, which is not the same as a score of zero.
    """
    raw: dict[str, float] = {}
    for key, value in (scores or {}).items():
        try:
            raw[str(key).upper()] = float(value)
        except (TypeError, ValueError):
            continue

    winner = max(raw, key=raw.get) if raw else None
    if raw:
        lo, hi = min(raw.values()), max(raw.values())
        span = (hi - lo) or 1.0
    else:
        lo, span = 0.0, 1.0

    rows: list[dict[str, Any]] = []
    for key in SCENARIO_KEYS:
        present = key in raw
        value = raw.get(key, lo)
        fill = min(1.0, max(0.0, (value - lo) / span)) if present else 0.0
        is_winner = present and key == winner
        rows.append(
            {
                "key": key,
                "label": SCENARIO_LABELS[key],
                "fill": fill,
                "score": f"{value:+.2f}" if present else "  --",
                "is_winner": is_winner,
                "bar_colour": hex_str(ACCENT if is_winner else TEXT_SECONDARY),
                "label_colour": hex_str(ACCENT if is_winner else TEXT_SECONDARY),
                "score_colour": hex_str(TEXT_PRIMARY if is_winner else TEXT_SECONDARY),
            }
        )
    return rows
