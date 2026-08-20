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
    readable_on,
)
from src.arcade.strategy import classify_action

# The two posture chips are FACTS, not warnings, so they wear text colour and
# the word carries the meaning (#964).
#
# They used to be a severity scale, and it put `Risk: AGGRESSIVE` - a setting
# somebody chose - in the same DANGER red as `⚠ Guardrail: minimum stint
# length not met`, a constraint violation, one line below it. Counting the
# window's reds found six different meanings on one colour: imperative action,
# posture, low confidence, radio alert, dead link, rule violation. The whole
# point of an alarm colour is pre-attentive triage, and six semantics deny the
# reader exactly that at the moment they need it.
#
# DANGER now belongs to alarm-class facts: the ALERT glyph and a dead
# connection. The guardrail line was the third and it is gone, because nothing
# could fill it (#974, `build_orchestrator`). The dicts stay rather than
# collapsing into a constant, because a posture the producer has never sent
# must still fall to TEXT_TERTIARY - which is what says "not reported" as
# against "reported and unremarkable".
_PACE_COLOURS: dict[str, tuple[int, int, int]] = {
    "PUSH": TEXT_SECONDARY,
    "NEUTRAL": TEXT_SECONDARY,
    "MANAGE": TEXT_SECONDARY,
    "LIFT_AND_COAST": TEXT_SECONDARY,
}

_RISK_COLOURS: dict[str, tuple[int, int, int]] = {
    "AGGRESSIVE": TEXT_SECONDARY,
    "BALANCED": TEXT_SECONDARY,
    "NEUTRAL": TEXT_SECONDARY,
    "CONSERVATIVE": TEXT_SECONDARY,
    "DEFENSIVE": TEXT_SECONDARY,
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


def _previous_call(latest: dict[str, Any], tail: list[dict[str, Any]] | None) -> str:
    """`was STAY OUT (0.58) · L22`, when this lap's call is not last lap's.

    A pit wall reads deltas, and this window had no first-class answer to
    "what changed since the last lap" (#968): the badge, the confidence,
    the four bars and all six cards overwrite in place ten times a second,
    and the only trace that anything moved was a monospace heading inside
    a tab panel, below the fold of attention.

    Read off `history_tail`, whose entries are real `LapDecision` fields.
    **Not parsed out of `memory_block`**, which the gate's proposal assumed
    was `lap 22: STAY_OUT (0.58)` and which is really a multi-line LLM
    prompt block ("DECISION MEMORY (your own previous calls this race):").

    `plan_changed` is the producer's own answer and it counts the ACTION
    only - measured over 40 lap pairs of a real race, the action moved on
    none of them while `pit_lap_target` moved on 25, so counting the target
    would open this on two laps in three and turn a signal into wallpaper.

    The tail's last entry is the CURRENT decision, because the producer
    appends to `history` and sets `latest` in the same breath
    (`src/arcade/strategy.py:439-440`), so the search is for the newest
    entry from an EARLIER lap rather than for `tail[-2]`.
    """
    if not latest.get("plan_changed") or not tail:
        return ""
    lap = latest.get("lap_number")
    earlier = [row for row in tail if isinstance(row.get("lap_number"), int)]
    if isinstance(lap, int):
        earlier = [row for row in earlier if row["lap_number"] < lap]
    if not earlier:
        return ""
    previous = max(earlier, key=lambda row: row["lap_number"])
    _, label = classify_action(str(previous.get("action") or "--"))
    try:
        confidence = f" ({float(previous.get('confidence') or 0.0):.2f})"
    except (TypeError, ValueError):
        confidence = ""
    return f"was {label}{confidence} · L{previous['lap_number']}"


def build_orchestrator(
    latest: dict[str, Any] | None, history_tail: list[dict[str, Any]] | None = None
) -> dict[str, Any]:
    """The action badge, the confidence bar, the two chips and the plan.

    **There is no guardrail line, and that is the fix rather than an omission
    (#974).** The window used to render `⚠ Guardrail: <reason>` from
    `latest["guardrail_reason"]`, a field typed, documented, styled and tested
    on a path that cannot deliver it. The chain, walked end to end:

    - `apply_guard_rails` produces a reason at exactly one production site,
      `src/strategy/inference/no_llm.py:302`.
    - `run_lap` hardcodes `guardrail_reason=None` for the `rich` profile
      (`src/strategy/inference/engine.py:303`), because rich mode puts the
      bounds in the LLM's prompt instead of applying them after the fact.
    - `src/arcade/strategy_pipeline.py:48` hardcodes `profile="rich"`, and
      `src/arcade/app.py` builds its request with a literal `no_llm=False`.

    So on every arcade path the value is None by construction, and the line
    was permanently blank.

    --- WHERE TO CHANGE IF THE ARCADE LEARNS TO RUN WITHOUT AN LLM ---
    Restore the field here, the `guardrail` key on `OrchestratorView`
    (`lib/agents.ts`), the render in `OrchestratorCard.tsx` and the
    `.orch-guardrail` rule, and give the raw hex back its row in
    `test_pitwall_tokens.py`.

    **What must NOT be done instead** is calling `apply_guard_rails` post-hoc
    on the rich path to fill the line. In rich mode the bounds live in the
    prompt and the model weighed them, so a deterministic check afterwards
    would report an override that never ran; the prompt and the deterministic
    mirror have also diverged (#716); and the bounds proscribe a real
    percentage of professional stops, so the line would wear the alarm colour
    on correct calls at that rate.
    """
    if not latest:
        return {
            "action": "--",
            "action_colour": hex_str(TEXT_TERTIARY),
            "action_text_colour": hex_str(readable_on(TEXT_TERTIARY)),
            "confidence": None,
            "confidence_fill": 0.0,
            "confidence_label": "Confidence: --",
            "confidence_colour": hex_str(TEXT_TERTIARY),
            "pace": "Pace: --",
            "pace_colour": hex_str(TEXT_TERTIARY),
            "risk": "Risk: --",
            "risk_colour": hex_str(TEXT_TERTIARY),
            "plan": "Pit: -- · Next: -- · UCUT: --",
            "changed": "",
        }

    action = str(latest.get("action") or "--")
    confidence = float(latest.get("confidence") or 0.0)
    pace_mode = latest.get("pace_mode")
    risk_posture = latest.get("risk_posture")
    badge_colour, badge_label = classify_action(action)

    return {
        "action": badge_label,
        "action_colour": hex_str(badge_colour),
        # The badge's own text colour, decided here rather than fixed to
        # white in the renderer. White on SUCCESS measures 2.54:1, so the
        # single most important element on the screen failed AA in the one
        # state - a guardrail veto - where a strategist most needs to read
        # it. `readable_on` picks whichever ground actually contrasts.
        "action_text_colour": hex_str(readable_on(badge_colour)),
        "confidence": confidence,
        # The bar's width, to the 0.1 % Qt's gradient stop resolves to
        # (`orchestrator_card.py::_bar_style` rounds the stop to 3 dp).
        # The client used to do this with `Math.round`, which is both
        # coarser and the exact kind of arithmetic the view exists to
        # keep out of the renderer.
        "confidence_fill": round(min(1.0, max(0.0, confidence)) * 100, 1),
        "confidence_label": f"Confidence: {confidence * 100:.0f}%",
        "confidence_colour": hex_str(_confidence_colour(confidence)),
        "pace": f"Pace: {pace_mode or '--'}",
        "pace_colour": hex_str(_PACE_COLOURS.get(str(pace_mode or "").upper(), TEXT_TERTIARY)),
        "risk": f"Risk: {risk_posture or '--'}",
        "risk_colour": hex_str(_RISK_COLOURS.get(str(risk_posture or "").upper(), TEXT_TERTIARY)),
        "plan": _plan_line(latest, action),
        "changed": _previous_call(latest, history_tail),
    }


# The narrowest bar a SCORED candidate may draw. It exists so that "was
# scored and came last" cannot render as "was never scored" (#963): min-max
# sends the worst candidate to zero by construction, and an absent one draws
# zero too, so the two were the same pixels - an executed diff over the two
# bar strips found nought differing pixels.
_SCORED_FLOOR = 0.06


def build_scenarios(
    scores: dict[str, Any] | None,
    enacted_action: str | None = None,
) -> list[dict[str, Any]]:
    """Four rows: label, bar fill in [0, 1], signed score, and who owns the call.

    The fill is min-max normalised across whichever of the four keys are
    present, floored at `_SCORED_FLOOR` so a scored candidate always draws
    ink. An absent key draws NO TRACK AT ALL and prints `--`; that is what
    keeps "nobody scored this" distinguishable from "this one came last",
    which the sentence used to claim while the pixels said otherwise.

    **The bar encodes RANK, not margin.** Min-max over signed gains is the
    only scale-free comparison available here, so `+0.70` against `+0.69`
    still draws the same widths as `+0.70` against `+0.10`; the number
    beside the bar is what carries the margin. Changing that is a decision
    about what the bar MEANS and needs its own argument.

    `enacted_action` is the action the orchestrator actually published, and
    it is not always the Monte Carlo winner - a guardrail can veto the top
    scenario (#962). When they diverge the panel used to crown the vetoed
    plan in full ACCENT while the badge one card up said the opposite, so
    the "why" panel read as the opposite of the call. The enacted row now
    takes the highlight; the vetoed winner KEEPS ITS FILL, because the
    Monte Carlo really did score it highest and that is true, and loses
    only the regalia, gaining a `VETOED` mark instead.
    """
    raw: dict[str, float] = {}
    for key, value in (scores or {}).items():
        try:
            raw[str(key).upper()] = float(value)
        except (TypeError, ValueError):
            continue

    # A TIE HAS NO WINNER. `max` picks whichever key it met first, which
    # invented a leader out of two equal scores and then marked the loser of
    # that coin toss `NOT TAKEN` - a claim about a decision nobody made.
    top = max(raw.values()) if raw else None
    leaders = [key for key, value in raw.items() if value == top]
    winner = leaders[0] if len(leaders) == 1 else None

    enacted = str(enacted_action or "").upper() or None
    if enacted is None:
        # Nothing published - the idle branch, and any producer that has not
        # sent an action. The simulation's preference stands in, which is
        # what every state but the overruled one already did.
        highlighted = winner
    elif enacted in raw:
        highlighted = enacted
    else:
        # **Published something these four rows do not describe.** `ALERT`
        # is the fifth member of the orchestrator's own action Literal and
        # it is not a scenario, so nothing here was enacted. Crowning the
        # Monte Carlo winner anyway is exactly the misread #962 fixed,
        # walking back in through the door left open for the idle case.
        highlighted = None
    unenacted = winner if (winner is not None and highlighted != winner) else None

    if raw:
        lo, hi = min(raw.values()), max(raw.values())
        span = hi - lo
    else:
        lo, span = 0.0, 0.0

    rows: list[dict[str, Any]] = []
    for key in SCENARIO_KEYS:
        present = key in raw
        value = raw.get(key, lo)
        # Every scored candidate FULL when they are all equal: min-max has
        # nothing to spread and flooring them all to 6 % said the opposite of
        # what a tie means.
        scaled = 1.0 if span == 0 else min(1.0, max(0.0, (value - lo) / span))
        fill = max(scaled, _SCORED_FLOOR) if present else 0.0
        is_highlighted = present and key == highlighted
        is_unenacted = key == unenacted
        if is_highlighted:
            accent = ACCENT
            score_colour = TEXT_PRIMARY
        elif is_unenacted:
            accent = TEXT_TERTIARY
            score_colour = TEXT_TERTIARY
        else:
            accent = TEXT_SECONDARY
            score_colour = TEXT_SECONDARY
        rows.append(
            {
                "key": key,
                "label": SCENARIO_LABELS[key],
                "fill": fill,
                # The bar's width in per cent, to the 0.1 % Qt's gradient
                # stop resolves to. Its twin one card up already came from
                # here; this one was still scaling 0-1 in the renderer,
                # unrounded, which is the arithmetic the view exists to
                # keep out of it.
                "fill_pct": round(fill * 100, 1),
                "score": f"{value:+.2f}" if present else "  --",
                # `is_winner` still means what it always meant - the top
                # Monte Carlo score - so a consumer asking "what did the
                # simulation prefer" still gets an honest answer. Whether
                # that plan was ENACTED is the separate flag beside it.
                "is_winner": present and key == winner,
                "is_enacted": is_highlighted,
                "is_scored": present,
                # `NOT TAKEN`, not `VETOED`. A veto names a MECHANISM, and
                # this code cannot see one: `guardrail_reason` never reaches
                # the window from any producer (#974), and the enacted action
                # can differ from the winner because the LLM synthesis chose
                # otherwise. What is certainly true is that the simulation
                # preferred this and it is not what the car is doing.
                "note": "NOT TAKEN" if is_unenacted else "",
                "bar_colour": hex_str(accent),
                "label_colour": hex_str(accent),
                "score_colour": hex_str(score_colour),
            }
        )
    return rows
