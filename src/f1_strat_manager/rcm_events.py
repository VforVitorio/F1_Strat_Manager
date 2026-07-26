"""The Race Control Message classifier, in one place so it cannot be measured twice.

Pure rule-based logic over strings: no model, no torch, no pandas. It was inlined in
``src/agents/radio_agent.py`` and separately PORTED into ``src/strategy/eval/nlp.py``,
whose docstring asserted the two were identical. They were not. Over the real
1515-row 2025 RCM corpus they disagreed on **427 messages, 28.2 percent**, and the
eval copy was the older side: #305's fix for ``SAFETY CAR IN THIS LAP`` never crossed
into it. The published coverage in ``documents/eval_reports/nlp.md`` therefore
measured a private copy rather than the parser that ships, over-stating it by 2.24
points on a row marked ``reproduced`` (#632).

The port existed because importing ``radio_agent`` loads every NLP model, which is a
real cost the harness was right to avoid. The answer was to move the logic somewhere
importable without that stack, not to copy it.

--- WHERE TO CHANGE IF THE FIA MESSAGE FORMAT CHANGES ---
Here, and only here. ``notebooks/nlp/N23_rcm_parser.ipynb`` is the historical origin
and is read-only; if it and this file disagree, THIS file is what runs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

_FLAG_MAP = {
    "RED": "RED_FLAG",
    "GREEN": "GREEN_FLAG",
    "CLEAR": "CLEAR_FLAG",
    "BLUE": "BLUE_FLAG",
    "CHEQUERED": "CHEQUERED_FLAG",
}

# NR-03 (#398) — flag values that resolve through the same scope-aware branch as
# a plain YELLOW. DOUBLE YELLOW is the highest-danger *local* flag (an incident
# sits directly on the racing line; cars must slow far more than for a single
# yellow) and is a frequent precursor to a full Safety Car. Before this fix it
# matched neither `_FLAG_MAP` nor the old exact `flag == "YELLOW"` check, so it
# fell all the way through to OTHER — invisible to `_SAFETY_FLAGS` and to N31.
# Folding it into the YELLOW branch below means it inherits YELLOW_FLAG /
# YELLOW_FLAG_SECTOR's existing alert status for free, with no edit needed to
# `_SAFETY_FLAGS` itself and no change to how a plain YELLOW is handled.
_YELLOW_FLAG_VALUES = {"YELLOW", "DOUBLE YELLOW"}

# NR-03 (#398) — message-keyword families that fell through to OTHER. Verified
# against the real 2025 rcm.parquet corpus (data/processed/race_radios/2025/**)
# except _SESSION_* (see _classify_rcm_event docstring for the caveat: no race
# in that corpus contains a red-flag suspension, so this family is unverified).
_TRACK_LIMITS_KEYWORDS = ("TRACK LIMITS", "TIME DELETED", "LAP DELETED", "DELETED")
_INVESTIGATION_KEYWORDS = ("UNDER INVESTIGATION", "NOTED")
_SESSION_SUSPENDED_KEYWORDS = ("SESSION SUSPENDED", "RACE SUSPENDED")
_SESSION_RESUMED_KEYWORDS = (
    "SESSION RESUMED",
    "RACE RESUMED",
    "RACE WILL RESUME",
    "SESSION WILL RESUME",
)
_SESSION_STARTED_KEYWORDS = ("SESSION WILL START", "RACE WILL START")


@dataclass
class RCMEvent:
    """A single Race Control Message row prepared for the RCM parser.

    message:
        Raw message string from FastF1 session.race_control_messages.
    flag:
        Flag type string (e.g. 'YELLOW', 'GREEN', 'SAFETY CAR'). Empty
        string when the RCM is informational and carries no flag.
    category:
        RCM category from FastF1 (e.g. 'SafetyCar', 'Flag', 'Other').
    lap:
        Race lap number at which the RCM was issued.
    racing_number:
        Car number referenced by the message, if any. None when no specific
        car is referenced (e.g. track-wide SC deployment).
    scope:
        Spatial scope of the message ('Track', 'Sector', 'Driver').
    """

    message: str
    flag: str
    category: str
    lap: int
    racing_number: Optional[str] = None
    scope: str = ""


def classify_rcm_event(event: "RCMEvent") -> str:
    """Map a raw RCMEvent to a canonical event type string.

    Priority: SafetyCar category -> flag keyword -> incident keyword ->
    NR-03 (#398) coverage additions -> OTHER. Mirrors the N23 rule-based
    classifier used in N24's run_rcm_pipeline.

    NR-03 (#398) note on placement: every new branch below sits AFTER the
    pre-existing DRS/collision/retired/penalty checks so no message that
    already resolves to a specific event type changes classification — the
    additions only catch what used to fall through to OTHER. One consequence,
    verified against the real 2025 rcm.parquet corpus: nearly every real
    "UNDER INVESTIGATION" / "NOTED" message also contains the word "INCIDENT"
    (e.g. "TURN 1 INCIDENT INVOLVING CAR 7 (DOO) NOTED - MOVING UNDER
    BRAKING"), so it is already caught by the COLLISION/CONTACT/INCIDENT
    branch above and never reaches the new INVESTIGATION branch. That overlap
    is pre-existing behaviour (unrelated to this fix, and out of scope to
    narrow here since doing so would reclassify messages that already resolve
    today) — the INVESTIGATION branch is genuine coverage only for messages
    that use "NOTED"/"UNDER INVESTIGATION" wording without those keywords.
    """
    cat = event.category.strip()
    flag = event.flag.strip().upper()
    msg = event.message.upper()
    # NR-03 (#398): scope casing is not guaranteed by the data source (FastF1
    # vs OpenF1-sourced corpora disagree on "Sector" vs "SECTOR"), so normalise
    # before comparing rather than relying on the exact-case match used before.
    scope = event.scope.strip().upper()

    if cat == "SafetyCar":
        # "SAFETY CAR IN THIS LAP" is the FIA end-of-neutralisation message (the
        # car comes into the pit lane this lap). It carries none of the other
        # keywords, so it used to fall through to SAFETY_CAR_DEPLOYED — which
        # kept the stateful tracker pinned ON for the rest of the race. Treat it
        # as ending. (NR-02 / NR-03, #305)
        ending = "ENDING" in msg or "IN THIS LAP" in msg
        if "VIRTUAL" in msg:
            return "VIRTUAL_SAFETY_CAR_ENDING" if ending else "VIRTUAL_SAFETY_CAR_DEPLOYED"
        if "IN THE PIT LANE" in msg:
            return "SAFETY_CAR_IN_PIT_LANE"
        if ending:
            return "SAFETY_CAR_ENDING"
        return "SAFETY_CAR_DEPLOYED"

    if flag in _FLAG_MAP:
        return _FLAG_MAP[flag]
    if flag in _YELLOW_FLAG_VALUES:
        return "YELLOW_FLAG_SECTOR" if scope == "SECTOR" else "YELLOW_FLAG"

    if "DRS ENABLED" in msg:
        return "DRS_ENABLED"
    if "DRS DISABLED" in msg:
        return "DRS_DISABLED"
    if any(k in msg for k in ("COLLISION", "CONTACT", "INCIDENT")):
        return "CAR_COLLISION"
    if "RETIRED" in msg:
        return "CAR_RETIRED"
    if "PENALTY" in msg:
        return "TIME_PENALTY"

    # ── NR-03 (#398) additive coverage ──────────────────────────────────────
    # Track-limit / lap-time deletions ("CAR 1 (VER) LAP DELETED - TRACK
    # LIMITS AT TURN 14 ..."): a penalty-risk signal for us or a rival that
    # was previously indistinguishable from any other unclassified message.
    if any(k in msg for k in _TRACK_LIMITS_KEYWORDS):
        return "LAP_DELETED"
    # Stewards reviewing an incident ("... UNDER INVESTIGATION" / "... NOTED"):
    # penalty anticipation, worth surfacing even though (see docstring) the
    # broader COLLISION/CONTACT/INCIDENT check above already intercepts the
    # large majority of real-world instances of this phrasing.
    if any(k in msg for k in _INVESTIGATION_KEYWORDS):
        return "INVESTIGATION"
    # Pit lane entry/exit status ("PIT LANE ENTRY CLOSED" / "... OPEN"): a hard
    # veto or permission on PIT_NOW, so CLOSED and OPEN are kept as distinct
    # event types rather than a single generic "pit lane event" bucket.
    if "PIT" in msg and "CLOSED" in msg:
        return "PIT_LANE_CLOSED"
    if "PIT" in msg and "OPEN" in msg:
        return "PIT_LANE_OPEN"
    # Session suspend/resume/start announcements. Unlike the families above,
    # no race in the local 2025 rcm.parquet corpus contains a red-flag
    # suspension, so these keyword sets are best-effort FIA phrasing and are
    # NOT corpus-verified — flagged in the PR/issue for validation against a
    # real red-flag race before being relied on for the restart-procedure logic
    # the audit describes (AUDIT_NLP_RADIO_PIPELINE.md section 5.2 item 4).
    if any(k in msg for k in _SESSION_SUSPENDED_KEYWORDS):
        return "SESSION_SUSPENDED"
    if any(k in msg for k in _SESSION_RESUMED_KEYWORDS):
        return "SESSION_RESUMED"
    if any(k in msg for k in _SESSION_STARTED_KEYWORDS):
        return "SESSION_STARTED"

    return "OTHER"
