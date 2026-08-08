"""The six reasoning tabs, with the Qt syntax highlighter ported to segments.

Two things live here.

**The tabs.** Composed exactly as `reasoning_tabs.py::update_from` does:
the orchestrator tab is the decision's own reasoning, plus the
DecisionMemory block **only on a lap where the call changed**; the five
agent tabs are the agent's reasoning followed by the `key = value` dump
from `reasoning_lines.py`. That conditional is copy, not an optimisation:
the block never appears in `reasoning` even when it drives the call, and
showing it unconditionally was measured as wallpaper.

**The highlighter.** `_ReasoningHighlighter` is a `QSyntaxHighlighter`
with five compiled rules, and it is the one part of this window that is
not a dict-in / string-out transform. Gate B named two ways to port it:
pre-process into HTML with `<span>` wraps, or a decoration layer in the
client.

This is the first, with the transport changed: the host emits **typed
segments** rather than an HTML string. Same place, same code, but the
free text of an LLM never becomes markup, and the client renders spans
instead of trusting `dangerouslySetInnerHTML` with a model's output. The
badge builders in `palette.py` are HTML because Qt needed them to be;
this had no such constraint, so it does not inherit one.

Qt's semantics are reproduced exactly: rules are applied in order and a
later rule OVERWRITES an earlier one where they overlap, which is what
`setFormat` does on a `QTextDocument`. A per-character colour array is
the honest way to say that; runs are emitted afterwards.
"""

from __future__ import annotations

import re
from typing import Any

from src.arcade.dashboard.reasoning_lines import LINE_BUILDERS, clean, compose
from src.arcade.palette import TEXT_PRIMARY, hex_str

# reasoning_tabs.py's five colours, in its order. The order matters: the
# action keywords are last, so `PIT_NOW` inside a percentage-bearing
# sentence still reads as an action.
_LAP = "#f472b6"
_QUANT = "#d946ef"
_PCT = "#facc15"
_DELTA = "#22d3ee"
_ACTION = "#facc15"

_RULES: tuple[tuple[re.Pattern[str], str, bool], ...] = (
    (re.compile(r"\blaps?\s+\d+(?:[-–]\d+)?\b"), _LAP, False),
    (re.compile(r"\b[Pp]\d{2}\b"), _QUANT, False),
    (re.compile(r"\b\d+(?:\.\d+)?%"), _PCT, False),
    (re.compile(r"[+\-]\d+\.\d+\s*s\b"), _DELTA, False),
    (re.compile(r"\b(PIT_NOW|STAY_OUT|UNDERCUT|OVERCUT)\b"), _ACTION, True),
)

DEFAULT_COLOUR = hex_str(TEXT_PRIMARY)

TABS: tuple[tuple[str, str], ...] = (
    ("orchestrator", "Orchestrator"),
    ("pace", "Pace"),
    ("tire", "Tire"),
    ("situation", "Situation"),
    ("radio", "Radio"),
    ("pit", "Pit"),
)


def highlight(text: str) -> list[dict[str, Any]]:
    """Split `text` into coloured runs, the way the Qt highlighter paints it.

    `QSyntaxHighlighter.highlightBlock` runs per block, but every rule
    here is anchored inside a line, so applying them over the whole
    string is equivalent and keeps the newlines in one segment.
    """
    if not text:
        return []
    colours: list[str | None] = [None] * len(text)
    bolds = [False] * len(text)
    for pattern, colour, bold in _RULES:
        for match in pattern.finditer(text):
            for index in range(match.start(), match.end()):
                colours[index] = colour
                bolds[index] = bold

    segments: list[dict[str, Any]] = []
    start = 0
    for index in range(1, len(text) + 1):
        same = (
            index < len(text) and colours[index] == colours[start] and bolds[index] == bolds[start]
        )
        if same:
            continue
        segments.append(
            {
                "text": text[start:index],
                "colour": colours[start] or DEFAULT_COLOUR,
                "bold": bolds[start],
            }
        )
        start = index
    return segments


def _orchestrator_body(latest: dict[str, Any]) -> str:
    """The decision's reasoning, plus the memory block on a changed lap only.

    DecisionMemory leaves no trace in `reasoning` even when it drives the
    call, so the block is rendered here rather than trusting the model to
    narrate its own continuity. Hidden on every other lap: the action
    changes on a small minority of them, and unconditional display was
    measured as wallpaper.
    """
    text = clean(latest.get("reasoning")) or "— no reasoning —"
    memory_block = latest.get("memory_block")
    if latest.get("plan_changed") and memory_block:
        text += "\n\n--- why this call changed ---\n" + str(memory_block)
    return text


def build_reasoning(latest: dict[str, Any] | None) -> list[dict[str, Any]]:
    """The six tabs, each already split into coloured segments."""
    if not latest:
        return [{"key": key, "label": label, "segments": []} for key, label in TABS]

    per = latest.get("per_agent") or {}
    tabs: list[dict[str, Any]] = []
    for key, label in TABS:
        if key == "orchestrator":
            body = _orchestrator_body(latest)
        else:
            agent_out = per.get(key) or {}
            metrics = LINE_BUILDERS[key](agent_out) if agent_out else []
            body = compose(clean(agent_out.get("reasoning")), metrics)
        tabs.append({"key": key, "label": label, "segments": highlight(body)})
    return tabs
