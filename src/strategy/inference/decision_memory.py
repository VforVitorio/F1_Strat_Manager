"""Per-race memory of the orchestrator's own previous recommendations.

Why this exists
---------------
The Layer 3 prompt is stateless. Consecutive laps are 99.02% identical text and
carry nothing that tells the model it is repeating itself, so it re-argues the
same case in fresh prose every lap. Measured over 41 laps at Lusail 2025 with no
memory, the orchestrator declared **80 distinct contingency triggers and reused
almost none**: across a race that is not one plan, it is 41 unrelated plans. With
its own previous contingencies echoed back it settles on six.

That is what this class is for, and it is a narrower claim than "memory makes the
system smarter" - but do not narrow it too far either, because the two halves blur
easily. On an ORDINARY green-flag lap the block does not change the call:
``action`` differed on 0 of 41 laps across a whole race. On the lap where a
contingency the model itself declared actually FIRES, it does, and that is the
entire point - under a Safety Car at Lusail 2025 lap 42 the orchestrator executed
its own one-lap-old plan on 8 of 8 runs against 0 of 8 without the block.

So: not a nudge applied to every lap, but a plan the model can still be holding
when the trigger arrives.

Where it lives, and why not in the engine
-----------------------------------------
``run_lap`` is pure per lap and ``tests/engine/test_engine_no_llm.py`` depends on
that: it calls the engine twice on one lap and asserts identical output. So the
accumulator lives in the CALLER (the CLI loop, the arcade connector, the backend
simulator's stream) and reaches the engine as a value. Each surface owns one
instance for the lifetime of one race, the same shape and lifetime as
``RaceControlStateTracker``.

``/recommend`` and the MCP tool cannot use this, and that is by design rather than
an oversight: both are stateless per request, with no race-scoped object to hang
an accumulator on, so the webapp Strategy tab keeps today's memoryless prompt.
Do not "fix" that by reimplementing a per-request accumulator on the endpoint
side; that is how ``_rcm_events_for_lap`` ended up existing twice with two
signatures.

--- WHERE TO CHANGE IF THE RECOMMENDATION SCHEMA CHANGES ---
``record`` reads ``action``, ``pit_lap_target`` and ``contingencies`` off a
``StrategyRecommendation``. Record from THAT object, never from a surface DTO:
``src/arcade/strategy.py``'s ``LapDecisionDTO`` and the backend's ``LapDecision``
both drop ``contingencies`` entirely, and the backend derives ``pit_lap_target``
from N28 on the no-llm branch and from the LLM on the rich one, so a DTO-sourced
memory would mean two different things on one surface.

Evidence for every field choice: documents/audits/AUDIT_ORCHESTRATOR_MEMORY.md.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# The synthesis schema caps contingencies at 4 (``_LLMSynthesis.contingencies``),
# and only the most recent lap's list is echoed, so the block cannot grow with race
# length. The cumulative reading was rejected: a trigger is free text with no
# evaluator, so nothing can retire one, and without memory the model produces about
# two brand-new triggers per lap.
MAX_CONTINGENCIES = 4

# How many past targets to show. Enough to make a drifting plan visible, short
# enough that the block stays a few lines.
TARGET_HISTORY = 5

# Appended to every block. Without it, memory measurably ANCHORED the model: at
# Lusail 2025 lap 44 (Norris's real stop) it agreed with the deterministic Monte
# Carlo on 4 of 10 runs against 6 of 10 with no memory at all. With this sentence,
# 10 of 10. It is not decoration and it is not optional.
COUNTERWEIGHT = (
    "  This history is context, NOT a commitment. Judge this lap on its own\n"
    "  evidence; a long hold is not itself a reason to keep holding.\n"
)

# Emitted only on a genuine repeated STAY_OUT. It used to live in the static
# prompt, where it fired on every lap: on lap 1, when there was no previous call
# to continue, and on the lap the call changed, when continuing was the wrong
# answer. Nothing could condition it there because the prompt had no idea what the
# last call was. Here the object that knows owns the claim, which is also why this
# is not a second parameter on the prompt builder travelling next to the block.
CONTINUATION = (
    "  This is a CONTINUING plan, not a fresh one: do not re-argue the same case\n"
    "  from scratch.\n"
)


@dataclass(frozen=True)
class _Entry:
    """One recorded lap: the parts of a recommendation the next lap should see."""

    lap: int
    action: str
    pit_lap_target: int | None
    contingencies: tuple[dict[str, str], ...]


@dataclass
class DecisionMemory:
    """Accumulates one race's recommendations and renders the prompt block.

    Responsibilities:
      * ``record`` one ``StrategyRecommendation`` per lap the surface actually
        decided on. Laps a surface skips (retired car, incomplete lap, a lap that
        raised) are simply not recorded, and the rendering makes the resulting gap
        visible rather than hiding it.
      * ``block`` renders what the LLM should see, or ``None`` before there is any
        history at all.

    Invariants:
      * Laps are recorded in strictly increasing order. Going backwards would make
        every span nonsense, so it raises rather than quietly accepting it.
      * The rendered block never states a bare "held for N laps" count. N counts
        DECISIONS; the laps that elapsed can be more. Every surface `continue`s
        past skipped laps, so a bare count is a claim the model cannot check.
    """

    _entries: list[_Entry] = field(default_factory=list)

    # ── recording ─────────────────────────────────────────────────────────
    def record(self, lap: int, recommendation: Any) -> None:
        """Store the parts of ``recommendation`` the next lap's prompt needs.

        Raises:
            ValueError: when ``lap`` is not after the last recorded lap. A replay
                that seeks backwards, or a caller recording the same lap twice,
                would otherwise produce a span that reads as a fact and is not one.
        """
        if self._entries and lap <= self._entries[-1].lap:
            raise ValueError(
                f"lap {lap} is not after the last recorded lap "
                f"{self._entries[-1].lap}; DecisionMemory is per race and forward-only"
            )
        self._entries.append(
            _Entry(
                lap=lap,
                action=str(getattr(recommendation, "action", "")),
                pit_lap_target=getattr(recommendation, "pit_lap_target", None),
                contingencies=tuple(
                    {
                        "trigger": str(c.trigger),
                        "switch_to": str(c.switch_to),
                        "priority": str(c.priority),
                    }
                    for c in (getattr(recommendation, "contingencies", None) or [])
                ),
            )
        )

    # ── what a surface needs to render the block ──────────────────────────
    def last_call_changed(self) -> bool:
        """True when the call just recorded differs from the one before it.

        Call it AFTER ``record``. Surfaces use it to decide whether to show the
        memory block already expanded: the block explains a decision, and the
        decision worth explaining is the one that moved.

        Only ``action`` counts, and that is a measured choice rather than a
        stylistic one. Over 40 lap pairs of a real race the action changed on
        0 of them while ``pit_lap_target`` moved on 25 (62%), so counting the
        target would open the panel on two laps in three and turn a signal into
        wallpaper. A target that drifts under an unchanged call is exactly what
        the block's drift line is for; it is not a change of plan.

        False with fewer than two entries: the first decision of a race is not
        a change, and there is nothing to compare it against.
        """
        if len(self._entries) < 2:
            return False
        return self._entries[-1].action != self._entries[-2].action

    # ── derived views, each one line of the block ─────────────────────────
    def _current_run(self) -> tuple[str, int, int]:
        """The unbroken run of the latest action: (action, first lap, decisions)."""
        action = self._entries[-1].action
        decisions = 0
        for entry in reversed(self._entries):
            if entry.action != action:
                break
            decisions += 1
        return action, self._entries[-decisions].lap, decisions

    def _recent_targets(self) -> list[int | None]:
        return [entry.pit_lap_target for entry in self._entries[-TARGET_HISTORY:]]

    def _live_contingencies(self) -> tuple[dict[str, str], ...]:
        """The most recent lap's contingencies, capped.

        Deliberately the last lap only. "Everything declared and not retired" is
        unimplementable: a trigger is prose, so no code can decide whether it fired.
        """
        return self._entries[-1].contingencies[:MAX_CONTINGENCIES]

    # ── rendering ─────────────────────────────────────────────────────────
    def _is_continuing_a_hold(self) -> bool:
        """True when the last call repeated a STAY_OUT, which is the only real hold.

        Two decisions minimum: one STAY_OUT is a call, not a continuation. Restricted
        to STAY_OUT because it is the only action a race can sit on. Repeating
        PIT_NOW is not a plan being continued, it is a stop that has not happened.
        """
        if len(self._entries) < 2:
            return False
        action, _since_lap, decisions = self._current_run()
        return action == "STAY_OUT" and decisions >= 2

    def _render_hold(self) -> str:
        action, since_lap, decisions = self._current_run()
        laps_spanned = self._entries[-1].lap - since_lap + 1
        if laps_spanned == decisions:
            unit = "lap" if decisions == 1 else "laps"
            return f"  Last call: {action}, held since lap {since_lap} ({decisions} {unit})."
        # The two numbers disagree, which means the surface skipped laps. Say both
        # rather than picking one: "held for N laps" would be false, and dropping
        # the span would hide that the race moved on without a decision.
        return (
            f"  Last call: {action}, held since lap {since_lap} "
            f"({decisions} decisions across {laps_spanned} laps; the rest were not "
            f"evaluated)."
        )

    def _render_targets(self) -> str:
        targets = self._recent_targets()
        shown = ", ".join("none" if t is None else str(t) for t in targets)
        known = [t for t in targets if t is not None]
        if len(known) < 2:
            return f"  Your pit_lap_target over the last {len(targets)} calls: {shown}."
        drift = known[-1] - known[0]
        return (
            f"  Your pit_lap_target over the last {len(targets)} calls: {shown} "
            f"(net drift {drift:+d} laps)."
        )

    def _render_contingencies(self) -> list[str]:
        live = self._live_contingencies()
        if not live:
            return ["  Contingencies you declared last lap: none."]
        lines = ["  Contingencies you declared last lap:"]
        lines.extend(
            f"    - [{c['priority']}] \"{c['trigger']}\" -> {c['switch_to']}" for c in live
        )
        return lines

    def block(self) -> str | None:
        """The prompt block, or ``None`` when there is no history to report.

        ``None`` rather than an empty string or a "held for 0 laps" line: on the
        first lap of a race there is no previous call, and inventing a statement
        about one is how a prompt teaches a model something untrue.
        """
        if not self._entries:
            return None
        lines = [
            "DECISION MEMORY (your own previous calls this race):",
            self._render_hold(),
            self._render_targets(),
            *self._render_contingencies(),
        ]
        # CONTINUATION and COUNTERWEIGHT are deliberately adjacent and deliberately
        # in this order: carry the plan, but do not treat carrying it as a promise.
        # Shipping the first without the second is the configuration that measurably
        # anchored the model at the decision lap.
        tail = (CONTINUATION if self._is_continuing_a_hold() else "") + COUNTERWEIGHT
        return "\n".join(lines) + "\n" + tail + "\n"
