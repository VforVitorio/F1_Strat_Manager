"""The deterministic pit guard rails, in a module that imports nothing heavy.

WHAT THESE ARE, AND WHAT THEY ARE NOT
--------------------------------------
These are **anti-hallucination bounds**, not a strategy model. They exist so a
language model cannot recommend a lap-2 stop because it felt like it. The
authoritative statement of the policy is prose, in the N28 pit agent's prompt
(``src/agents/pit_strategy_agent.py``, "Strategic guard-rails (HARD
constraints)"), and this module is the DETERMINISTIC MIRROR of that prose so the
offline ``no-llm`` path behaves like the LLM path.

That direction matters: **the prompt is the specification and this file is the
copy.** When they disagree, this file is wrong until proven otherwise.

They are *proscriptive* — they forbid an action. That is a different thing from
a *prescriptive* rail that forces one (the Safety Car ``PIT_NOW`` rail, rejected
in #464). A proscriptive bound on a generative model's output is legitimate with
or without a regulation behind it; the test it must pass is CALIBRATION — the
threshold has to sit where real strategy essentially never goes, so it separates
absurd from sane rather than unusual from usual. See #716, where the
minimum-stint threshold is measured against real stint lengths.

WHY THIS IS ITS OWN MODULE (#708):
These rules used to live in ``no_llm.py``, which imports the agent stack and
therefore loads model weights at import time. That made "what is the minimum
stint?" a question you could not ask without ``data/models/`` on disk — it broke
``f1-eval`` on any install without the weights, and it broke CI. A policy
constant should not cost a LightGBM load, so the rules live here and ``no_llm``
imports them. Anything that needs to MIRROR a rail must import it from here and,
better still, call ``apply_guard_rails`` rather than re-derive its boundaries:
the eval tier initially retyped ``remaining < 3`` against the rail's
``remaining <= 3`` and shipped a test that encoded the off-by-one.
"""

from __future__ import annotations

_PIT_ACTIONS = frozenset({"PIT_NOW", "UNDERCUT", "OVERCUT", "REACTIVE_SC"})
_NO_PIT_BEFORE_LAP = 5
_NO_PIT_LAST_N_LAPS = 3
_CLIFF_P10_SAFE = 2
_MIN_STINT_LAPS = {"SOFT": 8, "MEDIUM": 12, "HARD": 15}
_DEFAULT_MIN_STINT = 10


def apply_guard_rails(
    action: str,
    lap: int,
    total_laps: int,
    compound: str,
    tyre_life: int,
    cliff_p10: float = 99.0,
    sc_active: bool = False,
) -> tuple[str, str | None]:
    """Override *action* with STAY_OUT when a hard strategic constraint fires.

    Rules: no pit before lap 5; no pit in the last 3 laps unless the cliff is
    imminent (cliff_p10 < 2); minimum stint SOFT 8 / MEDIUM 12 / HARD 15. Returns
    ``(action, reason)`` with ``reason=None`` when no rail fired.

    Args:
        sc_active: A neutralisation (Safety Car or VSC) is deployed right now.
            It suspends the two bounds whose premise it falsifies, and NOT the
            third. Both of those exist because a stop costs ~22-25 s; under a
            neutralisation the field is delta-limited and queued, the relative
            loss collapses, and a bound written to catch nonsense must not be what
            blocks the most valuable stop in racing (#716).

    THE ONE BOUND A SAFETY CAR DOES NOT SUSPEND is the end-of-race one, and this
    is a DELIBERATE divergence from the prompt, which does exempt it. The prompt's
    stated rationale there is cost, but cost is not the binding objection so late:
    Art. 55.17 ends the race behind the Safety Car if it is still deployed on the
    final lap, so the position surrendered by stopping is unrecoverable **by
    regulation** rather than merely expensive. A neutralisation makes that more
    true, not less. Suspending this bound under SC would re-create the exact defect
    corrected in #464, where the system emitted PIT_NOW with the reason "too late
    to pit". Raised on #716 as a question for the prompt, not settled unilaterally.

    KNOWN REMAINING DIVERGENCE, also tracked in #716: the prompt exempts the
    early-race bound when "radio confirms damage/puncture/mechanical failure".
    That is not wired here, because mapping radio alerts onto a damage semantic is
    its own source of error and is not worth guessing at. The gap is narrow (it
    costs a legitimate early stop after first-lap contact) and it is documented
    rather than faked.
    """
    if action not in _PIT_ACTIONS:
        return action, None

    remaining_laps = total_laps - lap

    # Suspended under a neutralisation: the ~22-25 s premise behind this bound is
    # false while the field is queued behind the Safety Car.
    if lap < _NO_PIT_BEFORE_LAP and not sc_active:
        return "STAY_OUT", f"guard-rail: pit window not open (lap < {_NO_PIT_BEFORE_LAP})"

    # NOT suspended under a neutralisation. See the docstring: Art. 55.17 makes the
    # ceded position unrecoverable rather than expensive, which a Safety Car
    # aggravates. Only an imminent tyre failure still overrides it.
    if remaining_laps <= _NO_PIT_LAST_N_LAPS and cliff_p10 >= _CLIFF_P10_SAFE:
        return "STAY_OUT", f"guard-rail: too late to pit (<={_NO_PIT_LAST_N_LAPS} laps left)"

    # Suspended under a neutralisation, matching the prompt: a cheap stop makes a
    # short stint affordable, so "the set still has useful life" stops being the
    # deciding consideration.
    min_life = _MIN_STINT_LAPS.get(compound, _DEFAULT_MIN_STINT)
    if tyre_life < min_life and not sc_active:
        return (
            "STAY_OUT",
            f"guard-rail: minimum stint not reached ({compound} {tyre_life}/{min_life} laps)",
        )

    return action, None
