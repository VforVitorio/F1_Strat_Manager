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
absurd from sane rather than unusual from usual.

#716 ran that test on all four bounds against 1900 real green-flag stops. The two
lap-based ones passed untouched; the two minimum-stint ones overshot the ceiling by
between two and four times and were reset from the measurement. Every bound now
carries its measured veto share, or the article that makes it a fact, at its
definition below. A bound with neither does not belong in this file.

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

# THE CALIBRATION CEILING every bound below is HELD TO: a bound may veto at most 5%
# of the real green-flag stops in the measured sample. Above that it is separating
# unusual from usual rather than absurd from sane, which is the one job an
# anti-hallucination bound has.
#
# Held to, not necessarily set from. The two minimum-stint bounds failed the ceiling
# and were reset to the largest value that clears it, so for them the ceiling is also
# the derivation. The two lap-based bounds already cleared it and were left exactly
# where they were; they are NOT maximal under the rule, and moving them up to the
# largest passing value would be a change nobody asked for on evidence that only says
# they are not currently wrong.
#
# The sample is 1900 real green-flag stops across 70 races of 2023-2025 raw laps.
# `documents/eval_reports/stint_lengths.md` regenerates the four minimum-stint shares
# from these constants on every run, so that report always grades what is actually
# shipping, and `tests/eval/test_stint_lengths.py` asserts the ceiling holds on them.
# The two lap-based shares below have no such home: they were measured once over the
# same sample and are recorded here rather than in an artefact, so treat them as a
# dated finding rather than as something a report re-checks.
_CALIBRATION_CEILING = 0.05

# Vetoes 42/1900 = 2.21% of real stops. Unchanged by #716: it already cleared the
# ceiling comfortably.
_NO_PIT_BEFORE_LAP = 5

# Vetoes 26/1900 = 1.37%. Also unchanged, and the one bound that is partly a FACT
# rather than a calibration: under a Safety Car, Art. 55.17 ends the race behind it
# if it is still deployed on the final lap, so the position a late stop surrenders
# is unrecoverable BY REGULATION. That article does not reach a green-flag lap,
# where the bound rests on the ~22-25 s cost instead, and on this measurement.
#
# This is also the bound behind the four excluded stops in the six-race
# `decision-modes` subset (Monaco VER, Lusail STR and HAD, Monza OCO). #716's issue
# body attributes four stops to the early-race bound as well; measured on that
# subset the early-race bound excludes NONE, and `decision_modes.md` carries no
# `opening_laps` row at all.
_NO_PIT_LAST_N_LAPS = 3

_CLIFF_P10_SAFE = 2

# Recalibrated by #716 from SOFT 8 / MEDIUM 12 / HARD 15, which vetoed 15.5% /
# 17.0% / 12.2% of real stops: one stop in six, three times the ceiling. Each value
# is now the largest integer whose veto share stays at or under 5%.
#
#   SOFT   2 -> 3.2% (341 stops)   MEDIUM 7 -> 4.6% (896)   HARD 8 -> 4.7% (548)
#
# SOFT lands lowest because real SOFT stints genuinely are the shortest: 11 of them
# ran exactly one lap. The bound is not a model of degradation and must not be read
# as one. It only forbids stopping on a set fitted so recently that no strategy
# produced it, and where that line falls differs per compound because the evidence
# differs per compound.
_MIN_STINT_LAPS = {"SOFT": 2, "MEDIUM": 7, "HARD": 8}

# The bound every compound with no entry above resolves to, which in practice means
# INTERMEDIATE and WET. Recalibrated by #716 from 10, and it was the WORST of the
# set at 20.0% of real wet stops vetoed, because it was also the only one nothing
# had ever measured: the stint-length report dropped wet stops on the strength of a
# comment claiming they ran no minimum-stint rule at all. They run this one.
#
#   6 -> 4.55% (110 wet stops)
#
# KNOWN DIVERGENCE, and it is the reverse of the usual one: this bound has NO prose copy
# in either prompt. N28 and N31 both state the three dry minimums and say nothing about
# what happens on an INTERMEDIATE or a WET, so the offline path enforces a bound the LLM
# path was never told about. Documented rather than closed, because adding it means
# writing new prompt text and that is a behaviour change on the default path, not a
# recalibration. It matters most in exactly the races where it is least likely to be
# noticed.
_DEFAULT_MIN_STINT = 6


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

    Three bounds, in the order they are evaluated: the pit window is not open yet;
    it is too late to matter unless the cliff is imminent; the set was fitted too
    recently for any strategy to have produced this call. The numbers are the module
    constants and are deliberately not restated here, because a retyped boundary has
    shipped wrong in this codebase before. Returns ``(action, reason)`` with
    ``reason=None`` when no bound fired.

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
