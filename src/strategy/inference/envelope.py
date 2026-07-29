"""Operating-envelope contract for ML model inputs (#709).

WHY THIS EXISTS
----------------
Our models never refuse out-of-range input; they answer with the same
confidence whether the call falls inside what they were trained on or not.
N26's tire TCN did exactly that: inference skipped the notebook's per-stint
grouping and left-padded with a repeated first lap instead of zeros, so it was
fed sequences the training run never produced in roughly 87% of its calls, for
two years, before anyone noticed.

Exactly ONE call site has since patched the symptom by hand:
``src/agents/pit_strategy_agent.py`` clips ``tyre_life_in`` at 50 laps because
N15 clipped it at training time (cell 11). That constant is an operating
envelope written as a bare number.

``src/agents/tire_agent.py``'s ``lap_out_of_range`` looks like a second
instance and is **not** one, which is worth stating so nobody folds them
together later: it tests ``1 <= lap <= total_laps`` and whether the driver is
on track. Those are questions about whether the CALL makes sense, not about
whether the MODEL was trained here. Data validity and operating envelope are
different contracts, and only the second belongs in this module.

This module turns the one real instance into a rule instead of a habit. An
:class:`OperatingEnvelope` names the bounds a model was actually trained on;
:meth:`OperatingEnvelope.check` compares a feature vector against them and
returns an :class:`EnvelopeVerdict`.

LABELLING ONLY
--------------
A verdict is a label, nothing more. Checking a feature vector against an
envelope must NEVER touch, clip, or refuse a prediction by itself -- it only
tells the caller whether to trust the one it already has. Whether an agent
skips a tool call, downgrades its confidence, or ignores the verdict entirely
is a decision for that call site (tracked separately as #710), not for this
module.

WHY A MISSING FEATURE IS ITS OWN STATE, NOT A NUMBER
-----------------------------------------------------
A feature that is absent from the input (or NaN) is UNKNOWN: neither in-range
nor out-of-range. This repo has shipped real bugs from collapsing "we do not
know" into a numeric default that a real value could also legitimately take
(a missing ``Position`` defaulted to 0 once matched the car that had just
crashed, because 0 was also a valid grid position). ``check`` follows the same
rule here: an unknown feature is tracked in ``EnvelopeVerdict.unknown`` and is
never compared against a bound.

NO MANIFEST FORMAT TO PARSE
----------------------------
The task behind this module expected a loader that reads ranges out of a
``feature_manifest_*.json`` file. Both real manifests in this repo
(``data/processed/feature_manifest_laptime.json``,
``data/processed/tiredeg_feature_manifest.json``) and every
``data/models/*/model_config.json`` were read before writing this: none of
them carries a numeric min/max range, only feature name lists and categorical
encodings (``feature_manifest_laptime.json``'s ``categorical_encoding`` maps
compound strings to ints, for instance -- there is no bounds table anywhere).
So there is nothing to parse. ``OperatingEnvelope`` is built directly from
explicit bounds the caller sources itself (a notebook cell comment, a
training-config constant such as ``_MAX_TRAINED_TYRE_LIFE`` in
``pit_strategy_agent.py`` today). That constructor is the loader.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

# One feature's declared operating bound: (lower, upper), both inclusive.
Bound = tuple[float, float]


@dataclass(frozen=True)
class FeatureViolation:
    """One declared feature whose value fell outside its trained bound.

    Carries what a caller needs to log or raise without re-deriving anything:
    the value actually passed in, and the ``[lower, upper]`` bound it broke.
    """

    value: float
    lower: float
    upper: float


@dataclass(frozen=True)
class EnvelopeVerdict:
    """Result of checking one feature vector against an OperatingEnvelope.

    A pure label (see module docstring): nothing may use a verdict to alter a
    prediction, only to decide whether to trust one.

    Attributes:
        envelope_name: name of the OperatingEnvelope that produced this.
        violations: features that were present and numerically outside their
            declared bound, keyed by feature name.
        unknown: features the envelope declares but that were missing (or
            NaN) in the checked input. Kept separate from ``violations`` on
            purpose -- a model given no value for a feature was not given a
            bad one, and the two states must stay distinguishable at the
            call site.

    ``bool(verdict)`` is True only when every declared feature was present
    and within bounds, so a call site can write ``if not verdict: ...`` to
    mean "do not trust this call blindly", then inspect ``.violations`` and
    ``.unknown`` to say why.
    """

    envelope_name: str
    violations: Mapping[str, FeatureViolation]
    unknown: frozenset[str]

    @property
    def in_range(self) -> bool:
        """True iff there is no violation and no unknown feature."""
        return not self.violations and not self.unknown

    def __bool__(self) -> bool:
        return self.in_range


def _is_unknown(value: float | None) -> bool:
    """A value is unknown when it is None or NaN, and never coerced to a number.

    Guards the rule in the module docstring: a NaN slipping in from a pandas
    frame must be treated the same as an absent dict key, not compared
    numerically (NaN comparisons are silently always False, which would make
    an unknown value read as falsely in-range).
    """
    if value is None:
        return True
    return isinstance(value, float) and math.isnan(value)


@dataclass(frozen=True)
class OperatingEnvelope:
    """The input range a model's answer is actually valid over.

    Built directly from bounds the caller sourced from the training
    notebook or config (see module docstring: no manifest in this repo
    carries ranges today, so there is no separate file-loading path).
    Checking a feature vector against an envelope never changes what the
    model predicts; it only labels the call.
    """

    name: str
    bounds: Mapping[str, Bound]

    def __post_init__(self) -> None:
        """Reject an envelope whose own declared bounds are inverted."""
        for feature_name, (lower, upper) in self.bounds.items():
            if lower > upper:
                raise ValueError(
                    f"{self.name}: bound for '{feature_name}' is inverted "
                    f"(lower={lower} > upper={upper})"
                )

    def check(self, features: Mapping[str, float | None]) -> EnvelopeVerdict:
        """Compare a feature vector against this envelope's declared bounds.

        Only features this envelope declares are checked; extra keys in
        ``features`` are ignored. Never raises on an out-of-range or missing
        value -- what to do with the verdict is the caller's decision (#710).
        """
        violations: dict[str, FeatureViolation] = {}
        unknown: set[str] = set()
        for feature_name, (lower, upper) in self.bounds.items():
            value = features.get(feature_name)
            if _is_unknown(value):
                unknown.add(feature_name)
                continue
            if not (lower <= value <= upper):
                violations[feature_name] = FeatureViolation(value=value, lower=lower, upper=upper)
        return EnvelopeVerdict(
            envelope_name=self.name,
            violations=violations,
            unknown=frozenset(unknown),
        )
