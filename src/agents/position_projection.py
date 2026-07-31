"""Project where our car comes out, in cars rather than in seconds (#554).

The Monte Carlo layer used to score strategies in generic seconds divided by a
flat 1.5 s/position, over a sampled state containing no cars at all. Losing 20 s
costs zero positions with a 25 s cushion behind and three positions with cars at
+2 / +8 / +15 s, and only a model that knows *which cars are where* can tell the
difference. This module is that model: one pure primitive that turns per-rival
gaps into a projected end-of-window track position.

Nothing here loads a model, reads a file or touches the orchestrator. It takes
plain state in and returns arrays out, which is what lets the projection be
validated against real pit stops before a single call site changes.

SIGN CONVENTION, stated once and loudly because a docstring that lied about a
sign has already cost this project a bug: gaps follow the RaceStateManager
contract, ``gap_s = rival_elapsed_time - our_elapsed_time``. So

    gap_s < 0  =>  the rival is AHEAD of us (less race time elapsed)
    gap_s > 0  =>  the rival is BEHIND us

Losing time pushes our elapsed time up, which pushes every gap DOWN.

KNOWN v1 SIMPLIFICATIONS, named so nobody mistakes them for oversights. Each is
a real racing effect this model does not carry, left out because the alternative
was an unmeasured constant, which is what the redesign exists to remove:

- **A gap crossing counts as a position change.** In the pit-stop cases that
  dominate the window this is exact — you emerge where you emerge. On track it
  is optimistic: three tenths does not pass at Monaco and does at Monza, and
  nothing here knows the difference. N11 already models overtake probability and
  is the natural gate, but it was trained on observed gaps, so feeding it the
  counterfactual gaps a projection invents would run it off its own manifold.
- **The out-lap is treated like any other lap on fresh rubber.** In reality a new
  set needs a lap, sometimes a sector, to switch on, and on a hard compound in
  cold conditions the out-lap can be slower than the worn set it replaced. In our
  own dry-race sample the effect is small at the flying-lap level, under a tenth,
  and most of its true cost already sits inside the pit-loss figure — so it is
  left folded into the measured undercut band rather than modelled per lap.
- **Dirty air is priced at one moment, not continuously.** The measured
  clean-air gain enters only when a car directly ahead boxes, which is the
  moment it decides something. Running the whole window stuck in traffic is not
  otherwise penalised, so the projection understates how bad a lap behind a
  slower car is at a circuit like Suzuka.
- **Neutralisation hazard is flat across the race.** The measured per-circuit
  rate pools every lap, while the real thing spikes on lap one and around the
  pit windows.
- **Lapped cars are counted by elapsed time on the same lap number**, which
  matches the timing screen but does not model the unlapping procedure before a
  restart (Art. 55.13).

THREE HORIZONS FOR ONE FACT, stated once and prominently because the module used
to leave it implicit and that read as an inconsistency waiting to be "fixed"
(#742). ``rival_time_deltas``, ``_terminal_gaps`` and ``rank_targets`` each ask
whether a rival's outstanding pit stop should cost them time, and each answers
correctly for a DIFFERENT horizon, not for the same one three times::

    rival_time_deltas -> rival.is_pitting                              (inside the window)
    _terminal_gaps     -> rival.stop_pending is True and not is_pitting (race end)
    rank_targets        -> rival.stop_pending is True                   (after both pit cycles)

A rival who owes a stop but is not taking it inside the window genuinely loses
no time inside the window, so ``is_pitting`` is the right predicate there. The
same rival genuinely falls back by the race end whether or not they stop this
lap, so ``stop_pending`` is right at that horizon; ``_terminal_gaps`` then
excludes a rival who IS pitting right now, because ``rival_time_deltas`` already
charged that stop inside the window and charging it again would push them a
full pit cycle further back than they will ever be. ``rank_targets`` looks past
both pit cycles, where whether the stop happens on THIS lap stops mattering, so
it drops the ``is_pitting`` guard entirely. Making the three agree would not fix
a bug: it would delete the distinction between "now", "by the end of the race"
and "once everyone has stopped" that the three horizons exist to carry.

--- WHERE TO CHANGE IF THE RIVALS CONTRACT CHANGES ---
``RivalState`` mirrors the fields ``RaceStateManager.get_rival_states`` emits
(``interval_to_driver_s``, ``is_pitting``, ``lap_time_s``). If that contract
gains or renames a field, the adapter that builds ``RivalState`` moves with it;
this module never reads a DataFrame.
"""

from __future__ import annotations

import json
import math
import unicodedata
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Sequence

import numpy as np

_MEASURED_TABLES = Path(__file__).resolve().parents[2] / "data" / "mc_measured_v1.json"

# Fallbacks used only when the measured tables are unavailable (a wheel install
# without data/). They are the values measured on 2026-07-25 over 71 races, so a
# missing file degrades to the same numbers rather than to invented ones.
DEFAULT_UNDERCUT_BAND_S = 4.91
DEFAULT_NEUTRALISATION_RATE = 0.0179
DEFAULT_RACING_LAPS_UNDER_SC = 2.61
# Measured separately from the SC figure because a VSC leaves more of the window
# raceable (2.90 laps against 2.61). This constant was REFERENCED at the bottom of
# measured_racing_laps and never defined, so the fallback that exists to survive a
# missing data/ raised NameError on exactly the VSC branch it was there to serve.
DEFAULT_RACING_LAPS_UNDER_VSC = 2.90

# What a gap becomes when a car ahead exists but its interval was never measured.
# NOT zero: 0.0 reads as side by side, and both the clean-air band above and N27's
# sub-1.0s DRS window act on that, so a missing measurement would look like the most
# aggressive possible situation. Three places needed this number (the CLI, the arcade
# and the telemetry backend) and each had grown its own, which is how they drift.
#
# It is still a FABRICATED number and a real 2.0s gap is common, so it does not meet
# the rule that a default must never be a value the code can also legitimately find.
# It is less harmful than 0.0, not correct. The correct fix is RaceState.gap_ahead_s
# becoming `float | None` the way RivalState.gap_ahead_s already is here, where every
# consumer guards with `is not None`. That is a Pydantic contract change, so it is
# tracked rather than smuggled in.
GAP_UNKNOWN_FALLBACK_S = 2.0

# How close we must be for the car ahead to be costing us downforce. This is not
# a tuning knob: it is the proximity the clean-air table was measured at, so
# crediting the gain to a car eight seconds back would apply a number outside
# the sample it came from.
CLEAN_AIR_BAND_S = 2.0

# Margin is a tie-break, not a second currency: at most a third of a position.
MARGIN_WEIGHT = 0.1
# Capped at three seconds because that is roughly where the car behind stops
# being a threat within one lap: DRS arms inside one second (Art. 22.1), and by
# about three the follower is out of dirty air and cannot mount an attack next
# time by. Buffer beyond that is real comfort but no longer decision-relevant,
# so the tie-break saturates rather than rewarding a 40-second cushion forever.
MARGIN_CLIP_S = 3.0


@dataclass(frozen=True)
class RivalState:
    """One rival as the pit wall sees them at the decision lap.

    Attributes:
        driver:      FIA three-letter code.
        gap_s:       Signed seconds, rival minus us. Negative means ahead of us.
                     ``None`` means unknown, and an unknown gap keeps the rival
                     out of the projection entirely rather than at a made-up
                     zero — a searchable sentinel is how #428 happened.
        pace_delta_s: Seconds per lap this rival is slower than us (negative =
                     faster). Zero when unknown, which is the neutral
                     assumption rather than a guess in either direction.
        is_pitting:  They entered the pit lane on this lap. A fact from timing,
                     the only rival-strategy signal v1 trusts.
        stop_pending: Whether they still owe the Art. 30.5(m) stop. ``None``
                     when their compound history cannot settle it.
        stop_loss_s: Total pit loss if they stop in the window (lane traversal
                     plus the physical stop).
    """

    driver: str
    gap_s: float | None
    pace_delta_s: float = 0.0
    is_pitting: bool = False
    stop_pending: bool | None = None
    stop_loss_s: float = 0.0

    @property
    def is_ahead(self) -> bool:
        """Whether the rival is ahead of us right now.

        An unknown gap counts as no, and so does a NaN one: ``nan < 0`` is False
        anyway, but stating it keeps the rule visible next to the comparison
        rather than resting on IEEE-754 happening to agree with us.
        """
        return self.gap_s is not None and math.isfinite(float(self.gap_s)) and self.gap_s < 0

    @property
    def gap_ahead_s(self) -> float | None:
        """Positive seconds we sit behind this rival, or None if not ahead."""
        if not self.is_ahead:
            return None
        return -float(self.gap_s)


@dataclass(frozen=True)
class DriverPlan:
    """What one candidate strategy does to our own car over the window.

    Attributes:
        name:            STAY_OUT / PIT_NOW / UNDERCUT / OVERCUT.
        stops_in_window: Whether this candidate takes the stop inside the window.
        stop_offset_laps: Laps from now until that stop (0 = this lap). Ignored
                     when the candidate does not stop.
    """

    name: str
    stops_in_window: bool
    stop_offset_laps: int = 0


@dataclass(frozen=True)
class ProjectionConfig:
    """Measured constants and race context the projection needs.

    Everything here is either measured (see ``scripts/measure_mc_tables.py``) or
    a fact of the current race. No strategy opinions live in this object.

    Attributes:
        window_laps:      Decision horizon W.
        racing_laps:      Laps inside the window that will actually be raced.
                          Equals ``window_laps`` under green and drops toward
                          zero under a neutralisation, which is what makes the
                          Art. 55.17 endgame fall out of the arithmetic: with no
                          racing laps left, fresh tyres buy nothing.
        fresh_gain_s:     Seconds per lap a fresh tyre gains. The FALLBACK for
                          ``deg_cost_s``, used only when the tyre model gave no
                          reading; the two are the same quantity and are never
                          charged together.
        deg_cost_s:       Seconds per lap the CURRENT set costs versus fresh, read
                          from the tyre model rather than assumed. ``None`` when
                          the model had no reference to subtract, which leaves
                          ``fresh_gain_s`` in charge. Measured, not tuned: this is
                          a fact about the car's tyres, so it belongs here, whereas
                          a hand-picked weight on it would not.
        cliff_loss_s:     Seconds per lap lost past the tyre cliff. Charged only on
                          laps run PAST the cliff, so it does not overlap
                          ``deg_cost_s``, which is charged on every old-set lap.
                          A tyre ten laps from the cliff but 0.4 s off the pace was
                          previously priced identically to a fresh one.
        neutralisation_saving_s: Seconds a stop saves when taken under a
                          neutralisation (the field is queued, so the pit loss
                          costs less).
        undercut_band_s:  Beyond this gap an undercut effectively never works.
        future_neutralisation_prob: q_f, the chance a later neutralisation
                          covers a stop we have not taken yet.
        laps_remaining:   Laps left in the race, for the q_f estimate.
        mandatory_stop_pending: Whether WE still owe the Art. 30.5(m) stop.
                          ``None`` means unknown and disables the liability term
                          rather than assuming either way.
        margin_weight:    Weight of the seconds-margin tie-break.
        clean_air_gain_s: Seconds per lap gained running in air the car directly
                          ahead has just vacated, for this circuit. Only a plan
                          that runs on after the target boxes accrues it, which
                          is what makes an overcut a real move rather than a
                          worse pit stop. Zero under a neutralisation, where the
                          field runs to a delta and clear track buys nothing.
        neutralisation_onset_rate: Per-lap probability this circuit throws a
                          neutralisation. Distinct from
                          ``future_neutralisation_prob``, which is the same rate
                          integrated over the whole remaining race for a stop we
                          are deferring past the window. This one prices the
                          single extra lap a delayed stop spends waiting.
    """

    window_laps: int = 5
    racing_laps: float = 5.0
    fresh_gain_s: float = 0.25
    deg_cost_s: float | None = None
    cliff_loss_s: float = 0.80
    neutralisation_saving_s: float = 8.0
    undercut_band_s: float = DEFAULT_UNDERCUT_BAND_S
    future_neutralisation_prob: float = 0.0
    laps_remaining: int = 0
    mandatory_stop_pending: bool | None = None
    margin_weight: float = MARGIN_WEIGHT
    clean_air_gain_s: float = 0.0
    neutralisation_onset_rate: float = 0.0


@dataclass(frozen=True)
class ProjectionResult:
    """Per-draw output of a projection for one candidate.

    Attributes:
        positions:   Projected track position at the end of the window. This is
                     the REJOIN horizon, and it is what the 1810-stop ground
                     truth grades, so it must keep meaning exactly that.
        margins_s:   Seconds of buffer to the nearest projected car behind,
                     clipped, and 0.0 when nothing is behind us.
        terminal_positions:
                     Projected position once every KNOWN outstanding stop has
                     been served, ours and theirs. Scoring happens here rather
                     than at the window end, because a candidate that skips the
                     window has not avoided the mandatory stop, only deferred
                     it, and a rival who still owes one has not really passed us.
        rivals_used: How many rivals had a usable gap and entered the count.
    """

    positions: np.ndarray
    margins_s: np.ndarray
    terminal_positions: np.ndarray
    rivals_used: int


@lru_cache(maxsize=1)
def measured_tables(path: str | None = None) -> dict:
    """The committed measurements, read once and cached.

    Returns an empty dict when the file is absent (a wheel install without
    ``data/``), which leaves every caller on the module's DEFAULT_* constants.
    Those are not invented numbers: they are the values this same file held on
    2026-07-25, so a missing table degrades to the last measured state rather
    than to a guess. Regenerate with ``scripts/measure_mc_tables.py``.
    """
    tables_path = Path(path) if path else _MEASURED_TABLES
    if not tables_path.exists():
        return {}
    return json.loads(tables_path.read_text(encoding="utf-8"))


def measured_undercut_band_s() -> float:
    """The measured undercut band, or the last measured value if the file is gone."""
    band = measured_tables().get("undercut_band", {}).get("u_band_s")
    return float(band) if band else DEFAULT_UNDERCUT_BAND_S


def measured_neutralisation_rate(circuit: str | None = None) -> float:
    """Per-lap onset hazard for ``circuit``, never zero.

    A circuit we have never raced gets the pooled figure rather than zero, and so
    does a circuit that simply has not thrown a Safety Car in our three seasons.
    Zero is not a neutral default here: it drives ``q_f`` to 0, which tells the
    decision layer that no future neutralisation can ever turn up to cover a
    stop, and that biases the terminal liability upward on every lap.

    That second case is real, not hypothetical. Monza and Budapest both measure
    exactly 0 (0 onsets in 157 and 210 racing laps), and Monza is the archetypal
    Art. 55.17 circuit. A zero count is not evidence of a zero rate: the upper
    bound of that interval comfortably covers the pooled value, so the honest
    reading is "we have not seen one here", not "one cannot happen here".
    """
    table = measured_tables().get("neutralisation_rate", {})
    pooled = (table.get("pooled") or {}).get("rate")
    fallback = float(pooled) if pooled is not None else DEFAULT_NEUTRALISATION_RATE

    if not circuit:
        return fallback

    cell = (table.get("per_circuit") or {}).get(_resolved_circuit_key(circuit)) or {}
    rate = cell.get("rate")
    if rate is None:
        return fallback
    # An observed zero means "never seen", not "impossible". Fall back rather
    # than let a small sample silence the whole option-value term.
    if float(rate) <= 0.0:
        return fallback
    return float(rate)


def measured_clean_air_s(circuit: str | None = None) -> float:
    """Seconds per lap a follower gains at ``circuit`` once the car ahead boxes.

    Measured over 479 cases where a car sat within two seconds of, and directly
    behind, a driver who then pitted: their mean lap time over the three raced
    laps after the stop against the three before it, corrected by the measured
    lap-to-lap trend so the follower's own fuel burn and tyre wear are not read
    as clean air.

    The spread is the finding, which is why this is per circuit and not a single
    constant. It runs from roughly +0.77 s at Suzuka and +0.65 at Monaco down to
    zero or below at Monza and Spielberg — that is, largest exactly where
    following costs the most downforce, and absent where the tow is worth more
    than the clear track. Nothing in the measurement knows about downforce; the
    ordering came out of the lap times.

    Negative cells are returned as measured. A circuit where losing the car
    ahead costs you a tow is not a measurement error, and clamping it to zero
    would tell the decision layer that an overcut is free at Monza.

    Unknown circuits get the pooled figure, which is deliberately small.
    """
    table = measured_tables().get("clean_air", {})
    pooled = (table.get("pooled") or {}).get("corrected_mean_s")
    fallback = float(pooled) if pooled is not None else 0.0

    if not circuit:
        return fallback

    cell = (table.get("by_circuit") or {}).get(_resolved_circuit_key(circuit)) or {}
    gain = cell.get("corrected_mean_s")
    return float(gain) if gain is not None else fallback


# Circuits this repo files under more than one name. The measured tables and the
# traversal table are keyed by the circuit slug, but callers hand us whatever
# their surface holds: 2023 filed Barcelona under a country name, and Miami has
# three spellings across the repo. Left explicit so an unresolved name is a
# genuinely unknown circuit rather than a silent keyspace miss (#448).
_CIRCUIT_ALIASES: dict[str, str] = {
    "Spain": "Barcelona",
    "Miami Gardens": "Miami",
    "Miami_Gardens": "Miami",
}


def _key_candidates(name: str) -> list[str]:
    """Every spelling of ``name`` worth trying against a circuit-keyed table.

    Ordered cheapest-first: the name as given, its explicit alias, the
    underscored folder form, then the two resolvers in ``gp_slugs``. Returning a
    list rather than resolving eagerly keeps the caller in charge of which table
    it is matching against, since the two tables do not hold identical key sets.
    """
    from src.f1_strat_manager.gp_slugs import canonical_gp_name, slug_from_event_name

    spaced = name.replace("_", " ")
    candidates = [name, _CIRCUIT_ALIASES.get(name), spaced, _CIRCUIT_ALIASES.get(spaced)]
    for resolver in (slug_from_event_name, canonical_gp_name):
        try:
            candidates.append(resolver(name))
        except (ValueError, KeyError):
            continue

    # Accent-folded spellings, because two of our circuits carry diacritics
    # (São Paulo, Montréal) and they lose them whenever a name passes through a
    # non-UTF-8 console, a filename or a hand-typed argument. Matched by folding
    # BOTH sides so "Sao Paulo" finds "São Paulo" without a second alias entry.
    folded = {_fold_accents(c): c for c in candidates if c}
    for key in list(_traversal_table()) + list(
        (measured_tables().get("neutralisation_rate") or {}).get("per_circuit") or {}
    ):
        if _fold_accents(key) in folded:
            candidates.append(key)

    seen, ordered = set(), []
    for candidate in candidates:
        if candidate and candidate not in seen:
            seen.add(candidate)
            ordered.append(candidate)
    return ordered


def _fold_accents(text: str) -> str:
    """``São Paulo`` and ``Sao Paulo`` collapse to the same comparison key."""
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch)).casefold()


def _lookup_by_circuit(table: dict, name: str):
    """First value in ``table`` matching any accepted spelling of ``name``."""
    for candidate in _key_candidates(name):
        if candidate in table:
            return table[candidate]
    return None


@lru_cache(maxsize=64)
def _resolved_circuit_key(name: str) -> str:
    """The key a circuit-keyed measured table actually holds for ``name``."""
    per_circuit = (measured_tables().get("neutralisation_rate") or {}).get("per_circuit") or {}
    for candidate in _key_candidates(name):
        if candidate in per_circuit:
            return candidate
    return name


@lru_cache(maxsize=1)
def _traversal_table() -> dict:
    """Per-circuit pit-lane traversal seconds, keyed by the slug agents query with.

    Read straight from N15's committed model config rather than through the pit
    agent, because the decision layer needs this number on the no-LLM path too and
    constructing N28 there would load a stack of models to answer one lookup.

    Re-keyed through ``rekey_by_slug``: the table ships keyed by FastF1 event names
    whose overlap with the slug keyspace is exactly zero, which is how every
    circuit lookup silently missed and froze traversal at a single constant (#448).
    """
    config_path = Path(__file__).resolve().parents[2] / "data" / "models" / "pit_prediction"
    config_file = config_path / "model_config.json"
    if not config_file.exists():
        return {}

    from src.f1_strat_manager.gp_slugs import rekey_by_slug

    payload = json.loads(config_file.read_text(encoding="utf-8"))
    return rekey_by_slug(payload.get("circuit_traversal_lookup", {}), "circuit_traversal_lookup")


def traversal_seconds(gp_name: str | None) -> float | None:
    """Pit-lane traversal for ``gp_name``, or None when the circuit is unknown.

    None rather than a pooled average on purpose: the caller decides whether to
    fall back, and a silent average would hide a keyspace drift exactly the way
    #448 did. The real spread is 19.7 s at Budapest to 27.5 s at Marina Bay, which
    is the difference between a stop that costs a place and one that does not.
    """
    if not gp_name:
        return None
    value = _lookup_by_circuit(_traversal_table(), gp_name)
    return float(value) if value is not None else None


def measured_racing_laps(neutralisation: str = "sc") -> float:
    """Measured racing laps left inside the window under ``sc`` or ``vsc``."""
    kinds = measured_tables().get("sc_window", {}).get("by_kind", {})
    mean = (kinds.get(neutralisation) or {}).get("racing_laps_in_window", {}).get("mean")
    if mean is not None:
        return float(mean)
    return (
        DEFAULT_RACING_LAPS_UNDER_VSC if neutralisation == "vsc" else DEFAULT_RACING_LAPS_UNDER_SC
    )


def future_neutralisation_probability(rate_per_lap: float, laps_remaining: int) -> float:
    """Probability at least one neutralisation begins in the laps that remain.

    ``1 - exp(-rate * laps)`` rather than ``rate * laps``: the product form runs
    past 1 on a long run and would hand the decision layer a certainty it has no
    right to. Clamped anyway, because a probability that can leave [0, 1] is a
    bug waiting for its first extreme input.
    """
    if rate_per_lap <= 0 or laps_remaining <= 0:
        return 0.0
    probability = 1.0 - math.exp(-rate_per_lap * laps_remaining)
    return min(1.0, max(0.0, probability))


def _usable_rivals(rivals: Sequence[RivalState]) -> list[RivalState]:
    """Rivals whose gap is a real number.

    An unknown gap cannot be projected, so it is excluded rather than defaulted —
    the house rule that None means unknown. NaN and infinity are excluded on the
    same grounds and for a sharper reason: they are not merely unknown, they are
    contagious. One NaN gap turns every candidate's E, P10, P90 and score into
    ``nan`` while the payload still reports itself as scored.
    """
    return [
        rival for rival in rivals if rival.gap_s is not None and math.isfinite(float(rival.gap_s))
    ]


def _tyre_cost_s(config: ProjectionConfig, *, old_laps: float, fresh_laps: float) -> float:
    """Seconds this plan's tyre state costs, positive = worse for us.

    Sign convention differs from ``strategy_orchestrator._tyre_term`` because this
    module accumulates a LOSS while that one accumulates a gain. Same two mutually
    exclusive prices for the same physical thing: a measured cost on the laps spent
    on the old set, or the hardcoded fresh credit when the model gave no reading.

    THE ASYMMETRY WITH RIVALS IS REAL, AND IT IS A LIMITATION, NOT A CORRECTION
    ---------------------------------------------------------------------------
    This scorer works in gaps, and ``rival_time_deltas`` moves each rival by
    ``pace_delta_s * racing_laps``. That is not a reason to skip our own wear: a
    pace delta is a SNAPSHOT of the rival's relative pace at the current lap, and it
    says nothing about how the gap moves as our set degrades across the window.
    Charging our wear is exactly that extrapolation, and without it the projection
    prices a twenty-lap-old set identically to a fresh one for every lap it holds.

    What is genuinely missing is the mirror: **their** degradation is not modelled,
    because the single-driver boundary gives rivals timing-screen data only and there
    is no per-rival tyre state to run a TCN on. So a rival on older tyres than ours
    is credited with holding their current pace. That biases the comparison toward
    stopping, in the same direction and for a different reason than the term above,
    and it is a known limitation of the projection rather than something this
    function should compensate for by silently halving a measured cost.

    The legacy path has no rivals and never faced the question.
    """
    if config.deg_cost_s is None:
        return -fresh_laps * config.fresh_gain_s
    return old_laps * config.deg_cost_s


def driver_time_delta(
    plan: DriverPlan,
    pit_loss_s: np.ndarray,
    cliff_laps: np.ndarray,
    config: ProjectionConfig,
    stop_is_neutralised: np.ndarray | bool = False,
) -> np.ndarray:
    """Seconds we lose over the window under ``plan``, per draw.

    Three terms, all in seconds so they can be compared with a rival's:

    - the stop itself, discounted when it is taken under a neutralisation
      (the field is queued, so the same pit lane costs fewer seconds of race),
    - time lost running past the tyre cliff, over the laps this plan spends on
      the old set,
    - time gained on fresh rubber, over the racing laps that follow the stop,
    - time gained in clean air, over the laps this plan runs on after the car
      ahead has boxed,
    - the option value of those same laps: each one is another chance that a
      neutralisation turns up first and makes the stop cheap.

    A plan that does not stop pays no pit loss and gains nothing fresh; it just
    lives with its tyres. That asymmetry is the whole trade-off, and it is
    expressed here rather than asserted in a comment.

    The clean-air term is what separates an overcut from a late pit stop. It
    accrues over ``stop_offset_laps`` because those are exactly the laps run
    after the target vacated the road and before we take our own stop; every
    other plan has an offset of zero and the term vanishes for them without a
    branch. Caller's job to pass a gain of zero when we were never close enough
    to be in that wake, since the measurement only covers followers inside the
    dirty-air band.

    The waiting term is the other half of why a strategist delays a stop, and it
    rides on the same laps. It applies only to draws where the stop is NOT
    already neutralised: on those it has happened, and counting the chance of it
    happening as well would pay twice for one Safety Car.
    """
    draws = len(pit_loss_s)
    delta = np.zeros(draws, dtype=float)

    racing = float(config.racing_laps)
    if plan.stops_in_window:
        laps_before_stop = min(float(plan.stop_offset_laps), racing)
        laps_after_stop = max(0.0, racing - laps_before_stop)

        saving_if_it_comes = config.neutralisation_saving_s
        saving = np.where(stop_is_neutralised, saving_if_it_comes, 0.0)
        effective_loss = np.maximum(0.0, pit_loss_s - saving)

        worn_laps = np.maximum(0.0, laps_before_stop - cliff_laps)
        delta += effective_loss
        delta += worn_laps * config.cliff_loss_s
        delta += _tyre_cost_s(config, old_laps=laps_before_stop, fresh_laps=laps_after_stop)
        delta -= laps_before_stop * config.clean_air_gain_s

        waiting_pays = laps_before_stop * config.neutralisation_onset_rate * saving_if_it_comes
        delta -= np.where(stop_is_neutralised, 0.0, waiting_pays)
    else:
        worn_laps = np.maximum(0.0, racing - cliff_laps)
        delta += worn_laps * config.cliff_loss_s
        delta += _tyre_cost_s(config, old_laps=racing, fresh_laps=0.0)

    return delta


def rival_time_deltas(
    rivals: Sequence[RivalState],
    config: ProjectionConfig,
    draws: int,
) -> np.ndarray:
    """Seconds each rival loses over the window, shaped (draws, rivals).

    A rival costs themselves time two ways: by stopping (only when timing says
    they are in the pit lane right now — v1 trusts the fact, never a guess about
    their strategy) and by being slower than us lap after lap.

    Deterministic per rival today, so every draw carries the same column. It is
    still built at full width because the projection multiplies it against
    per-draw quantities, and because sampling rival stop durations is the
    natural next refinement.
    """
    usable = _usable_rivals(rivals)
    deltas = np.zeros((draws, len(usable)), dtype=float)

    for index, rival in enumerate(usable):
        loss = rival.stop_loss_s if rival.is_pitting else 0.0
        pace = rival.pace_delta_s * config.racing_laps
        deltas[:, index] = loss + pace

    return deltas


def _stop_residual_s(stop_loss_s: np.ndarray | float, config: ProjectionConfig) -> np.ndarray:
    """Seconds an outstanding stop still costs, after the option value of waiting.

    A future neutralisation might cover the stop cheaply, and that possibility is
    worth real seconds, so the raw pit loss is discounted by ``q_f * saving``.
    Floored at zero: a stop cannot end up gaining time. This is what turned the
    old flat Safety Car bonus into an option value and it is unchanged by the
    netting; only who it is applied TO changed.
    """
    discounted = np.asarray(stop_loss_s, dtype=float) - (
        config.future_neutralisation_prob * config.neutralisation_saving_s
    )
    return np.maximum(0.0, discounted)


def _deferral_tyre_liability_s(pit_loss_s: np.ndarray, config: ProjectionConfig) -> np.ndarray:
    """Seconds a NON-stopping plan's tyres cost between the window edge and the flag.

    The terminal netting already carries every KNOWN outstanding stop to a common
    horizon. What it did not carry is what the rubber costs while you defer, and for a
    car whose mandatory stop is already discharged that was the ONLY future cost it had
    -- so an elective stop's full pit loss stood against nothing at all.

    WHY THIS EXISTS, MEASURED RATHER THAN ARGUED
    ---------------------------------------------
    Over 694 real elective stops in 2023-24, the horizon a stop takes to repay itself
    from its own pace advantage has a median of **13 laps** (95% CI [12, 14]), and only
    **15.0%** repay inside five. So a five-lap window can price at most a seventh of the
    decision, and the remaining six sevenths were being compared against zero. The
    layer declined **69.9%** of elective stops against 26.7% of first stops; that gap
    is what this term addresses.

    The car's real choice is the cheaper of two futures, so the liability is their
    minimum rather than a horizon someone picked:

        stop later:   deg * k    + the residual that stop still costs
        run it out:   deg * R    + the cliff over the laps past it

    Both q_f-discounted the same way ``_stop_residual_s`` already discounts, because a
    neutralisation that turns up covers a deferred stop whichever branch wins.

    --- WHERE TO CHANGE IF THE SCOPING CHANGES ---
    Scoped by the CALLER to plans with no residual, and that scoping is a measured
    behaviour rather than a physical claim: the same unpriced wear exists for a car
    that still owes its stop, but that population already measures balanced, and
    charging it there moves first calls EARLIER -- the exact direction that cost five
    exact agreements when the wear term landed. Extending it is a separate decision and
    wants its own measurement, not symmetry.
    """
    if config.deg_cost_s is None:
        return np.zeros(len(pit_loss_s), dtype=float)

    remaining = float(max(0, config.laps_remaining - config.window_laps))
    if remaining <= 0.0:
        return np.zeros(len(pit_loss_s), dtype=float)

    # Stopping later still costs the stop, so the best later lap is as soon as the
    # window ends: every further lap adds wear without removing the pit loss. That
    # makes k = 0 from the window edge, and the branch collapses to the residual.
    stop_later = _stop_residual_s(pit_loss_s, config)

    # Or hold this set to the flag: wear on every lap, plus the cliff on the laps
    # past it. `cliff_laps` is a per-draw quantity the caller owns; the terminal
    # horizon uses the config's own window as the earliest the cliff can bite, which
    # keeps this function pure and its inputs already-measured.
    run_it_out = remaining * config.deg_cost_s + max(0.0, remaining - config.window_laps) * (
        config.cliff_loss_s
    )
    discounted_run = np.maximum(
        0.0, run_it_out - config.future_neutralisation_prob * config.neutralisation_saving_s
    )

    return np.minimum(stop_later, np.full(len(pit_loss_s), discounted_run, dtype=float))


def _terminal_gaps(
    usable: Sequence[RivalState],
    plan: DriverPlan,
    projected_gaps: np.ndarray,
    pit_loss_s: np.ndarray,
    config: ProjectionConfig,
) -> np.ndarray:
    """Window-end gaps carried forward to a race end where every KNOWN stop is served.

    The two-compound rule (Art. 30.5(m)) makes one stop mandatory, so a candidate
    that skips the window has not avoided the cost, only deferred it — and the
    same is true of a rival. Charging our deferral while exempting theirs is what
    made staying out look free and stopping look expensive: measured over real
    races, 73-84% of the cars counted as passing us still owed a stop of their
    own, and PIT_NOW won once in 110 laps.

    So both sides are carried to the same horizon::

        terminal_gap = projected_gap + their_residual - our_residual

    and the three cases the deleted rail was patching still fall out of the
    arithmetic, now by cancellation rather than by exemption:

    - already stopped (no obligation) -> our residual is zero, staying out costs
      nothing on this term;
    - leading a pack that all still owe a stop -> their residuals and ours are
      the same size and cancel, so holding the lead is free. The cancellation is
      EXACT only when the two losses match; in production ours is sampled and
      theirs is a per-rival prior, so a tight pack costs a fraction of a position
      on the draws where the sampled loss runs longer than the prior. That is a
      real difference between two cars' pit stops, not a modelling artefact, but
      "free" is the round-number version of it;
    - the race ending behind the Safety Car -> the config's racing laps go to
      zero, so nothing is gained by stopping in the first place.

    An unknown obligation contributes NOTHING in either direction, on the
    module's standing rule that a claim needs a fact. That is deliberately
    asymmetric with charging it: an unsettled obligation treated as a certainty
    invents twenty-odd seconds of somebody's race.

    **And that rule binds BOTH sides of the subtraction.** If we do not know
    whether WE still owe a stop, then crediting a rival's fall-back is not a
    claim about them, it is a claim that we will not fall back with them — a
    fact we do not have. The first version applied their residual anyway and
    handed a candidate that stays out a full terminal place on that bet: it
    scored an unknown obligation exactly like a known-discharged one. A plan
    that STOPS inside the window is settled either way, so the suppression is
    narrow, covering only the deferring case.

    A rival already serving their stop is excluded too: ``rival_time_deltas``
    has charged them inside the window already, and charging the same stop twice
    would push them a full pit cycle further back than they will ever be.
    """
    if not plan.stops_in_window and config.mandatory_stop_pending is None:
        return projected_gaps

    if plan.stops_in_window:
        our_residual = np.zeros(len(pit_loss_s), dtype=float)
    elif config.mandatory_stop_pending is True:
        our_residual = _stop_residual_s(pit_loss_s, config)
    elif config.mandatory_stop_pending is False:
        # The obligation is discharged, so there is no stop residual to carry -- but
        # deferring still costs rubber, and until this term existed an elective stop's
        # full pit loss stood against exactly zero. See _deferral_tyre_liability_s.
        our_residual = _deferral_tyre_liability_s(pit_loss_s, config)
    else:
        # `None` means the compound history could not settle it. The module's rule is
        # that a claim needs a fact, so an unknown obligation buys no correction in
        # either direction, exactly as it did before this term.
        our_residual = np.zeros(len(pit_loss_s), dtype=float)

    their_residual = np.array(
        [
            _stop_residual_s(rival.stop_loss_s, config)
            if rival.stop_pending is True and not rival.is_pitting
            else 0.0
            for rival in usable
        ],
        dtype=float,
    )

    return projected_gaps + their_residual[None, :] - our_residual[:, None]


def project_positions(
    rivals: Sequence[RivalState],
    plan: DriverPlan,
    config: ProjectionConfig,
    pit_loss_s: np.ndarray,
    cliff_laps: np.ndarray,
    stop_is_neutralised: np.ndarray | bool = False,
) -> ProjectionResult:
    """Project our end-of-window position among the actual cars, per draw.

    Every gap moves by the difference between what the rival loses and what we
    lose; a gap that crosses zero is a car changing sides. Counting the cars
    projected ahead gives the position directly, so "rejoining into traffic"
    needs no special case: every rival within our pit loss behind us is a
    position lost, counted by name.

    Returns positions, the seconds of margin to the nearest car behind (a
    tie-break with real strategic meaning, since two seconds of clear air beats
    a tenth at the same position), and the terminal liability.
    """
    usable = _usable_rivals(rivals)
    draws = len(pit_loss_s)

    our_delta = driver_time_delta(plan, pit_loss_s, cliff_laps, config, stop_is_neutralised)

    if not usable:
        # Nobody to be passed by, at either horizon.
        ones = np.ones(draws, dtype=float)
        return ProjectionResult(
            positions=ones,
            margins_s=np.zeros(draws, dtype=float),
            terminal_positions=ones,
            rivals_used=0,
        )

    current_gaps = np.asarray([rival.gap_s for rival in usable], dtype=float)
    their_deltas = rival_time_deltas(rivals, config, draws)

    projected_gaps = current_gaps[None, :] + their_deltas - our_delta[:, None]

    ahead = projected_gaps < 0
    positions = 1.0 + ahead.sum(axis=1)

    behind_gaps = np.where(ahead, np.inf, projected_gaps)
    nearest_behind = behind_gaps.min(axis=1)
    margins = np.clip(np.where(np.isinf(nearest_behind), 0.0, nearest_behind), 0.0, MARGIN_CLIP_S)

    terminal_gaps = _terminal_gaps(usable, plan, projected_gaps, pit_loss_s, config)

    return ProjectionResult(
        positions=positions,
        margins_s=margins,
        terminal_positions=1.0 + (terminal_gaps < 0).sum(axis=1),
        rivals_used=len(usable),
    )


def payoff(result: ProjectionResult, current_position: int, config: ProjectionConfig) -> np.ndarray:
    """Per-draw payoff in positions gained at the terminal horizon, margin-adjusted.

    Positions are the currency, which is the point of the redesign. The margin
    term is deliberately small (a tenth of a position per second, capped): it
    breaks ties between candidates that land on the same car count and smooths
    the quantile steps that an integer-valued score would otherwise have, but it
    can never outvote an actual position.

    Scored on ``terminal_positions``, not on the window-end ``positions``: both
    sides of a comparison must be at the same horizon or the arithmetic favours
    whichever one had its future cost left off. The margin stays on the WINDOW-end
    gaps on purpose — it is a tie-break about track position now, not a claim
    about the end of the race.
    """
    gained = float(current_position) - result.terminal_positions
    margin_bonus = config.margin_weight * result.margins_s
    return gained + margin_bonus


def undercut_targets(rivals: Sequence[RivalState], config: ProjectionConfig) -> list[str]:
    """Live rivals ahead of us and inside the measured undercut band.

    The band is measured, not assumed: across 716 real attempts, success falls
    from 86% under a second to under 1% beyond ten, and the committed band is the
    P90 of the gaps at which one actually worked. It replaces the old "within
    five positions" rule, which reasoned in a unit the pit lane does not use.

    Liveness is presence in this list — a car that crashed is simply not in it —
    never a DNF classification or a staleness threshold, because a car that
    finished can legitimately lag twenty laps behind in the data.

    A car already in the pit lane is NOT a target. You cannot undercut someone
    who is serving their stop as you decide: the whole move is to reach the pit
    lane before they do. Offering them as a target credited the undercut with a
    place it had no way to take.
    """
    band = config.undercut_band_s
    return [
        rival.driver
        for rival in _usable_rivals(rivals)
        if rival.is_ahead
        and not rival.is_pitting
        and rival.gap_ahead_s is not None
        and rival.gap_ahead_s <= band
    ]


def overcut_targets(rivals: Sequence[RivalState]) -> list[str]:
    """Live rivals ahead of us who are in the pit lane right now.

    An overcut needs someone to overcut: the payoff is holding track position
    while they serve their stop. v1 keys off the timing fact (``is_pitting``)
    and never off a guess about who might stop soon, which would be rival
    strategy modelling by another name. Surfaces whose rivals list carries no
    such flag get the measured stop-hazard prior instead, wired separately.
    """
    return [rival.driver for rival in _usable_rivals(rivals) if rival.is_ahead and rival.is_pitting]


@dataclass(frozen=True)
class TargetRanking:
    """One rival scored as a post-pit-cycle target.

    Attributes:
        driver:            FIA code.
        projected_gap_s:   Signed seconds to them once both pit cycles are done.
        current_gap_s:     Signed seconds now, for comparison.
        positions_apart:   How far apart the timing screen says we are.
    """

    driver: str
    projected_gap_s: float
    current_gap_s: float
    positions_apart: int


def rank_targets(
    rivals: Sequence[RivalState],
    config: ProjectionConfig,
    our_pit_loss_s: float,
) -> list[TargetRanking]:
    """Rank rivals by how close they will be once both pit cycles have played out.

    A strategist does not attack the car currently ahead, they attack the car
    they will be racing after the stops. Víctor's own example: leading a race
    while the car behind pits early and emerges tenth, the right target may be a
    car eight places down the screen, because after our own stop they come out
    in front of us.

    Deterministic and run once (no sampling): it selects who to attack, and the
    Monte Carlo then scores the attacking. Both go through the same
    ``_usable_rivals`` filter, so the selector and the scorer can never disagree
    about WHICH rivals are in play (a rival with an unknown gap cannot silently
    show up in one and not the other). That is the only thing shared: this
    function does NOT use the same stop-obligation predicate as the scorer's
    ``rival_time_deltas`` / ``_terminal_gaps``, and it should not be made to.
    See "THREE HORIZONS FOR ONE FACT" at the top of this module for why
    ``rank_targets`` charges ``stop_pending`` alone rather than the
    ``is_pitting`` guard the other two use (#742).
    """
    ranked: list[TargetRanking] = []

    for offset, rival in enumerate(_usable_rivals(rivals), start=1):
        # Charge a rival a pit loss only when we KNOW they still owe the stop.
        # Unknown is not "probably yes": an unsettled obligation treated as a
        # certainty invents 20-odd seconds of someone else's race.
        their_loss = rival.stop_loss_s if rival.stop_pending is True else 0.0
        projected = (
            rival.gap_s + their_loss - our_pit_loss_s + rival.pace_delta_s * config.racing_laps
        )
        ranked.append(
            TargetRanking(
                driver=rival.driver,
                projected_gap_s=round(float(projected), 3),
                current_gap_s=round(float(rival.gap_s), 3),
                positions_apart=offset,
            )
        )

    ranked.sort(key=lambda target: abs(target.projected_gap_s))
    return ranked
