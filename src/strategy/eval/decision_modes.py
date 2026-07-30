"""Eval report for the DECISION, not the rejoin: does the stack pick the right lap to stop?

``projection.py`` answers "given that the driver stopped on lap L, where does he
rejoin" — 86.5% within one place over 1810 real stops. That is pit-cycle
geometry, and it is the number this project has been quoting. It is not the
claim the system actually makes. The claim is *when to stop*, and until this
report existed nobody had measured it.

The gap survived because the projection number is good. A strong metric is
effective camouflage for the adjacent claim nobody tested.

WHAT THIS MEASURES, EXACTLY
---------------------------
For every real green-flag stop in the sampled races, the stack is driven over a
window of laps either side of the real stop lap and asked, lap by lap, what it
would do. The lap it would have chosen is the first **transition** — the lap
where it stops saying "stay out" and starts saying "box" — and the signed
distance to the real lap is the error. Signed, not absolute, for the reason
``GroundTruth.mean_signed_error`` exists: stopping consistently early or late is
a fixable bug, and a magnitude hides it.

A transition and not simply the first pit lap, because that is what #752
retired. A stack whose earliest pit ask has no evaluated non-pit lap before it
has no decision inside the window, and reporting the window's left edge as its
choice made the error a property of the window: widening it from 5 laps to 10
moved the entire mass from one boundary to the other and left zero at the old
one. Those stops are now ``no_boundary_in_window``: looked at, deliberately
unscored, because the honest statement is that the call came earlier than we
asked, not that it came on the lap we happened to start asking.

That bucket means *no locatable decision*, and nothing more. It is NOT a
description of what the stack did. On the measured 2025 Monza sample all four
occupants were committed when the window opened and then WITHDREW inside it, one
of them flipping to stay-out on the exact lap the team really stopped. See
``_render_table`` for the three distinct shapes that land there.

The stack runs on ``profile="no-llm"``, so the action comes from the Monte
Carlo layer and the guard rails **deterministically**. This is a deliberate
scope choice, not a shortcut: the LLM synthesis is stochastic and has been
shown to be steerable by the prompt, so measuring it would measure the sampler
as much as the strategy. What is under test here is the deterministic decision
layer.

WHAT THIS DOES NOT MEASURE (do not let a reader assume it does)
---------------------------------------------------------------
It is **not** counterfactual. It never claims "stopping on lap 28 instead of 32
would have finished higher" — answering that needs a full-field propagator, and
``RaceReplayEngine`` replays real laps by design while rivals get timing-only
data. Agreement with what the team actually did is also not proof of
correctness: the team can be wrong. This measures whether the decision layer
lands in the same place as a professional pit wall, which is evidence, not a
verdict.

WHY THERE IS NO CROSS-TIER MONOTONICITY GUARD
----------------------------------------------
The intended guard was "a freer tier must not score better than a constrained
one, or two errors are cancelling". It is not buildable across these two tiers:
projection error is in POSITIONS and decision error is in LAPS, and converting
the second into the first requires the counterfactual ruled out above. What is
buildable is the other half of the same idea — that the not-scored buckets must
not quietly absorb the failures — and that is ``coverage_verdict``.

WHERE TO CHANGE IF THINGS MOVE:
- ``src/strategy/inference/guard_rails.py`` owns the rails and the pit-action
  set. ``guard_rail_block`` CALLS the rail rather than restating its thresholds,
  because the first version retyped one boundary and got it wrong by one lap.
- ``SAMPLED_RACES`` is the subset and the reason it exists; see its comment
  before widening it.
- ``lap_inputs`` decides which laps are answerable and is deliberately free of
  agent imports, so the two bugs that lived there stay under test on a runner
  with no model weights.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from src.strategy.eval.report import build_header, write_report

# From the leaf module, never from ``no_llm``: that one imports the agent stack and
# loads model weights at import time, which would make this report — and every other
# ``f1-eval`` subcommand next to it — impossible to even import without ``data/models/``.
from src.strategy.inference.guard_rails import (
    _MIN_STINT_LAPS,
    _NO_PIT_BEFORE_LAP,
    _PIT_ACTIONS,
    apply_guard_rails,
)

# Laps either side of the real stop that the stack is asked about. Five matches
# the Monte Carlo decision window, so the question posed to the model is the one
# it was built to answer rather than a horizon it never considers.
DECISION_WINDOW_LAPS = 5

# A full sweep is not affordable. Measured on this machine: 0.51 s per lap through
# the no-llm stack, 28.8 s for one (race, driver) pair, and the real-stop sample
# spans roughly 1440 pairs — about 11.5 hours for one report. So the tier runs on a
# named, stratified subset instead, chosen for circuit archetype rather than
# convenience: two conventional high-degradation circuits, two street circuits where
# track position dominates, one low-downforce low-stop circuit and one fast circuit
# with variable weather. Widening this is a runtime decision, not a correctness one —
# but the report must keep saying which races it used.
SAMPLED_RACES: tuple[tuple[int, str], ...] = (
    (2023, "Barcelona"),  # conventional, high degradation
    (2023, "Monaco"),  # street, track position is everything
    (2024, "Silverstone"),  # fast, weather-variable
    (2024, "Marina_Bay"),  # street, high degradation
    (2025, "Lusail"),  # high degradation, stint-limited history
    (2025, "Monza"),  # low downforce, fewest stops
)

# Share of eligible stops that must actually be scored before the headline
# agreement figure means anything. Below this the not-scored buckets are large
# enough to be hiding the answer, which is exactly the failure mode a fourth
# "unavailable" gate state invites if nobody polices it.
MIN_SCORED_SHARE = 0.60

# The `alpha` the Monte Carlo scores with: `score = alpha*E[S] + (1-alpha)*P10[S]`, taken
# from `RaceState.risk_tolerance`. 0.5 is what every surface passes today, so it is what
# the tier measures by default. It is exposed rather than buried because a decision layer
# that declines 65% of real stops might simply be reading a cautious default as policy,
# and that is answerable by sweeping it rather than by arguing about it (#715).
DEFAULT_RISK_TOLERANCE = 0.5

# Buckets that mean "the rails made agreement impossible", as opposed to
# "the model declined to call it", which is a result and not an exclusion.
_GUARD_RAIL_BUCKETS = frozenset({"opening_laps", "closing_laps", "min_stint"})


@dataclass(frozen=True)
class StopVerdict:
    """One real stop, and what the decision layer would have done about it.

    ``offset_laps`` is signed (chosen minus actual) and is ``None`` whenever the
    stop was not scored, in which case ``bucket`` says why. None means "not
    measured" and never zero — a zero offset is perfect agreement, which is the
    opposite conclusion.
    """

    year: int
    race: str
    driver: str
    actual_lap: int
    chosen_lap: int | None
    offset_laps: int | None
    bucket: str


@dataclass(frozen=True)
class DecisionAgreement:
    """Aggregate agreement between the decision layer and the real pit wall."""

    offsets: np.ndarray
    guard_railed: int
    no_call: int
    races: int
    no_data: int = 0
    no_boundary: int = 0

    @property
    def sample_size(self) -> int:
        return int(self.offsets.size)

    @property
    def eligible(self) -> int:
        """Every stop the tier looked at, scored or not.

        ``no_boundary`` joined this sum in #752 and it has to: those stops WERE
        looked at. Leaving them out would shrink the denominator and inflate the
        scored share by exactly the stops the old code used to score wrongly.
        """
        return self.sample_size + self.guard_railed + self.no_call + self.no_data + self.no_boundary

    @property
    def exact(self) -> float:
        return float((self.offsets == 0).mean()) if self.sample_size else 0.0

    @property
    def within_one(self) -> float:
        return float((np.abs(self.offsets) <= 1).mean()) if self.sample_size else 0.0

    @property
    def within_two(self) -> float:
        return float((np.abs(self.offsets) <= 2).mean()) if self.sample_size else 0.0

    @property
    def mean_signed_error(self) -> float:
        return float(self.offsets.mean()) if self.sample_size else 0.0

    @property
    def mean_absolute_error(self) -> float:
        return float(np.abs(self.offsets).mean()) if self.sample_size else 0.0

    @property
    def scored_share(self) -> float:
        return self.sample_size / self.eligible if self.eligible else 0.0


# Which rail fired, keyed on a stable fragment of the reason it returns. Matching
# the message is not elegant, but re-deriving the boundaries is what produced an
# off-by-one against the rail's own ``remaining <= 3`` the first time this was
# written. ``test_every_bucket_is_reachable_through_the_real_rail`` fails loudly if
# a message is ever reworded, so a silent mis-bucket is not possible.
_RAIL_BUCKETS: tuple[tuple[str, str], ...] = (
    ("pit window not open", "opening_laps"),
    ("too late to pit", "closing_laps"),
    ("minimum stint", "min_stint"),
)


def guard_rail_block(
    actual_lap: int, total_laps: int, compound: str | None, tyre_life: int | None
) -> str | None:
    """Name the guard rail that makes agreement with this stop impossible, else None.

    Asks ``apply_guard_rails`` itself whether a stop on this lap would have been
    overridden, rather than restating its thresholds. A real stop inside a rail can
    never be agreed with no matter how good the strategy is, so folding it into the
    headline would measure the rail instead of the decision.

    When tyre life is unknown the minimum-stint rail simply cannot be evaluated, so
    the probe passes a life that satisfies every minimum and only the lap-based
    rails apply. That is a stated assumption, not a sentinel: it never becomes a
    value the caller can mistake for a measurement.
    """
    probe_life = max(_MIN_STINT_LAPS.values()) if tyre_life is None else tyre_life
    action, reason = apply_guard_rails(
        "PIT_NOW", actual_lap, total_laps, compound or "", probe_life
    )
    if action == "PIT_NOW" or reason is None:
        return None

    for marker, bucket in _RAIL_BUCKETS:
        if marker in reason:
            return bucket
    raise ValueError(f"unmapped guard-rail reason: {reason!r}")


def coverage_verdict(agreement: DecisionAgreement) -> str:
    """``ok`` when enough stops were actually scored, ``masked`` when they were not.

    The adapted form of the compensation guard. A tier that quietly routes most
    of its sample into "could not evaluate" reports a headline drawn from
    whatever is left, and the shape of what is left is not random: guard-railed
    and no-call stops are systematically the awkward ones. Surfacing this as a
    status is the whole point — an "unavailable" bucket that nobody polices is
    just a failure with better manners.
    """
    if agreement.eligible == 0:
        return "unavailable"
    return "ok" if agreement.scored_share >= MIN_SCORED_SHARE else "masked"


def _team_of(laps, driver: str) -> str | None:
    """The Team string this driver raced under, exactly as the parquet spells it."""
    rows = laps[laps["Driver"] == driver]["Team"].dropna()
    return str(rows.iloc[0]) if len(rows) else None


def lap_inputs(state: dict[str, Any]) -> dict[str, Any] | None:
    """The primitive fields a ``RaceState`` needs from a lap state, or None to skip.

    Pure and free of any agent import on purpose: this is the part that decides
    which laps are answerable at all, it is where two real bugs lived, and keeping
    it separate is what lets those bugs stay under test on a runner with no model
    weights on disk.

    Skips, and why each is a skip rather than a default:
    - no ``lap_number``: a retired car keeps yielding lap states with an EMPTY
      driver dict for the remainder of the race. Presence is the signal, never a
      lap-number threshold — a car that finished can be missing laps too, and the
      two ranges overlap.
    - no ``position``: the state manager returns None deliberately, because a
      sentinel position has already collided with a real one in this codebase. A
      lap with no position is not a lap the stack can be asked about.

    ``tyre_life`` is read the long way for the same family of reason: ``or 10``
    would turn a legitimate fresh tyre (0) into a ten-lap-old one and quietly move
    what the tyre agent answers.
    """
    car = state.get("driver") or {}
    if "lap_number" not in car or car.get("position") is None:
        return None

    tyre_life = car.get("tyre_life")
    weather = state.get("weather") or {}
    return {
        "lap": int(car["lap_number"]),
        "total_laps": int(state["session_meta"]["total_laps"]),
        "position": int(car["position"]),
        "compound": car.get("compound") or "MEDIUM",
        "tyre_life": 10 if tyre_life is None else int(tyre_life),
        "gap_ahead_s": car.get("gap_ahead_s") or 2.0,
        "air_temp": weather.get("air_temp") or 25.0,
        "track_temp": weather.get("track_temp") or 35.0,
    }


def _decisions_in_window(
    engine,
    laps_df,
    driver: str,
    low: int,
    high: int,
    risk_tolerance: float = DEFAULT_RISK_TOLERANCE,
) -> dict[int, str]:
    """Action the deterministic stack emits on each lap of ``[low, high]``.

    Only the laps inside the window are pushed through ``run_lap``; the replay
    generator itself is cheap and the agent stack is not, so skipping the rest of
    the race is where the runtime budget comes from.
    """
    from src.agents.strategy_orchestrator import RaceState
    import src.strategy.inference.engine as inference_engine

    actions: dict[int, str] = {}
    for state in engine.replay():
        inputs = lap_inputs(state)
        if inputs is None:
            continue
        if inputs["lap"] < low:
            continue
        if inputs["lap"] > high:
            break

        race_state = RaceState(
            driver=driver, pace_delta_s=0.0, risk_tolerance=risk_tolerance, **inputs
        )
        recommendation, _outputs, _timings = inference_engine.run_lap(
            race_state, laps_df, state, profile="no-llm", return_agent_outputs=True
        )
        actions[inputs["lap"]] = recommendation.action
    return actions


def _asks_to_stop(actions: dict[int, str], low: int, high: int) -> bool:
    """Whether the stack asked to stop anywhere in the window, transition or not.

    Only used to tell "it never wanted to stop" from "it already wanted to stop
    before we started asking". Those are opposite findings and the old bucketing
    could not distinguish them.
    """
    return any(actions.get(lap) in _PIT_ACTIONS for lap in range(low, high + 1))


def _pit_decision_lap(actions: dict[int, str], low: int, high: int) -> int | None:
    """The lap the stack DECIDED to stop on — the first non-pit → pit transition.

    WHY A TRANSITION AND NOT THE FIRST PIT LAP (#752)
    -------------------------------------------------
    This used to return the earliest lap in the window carrying a pit action. That
    reports the window's left edge for any stack that would pit on *every* lap of
    it, which is not a timing estimate — it is ``window_low`` wearing one.

    Measured before the change, 2025 Monza + Marina_Bay, moving only the eval's own
    window width:

        window = 5     12 offsets at -5   (the edge)   mean signed -3.08
        window = 10     0 offsets at -5, 10 at -10     mean signed -5.04

    The mass at -5 does not survive widening; it goes to **zero** and reappears at
    the new boundary. So the published -2.23 / -3.08 / -5.04 were three readings of
    the window, not three readings of the model.

    A transition is the smallest thing that is actually a decision: the stack said
    "stay out" and then said "box". Where no transition exists, this returns None
    and the caller records ``no_boundary_in_window`` rather than inventing a lap —
    which is the honest report for a stack that was already committed when the
    window opened, and the one the old code could not make.

    Requires ``lap - 1`` to have been EVALUATED, not merely absent: an unevaluated
    predecessor cannot witness a transition, so it is not counted as one. The
    caller widens the replay span by a lap so ``low`` itself can be judged.

    A RAILED PREDECESSOR IS NOT A WITNESS EITHER
    ---------------------------------------------
    Actions are recorded AFTER the guard rails (``no_llm.py``), and the opening
    rail forces STAY_OUT on every lap below ``_NO_PIT_BEFORE_LAP`` outside a
    neutralisation. So a stack that wants to box from lap 1 is recorded as
    ``STAY_OUT, STAY_OUT, STAY_OUT, STAY_OUT, PIT`` and the transition lands on
    ``_NO_PIT_BEFORE_LAP`` for every such stop — a constant of the rails wearing a
    timing estimate, which is the shape #752 retired, relocated from ``window_low``
    to the rail boundary. Those stops go to ``no_boundary_in_window`` instead.

    Rejecting on the lap number alone over-rejects under a Safety Car, where the
    rail is suspended and lap 5 may be a real transition. That is the conservative
    direction and the same one this whole metric took: refusing to score is honest,
    inventing a lap is not.
    """
    for lap in range(low, high + 1):
        if actions.get(lap) not in _PIT_ACTIONS:
            continue
        previous = actions.get(lap - 1)
        if previous is None or previous in _PIT_ACTIONS:
            continue
        if lap <= _NO_PIT_BEFORE_LAP:
            continue
        return lap
    return None


def _replay_span(stop_laps: list[int], total_laps: int) -> tuple[int, int]:
    """The laps to replay for one (race, driver), covering the union of the windows.

    One pass per driver rather than one per stop: two stops twenty laps apart still
    cost one replay, which is where the runtime budget for a stratified subset comes
    from.

    The span opens **one lap before** the earliest scoring window. A transition needs
    its predecessor to have been evaluated, so without that extra lap the first lap of
    every window would be permanently unjudgeable and the edge report would return
    through the back door: on 2025 Monza it is the difference between HAD's stop
    scoring at lap 27 and landing in ``no_boundary_in_window`` (#752).

    It is a separate function because that one lap was previously asserted by a comment
    and nothing else. Reverting it left the whole suite green while moving the published
    error by half its value, so it now has a home a test can reach.
    """
    low = max(1, min(stop_laps) - DECISION_WINDOW_LAPS - 1)
    high = min(total_laps, max(stop_laps) + DECISION_WINDOW_LAPS)
    return low, high


def _stop_context(laps, driver: str, lap: int) -> tuple[str | None, int | None]:
    """Compound and tyre life on the lap the car actually pitted."""
    row = laps[(laps["Driver"] == driver) & (laps["LapNumber"] == lap)]
    if not len(row):
        return None, None
    compound = row["Compound"].iloc[0]
    tyre_life = row["TyreLife"].iloc[0]
    return (
        None if compound is None or _is_missing(compound) else str(compound),
        None if _is_missing(tyre_life) else int(tyre_life),
    )


def _is_missing(value: Any) -> bool:
    """True for NaN/NaT/None, without assuming the value is numeric."""
    try:
        return bool(np.isnan(value))
    except (TypeError, ValueError):
        return value is None


def measure_decision_agreement(
    races: tuple[tuple[int, str], ...] = SAMPLED_RACES,
    risk_tolerance: float = DEFAULT_RISK_TOLERANCE,
) -> tuple[DecisionAgreement, list[StopVerdict]]:
    """Drive the deterministic stack over every real green-flag stop in ``races``.

    Args:
        risk_tolerance: the Monte Carlo's ``alpha``. Exposed so the decline rate can
            be swept against it instead of argued about: if 65% of real stops go
            uncalled at 0.5 and the number barely moves at 0.1 or 0.9, the default is
            not what is deciding, and the search moves to the scorer itself (#715).

    Raises FileNotFoundError when ``data/raw/`` is absent rather than returning an
    empty sample, because a tier that reports perfect agreement over zero stops is
    worse than one that reports nothing.
    """
    import pandas as pd

    from src.f1_strat_manager.laps_augment import augment_featured_laps
    from src.simulation.replay_engine import RaceReplayEngine

    # The sample definition comes from projection.py rather than being restated,
    # so both tiers grade the same stops. That identity is the only reason the two
    # sets of numbers can sit in the same sentence.
    from src.strategy.eval.projection import (
        _neutralised_laps,
        _raw_data_root,
        green_flag_stops,
    )

    raw = _raw_data_root()
    if raw is None:
        raise FileNotFoundError(
            "data/raw/ is not present; the decision tier needs the raw laps from the "
            "Hugging Face dataset (the featured parquet drops the pit laps it is keyed on)"
        )

    featured_by_year: dict[int, Any] = {}
    verdicts: list[StopVerdict] = []
    races_measured = 0

    for year, race in races:
        race_dir = raw / str(year) / race
        laps_path = race_dir / "laps.parquet"
        if not laps_path.exists():
            continue

        laps = pd.read_parquet(laps_path)
        if "PitInTime" not in laps.columns:
            continue

        if year not in featured_by_year:
            featured_path = f"data/processed/laps_featured_{year}.parquet"
            featured_by_year[year] = augment_featured_laps(pd.read_parquet(featured_path), year)
        featured = featured_by_year[year]

        neutralised = _neutralised_laps(laps)
        total_laps = int(laps["LapNumber"].max())
        races_measured += 1

        for driver, stop_laps in green_flag_stops(laps, neutralised).items():
            team = _team_of(laps, driver)
            if team is None:
                continue

            low, high = _replay_span(stop_laps, total_laps)
            engine = RaceReplayEngine(str(race_dir), driver, team, interval_seconds=0)
            actions = _decisions_in_window(engine, featured, driver, low, high, risk_tolerance)

            for stop_lap in stop_laps:
                compound, tyre_life = _stop_context(laps, driver, stop_lap)
                blocked = guard_rail_block(stop_lap, total_laps, compound, tyre_life)
                if blocked is not None:
                    verdicts.append(StopVerdict(year, race, driver, stop_lap, None, None, blocked))
                    continue

                window_low = max(1, stop_lap - DECISION_WINDOW_LAPS)
                window_high = min(total_laps, stop_lap + DECISION_WINDOW_LAPS)

                # "The stack never evaluated this window" and "the stack looked and
                # declined to stop" are different findings, and collapsing them would
                # charge a retirement to the model as a missed call.
                if not any(lap in actions for lap in range(window_low, window_high + 1)):
                    verdicts.append(
                        StopVerdict(year, race, driver, stop_lap, None, None, "no_data")
                    )
                    continue

                chosen = _pit_decision_lap(actions, window_low, window_high)
                if chosen is None:
                    # Two opposite findings the old bucketing merged into one. "It
                    # never asked to stop" is the model declining; "it asked, but no
                    # transition can be located" is the model already committed when
                    # we started looking, and reporting the window edge as its choice
                    # is what #752 retired. The second bucket says only that - it does
                    # NOT say the stack asked on every lap, which measured false on
                    # 4 of 4 real occupants (they withdrew mid-window).
                    unscored = (
                        "no_boundary_in_window"
                        if _asks_to_stop(actions, window_low, window_high)
                        else "no_call_in_window"
                    )
                    verdicts.append(StopVerdict(year, race, driver, stop_lap, None, None, unscored))
                    continue

                verdicts.append(
                    StopVerdict(year, race, driver, stop_lap, chosen, chosen - stop_lap, "scored")
                )

    offsets = np.array([v.offset_laps for v in verdicts if v.offset_laps is not None], dtype=int)
    agreement = DecisionAgreement(
        offsets=offsets,
        guard_railed=sum(1 for v in verdicts if v.bucket in _GUARD_RAIL_BUCKETS),
        no_call=sum(1 for v in verdicts if v.bucket == "no_call_in_window"),
        races=races_measured,
        no_data=sum(1 for v in verdicts if v.bucket == "no_data"),
        no_boundary=sum(1 for v in verdicts if v.bucket == "no_boundary_in_window"),
    )
    return agreement, verdicts


def _bucket_counts(verdicts: list[StopVerdict]) -> dict[str, int]:
    """How many stops landed in each bucket, so the exclusions stay countable."""
    counts: dict[str, int] = {}
    for verdict in verdicts:
        counts[verdict.bucket] = counts.get(verdict.bucket, 0) + 1
    return dict(sorted(counts.items()))


def _render_table(
    agreement: DecisionAgreement | None, verdicts: list[StopVerdict], status: str
) -> str:
    """Markdown body: the headline agreement, the buckets, and the caveats."""
    if agreement is None:
        return (
            "Not measured: `data/raw/` is absent, so there are no real stops to "
            "compare against. Pull the Hugging Face dataset and re-run.\n"
        )

    races = ", ".join(f"{year} {race}" for year, race in SAMPLED_RACES)
    lines = [
        "| Metric | Value | Meaning |",
        "| --- | --- | --- |",
        f"| Stops scored | {agreement.sample_size} of {agreement.eligible} "
        f"({agreement.scored_share:.1%}) | real green-flag stops the tier could grade |",
        f"| Exact lap | {agreement.exact:.1%} | chose the lap the team chose |",
        f"| Within 1 lap | {agreement.within_one:.1%} | same call, one lap either side |",
        f"| Within 2 laps | {agreement.within_two:.1%} | same strategic window |",
        f"| Mean signed error | {agreement.mean_signed_error:+.2f} laps | "
        "negative = stops earlier than the team |",
        f"| Mean absolute error | {agreement.mean_absolute_error:.2f} laps | magnitude |",
        f"| Coverage verdict | **{status}** | `masked` when under "
        f"{MIN_SCORED_SHARE:.0%} of eligible stops were scored |",
        "",
        "### Buckets",
        "",
        "| Bucket | Stops |",
        "| --- | --- |",
    ]
    lines += [f"| `{name}` | {count} |" for name, count in _bucket_counts(verdicts).items()]
    lines += [
        "",
        "`opening_laps` / `closing_laps` / `min_stint` are stops the guard rails make",
        "impossible to agree with, so they are excluded from the headline rather than",
        "counted as misses. `no_data` is a car that had already retired, so the stack",
        "never evaluated the window at all. `no_call_in_window` is different and is the",
        "number to watch: the stack looked and declined to stop anywhere near the real",
        "lap. Charging a retirement to the model as a missed call would flatter neither",
        "side honestly, so the two are never merged.",
        "",
        "`no_boundary_in_window` is the third case and it is the one this tier used to",
        "get wrong (#752). It means only this: the stack asked to stop somewhere in the",
        "window, but no STAY_OUT -> PIT transition could be located inside it. Read it",
        "as **no locatable decision**, never as a description of what the stack did.",
        "Three different shapes land here and they are not the same finding:",
        "",
        "- already asking when the window opened, and still asking on every lap;",
        "- already asking when the window opened, then **withdrawing** later - on the",
        "  measured 2025 Monza sample this was 4 of 4 occupants, one of them flipping to",
        "  STAY_OUT on the exact lap the team really stopped;",
        "- a lap inside the window that was never evaluated, so the only pit ask has no",
        "  witness for its predecessor.",
        "",
        "What they share is that the earliest pit ask has no evaluated non-pit lap before",
        "it, so any lap reported would be the window's left edge rather than the model's",
        "choice - which is why the retired `mean_signed_error` moved with the window width",
        "instead of with the model. A stop here is counted as looked-at and left unscored.",
        "The same applies at the opening guard rail: a transition on lap",
        f"{_NO_PIT_BEFORE_LAP} or earlier is the rail releasing, not the model deciding,",
        "so it is bucketed here too.",
        "",
        "### Scope",
        "",
        f"- Sampled races ({agreement.races} measured): {races}.",
        "- A full sweep of the real-stop sample is roughly 11.5 h of wall clock at",
        "  0.51 s per lap through the stack, so this is a stratified subset by circuit",
        "  archetype and **not** full coverage. Read every figure above as conditional",
        "  on these races.",
        '- Decisions come from `profile="no-llm"`: the deterministic Monte Carlo layer',
        "  plus the guard rails, never the LLM synthesis.",
        "- Agreement with the real pit wall is evidence, not correctness. The team can",
        "  be wrong, and this tier cannot tell when it was.",
        "",
    ]
    return "\n".join(lines)


def build_decision_modes_report() -> dict[str, Any]:
    """Write ``documents/eval_reports/decision_modes.{md,json}`` and return the payload."""
    try:
        agreement, verdicts = measure_decision_agreement()
    except FileNotFoundError:
        agreement, verdicts = None, []

    status = coverage_verdict(agreement) if agreement is not None else "unavailable"
    header = build_header(dataset="data/raw laps, stratified 6-race subset (RAW, not featured)")
    md_path, json_path = write_report(
        "decision_modes",
        header,
        _render_table(agreement, verdicts, status),
        {
            "status": status,
            "window_laps": DECISION_WINDOW_LAPS,
            "sampled_races": [{"year": year, "race": race} for year, race in SAMPLED_RACES],
            "agreement": None
            if agreement is None
            else {
                "sample_size": agreement.sample_size,
                "eligible": agreement.eligible,
                "scored_share": agreement.scored_share,
                "races": agreement.races,
                "exact": agreement.exact,
                "within_one": agreement.within_one,
                "within_two": agreement.within_two,
                "mean_signed_error": agreement.mean_signed_error,
                "mean_absolute_error": agreement.mean_absolute_error,
            },
            "buckets": _bucket_counts(verdicts),
            "verdicts": [
                {
                    "year": v.year,
                    "race": v.race,
                    "driver": v.driver,
                    "actual_lap": v.actual_lap,
                    "chosen_lap": v.chosen_lap,
                    "offset_laps": v.offset_laps,
                    "bucket": v.bucket,
                }
                for v in verdicts
            ],
        },
    )
    return {
        "md_path": str(md_path),
        "json_path": str(json_path),
        "status": status,
        "agreement": agreement,
        "verdicts": verdicts,
    }
