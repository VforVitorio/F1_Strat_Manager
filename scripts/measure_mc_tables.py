"""Measure the quantities the projection-based Monte Carlo layer needs (#553).

The MC redesign (#550) replaces hand-picked constants with measured tables. This
script produces them from the RAW per-race parquets and writes
``data/mc_measured_v1.json`` plus human-readable twins under ``data/eval/``.

Five tables, each value carrying its sample size and a 95% interval:

- ``sc_window``       expected GREEN laps inside the W-lap decision window while a
                      neutralisation is active, plus how long spells last and how
                      often they run to the flag (the Art. 55.17 endgame).
- ``neutralisation_rate``  per-circuit onset hazard, the input to
                      ``q_f = 1 - exp(-rate * laps_remaining)`` (clamped to [0, 1]).
- ``gap_density``     measured seconds between consecutive cars under green and
                      under neutralisation — the empirical answer to the
                      ``POS_GAP_S = 1.5`` constant the redesign retires.
- ``undercut_band``   undercut success by gap-to-target in SECONDS, so target
                      eligibility stops being the ad-hoc "within 5 positions".
- ``stop_hazard``     P(a driver pits within W laps | compound, tyre-life bin,
                      neutralisation), for the surfaces whose rivals list carries
                      no ``is_pitting`` flag.

Why RAW and never the featured parquet: N04's ``IsAccurate`` gate drops laps run
under a Safety Car, pit laps and out-laps by design. Every measurement here is
ABOUT those laps, so the featured frame would report a race with no neutralisations
at all. The featured frame is the agents' feature channel; ``data/raw`` is the
source of truth for track status.

Regenerate with::

    uv run python scripts/measure_mc_tables.py

The output is deterministic: same parquets in, byte-identical JSON out. A test
(``tests/test_mc_measured_tables.py``) asserts the committed file matches a fresh
run, so a silent data change cannot drift the tables the engine reads.

--- WHERE TO CHANGE IF THE RACE DATA LAYOUT CHANGES ---
``data/raw/<year>/<folder>/laps.parquet`` is read directly, and folder names are a
keyspace of their own (underscored, and 2023 filed Barcelona under ``Spain``).
``_slug_from_folder`` is the only place that knows this; every table is keyed by
the circuit slug the agents query with (``session_meta.gp_name``).
"""

from __future__ import annotations

import json
import logging
import math
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.f1_strat_manager.gp_slugs import (  # noqa: E402
    FOLDER_ALIASES,
    resolve_gp_slug,
    slug_from_event_name,
)

logger = logging.getLogger("measure_mc_tables")

SCHEMA_VERSION = 1
YEARS: tuple[int, ...] = (2023, 2024, 2025)
WINDOW_LAPS = 5

RAW_DIR = ROOT / "data" / "raw"
UNDERCUT_LABELS = ROOT / "data" / "processed" / "undercut_labeled" / "undercut_clean.parquet"
JSON_OUT = ROOT / "data" / "mc_measured_v1.json"
EVAL_DIR = ROOT / "data" / "eval"

# 2023 filed the Circuit de Barcelona race under a country-named folder while
# 2024/25 use the circuit name. Left explicit rather than guessed: an unresolvable
# folder is reported, never silently dropped (the #429 guard rule).
FOLDER_SLUG_ALIASES: dict[str, str] = {"Spain": "Barcelona"}

# Track-status digits, verified across 79k raw laps 2023-25: '4' only ever means
# Safety Car and '6' only ever means VSC deployed, so a substring test over the
# concatenated per-lap statuses is sound (#438 round 2).
STATUS_SAFETY_CAR = "4"
STATUS_VSC = "6"
STATUS_RED_FLAG = "5"

GREEN = "green"
SAFETY_CAR = "sc"
VIRTUAL_SAFETY_CAR = "vsc"
RED_FLAG = "red"

# Tyre-life bins for the stop-hazard table. Coarse on purpose: the table is read
# as an eligibility prior, and narrow bins would ship cells with n < 30.
TYRE_LIFE_BINS: tuple[tuple[str, int, int], ...] = (
    ("0-9", 0, 9),
    ("10-19", 10, 19),
    ("20-29", 20, 29),
    ("30+", 30, 10_000),
)

# Gap bins (seconds behind the target) for the undercut band.
UNDERCUT_GAP_BINS: tuple[tuple[str, float, float], ...] = (
    ("0-1", 0.0, 1.0),
    ("1-2", 1.0, 2.0),
    ("2-3", 2.0, 3.0),
    ("3-5", 3.0, 5.0),
    ("5-10", 5.0, 10.0),
    ("10+", 10.0, 1e9),
)

MIN_CELL_N = 30  # below this a cell is reported but flagged as thin


# ---------------------------------------------------------------------------
# Statistics helpers — every measured value ships with n and an interval
# ---------------------------------------------------------------------------


def _mean_ci(values: Iterable[float]) -> dict[str, Any]:
    """Mean with a normal-approximation 95% interval, plus n.

    Returns ``n = 0`` and null statistics for an empty sample rather than a
    default number: an unmeasured quantity must be visibly unmeasured.
    """
    sample = [float(v) for v in values if v is not None and not math.isnan(float(v))]
    n = len(sample)
    if n == 0:
        return {"mean": None, "ci95": [None, None], "n": 0}

    mean = sum(sample) / n
    if n == 1:
        return {"mean": round(mean, 4), "ci95": [None, None], "n": 1}

    variance = sum((v - mean) ** 2 for v in sample) / (n - 1)
    half_width = 1.96 * math.sqrt(variance / n)
    interval = [round(mean - half_width, 4), round(mean + half_width, 4)]
    return {"mean": round(mean, 4), "ci95": interval, "n": n}


def _proportion_ci(successes: int, trials: int) -> dict[str, Any]:
    """Proportion with a Wilson 95% interval, plus n.

    Wilson rather than the normal approximation because several cells sit near
    0 or 1 with modest n, where the normal interval runs outside [0, 1].
    """
    if trials == 0:
        return {"rate": None, "ci95": [None, None], "n": 0, "successes": 0}

    z = 1.96
    phat = successes / trials
    denominator = 1 + z**2 / trials
    centre = (phat + z**2 / (2 * trials)) / denominator
    margin = z * math.sqrt(phat * (1 - phat) / trials + z**2 / (4 * trials**2)) / denominator
    interval = [round(max(0.0, centre - margin), 4), round(min(1.0, centre + margin), 4)]
    return {
        "rate": round(phat, 4),
        "ci95": interval,
        "n": trials,
        "successes": successes,
        "thin": trials < MIN_CELL_N,
    }


def _quantiles(values: Iterable[float]) -> dict[str, Any]:
    """P10 / P50 / P90 of a sample, or nulls when it is empty."""
    series = pd.Series([float(v) for v in values], dtype="float64").dropna()
    if series.empty:
        return {"p10": None, "p50": None, "p90": None, "n": 0}
    return {
        "p10": round(float(series.quantile(0.10)), 4),
        "p50": round(float(series.quantile(0.50)), 4),
        "p90": round(float(series.quantile(0.90)), 4),
        "n": int(series.size),
    }


def _bin_label(value: float, bins: tuple[tuple[str, float, float], ...]) -> str | None:
    """Return the label of the first bin containing ``value`` (upper bound inclusive)."""
    for label, low, high in bins:
        if low <= value <= high:
            return label
    return None


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def _slug_from_folder(folder: str) -> str | None:
    """Map a ``data/raw/<year>/`` folder name to the circuit slug agents query with.

    Three keyspaces meet here and only two of them were ever reconciled (#448
    fixed FastF1 event names against slugs). Folder names are the third: they
    underscore multi-word slugs (``Las_Vegas``) and 2023 filed Barcelona under
    ``Spain``. Returns None for a folder that resolves to no known circuit, so
    the caller can report it instead of measuring a race under a key nothing
    will ever look up.
    """
    spaced = FOLDER_ALIASES.get(folder, folder.replace("_", " "))
    candidate = FOLDER_SLUG_ALIASES.get(spaced, spaced)
    try:
        resolve_gp_slug(candidate)
    except ValueError:
        return None
    return candidate


@dataclass(frozen=True)
class RaceLaps:
    """One race's raw laps plus the per-lap race-level track status.

    Attributes:
        year:          Season year.
        slug:          Circuit slug (the keyspace the agents query with).
        laps:          Raw laps frame, all drivers, all laps (nothing filtered).
        total_laps:    Highest lap number any driver completed.
        status_by_lap: Lap number to one of green / sc / vsc / red.
    """

    year: int
    slug: str
    laps: pd.DataFrame
    total_laps: int
    status_by_lap: dict[int, str]

    def is_neutralised(self, lap: int) -> bool:
        """Whether ``lap`` ran under any non-green condition."""
        return self.status_by_lap.get(lap, GREEN) != GREEN


def _status_by_lap(laps: pd.DataFrame) -> dict[int, str]:
    """Reduce every driver's TrackStatus string to one race-level status per lap.

    A lap counts as neutralised when ANY car reports the flag, because the flag
    is a property of the race, not of a car; drivers already past the incident
    keep a clean status for that lap. Precedence red > SC > VSC > green: a lap
    that saw an SC deployed and withdrawn is an SC lap for the decision layer,
    since no racing happened on it.
    """
    statuses: dict[int, str] = {}
    for lap, group in laps.groupby("LapNumber"):
        joined = "".join(group["TrackStatus"].dropna().astype(str))
        if STATUS_RED_FLAG in joined:
            statuses[int(lap)] = RED_FLAG
        elif STATUS_SAFETY_CAR in joined:
            statuses[int(lap)] = SAFETY_CAR
        elif STATUS_VSC in joined:
            statuses[int(lap)] = VIRTUAL_SAFETY_CAR
        else:
            statuses[int(lap)] = GREEN
    return statuses


def load_races(years: tuple[int, ...] = YEARS) -> list[RaceLaps]:
    """Load every raw race parquet for ``years``, keyed by circuit slug.

    Folders that resolve to no known circuit are reported and skipped loudly:
    a silently dropped race is a silently smaller n on every table below.
    """
    races: list[RaceLaps] = []
    unresolved: list[str] = []

    for year in years:
        year_dir = RAW_DIR / str(year)
        if not year_dir.is_dir():
            logger.warning("no raw data for %s at %s", year, year_dir)
            continue

        for folder in sorted(p.name for p in year_dir.iterdir() if p.is_dir()):
            laps_path = year_dir / folder / "laps.parquet"
            if not laps_path.exists():
                continue

            slug = _slug_from_folder(folder)
            if slug is None:
                unresolved.append(f"{year}/{folder}")
                continue

            laps = pd.read_parquet(laps_path)
            statuses = _status_by_lap(laps)
            races.append(
                RaceLaps(
                    year=year,
                    slug=slug,
                    laps=laps,
                    total_laps=max(statuses) if statuses else 0,
                    status_by_lap=statuses,
                )
            )

    if unresolved:
        logger.warning("skipped %d unresolvable race folder(s): %s", len(unresolved), unresolved)
    return races


# ---------------------------------------------------------------------------
# Table 1 — the neutralisation window (W_green and spell length)
# ---------------------------------------------------------------------------


def _spell_length_from(race: RaceLaps, lap: int) -> int:
    """Count consecutive neutralised laps starting at ``lap`` (inclusive)."""
    length = 0
    current = lap
    while current <= race.total_laps and race.is_neutralised(current):
        length += 1
        current += 1
    return length


def _green_laps_in_window(race: RaceLaps, lap: int, window: int) -> int:
    """Green laps among the ``window`` laps that follow ``lap``.

    Laps past the chequered flag are not green laps: a decision taken three laps
    from the end cannot bank five laps of racing, and the projection must see
    that, so the count is bounded by the race distance rather than assumed full.
    """
    upcoming = range(lap + 1, min(lap + window, race.total_laps) + 1)
    return sum(1 for nxt in upcoming if not race.is_neutralised(nxt))


def measure_sc_window(races: list[RaceLaps], window: int = WINDOW_LAPS) -> dict[str, Any]:
    """Expected green laps in the decision window while a neutralisation is active.

    This is the ``W_green`` the projection scales its per-lap accrual terms by.
    Reported per neutralisation kind: an SC bunches the field and runs for laps,
    a VSC is typically over within one or two, and the decision layer must not
    treat them alike (#471).

    Also reports how long spells last and how often one reaches the flag, which
    is the Art. 55.17 endgame the redesign wants to emerge from numbers rather
    than from a rail: when the race ends behind the Safety Car, a stop buys
    nothing because there are no green laps left to spend the fresh tyres on.
    """
    green_in_window: dict[str, list[int]] = defaultdict(list)
    spell_lengths: dict[str, list[int]] = defaultdict(list)
    spells_to_flag: dict[str, int] = defaultdict(int)
    spells_total: dict[str, int] = defaultdict(int)

    for race in races:
        for lap in range(1, race.total_laps + 1):
            status = race.status_by_lap.get(lap, GREEN)
            if status not in (SAFETY_CAR, VIRTUAL_SAFETY_CAR):
                continue

            green_in_window[status].append(_green_laps_in_window(race, lap, window))

            starts_spell = not race.is_neutralised(lap - 1) if lap > 1 else True
            if not starts_spell:
                continue

            length = _spell_length_from(race, lap)
            spell_lengths[status].append(length)
            spells_total[status] += 1
            if lap + length - 1 >= race.total_laps:
                spells_to_flag[status] += 1

    per_kind: dict[str, Any] = {}
    for kind in (SAFETY_CAR, VIRTUAL_SAFETY_CAR):
        per_kind[kind] = {
            "green_laps_in_window": _mean_ci(green_in_window[kind]),
            "green_laps_quantiles": _quantiles(green_in_window[kind]),
            "spell_length_laps": _mean_ci(spell_lengths[kind]),
            "spell_length_quantiles": _quantiles(spell_lengths[kind]),
            "runs_to_the_flag": _proportion_ci(spells_to_flag[kind], spells_total[kind]),
        }

    table = {
        "window_laps": window,
        "definition": (
            "For every lap run under a neutralisation, the number of GREEN laps among "
            "the next W laps, bounded by the race distance. Spell length counts "
            "consecutive neutralised laps from each spell's first lap."
        ),
        "by_kind": per_kind,
    }
    return table


# ---------------------------------------------------------------------------
# Table 2 — neutralisation onset hazard (the q_f input)
# ---------------------------------------------------------------------------


def measure_neutralisation_rate(races: list[RaceLaps]) -> dict[str, Any]:
    """Per-circuit per-lap hazard that a NEW neutralisation begins.

    Feeds the option-value term: ``q_f = 1 - exp(-rate * laps_remaining)`` is the
    probability that a future neutralisation turns up to cover a stop we have not
    taken yet. The exponential form is what keeps q_f a probability — the naive
    ``rate * laps_remaining`` exceeds 1 on a long green run and would hand the MC
    a nonsense certainty.

    Only GREEN laps are at risk of an onset, so they are the denominator; laps
    already neutralised cannot start a new spell.
    """
    onsets: dict[str, int] = defaultdict(int)
    green_laps: dict[str, int] = defaultdict(int)
    races_seen: dict[str, int] = defaultdict(int)

    for race in races:
        races_seen[race.slug] += 1
        for lap in range(1, race.total_laps + 1):
            previous_green = race.status_by_lap.get(lap - 1, GREEN) == GREEN if lap > 1 else True
            if race.status_by_lap.get(lap, GREEN) == GREEN:
                green_laps[race.slug] += 1
            elif previous_green:
                onsets[race.slug] += 1

    per_circuit = {
        slug: {
            **_proportion_ci(onsets[slug], green_laps[slug]),
            "races": races_seen[slug],
        }
        for slug in sorted(races_seen)
    }
    pooled = _proportion_ci(sum(onsets.values()), sum(green_laps.values()))

    table = {
        "definition": (
            "Onsets per green lap at risk, per circuit. q_f(laps_remaining) = "
            "1 - exp(-rate * laps_remaining), clamped to [0, 1]."
        ),
        "q_f_form": "1 - exp(-rate * laps_remaining)",
        "pooled": pooled,
        "per_circuit": per_circuit,
    }
    return table


# ---------------------------------------------------------------------------
# Table 3 — how many seconds a position is actually worth
# ---------------------------------------------------------------------------


def _lap_intervals(lap_rows: pd.DataFrame) -> list[float]:
    """Seconds between consecutive cars on one lap, from elapsed session time."""
    elapsed = lap_rows["Time"].dropna().sort_values()
    if elapsed.size < 2:
        return []
    diffs = elapsed.diff().dropna().dt.total_seconds()
    return [float(v) for v in diffs if v >= 0]


def measure_gap_density(races: list[RaceLaps]) -> dict[str, Any]:
    """Measured seconds between consecutive cars, under green and neutralised.

    The legacy scoring divided seconds by a flat ``POS_GAP_S = 1.5`` to convert a
    time loss into positions. This measures what that constant approximates, and
    separates the two regimes: under a Safety Car the field closes up, so the
    same 20-second pit loss costs a very different number of cars. The projection
    needs no such constant — it counts the actual cars — but publishing the
    measurement is what retires the constant honestly instead of by assertion.
    """
    intervals: dict[str, list[float]] = {GREEN: [], SAFETY_CAR: []}

    for race in races:
        for lap, lap_rows in race.laps.groupby("LapNumber"):
            status = race.status_by_lap.get(int(lap), GREEN)
            if status not in intervals:
                continue
            intervals[status].extend(_lap_intervals(lap_rows))

    table = {
        "definition": (
            "Seconds between consecutive cars on the same lap, from elapsed session "
            "time (the Time column), pooled over every lap of every race."
        ),
        "retires_constant": "POS_GAP_S = 1.5 s/position (legacy scoring path only)",
        "green": {**_mean_ci(intervals[GREEN]), **_quantiles(intervals[GREEN])},
        "safety_car": {**_mean_ci(intervals[SAFETY_CAR]), **_quantiles(intervals[SAFETY_CAR])},
    }
    return table


# ---------------------------------------------------------------------------
# Table 4 — the undercut band, in seconds
# ---------------------------------------------------------------------------


def _elapsed_at_lap(laps: pd.DataFrame, driver: str, lap: int) -> float | None:
    """Elapsed session seconds for ``driver`` at the end of ``lap``, or None."""
    row = laps[(laps["Driver"] == driver) & (laps["LapNumber"] == lap)]
    if row.empty:
        return None
    elapsed = row.iloc[0].get("Time")
    if elapsed is None or pd.isna(elapsed):
        return None
    return float(elapsed.total_seconds())


def _attempt_gap_seconds(race: RaceLaps, attacker: str, target: str, pit_lap: int) -> float | None:
    """Seconds the attacker sat behind the target on the lap BEFORE the stop.

    The decision lap is the one before the in-lap: that is the state a strategist
    actually has when calling the driver in. Positive means the target is ahead,
    which is the only orientation an undercut makes sense in.
    """
    decision_lap = pit_lap - 1
    if decision_lap < 1:
        return None

    attacker_time = _elapsed_at_lap(race.laps, attacker, decision_lap)
    target_time = _elapsed_at_lap(race.laps, target, decision_lap)
    if attacker_time is None or target_time is None:
        return None
    return attacker_time - target_time


def measure_undercut_band(races: list[RaceLaps]) -> dict[str, Any]:
    """Undercut success against the gap to the target, in seconds.

    Reuses N16's own labelled attempts (``undercut_clean.parquet``) rather than
    re-deriving what counts as an undercut: a second definition would drift from
    the model the MC already samples from. What this adds is the gap in SECONDS,
    which the labels do not carry — they encode the gap in positions, and the
    projection reasons in time.

    The keyspace trap: those labels are keyed by FastF1 event name while the raw
    folders are keyed by circuit, with zero overlap between the two (#448). The
    join goes through ``slug_from_event_name`` and reports its own match rate, so
    a future rename shows up as a visible drop rather than a quiet n of zero.
    """
    if not UNDERCUT_LABELS.exists():
        logger.warning("no undercut labels at %s — table skipped", UNDERCUT_LABELS)
        return {"available": False, "reason": f"missing {UNDERCUT_LABELS.name}"}

    attempts = pd.read_parquet(UNDERCUT_LABELS)
    races_by_key = {(race.year, race.slug): race for race in races}

    matched: list[tuple[float, bool]] = []
    unmatched_race = 0
    unmatched_gap = 0

    for _, attempt in attempts.iterrows():
        slug = slug_from_event_name(str(attempt["GP_Name"]))
        race = races_by_key.get((int(attempt["Year"]), slug))
        if race is None:
            unmatched_race += 1
            continue

        gap = _attempt_gap_seconds(
            race,
            attacker=str(attempt["Driver_X"]),
            target=str(attempt["Driver_Y"]),
            pit_lap=int(attempt["Lap_X_pits"]),
        )
        if gap is None or gap <= 0:
            # gap <= 0 means the "target" was already behind the attacker on the
            # decision lap: not an undercut, whatever the label pairing says.
            unmatched_gap += 1
            continue

        matched.append((gap, bool(attempt["undercut_success"])))

    by_bin: dict[str, Any] = {}
    for label, low, high in UNDERCUT_GAP_BINS:
        cell = [success for gap, success in matched if low <= gap < high]
        by_bin[label] = _proportion_ci(sum(cell), len(cell))

    successful_gaps = [gap for gap, success in matched if success]
    band = _quantiles(successful_gaps)

    table = {
        "definition": (
            "N16's labelled undercut attempts, enriched with the elapsed-time gap to "
            "the target on the lap BEFORE the stop. Attempts whose target was already "
            "behind on that lap are excluded."
        ),
        "source_labels": str(UNDERCUT_LABELS.relative_to(ROOT)).replace("\\", "/"),
        "attempts_total": int(len(attempts)),
        "attempts_matched": len(matched),
        "dropped_no_race": unmatched_race,
        "dropped_no_gap_or_behind": unmatched_gap,
        "overall": _proportion_ci(sum(1 for _, s in matched if s), len(matched)),
        "by_gap_bin_seconds": by_bin,
        "successful_attempt_gap_quantiles": band,
        "u_band_s": band["p90"],
        "u_band_note": (
            "P90 of the gaps at which a real undercut succeeded: beyond it, success is "
            "rare enough that the candidate should not be offered a target."
        ),
    }
    return table


# ---------------------------------------------------------------------------
# Table 5 — will that rival stop soon?
# ---------------------------------------------------------------------------


def _driver_pit_laps(laps: pd.DataFrame) -> dict[str, set[int]]:
    """Laps at whose end each driver entered the pit lane."""
    pitting = laps[laps["PitInTime"].notna()]
    pit_laps: dict[str, set[int]] = defaultdict(set)
    for _, row in pitting.iterrows():
        pit_laps[str(row["Driver"])].add(int(row["LapNumber"]))
    return pit_laps


def measure_stop_hazard(races: list[RaceLaps], window: int = WINDOW_LAPS) -> dict[str, Any]:
    """P(a driver pits within the next W laps | compound, tyre life, neutralisation).

    OVERCUT eligibility in the redesign keys off a rival actually being in the pit
    lane, which the rivals list reports as ``is_pitting``. Two surfaces build their
    rivals from the featured parquet and carry no such flag, so on those the fact
    is unavailable and this measured prior stands in for it.

    It is deliberately a prior and not a prediction: it answers "how often does a
    car on this compound at this tyre age stop within five laps", never "will this
    car stop", which would need the rival's strategy — the full-race modelling the
    redesign rules out.
    """
    cells: dict[tuple[str, str, str], list[bool]] = defaultdict(list)

    for race in races:
        pit_laps = _driver_pit_laps(race.laps)
        for _, row in race.laps.iterrows():
            compound = str(row.get("Compound", "") or "")
            tyre_life = row.get("TyreLife")
            lap = row.get("LapNumber")
            if not compound or pd.isna(tyre_life) or pd.isna(lap):
                continue

            bin_label = _bin_label(float(tyre_life), TYRE_LIFE_BINS)
            if bin_label is None:
                continue

            lap = int(lap)
            regime = SAFETY_CAR if race.is_neutralised(lap) else GREEN
            upcoming = range(lap + 1, min(lap + window, race.total_laps) + 1)
            stops_soon = any(nxt in pit_laps[str(row["Driver"])] for nxt in upcoming)
            cells[(compound, bin_label, regime)].append(stops_soon)

    by_cell = {
        f"{compound}|{bin_label}|{regime}": _proportion_ci(sum(observations), len(observations))
        for (compound, bin_label, regime), observations in sorted(cells.items())
    }

    table = {
        "definition": (
            "Share of laps after which the driver entered the pit lane within the next "
            "W laps, grouped by compound, tyre-life bin and whether the lap was "
            "neutralised. Key format: compound|tyre_life_bin|regime."
        ),
        "window_laps": window,
        "tyre_life_bins": [label for label, _, _ in TYRE_LIFE_BINS],
        "min_cell_n": MIN_CELL_N,
        "by_cell": by_cell,
    }
    return table


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def build_tables(races: list[RaceLaps]) -> dict[str, Any]:
    """Run every measurement and assemble the versioned payload."""
    payload = {
        "schema_version": SCHEMA_VERSION,
        "generated_by": "scripts/measure_mc_tables.py",
        "source": "data/raw/<year>/<folder>/laps.parquet (RAW, never the featured parquet)",
        "years": list(YEARS),
        "races_measured": len(races),
        "window_laps": WINDOW_LAPS,
        "sc_window": measure_sc_window(races),
        "neutralisation_rate": measure_neutralisation_rate(races),
        "gap_density": measure_gap_density(races),
        "undercut_band": measure_undercut_band(races),
        "stop_hazard": measure_stop_hazard(races),
    }
    return payload


def _sc_window_rows(tables: dict[str, Any]) -> list[dict[str, Any]]:
    """Flatten the neutralisation-window table into CSV rows."""
    rows = []
    for kind, stats in tables["sc_window"]["by_kind"].items():
        rows.append(
            {
                "kind": kind,
                "green_laps_in_window_mean": stats["green_laps_in_window"]["mean"],
                "green_laps_in_window_n": stats["green_laps_in_window"]["n"],
                "spell_length_mean": stats["spell_length_laps"]["mean"],
                "spell_length_p90": stats["spell_length_quantiles"]["p90"],
                "spells_n": stats["runs_to_the_flag"]["n"],
                "runs_to_the_flag_rate": stats["runs_to_the_flag"]["rate"],
            }
        )
    return rows


def _gap_density_rows(tables: dict[str, Any]) -> list[dict[str, Any]]:
    """Flatten the gap-density table into CSV rows."""
    rows = []
    for regime in ("green", "safety_car"):
        stats = tables["gap_density"][regime]
        rows.append(
            {
                "regime": regime,
                "mean_s": stats["mean"],
                "p10_s": stats["p10"],
                "p50_s": stats["p50"],
                "p90_s": stats["p90"],
                "n": stats["n"],
            }
        )
    return rows


def _undercut_rows(tables: dict[str, Any]) -> list[dict[str, Any]]:
    """Flatten the undercut band into CSV rows, one per gap bin."""
    band = tables["undercut_band"]
    if not band.get("by_gap_bin_seconds"):
        return []
    rows = []
    for label, stats in band["by_gap_bin_seconds"].items():
        rows.append(
            {
                "gap_bin_s": label,
                "success_rate": stats["rate"],
                "ci95_low": stats["ci95"][0],
                "ci95_high": stats["ci95"][1],
                "n": stats["n"],
            }
        )
    return rows


def _write_twin(stem: str, title: str, rows: list[dict[str, Any]]) -> None:
    """Write the CSV (dot decimal) and Markdown (comma decimal) twins for one table.

    Mirrors the existing ``data/eval`` convention: the CSV is pandas-loadable for
    ``df.to_latex``, the Markdown is ready to paste into the thesis.
    """
    if not rows:
        return

    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(EVAL_DIR / f"{stem}.csv", index=False)

    def _format(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, float):
            return f"{value:.3f}".replace(".", ",")
        return str(value)

    header = "| " + " | ".join(frame.columns) + " |"
    divider = "|" + "---|" * len(frame.columns)
    body = [
        "| " + " | ".join(_format(value) for value in row) + " |"
        for row in frame.itertuples(index=False)
    ]
    markdown = "\n".join([f"## {title}", "", header, divider, *body, ""])
    (EVAL_DIR / f"{stem}.md").write_text(markdown, encoding="utf-8")


def write_outputs(tables: dict[str, Any]) -> None:
    """Write the versioned JSON and the human-readable eval twins."""
    JSON_OUT.write_text(json.dumps(tables, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    logger.info("wrote %s", JSON_OUT.relative_to(ROOT))

    _write_twin("mc_sc_window", "Neutralisation window (W=5)", _sc_window_rows(tables))
    _write_twin("mc_gap_density", "Seconds between consecutive cars", _gap_density_rows(tables))
    _write_twin("mc_undercut_band", "Undercut success by gap to target", _undercut_rows(tables))


def main() -> int:
    """Load every raw race, measure the five tables, write the artefacts."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    races = load_races()
    if not races:
        logger.error("no races loaded from %s — is data/raw populated?", RAW_DIR)
        return 2

    logger.info("loaded %d races (%s)", len(races), ", ".join(str(y) for y in YEARS))
    tables = build_tables(races)
    write_outputs(tables)

    sc_stats = tables["sc_window"]["by_kind"][SAFETY_CAR]
    logger.info(
        "SC: %.2f green laps in a %d-lap window (n=%d); spells run to the flag %.1f%% of the time",
        sc_stats["green_laps_in_window"]["mean"] or 0.0,
        WINDOW_LAPS,
        sc_stats["green_laps_in_window"]["n"],
        100 * (sc_stats["runs_to_the_flag"]["rate"] or 0.0),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
