"""Score the LLM path's per-lap decisions against the real 2025 pit wall.

The companion to ``decision_modes.py``, which measures the DETERMINISTIC layer
(Monte Carlo plus guard rails, zero API calls). Twelve of the fourteen
``StrategyRecommendation`` fields are written by the LLM, so the deterministic
tier measures the part of the product the product does not ship alone. This
module grades the same kind of question over the same kind of sample, on rows
produced by ``scripts/measure_llm_windows.py``.

WHAT IS SHARED WITH THE DETERMINISTIC TIER, AND WHY THAT MATTERS
----------------------------------------------------------------
The sample definition (``green_flag_stops``), the rail exclusions
(``guard_rail_block``), the pit-action set, the transition rule
(``_pit_decision_lap``) and the aggregate type (``DecisionAgreement``) are all
IMPORTED, never restated. Two tiers whose numbers get compared to each other
must be measuring the same thing, and the way this repository has broken that
before is by keeping a private copy of the definition inside the harness. If a
number here is compared to a number there, the comparison is only honest
because these definitions are literally the same objects.

WHAT IS DIFFERENT, AND HAS TO BE SAID EVERY TIME
------------------------------------------------
1. **The sample is smaller and it is chosen, not enumerated.** The
   deterministic tier can afford to sweep every green-flag stop of six races at
   0.51 s/lap. This path costs 15.93 s/lap, so it runs over named windows. A
   figure from here describes those windows and nothing wider.
2. **It is not deterministic.** The orchestrator requests ``temperature=0`` and
   ``gpt-5.4-mini`` discards it. A single pass is one sample of a distribution,
   which is why ``pass_index`` exists and why :func:`repeat_disagreement` is
   reported next to any agreement figure drawn from repeats.
3. **Agreement is still not correctness.** The team can be wrong; Qatar 2025 is
   in this sample precisely because the press says it was. Nothing here is a
   counterfactual: ``RaceReplayEngine`` replays laps that happened.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.strategy.eval.decision_modes import (
    DECISION_WINDOW_LAPS,
    DecisionAgreement,
    StopVerdict,
    _pit_decision_lap,
    _stop_context,
    coverage_verdict,
    guard_rail_block,
)
from src.strategy.eval.projection import _neutralised_laps, green_flag_stops

# Raw-folder name per featured-artefact name. The eval path reads RAW folders and
# the featured parquet spells four of them differently; a single mismatch runs a
# whole race on another race's data, which has already happened here once.
_RAW_FOLDER: dict[str, str] = {
    "Miami": "Miami_Gardens",
    "Marina Bay": "Marina_Bay",
    "São Paulo": "São_Paulo",
    "Las Vegas": "Las_Vegas",
    "Mexico City": "Mexico_City",
}


@dataclass(frozen=True)
class WindowKey:
    """One measured (race, driver, pass) window."""

    race: str
    driver: str
    pass_index: int


def load_rows(path: Path) -> list[dict[str, Any]]:
    """Every per-lap row in a measurement JSONL, skipping a torn trailing line."""
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def actions_by_window(rows: list[dict[str, Any]]) -> dict[WindowKey, dict[int, str]]:
    """``{window: {lap: action}}``, which is the shape the transition rule wants.

    A later row for the same lap wins. That only happens when a window is
    re-measured into the same file, and the newest measurement is the one the
    caller asked for.
    """
    actions: dict[WindowKey, dict[int, str]] = defaultdict(dict)
    for row in rows:
        key = WindowKey(str(row["race"]), str(row["driver"]), int(row.get("pass_index", 0)))
        action = (row.get("recommendation") or {}).get("action")
        if action is None:
            continue
        actions[key][int(row["lap"])] = str(action)
    return dict(actions)


def _raw_laps(race: str, year: int, raw_root: Path) -> pd.DataFrame:
    """The raw laps frame for a race, resolving the folder-vs-featured keyspace."""
    folder = _RAW_FOLDER.get(race, race)
    return pd.read_parquet(raw_root / str(year) / folder / "laps.parquet")


def score_window(
    key: WindowKey,
    actions: dict[int, str],
    laps: pd.DataFrame,
    year: int,
) -> list[StopVerdict]:
    """Verdicts for every real green-flag stop this window could speak to.

    A stop is only considered when the window actually covers laps around it:
    the point of the exercise is what the stack said near the real decision, and
    grading a stop the window never approached would count the sampling as a
    model failure.
    """
    neutralised = _neutralised_laps(laps)
    stops = green_flag_stops(laps, neutralised).get(key.driver, [])
    total_laps = int(laps["LapNumber"].max())
    evaluated = set(actions)

    verdicts: list[StopVerdict] = []
    for stop_lap in stops:
        window_low = max(1, stop_lap - DECISION_WINDOW_LAPS)
        window_high = min(total_laps, stop_lap + DECISION_WINDOW_LAPS)
        if not evaluated & set(range(window_low, window_high + 1)):
            continue

        compound, tyre_life = _stop_context(laps, key.driver, stop_lap)
        blocked = guard_rail_block(stop_lap, total_laps, compound, tyre_life)
        if blocked:
            verdicts.append(StopVerdict(year, key.race, key.driver, stop_lap, None, None, blocked))
            continue

        chosen = _pit_decision_lap(actions, window_low, window_high)
        if chosen is None:
            asked = any(
                actions.get(lap) in {"PIT_NOW", "UNDERCUT", "OVERCUT", "REACTIVE_SC"}
                for lap in range(window_low, window_high + 1)
            )
            bucket = "no_boundary_in_window" if asked else "no_call_in_window"
            verdicts.append(StopVerdict(year, key.race, key.driver, stop_lap, None, None, bucket))
            continue

        verdicts.append(
            StopVerdict(year, key.race, key.driver, stop_lap, chosen, chosen - stop_lap, "scored")
        )
    return verdicts


def aggregate(verdicts: list[StopVerdict]) -> DecisionAgreement:
    """Fold verdicts into the same aggregate type the deterministic tier reports."""
    offsets = np.array([v.offset_laps for v in verdicts if v.offset_laps is not None], dtype=float)
    buckets = Counter(v.bucket for v in verdicts)
    rail_buckets = ("opening_laps", "closing_laps", "min_stint")
    agreement = DecisionAgreement(
        offsets=offsets,
        guard_railed=sum(buckets[name] for name in rail_buckets),
        no_call=buckets["no_call_in_window"],
        races=len({v.race for v in verdicts}),
        no_data=buckets["no_data"],
        no_boundary=buckets["no_boundary_in_window"],
    )
    return agreement


def repeat_disagreement(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """How often two passes over the same lap disagree, field by field.

    This is the number that says whether a single pass is quotable at all. The
    orchestrator's ``temperature=0`` does not survive into the client for the
    shipped model, so a run-to-run difference is the expected state and its size
    is the question.
    """
    # Keyed on the WINDOW as well as the lap. The same lap measured from two
    # different window starts is not a repeat of itself: the ``DecisionMemory``
    # block entering the orchestrator prompt is warmer the further back the
    # window began. Pooling them charges context variation to run-to-run noise,
    # which overstated `pit_lap_target` by 9.1 points and `pace_mode` by 4.6 the
    # first time these figures were published. Rows written before the recorder
    # stored window bounds fall back to the old pooled key, which is the honest
    # degradation: it cannot separate what was never recorded.
    by_lap: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (
            str(row["race"]),
            str(row["driver"]),
            int(row["lap"]),
            row.get("window_low"),
            row.get("window_high"),
        )
        by_lap[key].append(row)

    repeated = {lap: rs for lap, rs in by_lap.items() if len(rs) > 1}
    if not repeated:
        return {"laps_with_repeats": 0}

    fields = ("action", "confidence", "pit_lap_target", "compound_next", "pace_mode")
    differing = {name: 0 for name in fields}
    for candidates in repeated.values():
        for name in fields:
            values = {
                json.dumps((r.get("recommendation") or {}).get(name), default=str)
                for r in candidates
            }
            if len(values) > 1:
                differing[name] += 1

    result = {
        "laps_with_repeats": len(repeated),
        "laps_differing": differing,
        "share_differing": {
            name: round(count / len(repeated), 3) for name, count in differing.items()
        },
    }
    return result


def llm_field_census(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """What the LLM actually filled in, over every measured lap.

    Exists because the deterministic tier cannot report any of it: eleven of
    these fields do not exist on the no-llm path, which emits an argmax and a
    fixed reasoning string. Whether the LLM populates a planning field at all is
    a result in its own right.
    """
    recommendations = [r.get("recommendation") or {} for r in rows]
    if not recommendations:
        return {}

    confidences = [
        float(r["confidence"])
        for r in recommendations
        if isinstance(r.get("confidence"), (int, float))
    ]
    populated = {
        name: sum(1 for r in recommendations if r.get(name) not in (None, "", []))
        for name in (
            "pit_lap_target",
            "compound_next",
            "undercut_target",
            "target_lap_time_s",
            "pace_mode",
            "risk_posture",
            "contingencies",
        )
    }
    census = {
        "laps": len(recommendations),
        "actions": dict(Counter(str(r.get("action")) for r in recommendations).most_common()),
        "pace_modes": dict(Counter(str(r.get("pace_mode")) for r in recommendations).most_common()),
        "risk_postures": dict(
            Counter(str(r.get("risk_posture")) for r in recommendations).most_common()
        ),
        "confidence": {
            "n": len(confidences),
            "mean": round(float(np.mean(confidences)), 3) if confidences else None,
            "median": round(float(np.median(confidences)), 3) if confidences else None,
            "min": round(float(np.min(confidences)), 3) if confidences else None,
            "max": round(float(np.max(confidences)), 3) if confidences else None,
        },
        "fields_populated": populated,
    }
    return census


def measure(jsonl: Path, year: int, raw_root: Path) -> dict[str, Any]:
    """Everything scoreable in one measurement file."""
    rows = load_rows(jsonl)
    windows = actions_by_window(rows)

    laps_cache: dict[str, pd.DataFrame] = {}
    verdicts: list[StopVerdict] = []
    for key, actions in sorted(windows.items(), key=lambda kv: (kv[0].race, kv[0].driver)):
        if key.race not in laps_cache:
            laps_cache[key.race] = _raw_laps(key.race, year, raw_root)
        verdicts += score_window(key, actions, laps_cache[key.race], year)

    agreement = aggregate(verdicts)
    seconds = [float(r.get("seconds", 0.0)) for r in rows]
    # DISTINCT laps, not rows. A resumed run re-executes any window that is not
    # complete on disk and appends its laps again, so the row count over-states
    # coverage by exactly the re-runs. `actions_by_window` already resolves the
    # duplicate by taking the later row; the count has to resolve it too, or the
    # coverage line reports more laps than were ever distinctly evaluated.
    distinct = {
        (str(r["race"]), str(r["driver"]), int(r["lap"]), int(r.get("pass_index", 0))) for r in rows
    }
    result = {
        "source": str(jsonl),
        "laps_measured": len(distinct),
        "rows_on_disk": len(rows),
        "windows": len(windows),
        "wall_clock_seconds": round(sum(seconds), 1),
        "tokens": {
            "prompt": sum(int(r.get("prompt_tokens", 0)) for r in rows),
            "completion": sum(int(r.get("completion_tokens", 0)) for r in rows),
            "calls": sum(int(r.get("llm_calls", 0)) for r in rows),
            # None, not 0, when no row carries the field: "measured zero" and "never
            # recorded" are different statements and the report prints different
            # sentences for them.
            "cached_prompt": (
                sum(int(r.get("cached_prompt_tokens", 0)) for r in rows)
                if any("cached_prompt_tokens" in r for r in rows)
                else None
            ),
        },
        "agreement": {
            "scored": agreement.sample_size,
            "eligible": agreement.eligible,
            "scored_share": round(agreement.scored_share, 4),
            "coverage_verdict": coverage_verdict(agreement),
            "exact": round(agreement.exact, 4),
            "within_one": round(agreement.within_one, 4),
            "within_two": round(agreement.within_two, 4),
            "mean_signed_error": round(agreement.mean_signed_error, 3),
            "mean_absolute_error": round(agreement.mean_absolute_error, 3),
        },
        "buckets": dict(Counter(v.bucket for v in verdicts).most_common()),
        "verdicts": [
            {
                "race": v.race,
                "driver": v.driver,
                "actual_lap": v.actual_lap,
                "chosen_lap": v.chosen_lap,
                "offset_laps": v.offset_laps,
                "bucket": v.bucket,
            }
            for v in verdicts
        ],
        "llm_fields": llm_field_census(rows),
        "repeats": repeat_disagreement(rows),
    }
    return result


def _self_check() -> None:
    """Smallest runnable check: the transition rule and the aggregate agree."""
    actions = {14: "STAY_OUT", 15: "STAY_OUT", 16: "STAY_OUT", 17: "PIT_NOW", 18: "PIT_NOW"}
    assert _pit_decision_lap(actions, 14, 24) == 17

    verdicts = [
        StopVerdict(2025, "Budapest", "LEC", 19, 17, -2, "scored"),
        StopVerdict(2025, "Budapest", "LEC", 40, None, None, "no_call_in_window"),
        StopVerdict(2025, "Lusail", "PIA", 24, 24, 0, "scored"),
    ]
    agreement = aggregate(verdicts)
    assert agreement.sample_size == 2, agreement
    assert agreement.eligible == 3, agreement
    assert agreement.no_call == 1, agreement
    assert abs(agreement.exact - 0.5) < 1e-9, agreement
    assert abs(agreement.mean_signed_error - (-1.0)) < 1e-9, agreement
    # 2 of 3 is 66.7%, above the 60% gate, so this reads ``ok``. Asserted as the
    # value rather than as a guess: the first version of this line assumed
    # ``masked`` and the check caught it, which is the whole reason it exists.
    assert coverage_verdict(agreement) == "ok", agreement
    assert (
        coverage_verdict(
            aggregate(
                verdicts + [StopVerdict(2025, "X", "D", 9, None, None, "no_call_in_window")] * 3
            )
        )
        == "masked"
    )

    rows = [
        {
            "race": "X",
            "driver": "D",
            "lap": 5,
            "recommendation": {"action": "STAY_OUT", "confidence": 0.9},
        },
        {
            "race": "X",
            "driver": "D",
            "lap": 5,
            "recommendation": {"action": "PIT_NOW", "confidence": 0.9},
        },
    ]
    spread = repeat_disagreement(rows)
    assert spread["laps_with_repeats"] == 1, spread
    assert spread["laps_differing"]["action"] == 1, spread
    assert spread["laps_differing"]["confidence"] == 0, spread
    print("llm_decision self-check OK")


if __name__ == "__main__":
    _self_check()
