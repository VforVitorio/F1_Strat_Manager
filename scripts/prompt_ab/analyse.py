"""Stage 4: diff two passes against a noise floor. No API calls.

The only honest way to read an A/B on this stack. Give it three passes - two runs
of the SAME variant and one of the other - and it prints the signal beside the
floor, per field. A field that moves as often between two identical runs as it does
between the variants has not been shown to move at all.

It also reports the two WITHIN-pass statistics, which carry none of the cross-pass
sampling noise and are where the memory effect actually showed up: how many distinct
contingency triggers a pass declared, and how far ``pit_lap_target`` travelled
lap to lap.

Usage:
    python -m scripts.prompt_ab.analyse --floor-a pass_a.json --floor-b pass_a2.json \\
        --other pass_memory.json
"""

from __future__ import annotations

import argparse
import json
import re
from difflib import SequenceMatcher
from pathlib import Path

SCALAR_FIELDS = (
    "action",
    "confidence",
    "pit_lap_target",
    "compound_next",
    "undercut_target",
    "pace_mode",
    "target_lap_time_s",
    "risk_posture",
    "expected_stint_end",
)
ALL_FIELDS = (*SCALAR_FIELDS, "contingencies", "key_risks", "reasoning")


def _load(path: Path) -> tuple[dict, dict[int, dict]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload, {row["lap"]: row for row in payload["rows"]}


def _numeric_claims(text: str) -> set[str]:
    """The numbers a prose field asserts - its substance, as against its wording."""
    return set(re.findall(r"-?\d+\.?\d*", text or ""))


def _differs(left, right, field: str) -> tuple[bool, bool]:
    """Return (differs at all, differs in substance) for one field."""
    if field == "reasoning":
        return left != right, _numeric_claims(left) != _numeric_claims(right)
    if field == "contingencies":
        switches_left = sorted(c["switch_to"] for c in left)
        switches_right = sorted(c["switch_to"] for c in right)
        triggers_left = {c["trigger"].lower().strip() for c in left}
        triggers_right = {c["trigger"].lower().strip() for c in right}
        substantive = switches_left != switches_right or len(triggers_left) != len(triggers_right)
        return left != right, substantive
    if field == "key_risks":
        return left != right, len(left) != len(right)
    return left != right, left != right


def _count_differences(a: dict[int, dict], b: dict[int, dict], laps: list[int]) -> dict:
    return {
        field: sum(_differs(a[lap][field], b[lap][field], field)[1] for lap in laps)
        for field in ALL_FIELDS
    }


def _reasoning_similarity(a: dict[int, dict], b: dict[int, dict], laps: list[int]) -> float:
    ratios = sorted(
        SequenceMatcher(None, a[lap]["reasoning"], b[lap]["reasoning"]).ratio() for lap in laps
    )
    return ratios[len(ratios) // 2]


def _target_movement(rows: dict[int, dict], laps: list[int]) -> tuple[int, int, int]:
    """(lap-to-lap changes, comparable pairs, total absolute movement)."""
    values = [rows[lap]["pit_lap_target"] for lap in laps]
    pairs = [(x, y) for x, y in zip(values, values[1:]) if x is not None and y is not None]
    changes = sum(1 for x, y in pairs if x != y)
    movement = sum(abs(y - x) for x, y in pairs)
    return changes, len(pairs), movement


def _distinct_triggers(rows: dict[int, dict], laps: list[int]) -> tuple[int, int]:
    """(distinct triggers, total declarations) - the largest effect in the audit."""
    seen: set[str] = set()
    declarations = 0
    for lap in laps:
        for contingency in rows[lap]["contingencies"]:
            seen.add(contingency["trigger"].lower().strip())
            declarations += 1
    return len(seen), declarations


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--floor-a", required=True, type=Path, help="variant X, run 1")
    parser.add_argument("--floor-b", required=True, type=Path, help="variant X, run 2")
    parser.add_argument("--other", required=True, type=Path, help="the other variant")
    args = parser.parse_args()

    meta_a, rows_a = _load(args.floor_a)
    meta_b, rows_b = _load(args.floor_b)
    meta_o, rows_o = _load(args.other)
    laps = sorted(set(rows_a) & set(rows_b) & set(rows_o))

    print(
        "usage:",
        {
            p.name: m["usage"]
            for p, m in ((args.floor_a, meta_a), (args.floor_b, meta_b), (args.other, meta_o))
        },
    )
    print(f"\ncommon laps: {len(laps)}")

    floor = _count_differences(rows_a, rows_b, laps)
    signal = _count_differences(rows_a, rows_o, laps)

    print(f"\n{'field':<20}{'noise':>8}{'signal':>8}{'delta':>8}")
    print("-" * 44)
    for field in ALL_FIELDS:
        print(f"{field:<20}{floor[field]:>8}{signal[field]:>8}{signal[field] - floor[field]:>+8}")
    print(
        f"\nreasoning similarity  floor {_reasoning_similarity(rows_a, rows_b, laps):.4f}"
        f"   signal {_reasoning_similarity(rows_a, rows_o, laps):.4f}"
    )

    print("\nWITHIN-pass statistics (no cross-pass sampling noise)")
    print(f"{'pass':<28}{'target changes':>16}{'movement':>10}{'triggers':>10}{'declared':>10}")
    print("-" * 74)
    for name, rows in (
        (args.floor_a.name, rows_a),
        (args.floor_b.name, rows_b),
        (args.other.name, rows_o),
    ):
        changes, pairs, movement = _target_movement(rows, laps)
        distinct, declarations = _distinct_triggers(rows, laps)
        print(f"{name:<28}{f'{changes}/{pairs}':>16}{movement:>10}{distinct:>10}{declarations:>10}")


if __name__ == "__main__":
    main()
