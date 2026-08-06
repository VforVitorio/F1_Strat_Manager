"""Render the 2025 LLM-mode measurement into a markdown report plus its JSON.

Consumes the JSONL files `scripts/measure_llm_windows.py` writes and scores them
with `src/strategy/eval/llm_decision.py`, which imports the deterministic tier's
definitions rather than restating them. The paired deterministic arm is scored
from its own JSONL over the SAME windows, so the two columns are comparable by
construction instead of by assertion.

Every rate is printed with the population it describes on the same line. That is
not decoration: the sample is a seeded draw over rail-eligible green-flag stops of
nine races, so a bare percentage here would be read as a season figure and it is
not one.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.strategy.eval.llm_decision import measure

REPO_ROOT = Path(__file__).resolve().parents[1]

# Read 2026-08-06 from the providers' published pricing. Kept here, next to the
# date, rather than inside the meter: a price is a fact about a day.
_PRICE_PER_MTOK = {
    "gpt-4.1-mini": (0.40, 1.60),
    "gpt-5.4-mini": (0.75, 4.50),
}


def _rate(part: int, whole: int) -> str:
    return f"{part}/{whole} ({part / whole:.1%})" if whole else f"{part}/0 (n/a)"


def _bucket_table(buckets: dict[str, int], total: int) -> list[str]:
    lines = ["| bucket | stops | share of eligible |", "| --- | --- | --- |"]
    for name, count in sorted(buckets.items(), key=lambda kv: -kv[1]):
        lines.append(
            f"| `{name}` | {count} | {count / total:.1%} |"
            if total
            else f"| `{name}` | {count} | n/a |"
        )
    return lines


def _paired_rows(llm: dict, det: dict) -> list[str]:
    """Verdict-by-verdict contrast on the windows both arms actually measured."""
    key = lambda v: (v["race"], v["driver"], v["actual_lap"])  # noqa: E731
    llm_by = {key(v): v for v in llm["verdicts"]}
    det_by = {key(v): v for v in det["verdicts"]}
    shared = sorted(set(llm_by) & set(det_by))

    agree_bucket = sum(1 for k in shared if llm_by[k]["bucket"] == det_by[k]["bucket"])
    both_scored = [k for k in shared if llm_by[k]["bucket"] == det_by[k]["bucket"] == "scored"]
    same_lap = sum(1 for k in both_scored if llm_by[k]["chosen_lap"] == det_by[k]["chosen_lap"])

    det_scored_llm_not = [
        k for k in shared if det_by[k]["bucket"] == "scored" and llm_by[k]["bucket"] != "scored"
    ]
    llm_scored_det_not = [
        k for k in shared if llm_by[k]["bucket"] == "scored" and det_by[k]["bucket"] != "scored"
    ]
    det_exact_llm_lost = [k for k in det_scored_llm_not if det_by[k]["offset_laps"] == 0]

    lines = [
        "",
        "## Paired contrast: the same stops, both profiles",
        "",
        "The only legitimate LLM-vs-deterministic comparison. Both arms ran the identical",
        "windows through the identical harness; only `profile` differs. The published",
        "`decision_modes.md` numbers are NOT comparable to these: different sample.",
        "",
        f"- stops measured by both arms: **{len(shared)}**",
        f"- same bucket in both arms: **{_rate(agree_bucket, len(shared))}**",
        f"- scored by both arms: **{len(both_scored)}**, of which same chosen lap: "
        f"**{_rate(same_lap, len(both_scored))}**",
        f"- **deterministic scored, LLM did not: {len(det_scored_llm_not)}** "
        f"(of these, {len(det_exact_llm_lost)} were EXACT agreements the LLM path lost)",
        f"- LLM scored, deterministic did not: {len(llm_scored_det_not)}",
        "",
    ]
    if det_exact_llm_lost:
        lines += [
            "Exact agreements present in the deterministic arm and absent in the LLM arm:",
            "",
            "| race | driver | real stop lap | LLM bucket |",
            "| --- | --- | --- | --- |",
        ]
        lines += [
            f"| {race} | {driver} | {lap} | `{llm_by[(race, driver, lap)]['bucket']}` |"
            for race, driver, lap in det_exact_llm_lost
        ]
        lines.append("")
    return lines


def _cost_lines(llm: dict) -> list[str]:
    tokens = llm["tokens"]
    laps = llm["laps_measured"]
    hours = llm["wall_clock_seconds"] / 3600
    lines = [
        "",
        "## What it cost",
        "",
        f"- **{laps} evaluated laps**, {llm['windows']} windows",
        f"- **{hours:.2f} h** inside `run_lap` (excludes per-process boot)",
        f"- {tokens['calls']} LLM calls, {tokens['prompt']:,} prompt + "
        f"{tokens['completion']:,} completion tokens",
        f"- {tokens['calls'] / laps:.1f} calls/lap, "
        f"{(tokens['prompt'] + tokens['completion']) / laps:,.0f} tokens/lap, "
        f"{llm['wall_clock_seconds'] / laps:.2f} s/lap",
        "",
        "Prices read 2026-08-06: `gpt-4.1-mini` $0.40/$1.60 per 1M, `gpt-5.4-mini` "
        "$0.75/$4.50 per 1M. Zero prompt tokens are cacheable here (measured), so the "
        "prompt bill is paid in full on every lap.",
        "",
    ]
    return lines


def _regime_lines(llm: dict) -> list[str]:
    """Both readings of the two contested population questions, from one measurement.

    Whether the wet race and the directive-forced stops belong inside the headline
    is a reporting choice, not a run-time one: the rows are the same either way.
    Printing both is strictly more honest than picking one and saying so.
    """
    wet = {"Silverstone"}
    forced = {("Lusail", 32)}
    inside = llm["verdicts"]
    core = [
        v for v in inside if v["race"] not in wet and (v["race"], v["actual_lap"]) not in forced
    ]

    def summary(verdicts: list[dict]) -> str:
        scored = [v for v in verdicts if v["bucket"] == "scored"]
        if not verdicts:
            return "no stops"
        exact = sum(1 for v in scored if v["offset_laps"] == 0)
        within_one = sum(1 for v in scored if abs(v["offset_laps"]) <= 1)
        no_call = sum(1 for v in verdicts if v["bucket"] == "no_call_in_window")
        return (
            f"{len(verdicts)} eligible, {len(scored)} scored "
            f"({len(scored) / len(verdicts):.1%}), exact {_rate(exact, len(scored))}, "
            f"within 1 {_rate(within_one, len(scored))}, "
            f"declined {_rate(no_call, len(verdicts))}"
        )

    lines = [
        "",
        "## Both readings of the population, from the same rows",
        "",
        "| reading | result |",
        "| --- | --- |",
        f"| **all drawn stops** (wet + directive-forced included) | {summary(inside)} |",
        f"| dry, freely-chosen stops only (Silverstone and the Lusail L32 trio removed) "
        f"| {summary(core)} |",
        "",
        "Neither is more correct. The first says *2025 as it happened*, including a wet race "
        "and a set of stops forced by a Pirelli maximum-stint directive the system cannot "
        "observe (the directive is not in the RAG corpus, which holds season rulebooks only; "
        "the 25-lap Qatar limit is confirmed by Pirelli's own press release). The second says "
        "*the population the system was built for*. Quoting either without naming it is the "
        "error.",
        "",
    ]
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--llm", required=True, nargs="+", help="LLM-arm JSONL file(s)")
    parser.add_argument("--det", required=True, nargs="+", help="Deterministic-arm JSONL file(s)")
    parser.add_argument("--out", default="documents/eval_reports/llm_2025/REPORT.md")
    parser.add_argument("--year", type=int, default=2025)
    args = parser.parse_args()

    raw_root = REPO_ROOT / "data/raw"
    merged: dict[str, dict] = {}
    for label, paths in (("llm", args.llm), ("det", args.det)):
        combined = REPO_ROOT / f"documents/eval_reports/llm_2025/_merged_{label}.jsonl"
        combined.write_text(
            "".join(Path(p).read_text(encoding="utf-8") for p in paths), encoding="utf-8"
        )
        merged[label] = measure(combined, args.year, raw_root)

    llm, det = merged["llm"], merged["det"]
    agree = llm["agreement"]

    lines = [
        "# 2025 season, LLM mode: what the shipped system recommends",
        "",
        "Generated by `scripts/report_llm_2025.py`. Sample and protocol: "
        "`documents/audits/MEASUREMENT_2025_METHODOLOGY.md`. Session record: "
        "`documents/audits/MEASUREMENT_SESSION_2025_LOG.md`.",
        "",
        "**Population, stated once and applying to every rate below:** rail-eligible "
        "green-flag pit stops of the 2025 season, drawn by a seeded uniform draw "
        "(`seed 20250806`, at most 3 windows per race-lap) from nine races covering all four "
        "circuit clusters. It is **not** a season-wide figure and it is not comparable to the "
        "published `decision_modes.md` numbers, whose sample is different.",
        "",
        "## Headline",
        "",
        "| metric | LLM (`rich`) | deterministic (`no-llm`) |",
        "| --- | --- | --- |",
        f"| stops eligible | {agree['eligible']} | {det['agreement']['eligible']} |",
        f"| stops scored | {_rate(agree['scored'], agree['eligible'])} | "
        f"{_rate(det['agreement']['scored'], det['agreement']['eligible'])} |",
        f"| coverage verdict | **{agree['coverage_verdict']}** | "
        f"**{det['agreement']['coverage_verdict']}** |",
        f"| exact lap | {agree['exact']:.1%} | {det['agreement']['exact']:.1%} |",
        f"| within 1 lap | {agree['within_one']:.1%} | {det['agreement']['within_one']:.1%} |",
        f"| within 2 laps | {agree['within_two']:.1%} | {det['agreement']['within_two']:.1%} |",
        f"| mean signed error | {agree['mean_signed_error']:+.2f} | "
        f"{det['agreement']['mean_signed_error']:+.2f} |",
        "",
        "`mean signed error` is **not quotable as a system property**: it moves with "
        "`DECISION_WINDOW_LAPS`. It is here so the direction of the bias is visible.",
        "",
        "### Buckets, LLM arm",
        "",
    ]
    lines += _bucket_table(llm["buckets"], agree["eligible"])
    lines += ["", "### Buckets, deterministic arm", ""]
    lines += _bucket_table(det["buckets"], det["agreement"]["eligible"])
    lines += _paired_rows(llm, det)
    lines += _regime_lines(llm)

    census = llm["llm_fields"]
    if census:
        lines += [
            "",
            "## What the LLM actually filled in",
            "",
            "Eleven of these fields do not exist on the deterministic path, which emits an "
            "argmax and a fixed reasoning string. Whether the planner populates them at all "
            "is a result.",
            "",
            f"- actions: `{census['actions']}`",
            f"- pace modes: `{census['pace_modes']}`",
            f"- risk postures: `{census['risk_postures']}`",
            f"- confidence: mean {census['confidence']['mean']}, median "
            f"{census['confidence']['median']}, range "
            f"[{census['confidence']['min']}, {census['confidence']['max']}] over "
            f"{census['confidence']['n']} laps",
            f"- fields populated (of {census['laps']} laps): `{census['fields_populated']}`",
            "",
        ]

    repeats = llm["repeats"]
    if repeats.get("laps_with_repeats"):
        lines += [
            "",
            "## Stability across repeat passes",
            "",
            "The orchestrator requests `temperature=0` and `gpt-5.4-mini` discards it, so the "
            "path is not deterministic. A pre-committed rule, registered before these numbers "
            "existed: if repeated windows disagree on the verdict for more than 20% of "
            "repeats, the single-pass headline is not quotable on its own.",
            "",
            f"- laps measured more than once: **{repeats['laps_with_repeats']}**",
            "",
            "| field | laps differing | share |",
            "| --- | --- | --- |",
        ]
        lines += [
            f"| `{name}` | {count} | {repeats['share_differing'][name]:.1%} |"
            for name, count in repeats["laps_differing"].items()
        ]
        lines.append("")

    lines += _cost_lines(llm)

    out_path = REPO_ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out_path.with_suffix(".json")).write_text(
        json.dumps({"llm": llm, "deterministic": det}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"wrote {out_path}")
    print(json.dumps({"llm": agree, "deterministic": det["agreement"]}, indent=2))


if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(REPO_ROOT))
    main()
