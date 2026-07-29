"""``f1-eval`` console entry point: regenerate the eval reports.

Thin argparse dispatch over the three report builders. Each subcommand writes
its versioned markdown + JSON under ``documents/eval_reports/`` and prints a
one-line summary of where it landed and what it flagged.

    f1-eval registry       # E-08 consolidated metrics table
    f1-eval calibration    # E-03 reliability/Brier/ECE + quantile coverage
    f1-eval models         # reproduce headline numbers vs the model_configs
    f1-eval hygiene        # E-02 threshold provenance + leakage verdicts (#207)
    f1-eval nlp            # NLP per-stage eval: sentiment + gated stages (#304)
    f1-eval projection     # MC projection accuracy vs real stops + the measured tables
    f1-eval alert-llm      # PROXY alert precision via an LLM judge (#304; spends API calls)
    f1-eval decision-modes # does the stack pick the right lap to STOP (#708; takes minutes)
    f1-eval all            # every report EXCEPT the two opt-in ones below

Two commands stay out of ``all`` because they cost something a routine run should
not silently spend: ``alert-llm`` spends API calls, and ``decision-modes`` drives
the whole agent stack over hundreds of laps and takes minutes.
"""

from __future__ import annotations

import argparse
from typing import Any, Callable

from src.strategy.eval.alert_llm import build_alert_llm_report
from src.strategy.eval.calibration import build_calibration_report
from src.strategy.eval.decision_modes import build_decision_modes_report
from src.strategy.eval.hygiene import build_hygiene_report
from src.strategy.eval.nlp import build_nlp_report
from src.strategy.eval.projection import build_projection_report
from src.strategy.eval.registry import build_registry
from src.strategy.eval.reproduce import build_reproduction_report


def _summarise(payload: dict[str, Any], rows_key: str, flag_statuses: tuple[str, ...]) -> str:
    """One-line summary: report path + count of flagged rows."""
    rows = payload.get(rows_key, [])
    flagged = [r for r in rows if r.get("status") in flag_statuses]
    tail = f"; {len(flagged)} flagged ({', '.join(flag_statuses)})" if flag_statuses else ""
    return f"-> {payload['md_path']} ({len(rows)} rows{tail})"


def _run_registry() -> None:
    payload = build_registry()
    divergences = [e for e in payload["entries"] if not e["canonical"]]
    print(
        f"registry -> {payload['md_path']} ({len(payload['entries'])} entries; {len(divergences)} divergences reconciled)"
    )


def _run_calibration() -> None:
    payload = build_calibration_report()
    print("calibration " + _summarise(payload, "results", ("drift", "pending")))


def _run_models() -> None:
    payload = build_reproduction_report()
    print("models " + _summarise(payload, "results", ("delta", "pending")))


def _run_hygiene() -> None:
    payload = build_hygiene_report()
    findings = payload.get("findings", [])
    flagged = [f for f in findings if f.get("verdict") in ("contaminated", "underdocumented")]
    print(f"hygiene -> {payload['md_path']} ({len(findings)} items; {len(flagged)} flagged)")


def _run_nlp() -> None:
    payload = build_nlp_report()
    print("nlp " + _summarise(payload, "results", ("flagged", "delta", "blocked", "pending")))


def _run_projection() -> None:
    payload = build_projection_report()
    truth = payload["ground_truth"]
    accuracy = (
        "not measured (no data/raw)"
        if truth is None
        else f"{truth.within_one:.1%} within one position over {truth.sample_size} real stops"
    )
    print(f"projection -> {payload['md_path']} ({accuracy}; {len(payload['tables'])} tables)")


def _run_decision_modes() -> None:
    payload = build_decision_modes_report()
    agreement = payload["agreement"]
    accuracy = (
        "not measured (no data/raw)"
        if agreement is None
        else (
            f"{agreement.within_one:.1%} within one lap over {agreement.sample_size} "
            f"of {agreement.eligible} real stops"
        )
    )
    print(f"decision-modes -> {payload['md_path']} ({accuracy}; coverage {payload['status']})")


def _run_alert_llm() -> None:
    payload = build_alert_llm_report()
    result = payload["result"]
    precision = "-" if result["proxy_precision"] is None else f"{result['proxy_precision']:.4f}"
    print(
        f"alert-llm -> {payload['md_path']} (proxy precision {precision} over "
        f"{result['predicted_alerts']} predicted alerts; PROXY - needs human review)"
    )


_COMMANDS: dict[str, Callable[[], None]] = {
    "registry": _run_registry,
    "calibration": _run_calibration,
    "models": _run_models,
    "hygiene": _run_hygiene,
    "nlp": _run_nlp,
    "projection": _run_projection,
}


def main(argv: list[str] | None = None) -> int:
    """Parse args and run the requested report builder(s)."""
    parser = argparse.ArgumentParser(
        prog="f1-eval", description="F1 StratLab ML evaluation harness (#206)."
    )
    parser.add_argument(
        "command",
        choices=[*_COMMANDS, "alert-llm", "decision-modes", "all"],
        help="which report to regenerate",
    )
    args = parser.parse_args(argv)

    # Both of these are opt-in only, so `all` never runs them: alert-llm spends API
    # calls, and decision-modes drives the agent stack over hundreds of laps.
    if args.command == "alert-llm":
        _run_alert_llm()
        return 0
    if args.command == "decision-modes":
        _run_decision_modes()
        return 0

    commands = _COMMANDS.values() if args.command == "all" else [_COMMANDS[args.command]]
    for run in commands:
        run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
