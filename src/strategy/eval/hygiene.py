"""E-02 threshold-provenance + aggregate-feature leakage verification (#207).

Gates the IEEE paper freeze: rules out test-2025 contamination of the reported
numbers. Every tuned decision threshold and every historical aggregate feature
gets a verdict (clean / contaminated / underdocumented) traced to the training
cell that produced it. The verdicts are the product of a read-only audit of
``notebooks/strategy/`` (UNTOUCHABLE - this module only records the findings and
demonstrates the correction; it does not modify a notebook).

KEY CONCLUSION FOR THE PAPER: the two contaminated items (overtake threshold
0.7976, safety-car threshold 0.2335 + its 3/5/7-lap window) were selected on
the 2025 test set, so they inflate only the OPERATING-POINT metrics
(precision/recall/F1 at that threshold). The threshold-FREE headline numbers the
paper actually reports - AUC-PR and AUC-ROC - are computed from probabilities
and are unaffected, so they clear. This module re-selects the overtake threshold
on val-2024 (the honest split) to show the operating-point delta.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from src.strategy.eval.calibration import load_overtake_predictions
from src.strategy.eval.report import build_header, write_report

HYGIENE_NAME = "hygiene"

# Verdicts (fixed vocabulary; kept as module constants so the report writer and
# the golden test share one spelling).
CLEAN = "clean"
CONTAMINATED = "contaminated"
UNDERDOCUMENTED = "underdocumented"

# Overtake threshold sweep grid, mirrored from N12 Step 5 so the re-selection
# uses the same candidate set the notebook did.
_THRESHOLD_GRID = np.linspace(0.05, 0.92, 200)
_LEAKED_OVERTAKE_THRESHOLD = 0.7976


@dataclass
class ProvenanceEntry:
    """One threshold/feature provenance verdict.

    ``kind`` is ``threshold`` or ``aggregate_feature``; ``verdict`` is one of the
    module constants; ``selection`` names the split/window the value was chosen
    or computed on; ``evidence`` points at the training notebook + cell; and
    ``impact`` states what the verdict means for the paper's numbers.
    """

    item: str
    kind: str
    model: str
    verdict: str
    selection: str
    evidence: str
    impact: str


def audit_findings() -> list[ProvenanceEntry]:
    """The read-only provenance audit of ``notebooks/strategy/`` (issue #207).

    Adversarially verified: the two contaminated threshold verdicts were
    confirmed against the exact selection cells; every aggregate feature was
    checked for a fit-on-train / past-season-only implementation.
    """
    return [
        ProvenanceEntry(
            "optimal_threshold=0.7976",
            "threshold",
            "overtake",
            CONTAMINATED,
            "argmax-F1 on the 2025 test set",
            "N12_overtake_model.ipynb (threshold-analysis / step-5 cell)",
            "operating-point P/R/F1 optimistic; AUC-PR 0.5491 / AUC-ROC 0.8758 unaffected (threshold-free)",
        ),
        ProvenanceEntry(
            "best_threshold=0.2335 + 3/5/7-lap window",
            "threshold",
            "safety_car",
            CONTAMINATED,
            "argmax-F2 threshold AND the best target-window both on the 2025 test set",
            "N14_sc_model.ipynb (PR-curve + window-comparison cells)",
            "threshold -> operating-point optimistic; window chosen by max-lift on test -> the headline "
            "AUC-PR 0.0723 is itself optimistic (a max over 3 candidates on test), NOT clean",
        ),
        ProvenanceEntry(
            "best_threshold=0.522",
            "threshold",
            "undercut",
            CLEAN,
            "argmax-F1 on the calibrated val-2024 split",
            'N16_undercut.ipynb ("Threshold on calibrated val 2024")',
            "textbook-correct; the 2025 test set is only applied afterwards",
        ),
        ProvenanceEntry(
            "circuit_sc_rate",
            "aggregate_feature",
            "safety_car",
            CLEAN,
            "past-season only (year < yr); 2023 rows get a fixed SC_PRIOR=0.15",
            "N13_sc_eda.ipynb (compute_circuit_sc_rate)",
            "no test-2025 in its own aggregate",
        ),
        ProvenanceEntry(
            "team_year_median",
            "aggregate_feature",
            "pit_duration",
            CLEAN,
            "lookup fit on train only, applied to test with recent-year fallback",
            "N15_pit_duration.ipynb (add_team_year_median)",
            "companion circuit_traversal / baseline are likewise train-fit",
        ),
        ProvenanceEntry(
            "circuit_undercut_rate + team_x_undercut_rate",
            "aggregate_feature",
            "undercut",
            CLEAN,
            "target encoding fit on train only, train-mean fallback on test",
            "N16_undercut.ipynb (compute_target_encoding)",
            "split precedes the encoding; no train<-test leak",
        ),
        ProvenanceEntry(
            "circuit_cluster (k-means)",
            "aggregate_feature",
            "overtake/safety_car/laptime",
            UNDERDOCUMENTED,
            "k-means fit window not year-restricted in code; 2025 holdout is intent-only",
            "N03_circuit_clustering.ipynb (load_all_races / fit_kmeans_final)",
            "mild NON-target risk (unsupervised geometry bucket); pin a year filter to close",
        ),
        ProvenanceEntry(
            "year_circuit_median / team_pace_rank",
            "aggregate_feature",
            "laptime",
            CLEAN,
            "within-session per-year aggregates; flagged LEAKY and removed from the reported model",
            "N06_laptime_model.ipynb (add_context_features / FEATURES_PROD)",
            "not in the reported IEEE delta model; 2025 held out until final eval",
        ),
    ]


def _operating_point(y: np.ndarray, proba: np.ndarray, threshold: float) -> dict[str, float]:
    """Precision / recall / F1 of ``proba >= threshold`` against ``y``."""
    pred = (proba >= threshold).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4)}


def correct_overtake_threshold() -> dict[str, Any] | None:
    """Re-select the overtake threshold on val-2024 and show the operating-point delta.

    The E-02 correction for the worst contaminated item: pick the threshold by
    argmax-F1 on the 2024 validation slice (never touching test), then report its
    operating point on the 2025 test set next to the leaked 0.7976's operating
    point on the same test set. The leaked threshold's F1 is maximal on test by
    construction (it was fit there); the corrected threshold's F1 is the honest
    number. Returns ``None`` if the holdout is absent.
    """
    val = load_overtake_predictions(year=2024)
    test = load_overtake_predictions(year=2025)
    if val is None or test is None:
        return None

    y_val, proba_val_raw, _ = val
    y_test, proba_test_raw, _ = test

    f1s = [_operating_point(y_val, proba_val_raw, t)["f1"] for t in _THRESHOLD_GRID]
    corrected_threshold = float(_THRESHOLD_GRID[int(np.argmax(f1s))])

    return {
        "leaked_threshold": _LEAKED_OVERTAKE_THRESHOLD,
        "corrected_threshold": round(corrected_threshold, 4),
        "leaked_test_operating_point": _operating_point(
            y_test, proba_test_raw, _LEAKED_OVERTAKE_THRESHOLD
        ),
        "corrected_test_operating_point": _operating_point(
            y_test, proba_test_raw, corrected_threshold
        ),
        "note": "threshold selected on val-2024; both operating points evaluated on the 2025 test set",
    }


def _render(findings: list[ProvenanceEntry], correction: dict[str, Any] | None) -> str:
    """Render the hygiene report: verdict table + correction + paper conclusion."""
    header = "| item | kind | model | verdict | selection | evidence |"
    rule = "|---|---|---|---|---|---|"
    order = {CONTAMINATED: 0, UNDERDOCUMENTED: 1, CLEAN: 2}
    ordered = sorted(findings, key=lambda f: order.get(f.verdict, 3))
    rows = [
        f"| {f.item} | {f.kind} | {f.model} | **{f.verdict}** | {f.selection} | {f.evidence} |"
        for f in ordered
    ]

    lines = [header, rule, *rows, "", "## Correction (overtake threshold)", ""]
    if correction is None:
        lines.append("- holdout absent on disk; correction not computed this run")
    else:
        leaked = correction["leaked_test_operating_point"]
        fixed = correction["corrected_test_operating_point"]
        lines += [
            f"- leaked threshold {correction['leaked_threshold']} (selected on test): "
            f"P {leaked['precision']} / R {leaked['recall']} / F1 {leaked['f1']} on 2025 test",
            f"- corrected threshold {correction['corrected_threshold']} (selected on val-2024): "
            f"P {fixed['precision']} / R {fixed['recall']} / F1 {fixed['f1']} on 2025 test",
            f"- {correction['note']}",
        ]

    contaminated = [f for f in findings if f.verdict == CONTAMINATED]
    lines += [
        "",
        "## Conclusion for the paper freeze",
        "",
        f"- {len(contaminated)} contaminated items (overtake threshold; safety-car threshold + window). "
        "All other thresholds and aggregate features are clean or non-target.",
        "- **Overtake headline clears**: AUC-PR 0.5491 / AUC-ROC 0.8758 are threshold-free and involve no "
        "window selection; the threshold leakage touches only their operating point.",
        "- **Safety-car headline is optimistic, NOT clean**: AUC-PR 0.0723 is the max-lift window among "
        "{3,5,7} laps selected on test-2025, so the reported number is itself selection-biased (a max over "
        "3 candidates on test), on top of the operating-point threshold leakage.",
        "- **Action before freeze**: (1) re-select the overtake + SC operating thresholds on val-2024 "
        "(the undercut N16 pattern) - the overtake correction above shows the honest operating point; "
        "(2) re-select the SC target window on the CV/val split (not test) and re-report SC AUC-PR, or "
        "caveat it as test-selected; (3) pin a year filter in N03 `load_all_races` to close the "
        "circuit_cluster underdocumentation.",
        "- Every other headline (undercut 0.6739, pit 0.487, pace 0.4104, tire 0.7078, sentiment 0.84) is "
        "unaffected by these findings.",
    ]
    return "\n".join(lines)


def build_hygiene_report() -> dict[str, Any]:
    """Regenerate the signed hygiene report (the #207 deliverable)."""
    findings = audit_findings()
    correction = correct_overtake_threshold()
    header = build_header(dataset="notebooks/strategy audit + 2024/2025 overtake holdout")
    payload = {"findings": [asdict(f) for f in findings], "overtake_correction": correction}
    md_path, json_path = write_report(HYGIENE_NAME, header, _render(findings, correction), payload)
    return {
        "header": asdict(header),
        "md_path": str(md_path),
        "json_path": str(json_path),
        **payload,
    }
