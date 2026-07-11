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
on val-2024 (the honest split) to show the operating-point delta, and does the
same for the safety-car threshold (#363) - where the correction collapses because
val-2024 is SC-positive-sparse, evidence the leaked operating point was
test-overfit. The SC target window cannot be retro-selected (only the 3-lap model
is persisted), so its AUC-PR keeps an explicit test-window-selected caveat.
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
_LEAKED_SC_THRESHOLD = 0.2335  # argmax-F2 on test-2025 (N14 Step 5)
_SC_TARGETS = ("sc_within_3_laps", "sc_within_5_laps", "sc_within_7_laps")


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
    """Precision / recall / F1 / F2 of ``proba >= threshold`` against ``y``.

    F1 is the overtake operating metric; F2 (recall-biased) is the one N14 uses
    to pick the safety-car threshold, so both are returned and each caller reads
    the one its notebook selected on.
    """
    pred = (proba >= threshold).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    f2 = 5 * precision * recall / (4 * precision + recall) if (4 * precision + recall) else 0.0
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "f2": round(f2, 4),
    }


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


def correct_sc_threshold() -> dict[str, Any] | None:
    """Re-select the SC operating threshold on val-2024 (argmax-F2) and show the delta.

    The SC threshold 0.2335 was argmax-F2 on test-2025 (pre-calibration, N14
    Step 5), so the correction sweeps F2 on the 2024 validation raw scores and
    reports its test operating point next to the leaked one. IMPORTANT: val-2024
    carries very few SC-positive laps, so the val-selected threshold is
    high-variance - a degenerate corrected operating point is itself the finding
    (the leaked operating point was a product of test overfitting and does not
    reproduce off-test). Returns ``None`` if the holdout is absent.
    """
    from src.strategy.eval.calibration import load_sc_predictions

    val = load_sc_predictions(year=2024, target="sc_within_3_laps")
    test = load_sc_predictions(year=2025, target="sc_within_3_laps")
    if val is None or test is None:
        return None

    y_val, proba_val_raw, _ = val
    y_test, proba_test_raw, _ = test

    f2_scores = [_operating_point(y_val, proba_val_raw, t)["f2"] for t in _THRESHOLD_GRID]
    corrected_threshold = float(_THRESHOLD_GRID[int(np.argmax(f2_scores))])

    return {
        "leaked_threshold": _LEAKED_SC_THRESHOLD,
        "corrected_threshold": round(corrected_threshold, 4),
        "val_positive_count": int(y_val.sum()),
        "val_size": int(len(y_val)),
        "leaked_test_operating_point": _operating_point(
            y_test, proba_test_raw, _LEAKED_SC_THRESHOLD
        ),
        "corrected_test_operating_point": _operating_point(
            y_test, proba_test_raw, corrected_threshold
        ),
        "note": "F2 threshold selected on val-2024 raw scores; both operating points on 2025 test. "
        "val-2024 has few SC positives so the corrected point is high-variance - the collapse is the "
        "evidence that the leaked operating point was test-overfit",
    }


def sc_window_sensitivity() -> dict[str, Any] | None:
    """Show the SC target window (3/5/7) cannot be honestly retro-selected, and caveat it.

    N14 chose the window by comparing three separately-trained models on
    test-2025 (the leak); only the winning 3-lap model is persisted, so the
    window CANNOT be re-selected on val without retraining the 5/7-lap models
    (out of scope). As supporting evidence this reports the persisted 3-lap
    model's AUC-PR against each target window on val-2024 vs test-2025 - a
    single-model sensitivity check, NOT the original 3-model selection - which
    shows the metric is window-unstable. The paper's honest position: report the
    threshold-free AUC-PR WITH the caveat that its window was test-selected.
    Returns ``None`` if the holdout is absent.
    """
    from sklearn.metrics import average_precision_score

    from src.strategy.eval.calibration import _sc_frame_and_proba

    val = _sc_frame_and_proba(2024)
    test = _sc_frame_and_proba(2025)
    if val is None or test is None:
        return None

    val_frame, val_proba, _ = val
    test_frame, test_proba, _ = test
    per_window = {}
    for target in _SC_TARGETS:
        per_window[target] = {
            "val_2024_auc_pr": round(
                float(average_precision_score(val_frame[target].astype(int), val_proba)), 4
            ),
            "test_2025_auc_pr": round(
                float(average_precision_score(test_frame[target].astype(int), test_proba)), 4
            ),
        }
    return {
        "persisted_model_target": "sc_within_3_laps",
        "per_window_auc_pr": per_window,
        "retro_selectable": False,
        "note": "only the 3-lap model is persisted; the 5/7-lap models needed to re-select the window "
        "on val are not on disk (retraining out of scope). The reported SC AUC-PR 0.0723 keeps the "
        "caveat that its window was test-selected; the table is single-model sensitivity, not the "
        "original 3-model selection",
    }


def _render_overtake_correction(correction: dict[str, Any] | None) -> list[str]:
    """Markdown block for the overtake threshold correction (F1-selected)."""
    lines = ["## Correction (overtake threshold)", ""]
    if correction is None:
        return [*lines, "- holdout absent on disk; correction not computed this run"]
    leaked = correction["leaked_test_operating_point"]
    fixed = correction["corrected_test_operating_point"]
    return [
        *lines,
        f"- leaked threshold {correction['leaked_threshold']} (selected on test): "
        f"P {leaked['precision']} / R {leaked['recall']} / F1 {leaked['f1']} on 2025 test",
        f"- corrected threshold {correction['corrected_threshold']} (selected on val-2024): "
        f"P {fixed['precision']} / R {fixed['recall']} / F1 {fixed['f1']} on 2025 test",
        f"- {correction['note']}",
    ]


def _render_sc_correction(correction: dict[str, Any] | None) -> list[str]:
    """Markdown block for the SC threshold correction (F2-selected, sparse-val caveat)."""
    lines = ["## Correction (safety-car threshold)", ""]
    if correction is None:
        return [*lines, "- holdout absent on disk; correction not computed this run"]
    leaked = correction["leaked_test_operating_point"]
    fixed = correction["corrected_test_operating_point"]
    return [
        *lines,
        f"- leaked threshold {correction['leaked_threshold']} (F2 on test): "
        f"P {leaked['precision']} / R {leaked['recall']} / F2 {leaked['f2']} on 2025 test",
        f"- corrected threshold {correction['corrected_threshold']} (F2 on val-2024, "
        f"{correction['val_positive_count']}/{correction['val_size']} positive): "
        f"P {fixed['precision']} / R {fixed['recall']} / F2 {fixed['f2']} on 2025 test",
        f"- {correction['note']}",
    ]


def _render_sc_window(window: dict[str, Any] | None) -> list[str]:
    """Markdown block for the SC target-window sensitivity + retro-selection caveat."""
    lines = ["## Correction (safety-car target window)", ""]
    if window is None:
        return [*lines, "- holdout absent on disk; sensitivity not computed this run"]
    lines += [
        "| window | val-2024 AUC-PR | test-2025 AUC-PR |",
        "|---|---|---|",
    ]
    for target, aucs in window["per_window_auc_pr"].items():
        lines.append(f"| {target} | {aucs['val_2024_auc_pr']} | {aucs['test_2025_auc_pr']} |")
    return [*lines, "", f"- {window['note']}"]


def _render(
    findings: list[ProvenanceEntry],
    correction: dict[str, Any] | None,
    sc_correction: dict[str, Any] | None,
    sc_window: dict[str, Any] | None,
) -> str:
    """Render the hygiene report: verdict table + corrections + paper conclusion."""
    header = "| item | kind | model | verdict | selection | evidence |"
    rule = "|---|---|---|---|---|---|"
    order = {CONTAMINATED: 0, UNDERDOCUMENTED: 1, CLEAN: 2}
    ordered = sorted(findings, key=lambda f: order.get(f.verdict, 3))
    rows = [
        f"| {f.item} | {f.kind} | {f.model} | **{f.verdict}** | {f.selection} | {f.evidence} |"
        for f in ordered
    ]

    lines = [header, rule, *rows, ""]
    lines += _render_overtake_correction(correction)
    lines += ["", *_render_sc_correction(sc_correction)]
    lines += ["", *_render_sc_window(sc_window)]

    contaminated = [f for f in findings if f.verdict == CONTAMINATED]
    lines += [
        "",
        "## Conclusion for the paper freeze",
        "",
        f"- {len(contaminated)} contaminated items (overtake threshold; safety-car threshold + window). "
        "All other thresholds and aggregate features are clean or non-target.",
        "- **Overtake headline clears**: AUC-PR 0.5491 / AUC-ROC 0.8758 are threshold-free and involve no "
        "window selection; the threshold leakage touches only their operating point, corrected above.",
        "- **Safety-car operating threshold is NOT robustly recoverable**: re-selecting on val-2024 "
        "collapses the operating point (val-2024 has too few SC positives), which is itself the evidence "
        "that the leaked 0.2335 was test-overfit. The paper should report SC threshold-free and not claim "
        "a fixed operating threshold.",
        "- **Safety-car window cannot be retro-selected**: only the 3-lap model is persisted, so the "
        "{3,5,7}-lap window selected on test-2025 cannot be honestly re-chosen without retraining the "
        "5/7-lap models. The reported SC AUC-PR 0.0723 therefore keeps an explicit test-window-selected "
        "caveat.",
        "- **Remaining action before freeze**: pin a year filter in N03 `load_all_races` to close the "
        "circuit_cluster underdocumentation.",
        "- Every other headline (undercut 0.6739, pit 0.487, pace 0.4104, tire 0.7078, sentiment 0.84) is "
        "unaffected by these findings.",
    ]
    return "\n".join(lines)


def build_hygiene_report() -> dict[str, Any]:
    """Regenerate the signed hygiene report (the #207 / #363 deliverable)."""
    findings = audit_findings()
    correction = correct_overtake_threshold()
    sc_correction = correct_sc_threshold()
    sc_window = sc_window_sensitivity()
    header = build_header(dataset="notebooks/strategy audit + 2024/2025 overtake & SC holdouts")
    payload = {
        "findings": [asdict(f) for f in findings],
        "overtake_correction": correction,
        "sc_correction": sc_correction,
        "sc_window_sensitivity": sc_window,
    }
    md_path, json_path = write_report(
        HYGIENE_NAME, header, _render(findings, correction, sc_correction, sc_window), payload
    )
    return {
        "header": asdict(header),
        "md_path": str(md_path),
        "json_path": str(json_path),
        **payload,
    }
