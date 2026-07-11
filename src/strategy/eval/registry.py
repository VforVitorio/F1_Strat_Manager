"""E-08 consolidated metrics registry: the single citable source of truth.

Today the headline numbers live scattered across ``model_config`` JSONs,
notebook cells, thesis chapters and memory files, and at least one diverges
(pace MAE 0.392 notebook-era vs 0.4104 thesis-final). Every downstream
document (IEEE paper, AEPIA 5-pager, docs site, model cards) copies numbers by
hand from that mess. This module regenerates ONE versioned table so the drift
ends here.

Two kinds of entry:

- **config-sourced** - the model pins its own numbers in a ``model_config``
  JSON (overtake, undercut, pit); the registry just surfaces them verbatim.
  ``reproduce.py`` is what re-derives and checks they still reproduce.
- **doc-pinned** - the number is NOT in any config (pace, tire-deg,
  sentiment); it lives in the thesis (Tabla 6.1) / IEEE report. The
  registry records the canonical (thesis-final) value AND the divergent
  published/notebook-era value, so the reconciliation is documented in one
  place forever. Docs reconciliation (#213) propagates the canonical from here;
  provenance *verification* of these splits is Phase 2 (#207), not this phase.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any

from src.f1_strat_manager.data_cache import get_models_root
from src.strategy.eval.report import build_header, write_report

REGISTRY_NAME = "metrics_registry"


@dataclass
class MetricEntry:
    """One headline metric consolidated into the registry.

    Fields:
    - ``model`` / ``family`` - which predictor and its problem type.
    - ``metric`` / ``value`` / ``threshold`` - the number and its decision
      threshold (``None`` for regressors).
    - ``split`` - the train/test season split behind the number.
    - ``source`` - where the number came from (config path or thesis).
    - ``canonical`` - True for the citable authority value; a False entry is a
      superseded/divergent value kept only to document the reconciliation.
    - ``note`` - provenance / divergence commentary.
    """

    model: str
    family: str
    metric: str
    value: float
    threshold: float | None
    split: str
    source: str
    canonical: bool
    note: str


def _config_entries() -> list[MetricEntry]:
    """Read headline metrics straight from the ``model_config`` JSONs.

    These models pin their own numbers, so the registry surfaces them without
    re-deriving. The broken pit P05-P95 coverage (0.7047 vs 0.90 nominal) is
    published inside its config too - the registry does not hide it; the
    calibration report is what flags it as drift.
    """
    models = get_models_root()

    overtake = json.loads(
        (models / "overtake_probability" / "model_config.json").read_text(encoding="utf-8")
    )
    undercut = json.loads(
        (models / "pit_prediction" / "model_config_undercut_v1.json").read_text(encoding="utf-8")
    )
    pit = json.loads((models / "pit_prediction" / "model_config.json").read_text(encoding="utf-8"))
    sc = json.loads(
        (models / "safety_car_probability" / "feature_list_v1.json").read_text(encoding="utf-8")
    )

    ov_split = f"train {overtake['train_seasons']} / test {overtake['test_season']}"
    uc_split = f"train {undercut['train_years']} / test {undercut['test_year']}"
    pit_split = f"train {pit['train_years']} / test {pit['test_year']}"
    sc_split = f"train {sc['train_years']} / test {sc['test_year']}"

    return [
        MetricEntry(
            "overtake",
            "classification",
            "auc_pr_test",
            overtake["auc_pr_test"],
            overtake["optimal_threshold"],
            ov_split,
            "config: overtake_probability/model_config.json",
            True,
            f"LightGBM + Platt(val 2024); auc_roc {overtake['auc_roc_test']}",
        ),
        MetricEntry(
            "undercut",
            "classification",
            "auc_pr_test",
            undercut["metrics"]["auc_pr_test"],
            undercut["best_threshold"],
            uc_split,
            "config: pit_prediction/model_config_undercut_v1.json",
            True,
            f"LightGBM + Platt(val 2024); baseline_pr {undercut['metrics']['baseline_pr']}",
        ),
        MetricEntry(
            "pit_duration",
            "quantile",
            "p50_mae_test_s",
            pit["eval"]["p50_mae_test"],
            None,
            pit_split,
            "config: pit_prediction/model_config.json",
            True,
            f"HistGBT P05/P50/P95; baseline_mae {pit['eval']['baseline_mae_test']}; "
            f"P05-P95 coverage {pit['eval']['p05_p95_coverage_test']} vs 0.90 nominal (see calibration report)",
        ),
        MetricEntry(
            "safety_car",
            "classification",
            "auc_pr_test",
            sc["metrics"]["test_auc_pr"],
            sc["best_threshold"],
            sc_split,
            "config: safety_car_probability/feature_list_v1.json",
            True,
            f"LightGBM + Platt(val 2024); baseline {sc['metrics']['baseline_auc_pr']}; "
            f"3-lap lift {sc['target_comparison']['3-lap']['lift']}; target sc_within_3_laps",
        ),
    ]


def _doc_pinned_entries() -> list[MetricEntry]:
    """Encode the metrics that live in the thesis, not in any config.

    Each divergent number produces two rows: the canonical (thesis-final) row
    and the superseded row, so the reconciliation is self-documenting. Numbers
    are sourced to the thesis Tabla 6.1 / IEEE report and issue #213; the
    precise season splits for these are marked "verify in #207" rather than
    asserted, because split provenance is exactly Phase 2's job.
    """
    verify = "split provenance pending (#207)"
    entries: list[MetricEntry] = []

    # Pace / lap-time delta (XGBoost, N06) - the canonical 0.392 -> 0.4104 case.
    entries.append(
        MetricEntry(
            "pace",
            "regression",
            "mae_test_s",
            0.4104,
            None,
            "test 2025",
            "thesis Tabla 6.1 (final)",
            True,
            "XGBoost delta lap time; " + verify,
        )
    )
    entries.append(
        MetricEntry(
            "pace",
            "regression",
            "mae_test_s",
            0.392,
            None,
            "test 2025",
            "notebook N06 (superseded)",
            False,
            "notebook-era value; superseded by thesis-final 0.4104",
        )
    )

    # Tire degradation (TCN + MC Dropout, N07-10) - global MAE; best compound C2.
    entries.append(
        MetricEntry(
            "tire_degradation",
            "regression",
            "mae_test_s",
            0.7078,
            None,
            "test 2025",
            "thesis Tabla 6.1 (global)",
            True,
            "TireDegTCN; per-compound best C2 0.5501s; missed R2 target; " + verify,
        )
    )

    # Radio sentiment (RoBERTa, N20) - the 87.5% -> 0.84 case.
    entries.append(
        MetricEntry(
            "sentiment",
            "nlp",
            "accuracy",
            0.84,
            None,
            "held-out radio set",
            "thesis Tabla 6.1 (final)",
            True,
            "RoBERTa; macro-F1 0.75; NLP harness in #304",
        )
    )
    entries.append(
        MetricEntry(
            "sentiment",
            "nlp",
            "accuracy",
            0.875,
            None,
            "held-out radio set",
            "published-era (superseded)",
            False,
            "87.5% published; superseded by thesis-final 0.84",
        )
    )

    return entries


def collect_entries() -> list[MetricEntry]:
    """All registry entries: config-sourced first, then doc-pinned."""
    return _config_entries() + _doc_pinned_entries()


def _render_table(entries: list[MetricEntry]) -> str:
    """Render the registry as a markdown table + a divergences section."""
    header = "| model | metric | value | thr | split | canonical | source |"
    rule = "|---|---|---|---|---|---|---|"
    rows = []
    for entry in entries:
        thr = "-" if entry.threshold is None else f"{entry.threshold:g}"
        mark = "yes" if entry.canonical else "no"
        rows.append(
            f"| {entry.model} | {entry.metric} | {entry.value:g} | {thr} | "
            f"{entry.split} | {mark} | {entry.source} |"
        )

    divergences = [e for e in entries if not e.canonical]
    div_lines = ["", "## Divergences reconciled", ""]
    if divergences:
        for entry in divergences:
            canonical = next(
                c
                for c in entries
                if c.canonical and c.model == entry.model and c.metric == entry.metric
            )
            div_lines.append(
                f"- **{entry.model} {entry.metric}**: {entry.value:g} "
                f"({entry.source}) -> **{canonical.value:g}** ({canonical.source})"
            )
    else:
        div_lines.append("- none")

    return "\n".join([header, rule, *rows, *div_lines])


def build_registry() -> dict[str, Any]:
    """Regenerate the consolidated registry and write the versioned report.

    Returns the payload dict (also written to
    ``documents/eval_reports/metrics_registry.json``). This is the single
    citable source that replaces the scattered numbers.
    """
    entries = collect_entries()
    models = get_models_root()
    artifacts = {
        "overtake_cfg": models / "overtake_probability" / "model_config.json",
        "undercut_cfg": models / "pit_prediction" / "model_config_undercut_v1.json",
        "pit_cfg": models / "pit_prediction" / "model_config.json",
        "sc_cfg": models / "safety_car_probability" / "feature_list_v1.json",
    }
    header = build_header(dataset="model_configs + thesis Tabla 6.1", artifacts=artifacts)
    payload = {"entries": [asdict(e) for e in entries]}

    md_path, json_path = write_report(REGISTRY_NAME, header, _render_table(entries), payload)
    return {
        "header": asdict(header),
        "md_path": str(md_path),
        "json_path": str(json_path),
        **payload,
    }


def load_registry() -> dict[str, Any]:
    """Read the committed registry JSON back (for #213 / #304 / tests).

    Raises ``FileNotFoundError`` if the registry has never been generated, so
    a stale consumer fails loudly instead of silently using no data.
    """
    from src.strategy.eval.report import eval_reports_dir

    path = eval_reports_dir() / f"{REGISTRY_NAME}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"registry not generated yet; run `f1-eval registry` (looked in {path})"
        )
    return json.loads(path.read_text(encoding="utf-8"))
