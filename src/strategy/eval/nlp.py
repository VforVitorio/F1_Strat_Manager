"""NLP evaluation harness on the shared eval package (issue #304).

Per-stage evaluation of the radio NLP pipeline, built ON ``src/strategy/eval``
(NOT a parallel harness - it reuses report.py's header + writer). This phase
reproduces the RoBERTa sentiment stage on a fixed labeled set; the other stages
are gated with their exact blocker, the same honest-delta convention the ML-eval
harness (#206) uses.

Stage status this phase:
- **sentiment** (RoBERTa) - reproduced over the fixed 530-row labeled radio set
  (accuracy + macro-F1), compared to the thesis-final 0.84 / macro-F1 0.75. The
  labeled CSV is the full set (train+test), so the number is expected to run
  optimistic vs the held-out 0.84; the held-out split is not pinned on disk.
- **intent** (SetFit) - BLOCKED: setfit does not import under transformers 5.3.0
  (#303), so the intent stage cannot be reproduced until that is fixed.
- **NER / RCM** - pending: their model reconstruction is not wired this phase.
- **alert precision** (the MoE-routing signal the orchestrator depends on) -
  pending a labeled alert ground-truth set; none exists on disk today. This is
  the metric #304 most wants and it is a data-collection task, not a code one.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any

from src.f1_strat_manager.data_cache import get_data_root, get_models_root
from src.strategy.eval.report import build_header, write_report

NLP_NAME = "nlp"
_SENTIMENT_ACC = 0.84  # thesis-final RoBERTa accuracy
_SENTIMENT_MACRO_F1 = 0.75  # thesis-final macro-F1
_TOLERANCE = 0.03
_SENTIMENT_BATCH = 32


@dataclass
class StageResult:
    """One NLP stage measurement.

    ``status`` is ``reproduced`` (within tolerance of the thesis number),
    ``delta`` (measured but diverges - e.g. full-set vs held-out), ``blocked``
    (a dependency prevents the stage from running at all), or ``pending`` (not
    wired / no ground-truth this phase).
    """

    stage: str
    metric: str
    value: float | None
    reference: float | None
    status: str
    detail: str


def _load_roberta_sentiment() -> tuple[Any, Any, str, list[str], int] | None:
    """Reconstruct the RoBERTa sentiment classifier from its Lightning ``.ckpt``.

    The checkpoint stores a ``model.roberta.* / model.classifier.*`` state dict
    (a ``RobertaForSequenceClassification`` wrapped by a Lightning module); we
    strip the ``model.`` prefix and load it into a fresh HF head configured from
    ``model_config.json``. Returns ``(model, tokenizer, device, names, max_len)``
    or ``None`` when the artifacts are absent.
    """
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    model_dir = get_models_root() / "nlp" / "sentiment_classifier_v1"
    ckpt = model_dir / "best_roberta_sentiment.ckpt"
    cfg_path = model_dir / "model_config.json"
    if not (ckpt.exists() and cfg_path.exists()):
        return None

    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg["model_name"], num_labels=cfg["num_labels"]
    )
    state = torch.load(ckpt, map_location="cpu", weights_only=False)["state_dict"]
    prefix = "model."
    stripped = {
        k[len(prefix) :]: v
        for k, v in state.items()
        if k.startswith(prefix + "roberta") or k.startswith(prefix + "classifier")
    }
    model.load_state_dict(stripped, strict=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()
    return model, tokenizer, device, cfg["sentiment_names"], cfg["max_length"]


def reproduce_sentiment() -> list[StageResult]:
    """Reproduce RoBERTa sentiment accuracy + macro-F1 on the fixed labeled set.

    Returns a ``pending`` row instead of raising when the checkpoint or the
    labeled CSV is absent, so a partial install still reports.
    """
    import torch
    from sklearn.metrics import accuracy_score, f1_score

    loaded = _load_roberta_sentiment()
    labeled = get_data_root() / "processed" / "radio_nlp" / "radio_labeled_data.csv"
    if loaded is None or not labeled.exists():
        return [
            StageResult(
                "sentiment",
                "accuracy",
                None,
                _SENTIMENT_ACC,
                "pending",
                "roberta ckpt or radio_labeled_data.csv absent",
            )
        ]

    import pandas as pd

    model, tokenizer, device, names, max_len = loaded
    df = pd.read_csv(labeled)
    texts = df["radio_message"].astype(str).tolist()
    y_true = df["sentiment"].tolist()

    preds: list[str] = []
    with torch.no_grad():
        for start in range(0, len(texts), _SENTIMENT_BATCH):
            batch = texts[start : start + _SENTIMENT_BATCH]
            enc = tokenizer(
                batch, truncation=True, padding=True, max_length=max_len, return_tensors="pt"
            ).to(device)
            idx = model(**enc).logits.argmax(dim=1).tolist()
            preds.extend(names[j] for j in idx)

    accuracy = float(accuracy_score(y_true, preds))
    macro_f1 = float(f1_score(y_true, preds, average="macro"))
    caveat = (
        "full labeled set (train+test); optimistic vs the held-out 0.84 (split not pinned on disk)"
    )
    return [
        StageResult(
            "sentiment",
            "accuracy",
            round(accuracy, 4),
            _SENTIMENT_ACC,
            _status(accuracy, _SENTIMENT_ACC),
            f"n={len(y_true)}; {caveat}",
        ),
        StageResult(
            "sentiment",
            "macro_f1",
            round(macro_f1, 4),
            _SENTIMENT_MACRO_F1,
            _status(macro_f1, _SENTIMENT_MACRO_F1),
            f"n={len(y_true)}; 3-class",
        ),
    ]


def _status(value: float, reference: float) -> str:
    """``reproduced`` when within tolerance of the thesis number, else ``delta``."""
    return "reproduced" if abs(value - reference) <= _TOLERANCE else "delta"


def _gated_stages() -> list[StageResult]:
    """The stages that cannot run this phase, each with its exact blocker."""
    return [
        StageResult(
            "intent",
            "accuracy",
            None,
            None,
            "blocked",
            "setfit does not import under transformers 5.3.0 (#303); stage cannot run",
        ),
        StageResult(
            "ner",
            "entity_f1",
            None,
            None,
            "pending",
            "GLiNER/BERT reconstruction not wired this phase",
        ),
        StageResult(
            "rcm",
            "accuracy",
            None,
            None,
            "pending",
            "RCM classifier reconstruction not wired this phase",
        ),
        StageResult(
            "alert_precision",
            "precision",
            None,
            None,
            "pending",
            "no labeled alert ground-truth on disk (the MoE-routing metric #304 targets; data-collection task)",
        ),
    ]


def collect_results() -> list[StageResult]:
    """All NLP stage results: reproduced sentiment first, then gated stages."""
    return [*reproduce_sentiment(), *_gated_stages()]


def _render(results: list[StageResult]) -> str:
    """Render the NLP eval as a markdown table, runnable stages first."""
    order = {"delta": 0, "reproduced": 1, "blocked": 2, "pending": 3}
    ordered = sorted(results, key=lambda r: order.get(r.status, 4))
    header = "| stage | metric | value | reference | status | detail |"
    rule = "|---|---|---|---|---|---|"
    rows = []
    for r in ordered:
        value = "-" if r.value is None else f"{r.value:.4f}"
        reference = "-" if r.reference is None else f"{r.reference:g}"
        rows.append(f"| {r.stage} | {r.metric} | {value} | {reference} | {r.status} | {r.detail} |")
    return "\n".join([header, rule, *rows])


def build_nlp_report() -> dict[str, Any]:
    """Regenerate the NLP eval report (the #304 deliverable)."""
    results = collect_results()
    ckpt = get_models_root() / "nlp" / "sentiment_classifier_v1" / "best_roberta_sentiment.ckpt"
    header = build_header(
        dataset="radio_labeled_data.csv (fixed set)", artifacts={"roberta_sentiment": ckpt}
    )
    payload = {"results": [asdict(r) for r in results]}
    md_path, json_path = write_report(NLP_NAME, header, _render(results), payload)
    return {
        "header": asdict(header),
        "md_path": str(md_path),
        "json_path": str(json_path),
        **payload,
    }
