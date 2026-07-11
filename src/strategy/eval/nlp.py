"""NLP evaluation harness on the shared eval package (issues #303, #304).

Per-stage evaluation of the radio NLP pipeline, built ON ``src/strategy/eval``
(NOT a parallel harness - it reuses report.py's header + writer). This phase
reproduces the sentiment and intent stages on fixed labeled sets, records the
two #303 hygiene verdicts, and gates the stages not yet wired.

Stage status this phase:
- **sentiment** (RoBERTa) - reproduced over the fixed 530-row labeled radio set
  (accuracy + macro-F1), compared to the thesis-final 0.84 / macro-F1 0.75. The
  labeled CSV is the full set (train+test), so the number runs optimistic vs the
  held-out 0.84; the held-out split is not pinned on disk.
- **intent** (SetFit ModernBERT) - reproduced SETFIT-FREE (#303). The deployed
  model is a SentenceTransformer body + a pickled sklearn head; ``import setfit``
  fails under transformers 5.3.0 (it imports ``default_logdir``, removed in 5.x),
  so the harness classifies with the two pieces directly. accuracy + macro/
  weighted-F1 on the full labeled set (optimistic vs the published 0.5934 test
  weighted-F1).
- **NR-08** (intent predict_proba order) - the production ``radio_agent`` decodes
  ``predict_proba`` with the hardcoded ``intent_names`` tuple; verified aligned
  with the head's class order, so there is no confidence swap (a ``flagged`` row
  would mean a live bug). ``src/agents`` is untouchable; the harness only records
  the verdict.
- **NR-07** (dead NER classes) - entity types the frozen NER eval predicts at
  ~0 B-F1 are flagged untrustworthy; the #304 NER stage re-measures them.
- **NER F1 / RCM / alert precision** - pending (#304 phases).
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
_INTENT_PUBLISHED_WEIGHTED_F1 = 0.5934  # deployed SetFit ModernBERT test weighted-F1 (model_config)
_TOLERANCE = 0.03
_SENTIMENT_BATCH = 32
_INTENT_DIR = "intent_setfit_modernbert_v1"
_NER_DIR = "ner_v1"
_DEAD_NER_F1 = 0.15  # entity types with frozen-eval B-F1 below this are flagged dead (NR-07)


@dataclass
class StageResult:
    """One NLP stage measurement or hygiene verdict.

    ``status`` is ``reproduced`` (a metric within tolerance of the reference, or
    a hygiene verdict that clears), ``delta`` (measured but diverges - e.g.
    full-set vs held-out), ``flagged`` (a hygiene finding the paper must
    surface - NR-07 dead classes, an NR-08 swap), ``blocked`` (a dependency
    prevents the stage from running at all), or ``pending`` (not wired / no
    ground-truth this phase).
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


def _load_intent_setfit_free() -> tuple[Any, Any, dict[int, str]] | None:
    """Load the intent classifier WITHOUT importing setfit (#303 blocker).

    The deployed intent model is a SentenceTransformer body + a pickled sklearn
    ``LogisticRegression`` head. ``import setfit`` fails under transformers 5.3.0
    (it imports ``default_logdir``, removed in 5.x), so the harness reconstructs
    the two pieces directly: encode with the SentenceTransformer, classify with
    the head. Returns ``(sentence_transformer, head, idx_to_name)`` or ``None``
    when the artifacts are absent.
    """
    import joblib
    from sentence_transformers import SentenceTransformer

    model_dir = get_models_root() / "nlp" / _INTENT_DIR
    head_path = model_dir / "model_head.pkl"
    cfg_path = model_dir / "model_config.json"
    if not (head_path.exists() and cfg_path.exists() and (model_dir / "config.json").exists()):
        return None

    mapping = json.loads(cfg_path.read_text(encoding="utf-8"))["intent_mapping"]
    idx_to_name = {v: k for k, v in mapping.items()}
    st = SentenceTransformer(str(model_dir))
    head = joblib.load(head_path)
    return st, head, idx_to_name


def reproduce_intent() -> list[StageResult]:
    """Reproduce intent accuracy + macro/weighted-F1 on the fixed labeled set.

    Setfit-free (see ``_load_intent_setfit_free``). The labeled CSV is the full
    train+test set, so the numbers run optimistic vs the deployed model's
    published test weighted-F1 (0.5934); the held-out split is not pinned on
    disk. Only ``weighted_f1`` has a published anchor; accuracy + macro-F1 are
    new measurements with no thesis reference. Returns a ``pending`` row when the
    artifacts or CSV are absent.
    """
    from sklearn.metrics import accuracy_score, f1_score

    loaded = _load_intent_setfit_free()
    labeled = get_data_root() / "processed" / "radio_nlp" / "intent_labeled_data.csv"
    if loaded is None or not labeled.exists():
        return [
            StageResult(
                "intent",
                "accuracy",
                None,
                None,
                "pending",
                "intent model or intent_labeled_data.csv absent",
            )
        ]

    import pandas as pd

    st, head, idx_to_name = loaded
    df = pd.read_csv(labeled)
    texts = df["message"].astype(str).tolist()
    y_true = df["intent"].tolist()
    embeddings = st.encode(texts, show_progress_bar=False)
    preds = [idx_to_name[int(c)] for c in head.predict(embeddings)]

    accuracy = float(accuracy_score(y_true, preds))
    macro_f1 = float(f1_score(y_true, preds, average="macro"))
    weighted_f1 = float(f1_score(y_true, preds, average="weighted"))
    n = len(y_true)
    caveat = "full labeled set (train+test); optimistic vs the published 0.5934 test weighted-F1"
    return [
        StageResult(
            "intent", "accuracy", round(accuracy, 4), None, "reproduced", f"n={n}; {caveat}"
        ),
        StageResult(
            "intent",
            "macro_f1",
            round(macro_f1, 4),
            None,
            "reproduced",
            f"n={n}; 5-class, no anchor",
        ),
        StageResult(
            "intent",
            "weighted_f1",
            round(weighted_f1, 4),
            _INTENT_PUBLISHED_WEIGHTED_F1,
            _status(weighted_f1, _INTENT_PUBLISHED_WEIGHTED_F1),
            f"n={n}; vs deployed test weighted-F1 (full-set optimistic)",
        ),
    ]


def verify_intent_column_order() -> StageResult:
    """NR-08: verify the production ``predict_proba`` decode reads the right class.

    ``radio_agent.predict_intent`` decodes ``predict_proba(...)[label_idx]`` with
    ``label_idx`` from the hardcoded ``intent_names`` order. That is correct only
    if the head's ``classes_`` columns line up with ``intent_names`` via
    ``intent_mapping``. This loads ONLY the 30 KB head + config (not the 568 MB
    encoder) and checks the alignment column-by-column. ``reproduced`` = fully
    aligned (no confidence swap); ``flagged`` = a live confidence-swap bug.
    """
    import joblib

    model_dir = get_models_root() / "nlp" / _INTENT_DIR
    head_path = model_dir / "model_head.pkl"
    cfg_path = model_dir / "model_config.json"
    if not (head_path.exists() and cfg_path.exists()):
        return StageResult(
            "intent", "predict_proba_order", None, 1.0, "pending", "intent head/config absent"
        )

    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    intent_names = cfg["intent_names"]
    inverse = {v: k for k, v in cfg["intent_mapping"].items()}
    classes = [int(c) for c in joblib.load(head_path).classes_]

    if len(classes) != len(intent_names):
        return StageResult(
            "intent",
            "predict_proba_order",
            0.0,
            1.0,
            "flagged",
            f"head has {len(classes)} classes vs {len(intent_names)} intent_names",
        )

    aligned = sum(1 for pos, name in enumerate(intent_names) if inverse.get(classes[pos]) == name)
    fraction = aligned / len(intent_names)
    swap_note = "no swap" if aligned == len(intent_names) else "CONFIDENCE SWAP - live bug"
    return StageResult(
        "intent",
        "predict_proba_order",
        round(fraction, 4),
        1.0,
        "reproduced" if aligned == len(intent_names) else "flagged",
        f"head.classes_={classes} map to intent_names via intent_mapping; "
        f"{aligned}/{len(intent_names)} columns aligned ({swap_note})",
    )


def dead_ner_classes() -> tuple[list[str], StageResult]:
    """NR-07: flag NER entity types the frozen model predicts at ~0 B-F1.

    Annotation support is balanced, so "dead" is a MODEL failure, not a data
    gap: the production NER model_config's per-class B-F1 (from the training
    eval) exposes types the model never gets right. Those must not be surfaced
    as reliable; the #304 NER stage re-measures them. Returns
    ``(dead_type_names, flag_row)``.
    """
    cfg_path = get_models_root() / "nlp" / _NER_DIR / "model_config.json"
    if not cfg_path.exists():
        return [], StageResult(
            "ner", "dead_classes", None, None, "pending", "ner model_config absent"
        )

    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    per_class = cfg["results"]["bert_large_conll03_bio"]["per_class_f1"]
    dead = []
    for etype in cfg["entity_types"]:
        b_f1 = per_class.get("B-" + etype.upper().replace(" ", "_"))
        if b_f1 is not None and b_f1 < _DEAD_NER_F1:
            dead.append(etype)

    detail = (
        f"{len(dead)} entity types with frozen-eval B-F1 < {_DEAD_NER_F1} "
        f"({', '.join(dead) or 'none'}); suppressed as untrustworthy, re-measured in #304"
    )
    return dead, StageResult("ner", "dead_classes", float(len(dead)), 0.0, "flagged", detail)


def _status(value: float, reference: float) -> str:
    """``reproduced`` when within tolerance of the reference number, else ``delta``."""
    return "reproduced" if abs(value - reference) <= _TOLERANCE else "delta"


def _gated_stages() -> list[StageResult]:
    """The stages that cannot run this phase, each with its exact blocker (#304)."""
    return [
        StageResult(
            "ner",
            "entity_f1",
            None,
            None,
            "pending",
            "BERT-bio entity-level F1 reproduction not wired this phase (#304)",
        ),
        StageResult(
            "rcm",
            "accuracy",
            None,
            None,
            "pending",
            "RCM parser reproduction not wired this phase (#304)",
        ),
        StageResult(
            "alert_precision",
            "precision",
            None,
            None,
            "pending",
            "no labeled alert ground-truth on disk (the MoE-routing metric #304 targets; data task)",
        ),
    ]


def collect_results() -> list[StageResult]:
    """All NLP stage results: reproduced stages, the #303 hygiene verdicts, then gated stages."""
    _dead, dead_row = dead_ner_classes()
    return [
        *reproduce_sentiment(),
        *reproduce_intent(),
        verify_intent_column_order(),
        dead_row,
        *_gated_stages(),
    ]


def _render(results: list[StageResult]) -> str:
    """Render the NLP eval as a markdown table, findings + runnable stages first."""
    order = {"flagged": 0, "delta": 1, "reproduced": 2, "blocked": 3, "pending": 4}
    ordered = sorted(results, key=lambda r: order.get(r.status, 5))
    header = "| stage | metric | value | reference | status | detail |"
    rule = "|---|---|---|---|---|---|"
    rows = []
    for r in ordered:
        value = "-" if r.value is None else f"{r.value:.4f}"
        reference = "-" if r.reference is None else f"{r.reference:g}"
        rows.append(f"| {r.stage} | {r.metric} | {value} | {reference} | {r.status} | {r.detail} |")
    return "\n".join([header, rule, *rows])


def build_nlp_report() -> dict[str, Any]:
    """Regenerate the NLP eval report (the #303/#304 deliverable)."""
    results = collect_results()
    models = get_models_root()
    artifacts = {
        "roberta_sentiment": models
        / "nlp"
        / "sentiment_classifier_v1"
        / "best_roberta_sentiment.ckpt",
        "intent_head": models / "nlp" / _INTENT_DIR / "model_head.pkl",
    }
    header = build_header(
        dataset="radio_labeled_data.csv + intent_labeled_data.csv (fixed sets)", artifacts=artifacts
    )
    payload = {"results": [asdict(r) for r in results]}
    md_path, json_path = write_report(NLP_NAME, header, _render(results), payload)
    return {
        "header": asdict(header),
        "md_path": str(md_path),
        "json_path": str(json_path),
        **payload,
    }
