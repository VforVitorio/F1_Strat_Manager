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
  ~0 B-F1 are flagged untrustworthy, separately from the NER score below.
- **ner** (BERT-bio) - entity-level micro-F1 reproduced with seqeval over the
  annotation set, vs the frozen 0.4151 headline (#304).
- **rcm** - the N23 rule-based classifier is ported into the harness and run
  over the persisted 2025 RCM corpus; coverage (1 - OTHER-rate) per category vs
  the frozen config (#304).
- **alert precision** - pending a labeled alert ground-truth set (#304 phase 3).
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
_NER_PROD_MODEL = "bert_bio_v1"  # production NER model under ner_v1/
_NER_HEADLINE_F1 = 0.4151  # frozen bert_large_conll03_bio entity-F1 (model_config)
_NER_MAX_LEN = 128  # must match the N22 training config
_DEAD_NER_F1 = 0.15  # entity types with frozen-eval B-F1 below this are flagged dead (NR-07)
_RCM_CORPUS_GLOB = "race_radios/2025/*/rcm.parquet"  # persisted FastF1 RCM corpus


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


def _load_ner_model() -> tuple[Any, Any, dict[int, str], str] | None:
    """Load the production BERT-bio NER model (additive replica of radio_agent).

    Mirrors ``radio_agent._load_ner_model`` but stays in the harness because
    importing ``src.agents`` loads every NLP model at import time. The
    checkpoint's classifier head is 19-label BIO; ``from_pretrained`` warns about
    the base model's 9-label head, then ``load_state_dict`` overwrites it with
    the trained weights. Returns ``(tokenizer, model, id2label, device)`` or
    ``None`` when the artifacts are absent.
    """
    import torch
    from transformers import AutoTokenizer, BertForTokenClassification

    ner_dir = get_models_root() / "nlp" / _NER_DIR / _NER_PROD_MODEL
    cfg_path = ner_dir / "model_config.json"
    state_path = ner_dir / "bert_bio_state_dict.pt"
    if not (cfg_path.exists() and state_path.exists()):
        return None

    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    label2id = cfg["label2id"]
    id2label = {int(k): v for k, v in cfg["id2label"].items()}
    base = cfg.get("model_name", "dbmdz/bert-large-cased-finetuned-conll03-english")
    tokenizer = AutoTokenizer.from_pretrained(str(ner_dir), use_fast=True)
    model = BertForTokenClassification.from_pretrained(
        base, num_labels=len(label2id), ignore_mismatched_sizes=True
    )
    model.load_state_dict(torch.load(state_path, map_location="cpu", weights_only=False))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()
    return tokenizer, model, id2label, device


def _ner_predicted_tags(
    words: list[str], tokenizer: Any, model: Any, id2label: dict[int, str], device: str
) -> list[str]:
    """Word-level BIO tags for a pre-split word list (replica of the radio_agent decode).

    Keeps only the first subword tag per word, mirroring
    ``radio_agent._decode_word_tags`` so the harness scores the exact tags the
    production pipeline would emit.
    """
    import torch

    enc = tokenizer(
        words,
        is_split_into_words=True,
        add_special_tokens=True,
        max_length=_NER_MAX_LEN,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    word_ids = enc.word_ids(batch_index=0)
    with torch.no_grad():
        logits = (
            model(
                input_ids=enc["input_ids"].to(device),
                attention_mask=enc["attention_mask"].to(device),
            )
            .logits[0]
            .cpu()
        )
    predicted_ids = logits.argmax(dim=-1).tolist()
    first_tag_per_word: dict[int, str] = {}
    for token_i, word_i in enumerate(word_ids):
        if word_i is not None and word_i not in first_tag_per_word:
            first_tag_per_word[word_i] = id2label.get(predicted_ids[token_i], "O")
    return [first_tag_per_word.get(i, "O") for i in range(len(words))]


def _gold_bio_tags(text: str, entities: list) -> tuple[list[str], list[str]]:
    """Convert char-offset gold entities to word-level BIO over ``text.split()``.

    ``predict_entities`` splits on whitespace, so the gold tags must use the same
    word boundaries to line up with the model's per-word predictions. A word is
    tagged with an entity when its character span overlaps the gold span (B- on
    the first overlapping word, I- after). Returns ``(words, gold_tags)``.
    """
    words = text.split()
    offsets = []
    cursor = 0
    for word in words:
        start = text.index(word, cursor)
        offsets.append((start, start + len(word)))
        cursor = start + len(word)

    tags = ["O"] * len(words)
    for entity in entities:
        if not (isinstance(entity, (list, tuple)) and len(entity) == 3):
            continue
        char_start, char_end, label = entity
        opening = True
        for word_i, (word_start, word_end) in enumerate(offsets):
            if word_start < char_end and word_end > char_start:
                tags[word_i] = ("B-" if opening else "I-") + str(label)
                opening = False
    return words, tags


def _iter_ner_annotations() -> list[tuple[str, list]]:
    """``(text, gold_entities)`` pairs from the annotation set (radio_message as text).

    Records whose ``annotations`` payload is malformed keep their text with an
    empty gold set, so they still contribute to precision (a spurious prediction
    on a no-entity message is a false positive).
    """
    path = get_data_root() / "processed" / "radio_nlp" / "f1_radio_entity_annotations.json"
    if not path.exists():
        return []
    records = json.loads(path.read_text(encoding="utf-8"))
    pairs = []
    for record in records:
        text = str(record.get("radio_message", ""))
        annotation = record.get("annotations")
        well_formed = (
            isinstance(annotation, list)
            and len(annotation) == 2
            and isinstance(annotation[1], dict)
        )
        entities = annotation[1].get("entities", []) if well_formed else []
        if text.strip():
            pairs.append((text, entities))
    return pairs


def reproduce_ner() -> list[StageResult]:
    """Reproduce BERT-bio entity-level micro-F1 on the annotation set (seqeval).

    Full-set score (train+test), so it runs slightly optimistic vs the frozen
    0.4151 headline; the 4 NR-07 dead classes stay in the score but are flagged
    separately. Returns a ``pending`` row when the model or annotations are
    absent.
    """
    from seqeval.metrics import f1_score, precision_score, recall_score

    loaded = _load_ner_model()
    pairs = _iter_ner_annotations()
    if loaded is None or not pairs:
        return [
            StageResult(
                "ner",
                "entity_f1",
                None,
                _NER_HEADLINE_F1,
                "pending",
                "ner model or annotations absent",
            )
        ]

    tokenizer, model, id2label, device = loaded
    gold_seqs, pred_seqs = [], []
    for text, entities in pairs:
        words, gold = _gold_bio_tags(text, entities)
        if not words:
            continue
        gold_seqs.append(gold)
        pred_seqs.append(_ner_predicted_tags(words, tokenizer, model, id2label, device))

    micro_f1 = float(f1_score(gold_seqs, pred_seqs))
    precision = float(precision_score(gold_seqs, pred_seqs))
    recall = float(recall_score(gold_seqs, pred_seqs))
    dead, _ = dead_ner_classes()
    detail = (
        f"n={len(gold_seqs)}; micro entity-F1 (P {precision:.3f}/R {recall:.3f}); full-set optimistic; "
        f"{len(dead)} dead classes flagged separately"
    )
    return [
        StageResult(
            "ner",
            "entity_f1",
            round(micro_f1, 4),
            _NER_HEADLINE_F1,
            _status(micro_f1, _NER_HEADLINE_F1),
            detail,
        )
    ]


def _classify_rcm_event(category: str, flag: Any, scope: Any, sector: Any, message: str) -> str:
    """Rule-based RCM event classifier ported from N23 (``_classify_event``).

    Reads the persisted lowercase RCM schema (category/flag/scope/sector/
    message). The branch logic is identical to the N23 notebook parser and the
    inlined ``radio_agent`` copy; only the field access is adapted to the
    on-disk column names. Kept in the harness because importing ``radio_agent``
    loads every NLP model.

    --- WHERE TO CHANGE IF THE PARSER CHANGES ---
    ``notebooks/nlp/N23_rcm_parser.ipynb`` (_classify_event) is the source of
    truth; mirror any edit there into this port and the radio_agent copy.
    """
    import pandas as pd

    cat = str(category).strip()
    flag_up = str(flag).strip().upper()
    msg = str(message).upper()

    if cat == "SafetyCar":
        if "VIRTUAL" in msg:
            return (
                "VIRTUAL_SAFETY_CAR_DEPLOYED" if "DEPLOYED" in msg else "VIRTUAL_SAFETY_CAR_ENDING"
            )
        if "DEPLOYED" in msg:
            return "SAFETY_CAR_DEPLOYED"
        if "PIT LANE" in msg or "IN THIS LAP" in msg:
            return "SAFETY_CAR_IN_PIT_LANE"
        if "ENDING" in msg or "WITHDRAWN" in msg:
            return "SAFETY_CAR_ENDING"
        return "OTHER"

    if cat == "Flag":
        if flag_up == "CHEQUERED" or "CHEQUERED" in msg:
            return "CHEQUERED_FLAG"
        if flag_up == "BLUE":
            return "BLUE_FLAG"
        if flag_up == "BLACK AND WHITE":
            return "BLACK_AND_WHITE_FLAG"
        if flag_up in ("VIRTUAL_SAFETY_CAR", "VSC"):
            return "VIRTUAL_SAFETY_CAR_DEPLOYED"
        if flag_up == "SAFETY_CAR":
            return "SAFETY_CAR_DEPLOYED"
        if flag_up == "RED" or "RED FLAG" in msg:
            return "RED_FLAG"
        if flag_up == "GREEN" or "GREEN FLAG" in msg:
            return "GREEN_FLAG"
        if flag_up == "CLEAR":
            return "CLEAR_FLAG"
        if flag_up in ("YELLOW", "DOUBLE YELLOW"):
            if str(scope).strip() == "Sector" or pd.notna(sector):
                return "YELLOW_FLAG_SECTOR"
            return "YELLOW_FLAG"
        return "OTHER"

    if cat == "Drs":
        return "DRS_ENABLED" if "ENABLED" in msg else "DRS_DISABLED"

    if cat == "CarEvent":
        if "RETIRED" in msg or "ABANDON" in msg:
            return "CAR_RETIRED"
        if "COLLISION" in msg or "CONTACT" in msg:
            return "CAR_COLLISION"
        if "MECHANICAL" in msg or "ENGINE" in msg or "GEARBOX" in msg:
            return "CAR_MECHANICAL"
        return "OTHER"

    if cat == "Other":
        if "DRS ENABLED" in msg:
            return "DRS_ENABLED"
        if "DRS DISABLED" in msg:
            return "DRS_DISABLED"
        if (
            "TRACK LIMITS" in msg
            or "TIME DELETED" in msg
            or "LAP DELETED" in msg
            or "DELETED" in msg
        ):
            return "LAP_DELETED"
        if "UNDER INVESTIGATION" in msg or "FIA STEWARDS" in msg or "NOTED" in msg:
            return "INVESTIGATION"
        if "PENALTY" in msg or ("TIME" in msg and "SECOND" in msg):
            return "TIME_PENALTY"
        if "PIT EXIT" in msg or "PIT LANE" in msg:
            return "PIT_EXIT"
        track_surface = "TRACK" in msg and (
            "CONDITION" in msg or "SLIPPERY" in msg or "SURFACE" in msg
        )
        other_surface = any(k in msg for k in ("DEBRIS", "FLUID", "LOW GRIP", "RAIN", "AWNING"))
        if track_surface or other_surface:
            return "TRACK_CONDITION"
        if "LAPPED" in msg and "OVERTAKE" in msg:
            return "LAPPED_CARS_OVERTAKE"
        if "ALL CARS MAY OVERTAKE" in msg:
            return "SAFETY_CAR_ENDING"
        return "OTHER"

    return "OTHER"


def reproduce_rcm() -> list[StageResult]:
    """Reproduce the RCM parser coverage (1 - OTHER-rate) per FastF1 category.

    Runs the ported rule-based classifier over the persisted 2025 RCM corpus and
    reports overall + per-category coverage against the frozen config
    (Flag 1.0 / Other 0.928 / Drs 1.0 / SafetyCar 1.0). The corpus (all 2025
    races) differs from the N23 sample, so the numbers are close but not
    identical. Returns a ``pending`` row when the corpus is absent.
    """
    from collections import Counter

    import pandas as pd

    corpus = sorted((get_data_root() / "processed").glob(_RCM_CORPUS_GLOB))
    if not corpus:
        return [
            StageResult("rcm", "coverage", None, None, "pending", "2025 rcm corpus absent on disk")
        ]

    total: Counter = Counter()
    other: Counter = Counter()
    for path in corpus:
        df = pd.read_parquet(path)
        for _, row in df.iterrows():
            category = str(row["category"]).strip()
            event = _classify_rcm_event(
                row["category"], row["flag"], row["scope"], row["sector"], row["message"]
            )
            total[category] += 1
            if event == "OTHER":
                other[category] += 1

    n = sum(total.values())
    coverage = 1 - sum(other.values()) / n
    per_category = ", ".join(f"{cat} {1 - other[cat] / total[cat]:.3f}" for cat in sorted(total))
    detail = (
        f"n={n} across {len(corpus)} 2025 races; per-category [{per_category}]; "
        "vs config Flag 1.0/Other 0.928/Drs 1.0/SafetyCar 1.0 (N23 sample differs)"
    )
    return [StageResult("rcm", "coverage", round(coverage, 4), None, "reproduced", detail)]


def _status(value: float, reference: float) -> str:
    """``reproduced`` when within tolerance of the reference number, else ``delta``."""
    return "reproduced" if abs(value - reference) <= _TOLERANCE else "delta"


def _gated_stages() -> list[StageResult]:
    """The stages that cannot run this phase, each with its exact blocker (#304)."""
    return [
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
        *reproduce_ner(),
        *reproduce_rcm(),
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
        "ner_bert_bio": models / "nlp" / _NER_DIR / _NER_PROD_MODEL / "bert_bio_state_dict.pt",
    }
    header = build_header(
        dataset="radio + intent labeled CSVs, entity annotations, 2025 RCM corpus",
        artifacts=artifacts,
    )
    payload = {"results": [asdict(r) for r in results]}
    md_path, json_path = write_report(NLP_NAME, header, _render(results), payload)
    return {
        "header": asdict(header),
        "md_path": str(md_path),
        "json_path": str(json_path),
        **payload,
    }
