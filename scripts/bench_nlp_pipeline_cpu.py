"""NLP pipeline CPU latency benchmark, replicates N24 inline.

Loads the three N24 NLP models (sentiment, intent, NER) using the
exact same loaders the notebook uses, then times two entry points
(``run_pipeline`` for team-radio strings, ``run_rcm_pipeline`` for
race-control rows) on a fixed bank of 8 representative messages. The
benchmark intentionally inlines the loaders rather than importing
from the notebook so that the only Python dependency at runtime is
``src.nlp`` and the model artefacts themselves.

Use ``--device both`` to emit one row per entry point on each device
when CUDA is available; ``cpu`` and ``cuda`` produce a single device
configuration each.

Usage::

    uv run scripts/bench_nlp_pipeline_cpu.py [--device cpu|cuda|both]
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

# ---------------------------------------------------------------------------
# Repo-root path injection: must happen before any src.* import
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = next(
    (p for p in [_SCRIPT_DIR, *_SCRIPT_DIR.parents] if (p / ".git").exists()),
    _SCRIPT_DIR.parent,
)
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Silence transformers / setfit log noise so the Rich panel stays readable.
warnings.filterwarnings("ignore", category=FutureWarning)
for noisy in ("transformers", "setfit", "sentence_transformers", "torch"):
    logging.getLogger(noisy).setLevel(logging.ERROR)
import transformers.training_args as _tra  # noqa: E402

# SetFit compatibility shim: same one N24 applies at the top of the notebook.
if not hasattr(_tra, "default_logdir"):
    _tra.default_logdir = lambda: "runs"

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from scripts.bench._common import (  # noqa: E402
    BenchResult,
    export_csv,
    export_markdown,
    make_start_panel,
    render_results_table,
    time_function,
)
from scripts.cli.theme import console  # noqa: E402

_DATA_ROOT = _REPO_ROOT / "data"
_MODELS_DIR = _DATA_ROOT / "models" / "nlp"
_EVAL_DIR = _DATA_ROOT / "eval"

_SENTIMENT_DIR = _MODELS_DIR / "bert_sentiment_v1"
_INTENT_DIR = _MODELS_DIR / "intent_setfit_modernbert_v1"
_NER_DIR = _MODELS_DIR / "ner_v1" / "bert_bio_v1"

# Same 8 messages N24's GPU benchmark uses, copied verbatim so the CPU
# numbers in this artefact are directly comparable to the notebook's.
_BENCHMARK_MESSAGES: tuple[str, ...] = (
    "Box this lap, box this lap.",
    "We have a hydraulics issue, engine temp is rising.",
    "Verstappen is 1.8 seconds ahead, DRS enabled.",
    "There is rain coming in sector 2, be careful.",
    "P3 on track, gap to Hamilton is 0.4 seconds.",
    "Stay out, stay out, the undercut won't work here.",
    "Front-left degradation is critical, we need to box.",
    "Safety car deployed, pit this lap.",
)

# RCM rows mirror those used in N24's RCM demo, kept identical so the
# rule-based parser sees the same input shape it does in the notebook.
_RCM_ROWS: tuple[dict[str, Any], ...] = (
    {
        "Category": "SafetyCar",
        "Flag": "SAFETY_CAR",
        "Message": "SAFETY CAR DEPLOYED",
        "Scope": "Track",
        "Sector": None,
        "RacingNumber": None,
    },
    {
        "Category": "Flag",
        "Flag": "YELLOW",
        "Message": "DOUBLE YELLOW FLAG",
        "Scope": "Sector",
        "Sector": 2,
        "RacingNumber": None,
    },
    {
        "Category": "Drs",
        "Flag": None,
        "Message": "DRS ENABLED",
        "Scope": "Track",
        "Sector": None,
        "RacingNumber": None,
    },
    {
        "Category": "Other",
        "Flag": None,
        "Message": "CAR 44 RETIRED FROM THE RACE",
        "Scope": "Track",
        "Sector": None,
        "RacingNumber": "44",
    },
    {
        "Category": "Other",
        "Flag": None,
        "Message": "LAP 23 TURN 5 TIME DELETED - TRACK LIMITS",
        "Scope": None,
        "Sector": None,
        "RacingNumber": "1",
    },
    {
        "Category": "Flag",
        "Flag": "GREEN",
        "Message": "GREEN FLAG",
        "Scope": "Track",
        "Sector": None,
        "RacingNumber": None,
    },
    {
        "Category": "CarEvent",
        "Flag": None,
        "Message": "CAR 1 COLLISION WITH CAR 4",
        "Scope": "Track",
        "Sector": None,
        "RacingNumber": "1",
    },
    {
        "Category": "Other",
        "Flag": None,
        "Message": "VIRTUAL SAFETY CAR DEPLOYED",
        "Scope": "Track",
        "Sector": None,
        "RacingNumber": None,
    },
)

_SENTIMENT_LABELS = ("negative", "neutral", "positive")
_INTENT_LABELS = ("INFORMATION", "PROBLEM", "ORDER", "WARNING", "QUESTION")
_NER_MAX_LEN = 128


# ---------------------------------------------------------------------------
# Inline replicas of N24's loaders + predictors
# ---------------------------------------------------------------------------


@dataclass
class _NlpModels:
    """Container holding everything ``_run_pipeline`` needs in a single dict.

    Keeping the loaded objects together lets the benchmark closure read
    ``self.models`` without juggling six separate fields, while the
    type annotations on each attribute keep the structure documented
    for readers who only see the dataclass.
    """

    sentiment_model: Any
    sentiment_tokenizer: Any
    intent_model: Any
    ner_model: Any
    ner_tokenizer: Any
    ner_label2id: dict[str, int]
    ner_id2label: dict[int, str]
    device: str


def _load_sentiment(device: str):
    """Load the fine-tuned RoBERTa sentiment classifier (mirrors N24)."""
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    base_model = "roberta-base"
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForSequenceClassification.from_pretrained(base_model, num_labels=3)
    state_dict = torch.load(
        _SENTIMENT_DIR / "best_roberta_sentiment_model.pt",
        map_location=device,
        weights_only=False,
    )
    if any(k.startswith("model.") for k in state_dict):
        state_dict = {k[len("model.") :]: v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    return model.to(device).eval(), tokenizer


def _load_intent(device: str):
    """Load the SetFit + ModernBERT intent classifier (mirrors N24).

    SetFit's :class:`SetFitModel` does not honour an explicit ``device``
    argument when loading from disk, so the body of the model is moved
    after construction. The encoder lives under ``model_body`` and the
    classification head is a small numpy-backed sklearn-style object;
    only the encoder needs the device move.
    """
    from setfit import SetFitModel

    model = SetFitModel.from_pretrained(str(_INTENT_DIR))
    if hasattr(model, "model_body"):
        try:
            model.model_body.to(device)
        except Exception:
            # Older SetFit versions silently ignore the move; fall back to default.
            pass
    return model


def _load_ner(device: str):
    """Load the fine-tuned BERT-large CoNLL-03 BIO NER model (mirrors N24)."""
    from transformers import AutoTokenizer, BertForTokenClassification

    cfg = json.loads((_NER_DIR / "model_config.json").read_text())
    label2id = cfg["label2id"]
    id2label = {int(k): v for k, v in cfg["id2label"].items()}
    base_model = cfg.get("model_name", "dbmdz/bert-large-cased-finetuned-conll03-english")
    tokenizer = AutoTokenizer.from_pretrained(str(_NER_DIR), use_fast=True)
    model = BertForTokenClassification.from_pretrained(
        base_model,
        num_labels=len(label2id),
        ignore_mismatched_sizes=True,
    )
    state_dict = torch.load(
        _NER_DIR / "bert_bio_state_dict.pt",
        map_location=device,
        weights_only=False,
    )
    model.load_state_dict(state_dict)
    return model.to(device).eval(), tokenizer, label2id, id2label


def _build_models(device: str) -> _NlpModels:
    """Load all three models on ``device`` and return them as a dataclass.

    Equivalent to N24's ``build_pipeline()``; raises a clear error
    when any of the three artefact directories is missing rather than
    falling back to silent training (the bench is read-only by spec).
    """
    for required in (
        _SENTIMENT_DIR / "best_roberta_sentiment_model.pt",
        _INTENT_DIR / "config.json",
        _NER_DIR / "bert_bio_state_dict.pt",
    ):
        if not required.exists():
            raise FileNotFoundError(
                f"Required NLP artefact missing: {required}. "
                "Aborting — model retraining is out of scope for this benchmark."
            )

    sentiment_model, sentiment_tokenizer = _load_sentiment(device)
    intent_model = _load_intent(device)
    ner_model, ner_tokenizer, label2id, id2label = _load_ner(device)
    return _NlpModels(
        sentiment_model=sentiment_model,
        sentiment_tokenizer=sentiment_tokenizer,
        intent_model=intent_model,
        ner_model=ner_model,
        ner_tokenizer=ner_tokenizer,
        ner_label2id=label2id,
        ner_id2label=id2label,
        device=device,
    )


def _predict_sentiment(text: str, models: _NlpModels) -> tuple[str, float]:
    """Return ``(label, confidence)`` for ``text``.

    Mirrors N24's ``predict_sentiment``. Confidence is the softmax
    probability of the argmax class, returned as a float in [0, 1].
    """
    inputs = models.sentiment_tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128,
    )
    inputs = {k: v.to(models.device) for k, v in inputs.items()}
    with torch.no_grad():
        logits = models.sentiment_model(**inputs).logits
    probs = torch.softmax(logits, dim=-1).squeeze().cpu().numpy()
    pred_idx = int(np.argmax(probs))
    return _SENTIMENT_LABELS[pred_idx], float(probs[pred_idx])


def _predict_intent(text: str, models: _NlpModels) -> tuple[str, float]:
    """Return ``(label, confidence)`` from the SetFit intent classifier.

    SetFit's ``predict_proba`` returns a 2-D array (one row per input);
    the bench passes a single string so the confidence is read from
    row 0 at the index matching the predicted class.
    """
    pred_str = models.intent_model.predict([text])[0]
    probs = models.intent_model.predict_proba([text])[0]
    label_idx = _INTENT_LABELS.index(pred_str)
    return pred_str, float(probs[label_idx])


def _predict_entities(text: str, models: _NlpModels) -> list[dict[str, str]]:
    """Decode word-level BIO spans from the NER model (mirrors N24).

    The implementation is a near-exact transcription of N24's
    ``predict_entities``. Splitting on whitespace, encoding with
    ``is_split_into_words=True``, mapping the first sub-token of each
    word to its predicted tag, and stitching contiguous ``B-/I-``
    runs into spans is what the agent's :mod:`src.agents.radio_agent`
    does at simulation time; the benchmark must match that surface
    so the latency is comparable.
    """
    words = text.split()
    enc = models.ner_tokenizer(
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
            models.ner_model(
                input_ids=enc["input_ids"].to(models.device),
                attention_mask=enc["attention_mask"].to(models.device),
            )
            .logits[0]
            .cpu()
        )
    pred_ids = logits.argmax(dim=-1).tolist()

    word_tags: dict[int, str] = {}
    for tok_i, wid in enumerate(word_ids):
        if wid is not None and wid not in word_tags:
            word_tags[wid] = models.ner_id2label.get(pred_ids[tok_i], "O")

    spans: list[dict[str, str]] = []
    current_type: Optional[str] = None
    span_words: list[str] = []
    for wi, word in enumerate(words):
        tag = word_tags.get(wi, "O")
        if tag.startswith("B-"):
            if current_type is not None:
                spans.append(
                    {"text": " ".join(span_words), "label": current_type.lower().replace("_", " ")}
                )
            current_type, span_words = tag[2:], [word]
        elif tag.startswith("I-") and current_type == tag[2:]:
            span_words.append(word)
        else:
            if current_type is not None:
                spans.append(
                    {"text": " ".join(span_words), "label": current_type.lower().replace("_", " ")}
                )
            current_type = None
            span_words = []
            if tag.startswith("B-"):
                current_type, span_words = tag[2:], [word]
    if current_type is not None:
        spans.append(
            {"text": " ".join(span_words), "label": current_type.lower().replace("_", " ")}
        )
    return spans


def _run_pipeline(text: str, models: _NlpModels) -> dict[str, Any]:
    """Run sentiment + intent + NER on a single team-radio string.

    Mirrors N24's ``run_pipeline``. The caller is responsible for
    deciding whether the result is forwarded to the orchestrator;
    this function only synthesises the structured analysis dict that
    the radio agent expects.
    """
    sentiment, sentiment_score = _predict_sentiment(text, models)
    intent, intent_confidence = _predict_intent(text, models)
    entities = _predict_entities(text, models)
    return {
        "message": text,
        "timestamp": datetime.utcnow().isoformat(),
        "analysis": {
            "sentiment": sentiment,
            "sentiment_score": round(sentiment_score, 4),
            "intent": intent,
            "intent_confidence": round(intent_confidence, 4),
            "entities": entities,
            "rcm": None,
        },
    }


# ---------------------------------------------------------------------------
# RCM rule-based parser (no ML, direct transcription of N24)
# ---------------------------------------------------------------------------


def _extract_car_number(message: str, racing_number: Any) -> Optional[str]:
    if racing_number is not None and pd.notna(racing_number) and str(racing_number).strip():
        return str(racing_number).strip()
    m = re.search(r"\bCAR\s+(\d+)\b", message, re.IGNORECASE)
    return m.group(1) if m else None


def _classify_rcm_event(row: dict[str, Any]) -> str:
    cat = str(row.get("Category", "")).strip()
    flag = str(row.get("Flag", "")).strip().upper()
    msg = str(row.get("Message", "")).upper()

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
        if flag == "CHEQUERED" or "CHEQUERED" in msg:
            return "CHEQUERED_FLAG"
        if flag == "BLUE":
            return "BLUE_FLAG"
        if flag == "BLACK AND WHITE":
            return "BLACK_AND_WHITE_FLAG"
        if flag in ("VIRTUAL_SAFETY_CAR", "VSC"):
            return "VIRTUAL_SAFETY_CAR_DEPLOYED"
        if flag == "SAFETY_CAR":
            return "SAFETY_CAR_DEPLOYED"
        if flag == "RED" or "RED FLAG" in msg:
            return "RED_FLAG"
        if flag == "GREEN" or "GREEN FLAG" in msg:
            return "GREEN_FLAG"
        if flag == "CLEAR":
            return "CLEAR_FLAG"
        if flag in ("YELLOW", "DOUBLE YELLOW"):
            scope = str(row.get("Scope", "")).strip()
            sector = row.get("Sector")
            # "Is a sector present?" means two different things by type, and packing
            # both into one guard hid which: an empty string counts as absent, a NaN
            # counts as absent, anything else counts as present.
            if isinstance(sector, str):
                has_sector = bool(sector)
            else:
                has_sector = bool(pd.notna(sector))

            if scope == "Sector" or has_sector:
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
        if "RETIRED" in msg or "ABANDON" in msg:
            return "CAR_RETIRED"
        if "COLLISION" in msg or "CONTACT" in msg:
            return "CAR_COLLISION"
        if "DRS ENABLED" in msg:
            return "DRS_ENABLED"
        if "DRS DISABLED" in msg:
            return "DRS_DISABLED"
        if "TRACK LIMITS" in msg or "DELETED" in msg:
            return "LAP_DELETED"
        if "UNDER INVESTIGATION" in msg or "NOTED" in msg:
            return "INVESTIGATION"
        if "PENALTY" in msg or ("TIME" in msg and "SECOND" in msg):
            return "TIME_PENALTY"
        if "PIT EXIT" in msg or "PIT LANE" in msg:
            return "PIT_EXIT"
        if (
            ("TRACK" in msg and ("CONDITION" in msg or "SLIPPERY" in msg))
            or "DEBRIS" in msg
            or "FLUID" in msg
            or "LOW GRIP" in msg
            or "RAIN" in msg
        ):
            return "TRACK_CONDITION"
        if "LAPPED" in msg and "OVERTAKE" in msg:
            return "LAPPED_CARS_OVERTAKE"
        if "ALL CARS MAY OVERTAKE" in msg:
            return "SAFETY_CAR_ENDING"
        if "VIRTUAL SAFETY CAR DEPLOYED" in msg:
            return "VIRTUAL_SAFETY_CAR_DEPLOYED"
        return "OTHER"

    return "OTHER"


def _parse_rcm_row(row: dict[str, Any]) -> dict[str, Any]:
    event_type = _classify_rcm_event(row)
    sector = row.get("Sector")
    return {
        "event_type": event_type,
        "category": str(row.get("Category", "")).strip(),
        "flag": row.get("Flag") if pd.notna(row.get("Flag", None)) else None,
        "scope": row.get("Scope") if pd.notna(row.get("Scope", None)) else None,
        "sector": int(sector) if sector is not None and pd.notna(sector) else None,
        "car_number": _extract_car_number(str(row.get("Message", "")), row.get("RacingNumber")),
        "message_raw": str(row.get("Message", "")).strip(),
    }


def _run_rcm_pipeline(rcm_row: dict[str, Any], models: _NlpModels) -> dict[str, Any]:
    """Run the rule-based RCM parser on a single race-control row.

    ``models`` is accepted for API parity with :func:`_run_pipeline`
    even though the RCM agent is rule-based and ignores every model
    in the bundle. This makes the timed closure interchangeable in
    the runner without a special case.
    """
    if hasattr(rcm_row, "to_dict"):
        rcm_row = rcm_row.to_dict()
    rcm_result = _parse_rcm_row(rcm_row)
    return {
        "message": rcm_row.get("Message", ""),
        "timestamp": datetime.utcnow().isoformat(),
        "analysis": {
            "sentiment": None,
            "sentiment_score": None,
            "intent": None,
            "intent_confidence": None,
            "entities": None,
            "rcm": rcm_result,
        },
    }


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


class NlpPipelineRunner:
    """Time both NLP entry points on every requested device.

    The runner loads the three models once per requested device, then
    iterates the 8-message bank ``n_runs`` times so the total
    measurement count matches N24's GPU benchmark (``8 * 20 = 160``
    by default). Results are returned as a list of
    :class:`BenchResult` rows with the device and entry point in the
    metrics dict.
    """

    def __init__(self, n_warmup: int = 5, n_runs: int = 20) -> None:
        self.n_warmup = int(n_warmup)
        self.n_runs = int(n_runs)

    def _bench_one_device(self, device: str) -> list[BenchResult]:
        """Return one row per entry point on ``device``."""
        models = _build_models(device)

        # ── run_pipeline: sentiment + intent + NER on every BENCHMARK message ──
        msg_idx_pipe = {"i": 0}

        def _call_pipeline() -> None:
            text = _BENCHMARK_MESSAGES[msg_idx_pipe["i"] % len(_BENCHMARK_MESSAGES)]
            _run_pipeline(text, models)
            msg_idx_pipe["i"] += 1

        n_total_pipeline = self.n_runs * len(_BENCHMARK_MESSAGES)
        latency_pipeline = time_function(
            _call_pipeline,
            n_warmup=self.n_warmup * len(_BENCHMARK_MESSAGES),
            n_runs=n_total_pipeline,
        )

        # ── run_rcm_pipeline: rule-based, iterate the RCM rows ──
        msg_idx_rcm = {"i": 0}

        def _call_rcm() -> None:
            row = _RCM_ROWS[msg_idx_rcm["i"] % len(_RCM_ROWS)]
            _run_rcm_pipeline(row, models)
            msg_idx_rcm["i"] += 1

        n_total_rcm = self.n_runs * len(_RCM_ROWS)
        latency_rcm = time_function(
            _call_rcm,
            n_warmup=self.n_warmup * len(_RCM_ROWS),
            n_runs=n_total_rcm,
        )

        return [
            BenchResult(
                name=f"{device}/run_pipeline",
                metrics={
                    "device": device,
                    "entry_point": "run_pipeline",
                    "mean_ms": latency_pipeline["mean_ms"],
                    "p50_ms": latency_pipeline["p50_ms"],
                    "p95_ms": latency_pipeline["p95_ms"],
                    "n_runs": latency_pipeline["n_runs"],
                    "notes": "8 messages x N runs, sentiment + intent + NER",
                },
            ),
            BenchResult(
                name=f"{device}/run_rcm_pipeline",
                metrics={
                    "device": device,
                    "entry_point": "run_rcm_pipeline",
                    "mean_ms": latency_rcm["mean_ms"],
                    "p50_ms": latency_rcm["p50_ms"],
                    "p95_ms": latency_rcm["p95_ms"],
                    "n_runs": latency_rcm["n_runs"],
                    "notes": "8 RCM rows x N runs, rule-based parser only",
                },
            ),
        ]

    def run(self, devices: list[str]) -> list[BenchResult]:
        """Run the benchmark on each requested device, in order."""
        rows: list[BenchResult] = []
        for device in devices:
            rows.extend(self._bench_one_device(device))
        return rows


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

_COLUMNS = ["row", "device", "entry_point", "mean_ms", "p50_ms", "p95_ms", "n_runs", "notes"]
_TITLE = "NLP pipeline latency (CPU vs GPU)"


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="NLP pipeline (N24) latency benchmark.")
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=("cpu", "cuda", "both"),
        help="Device(s) to benchmark (default cpu).",
    )
    parser.add_argument(
        "--n-warmup", type=int, default=5, help="Warm-up passes over the message bank (default 5)."
    )
    parser.add_argument(
        "--n-runs", type=int, default=20, help="Measured passes over the message bank (default 20)."
    )
    return parser.parse_args(argv)


def _resolve_devices(device_arg: str) -> list[str]:
    """Translate the CLI ``--device`` argument into a concrete device list.

    ``both`` expands to ``[cpu, cuda]`` only when CUDA is available;
    otherwise it downgrades to CPU and prints a warning so the operator
    notices the GPU row is missing rather than assuming it was skipped
    on purpose.
    """
    if device_arg == "cpu":
        return ["cpu"]
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested but torch.cuda.is_available() is False")
        return ["cuda"]
    # both
    if torch.cuda.is_available():
        return ["cpu", "cuda"]
    console.print("[yellow]CUDA unavailable — falling back to CPU only.[/yellow]")
    return ["cpu"]


def main(argv: Optional[list[str]] = None) -> int:
    args = _parse_args(argv)
    console.print(
        make_start_panel(
            "bench_nlp_pipeline_cpu.py",
            f"NLP pipeline (N24) latency, device={args.device}, n_runs={args.n_runs}.",
        )
    )

    devices = _resolve_devices(args.device)
    runner = NlpPipelineRunner(n_warmup=args.n_warmup, n_runs=args.n_runs)
    results = runner.run(devices)

    md_path = _EVAL_DIR / "nlp_pipeline_cpu.md"
    csv_path = _EVAL_DIR / "nlp_pipeline_cpu.csv"
    export_markdown(results, md_path, _TITLE, _COLUMNS)
    export_csv(results, csv_path, _COLUMNS)

    console.print(render_results_table(results, _TITLE, _COLUMNS))
    console.print(f"[green]Markdown:[/green] {md_path.resolve()}")
    console.print(f"[green]CSV:     [/green] {csv_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
