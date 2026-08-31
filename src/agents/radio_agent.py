"""src/agents/radio_agent.py

Radio Agent: extraction from N29_radio_agent.ipynb.

Two-stream NLP pipeline:
  - driver radio: RoBERTa-base sentiment + SetFit intent + BERT-large NER
  - Race Control Messages: deterministic rule-based parser (N23/N24)

Design (N06 pattern): NLP inference always runs before the LLM.
Alerts are built deterministically from NLP is_alert flags. The LLM
receives pre-processed JSON results and acts as a reasoning layer only.
The LLM cannot miss or hallucinate alerts.

Entry points
------------
run_radio_agent(lap_state)
    Self-contained: everything it needs comes from lap_state itself.
    lap_state keys: lap (int), radio_msgs (list[RadioMessage]),
    rcm_events (list[RCMEvent]).

run_radio_agent_from_state(lap_state, laps_df)
    RSM adapter: builds SESSION_META from laps_df, no FastF1 session needed.
    Called by strategy_orchestrator when running from RaceStateManager context.

Tools (LangChain)
-----------------
process_radio_tool: NLP pipeline on a single radio message (testing utility)
process_rcm_tool:   RCM parser on a single event (testing utility)

Note: tools wrap the inference helpers for isolated testing. In production,
run_radio_agent() calls run_pipeline() and run_rcm_pipeline() directly.
"""

import importlib.util
import json
import sys
import threading
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch

warnings.filterwarnings("ignore")

# ── Pydantic for structured LLM output ────────────────────────────────────────
from pydantic import BaseModel, Field as PydanticField

# ── Optional LangChain imports ──────────────────────────────────────
# langchain_core stays eager because the @tool decorators below run at import.
# Probed, not imported. `import langchain_openai` costs 14.3 s measured: it drags
# the langgraph stack and transformers in behind it, and every consumer of the
# name sits inside a factory that already builds its client on first call. So an
# eager import charged that to `f1-sim --help`, to a --no-llm run, and to every
# surface that merely touches this module. find_spec answers the only question
# asked here, is it installed, in about a millisecond and executes nothing.
try:
    from langchain_core.tools import tool
    from langchain_core.messages import HumanMessage, SystemMessage
    _LC_CORE_OK = True
except ImportError:
    _LC_CORE_OK = False

    def tool(fn):
        """No-op decorator when langchain_core is not installed."""
        return fn

_LC_OK = _LC_CORE_OK and importlib.util.find_spec("langchain_openai") is not None


# ── Repo root (with root-stop guard for uv tool install) ─────────────────────
_REPO_ROOT = Path(__file__).resolve()
while not (_REPO_ROOT / ".git").exists():
    if _REPO_ROOT.parent == _REPO_ROOT:
        break
    _REPO_ROOT = _REPO_ROOT.parent

if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Resolve the data root (models + processed parquets) through the cache
# helper when importable; fall back to the repo-relative layout for bare
# dev checkouts where the helper is not yet on the path.
try:
    from src.f1_strat_manager.data_cache import get_data_root as _get_data_root
    _DATA_ROOT = _get_data_root()
except (ImportError, OSError, RuntimeError):
    # Every way get_data_root() can fail, enumerated against its body in
    # src/f1_strat_manager/data_cache.py: ImportError from the import itself on a
    # bare dev checkout; OSError from the three mkdir() calls (read-only mount,
    # permissions); RuntimeError from Path.home() and Path.expanduser(), which
    # raise when no home directory resolves. That last one is not hypothetical:
    # the Path.home() branch IS the `uv tool install` path this block exists to
    # serve, and a container with no HOME would take it. Falling back to the
    # repo-relative data/ is right for all three.
    _DATA_ROOT = _REPO_ROOT / "data"

# ── Module-level globals ───────────────────────────────────────────────────────
# Left over from the notebook's global-state design. Only run_radio_agent_from_state
# writes SESSION_META (for total_laps/year bookkeeping, see its docstring: "not
# queried during inference"), and nothing in this module ever reads it back.
# run_radio_agent itself is self-contained through lap_state and never touches it.
# The LAPS and RCM_DF frames that used to sit here are gone: LAPS was a full copy
# of the race frame taken on every lap and read by nothing, and RCM_DF was never
# even written.
SESSION_META: dict = {}


# ── RCM classification constants ──────────────────────────────────────────────
_SAFETY_FLAGS = {
    "SAFETY_CAR_DEPLOYED",
    "SAFETY_CAR_IN_PIT_LANE",
    "SAFETY_CAR_ENDING",
    "VIRTUAL_SAFETY_CAR_DEPLOYED",
    "VIRTUAL_SAFETY_CAR_ENDING",
    "RED_FLAG",
    "YELLOW_FLAG",
    "YELLOW_FLAG_SECTOR",
    # A time-penalty ruling is strategically relevant whether it lands on us (we
    # serve it) or a rival (they lose track position, changing our undercut /
    # overcut calculus), so it is forwarded as an alert and routed to N30's
    # regulation lookup. The orchestrator's _RCM_PENALTY_EVENT_TYPES already includes it;
    # this is the upstream half that lets it actually reach the alert list
    # (#398 follow-up). Cost is bounded by the chat rate-limit + token cap.
    "TIME_PENALTY",
}


# -- RCM classification: imported, never redeclared ---------------------------
# RCMEvent, the flag maps and the classifier used to live here, and the eval
# harness kept its own port because importing this module loads every NLP model.
# That port drifted: over the real 1515-row 2025 corpus the two disagreed on 427
# messages (28.2%), this side being the correct one on "SAFETY CAR IN THIS LAP"
# because #305's fix never crossed. So the published RCM coverage measured the copy
# rather than what ships. The logic is pure string rules with no model behind it, so
# it now lives where the harness can import it without the model stack (#632).
from src.f1_strat_manager.rcm_events import (  # noqa: E402
    RCMEvent,
    classify_rcm_event as _classify_rcm_event,
)

from src.agents._shared_defaults import LLM_MAX_RETRIES





# ==============================================================================
# Input/output dataclasses
# ==============================================================================

@dataclass
class RadioMessage:
    """A single driver radio message ready for NLP processing.

    driver:
        Three-letter driver code (e.g. 'NOR'). Used to tag the processed
        event in RadioOutput so N31 can attribute alerts to specific drivers.
    lap:
        Race lap number at which the message was recorded. Aligns the radio
        event with the lap-level state consumed by N31.
    text:
        Transcribed radio text, either from Whisper (real audio) or a mock
        string injected during replay simulation.
    timestamp:
        Optional ISO8601 timestamp from the audio source. Not used for
        inference but preserved in the output for post-race logging.
    """

    driver:    str
    lap:       int
    text:      str
    timestamp: Optional[str] = None




@dataclass
class RadioOutput:
    """Structured output of the Radio Agent for one lap window.

    radio_events:
        List of run_pipeline() dicts, one per RadioMessage processed.
        Each dict carries sentiment, intent, entities and the original text.
    rcm_events:
        List of run_rcm_pipeline() dicts, one per RCMEvent processed.
        Each dict carries event_type, flag, car_number and the raw message.
    alerts:
        Filtered subset of radio_events and rcm_events flagged as critical.
        Radio alerts: intent in CFG.alert_intents (PROBLEM, WARNING).
        RCM alerts: event_type in _SAFETY_FLAGS.
        Always deterministic, never modified by the LLM.
        N31 reads this field first to decide whether to escalate to N30 (RAG).
    reasoning:
        Human-readable synthesis from the LLM explaining which events were
        detected and why certain alerts were raised.
    corrections:
        List of NLP mismatches flagged by the LLM when the model classification
        contradicts the message content. Each entry is a dict with keys:
        driver, original_intent, suggested_intent, span, reason.
        Alerts are NOT modified based on corrections. N31 receives both
        the deterministic alerts and the LLM's mismatch assessment and
        decides how to weight them.
    """

    radio_events: list = field(default_factory=list)
    rcm_events:   list = field(default_factory=list)
    alerts:       list = field(default_factory=list)
    reasoning:    str  = ""
    corrections:  list = field(default_factory=list)


# ==============================================================================
# RadioAgentCFG: holds the three NLP models, loaded on first use
# ==============================================================================

@dataclass
class RadioAgentCFG:
    """Runtime configuration for the Radio Agent.

    Holds the three N24 NLP models (sentiment, intent, NER) behind the lazy
    `pipeline` property consumed by run_pipeline(). Device is auto-detected
    from CUDA availability so the module runs on CPU when no GPU is present.

    model_name:
        LM Studio local model identifier for the LLM synthesizer. Must match
        the model currently loaded in LM Studio.
    device:
        Torch device for all three NLP models. Resolved at init time from
        CUDA availability. The same device is used across all inference calls.
    pipeline:
        Dict built on first access with keys sentiment_model,
        sentiment_tokenizer, intent_model, ner_model, ner_tokenizer,
        ner_label2id, ner_id2label. Same schema as N24's build_pipeline().
    alert_intents:
        Intent labels that trigger an alert entry in RadioOutput. PROBLEM and
        WARNING from N21 SetFit are the two signals relevant for N31 escalation.
    intent_names:
        Ordered label list matching N21 SetFit training output. Used to decode
        predict_proba() indices into human-readable labels.
    sentiment_labels:
        Ordered label list matching N20 RoBERTa training output (neg/neu/pos).
    ner_max_len:
        Maximum token length for BERT-large NER tokenisation. Must match the
        N22 training config to avoid truncation artifacts on long messages.
        Also used as the sentiment tokenizer's max_length.
    """

    model_name:       str   = "gpt-4.1-mini"
    device:           str   = field(
        default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu"
    )
    alert_intents:    tuple = ("PROBLEM", "WARNING")
    intent_names:     tuple = ("INFORMATION", "PROBLEM", "ORDER", "WARNING", "QUESTION")
    sentiment_labels: tuple = ("negative", "neutral", "positive")
    ner_max_len:      int   = 128

    # ------------------------------------------------------------------
    # Private model-loading helpers
    # ------------------------------------------------------------------

    def _load_sentiment_model(self, nlp_dir: Path, device: str) -> tuple:
        """Load RoBERTa-base sentiment model and tokenizer from disk.

        Reads the state dict saved by N20, strips any 'model.' key prefix
        introduced by the N20 training wrapper, then moves the model to the
        target device and sets it to eval mode.

        nlp_dir:
            Root NLP models directory; the sentiment checkpoint lives under
            bert_sentiment_v1/ inside it.
        device:
            Torch device string ('cuda' or 'cpu') for model placement.

        Returns (sentiment_tokenizer, sentiment_model).
        """
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        _s_dir  = nlp_dir / "bert_sentiment_v1"
        s_tok   = AutoTokenizer.from_pretrained("roberta-base")
        s_model = AutoModelForSequenceClassification.from_pretrained(
            "roberta-base", num_labels=3
        )
        _s_sd = torch.load(
            _s_dir / "best_roberta_sentiment_model.pt",
            map_location=device, weights_only=False,
        )
        if any(k.startswith("model.") for k in _s_sd):
            _s_sd = {k[len("model."):]: v for k, v in _s_sd.items()}
        s_model.load_state_dict(_s_sd)
        s_model = s_model.to(device).eval()
        return s_tok, s_model

    def _load_intent_model(self, nlp_dir: Path) -> tuple:
        """Load SetFit + ModernBERT intent classifier from disk.

        The model directory must contain a valid SetFit checkpoint saved by N21.
        Returns the loaded model; intent_names come from self.intent_names and
        are not re-read from disk.

        nlp_dir:
            Root NLP models directory; the intent checkpoint lives under
            intent_setfit_modernbert_v1/ inside it.

        Returns (intent_model, intent_names).
        """
        # Compatibility shim: SetFit 0.9.x imports default_logdir from
        # transformers.training_args which was removed in transformers >= 5.0.
        import transformers.training_args as _ta
        if not hasattr(_ta, "default_logdir"):
            import datetime, os
            def _default_logdir() -> str:
                ts = datetime.datetime.now().strftime("%b%d_%H-%M-%S")
                return os.path.join("runs", ts)
            _ta.default_logdir = _default_logdir

        from setfit import SetFitModel
        _i_dir  = nlp_dir / "intent_setfit_modernbert_v1"
        i_model = SetFitModel.from_pretrained(str(_i_dir))
        return i_model, self.intent_names

    def _load_ner_model(self, nlp_dir: Path, device: str) -> tuple:
        """Load BERT-large CoNLL-03 BIO NER model, tokenizer, and label maps.

        Reads model_config.json for label2id/id2label and the base model name,
        loads the state dict saved by N22, then moves the model to the target
        device and sets it to eval mode.

        nlp_dir:
            Root NLP models directory; the NER checkpoint lives under
            ner_v1/bert_bio_v1/ inside it.
        device:
            Torch device string ('cuda' or 'cpu') for model placement.

        Returns (ner_tokenizer, ner_model, label2id, id2label).
        """
        from transformers import AutoTokenizer, BertForTokenClassification
        _n_dir  = nlp_dir / "ner_v1" / "bert_bio_v1"
        _n_cfg  = json.loads((_n_dir / "model_config.json").read_text())
        n_l2i   = _n_cfg["label2id"]
        n_i2l   = {int(k): v for k, v in _n_cfg["id2label"].items()}
        _n_base = _n_cfg.get(
            "model_name", "dbmdz/bert-large-cased-finetuned-conll03-english"
        )
        n_tok   = AutoTokenizer.from_pretrained(str(_n_dir), use_fast=True)
        n_model = BertForTokenClassification.from_pretrained(
            _n_base, num_labels=len(n_l2i), ignore_mismatched_sizes=True
        )
        _n_sd = torch.load(
            _n_dir / "bert_bio_state_dict.pt",
            map_location=device, weights_only=False,
        )
        n_model.load_state_dict(_n_sd)
        n_model = n_model.to(device).eval()
        return n_tok, n_model, n_l2i, n_i2l

    # ------------------------------------------------------------------
    # Initialiser
    # ------------------------------------------------------------------

    def __post_init__(self):
        self._pipeline: Optional[dict] = None
        self._pipeline_lock = threading.Lock()

    @property
    def pipeline(self) -> dict:
        """The three NLP models, built on first use rather than at import.

        Building them reads three checkpoints off disk and costs 17 s measured,
        warm. It used to run in __post_init__, so merely importing this module
        paid the whole bill: importing it took 25.8 s after torch against 8.0 s
        now. scripts/run_simulation_cli.py imports it above its argparse block,
        which is part of why `f1-sim --help` took 12.6 s to print a usage
        string, and every surface that touches the agents module paid it whether
        or not a radio message was ever analysed.

        Every consumer of the models goes through here, so the cost now lands on
        the first message instead. The rest of the config (device, label lists,
        alert intents, ner_max_len) stays eager because it is free, which is
        what keeps _intent_name_to_idx below a plain import-time expression.

        Locked, because the eager version was serialised by the import lock and a
        property is not. Two concurrent requests through the backend's strategy
        endpoint would otherwise each spend the 17 s building a pipeline the
        other was already building. The double check keeps the steady-state read
        lock-free.

        Callers that want the cost paid up front read this property somewhere
        they control, which is what scripts/run_simulation_cli.py does in its
        pre-warm block. Importing the module no longer loads anything.
        """
        if self._pipeline is not None:
            return self._pipeline

        with self._pipeline_lock:
            if self._pipeline is not None:
                return self._pipeline

            nlp_dir = _DATA_ROOT / "models" / "nlp"

            s_tok,  s_model                = self._load_sentiment_model(nlp_dir, self.device)
            i_model, _intent_names         = self._load_intent_model(nlp_dir)
            n_tok,  n_model, n_l2i, n_i2l = self._load_ner_model(nlp_dir, self.device)

            self._pipeline = {
                "sentiment_model":     s_model,
                "sentiment_tokenizer": s_tok,
                "intent_model":        i_model,
                "ner_model":           n_model,
                "ner_tokenizer":       n_tok,
                "ner_label2id":        n_l2i,
                "ner_id2label":        n_i2l,
            }
        return self._pipeline


# ── Module-level singletons ────────────────────────────────────────────────────
CFG = RadioAgentCFG()

# O(1) intent label lookup: populated immediately after CFG is created
_intent_name_to_idx: dict = {name: i for i, name in enumerate(CFG.intent_names)}


# ── LLM availability predicate ─────────────────────────────────────────────────
# Mirrors scripts/run_simulation_cli.py::_is_llm_unavailable so the radio agent
# can degrade gracefully when the synthesis layer is offline (no provider,
# --no-llm mode, network glitch, LM Studio without a loaded model). Kept inline
# instead of imported to avoid a script -> module dependency.
_LLM_SYNTH_ERR_TYPES = (
    "Connection", "APIConnection", "OpenAI", "HTTP", "Timeout",
    "RemoteDisconnected", "BadRequest", "NotFound", "Authentication",
    "APIError", "APIStatusError", "RateLimit", "InternalServerError",
    "ServiceUnavailable", "PermissionDenied",
)
_LLM_SYNTH_ERR_MSGS = (
    "Connection error", "connect ECONNREFUSED", "No models loaded",
    "model_not_found", "invalid_api_key", "Could not connect",
    "ENOTFOUND", "getaddrinfo failed", "F1_LLM_PROVIDER",
)


def _is_llm_synthesis_unavailable(exc: Exception) -> bool:
    """Return True when the radio synthesis LLM call cannot reach a backend.

    Used by run_radio_agent to swap stage 3 for an empty reasoning string when
    the LLM is offline. Errors unrelated to LLM connectivity (NLP model bugs,
    bad lap_state, missing fields) must NOT match: those should propagate up.
    """
    tn  = type(exc).__name__
    msg = str(exc)[:300]
    return any(k in tn  for k in _LLM_SYNTH_ERR_TYPES) or \
           any(k in msg for k in _LLM_SYNTH_ERR_MSGS)


# ==============================================================================
# NLP inference helpers (adapted from N24)
# ==============================================================================

def predict_sentiment(text: str) -> tuple:
    """Run RoBERTa-base sentiment classifier on a single text.

    Uses CFG.pipeline and CFG.device. Returns (label, confidence) where
    label is one of CFG.sentiment_labels (negative/neutral/positive) and
    confidence is the softmax probability of the predicted class.
    """
    model, tokenizer = (
        CFG.pipeline["sentiment_model"],
        CFG.pipeline["sentiment_tokenizer"],
    )
    inputs = tokenizer(
        text, return_tensors="pt", truncation=True,
        padding=True, max_length=CFG.ner_max_len,
    )
    inputs = {k: v.to(CFG.device) for k, v in inputs.items()}
    with torch.no_grad():
        logits = model(**inputs).logits
    probs    = torch.softmax(logits, dim=-1).squeeze().cpu().numpy()
    pred_idx = int(np.argmax(probs))
    return CFG.sentiment_labels[pred_idx], float(probs[pred_idx])


def predict_intent(text: str) -> tuple:
    """Run SetFit + ModernBERT intent classifier on a single text.

    Uses CFG.pipeline. Returns (label, confidence) where label is one of
    CFG.intent_names (INFORMATION/PROBLEM/ORDER/WARNING/QUESTION).
    Confidence is the predict_proba score for the predicted class.
    """
    model     = CFG.pipeline["intent_model"]
    pred_str  = model.predict([text])[0]
    probs     = model.predict_proba([text])[0]
    label_idx = _intent_name_to_idx.get(pred_str, 0)
    return pred_str, float(probs[label_idx])


def _tokenize_words(words: list) -> tuple:
    """Tokenise a pre-split word list and return (encoding, word_ids).

    Uses is_split_into_words=True so the tokenizer preserves word boundaries.
    word_ids maps each subword token index to its originating word index,
    needed to assign a single BIO tag per word from the model predictions.
    """
    tokenizer = CFG.pipeline["ner_tokenizer"]
    enc       = tokenizer(
        words, is_split_into_words=True, add_special_tokens=True,
        max_length=CFG.ner_max_len, padding="max_length",
        truncation=True, return_tensors="pt",
    )
    return enc, enc.word_ids(batch_index=0)


def _decode_word_tags(enc, word_ids: list) -> dict:
    """Run the NER model and map the first subword of each word to its predicted tag.

    Only the first subword token per word is kept to avoid double-counting
    multi-piece words. Special tokens map to None in word_ids and are skipped.
    Returns a dict mapping word index → BIO tag string (e.g. 'B-PER', 'O').
    """
    model    = CFG.pipeline["ner_model"]
    id2label = CFG.pipeline["ner_id2label"]
    with torch.no_grad():
        logits = model(
            input_ids=enc["input_ids"].to(CFG.device),
            attention_mask=enc["attention_mask"].to(CFG.device),
        ).logits[0].cpu()
    pred_ids  = logits.argmax(dim=-1).tolist()
    word_tags: dict = {}
    for tok_i, wid in enumerate(word_ids):
        if wid is not None and wid not in word_tags:
            word_tags[wid] = id2label.get(pred_ids[tok_i], "O")
    return word_tags


def _collect_bio_spans(words: list, word_tags: dict) -> list:
    """Collapse contiguous B-/I- tag sequences into entity span dicts.

    A new span opens on every B- tag. An I- tag extends the current span
    only when its type matches the open span. Returns a list of
    {text, label} dicts; empty list when no named entities are detected.
    """
    spans, current_type, span_words = [], None, []
    for wi, word in enumerate(words):
        tag = word_tags.get(wi, "O")
        if tag.startswith("B-"):
            if current_type:
                spans.append({
                    "text":  " ".join(span_words),
                    "label": current_type.lower().replace("_", " "),
                })
            current_type, span_words = tag[2:], [word]
        elif tag.startswith("I-") and current_type == tag[2:]:
            span_words.append(word)
        else:
            if current_type:
                spans.append({
                    "text":  " ".join(span_words),
                    "label": current_type.lower().replace("_", " "),
                })
            current_type, span_words = None, []
    if current_type:
        spans.append({
            "text":  " ".join(span_words),
            "label": current_type.lower().replace("_", " "),
        })
    return spans


def predict_entities(text: str) -> list:
    """Decode named entities from a radio message using BERT-large CoNLL-03 BIO.

    Orchestrates tokenisation → tag decoding → BIO span collection.
    Returns a list of {text, label} dicts, empty when no entities are found.
    """
    words          = text.split()
    enc, word_ids  = _tokenize_words(words)
    word_tags      = _decode_word_tags(enc, word_ids)
    return _collect_bio_spans(words, word_tags)


def run_pipeline(text: str, rcm_result: Optional[dict] = None) -> dict:
    """Chain sentiment → intent → NER on a single radio message.

    text:
        Transcribed radio text to process (Whisper output or mock string).
    rcm_result:
        Optional pre-parsed RCM dict to attach race control context to a
        driver radio event. Stored under analysis.rcm for N31 inspection.

    Returns the agreed JSON schema consumed by N31:
        message, timestamp, analysis (sentiment, sentiment_score, intent,
        intent_confidence, entities, rcm).
    """
    sentiment, sentiment_score = predict_sentiment(text)
    intent, intent_confidence  = predict_intent(text)
    entities                   = predict_entities(text)
    return {
        "message":   text,
        "timestamp": datetime.utcnow().isoformat(),
        "analysis": {
            "sentiment":         sentiment,
            "sentiment_score":   round(sentiment_score, 4),
            "intent":            intent,
            "intent_confidence": round(intent_confidence, 4),
            "entities":          entities,
            "rcm":               rcm_result,
        },
    }


# ==============================================================================
# RCM parser (inlined from N23/N24)
# ==============================================================================



def run_rcm_pipeline(event: "RCMEvent") -> dict:
    """Parse a single RCMEvent into the canonical output schema.

    event:
        RCMEvent dataclass instance with message, flag, category, lap, etc.

    Returns a dict with keys: event_type, flag, message, lap, car_number,
    is_alert (True when event_type is in _SAFETY_FLAGS). is_alert is the
    signal N31 reads to escalate to the RAG agent (N30).
    """
    event_type = _classify_rcm_event(event)
    return {
        "event_type": event_type,
        "flag":       event.flag,
        "message":    event.message,
        "lap":        event.lap,
        "car_number": event.racing_number,
        "is_alert":   event_type in _SAFETY_FLAGS,
    }


# ==============================================================================
# LangChain tools (testing utilities, not used in production flow)
# ==============================================================================

@tool
def process_radio_tool(driver: str, lap: int, text: str) -> str:
    """Process a transcribed driver radio message through the NLP pipeline.

    Testing utility: not part of the agent's main flow. In production,
    run_radio_agent() calls run_pipeline() directly before the LLM synthesis step.
    Useful for manual inspection of individual radio messages in isolation.

    Runs sentiment (RoBERTa-base), intent (SetFit + ModernBERT) and named entity
    recognition (BERT-large CoNLL-03 BIO) on the transcribed text.

    Args:
        driver: Three-letter driver code (e.g. 'NOR').
        lap: Race lap number at which the message was transmitted.
        text: Transcribed radio text from Whisper or a mock string.

    Returns:
        Structured string: "driver={driver} | lap={lap} | intent={intent}
        ({intent_confidence:.3f}) | sentiment={sentiment} ({sentiment_score:.3f})
        | entities=[{entity} ({label}), ...] | is_alert={bool}"
    """
    result       = run_pipeline(text)
    ana          = result["analysis"]
    intent       = ana["intent"]
    is_alert     = intent in CFG.alert_intents
    entities_str = (
        ", ".join(f"{e['text']} ({e['label']})" for e in ana["entities"]) or "none"
    )
    return (
        f"driver={driver} | lap={lap} | "
        f"intent={intent} ({ana['intent_confidence']:.3f}) | "
        f"sentiment={ana['sentiment']} ({ana['sentiment_score']:.3f}) | "
        f"entities=[{entities_str}] | is_alert={is_alert}"
    )


@tool
def process_rcm_tool(
    message: str, flag: str, category: str, lap: int,
    scope: str = "", racing_number: str = "",
) -> str:
    """Parse a Race Control Message and flag safety-critical events.

    Testing utility: not part of the agent's main flow. In production,
    run_radio_agent() calls run_rcm_pipeline() directly before the LLM synthesis step.
    Useful for manual inspection of individual RCM rows in isolation.

    Applies the N23 rule-based classifier to determine the event type and whether
    it is safety-critical (SAFETY_CAR_*, RED_FLAG, YELLOW_FLAG variants).

    Args:
        message: Raw message string from session.race_control_messages.
        flag: Flag type string from FastF1 (e.g. 'YELLOW', 'GREEN', 'SAFETY CAR').
        category: RCM category from FastF1 (e.g. 'SafetyCar', 'Flag', 'Other').
        lap: Race lap number at which the message was issued.
        scope: Spatial scope string ('Track', 'Sector', 'Driver'). Empty if not set.
        racing_number: Car number referenced by the message. Empty string if not set.

    Returns:
        Structured string: "lap={lap} | event_type={event_type} | flag={flag}
        | car={car_number or N/A} | is_alert={bool} | message=\"{message}\""
    """
    event = RCMEvent(
        message=message, flag=flag, category=category, lap=lap,
        scope=scope, racing_number=racing_number or None,
    )
    result = run_rcm_pipeline(event)
    return (
        f"lap={lap} | event_type={result['event_type']} | "
        f"flag={result['flag']} | car={result['car_number'] or 'N/A'} | "
        f"is_alert={result['is_alert']} | message=\"{message}\""
    )


# ==============================================================================
# LLM synthesizer: lazy singleton, structured output
# ==============================================================================

_RADIO_SYSTEM_PROMPT = """You are the Radio Agent for an F1 race strategy system.

You receive pre-processed NLP results for driver radio messages and Race Control Messages (RCM).
Your job is to synthesise the data into a strategic summary for the Race Orchestrator (N31).

For reasoning: 2-3 sentences connecting the events to concrete strategic decisions
(pit, stay out, prepare for SC, tyre management, etc.).

For corrections: compare each message's original text against its NLP-assigned intent.
If they clearly contradict (a positive/neutral message classified as PROBLEM, or a critical
message classified as INFORMATION), add a correction entry with the verbatim span from the
message that contradicts the model's label, and a one-line reason.
Leave corrections empty if all classifications look correct.
If corrections exist, factor them into reasoning: a likely-misclassified PROBLEM alert
should carry lower urgency than a well-supported one.

Base your response only on the provided data. Do not invent events.
"""


class CorrectionEntry(BaseModel):
    """A single NLP mismatch flagged by the LLM synthesizer.

    driver:
        Three-letter driver code of the misclassified message.
    original_intent:
        Intent label produced by the N21 SetFit model.
    suggested_intent:
        Intent label the LLM believes is correct based on message content.
    span:
        Verbatim substring from the original message that contradicts the
        model's label, the specific phrase the LLM used as evidence.
    reason:
        One-line explanation of why the model label is incorrect.
    """

    driver:           str = PydanticField(description="Three-letter driver code")
    original_intent:  str = PydanticField(description="Intent label from N21 SetFit")
    suggested_intent: str = PydanticField(description="Intent label the LLM believes is correct")
    span:             str = PydanticField(description="Verbatim substring from the message supporting the correction")
    reason:           str = PydanticField(description="One-line explanation of the mismatch")


class RadioSynthesis(BaseModel):
    """Structured output from the Radio Agent LLM synthesizer.

    reasoning:
        2-3 sentences connecting the detected events to concrete strategic
        decisions (pit, stay out, SC preparation, etc.). If corrections exist,
        likely-misclassified alerts are weighted with lower urgency here.
    corrections:
        List of NLP mismatches detected by the LLM. Empty if all model
        classifications are consistent with the message content.
    """

    reasoning:   str                    = PydanticField(
        description="2-3 sentences on strategic implications"
    )
    corrections: list[CorrectionEntry]  = PydanticField(
        default_factory=list,
        description="NLP mismatches, empty list if none",
    )


# Lazy singleton: created on first call to avoid LLM connection at import time
_structured_llm = None


def _get_radio_llm():
    """Return the cached structured-output LLM, creating it on first call.

    Uses LM Studio at localhost:1234 with parallel_tool_calls disabled to avoid
    Jinja NullValue errors. Returns a Runnable that produces RadioSynthesis objects.
    Raises ImportError when langchain_openai is not installed.
    """
    import os
    global _structured_llm
    if _structured_llm is None:
        if not _LC_OK:
            raise ImportError(
                "langchain_openai is not installed — cannot run LLM synthesis. "
                "Install it with: pip install langchain-openai"
            )
        from langchain_openai import ChatOpenAI
        provider = os.environ.get('F1_LLM_PROVIDER', 'lmstudio')
        if provider == 'openai':
            # parallel_tool_calls is NOT sent, because OpenAI rejects it when no tools are specified
            base_llm = ChatOpenAI(
                model=CFG.model_name,
                temperature=0.0,
                timeout=120,
                max_retries=LLM_MAX_RETRIES,
            )
        else:
            base_llm = ChatOpenAI(
                model=CFG.model_name,
                base_url="http://localhost:1234/v1",
                api_key="lm-studio",
                temperature=0.0,
                model_kwargs={"parallel_tool_calls": False},
                timeout=120,
                max_retries=LLM_MAX_RETRIES,
            )
        _structured_llm = base_llm.with_structured_output(RadioSynthesis)
    return _structured_llm


def _build_synthesis_prompt(lap: int, radio_results: list, rcm_results: list) -> str:
    """Build the human message the LLM receives with pre-processed NLP JSON results.

    lap:
        Current race lap number, appearing as the first line so the LLM
        has temporal context before reading the event lists.
    radio_results:
        List of dicts from run_pipeline(), formatted as indented JSON so the
        LLM reads structured data rather than raw text (N06 pattern).
    rcm_results:
        List of dicts from run_rcm_pipeline(), formatted the same way.

    Returns a prompt string combining both lists under labelled headers.
    """
    radio_json = json.dumps(radio_results, indent=2, ensure_ascii=False)
    rcm_json   = json.dumps(rcm_results,   indent=2, ensure_ascii=False)
    return (
        f"Lap {lap}.\n\n"
        f"RADIO MESSAGES — NLP RESULTS:\n{radio_json}\n\n"
        f"RCM EVENTS:\n{rcm_json}"
    )


# ==============================================================================
# Alert builder (deterministic, runs before LLM)
# ==============================================================================

def _build_alerts(
    radio_results: list, rcm_results: list, radio_msgs: list,
) -> list:
    """Build the alerts list deterministically from NLP inference results.

    Radio alerts: intent in CFG.alert_intents (PROBLEM, WARNING).
    RCM alerts: is_alert=True from run_rcm_pipeline (SAFETY_CAR_*, RED_FLAG, etc.).
    Alerts are dicts so N31 can access structured fields without re-parsing.

    radio_results:
        List of dicts from run_pipeline(), one per RadioMessage. Each dict has
        keys 'message' and 'analysis' (with 'intent', 'sentiment', 'entities').
    rcm_results:
        List of dicts from run_rcm_pipeline(), one per RCMEvent. Each dict has
        keys 'event_type', 'message', 'lap', 'is_alert'.
    radio_msgs:
        Original RadioMessage objects in the same order as radio_results.
        Used to extract the driver abbreviation for radio alert entries.

    Returns a list of alert dicts. Radio alerts: source='radio', driver, intent,
    sentiment, entities, message. RCM alerts: source='rcm', event_type, message, lap.
    """
    alerts = []
    for i, result in enumerate(radio_results):
        if result["analysis"]["intent"] in CFG.alert_intents:
            driver = radio_msgs[i].driver if i < len(radio_msgs) else "UNKNOWN"
            alerts.append({
                "source":    "radio",
                "driver":    driver,
                "intent":    result["analysis"]["intent"],
                "sentiment": result["analysis"]["sentiment"],
                "entities":  result["analysis"]["entities"],
                "message":   result["message"],
            })
    for result in rcm_results:
        if result["is_alert"]:
            alerts.append({
                "source":     "rcm",
                "event_type": result["event_type"],
                "message":    result["message"],
                "lap":        result["lap"],
            })
    return alerts


# ==============================================================================
# Persist helper (optional side effect)
# ==============================================================================

def _save_nlp_json(lap: int, radio_results: list, rcm_results: list) -> Path:
    """Persist the NLP inference results for one lap to disk as a JSON file.

    lap:
        Current race lap number, used in the output filename.
    radio_results:
        List of dicts from run_pipeline(), serialised as-is.
    rcm_results:
        List of dicts from run_rcm_pipeline(), serialised as-is.

    Saved to data/processed/radio_outputs/ with a lap+timestamp filename.
    This is a pure side effect. The agent flow uses the in-memory dicts.
    Useful for audit trails, debugging and post-race replay.

    Returns the Path of the saved file, or None when the directory or file
    cannot be written (read-only mount).
    """
    out_dir   = _DATA_ROOT / "processed" / "radio_outputs"
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
    except OSError:
        return None  # read-only mount in Docker, skip persist
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    path      = out_dir / f"radio_nlp_lap{lap:03d}_{timestamp}.json"
    payload   = {"lap": lap, "radio_results": radio_results, "rcm_results": rcm_results}
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False, default=str)
    except OSError:
        return None  # read-only mount in Docker, skip persist
    return path


# ==============================================================================
# Entry points
# ==============================================================================

def run_radio_agent(lap_state: dict, persist: bool = False) -> "RadioOutput":
    """Run the Radio Agent for one lap window and return a structured RadioOutput.

    Follows the N06 design: NLP inference runs first for all inputs, then the
    LLM receives the pre-processed JSON results and synthesises REASONING and
    CORRECTIONS via with_structured_output(RadioSynthesis). Alerts are built
    deterministically from NLP outputs, so the LLM cannot miss or hallucinate them.

    The caller (N31) is responsible for pre-filtering radio_msgs and rcm_events
    to the relevant lap window before calling this function.

    lap_state keys:
        lap (int): Current race lap number.
        radio_msgs (list[RadioMessage]): Transcribed or mocked driver radio messages.
        rcm_events (list[RCMEvent]): Race control messages for this lap window.
    persist:
        When True, saves the NLP JSON to data/processed/radio_outputs/ before
        passing results to the LLM. Default False for N31 loop calls where
        disk I/O would accumulate over many laps.

    Returns a RadioOutput with radio_events, rcm_events, alerts, reasoning,
    and corrections populated.
    """
    lap        = lap_state["lap"]

    # Normalise radio_msgs: accept both RadioMessage dataclasses and plain dicts
    _raw_radio = lap_state.get("radio_msgs", [])
    radio_msgs: list[RadioMessage] = []
    for m in _raw_radio:
        if isinstance(m, RadioMessage):
            radio_msgs.append(m)
        elif isinstance(m, dict):
            radio_msgs.append(RadioMessage(
                driver    = m.get("driver", "UNK"),
                lap       = m.get("lap", lap),
                text      = m.get("text", ""),
                timestamp = m.get("timestamp"),
            ))

    # Normalise rcm_events: accept both RCMEvent dataclasses and plain dicts
    _raw_rcm = lap_state.get("rcm_events", [])
    rcm_events: list[RCMEvent] = []
    for ev in _raw_rcm:
        if isinstance(ev, RCMEvent):
            rcm_events.append(ev)
        elif isinstance(ev, dict):
            rcm_events.append(RCMEvent(
                message       = ev.get("message", ""),
                flag          = ev.get("flag", ""),
                category      = ev.get("category", "Other"),
                lap           = ev.get("lap", lap),
                racing_number = ev.get("racing_number"),
                scope         = ev.get("scope", ""),
            ))

    # Stage 1: NLP inference (N06 pattern: models run before LLM)
    radio_results = [run_pipeline(msg.text) for msg in radio_msgs]
    rcm_results   = [run_rcm_pipeline(ev) for ev in rcm_events]

    # Optional side effect: persist NLP JSON for audit trail
    if persist:
        _save_nlp_json(lap, radio_results, rcm_results)

    # Stage 2: Deterministic alerts from NLP results (before LLM)
    alerts = _build_alerts(radio_results, rcm_results, radio_msgs)

    # Stage 3: LLM synthesises structured RadioSynthesis via with_structured_output.
    # Wrapped in try/except so a missing or unreachable LLM backend (no provider,
    # --no-llm mode, network glitch, LM Studio without a loaded model) does not
    # discard the load-bearing stages 1+2. radio_events / rcm_events / alerts are
    # the contract the orchestrator and the inference panel actually consume.
    # reasoning + corrections are presentation-only and acceptable to drop when
    # the synthesis layer is offline.
    try:
        prompt    = _build_synthesis_prompt(lap, radio_results, rcm_results)
        synthesis: RadioSynthesis = _get_radio_llm().invoke([
            SystemMessage(content=_RADIO_SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ])
        reasoning   = synthesis.reasoning
        corrections = [c.model_dump() for c in synthesis.corrections]
    except Exception as exc:
        if not _is_llm_synthesis_unavailable(exc):
            raise
        reasoning   = "[no-LLM mode — radio synthesis skipped, NLP stages 1+2 still applied]"
        corrections = []

    return RadioOutput(
        radio_events=radio_results,
        rcm_events=rcm_results,
        alerts=alerts,
        reasoning=reasoning,
        corrections=corrections,
    )


def run_radio_agent_from_state(
    lap_state: dict,
    laps_df: pd.DataFrame,
    persist: bool = False,
) -> "RadioOutput":
    """RSM adapter: run the Radio Agent without a live FastF1 session.

    Populates the module-level SESSION_META global from laps_df for
    bookkeeping (see the module docstring); the actual radio and RCM processing
    runs through run_radio_agent(lap_state, persist=persist), which reads
    everything it needs from lap_state directly and never touches those globals.
    radio_msgs and rcm_events must still be provided in lap_state: the
    orchestrator pre-filters them to the current lap window before handing them
    to this function.

    lap_state keys:
        lap (int): Current race lap number.
        radio_msgs (list[RadioMessage]): Pre-filtered driver radio messages.
        rcm_events (list[RCMEvent]): Pre-filtered race control messages.
        session_meta (dict, optional): Contains 'gp' key for SESSION_META.
    laps_df:
        Full laps DataFrame from RaceStateManager. Used only to populate
        SESSION_META.total_laps and SESSION_META.year, not queried during
        inference (radio_msgs and rcm_events are already in lap_state).
    persist:
        Forwarded to run_radio_agent: see its docstring.

    Returns a RadioOutput identical to what run_radio_agent() would produce.
    """
    global SESSION_META

    SESSION_META = {
        "year":       int(laps_df["Year"].iloc[0]) if "Year" in laps_df.columns else 0,
        "gp":         lap_state.get("session_meta", {}).get("gp", ""),
        "total_laps": int(laps_df["LapNumber"].max()),
        "session":    None,
    }
    return run_radio_agent(lap_state, persist=persist)
