"""LLM-judge alert precision over UNLABELED radios (#304 follow-up).

The gold alert precision (0.9185, in ``nlp.py``) is measured on the hand-labeled
intent set. This module extends coverage to the ~154 radios that carry NO hand
label, using an LLM as a PROXY judge: it flags the radios the intent model calls
an alert (PROBLEM/WARNING), asks the LLM whether each is a genuine pit-wall alert,
and reports the proxy precision.

IMPORTANT - this is a PROXY, not ground truth:
- LLM verdicts are not gold; the number is indicative, not defensible for the
  paper on its own. The paper-grade alert precision is the gold-based one.
- Non-deterministic (an external model), so it is deliberately kept OUT of the
  reproducible ``f1-eval nlp`` report and behind its own ``f1-eval alert-llm``
  command that spends API calls only when invoked.
- FLAG FOR VICTOR: for a paper claim on the unlabeled set, these proxy labels
  need a human-review pass.

LLM provider follows ``F1_LLM_PROVIDER`` (openai / lmstudio), never Anthropic.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from typing import Any

from src.f1_strat_manager.data_cache import get_data_root
from src.strategy.eval.nlp import _ALERT_INTENTS, _load_intent_setfit_free
from src.strategy.eval.report import build_header, write_report

ALERT_LLM_NAME = "alert_llm"
_JUDGE_MODEL = "gpt-4.1-mini"
_DEFAULT_SAMPLE = 80

_JUDGE_SYSTEM = (
    "You are an F1 race strategist on the pit wall. A radio message is an ALERT only if it "
    "reports a genuine problem or a warning the strategist must act on (car damage, tyre/brake "
    "issues, hazards, incidents, urgent instructions). Neutral information, questions, banter, or "
    "routine updates are NOT alerts. Answer with a single word: YES or NO."
)


@dataclass
class AlertLLMResult:
    """The proxy alert-precision measurement over the unlabeled sample."""

    sample_size: int
    predicted_alerts: int
    judged_true: int
    proxy_precision: float | None
    detail: str


def _make_judge_llm() -> Any:
    """Instantiate the judge LLM per ``F1_LLM_PROVIDER`` (openai default here, or lmstudio).

    Mirrors the provider switch in ``radio_agent`` so the harness reaches the same
    backend the agents use. Temperature 0 for as-deterministic-as-possible verdicts.
    """
    from langchain_openai import ChatOpenAI

    provider = os.environ.get("F1_LLM_PROVIDER", "openai")
    if provider == "lmstudio":
        return ChatOpenAI(
            model="local-model",
            base_url="http://localhost:1234/v1",
            api_key="lm-studio",
            temperature=0,
        )
    return ChatOpenAI(model=_JUDGE_MODEL, temperature=0)


def _unlabeled_radios(sample_size: int) -> list[str] | None:
    """The first ``sample_size`` radios (sorted, deterministic) absent from the intent set."""
    import pandas as pd

    raw_path = get_data_root() / "processed" / "radio_nlp" / "radios_raw.csv"
    labeled_path = get_data_root() / "processed" / "radio_nlp" / "intent_labeled_data.csv"
    if not (raw_path.exists() and labeled_path.exists()):
        return None

    labeled = set(pd.read_csv(labeled_path)["message"].astype(str))
    raw = pd.read_csv(raw_path)["text"].astype(str)
    unlabeled = sorted(t for t in raw if t.strip() and t not in labeled)
    return unlabeled[:sample_size]


def _predicted_alerts(texts: list[str]) -> list[str] | None:
    """Radios the intent model flags as an alert (intent in PROBLEM/WARNING)."""
    loaded = _load_intent_setfit_free()
    if loaded is None:
        return None
    st, head, idx_to_name = loaded
    preds = [idx_to_name[int(c)] for c in head.predict(st.encode(texts, show_progress_bar=False))]
    return [text for text, pred in zip(texts, preds) if pred in _ALERT_INTENTS]


def _judge(llm: Any, text: str) -> bool:
    """Ask the LLM whether one radio is a genuine pit-wall alert (YES/NO)."""
    response = llm.invoke([("system", _JUDGE_SYSTEM), ("human", text)])
    return "yes" in str(response.content).strip().lower()[:5]


def run_alert_precision_llm(sample_size: int = _DEFAULT_SAMPLE) -> AlertLLMResult:
    """Compute proxy alert precision over the unlabeled sample via the LLM judge.

    Returns a degraded result (``proxy_precision=None``) when the intent model,
    the corpus, or the LLM backend is unavailable, rather than raising.
    """
    texts = _unlabeled_radios(sample_size)
    if not texts:
        return AlertLLMResult(0, 0, 0, None, "intent set or radios_raw.csv absent")

    alerts = _predicted_alerts(texts)
    if alerts is None:
        return AlertLLMResult(len(texts), 0, 0, None, "intent model absent")
    if not alerts:
        return AlertLLMResult(len(texts), 0, 0, None, "no predicted alerts in the sample")

    try:
        llm = _make_judge_llm()
        verdicts = [_judge(llm, text) for text in alerts]
    except Exception as exc:  # noqa: BLE001 - any LLM/transport failure degrades to a note
        return AlertLLMResult(len(texts), len(alerts), 0, None, f"LLM judge unavailable: {type(exc).__name__}")

    judged_true = sum(verdicts)
    precision = judged_true / len(alerts)
    detail = (
        f"PROXY: {judged_true}/{len(alerts)} predicted alerts judged genuine by {_JUDGE_MODEL}; "
        "LLM labels are not ground truth - needs human review before any paper claim"
    )
    return AlertLLMResult(len(texts), len(alerts), judged_true, round(precision, 4), detail)


def _render(result: AlertLLMResult) -> str:
    """Render the proxy alert-precision report with its PROXY caveat up front."""
    precision = "-" if result.proxy_precision is None else f"{result.proxy_precision:.4f}"
    return "\n".join(
        [
            "> PROXY METRIC - LLM-judged, not ground truth. Needs a human-review pass before any "
            "paper claim. The paper-grade alert precision is the gold-based one in the nlp report.",
            "",
            "| metric | value |",
            "|---|---|",
            f"| sample size (unlabeled) | {result.sample_size} |",
            f"| predicted alerts | {result.predicted_alerts} |",
            f"| judged genuine | {result.judged_true} |",
            f"| proxy precision | {precision} |",
            "",
            result.detail,
        ]
    )


def build_alert_llm_report() -> dict[str, Any]:
    """Regenerate the proxy alert-precision report (spends API calls)."""
    result = run_alert_precision_llm()
    provider = os.environ.get("F1_LLM_PROVIDER", "openai")
    header = build_header(
        dataset="radios_raw.csv unlabeled subset", llm=f"{provider}/{_JUDGE_MODEL}"
    )
    payload = {"result": asdict(result)}
    md_path, json_path = write_report(ALERT_LLM_NAME, header, _render(result), payload)
    return {"header": asdict(header), "md_path": str(md_path), "json_path": str(json_path), **payload}
