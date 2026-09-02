"""Golden tests for the LLM-judge alert-precision module (#304 follow-up).

The render + degradation paths are hermetic. The corpus selection is data-tier
(needs the CSVs). The LLM call itself is neither run nor mocked here - it spends
API calls and is non-deterministic, so it is exercised only via ``f1-eval
alert-llm``.
"""

from __future__ import annotations

from pathlib import Path

import pytest


def _radio_nlp(name: str) -> Path:
    """A corpus file at the path the stage reads, not at the repo's copy of it.

    `alert_llm.py` reads `get_data_root()`, which `F1_STRAT_DATA_ROOT` moves. A probe
    anchored on the repo does not follow it, and a probe that HITS while the code
    misses turns this skip into a hard failure.
    """
    from src.f1_strat_manager.data_cache import get_data_root

    return get_data_root() / "processed" / "radio_nlp" / name


_HAS_CORPUS = (
    _radio_nlp("radios_raw.csv").exists() and _radio_nlp("intent_labeled_data.csv").exists()
)


def test_render_leads_with_proxy_caveat():
    """The report must open with the PROXY / human-review caveat, not a bare number."""
    from src.strategy.eval.alert_llm import AlertLLMResult, _render

    body = _render(AlertLLMResult(80, 20, 15, 0.75, "detail"))
    assert body.splitlines()[0].startswith("> PROXY")
    assert "human-review" in body
    assert "0.7500" in body


def test_render_handles_missing_precision():
    """A degraded result (no precision) still renders without raising."""
    from src.strategy.eval.alert_llm import AlertLLMResult, _render

    body = _render(AlertLLMResult(0, 0, 0, None, "LLM judge unavailable"))
    assert "| proxy precision | - |" in body


@pytest.mark.data
@pytest.mark.skipif(not _HAS_CORPUS, reason="radio corpus not downloaded")
def test_unlabeled_radios_excludes_the_labeled_set():
    """The sample is drawn only from radios absent from the hand-labeled intent set."""
    import pandas as pd

    from src.strategy.eval.alert_llm import _unlabeled_radios

    sample = _unlabeled_radios(30)
    assert sample is not None and len(sample) == 30
    labeled = set(pd.read_csv(_radio_nlp("intent_labeled_data.csv"))["message"].astype(str))
    assert all(text not in labeled for text in sample)
