"""Golden tests for the #303/#304 NLP eval harness.

The gated-stage contract is hermetic (authored data). The intent reproduction +
NR-08 column-order verdict are data-tier: they load pickled weights, so they run
locally and skip on CI without ``data/models/``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent.parent
_INTENT_DIR = ROOT / "data" / "models" / "nlp" / "intent_setfit_modernbert_v1"
_HAS_INTENT_HEAD = (_INTENT_DIR / "model_head.pkl").exists()
_HAS_NER_CFG = (ROOT / "data" / "models" / "nlp" / "ner_v1" / "model_config.json").exists()


def test_nlp_gated_stages_are_pending_after_303():
    """After #303 only NER F1, RCM and alert precision stay gated; intent is not."""
    from src.strategy.eval.nlp import _gated_stages

    by_stage = {r.stage: r for r in _gated_stages()}
    assert set(by_stage) == {"ner", "rcm", "alert_precision"}
    assert all(r.status == "pending" for r in by_stage.values())
    assert "303" in by_stage["ner"].detail or "#304" in by_stage["ner"].detail


@pytest.mark.data
@pytest.mark.skipif(not _HAS_INTENT_HEAD, reason="intent head absent (CI runner without weights)")
def test_intent_predict_proba_order_is_aligned():
    """NR-08: the production predict_proba decode reads the correct class (no swap)."""
    from src.strategy.eval.nlp import verify_intent_column_order

    verdict = verify_intent_column_order()
    assert verdict.status == "reproduced"
    assert verdict.value == 1.0


@pytest.mark.data
@pytest.mark.skipif(not _HAS_INTENT_HEAD, reason="intent model absent (CI runner without weights)")
def test_intent_reproduction_is_sane():
    """Setfit-free reproduction runs and beats chance on the labeled set."""
    from src.strategy.eval.nlp import reproduce_intent

    rows = {r.metric: r for r in reproduce_intent()}
    assert rows["accuracy"].status != "pending"
    assert rows["accuracy"].value is not None and rows["accuracy"].value > 0.5
    assert "weighted_f1" in rows


@pytest.mark.data
@pytest.mark.skipif(not _HAS_NER_CFG, reason="ner model_config absent (CI runner without weights)")
def test_dead_ner_classes_flags_zero_f1_types():
    """NR-07: entity types with ~0 frozen-eval B-F1 are flagged untrustworthy."""
    from src.strategy.eval.nlp import dead_ner_classes

    dead, row = dead_ner_classes()
    assert row.status == "flagged"
    assert row.value == float(len(dead))
    assert len(dead) >= 1
