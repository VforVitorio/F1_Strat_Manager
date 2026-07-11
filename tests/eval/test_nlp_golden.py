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
_HAS_NER_MODEL = (
    ROOT / "data" / "models" / "nlp" / "ner_v1" / "bert_bio_v1" / "bert_bio_state_dict.pt"
).exists()
_HAS_RCM_CORPUS = bool(list((ROOT / "data" / "processed").glob("race_radios/2025/*/rcm.parquet")))


def test_no_nlp_stage_is_gated_after_304():
    """After #304 every NLP stage reproduces; nothing stays gated."""
    from src.strategy.eval.nlp import _gated_stages

    assert _gated_stages() == []


def test_rcm_classifier_maps_known_events():
    """Hermetic: the ported N23 rule-based classifier maps representative rows."""
    from src.strategy.eval.nlp import _classify_rcm_event

    assert (
        _classify_rcm_event("SafetyCar", "", None, None, "SAFETY CAR DEPLOYED")
        == "SAFETY_CAR_DEPLOYED"
    )
    assert _classify_rcm_event("Flag", "BLUE", None, None, "WAVED BLUE FLAG") == "BLUE_FLAG"
    assert _classify_rcm_event("Drs", "", None, None, "DRS ENABLED") == "DRS_ENABLED"
    yellow_sector = _classify_rcm_event(
        "Flag", "DOUBLE YELLOW", "Sector", 20, "DOUBLE YELLOW SECTOR 20"
    )
    assert yellow_sector == "YELLOW_FLAG_SECTOR"
    assert _classify_rcm_event("Other", "", None, None, "SOMETHING UNMAPPED") == "OTHER"


def test_gold_bio_tags_align_to_split_words():
    """Hermetic: char-offset gold entities become word-level BIO over text.split()."""
    from src.strategy.eval.nlp import _gold_bio_tags

    text = "Box now Hamilton"
    # "Hamilton" starts at char 8
    words, tags = _gold_bio_tags(text, [[8, 16, "ACTION"]])
    assert words == ["Box", "now", "Hamilton"]
    assert tags == ["O", "O", "B-ACTION"]


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
@pytest.mark.skipif(not _HAS_INTENT_HEAD, reason="intent model absent (CI runner without weights)")
def test_alert_precision_from_gold_is_high():
    """Alert precision (intent PROBLEM/WARNING) is real (gold-derived) and clears 0.8."""
    from src.strategy.eval.nlp import reproduce_alert_precision

    rows = {r.metric: r for r in reproduce_alert_precision()}
    assert rows["precision"].status == "reproduced"
    assert rows["precision"].value is not None and rows["precision"].value > 0.8


@pytest.mark.data
@pytest.mark.skipif(not _HAS_NER_CFG, reason="ner model_config absent (CI runner without weights)")
def test_dead_ner_classes_flags_zero_f1_types():
    """NR-07: entity types with ~0 frozen-eval B-F1 are flagged untrustworthy."""
    from src.strategy.eval.nlp import dead_ner_classes

    dead, row = dead_ner_classes()
    assert row.status == "flagged"
    assert row.value == float(len(dead))
    assert len(dead) >= 1


@pytest.mark.data
@pytest.mark.skipif(not _HAS_NER_MODEL, reason="ner weights absent (CI runner without weights)")
def test_ner_entity_f1_reproduces_headline():
    """Entity micro-F1 reproduces the frozen 0.4151 headline within tolerance (full-set optimistic)."""
    from src.strategy.eval.nlp import reproduce_ner

    rows = {r.metric: r for r in reproduce_ner()}
    assert rows["entity_f1"].status == "reproduced"
    assert rows["entity_f1"].value is not None and 0.3 < rows["entity_f1"].value < 0.6


@pytest.mark.data
@pytest.mark.skipif(not _HAS_RCM_CORPUS, reason="2025 rcm corpus absent (CI runner without data)")
def test_rcm_coverage_is_high():
    """The ported parser leaves few OTHER events; overall coverage clears 0.9."""
    from src.strategy.eval.nlp import reproduce_rcm

    rows = {r.metric: r for r in reproduce_rcm()}
    assert rows["coverage"].status == "reproduced"
    assert rows["coverage"].value is not None and rows["coverage"].value > 0.9
