"""Golden tests for the #304 NLP eval harness.

Hermetic: asserts the gated-stage contract (intent blocked on #303, alert
precision pending a ground-truth set). The heavy RoBERTa sentiment reproduction
is validated by running ``f1-eval nlp`` (it loads a 1.4 GB checkpoint), not in
the routine test suite.
"""

from __future__ import annotations


def test_nlp_gated_stages_carry_their_blockers():
    """Intent is blocked on #303; alert precision + NER + RCM are pending."""
    from src.strategy.eval.nlp import _gated_stages

    by_stage = {r.stage: r for r in _gated_stages()}
    assert by_stage["intent"].status == "blocked"
    assert "303" in by_stage["intent"].detail
    assert by_stage["alert_precision"].status == "pending"
    assert by_stage["ner"].status == "pending"
    assert by_stage["rcm"].status == "pending"
