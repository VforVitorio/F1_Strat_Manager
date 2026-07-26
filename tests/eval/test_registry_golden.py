"""Golden acceptance tests for the eval harness (#206).

Data-tier: these need ``data/models/`` + the labeled holdouts, so they are
skipped on CI runners (marked ``data`` and guarded like the other model-backed
tests) and run locally where the weights are present. They lock the harness's
retro-validation contract - the three claims #206 must keep true:

- the registry reconciles the pace 0.392 -> 0.4104 divergence to the canonical;
- calibration "finds" the known-broken pit P05-P95 coverage (177/252) as drift;
- reproduction re-derives overtake AUC-PR back to its published 0.5491.

They call the ``collect_*`` functions (pure, no report I/O) so the assertions
test the harness logic, not the markdown writer.

WARNING, and it has already cost this repo once. Because CI cannot run these, a
red test here is INVISIBLE to every pull request. The pit-coverage golden was
written against 0.7047, the holdout was regenerated to 0.7024 in the same week,
and the assertion sat failing for months with every PR going green over it
(#634). **Run this directory locally before promoting to main.** A green CI is
not evidence that these passed; it is evidence that they did not run.
"""

from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent.parent
_HAS_MODELS = (ROOT / "data" / "models" / "overtake_probability" / "model_config.json").exists()

pytestmark = [
    pytest.mark.data,
    pytest.mark.skipif(not _HAS_MODELS, reason="data/models/ absent (CI runner without weights)"),
]


def test_registry_reconciles_pace_divergence():
    """The registry pins pace canonical = 0.4104 and keeps 0.392 as superseded."""
    from src.strategy.eval.registry import collect_entries

    pace = [e for e in collect_entries() if e.model == "pace" and e.metric == "mae_test_s"]
    canonical = [e for e in pace if e.canonical]
    superseded = [e for e in pace if not e.canonical]

    assert len(canonical) == 1, "exactly one canonical pace entry"
    assert canonical[0].value == pytest.approx(0.4104)
    assert any(e.value == pytest.approx(0.392) for e in superseded), "0.392 kept for provenance"


def test_calibration_flags_pit_coverage_drift():
    """The pit P05-P95 coverage surfaces and is flagged as drift.

    The drift assertion is what this test is FOR. The exact value is provenance,
    and it moved once: this test was written against 0.7047, then `ccc213d`
    regenerated the N15 holdout from raw laps, computed 0.7024, wrote that into
    the report, and left the assertion on the old number. So the golden was born
    red and stayed red, because `tests/eval/` is data-gated and CI runners have
    no dataset to run it with. Nobody was ever told (#634).

    177/252 is the current value, re-derived here rather than copied: the alias
    fix in #629 was checked against it and moves it by exactly nothing (it moved
    P50 MAE by -0.0045 s instead).
    """
    from src.strategy.eval.calibration import collect_results

    pit = [
        r for r in collect_results() if r.model == "pit_duration" and r.metric == "p05_p95_coverage"
    ]
    assert len(pit) == 1
    assert pit[0].value == pytest.approx(177 / 252, abs=1e-6)
    assert pit[0].status == "drift", "coverage below 0.90 nominal must flag drift"


def test_reproduction_matches_overtake_auc_pr():
    """Overtake AUC-PR re-derives to the published 0.5491 within tolerance."""
    from src.strategy.eval.reproduce import collect_results

    overtake = [r for r in collect_results() if r.model == "overtake" and r.metric == "auc_pr_test"]
    assert len(overtake) == 1
    assert overtake[0].status == "reproduced"
    assert overtake[0].reproduced == pytest.approx(0.5491, abs=0.01)
