"""Additive evaluation harness for the F1 StratLab ML predictors (issue #206).

The measurement foundation the paper builds on: a reproducible metrics
registry (E-08) + calibration verification (E-03), launched by the ``f1-eval``
console script. Nothing here touches the UNTOUCHABLE trees (``src/agents/``
internals, ``notebooks/``, ``run_simulation_cli.py``); it only reads frozen
artifacts under ``data/models/`` and the labeled holdouts under
``data/processed/``.

Public API:
- ``build_registry`` / ``load_registry`` - the consolidated metrics table.
- ``build_calibration_report`` - reliability/Brier/ECE + quantile coverage.
- ``build_reproduction_report`` - headline numbers re-derived vs the configs.
"""

from src.strategy.eval.calibration import build_calibration_report
from src.strategy.eval.hygiene import build_hygiene_report
from src.strategy.eval.nlp import build_nlp_report
from src.strategy.eval.registry import build_registry, load_registry
from src.strategy.eval.reproduce import build_reproduction_report

__all__ = [
    "build_registry",
    "load_registry",
    "build_calibration_report",
    "build_reproduction_report",
    "build_hygiene_report",
    "build_nlp_report",
]
