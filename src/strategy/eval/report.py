"""E-15 provenance-header contract shared by every eval report.

Every report the harness emits (metrics registry, calibration, regeneration)
carries the same header so any number can be traced back to the exact
artifacts, dataset snapshot, regulation era and harness version that produced
it. Without this stamp a 2026 retraining makes every old report ambiguous
(which model? which season? which code?).

WHERE TO CHANGE IF THE PROVENANCE CONTRACT CHANGES:
- add a field to ``ReportHeader`` + set it in ``build_header``; every report
  writer downstream picks it up for free (registry.py, calibration.py, ...).
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.f1_strat_manager.data_cache import _find_repo_root

SCHEMA_VERSION = "1"

# The regulation era these numbers are valid for, and NOT the range of seasons they
# were measured on. The ground-effect ruleset runs 2022 to 2025, so a number measured
# on 2023-2025 cars is valid across it; but there is no 2022 season anywhere in
# `data/`, every model trains on 2023-2024 and tests on 2025, the measured Monte
# Carlo tables cover 2023-2025, and the RAG corpus holds the 2023/24/25 rulebooks
# only. Two independent readers took this tag for a data range, so it is worth the
# sentence: a 2022-specific question has no authoritative source in this repo.
ERA_TAG = "2022-2025"


def _harness_sha() -> str:
    """Short git SHA of the repo the harness runs from, or ``unknown``.

    Uses ``git describe --always --dirty --exclude='*'`` rather than a bare
    ``rev-parse --short HEAD``. The ``--exclude`` pattern matches every tag,
    so ``describe`` has no tag left to describe from and falls back to the
    same short hash ``--always`` gives on its own; without it, ``describe``
    walks back to the nearest reachable tag instead and stamps a long,
    tag-relative description (e.g. ``legacy-2026-07-13-1098-g8b6cb305``) in
    place of the short sha the header is supposed to pin. The one addition
    over a bare ``rev-parse`` is the ``-dirty`` suffix, appended when the
    working tree does not match the stamped commit: reports are regenerated
    on a dirty tree before the change that motivated the regeneration gets
    committed, so the sha used to be silently one commit stale with nothing
    marking it (#1152).

    Returns ``unknown`` when running outside a checkout (e.g. an installed
    tool venv) so a report is never blocked on git being available.
    """
    repo = _find_repo_root()
    if repo is None:
        return "unknown"
    try:
        out = subprocess.run(
            ["git", "-C", str(repo), "describe", "--always", "--dirty", "--exclude=*"],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
        return out.stdout.strip()
    except (subprocess.SubprocessError, OSError):
        return "unknown"


def artifact_hash(path: Path) -> str:
    """First 12 hex chars of the SHA-256 of a model artifact.

    Short enough to stay readable inside a committed markdown header, long
    enough to pin an artifact version unambiguously in practice.
    """
    digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    return digest[:12]


@dataclass
class ReportHeader:
    """Provenance stamp attached to every eval report (E-15).

    Invariants:
    - ``harness_sha`` pins the code and ``artifacts`` pins the model weights;
      together they make a report reproducible, and a ``-dirty`` suffix on
      ``harness_sha`` says plainly when that does NOT hold.
    - ``generated_at`` is provenance only and is NOT part of report equality
      (two runs of the same code on the same data are "equal" bar the clock).
    - ``llm`` is ``none`` for pure-ML reports; NLP/orchestrator reports that
      call a model record ``provider/model/version`` (never Anthropic).
    """

    harness_sha: str
    dataset: str
    seed_policy: str
    era_tag: str = ERA_TAG
    llm: str = "none"
    schema_version: str = SCHEMA_VERSION
    generated_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")
    )
    artifacts: dict[str, str] = field(default_factory=dict)


def build_header(
    *,
    dataset: str,
    seed_policy: str = "deterministic",
    llm: str = "none",
    artifacts: dict[str, Path] | None = None,
) -> ReportHeader:
    """Assemble a report header, hashing each artifact path that exists.

    Missing artifact paths are skipped rather than raising, so a partial
    install still produces a report (with a shorter artifact list) instead of
    crashing the whole run.
    """
    hashed = {
        name: artifact_hash(path) for name, path in (artifacts or {}).items() if Path(path).exists()
    }
    return ReportHeader(
        harness_sha=_harness_sha(),
        dataset=dataset,
        seed_policy=seed_policy,
        llm=llm,
        artifacts=hashed,
    )


def eval_reports_dir() -> Path:
    """``documents/eval_reports/`` (created on demand): committed, diffable.

    Falls back to a user-cache directory when running outside a checkout so
    the writer never fails; in that mode the reports simply are not version
    controlled.
    """
    repo = _find_repo_root()
    base = (
        (repo / "documents" / "eval_reports")
        if repo
        else (Path.home() / ".f1-strat" / "eval_reports")
    )
    base.mkdir(parents=True, exist_ok=True)
    return base


def write_report(
    name: str,
    header: ReportHeader,
    table_md: str,
    payload: dict[str, Any],
) -> tuple[Path, Path]:
    """Write a report as a diffable markdown table + a machine-readable JSON.

    The markdown is the human / paper-facing citable artifact; the JSON is
    what downstream consumers (#213 docs reconciliation, #304 NLP harness,
    the golden tests) read so they never have to re-parse prose.

    Returns the ``(markdown_path, json_path)`` pair.
    """
    out = eval_reports_dir()
    md_path = out / f"{name}.md"
    json_path = out / f"{name}.json"
    md_path.write_text(_render_md(name, header, table_md), encoding="utf-8")
    json_path.write_text(
        json.dumps({"header": asdict(header), **payload}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return md_path, json_path


def _render_md(name: str, header: ReportHeader, table_md: str) -> str:
    """Render a report: title, provenance header block, then the table body."""
    artifacts = ", ".join(f"{k}=`{v}`" for k, v in header.artifacts.items()) or "none"
    lines = [
        f"# {name}",
        "",
        f"- harness `{header.harness_sha}` · schema v{header.schema_version} · generated {header.generated_at}",
        f"- era {header.era_tag} · dataset {header.dataset} · seed {header.seed_policy} · llm {header.llm}",
        f"- artifacts: {artifacts}",
        "",
        table_md,
        "",
    ]
    return "\n".join(lines)
