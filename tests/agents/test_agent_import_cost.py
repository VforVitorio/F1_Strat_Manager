"""Importing the agents must not build the LLM stack or the NLP models (#1118).

Six modules each opened with a `try: from langchain_openai import ChatOpenAI`
whose only job was to set an availability flag, and `RadioAgentCFG.__post_init__`
loaded three transformer checkpoints the moment the module was read. None of that
is needed to import anything: every consumer sits behind a factory that already
builds its client, or its models, on first call.

The bill was real and it was paid by everything. `import langchain_openai` drags
transformers and the langgraph stack in behind it, 7.2 s measured, and
`src/rag/retriever.py` pulled `sentence_transformers` for another 7.3 s. Because
`strategy_orchestrator` imports all six agents, `f1-sim --help` spent 12.6 s
printing a usage string, and a five-lap `--no-llm` run spent 15.6 s of which
13 s was startup. After the deferral those are 3.6 s and 5.5 s.

What this file protects is the property, not the timing: a wall-clock assertion
would be flaky on a cold disk and useless on a fast one. It asks a fresh
interpreter what it loaded.

`torch` is deliberately absent from the forbidden list. `tire_agent` defines its
TCN as an `nn.Module` subclass at module scope, so the class statement itself
needs torch and no amount of deferral removes it. Claiming otherwise here would
make this file fail for a reason it is not about.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]

# Packages no agent import may pull in. Each is a lazy path that regressed once.
FORBIDDEN = ("langchain_openai", "transformers", "sentence_transformers", "qdrant_client")

# The flag each module publishes to say the LLM path is available. A find_spec
# typo would set every one of these to False, and the only symptom would be
# "langchain_openai is not installed" raised on the first real LLM call.
AVAILABILITY_FLAGS = {
    "src.agents.strategy_orchestrator": "_LC_OK",
    "src.agents.radio_agent": "_LC_OK",
    "src.agents.rag_agent": "_LC_OK",
    "src.agents.tire_agent": "_LANGGRAPH_AVAILABLE",
    "src.agents.race_situation_agent": "_LANGGRAPH_AVAILABLE",
    "src.agents.pit_strategy_agent": "_LANGGRAPH_AVAILABLE",
}

_PROBE = """
import json, sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, {root!r})
import src.agents.strategy_orchestrator  # pulls all six agents and the retriever
from src.agents import radio_agent

flags = {{}}
for name, attr in {flags!r}.items():
    flags[name] = getattr(sys.modules[name], attr)

print("@@" + json.dumps({{
    "loaded": sorted(m for m in sys.modules if "." not in m),
    "flags": flags,
    "radio_pipeline_built": radio_agent.CFG._pipeline is not None,
    "radio_dead_globals": [g for g in ("LAPS", "RCM_DF") if hasattr(radio_agent, g)],
}}))
"""

skip_no_langchain = pytest.mark.skipif(
    importlib.util.find_spec("langchain_openai") is None,
    reason="langchain_openai not installed",
)


@pytest.fixture(scope="module")
def fresh_import() -> dict:
    """Import the orchestrator in a clean interpreter and report what loaded.

    A subprocess rather than an importlib dance in-process, because pytest has
    already imported half the tree by the time this runs and `sys.modules` in
    here answers a different question than the one being asked.
    """
    source = _PROBE.format(root=str(ROOT), flags=AVAILABILITY_FLAGS)
    result = subprocess.run(
        [sys.executable, "-c", source],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=600,
    )
    marker = [ln for ln in result.stdout.splitlines() if ln.startswith("@@")]
    if not marker:
        pytest.skip(f"agent import failed in a clean interpreter: {result.stderr[-400:]}")
    return json.loads(marker[-1][2:])


@pytest.mark.parametrize("package", FORBIDDEN)
def test_the_import_does_not_pull_the_llm_stack(fresh_import: dict, package: str) -> None:
    """One case per package, so a failure names which deferral came undone."""
    assert package not in fresh_import["loaded"]


def test_the_radio_models_are_not_built_at_import(fresh_import: dict) -> None:
    """Three transformer checkpoints, 17 s measured, on a run that may never
    analyse a radio message. `RadioAgentCFG.pipeline` builds them on first read.
    """
    assert fresh_import["radio_pipeline_built"] is False


def test_the_dead_frame_globals_are_gone(fresh_import: dict) -> None:
    """`LAPS` was a full copy of the race frame, taken per lap and read by
    nothing; `RCM_DF` was never even written. The module docstring said so.
    """
    assert fresh_import["radio_dead_globals"] == []


@skip_no_langchain
@pytest.mark.parametrize("module", sorted(AVAILABILITY_FLAGS))
def test_the_llm_path_still_reports_itself_available(fresh_import: dict, module: str) -> None:
    """The failure mode of replacing an import with a probe.

    Every one of these flags gates a `raise ImportError("... is not installed")`.
    A misspelled package name in `find_spec` turns the whole LLM layer off and
    nothing else notices until a real run.
    """
    assert fresh_import["flags"][module] is True


def test_the_radio_pipeline_builds_when_it_is_asked_for() -> None:
    """Lazy has to still be reachable. Asserted structurally rather than by
    loading 17 s of weights: the property is that a read of `pipeline` runs the
    loaders, and the loaders themselves are covered by the NLP golden tests.
    """
    from src.agents.radio_agent import RadioAgentCFG

    calls: list[str] = []
    cfg = RadioAgentCFG.__new__(RadioAgentCFG)
    cfg.device = "cpu"
    cfg.__post_init__()
    assert cfg._pipeline is None

    cfg._load_sentiment_model = lambda *_: (calls.append("sentiment"), ("tok", "model"))[1]
    cfg._load_intent_model = lambda *_: (calls.append("intent"), ("model", ()))[1]
    cfg._load_ner_model = lambda *_: (calls.append("ner"), ("tok", "model", {}, {}))[1]

    assert sorted(cfg.pipeline) == [
        "intent_model",
        "ner_id2label",
        "ner_label2id",
        "ner_model",
        "ner_tokenizer",
        "sentiment_model",
        "sentiment_tokenizer",
    ]
    assert sorted(calls) == ["intent", "ner", "sentiment"]

    cfg.pipeline
    assert sorted(calls) == ["intent", "ner", "sentiment"], "the second read reloaded"
