"""The CLI pre-warm must actually load the NLP models, not just import them.

`_prewarm_agents` exists to move model-loading latency out of the first lap and
to keep transformers' LOAD REPORT off the Live display, and it used to get the
three N24 checkpoints for free: `from src.agents.radio_agent import CFG` built
them as a side effect of `RadioAgentCFG.__post_init__`. Making that load lazy
(#1118) turned the import into a no-op and moved the 14 s somewhere much worse.

Measured on Lusail laps 20-22, `--no-llm --no-real-radios --radio-every 1`, which
is the smallest command that reaches the NLP pipeline:

| | LOAD REPORT lines in the output | lap loop | avg/lap |
|---|---|---|---|
| before the lazy property | 0 | 3.1 s | 1.0 s |
| lazy, pre-warm not updated | 2 | 21.7 s | 7.1 s |
| lazy, pre-warm reading .pipeline | 0 | 2.1 s | 0.6 s |

The middle row is the regression: the load landed inside the Live loop, where
`_devnull_fds()` does not reach, so transformers printed nine rows of a weight
report straight through the rendered panels.

The pre-warm's own docstring already promised this. It lists three jobs, one of
which is "suppresses tqdm progress bars and NLP weight LOAD REPORTs at C level",
and says the radio CFG stays pre-warmed unconditionally. A promise in a docstring
is not a check, which is why this file exists.

Nothing here loads a real checkpoint: the three loaders are replaced with stubs,
so what is under test is whether `_prewarm_agents` reaches them at all.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

from tests.conftest import skip_no_tire_models

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Importing the CLI reaches strategy_orchestrator and therefore tire_agent, whose
# module-level config reads data/models/tire_degradation/routing_config.json and
# raises FileNotFoundError when the weights are absent. That is an OSError, not an
# ImportError, so importorskip does not catch it and the job goes red on a runner
# that was never meant to have the file. The marker goes on the tests that import
# the CLI, not on the module, because the source check below has to keep running
# where the weights do not exist. Otherwise the only guard against this regression
# would be one that never executes in CI.
CLI_SOURCE = ROOT / "scripts" / "run_simulation_cli.py"


@pytest.fixture
def stubbed_radio_cfg(monkeypatch: pytest.MonkeyPatch):
    """A fresh un-built CFG whose loaders are free and counted."""
    radio_agent = pytest.importorskip("src.agents.radio_agent")
    calls: list[str] = []

    cfg = radio_agent.RadioAgentCFG.__new__(radio_agent.RadioAgentCFG)
    cfg.device = "cpu"
    cfg.__post_init__()
    monkeypatch.setattr(
        cfg, "_load_sentiment_model", lambda *_: (calls.append("sentiment"), ("t", "m"))[1]
    )
    monkeypatch.setattr(
        cfg, "_load_intent_model", lambda *_: (calls.append("intent"), ("m", ()))[1]
    )
    monkeypatch.setattr(
        cfg, "_load_ner_model", lambda *_: (calls.append("ner"), ("t", "m", {}, {}))[1]
    )
    monkeypatch.setattr(radio_agent, "CFG", cfg)
    return cfg, calls


def _prewarm_ast() -> ast.FunctionDef:
    """The `_prewarm_agents` definition, parsed rather than imported."""
    tree = ast.parse(CLI_SOURCE.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "_prewarm_agents":
            return node
    raise AssertionError("_prewarm_agents is gone from run_simulation_cli.py")


def _suppressed_block(fn: ast.FunctionDef) -> ast.With:
    """The `with _devnull_fds():` body, which is where output is safe to emit."""
    for node in ast.walk(fn):
        if isinstance(node, ast.With):
            for item in node.items:
                call = item.context_expr
                if isinstance(call, ast.Call) and getattr(call.func, "id", None) == "_devnull_fds":
                    return node
    raise AssertionError("_prewarm_agents no longer wraps its work in _devnull_fds()")


def test_the_prewarm_reads_the_pipeline_inside_the_suppressed_block() -> None:
    """The structural half, which runs on a machine with no weights at all.

    Parsed, not grepped: what matters is that the read happens INSIDE the
    `_devnull_fds()` context, and a text search cannot tell that from a read two
    lines after it, which is the shape that leaks the load report.
    """
    reads = [
        node
        for node in ast.walk(_suppressed_block(_prewarm_ast()))
        if isinstance(node, ast.Attribute) and node.attr == "pipeline"
    ]
    assert reads, (
        "nothing in the suppressed block reads .pipeline, so the NLP models load "
        "on the first lap instead, printing their load report into the Live view"
    )


@skip_no_tire_models
@pytest.mark.parametrize("no_llm", [True, False])
def test_the_prewarm_builds_the_radio_pipeline(stubbed_radio_cfg, no_llm: bool) -> None:
    """Both modes, because the docstring says this one is unconditional.

    `--no-llm` deliberately skips the tire, situation and pit singletons (#389).
    The radio CFG is named as the exception, along with the pace agent, since
    both run in either mode.
    """
    cli = pytest.importorskip("scripts.run_simulation_cli")
    cfg, calls = stubbed_radio_cfg

    cli._prewarm_agents(no_llm=no_llm)

    assert cfg._pipeline is not None, "the pre-warm left the NLP models unbuilt"
    assert sorted(calls) == ["intent", "ner", "sentiment"]


@skip_no_tire_models
def test_the_prewarm_swallows_a_failing_load(stubbed_radio_cfg, monkeypatch) -> None:
    """It is best-effort by design, and must stay that way.

    The `except Exception: pass` around the block is deliberate: a machine with
    no weights should reach the run loop and fail there with a real message,
    not die during a warm-up. Reading `.pipeline` raises where importing the
    name did not, so this is a new way for the pre-warm to throw.
    """
    cli = pytest.importorskip("scripts.run_simulation_cli")
    cfg, _ = stubbed_radio_cfg

    def boom(*_args: object) -> None:
        raise FileNotFoundError("no checkpoint here")

    monkeypatch.setattr(cfg, "_load_sentiment_model", boom)

    cli._prewarm_agents(no_llm=True)
    assert cfg._pipeline is None
