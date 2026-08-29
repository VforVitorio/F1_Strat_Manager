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

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


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
