"""`--provider` overrides `.env`; its absence must not (#805).

`.env.example` ships `F1_LLM_PROVIDER=openai` and `INSTALL.md` tells users to copy it,
but the CLI wrote `os.environ["F1_LLM_PROVIDER"] = args.provider` unconditionally and the
flag defaulted to `"lmstudio"`. So a correctly configured `.env` was ignored on every run,
every lap failed with `APIConnectionError`, and the process still exited 0.

An argparse default is indistinguishable from an explicit value at the call site, which is
the whole mechanism: the fix is that the flag defaults to None and the write only happens
when it was actually passed.
"""

from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent.parent
_CLI = ROOT / "scripts" / "run_simulation_cli.py"


def _provider_block() -> str:
    """The `--provider` argument declaration, as source.

    Asserted on the text rather than by importing the CLI, because `_parse_args` builds
    its parser inline and importing `run_simulation_cli` reads model weights, so a unit
    test cannot construct the parser. What is in question is not how argparse treats
    `default=None`, it is whether the code says it.
    """
    source = _CLI.read_text(encoding="utf-8")
    start = source.index('"--provider",')
    return source[start : source.index('p.add_argument(\n        "--interval"', start)]


def test_the_flag_defaults_to_unset_not_to_a_provider():
    """A default here outranks `.env`, which is exactly the bug."""
    block = _provider_block()

    assert "default=None," in block, (
        "an argparse default cannot be told apart from an explicit choice, so any "
        f"default here silently wins over the .env the project asks users to configure. "
        f"Block reads:\n{block}"
    )


@pytest.mark.parametrize("provider", ["openai", "lmstudio"])
def test_both_providers_are_still_accepted(provider):
    """Passing it must keep working, including passing the fallback explicitly."""
    assert provider in _provider_block()


def test_the_env_is_only_written_when_the_flag_was_passed():
    """The guard, asserted on the source rather than by importing the whole CLI.

    Importing `run_simulation_cli` builds the agent stack and reads model weights, so a
    unit test cannot call the function this guard lives in. What matters is the relation:
    the write must be conditional on the flag being set, not merely on LLM mode.
    """
    source = _CLI.read_text(encoding="utf-8")

    assert 'os.environ["F1_LLM_PROVIDER"] = args.provider' in source, (
        "the write moved or was renamed; this test is describing code that no longer exists"
    )
    write_at = source.index('os.environ["F1_LLM_PROVIDER"] = args.provider')
    guard = source[:write_at].rsplit("if ", 1)[-1]

    assert "args.provider is not None" in guard, (
        f"the env write is guarded by {guard.splitlines()[0]!r}, which does not check "
        f"whether --provider was actually passed"
    )
