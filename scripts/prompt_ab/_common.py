"""Shared plumbing for the prompt A/B harness.

Everything here exists so the measurement runs against the SHIPPED objects. It
calls ``_build_orchestrator_prompt``, ``_get_orchestrator_llm`` and
``_assemble_recommendation`` directly and never reimplements them: an eval harness
that measures a private copy of the code is a recorded failure in this repo, and
two published numbers were produced that way.

--- WHERE TO CHANGE IF THE PROMPT GAINS A MEMORY PARAMETER ---
``inject_memory_block`` splices the block in by string surgery, above the
``RACE CONTEXT:`` heading. That is a harness-only workaround for the fact that
``_build_orchestrator_prompt`` has no ``memory_block`` argument yet. The moment it
gains one, delete the splice and pass the argument, or this harness starts
measuring a prompt assembled differently from the one production builds.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# The heading the memory block is spliced above: after the guard-rails and the
# STAY_OUT framing, immediately before the per-lap facts.
MEMORY_ANCHOR = "RACE CONTEXT:\n"


def load_env() -> str:
    """Load the repo ``.env`` and return the resolved provider.

    ``load_dotenv()`` with no argument searches upward from the CALLING FILE, and
    this file lives inside the repo but a caller may not, so the path is explicit.
    Getting this wrong is silent: the provider falls back to LM Studio and every
    call dies against ``localhost:1234``.
    """
    from dotenv import load_dotenv

    load_dotenv(REPO_ROOT / ".env")
    return os.environ.get("F1_LLM_PROVIDER", "lmstudio")


class Usage:
    """LangChain callback that totals tokens and calls across a run.

    ``with_structured_output`` returns the parsed model and drops the raw response,
    so the token counts are only reachable from a callback. Reported in the output
    JSON because "how much did this cost" is the first question asked of any
    measurement that spends API calls.
    """

    def __init__(self) -> None:
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.calls = 0

    # LangChain's BaseCallbackHandler protocol; subclassed lazily in `handler()`
    # so importing this module does not require langchain to be installed.
    def account(self, response: Any) -> None:
        self.calls += 1
        usage = (getattr(response, "llm_output", None) or {}).get("token_usage") or {}
        if not usage:
            for generation in response.generations:
                for item in generation:
                    metadata = getattr(item.message, "usage_metadata", None) or {}
                    usage = {
                        "prompt_tokens": metadata.get("input_tokens", 0),
                        "completion_tokens": metadata.get("output_tokens", 0),
                    }
        self.prompt_tokens += usage.get("prompt_tokens", 0)
        self.completion_tokens += usage.get("completion_tokens", 0)

    def handler(self):
        """A LangChain callback handler that feeds ``account``."""
        from langchain_core.callbacks import BaseCallbackHandler

        totals = self

        class _Handler(BaseCallbackHandler):
            def on_llm_end(self, response, **kwargs):  # noqa: ANN001, ANN003
                totals.account(response)

        return _Handler()

    def as_dict(self) -> dict[str, int]:
        return {
            "calls": self.calls,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
        }


def invoke_with_retry(llm, prompt: str, usage: Usage, attempts: int = 4):
    """Call the LLM, retrying transient provider failures.

    The shipped client is ``timeout=120, max_retries=1``, which is right for a live
    race lap and wrong for a 41-lap measurement pass: one timeout killed a pass at
    lap 22 during the audit. Retrying here rather than widening the client keeps
    the harness measuring the client production actually uses.
    """
    for attempt in range(attempts):
        try:
            return llm.invoke(prompt, config={"callbacks": [usage.handler()]})
        except Exception as exc:  # noqa: BLE001 - harness, any provider error is retryable
            if attempt == attempts - 1:
                raise
            print(f"  retry {attempt + 1} after {type(exc).__name__}", flush=True)
    raise AssertionError("unreachable")


def inject_memory_block(prompt: str, block: str | None) -> str:
    """Splice the memory block above ``RACE CONTEXT:``; a falsy block is a no-op."""
    if not block:
        return prompt
    index = prompt.index(MEMORY_ANCHOR)
    return prompt[:index] + block + prompt[index:]


def build_prompt(record: dict[str, Any], memory_block: str | None = None) -> str:
    """Build one lap's orchestrator prompt from a cached record."""
    from src.agents.strategy_orchestrator import _build_orchestrator_prompt

    prompt = _build_orchestrator_prompt(
        race_state=record["race_state"],
        mc_results=record["mc_results"],
        best_mc=record["best_mc"],
        pace_out=record["pace_out"],
        tire_out=record["tire_out"],
        situation_out=record["situation_out"],
        pit_out=record["pit_out"],
        radio_out=record["radio_out"],
        regulation_context=record["regulation_context"],
    )
    return inject_memory_block(prompt, memory_block)


def assemble(record: dict[str, Any], synthesis):
    """Run the real assembly so clamps and guards apply exactly as in production."""
    from src.agents.strategy_orchestrator import _assemble_recommendation

    return _assemble_recommendation(
        synthesis,
        record["pit_out"],
        record["mc_results"],
        record["regulation_context"],
        sc_currently_active=record["sc_active"],
        live_drivers=None,
        cliff_p50=record["tire_out"].laps_to_cliff_p50,
        total_laps=record["race_state"].total_laps,
    )


def checkpoint(path: Path, payload: dict[str, Any]) -> None:
    """Write the whole result after every lap.

    A 41-lap pass is roughly 25 minutes of API calls. Buffering to the end means one
    provider timeout costs the entire run, which is exactly what happened before
    this existed.
    """
    path.write_text(json.dumps(payload, indent=1), encoding="utf-8")


def recommendation_row(lap: int, recommendation) -> dict[str, Any]:
    """Flatten a recommendation into the JSON row the analysis reads."""
    row = {
        "lap": lap,
        "action": str(recommendation.action),
        "reasoning": recommendation.reasoning,
        "confidence": recommendation.confidence,
        "pit_lap_target": recommendation.pit_lap_target,
        "compound_next": recommendation.compound_next,
        "undercut_target": recommendation.undercut_target,
        "pace_mode": str(recommendation.pace_mode),
        "target_lap_time_s": recommendation.target_lap_time_s,
        "risk_posture": str(recommendation.risk_posture),
        "contingencies": [
            {
                "trigger": c.trigger,
                "switch_to": c.switch_to,
                "priority": c.priority,
                "rationale": c.rationale,
            }
            for c in recommendation.contingencies
        ],
        "key_risks": list(recommendation.key_risks),
        "expected_stint_end": recommendation.expected_stint_end,
    }
    return row
