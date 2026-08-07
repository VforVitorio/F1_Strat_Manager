"""Count the LLM tokens a run of the agent stack actually spends, per model.

Every figure this project publishes today comes from ``profile="no-llm"``, which
makes zero API calls. Measuring the shipped path means paying for it, and the
price is unknown until it is measured: the stack builds five distinct
``ChatOpenAI`` clients (four sub-agents plus the orchestrator), the sub-agents
are LangGraph ReAct loops that call out an unbounded number of times per lap,
and nothing in the CLI reports any of it.

HOW IT ATTACHES, AND WHY THAT WAY
---------------------------------
By patching ``ChatOpenAI.__init__`` to append a handler to ``callbacks``, once,
on the class object every agent module imports. The two alternatives were both
worse:

- Passing ``config={"callbacks": [...]}`` at each call site means editing the
  agents, and the sub-agent calls happen inside LangGraph's ReAct loop where
  there is no call site to edit.
- ``register_configure_hook`` (what ``langchain_community``'s
  ``get_openai_callback`` uses) rides a ``ContextVar``, which does not follow
  the run into a worker thread. ``langchain_community`` is not installed here
  anyway.

The patch is idempotent and additive: a client that was given its own callbacks
keeps them.

WHAT IT DOES NOT DO
-------------------
It does not price anything. Token counts are measured; a price per million is a
published number that changes, so it lives in the caller's report next to the
date it was read, never hard-coded here where it would silently rot into a
wrong cost. It also does not count Whisper, the embeddings or the local
LM Studio path, which cost wall clock rather than API money.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler


@dataclass
class ModelUsage:
    """Token totals for one model name, accumulated over a run."""

    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_prompt_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def as_dict(self) -> dict[str, int]:
        result = {
            "calls": self.calls,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "cached_prompt_tokens": self.cached_prompt_tokens,
            "total_tokens": self.total_tokens,
        }
        return result


class TokenMeter(BaseCallbackHandler):
    """A LangChain callback that tallies token usage per model name.

    Thread-safe because the ReAct sub-agents are free to run their tool loops
    off the calling thread, and a lost increment would under-report the cost,
    which is the direction that matters least to notice and most to get wrong.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.by_model: dict[str, ModelUsage] = {}
        self.unattributed_calls = 0

    def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        """Record one completed LLM call.

        Reads ``llm_output`` first because that is where ``langchain_openai``
        puts the provider's own accounting, including the cached-prompt split.
        Falls back to the message's ``usage_metadata`` for clients that do not
        populate ``llm_output``. A call that carries neither is counted in
        ``unattributed_calls`` rather than dropped: an invisible call is how a
        cost estimate comes out low and confident.
        """
        usage, model = self._extract(response)
        if usage is None:
            with self._lock:
                self.unattributed_calls += 1
            return

        with self._lock:
            entry = self.by_model.setdefault(model, ModelUsage())
            entry.calls += 1
            entry.prompt_tokens += int(usage.get("prompt_tokens", 0) or 0)
            entry.completion_tokens += int(usage.get("completion_tokens", 0) or 0)
            entry.cached_prompt_tokens += int(usage.get("cached_tokens", 0) or 0)

    @staticmethod
    def _extract(response: Any) -> tuple[dict[str, Any] | None, str]:
        """Pull (token usage, model name) out of an ``LLMResult``, or (None, '')."""
        llm_output = getattr(response, "llm_output", None) or {}
        model = str(llm_output.get("model_name") or "unknown")

        token_usage = llm_output.get("token_usage")
        if token_usage:
            details = token_usage.get("prompt_tokens_details") or {}
            cached = details.get("cached_tokens") if isinstance(details, dict) else None
            usage = dict(token_usage)
            usage["cached_tokens"] = cached or 0
            return usage, model

        message = TokenMeter._first_message(response)
        metadata = getattr(message, "usage_metadata", None)
        if not metadata:
            return None, model

        model = str(getattr(message, "response_metadata", {}).get("model_name") or model)
        input_details = metadata.get("input_token_details") or {}
        usage = {
            "prompt_tokens": metadata.get("input_tokens", 0),
            "completion_tokens": metadata.get("output_tokens", 0),
            "cached_tokens": input_details.get("cache_read", 0),
        }
        return usage, model

    @staticmethod
    def _first_message(response: Any) -> Any:
        """The first generation's message, or None when the shape is unexpected."""
        generations = getattr(response, "generations", None) or []
        if not generations or not generations[0]:
            return None
        return getattr(generations[0][0], "message", None)

    def totals(self) -> ModelUsage:
        """One ``ModelUsage`` summing every model, for the headline row."""
        combined = ModelUsage()
        with self._lock:
            for entry in self.by_model.values():
                combined.calls += entry.calls
                combined.prompt_tokens += entry.prompt_tokens
                combined.completion_tokens += entry.completion_tokens
                combined.cached_prompt_tokens += entry.cached_prompt_tokens
        return combined

    def as_dict(self) -> dict[str, Any]:
        """The full tally, ready to serialise into a report."""
        with self._lock:
            per_model = {name: usage.as_dict() for name, usage in self.by_model.items()}
        result = {
            "by_model": per_model,
            "total": self.totals().as_dict(),
            "unattributed_calls": self.unattributed_calls,
        }
        return result


_installed_meter: TokenMeter | None = None


def install() -> TokenMeter:
    """Attach a process-wide meter to every ``ChatOpenAI`` built from now on.

    Returns the existing meter on a second call rather than stacking a second
    patch, so an import-order accident cannot double-count.

    --- WHERE TO CHANGE IF THINGS MOVE ---
    - The agents build their clients in ``_get_*_llm`` style functions in
      ``src/agents/``; all of them import ``ChatOpenAI`` from ``langchain_openai``,
      which is why patching the class reaches all five.
    - A client constructed BEFORE this call is not metered. Call ``install()``
      before importing or invoking anything in ``src/agents/``.
    """
    global _installed_meter
    if _installed_meter is not None:
        return _installed_meter

    import langchain_openai

    meter = TokenMeter()
    original_init = langchain_openai.ChatOpenAI.__init__

    def init_with_meter(self: Any, *args: Any, **kwargs: Any) -> None:
        callbacks = list(kwargs.get("callbacks") or [])
        callbacks.append(meter)
        kwargs["callbacks"] = callbacks
        original_init(self, *args, **kwargs)

    langchain_openai.ChatOpenAI.__init__ = init_with_meter
    _installed_meter = meter
    return meter


def _self_check() -> None:
    """Smallest runnable check: the extractor reads both usage shapes."""

    class FakeMessage:
        usage_metadata = {
            "input_tokens": 100,
            "output_tokens": 20,
            "input_token_details": {"cache_read": 64},
        }
        response_metadata = {"model_name": "gpt-4.1-mini"}

    class FakeGeneration:
        message = FakeMessage()

    class FakeResponse:
        llm_output = {
            "model_name": "gpt-4.1-mini",
            "token_usage": {
                "prompt_tokens": 100,
                "completion_tokens": 20,
                "prompt_tokens_details": {"cached_tokens": 64},
            },
        }
        generations = [[FakeGeneration()]]

    class FakeResponseNoLlmOutput:
        llm_output = None
        generations = [[FakeGeneration()]]

    class FakeResponseEmpty:
        llm_output = None
        generations = []

    meter = TokenMeter()
    meter.on_llm_end(FakeResponse())
    meter.on_llm_end(FakeResponseNoLlmOutput())
    meter.on_llm_end(FakeResponseEmpty())

    total = meter.totals()
    assert total.calls == 2, total
    assert total.prompt_tokens == 200, total
    assert total.completion_tokens == 40, total
    assert total.cached_prompt_tokens == 128, total
    assert meter.unattributed_calls == 1, meter.unattributed_calls
    assert set(meter.by_model) == {"gpt-4.1-mini"}, meter.by_model
    print("token_meter self-check OK")


if __name__ == "__main__":
    _self_check()
