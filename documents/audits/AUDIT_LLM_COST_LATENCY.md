# AUDIT - LLM cost & latency (provider layer across all surfaces)

**Scope:** every LLM call in the system: which layer makes it, with which model, how many tokens, how often, and what it costs in latency (all providers) and money (OpenAI). Covers per-layer model selection vs the intended policy, token budgets, prompt/response caching, per-surface critical-path latency, and the provider abstraction (OpenAI / LM Studio: timeouts, retries, streaming, fallback).
**Hard constraints honored in every remedy:** plan only, no code; `src/agents/` internals, `scripts/run_simulation_cli.py` (PMV) and `notebooks/**` UNTOUCHABLE (additive / config-side only); LLM = OpenAI or LM Studio, never Anthropic; every model recommendation stays within OpenAI / LM Studio.
**Inputs read:** `src/agents/{strategy_orchestrator,pace_agent,tire_agent,race_situation_agent,pit_strategy_agent,radio_agent,rag_agent}.py`, `src/agents/README.md`, `src/telemetry/backend/services/chatbot/{llm_service,chat_engine,mcp_bridge}.py`, `src/telemetry/backend/services/simulation/simulator.py`, `src/telemetry/backend/api/v1/endpoints/voice.py`, `src/telemetry/backend/mcp_tools.py`, `src/arcade/{strategy,strategy_pipeline}.py`, `scripts/run_simulation_cli.py` (read only), `.env.example`, sibling audits (`AUDIT_P2B_CORE_COMPUTE.md`, `AUDIT_SECURITY.md`, `AUDIT_ML_AGENTS_EVAL.md`, `AUDIT_P4_CLI.md`, `AUDIT_DOCS_ACCURACY.md`).
**Division of labor with siblings:** P2b owns *how many LLM turns the engine makes per lap* (probe duplication F1, ReAct turn inflation F3, RAG cache F4, silent-radio F5, N31 cadence F11). Security owns the *abuse* angle (S-2 cost-cap/allowlist, S-5 backend timeout, S-7 rate limiting). This audit owns everything in between: **model policy and configurability, token budgets and money cost, prompt-cache exploitation, provider-client hardening (timeouts/retries/streaming/fallback), and cost observability.** Where a lever belongs to a sibling it is cross-referenced, not re-registered.

---

## 1. Executive summary

The system runs two **independent LLM stacks**: LangChain `ChatOpenAI` singletons inside the seven agents (`src/agents/`), and a raw-`requests` service in the backend chat (`llm_service.py`). Both are provider-agnostic (OpenAI / LM Studio via `F1_LLM_PROVIDER`) and both are consistent with the intended per-layer model policy in their **hardcoded defaults**: sub-agents N25-N29 and N30 default to `gpt-4.1-mini`, the orchestrator N31 and the chat default to `gpt-5.4-mini`. The policy is therefore implemented, but it is implemented as **eight scattered signature/CFG defaults with no per-layer override**, and `src/agents/README.md:168` documents a third, stale answer (`gpt-4.1`).

The dominant cost findings:

1. **Latency risk is unbounded.** No agent-side `ChatOpenAI` sets a timeout or retry policy (7 construction sites), so every call inherits the openai-python defaults (600 s timeout, 2 retries): a stalled LM Studio can pin one lap for up to ~30 minutes. The backend service is worse: `DEFAULT_TIMEOUT = None` for LM Studio (`llm_service.py:40`, already SECURITY S-5). This is the single cheapest high-value fix in the whole LLM layer.
2. **Blocking calls sit inside the async event loop.** `chat_engine` admits it in its own docstring: the LLM provider calls are synchronous `requests` executed in the FastAPI event loop. One slow chat turn freezes every concurrent consumer of the backend, including the SSE simulation stream and voice.
3. **Money cost is unmeasured and unmeasurable today.** Token `usage` from responses is discarded everywhere except a single `total_tokens` field on the chat `done` event (`chat_engine.py:537`). There is no per-layer ledger, no per-race cost report, and no visibility into OpenAI cached-token discounts. Every number in this audit's budget table is an estimate that Phase 0 must replace with measurements.
4. **Prompt caching is left on the table.** The N31 synthesis prompt interleaves ~1,300 tokens of static text (guardrails, rubric, field spec) with dynamic blocks in a single user message, which caps OpenAI's automatic prefix cache at the first dynamic byte. Restructured static-first (system message), the N31 call becomes ~60-70% cacheable input at OpenAI's cached-token discount, on every lap.
5. **The chat quietly doubles its own cost.** Once a conversation exceeds 10 messages, `build_messages` fires a hidden *extra* LLM call to summarize old history (`llm_service.py:_compress_chat_history`), and never persists the summary, so **every subsequent turn pays the compression call again**.

Estimated OpenAI money cost is modest at mini-class rates (order of ~$1-2 per full 57-lap CLI race in LLM mode, before the P2b turn diet), so the primary optimization currency is **latency and robustness**, with cost observability as the enabler for both the ML-eval ablations and the security cost caps.

Plan: 5 phases, all additive/config-side. Phase 0 builds the token/cost ledger and baselines a race + a chat session. Phase 1 hardens the provider layer (timeouts, retries, async offload, preflight). Phase 2 centralizes model selection per layer (env-driven, one source of truth). Phase 3 restructures prompts for cache hits and removes the hidden chat costs. Phase 4 ships native streaming for the chat. The biggest *absolute* cost lever (fewer LLM turns per lap) stays owned by P2b Phases 1-2; nothing here depends on it, and everything here also benefits the P2b end state.

---

## 2. Per-layer LLM-call inventory

Models shown are the **code defaults** actually in force (`F1_LLM_PROVIDER=openai` path; on LM Studio the model string is mostly ignored by the server and the loaded local model answers). Token figures are estimates at ~4 chars/token, to be replaced by Phase 0 measurements. "Turns" = HTTP round trips to the provider.

| Layer | Call site | Model (code default) | Trigger / frequency | Turns | Est. tokens in / out per invocation |
|---|---|---|---|---|---|
| N25 Pace | `pace_agent.py` (ReAct exists at `:606` but is NOT used in the per-lap path; `run()` builds `reasoning` as a template string `:524`) | `gpt-4.1-mini` (idle) | never in sim loop | 0 | 0 |
| N26 Tire | `tire_agent.py:1162` ReAct invoke (`create_agent`, 2 tools, system prompt ~500 tok) | `gpt-4.1-mini` | every lap (always-on) | ~3 | ~3-4k / ~0.3-0.5k |
| N27 Situation | `race_situation_agent.py:1145` ReAct invoke (2 tools, system ~560 tok) | `gpt-4.1-mini` | every lap (always-on) | ~3 | ~3-4k / ~0.3-0.5k |
| N29 Radio | `radio_agent.py:996` single structured call (`with_structured_output(RadioSynthesis)`, system ~260 tok + NLP JSON `indent=2`) | `gpt-4.1-mini` (`CFG.model_name`, `radio_agent.py:236`) | every lap, **even with 0 messages** (P2b F5) | 1 | ~0.5-1.5k / ~0.1-0.2k |
| N28 Pit | `pit_strategy_agent.py:996` ReAct invoke (3 tools, system ~850 tok) | `gpt-4.1-mini` | conditional: tyre `PIT_SOON`, radio PROBLEM/WARNING, SC active (`_decide_agents_to_call`) | up to 4 | ~4-6k / ~0.4-0.6k |
| N30 RAG | `rag_agent.py:186` ReAct invoke (1 retrieval tool, system ~160 tok, retrieved chunks echoed as tool result) | `gpt-4.1-mini` (hardcoded `rag_agent.py:151-154`) | conditional: N28 active, `sc_prob>0.30`, PENALTY/WARNING alert, SC active. Only 3 canned questions exist (`strategy_orchestrator._build_rag_question`) | 2-3 | ~2-4k / ~0.2-0.3k |
| N31 Orchestrator | `strategy_orchestrator.py:1299/1417` single structured call (`_LLMSynthesis`, 12 LLM-filled fields incl. nested contingencies); prompt template ~7.1k chars static + dynamic blocks | **`gpt-5.4-mini`** (`OrchestratorCFG.model_name:104`) | every lap in LLM mode, all surfaces | 1 | ~2.3-3k / ~0.4-0.8k |
| CLI probe layer | `run_simulation_cli.py:1961` `_probe_core_agents` re-runs N25/N26/N27/N29 for the detail panel | same as above | every lap, CLI LLM mode only | ~7 duplicate | duplicates rows 2-4 (P2b F1) |
| Chat: tool choice | `chat_engine.py:163` first `send_message(tools=...)` (system ~850 tok + ~14 MCP tool schemas + history ≤10 msgs) | **`gpt-5.4-mini`** (`OPENAI_CHAT_MODEL` env, default `llm_service.py:34`) | every chat turn | 1 | ~3-6k / ~0.1-0.4k |
| Chat: summary | `chat_engine.py:281` second `send_message` (no tools, + tool payload capped at 4,000 chars, `chat_engine.py:430`) | `gpt-5.4-mini` | every tool-using turn | 1 | ~4-7k / ~0.2-0.8k (cap 800) |
| Chat: history compression | `llm_service.py:_compress_chat_history` (max_tokens=300), called from `build_messages` | `gpt-5.4-mini` | **every turn once history >10 messages** (summary never persisted) | 1 | ~1-3k / ~0.3k |
| Voice chat | `voice.py:293` single `send_message` (own system prompt, single-turn, max_tokens=220) | `gpt-5.4-mini` | per voice interaction | 1 | ~1k / ~0.2k |
| Backend SSE sim | `simulator.py:775` `run_strategy_orchestrator_from_state` per lap | rows N26-N31 above (no probe duplication) | per simulated lap | ~8-9 quiet | as rows above |
| Arcade | `strategy_pipeline.py` same pipeline, single pass, paced by replay; stale-lap skip saves calls on seek (`strategy.py:252-260`) | rows N26-N31 above | per replayed lap | ~8-9 quiet | as rows above |

**Per-lap totals (quiet lap, estimates):** CLI LLM mode ~15 turns, ~17-21k input / ~2-3k output tokens (probe duplication included). Arcade / backend SSE ~8-9 turns, ~9-12k input / ~1.5-2k output. Activated laps (N28+N30) add ~6-10k input. Per-chat-turn (tool path): 2-3 turns, ~8-13k input / ~0.5-1.5k output.

---

## 3. Model selection: intended policy vs code

Intended policy (user-confirmed, memory `feedback_llm_model_selection`): **`gpt-4.1-mini` for sub-agents N25-N29, a more capable model (`gpt-5.4-mini`) for the N31 orchestrator synthesis.**

| Layer | Policy | Code default | Where | Verdict |
|---|---|---|---|---|
| N25-N29 sub-agents | gpt-4.1-mini | gpt-4.1-mini | signature defaults: `pace_agent.py:609`, `pit_strategy_agent.py:97/793`, `race_situation_agent.py:108/949`, `radio_agent.py:236` | ✅ matches |
| N30 RAG | (not explicitly in policy; sub-agent class) | gpt-4.1-mini **hardcoded**, no parameter at all | `rag_agent.py:151-154` | ✅ matches, least configurable site |
| N31 orchestrator | gpt-5.4-mini | gpt-5.4-mini | `OrchestratorCFG.model_name`, `strategy_orchestrator.py:104` | ✅ matches |
| Chat / voice | (no explicit policy) | gpt-5.4-mini, env-overridable via `OPENAI_CHAT_MODEL` | `llm_service.py:34` | ✅ reasonable; the only layer with an env override |
| Documentation | - | `gpt-4.1` for N31 | `src/agents/README.md:167-168` | ❌ stale (cross-ref DOCS_ACCURACY F-11 family) |

**Configurability assessment:** the policy is real but frozen. There is no `F1_LLM_MODEL_*` environment variable, no shared config module, and `.env.example` documents neither `OPENAI_CHAT_MODEL` nor any model knob (it also documents `LM_STUDIO_BASE_URL`, which **no code reads**: the backend reads `LM_STUDIO_HOST` at `llm_service.py:26` and the agents hardcode `http://localhost:1234/v1` in 7 places). Changing the sub-agent model today means editing 6+ defaults inside UNTOUCHABLE files; changing the orchestrator model means editing `OrchestratorCFG`. The one additive escape hatch that already exists and is precedented: `OrchestratorCFG` is a module-level mutable singleton (`CFG = OrchestratorCFG()`, `strategy_orchestrator.py:112`) consumed lazily at first `_get_orchestrator_llm()` call, exactly the mechanism `simulator.py:183` already uses to propagate the provider. External config code can set `CFG.model_name` before first use without touching internals. The ReAct builders also accept `model_name=` parameters (`get_react_agent`), which the future P2b engine can thread through.

---

## 4. Token budget

Measured prompt sizes (chars, ~4 chars/token):

| Prompt | Size | Est. tokens | Notes |
|---|---|---|---|
| `_TIRE_SYSTEM_PROMPT` | 2,006 | ~500 | resent on every ReAct turn, every lap |
| `_RACE_SITUATION_SYSTEM_PROMPT` | 2,255 | ~560 | idem |
| `_PIT_STRATEGY_SYSTEM_PROMPT` | 3,403 | ~850 | conditional laps |
| `_RADIO_SYSTEM_PROMPT` | 1,022 | ~260 | + NLP results as `indent=2` JSON (30-50% avoidable whitespace/structure bloat) |
| RAG `_SYSTEM_PROMPT` | 630 | ~160 | + retrieved chunks in tool result |
| N31 synthesis prompt template | 7,069 (static source span) | ~1,300 static + ~0.5-1k dynamic | single user message; static text (guardrails :854-873, rubric :884-899, field spec :900-931) wraps dynamic blocks; plus the `_LLMSynthesis` structured-output schema (12 fields, nested `Contingency`) serialized as a tool definition on every call |
| Chat engine `_SYSTEM_PROMPT` | 3,417 | ~850 | + ~14 MCP tool schemas from `list_openai_tools()` on the first call of every turn |
| `llm_service.build_messages` fallback prompt | 3,286 | ~820 | near-duplicate of the chat prompt, pre-MCP wording ("You do NOT call tools yourself"), live only for callers that omit `system_prompt` (currently none; voice passes its own) |
| Voice system prompt | (built at `voice.py:_build_voice_system_prompt`) | ~0.3-0.5k | single-turn |

**Race-level budget (CLI, LLM mode, 57 laps, estimates pending Phase 0):** ~15 turns/lap x 57 laps ≈ 850 turns; ~1.0-1.2M input + ~0.15M output tokens on `gpt-4.1-mini` calls, plus 57 N31 calls ≈ ~150-170k input + ~30-45k output on `gpt-5.4-mini`. Arcade/backend: roughly half the mini-class volume (no probe duplication).

**Money (OpenAI only; LM Studio is free, its currency is latency):** at published `gpt-4.1-mini` rates ($0.40 / $1.60 per M input/output, knowledge-cutoff prices, verify current sheet), the mini-class share of a CLI race is ~$0.45-0.75. `gpt-5.4-mini` rates must be taken from the current OpenAI price sheet (post-cutoff model); at mini-class-like rates the N31 share is cents, at 4.1-full-like rates it approaches ~$0.5-1/race. **Order of magnitude: ~$1-2 per full CLI race today, halved by removing probe duplication (P2b F1), cut ~3-4x more by the P2b turn diet, and a further ~30-50% of the remaining input cost recoverable via prompt caching (§5).** Chat: ~$0.005-0.02 per tool-using turn at mini-class rates. These are planning numbers, not billing numbers: Phase 0 exists precisely because no measured figure exists anywhere in the repo today.

---

## 5. Caching assessment

What exists today: module-level client/agent singletons (`_get_orchestrator_llm`, `_get_radio_llm`, per-agent `_react_agent`), which cache *construction*, not calls. The Whisper JSON transcription cache is transport-side and out of scope here. **No prompt caching, no response caching, no memoization of any LLM call anywhere.**

Opportunities, in value order:

1. **OpenAI automatic prompt caching on N31 (new, this audit).** OpenAI applies a cached-token discount to request prefixes ≥1,024 tokens that repeat across calls. The N31 prompt has ~1,300 static tokens but they sandwich dynamic content: static intro + guardrails, then dynamic RACE CONTEXT / sub-agent blocks / MC table / reg block, then static rubric + field spec (`strategy_orchestrator.py:854-931`). The cacheable prefix therefore ends after ~600 tokens (below threshold). Restructure additively (the prompt builder is a private helper, but the *shape* change can land in the P2b engine's synthesis step, or as a sanctioned prompt-order change): put ALL static text first as a `SystemMessage` (guardrails + rubric + field spec + the structured-output instruction), dynamic blocks last in a short `HumanMessage`. Result: >1,300-token stable prefix, cache-eligible on every lap after the first, on both the input price and time-to-first-token. The same reordering is prompt-engineering-neutral (content identical, order changed), but must be A/B checked against the ML-eval conformance battery (AUDIT_ML_AGENTS_EVAL E-05/R-4) since rubric position can affect adherence.
2. **Sub-agent ReAct calls are already prefix-stable** (static system prompt + static tool schemas first): on OpenAI they may already collect discounts where the prefix crosses 1,024 tokens (tire ~500 + 2 tool schemas likely crosses; pit almost certainly). Nobody knows, because `usage.prompt_tokens_details.cached_tokens` is discarded (finding L-4). Phase 0 makes this visible before optimizing further.
3. **Chat turn prefix** (system ~850 + ~14 tool schemas ≈ 2-5k tokens, stable across turns and users) is prime cache territory and likely already partially discounted on OpenAI. Two things break the prefix today: the compressed-history summary is inserted as a *second system message right after the first* (`llm_service.py:472-476`), and `context` mutates the tail of the first system message (`:432-443`). Keep the static prompt byte-stable and append dynamic context/summary after it (order within the messages array already does this for context? No: context is concatenated INTO the system prompt string, changing the first block). Move context into a separate trailing message.
4. **RAG answer cache**: owned by P2b F4 (3 canned questions, `lru_cache` + warmup pre-answer). Cross-reference only; note here that it also zeroes the N30 *token* cost, not just latency.
5. **N31 event-triggered cadence**: owned by P2b F11 (reuse previous recommendation between triggers). The cost view: cruise sequences drop to ~1/K of both tokens and dollars.
6. **Chat history summary memoization**: this audit's L-6 (see findings): persist/roll forward the summary instead of recomputing per turn.
7. **LM Studio side**: prompt caching is server-dependent (llama.cpp-based versions reuse KV cache for common prefixes automatically on same-slot requests). The same static-prefix-first restructuring maximizes whatever the local server offers; no code needed beyond item 1. Verify per LM Studio release in Phase 0.

---

## 6. Latency per surface: what the LLM adds and where it sits

| Surface | LLM on critical path? | LLM contribution (today) | Cross-refs |
|---|---|---|---|
| CLI `f1-sim` LLM mode | Yes, hard: the Rich row for lap N renders only after ~15 turns | ~17-50 s/lap quiet, ~25-70 s activated (P2b §2.1); worst case unbounded (no timeouts, L-1) | P2b F1/F3; P4 C-04 |
| CLI `--no-llm` | Should be no; today: crashes (#166) and, post-fix, still constructs clients and pays retry backoff when provider down; silently calls LLM if LM Studio is up | ~5-8 s/lap of pure retry sleep when down (P2b §2.3) | P2b F2/F8; P4 C-01/C-06; issue #166 |
| Arcade | Yes, but paced: agent thread blocks until the replay reaches the lap; seek skips stale laps | ~8-30 s/lap budget, absorbed by replay pacing; dashboard recommendation lags a lap on slow turns | P2b §2.2 |
| Streamlit chat | Yes: 2 LLM calls per tool turn (+1 hidden compression once history >10), zero streaming, all blocking the backend event loop | full-generation wait before the single `token` event; a slow turn also stalls other backend consumers (L-3) | this audit L-3/L-6/L-8; memory `project_chat_mcp_refactor` (native streaming = known pending) |
| Backend SSE sim (`/strategy/simulate`) | Yes in LLM mode: orchestrator per lap inside the SSE generator | same ~8-9 turns/lap as Arcade; plus event-loop contention with chat (L-3) | P2b; SECURITY S-7 (abuse) |
| Voice | Yes: STT → 1 LLM call (max_tokens=220) → TTS | ~1-5 s LLM slice; LM Studio path has no timeout (S-5 reaches here via `send_message`) | SECURITY S-4/S-5 |

---

## 7. Provider abstraction assessment (OpenAI vs LM Studio)

| Concern | Agents stack (LangChain, `src/agents/`) | Backend stack (`llm_service.py`) |
|---|---|---|
| Provider switch | `F1_LLM_PROVIDER` read lazily at first client build (per-process singletons); CLI sets it at runtime (`run_simulation_cli.py:1586`) | `F1_LLM_PROVIDER`/`LLM_PROVIDER` read at **import time** into module constants (`llm_service.py:23-40`): switching requires a process restart, and runtime `os.environ` writes (the CLI pattern) would not affect an already-imported backend |
| Base URL | hardcoded `http://localhost:1234/v1` (7 sites) | `LM_STUDIO_HOST` env (host only, port fixed). `.env.example` documents `LM_STUDIO_BASE_URL`, which nothing reads |
| Timeout | none set → openai-python default 600 s | 60 s OpenAI, **None** LM Studio (S-5) |
| Retries | none set → openai-python default 2 (with backoff; harmful in `--no-llm`, P2b F8) | none (raw requests, 0 retries) |
| Streaming | not used (structured output; fine) | `stream_message` exists, unused by `chat_engine` |
| Fallback / degradation | exception-matching heuristic downgrades agents to stubs (`_is_llm_unavailable`, CLI `:432` + `simulator.py:288`); no health preflight, no circuit breaker | `check_health()` exists (`llm_service.py:71`) but is not gated before expensive flows; `_safe_send` degrades to a canned message |
| Model resolution | per-layer hardcoded defaults (§3) | `_default_model` drops any override not starting with `gpt`/`o` (`llm_service.py:53-61`): silently swallows other valid OpenAI ids and future families |
| Generation params | temperature 0.0 everywhere (good for determinism) | defaults temperature 0.7 / max_tokens 1000 in `send_message`; engine passes 0.3 / 800; voice 0.6 / 220; compression 0.3 / 300. Scattered, undocumented intent |

Two stacks, four env vars (`F1_LLM_PROVIDER`, `LLM_PROVIDER`, `OPENAI_CHAT_MODEL`, `LM_STUDIO_HOST`), one documented-but-dead var (`LM_STUDIO_BASE_URL`), zero shared configuration. This is the structural root under findings L-1, L-2 and L-7.

---

## 8. Findings register (P0 → P3)

Cross-referenced sibling findings are not re-registered: probe duplication (P2b F1), ReAct turn inflation (P2b F3), RAG cacheability (P2b F4), silent-radio LLM (P2b F5), sequential always-on (P2b F6), attempt-and-catch no-LLM (P2b F8), warmup gaps (P2b F9), N31 cadence (P2b F11), `--no-llm` crash (#166 / P2b F2 / P4 C-01), backend LM Studio timeout (SECURITY S-5), cost caps and rate limiting (SECURITY S-2/S-7).

### P0 - unbounded latency on the critical path

| ID | Finding | Evidence | Size |
|---|---|---|---|
| **L-1** | **No timeout or retry policy on any agent-side LLM client.** All 7 `ChatOpenAI` construction sites pass neither `timeout` nor `max_retries`, inheriting openai-python defaults (600 s timeout, 2 retries): a hung LM Studio or a network black hole can pin a lap (CLI, Arcade, backend SSE) for up to ~30 minutes per call, and the retry default is exactly what makes broken `--no-llm` pay ~1.5-2 s of backoff per agent (P2b F8). Remedy (additive/config-side): a per-call latency budget from env (e.g. `F1_LLM_TIMEOUT_S`, default ~30-60 s; `max_retries` 0-1), injected at client construction. For the agents this lands where clients are built for them: the P2b engine threads `model_name`/client kwargs through the existing `get_react_agent(...)` parameters, and `OrchestratorCFG` mutation (the `simulator.py:183` precedent) covers N31 without touching internals; interim, an env-var-driven `OPENAI_TIMEOUT`-style wrapper is acceptable if verified. Complements (does not replace) SECURITY S-5/C1, which fixes the backend `requests` path. | `pace_agent.py:644-646`, `pit_strategy_agent.py:826-835`, `race_situation_agent.py:981-983`, `radio_agent.py:793-800`, `rag_agent.py:151-158`, `strategy_orchestrator.py:138-146`; backend `llm_service.py:40` | S |

### P1 - major cost/latency/robustness levers

| ID | Finding | Evidence | Size |
|---|---|---|---|
| **L-2** | **Per-layer model policy is frozen in 8 hardcoded sites with no configuration surface.** Policy verified correct in defaults (§3) but: no env override for any agent layer (chat alone has `OPENAI_CHAT_MODEL`); N30 accepts no model parameter at all; `.env.example` documents no model knob; README contradicts code. Consequences: model migrations (e.g. a cheaper tool-choice model for the chat's first call, or a 2026-season model bump) require edits inside UNTOUCHABLE files; A/B model ablations (ML-eval "LLM model swap" arm) have no switch. Remedy: one additive `src/strategy/inference/llm_config.py` (or equivalent) reading `F1_LLM_MODEL_AGENTS`, `F1_LLM_MODEL_ORCHESTRATOR`, `OPENAI_CHAT_MODEL`, `F1_LLM_BASE_URL`, `F1_LLM_TIMEOUT_S`; consumed via `OrchestratorCFG` mutation + `get_react_agent(model_name=...)` params from the P2b engine + `llm_service` reading the same module; document all vars in `.env.example`; fix `src/agents/README.md:167-168`. Models stay OpenAI / LM Studio only. | `strategy_orchestrator.py:104`, `rag_agent.py:151-154`, signature defaults across the 5 sub-agents, `llm_service.py:34`, `.env.example`, `src/agents/README.md:168` | S/M |
| **L-3** | **Synchronous LLM calls block the FastAPI event loop.** `chat_engine` runs blocking `requests.post` inside async generators (self-documented at `chat_engine.py:22-25`); `send_message`/`stream_message` are sync; voice and the MCP-dispatched tools share the loop. One slow chat turn (or a stalled LM Studio with S-5's infinite timeout) freezes SSE sim streams, voice, and all other requests on the worker. Remedy: wrap provider calls in `asyncio.to_thread(...)` (minimal) or migrate `llm_service` to `httpx.AsyncClient` (better, enables L-8 streaming); backend-only change, no agent impact. | `chat_engine.py:22-25` (docstring), `:317-338` (`_safe_send` → sync `send_message`), `llm_service.py:226/291` | S/M |
| **L-4** | **Zero token/cost observability.** Response `usage` is discarded at every agent call site (LangChain result metadata unread); the chat keeps only `total_tokens` of the *last* call per turn (`chat_engine.py:531-537`); nothing records per-layer tokens, cached_tokens, latency, or per-race dollars. Without a ledger: this audit's budget stays an estimate, prompt-cache wins (L-5) are unverifiable, SECURITY's cost caps (S-2/S-7) have no meter to enforce against, and the ML-eval cost-vs-quality ablations have no cost axis. Remedy: additive usage ledger: a LangChain callback (agents) + a `send_message` hook (backend) appending `{ts, surface, layer, model, prompt_tokens, cached_tokens, completion_tokens, latency_ms}` to a JSONL under `data/telemetry/llm_usage/` (gitignored), plus a tiny per-race/per-session summary report. Feeds the P2b Phase-0 timing harness rather than duplicating it. | `chat_engine.py:531-537`; absence everywhere else (scan) | S/M |
| **L-5** | **Prompt-cache blindness: N31's static ~1,300 tokens are ordered to defeat OpenAI prefix caching.** Static intro+guardrails, then dynamic context, then static rubric+field-spec, all in one user message (§5.1): cacheable prefix ~600 tokens, below OpenAI's 1,024 threshold, so every lap pays full input price on ~2.3-3k tokens x 57 laps x every race. Chat prefix similarly self-sabotages by concatenating dynamic `context` into the system prompt (`llm_service.py:432-443`) and inserting the history summary immediately after it. Remedy: static-first restructure (system message carries guardrails+rubric+spec; short dynamic user message), keep the chat system prompt byte-stable and move context/summary to trailing messages; verify via L-4's `cached_tokens` before/after; A/B conformance per ML-eval E-05 battery. Also maximizes LM Studio KV-prefix reuse. | `strategy_orchestrator.py:854-931` (static/dynamic interleave), `llm_service.py:432-476` | M |
| **L-6** | **Hidden extra LLM call per chat turn once history exceeds 10 messages, recomputed forever.** `build_messages` calls `_compress_chat_history` (a full blocking LLM round trip) whenever `len(text_history) > 10`, and the summary is never persisted anywhere: with Streamlit resending the full session history every turn (`chat.py:291` → `get_chat_history()` → `chat_state.py:225-232`), every turn after the threshold pays compression again (and the first LLM call of the turn waits behind it, doubling perceived latency). Remedy: memoize the summary keyed by the compressed-prefix content hash (module-level LRU is enough for the single-user Streamlit case), or store it in the session state and roll it forward; also cap the resent raw history client-side. | `llm_service.py:449-476` (`build_messages`), `:341-397` (`_compress_chat_history`), `frontend/utils/chat_state.py:225-232` | S |

### P2 - structural hygiene, robustness, UX latency

| ID | Finding | Evidence | Size |
|---|---|---|---|
| **L-7** | **Dual provider stacks with divergent env semantics and dead config.** Import-time provider freeze in `llm_service` (runtime `F1_LLM_PROVIDER` changes ignored, unlike the agents); `LM_STUDIO_BASE_URL` documented in `.env.example` but read by nothing (backend reads `LM_STUDIO_HOST`, agents hardcode the URL); `_default_model` silently swallows non-`gpt`/`o` overrides; no shared health preflight before expensive flows (a race run starts and only discovers a dead provider lap by lap through exception matching). Remedy: fold both stacks onto the L-2 config module; make provider resolution lazy in `llm_service`; align env names and document them; add an optional preflight (`check_health()` already exists) gating LLM-mode runs with a clear "provider unreachable, use --no-llm" message. | `llm_service.py:23-40,53-61`, `.env.example`, agent hardcoded URLs (7 sites), `llm_service.py:71` unused-as-gate | M |
| **L-8** | **No native streaming for the chat summary.** The engine deliberately removed fake chunking; real `stream=True` was postponed (memory `project_chat_mcp_refactor`). Perceived latency for a summary = full generation time (~2-8 s local) delivered as one SSE `token` event. Remedy: consume `stream_message` (or the L-3 httpx async client with `stream=True`) in `_stream_tool_response`/`_stream_plain_response` and emit real deltas; SSE plumbing and frontend renderer already handle token events. Zero cost impact, large perceived-latency win. | `chat_engine.py:27-32` (docstring), `:281-289`; `llm_service.py:252` (`stream_message` unused) | M |
| **L-9** | **No degradation policy tiering across surfaces.** Each surface improvises: agents downgrade to stubs via string-matching on exception names (fragile: `_LLM_ERR_TYPES` substrings, CLI `:395-425`), chat degrades to a canned string, voice returns an apology, the backend SSE propagates per-lap errors. There is no shared "provider down → declared no-LLM degradation" switch, and no distinction between transient (retry) and hard (degrade) failures. Remedy: define the tiering once in the L-2/L-7 config module (error taxonomy + per-surface policy); the P2b engine's `no-llm` profile becomes the explicit degrade target instead of an accidental one. | CLI `:386-447`, `simulator.py:288-302`, `chat_engine.py:317-338`, `voice.py:303-311` | S/M |

### P3 - micro / docs

| ID | Finding | Evidence | Size |
|---|---|---|---|
| **L-10** | README model table stale: N31 documented as `gpt-4.1`, code says `gpt-5.4-mini`. One-line docs fix, batch with L-2's `.env.example` documentation. Cross-ref DOCS_ACCURACY F-11 (provider defaults) - same family, different line. | `src/agents/README.md:167-168` vs `strategy_orchestrator.py:104` | S |
| **L-11** | Radio synthesis prompt serializes NLP results with `indent=2` and full pipeline dicts: ~30-50% of that block is whitespace/structural bloat, paid every lap with radio traffic. Compact separators + a field projection (intent, sentiment, entities, text) shrink it with zero information loss. Config-side: the prompt builder is inside `src/agents/`, so land it as part of the P2b direct-mode synthesis (which rebuilds these prompts additively) rather than an in-place edit. | `radio_agent.py:820-833` (`_build_synthesis_prompt`) | S |
| **L-12** | Generation-parameter scatter with undocumented intent: `send_message` defaults (0.7/1000) never used deliberately, engine 0.3/800, voice 0.6/220, compression 0.3/300, agents 0.0. Document the per-layer intent in the L-2 config module and drop the misleading defaults. | `llm_service.py:154-157`, `chat_engine.py:134-136`, `voice.py:295-298` | S |

---

## 9. Phased plan (chunkable → GitHub sub-issues; S/M/L)

All phases are additive/config-side; none touches `src/agents/` internals, the PMV, or notebooks. Phases 0-2 have no dependency on the P2b engine; Phases 3's N31 item and L-11 land best *inside* the P2b engine work (coordinate, do not duplicate).

### Phase 0 - Token & cost observability (S)
| Chunk | What | Exit criterion |
|---|---|---|
| 0.1 | Usage ledger: LangChain callback (agents) + `send_message` hook (backend) → JSONL `{surface, layer, model, prompt/cached/completion tokens, latency_ms}` + per-race and per-chat-session summary script (L-4) | ledger populated on a Budapest/NOR LLM-mode run + a 12-message chat session |
| 0.2 | Baseline report: one full race per surface profile (CLI LLM, Arcade, backend SSE) + chat session, both providers; record cached_tokens as-is today; verify LM Studio version's prefix-cache behavior | §2/§4 estimate tables replaced with measured numbers; report committed under `documents/eval_reports/` or audit appendix |
| 0.3 | Verify `gpt-5.4-mini` current pricing + prompt-cache discount from the OpenAI price sheet; pin the cost model | cost-per-race figure with real rates |

### Phase 1 - Provider hardening (M)
| Chunk | What | Exit criterion |
|---|---|---|
| 1.1 | Timeouts + retry policy everywhere: env-driven `F1_LLM_TIMEOUT_S` / retries injected at client construction (agents via engine/CFG paths; backend `DEFAULT_TIMEOUT` for both providers, closing S-5/C1 jointly with the security plan) (L-1) | no LLM call in the tree can exceed the configured budget; TESTING_QA's "LM Studio no-timeout hang" test passes |
| 1.2 | Async offload in the backend: provider calls via `asyncio.to_thread` or httpx async (L-3) | concurrent SSE sim + chat turn: sim frames keep flowing during a slow chat LLM call |
| 1.3 | Preflight + degradation tiering: health gate before LLM-mode runs; shared error taxonomy replacing per-surface string matching (L-9, L-7 part) | dead-provider race run fails fast with a clear message instead of lap-by-lap exceptions |

### Phase 2 - Model policy configuration (S)
| Chunk | What | Exit criterion |
|---|---|---|
| 2.1 | `llm_config` single source of truth: `F1_LLM_MODEL_AGENTS` / `F1_LLM_MODEL_ORCHESTRATOR` / `OPENAI_CHAT_MODEL` / base URL / timeout, consumed by both stacks (CFG mutation + `get_react_agent` params + `llm_service`); lazy provider resolution; `.env.example` documents every var; kill dead `LM_STUDIO_BASE_URL` or wire it (L-2, L-7) | changing any layer's model = one env var; defaults unchanged (policy: 4.1-mini agents, 5.4-mini N31) |
| 2.2 | Docs sync: `src/agents/README.md` model table, CLAUDE/INSTALL mentions (L-10, batch with DOCS_ACCURACY follow-ups) | docs match code |
| 2.3 | Optional experiment (measure, then decide): cheaper model for the chat's first tool-choice call (routing is schema-driven; quality bar = tool-selection accuracy on a fixture set), keep `gpt-5.4-mini` for the summary | measured tool-choice accuracy delta + cost delta; adopt only if accuracy holds |

### Phase 3 - Prompt & cache diet (M)
| Chunk | What | Exit criterion |
|---|---|---|
| 3.1 | N31 static-first prompt restructure (system message: guardrails+rubric+spec; user message: dynamic blocks), landed with/inside the P2b engine synthesis step; A/B against the ML-eval conformance battery (L-5) | `cached_tokens > 0` on lap 2+ (OpenAI); no conformance regression on the guardrail battery |
| 3.2 | Chat prefix stabilization: byte-stable system prompt, context + history summary as trailing messages (L-5 chat part) | cached_tokens on chat turn 2+ |
| 3.3 | History-summary memoization + client-side history cap (L-6) | ≤1 compression call per conversation growth step; chat turn = 2 LLM calls max |
| 3.4 | Radio compact-JSON projection inside the P2b direct-mode synthesis (L-11); generation-param documentation (L-12) | radio prompt tokens −30-50% on radio laps |

### Phase 4 - Streaming UX (M)
| Chunk | What | Exit criterion |
|---|---|---|
| 4.1 | Native streaming for chat summary + plain replies: `stream=True` through the async client, real SSE `token` deltas (L-8; builds on 1.2) | first token visible < 1 s after summary stage on OpenAI; frontend renders progressively |
| 4.2 | (Optional, later) streamed N31 `reasoning` for Arcade/CLI panels once the P2b engine owns synthesis; structured fields stay non-streamed | deferred; only if panel UX demands it |

**Dependency chain:** 0.1-0.3 → 1.1-1.3 (hardening needs the ledger to prove no regression) → 2.x (config module absorbs 1.x knobs) → 3.x (cache wins measured via ledger) → 4.x. Coordinate 3.1/3.4 with P2b Phase 2 (turn diet) so prompts are restructured once, not twice.

---

## 10. Open questions

1. **`gpt-5.4-mini` economics:** current $/M input/output and cached-token discount (post-cutoff model; take from the live OpenAI price sheet in Phase 0.3). Also confirm whether the chat should stay on it or move the tool-choice call down (Phase 2.3).
2. **Cost caps:** does Víctor want hard per-race / per-session budget enforcement (ledger-backed kill switch), or is observability enough? (SECURITY S-2/S-7 propose caps for abuse; this is the product-side twin.)
3. **LM Studio prefix caching:** which LM Studio version is in use and does it reuse KV across requests with a common prefix? (Determines how much of L-5 pays off locally; OpenAI payoff is certain.)
4. **Prompt-order sensitivity:** does moving the rubric/field-spec before the dynamic blocks change guardrail adherence on the local models? (Gate 3.1 behind the ML-eval E-05 battery on both providers.)
5. **Latency budget targets per surface:** proposed defaults: 30 s/call CLI-LLM, 60 s chat, 20 s voice; confirm with the P2b measured budget before freezing env defaults.

---

## 11. Verification protocol

- **L-1/1.1:** unit-assert effective client timeout/retries for both providers (extend the SECURITY C1 assertion to the agent stack); chaos test: stall a fake provider socket, assert a lap fails within budget instead of hanging.
- **L-3/1.2:** concurrency test: start an SSE sim stream, fire a chat turn against a slow stub provider, assert sim frames keep arriving.
- **L-4/0.x:** ledger row count == number of provider HTTP calls on a fixture run (cross-check against a counting stub); per-race report totals reconcile with OpenAI dashboard usage for one real run.
- **L-5/3.1-3.2:** `usage.prompt_tokens_details.cached_tokens > 0` from lap/turn 2 on OpenAI; guardrail battery (ML-eval E-05) pass-rate unchanged pre/post restructure; action-match A/B on one GP replay (actions equal, prose may differ - same criterion as P2b's risk register).
- **L-6/3.3:** trace a 14-message conversation: exactly one compression call at the threshold crossing, turns remain 2 LLM calls.
- **L-2/2.1:** matrix smoke: `{F1_LLM_PROVIDER} x {default, overridden model}` boots each surface and the ledger shows the expected model string per layer.
- **L-8/4.1:** SSE capture shows >5 token events for a >100-token summary; time-to-first-token logged in the ledger.
- **Money:** after Phase 0, re-issue §4 as a measured table (per-surface per-race cost, both model classes), and re-baseline after P2b Phase 1/2 lands to confirm the combined reduction.

---

### Appendix A - Evidence index

| Topic | Anchors |
|---|---|
| Orchestrator model + LLM factory | `src/agents/strategy_orchestrator.py:104` (gpt-5.4-mini), `:119-152` (`_get_orchestrator_llm`, no timeout/retries) |
| Orchestrator prompt static/dynamic interleave | `src/agents/strategy_orchestrator.py:741-931` (`_build_orchestrator_prompt`; guardrails :854-873, rubric :884-899, field spec :900-931) |
| Sub-agent model defaults | `pace_agent.py:609`, `pit_strategy_agent.py:97/793`, `race_situation_agent.py:108/949`, `radio_agent.py:236`, `rag_agent.py:151-154` |
| Sub-agent per-lap LLM invokes | `tire_agent.py:1162`, `race_situation_agent.py:1145`, `radio_agent.py:996`, `pit_strategy_agent.py:996`, `rag_agent.py:186` |
| RAG canned questions | `strategy_orchestrator.py` `_build_rag_question` (3 templates) |
| Conditional routing | `strategy_orchestrator.py` `_decide_agents_to_call` |
| CLI probe duplication + provider env set | `scripts/run_simulation_cli.py:1961-1964`, `:1586` |
| Backend provider service | `llm_service.py:23-40` (import-time provider freeze, `DEFAULT_TIMEOUT=None` LM Studio), `:53-61` (`_default_model` filter), `:341-397` (`_compress_chat_history`), `:404-531` (`build_messages`, stale fallback prompt, context-in-system-prompt) |
| Chat engine flow | `chat_engine.py:22-32` (blocking-in-loop + no-streaming docstrings), `:55-120` (system prompt), `:163/281` (two calls per tool turn), `:430` (4,000-char tool payload cap), `:449-466` (`_trim_for_llm`), `:531-537` (`tokens_used` only) |
| Voice LLM call | `src/telemetry/backend/api/v1/endpoints/voice.py:285-298` |
| Arcade pacing / stale skip | `src/arcade/strategy.py:240-260`, `strategy_pipeline.py` (single-pass pipeline) |
| Backend SSE per-lap orchestrator | `src/telemetry/backend/services/simulation/simulator.py:772-775`, `:183` (CFG mutation precedent) |
| Env documentation | `.env.example` (F1_LLM_PROVIDER, OPENAI_API_KEY, dead LM_STUDIO_BASE_URL) |
| Stale docs | `src/agents/README.md:167-168` |
