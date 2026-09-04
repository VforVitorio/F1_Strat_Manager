# Backend API Reference (FastAPI)

## Overview

The backend is a FastAPI application at `src/telemetry/backend/`. It serves telemetry data, driver comparisons, chat (LM Studio proxy), and the N25-N31 strategy agent pipeline. All endpoints are prefixed with `/api/v1`.

Entry point: `backend/main.py`, creates the FastAPI app and registers all routers.

## Router map

There is no `auth` router. Authentication is a single ASGI middleware wrapping every router (see below), not a mounted endpoint set.

| Router | Prefix | Tags | Source |
|---|---|---|---|
| telemetry | `/api/v1/telemetry` | telemetry | `api/v1/endpoints/telemetry.py` |
| circuit_domination | `/api/v1/circuit-domination` | telemetry | `api/v1/endpoints/circuit_domination.py` |
| comparison | `/api/v1/comparison` | comparison | `api/v1/endpoints/comparison.py` |
| chat | `/api/v1/chat` | chat | `api/v1/endpoints/chat.py` |
| strategy | `/api/v1/strategy` | strategy | `api/v1/endpoints/strategy.py` |

Two more mount points sit outside the router list:

- **`GET /`** and **`GET /health`**, unauthenticated liveness endpoints, registered directly on the `FastAPI` app in `main.py`.
- **`/mcp`**, the FastMCP Streamable-HTTP server, mounted only when `F1_MCP_ENABLED=true` (off by default). The chat pipeline reaches the same tools in-process regardless of this flag, so leaving it unmounted removes an open network surface, not a feature. See "Authentication" and "MCP-Driven Tool Routing" below.

## Authentication

Every router (and the `/mcp` mount, when enabled) sits behind a single shared-secret ASGI middleware, `ApiKeyMiddleware` (`backend/core/auth.py`, Security A1 / issue #224). It is intentionally pure ASGI rather than `BaseHTTPMiddleware`, because the latter buffers the whole response body and would break the SSE streams (`/chat/tool-message-stream`, `/simulate`).

- **Header**: `X-API-Key`, compared against the `F1_API_KEY` env var with `hmac.compare_digest`.
- **Open paths**: `/` and `/health` always pass unauthenticated (uptime probes). `OPTIONS` (CORS preflight) always passes.
- **Safe-by-default when unset**: if `F1_API_KEY` is not set, every other request also passes, this is the local-dev default. The dangerous combination is a non-loopback bind (`F1_HOST` other than `127.0.0.1`/`localhost`/`::1`) with no key set: `enforce_startup_security()` refuses to boot in that case rather than come up open on the network.
- **WebSocket**: gated the same way; an unauthorized WS handshake gets a policy-violation close (code 1008) instead of a 401 body.

This means `F1_API_KEY` and `F1_HOST` (see [Setup and deployment](#/setup)) are the two env vars that decide whether the backend is safe to expose beyond localhost.

## Rate limiting

Every prediction and strategy endpoint (and `/simulate`) sits behind an in-process token-bucket limiter (`backend/core/rate_limit.py`) keyed on client IP. No external dependency, a stdlib bucket is enough for a single-process local backend. Buckets are per-route, so hammering `/pace` does not exhaust the `/recommend` bucket. The four chat routes carry buckets too (capacity 10, 20/min), so the chat surface is metered as well.

| Route | Burst capacity | Refill rate |
|---|---|---|
| `/pace`, `/tire`, `/situation`, `/pit`, `/radio` | 20 | 60/min |
| `/pace-range`, `/tire-range`, `/rag` | 5 | 10/min |
| `/recommend` | 5 | 10/min |
| `/simulate` | 3 | 3/min |

An exhausted bucket returns `429` with a `Retry-After` hint. Set `F1_RATE_LIMIT_OFF=1` to disable limiting entirely (load tests, benchmarking). A token is consumed only at request admission, so a long-lived SSE stream (`/simulate`, `/chat/tool-message-stream`) is metered once and then runs unmetered.

## Telemetry endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/api/v1/telemetry/data` | Fetch telemetry for year/gp/session/drivers |
| GET | `/api/v1/telemetry/gps` | List available GPs for a year |
| GET | `/api/v1/telemetry/sessions` | List sessions for a GP |
| GET | `/api/v1/telemetry/drivers` | List drivers for a session |
| GET | `/api/v1/telemetry/race-data` | Full-field featured-parquet frame for a GP (positions, lap times, inter-driver gaps), optionally filtered to driver codes |
| POST | `/api/v1/telemetry/prewarm` | Warm the session cache in the background; returns 202 immediately |

**Query parameters** vary by endpoint. `year` (int) and `gp` (str) are common to all of them; `session` (str) applies to the lap-time and telemetry endpoints; `drivers` (comma-separated) to the comparison ones. `/race-data` takes `driver`, **singular**, and treats it as an optional filter over the full-field frame.

`/race-data` computes the inter-driver gap columns (`GapToCarAhead`, `GapToCarBehind`) over the whole field first, then applies the optional `driver` filter afterwards: a single-car frame has no second car to measure a gap against, so filtering before computing the gaps used to return `null` on every lap whenever a `driver` was supplied. The gap-annotated frame is cached per `(year, gp)`, since it is a pure function of the static featured parquet.

## Comparison endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/api/v1/comparison/compare` | Compare fastest-lap telemetry between two drivers |

## Chat endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/api/v1/chat/health` | LM Studio health check |
| GET | `/api/v1/chat/models` | List available LM Studio models |
| GET | `/api/v1/chat/status` | Current backend stage for a `request_id` (smart-spinner poll) |
| POST | `/api/v1/chat/message` | Non-streaming chat message (raw LLM, no tools) |
| POST | `/api/v1/chat/stream` | Streaming chat response (raw LLM, no tools) |
| POST | `/api/v1/chat/tool-message` | Tool-aware chat -- JSON response |
| POST | `/api/v1/chat/tool-message-stream` | Tool-aware chat -- Server-Sent Events stream |

Chat proxies the configured LLM provider (LM Studio local or OpenAI cloud, switchable via `F1_LLM_PROVIDER`). Tool-aware endpoints route through the **MCP-driven `chat_engine`** (see below); raw `/message` and `/stream` skip the tool layer and return whatever the model writes.

### MCP-Driven Tool Routing

`/chat/tool-message` and `/chat/tool-message-stream` are powered by `services/chatbot/chat_engine.py`. The engine pulls every tool from the FastMCP server (`backend.mcp_tools.mcp`) via the in-process `fastmcp.Client`, exposes them to the LLM as OpenAI-style `tools=[...]` schemas, and dispatches the model's chosen tool back through the MCP client. There is **no parallel keyword/regex registry** anymore -- tool definitions live in one place and the LLM sees the same schemas an external MCP client (Claude Desktop, Cursor) would see when it dials `/mcp`.

The flow per request:

1. **Pull tool catalog** -- `mcp_bridge.list_openai_tools()` returns every Phase 1 `@mcp.tool` plus the Phase 2 telemetry tools auto-mounted from the FastAPI OpenAPI spec, formatted as `{"type": "function", "function": {...}}`.
2. **First LLM call** -- with `tools=` populated. The model decides whether to call a tool or reply in plain text. Casual greetings, meta questions ("what tools do you have?"), and general F1 knowledge are answered directly without dispatching.
3. **Tool dispatch** (only when the model returned a `tool_call`) -- `mcp_bridge.call_mcp_tool(name, args)` runs the tool through the FastMCP client and returns the structured data.
4. **`tool_result` SSE event** -- the structured payload is wrapped in `{tool_name, display_type, data, summary}` and emitted so the frontend can render the right component (chart / metrics / strategy card / table / text).
5. **Second LLM call** -- without `tools=`, feeding the tool's output back as a `role=tool` message so the model summarises the data in the user's language.

The streaming endpoint emits four SSE event types in order: `stage` (every checkpoint, also reflected in `/chat/status`), `tool_result` (rich payload), `token` (LLM text chunks), `done` (final marker with provider metadata).

### Tool results and display hints

Each tool is mapped to a `DisplayType` hint via `TOOL_DISPLAY_MAP` (`models/tool_schemas.py`); the frontend's chat renderer chooses a component based on the hint:

| DisplayType | Used by |
|---|---|
| `METRICS` | `predict_pace`, `predict_situation` |
| `STRATEGY_CARD` | `predict_tire`, `predict_pit`, `recommend_strategy` |
| `TABLE` | `analyze_radio` |
| `TEXT` | `query_regulations`, `list_gps`, `list_drivers`, `get_lap_range` |
| `CHART` | `get_lap_times`, `get_telemetry`, `compare_drivers`, `get_race_data` |

`chat_engine._trim_for_llm` caps long arrays before they are sent back to the LLM for summarisation; the unmodified payload still reaches the frontend on `tool_result.data` so charts retain the full series. The four telemetry tools are wired to `CHART` so the web app chat renders them as inline charts.

### Tool risk tiers and the chat allowlist (Security A2, #224)

`models/tool_schemas.py` classifies every dispatchable MCP tool into a `ToolRisk` tier: `READ_SAFE` (cheap, e.g. `predict_pace`), `READ_EXPENSIVE` (heavy but still read-only, e.g. `recommend_strategy`'s 500-sample Monte Carlo, or `query_regulations`'s RAG lookup), or `MUTATING` (writes/exports; none exist today). `CHAT_ALLOWED_TOOLS` is the default-deny set built from the first two tiers: a tool absent from `TOOL_RISK_MAP`, hallucinated by the LLM, or a newly added tool nobody classified yet, is refused by both `mcp_bridge` (before it reaches the LLM's tool list) and `chat_engine`'s dispatch guard (before it runs), and a `MUTATING` tool can never join the allowlist. The hard rule: no write/export tool may be added to the MCP server until it has a `TOOL_RISK_MAP` entry.

Every Phase 1 tool (`predict_pace`/`predict_tire`/`predict_situation`/`predict_pit`/`analyze_radio`/`recommend_strategy`) also normalises its `gp`/`driver`/`lap`/`year` arguments in `mcp_tools.py` before building the `lap_state` (`_normalize_gp_name`, `_normalize_driver_code`, `_normalize_lap`, `_normalize_year`). An unparseable lap number used to silently become lap 1 (#442); it now raises `ToolInputError`, which the `_catch_tool_input_error` decorator turns into a plain "X is invalid, here are the valid options" string for the LLM instead of a traceback, the same REFUSED shape the agent-level tool guards described in [Agents API reference](#/agents-api) already use.

### Smart-spinner stage tracker

The frontend mints a UUID, sends it on every chat request via the `X-Request-Id` header, and polls `/api/v1/chat/status?request_id=...` every second. The backend writes the current stage (`preparing_tools`, `model_choosing_tool`, `calling_<tool>`, `summarizing_with_llm`, ...) into a process-global tracker (`services/chatbot/stage_tracker.py`) at every checkpoint, cleared in a `try/finally` so the dict never leaks. The web app chat maps these stages to humanised labels so the spinner narrates the slow phases (model loading, tool execution).

### Module layout

`services/chatbot/` now contains only what the MCP-driven flow needs:

- `chat_engine.py`, async orchestrator (stream + sync entry points).
- `mcp_bridge.py`, async adapter to the FastMCP server (`list_openai_tools`, `call_mcp_tool`).
- `llm_service.py`, provider abstraction (LM Studio + OpenAI), now with `tools=` support.
- `stage_tracker.py`, per-request stage dict for the smart-spinner.
- `utils/`, empty placeholder; the legacy `tool_param_extractor`, `query_classifier`, `validators`, the per-handler files, the `router/` package and the `prompts/` directory were deleted along with the `/chat/query` endpoint.

## Voice endpoints (retired)

The `/api/v1/voice` router (STT, TTS and the STT to LLM to TTS pipeline) was retired in v2: it came from a course requirement and the web app ships without it. The implementation remains available in git history and in the `legacy_version` branch (the legacy Streamlit app was removed from the repo, #551).

## Strategy endpoints (N25-N31)

All strategy endpoints live under `/api/v1/strategy/`. They accept JSON bodies and return `StrategyResponse` envelopes.

### Consumers

The `/api/v1/strategy/simulate` SSE endpoint is consumed by the web app and by `curl` / `TestClient` smoke tests. The arcade replay no longer calls this endpoint, as of Phase 3.5 Proceso B (April 2026), the arcade owns its own strategy pipeline via [`src/arcade/strategy_pipeline.py`](#/arcade-strategy-pipeline).

### `POST /api/v1/strategy/simulate`

Streams per-lap strategy decisions as Server-Sent Events, rate-limited to 3 requests/minute per client (see "Rate limiting" above).

```python
class SimulateRequest(BaseModel):
    year: int = 2025  # 2023-2025
    gp: str
    driver: str
    team: str
    driver2: Optional[str] = None
    lap_range: Optional[tuple[int, int]] = None
    risk_tolerance: float = 0.5  # 0-1
    no_llm: bool = False
    provider: str = "lmstudio"  # "lmstudio" | "openai"
    interval_s: float = 0.0  # 0-10, artificial delay between laps
```

Event stream: one `start` event, then one `lap` (or `error`) event per processed lap, closed with a `summary` event. A blank SSE comment (`:\n\n`) is sent every 15 `lap` events as a heartbeat so long runs survive proxy idle timeouts.

### Metadata (GET)

| Path | Description |
|---|---|
| `/api/v1/strategy/available-gps` | GP names in the featured parquet |
| `/api/v1/strategy/available-drivers` | Driver codes for a GP |
| `/api/v1/strategy/lap-range` | Min/max lap for a driver at a GP |
| `/api/v1/strategy/lap-state` | Build a lap_state dict from parquet, agent-ready but not identical to the replay engine's (see below) |
| `/api/v1/strategy/radio-available-gps` | GPs with a recorded radio/RCM corpus |
| `/api/v1/strategy/radio-laps` | Laps with radio messages for a GP (optionally filtered by driver) |
| `/api/v1/strategy/radio-transcript` | Cached Whisper transcript for one driver/lap |

`/lap-state` also returns two Art. 30.5(m) (2024-25 numbering; it was 30.5(n) in 2023) stint-history keys the strategy layer's terminal-liability term depends on: `stint_flags` (the requested driver's `stops_made`, `compounds_used`, `mandatory_stop_pending`) and `rival_stop_pending` (a `{driver_code: mandatory_stop_pending}` map, one entry per rival in the response). Both come from `src/simulation/stint_history.py`, the same helper the replay engine calls, so the CLI, the Arcade and this endpoint read the same stop history and cannot disagree. Any of the three flags can be `null`: an unresolvable stint history (an invisible earlier stint that could hide a compound change) is reported as unknown rather than guessed.

`/radio-laps` and `/radio-transcript` cache their parquet and transcript-JSON reads in memory per `(year, gp)`, since the underlying radio corpus is static for the life of the process; the first request for a race pays the read cost, later ones are served from the cache.

### Where this `lap_state` differs from the replay engine's

Both producers emit the same five top-level keys plus the two stint-history ones, and every agent accepts either. They are **not** field-identical, and it is worth knowing which way, because a producer that diverged from this contract once made a whole strategy candidate permanently ineligible.

Measured on Lusail 2025 lap 30:

| | in `RaceStateManager` only | in this endpoint only |
|---|---|---|
| `driver` | `gap_to_leader_s`, `track_status`, `is_in_lap`, `is_out_lap` | `driver_number`, `gap_ahead_s` |
| `rivals[*]` | `gap_to_leader_s`, `speed_st`, `stint` | `gap_ahead_s` |

`weather.rainfall` also differs in type: this endpoint coerces it to `int`, the replay engine leaves it `None` when the reading is absent.

None of these are read by the projection, which needs `interval_to_driver_s` and `is_pitting`, and both producers emit those. A new consumer should read this table rather than assume the two are interchangeable.

### Agent endpoints (POST)

| Path | Request Body | Agent | Description |
|---|---|---|---|
| `/api/v1/strategy/pace` | `PaceRequest` | N25 | Lap time prediction + CI |
| `/api/v1/strategy/pace-range` | `PaceRangeRequest` | N25 | Batch predictions over a lap range (Model Lab chart) |
| `/api/v1/strategy/tire` | `TireRequest` | N26 | Tire cliff estimation |
| `/api/v1/strategy/tire-range` | `PaceRangeRequest` | N26 | Batch degradation over a lap range (actual vs predicted) |
| `/api/v1/strategy/situation` | `SituationRequest` | N27 | Overtake + SC probability |
| `/api/v1/strategy/pit` | `PitRequest` | N28 | Pit duration + undercut analysis |
| `/api/v1/strategy/radio` | `RadioRequest` | N29 | NLP radio pipeline |
| `/api/v1/strategy/rag` | `RagRequest` | N30 | Regulation retrieval |
| `/api/v1/strategy/recommend` | `RecommendRequest` | N31 | Full orchestrator pipeline |

`/tire-range` reuses `PaceRangeRequest`, same `{year, gp, driver, lap_start, lap_end}` shape as `/pace-range`, just routed to the TCN instead of the XGBoost model.

Every POST endpoint above (plus `/pace-range` and `/tire-range`) sits behind its own rate-limit bucket, see "Rate limiting" above.

### Request schemas

```python
class PaceRequest(BaseModel):
    lap_state: Dict[str, Any]


class TireRequest(BaseModel):
    lap_state: Dict[str, Any]


class SituationRequest(BaseModel):
    lap_state: Dict[str, Any]


class PitRequest(BaseModel):
    lap_state: Dict[str, Any]


class RadioRequest(BaseModel):
    lap_state: Dict[str, Any]
    radio_msgs: List[Dict[str, Any]] = []
    rcm_events: List[Dict[str, Any]] = []


class PaceRangeRequest(BaseModel):
    """Shared by /pace-range and /tire-range."""

    year: int = 2025
    gp: str
    driver: str
    lap_start: int
    lap_end: int


class RagRequest(BaseModel):
    question: str


class RecommendRequest(BaseModel):
    lap_state: Dict[str, Any]
    gp_name: str = ""
    year: int = 2025
    gap_ahead_s: float = 2.0
    pace_delta_s: float = 0.0
    risk_tolerance: float = 0.5
    radio_msgs: Optional[List[Dict[str, Any]]] = None
    rcm_events: Optional[List[Dict[str, Any]]] = None
    # Three-letter code of the rival selected in the Strategy tab. When set,
    # gap_ahead_s / pace_delta_s are measured against this car instead of the
    # positional car ahead (#431). None keeps the old positional behaviour.
    rival: Optional[str] = None
```

### Response schemas

All agent endpoints return the generic `StrategyResponse` envelope. Swagger also exposes a typed result model per agent (`PaceResult`, `TireResult`, `SituationResult`, `PitResult`, `RadioResult`, `RagResult`, mirroring the dataclass fields in [Agents API reference](#/agents-api)) for self-documentation, but the actual response body is the untyped envelope below:

```python
class StrategyResponse(BaseModel):
    agent: str  # e.g. "pace", "tire", "radio", "orchestrator"
    result: Dict[str, Any]
```

### Error handling

Strategy endpoints catch `(KeyError, TypeError, ValueError)` from the underlying agent and return **422** (a bad/incomplete input); any other exception returns **500**. Both cases share the same structured `StrategyError` body:

```json
{
  "error": "ValueError",
  "agent": "pace",
  "detail": "Missing feature: compound_id"
}
```

`/pace-range` and `/tire-range` also return `503` when the requested year's featured parquet is not cached, and `404` when the GP or driver is not found in it.

## CORS

`CORSMiddleware` allows a single origin: `FRONTEND_URL` (default `http://localhost:8501`), not a wildcard. Credentials are dropped (`allow_credentials=False`; the web app reaches the backend same-origin through its nginx / Vite `/api` proxy, so cross-origin browser requests are the exception, not the rule), and both the method and header allowlists are enumerated rather than `"*"`: `GET`/`POST`/`OPTIONS` and `Content-Type`/`Accept`/`X-Request-Id`. The `ApiKeyMiddleware` described under "Authentication" wraps CORS from the outside (registered after it in `main.py`), so an unauthenticated request is rejected before any CORS or routing logic runs; `OPTIONS` preflight is exempted so it still completes.

## Swagger / OpenAPI

Auto-generated at `http://localhost:8000/docs` when the backend is running.
