# Security Phase A — Design (issue #224, epic #223)

**Status:** PROPOSAL — pending Víctor's review + the §8 decisions before any code is written (DESIGN-BEFORE gate).
**Author:** Fable, orchestrated during the Sprint 4 session.
**Scope:** `src/telemetry/backend/` (a git submodule → implementation PRs land in `F1_Telemetry_Manager`, with a pointer bump in the parent). No code was written for this doc.

---

## 0. State-of-the-tree correction (read first)

The `AUDIT_SECURITY.md` snapshot (2026-07-05) is partly stale — three of its findings already shipped, which narrows Phase A to exactly the three sub-items of #224:

| Audit finding | Status today | Evidence |
|---|---|---|
| S-5 provider timeout (`DEFAULT_TIMEOUT=None`) | **DONE** | `services/chatbot/llm_service.py:45` — finite `F1_LLM_TIMEOUT` (60s OpenAI / 120s LM Studio), applied at both call sites |
| S-7 no rate limiting | **DONE** | `core/rate_limit.py` token bucket, wired on `/chat/*`, `/strategy/*`, `/voice/*`, `/simulate` |
| S-8 CORS credentials + wildcard | **DONE** | `main.py` CORS: `allow_credentials=False`, enumerated methods/headers, single `FRONTEND_URL` origin |

So Phase A = **A1** (auth + `/mcp` + bind) + **A2** (tool allowlist) + **A3** (per-request cost cap). A3 is smaller than the audit implied because the per-client rate limiter already exists.

---

## 1. Threat model (grounded)

The backend has **zero identity boundary** today. `main.py` mounts the full FastMCP tool catalogue at `app.mount("/mcp", mcp_app)` with no guard, and the Dockerfile starts `uvicorn --host 0.0.0.0`. The only `Depends` in the tree are data/rate-limit dependencies — none authenticate. On any non-localhost exposure:

- **(S-1) Anonymous access** — anyone reaching the port calls every prediction, the whole-race SSE sim, voice, and `/mcp` directly.
- **(S-2) Prompt-injection → tool execution** — `chat_engine` hands the model *every* tool (`mcp_bridge.list_openai_tools` returns all 14) and dispatches its pick blind; the only limiter is the happy accident that every tool is read-only today.
- **(S-5/cost) Unbounded spend** — `chat.py` passes client-controlled `max_tokens` straight into the loop; injected text can steer the model to `recommend_strategy` (500-sample Monte Carlo) or `get_telemetry` (outbound FastF1). The rate limiter caps *frequency*, not *per-request cost* or *tool reach*.

---

## 2. A1 — Auth (S-1)

**Recommendation: one static shared secret, enforced by a single ASGI middleware in `main.py`.** This is the minimal control that covers **both** the routers and the `/mcp` mount from one insertion point (a router-level `Depends` cannot cover an `app.mount()` sub-app; middleware runs before mount dispatch).

- New config `F1_API_KEY` in `core/config.py`; new `core/auth.py` (~30 lines) registered **before** CORS in `main.py`. Per request: `OPTIONS` → pass (CORS preflight); path in `{"/", "/health"}` → pass; else constant-time compare `X-API-Key` vs `F1_API_KEY`, else `401`.
- **Safe-by-default:** enforce **when `F1_API_KEY` is set**; when unset, log a loud startup warning and pass (local dev unchanged). Pair with a **startup guard** that refuses to boot if the bind is non-localhost **and** the key is unset — fail-closed on the only dangerous combination.
- **Bind:** Dockerfile `CMD` default `--host 127.0.0.1`; `0.0.0.0` becomes an explicit opt-in. This single line converts "anonymous-open to the world" into "anonymous-open to localhost only" even before the key is set.
- **Not** OAuth/JWT/user accounts: a single-user TFG deploy has no IdP or user store; a shared secret is O(1) to operate and rotate. Reversible: unset the key + localhost bind = today's behaviour.

Insertion points: `core/config.py`, `core/auth.py` (new), `main.py`, `Dockerfile`.

---

## 3. A1 — `/mcp` exposure

Mounted at `main.py` `app.mount("/mcp", mcp_app)` — a fully open Streamable-HTTP tool server. Two minimal controls:

1. The `require_api_key` middleware from §2 already sits above the mount, so `/mcp` inherits the key requirement for free.
2. A kill-switch `F1_MCP_ENABLED` (default `false` in prod, `true` for dev): guard the mount line with an `if`. Whether external MCP-client access is a wanted feature or an artifact of FastMCP is **OQ-3** — until decided, the safe default is "not exposed unless turned on".

---

## 4. A2 — Tool allowlist (S-2)

All 14 currently-invocable tools are read-only; the containment must exist **before** the first write/export tool, not after.

| Tool | Risk tier |
|---|---|
| `predict_pace/tire/situation/pit`, `list_available_gps/drivers`, `get_lap_range`, `analyze_radio` | READ_SAFE |
| `query_regulations`, `recommend_strategy` (500-sample MC), `compare_drivers`, `get_lap_times`, `get_telemetry`, `get_race_data` | READ_EXPENSIVE |
| *(any write / export / file / shell tool)* | MUTATING — none exist |

**Mechanism — explicit allowlist, default-deny:**
- `TOOL_RISK_MAP` + `CHAT_ALLOWED_TOOLS` in `models/tool_schemas.py`, keyed by the **real dispatched MCP names** (the enum drifts: `LIST_GPS="list_gps"` vs the tool `list_available_gps`).
- Enforce at two chokepoints: (1) filter in `mcp_bridge.list_openai_tools()` so the model can't *see* a non-allowed tool; (2) guard in `chat_engine._stream_tool_response` before dispatch so a hallucinated/ungated name is refused.
- **Hard rule:** no write/export/file/mutating tool may be added to the MCP server until this allowlist ships and that tool is deliberately classified.

Insertion points: `models/tool_schemas.py`, `services/chatbot/mcp_bridge.py`, `services/chatbot/chat_engine.py`. All additive; no `src/agents/` edits.

---

## 5. A3 — Cost cap (S-5)

Provider timeout is already done. Remaining, minimal (no session-accounting subsystem):
1. Clamp `max_tokens` server-side: add `F1_CHAT_MAX_TOKENS` (config) and `min()` it at the chat boundary so a client can't request a 100k-token completion.
2. Make the 1-tool-per-turn limit explicit (a named constant/comment on `_first_tool_call`) so a refactor can't silently allow N-tool chaining from one injected message.
3. Per-client frequency is already covered by `core/rate_limit.py`. Combined, injected text cannot pump `recommend_strategy`/`get_telemetry` beyond `rate-limit × 1 tool × clamped tokens`.

Insertion points: `core/config.py`, `api/v1/endpoints/chat.py`, `services/chatbot/chat_engine.py`.

---

## 6. Implementation plan (small PRs, submodule unless noted)

| # | PR | Touches | Repo |
|---|---|---|---|
| A1a | localhost-bind default + `/mcp` kill-switch flag | `Dockerfile`, `main.py`, `core/config.py` | submodule |
| A1b | API-key middleware + startup guard + `/health` | `core/auth.py` (new), `core/config.py`, `main.py` | submodule |
| A1c | document env vars | `.env.example` (root) + submodule `.env.example` | parent + submodule |
| A2 | risk map + allowlist filter + dispatch guard | `models/tool_schemas.py`, `mcp_bridge.py`, `chat_engine.py` | submodule |
| A3 | clamp `max_tokens`; pin 1-tool-per-turn invariant | `core/config.py`, `chat.py`, `chat_engine.py` | submodule |
| A-fin | bump submodule pointer | parent gitlink | parent |

Order A1a → A1b → A2 → A3. A1a alone already removes the worst exposure (world-open → localhost-open), so it is shippable on its own the same day. Each lands in `F1_Telemetry_Manager` first, then the parent bumps the pointer.

---

## 7. Verification plan (one executable check per control, `TestClient` + `FakeOpenAI`, never a real provider)

- **A1 auth:** unauth request to `/strategy/*`, `/chat/*`, `/mcp` (with key set) → **401** on all; with key → 200/valid stream; `/health` + `/` open without key; startup guard raises on non-localhost bind + unset key.
- **A1 `/mcp`:** `F1_MCP_ENABLED=false` → `GET /mcp` = **404**; true + no key → 401.
- **A2 allowlist:** stub the model to emit a synthetic MUTATING tool call → `chat_engine` **refuses** (no dispatch); a READ_SAFE tool still dispatches; `list_openai_tools()` excludes non-allowed names.
- **A3 cost cap:** `max_tokens=1_000_000` → clamped to `F1_CHAT_MAX_TOKENS`; a single turn dispatches ≤1 tool. Timeout regression: `DEFAULT_TIMEOUT is not None`.

Green on all = Phase A done; then a 🔴 Fable VERIFY-AFTER pass on the implementation.

---

## 8. Open questions for Víctor (decide before implementation)

1. **Deploy scope** — single-user localhost forever, or a shared/public deploy? Sets whether A1 is belt-and-braces or non-negotiable, and whether per-user anything is worth it (the recommendation assumes single-user → one shared key).
2. **Auth enforcement default** — safe-by-default (enforce only when `F1_API_KEY` is set, but refuse a non-localhost boot without a key), or **fail-closed always** (refuse to boot with no key even on localhost)?
3. **`/mcp` in prod** — keep external MCP-client access behind auth (`F1_MCP_ENABLED=true`), or default it **off** (treat it as an artifact, not a feature)?
4. **Header convention** — `X-API-Key` (recommended, unambiguous) vs `Authorization: Bearer` (reuses standard tooling but visually collides with the *outbound* OpenAI bearer in `llm_service`)?
