# Audit: Voice Stack (chat voice I/O)

**Scope:** the interactive voice loop of the chat: STT in (Whisper via HF transformers), single LLM turn, TTS out (Edge-TTS), the backend endpoint `src/telemetry/backend/api/v1/endpoints/voice.py`, the Streamlit voice UI plus the React+OGL audio orb, and the low-latency migration plans. This is distinct from the radio-transcription NLP pipeline (audited in #302). Plan only, no code.

**Cross-references (owned elsewhere, not duplicated here):** Security #223 (S-4 upload hardening), LLM-cost #261 (provider timeout, blocking LLM calls), NLP/radio #302 (radio Whisper + JSON cache), frontend migration #25 (React SPA).

## Executive summary

The voice stack is a working single-turn demo, not a production feature. The pipeline (record, `/voice-chat`, base64 MP3 back, autoplay) works end to end, the TTS-tuned system prompt (`voice.py:52`) and the LLM-failure spoken fallbacks (`voice.py:308`) are genuinely good, and the orb is already React (portable to #25 as-is). But the endpoint runs blocking Whisper inference and a blocking LLM call inside an async route (freezes the whole backend per turn), a size/duration validator that already exists in the codebase is simply not wired into the endpoint, no request in the loop has a timeout on either side, TTS failure throws away a completed STT+LLM turn, and the file carries doc drift from three TTS generations (pyttsx3, Nemotron/Qwen3, Edge-TTS). Warm-loop latency is roughly 2 to 4 s per turn: fine for a turn-based demo, far from the ~300 ms real-time target in the migration plan.

**Verdict: defer voice to fast-follow.** Ship v1 with voice flagged experimental after the cheap correctness fixes (Phases V1 and V2, all S/M). The latency rework (streaming, or the Nemotron/Qwen3 stack that was already attempted and reverted for dependency weight) is a fast-follow track that should wait for migration #25.

## Current architecture (verified against code, 2026-07-07)

- **STT:** `openai/whisper-small` via `transformers.pipeline`, in-process singleton (`backend/services/voice/stt_service.py:55`). Bytes go to a temp `.wav` file, ffmpeg decodes by content (ffmpeg is an undeclared system dependency).
- **LLM:** shared `llm_service` router (`F1_LLM_PROVIDER`, LM Studio or OpenAI), single turn, `stream=False`, temp 0.6, max_tokens 220 (`voice.py:293`).
- **TTS:** Edge-TTS (Azure Neural over Microsoft's public unauthenticated endpoint), async, buffered MP3, 4 curated English voices (`backend/services/voice/tts_service.py`). No API keys anywhere (good).
- **Frontend:** Streamlit components (`frontend/components/voice/voice_chat.py`, `voice_input.py`) + `voice_api.py` httpx client + `streamlit_audio_viz` custom component (React + OGL: `AudioOrb.tsx`, `Iridescence.tsx`) with idle/recording/processing/playing states.
- **Memory-file drift:** `project_voice_models.md` claims Nemotron/Qwen3 is "COMPLETE"; the later `project_voice_stack_migration.md` and the code confirm both were reverted (NeMo ~5 GB, qwen_tts ~600 MB) to Whisper + Edge-TTS. The code is the truth; the older memory is stale.

## Findings register

### P0

- **V-1. Blocking STT and LLM calls inside the async route freeze the backend event loop.** `voice.py:275` calls `stt.transcribe_audio()` (synchronous Whisper inference, plus a synchronous model download/load on the first request via the lazy singleton at `voice.py:37`) and `voice.py:293` calls `lm_send_message(stream=False)` (synchronous HTTP), both inside `async def voice_chat`. One voice turn stalls every other endpoint on the server (SSE sim stream, text chat, health) for seconds; a hung LLM provider stalls it indefinitely because there is no timeout (that timeout fix is owned by #261). Voice-specific fix: dispatch STT and the LLM call through `asyncio.to_thread` / `run_in_threadpool`, and warm the STT singleton at startup instead of on first request. Size: S.

### P1

- **V-2. Upload guard exists but is not wired (fix owned by Security #223 S-4).** `voice.py:98` validates extension only; `voice.py:139` does unbounded `audio.file.read()`. The voice-specific angle: `backend/services/voice/audio_processor.py:43` already implements the `MAX_AUDIO_SIZE` (25 MB) and `MAX_AUDIO_DURATION` (120 s) checks against `voice_config.py:23`, and no endpoint imports it. The constants are dead config; the fix is one import plus one call, executed under #223's spec. Size: S.
- **V-3. No timeout anywhere in the loop, on either side.** Frontend `voice_api.py:24` sets `TRANSCRIBE_TIMEOUT = SYNTHESIZE_TIMEOUT = VOICE_CHAT_TIMEOUT = None` (httpx waits forever; the `httpx.TimeoutException` handlers below are dead code), backend has no LLM timeout (#261), and `VOICE_API_TIMEOUT = 120` in `voice_config.py:31` is never read. Failure mode: orb stuck in "processing" forever. Fix: honor `VOICE_API_TIMEOUT` client-side; backend provider timeout via #261. Size: S.
- **V-4. Hardcoded CUDA device breaks CPU machines.** `voice_config.py:12` pins `WHISPER_DEVICE = 0` with no env override; on a machine without an NVIDIA GPU the STT singleton raises at init and every voice endpoint returns 503. Needs an env-var with CPU fallback (mirrors how the radio pipeline picks device). Size: S.
- **V-5. TTS failure discards a completed STT+LLM turn, and Edge-TTS is an unauthenticated internet dependency.** If `synthesize_speech_async` raises (`voice.py:327`), the whole request 500s even though transcript and response text exist. Edge-TTS calls Microsoft's public endpoint (documented in `tts_service.py:2-8`): offline demo rooms, rate limits, or upstream breakage kill voice entirely. Fix: degrade to a text-only response (return `transcript` + `response_text`, empty audio, a `tts_error` flag) and let the UI render the bubble without playback. Size: S.
- **V-6. Voice selection mutates a shared singleton.** `voice.py:325` calls `tts.set_voice(voice)` on the process-wide TTS instance (`tts_service.py:197`), so one user's voice choice leaks into every concurrent session and persists across requests. Fix: pass the voice per call into `synthesize_speech_async`. Size: S.

### P2

- **V-7. Latency: usable turn-based, not real time.** Warm loop is roughly 2 to 4 s (Whisper-small ~0.3 s GPU, non-streaming LLM 1 to 3 s, Edge-TTS 0.3 to 0.5 s per its own docstring, plus base64-in-JSON transport and a full Streamlit rerun); cold start adds model load/download. Nothing streams: `stream=False` at `voice.py:299`, TTS fully buffered (`tts_service.py:133`), audio fully base64-encoded (`voice.py:330`). The ~300 ms target in `project_voice_models.md` requires the streaming rework (Phase V4). Size: L (that is Phase V4, not a quick fix).
- **V-8. Two disjoint Whisper stacks.** Voice uses HF transformers `openai/whisper-small` (`stt_service.py:55`); the radio pipeline uses the `openai-whisper` package with the `turbo` model plus a JSON transcription cache (`src/nlp/pipeline.py:425`, #302's territory). Not sharing the cache is correct (live utterances are never repeated), but two frameworks and two weight sets in memory when both surfaces run is avoidable maintenance and VRAM cost. Unify the framework (not necessarily the model size) with #302. Size: M.
- **V-9. Contract and doc drift, including one live bug.** `/synthesize` returns Edge-TTS MP3 bytes labeled `media_type="audio/wav"` and `filename=speech.wav` (`voice.py:218`); its description still says pyttsx3 (`voice.py:193`); `/voice-chat`'s summary still says "STT (Nemotron) ... TTS (Qwen3)" (`voice.py:241`, `voice.py:250`); `duration` is always 0.0 (`stt_service.py:159`). Fix media type plus a docstring sweep. Size: S.
- **V-10. Single-turn with zero race context.** `chat_history=[]` and `context={}` (`voice.py:288`), so the voice assistant cannot follow up or see the lap state the text chat has. A voice copilot that forgets every turn is a demo ceiling, not a product. Wire the same history/context plumbing the text chat (MCP engine) uses. Size: M.

### P3

- **V-11. English-only voice UX.** All 4 curated voices are English (`tts_service.py:170`, `voice_chat.py:16`); Whisper auto-detects language (the `language` arg is informational only, `stt_service.py:96`), so a Spanish utterance yields a Spanish reply read by an English neural voice. Acceptable for the English-first project; document it or add one ES voice. Size: S.
- **V-12. Orb belongs to migration #25; minor flags.** The orb is already a React + OGL app (`streamlit_audio_viz/frontend/src/AudioOrb.tsx`, `Iridescence.tsx`, `useAudioLevel.ts`) wrapped as a Streamlit component; in the React SPA it should be extracted, not rewritten. Hardcoded `_RELEASE = True` dev flag in `streamlit_audio_viz/__init__.py:13` and a committed `node_modules` under the component are cleanup items for the move. Size: M (inside #25's scope).

## Phased plan (each phase = one future sub-issue)

| Phase | Title | Contents | Size |
|---|---|---|---|
| V1 | Unblock the loop and wire the guards | V-1 (to_thread + startup warm-up), V-2 (wire `audio_processor` under #223's spec), V-3 (timeouts, with #261), V-4 (device env-var) | M |
| V2 | Graceful degrade and contract fixes | V-5 (text-only TTS fallback), V-6 (per-request voice), V-9 (MP3 media type + doc sweep) | S |
| V3 | Conversational voice | V-10 (multi-turn history + race context via the chat MCP engine) | M |
| V4 | Low-latency track (fast-follow) | V-7: streaming LLM + chunked TTS playback first; only then re-evaluate Nemotron/Qwen3 per `project_voice_models.md` (already attempted once, reverted for dependency weight, so treat as research not default) | L |
| V5 | Orb into the React SPA | V-12, executed inside frontend migration #25 | M |

Sequencing: V1 and V2 are cheap, correctness-shaped, and worth landing in the v1 window (V1 also protects the non-voice backend from voice traffic). V3 onward is fast-follow, and V4/V5 should wait for #25 so the streaming transport is designed once for the SPA, not twice.

## Open questions

1. Offline demos: is an offline TTS fallback (pyttsx3-grade or Qwen3) a requirement for live defenses/demos, or is "no internet, no voice" acceptable? Decides how far V2's degrade path must go.
2. Should voice and text chat share one provider config forever, or does the deferred `F1_VOICE_LLM_PROVIDER` split (noted in `project_voice_stack_migration.md`) return in V3 when voice gets context (bigger prompts, latency-sensitive)?
3. Whisper unification (V-8): standardize on the `openai-whisper` package (radio's stack) or HF transformers (voice's stack)? Coordinate with #302 before either side refactors.
4. In the #25 SPA, does voice keep the request/response `/voice-chat` shape or move to a WebSocket for V4 streaming? Choosing now avoids building the REST path twice.
