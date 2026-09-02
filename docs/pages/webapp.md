# Web app

**The web app is the post-race surface: a React single-page app over the FastAPI backend, launched with `f1-webapp`.** It replaced the Streamlit frontend in v2.0.0, and it is used after a race rather than during one. The Arcade is the live surface; this one is for looking back at what happened and asking why.

## Launching it

```bash
f1-webapp
```

That wraps `docker compose up` and brings two services online: the FastAPI backend on `:8000` and the SPA served by nginx on `:8501`, with `/api` reverse-proxied to the backend so the browser only ever talks to one origin.

`f1-webapp --help` works without Docker installed, so the options can be read before committing to a pull.

A `.env` with `OPENAI_API_KEY` is required, or `F1_LLM_PROVIDER=lmstudio` when running a local model. The chat tab is the only part that needs it; everything else works without an LLM.

## What is in it

Six tabs, grouped into three sections by the nav rail (`webapp/src/app/Rail.tsx`):

| Section | Tab | What it answers |
|---|---|---|
| Telemetry | Dashboard | What the car did, lap by lap: speed traces, sector times, tyre life |
| Telemetry | Comparison | Two drivers at 60 fps over the same lap, synchronised |
| Telemetry | Lab | The ML models themselves, with their calibration and thresholds |
| Pit wall | Strategy | The strategy surface: scenarios, projections and the decision trail |
| Pit wall | Race | The multi-agent pit-wall call for any lap, with each agent's reasoning |
| Assist | Chat | Natural language over the finished race, streaming, rendering tool results inline |

The Chat tab is an MCP client: it reaches the same fourteen tools the agents expose, so an answer about lap 32 runs the same code path the Race tab does rather than a parallel implementation.

## How it relates to the other surfaces

All three surfaces call one shared inference engine, `src/strategy/inference/engine.py::run_lap`. The web app reaches it over HTTP through the backend, the Arcade calls it in-process, and the CLI calls it directly. That is deliberate: an earlier arrangement had the Arcade carrying its own copy of the orchestrator, it drifted, and it shipped wrong results.

Therefore, a strategy call shown in the web app is the same call the CLI would print for that lap. If they ever disagree, that is a bug rather than a difference of surface.

## Where the code lives

The SPA is a git submodule at `src/telemetry/`, under `webapp/`: React 19, Vite, TypeScript, Tailwind and ECharts. The backend it talks to is `src/telemetry/backend/`, documented in [Backend API](#/backend-api).

Its build artefact is not committed. CI builds it, and the compose file builds it for a local run.

## The surface it replaced

Streamlit served this role until v2.0.0. That page is kept as a historical record at [Streamlit frontend (legacy)](#/streamlit); the paths it describes no longer exist on `main`.
