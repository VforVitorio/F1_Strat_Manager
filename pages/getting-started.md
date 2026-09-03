# Getting started

**F1 StratLab is an open-source (Apache-2.0) multi-agent AI system for real-time Formula 1 race strategy**, combining seven ML models, six LangGraph sub-agents and one orchestrator. This page covers installation, for what it does and how it is wired, see the [architecture overview](#/architecture).

Three ways to get F1 StratLab running locally, from fastest to deepest.

<p align="center">
  <video src="/assets/demo/arcade-demo.mp4" poster="/assets/demo/arcade-demo-poster.jpg" width="760" autoplay loop muted playsinline preload="metadata" aria-label="F1 StratLab Arcade replay in action"></video>
  <br/>
  <sub>The <code>f1-arcade</code> replay: a 2D race with the strategy dashboard and live telemetry, all from one command.</sub>
</p>

## 1. Install the latest wheel

The quickest path. Installs the latest release into the current environment without cloning the repo.

```bash
uv pip install https://github.com/VforVitorio/F1-StratLab/releases/download/v2.6.1/f1_strat_manager-2.6.1-py3-none-any.whl
```

After install, seven console entry points are available:

```bash
f1-strat       # interactive launcher (recommended starting point)
f1-sim         # headless CLI simulation against a saved race
f1-arcade      # pyglet 2D replay plus the two PITWALL windows
f1-webapp      # post-race web app (wraps `docker compose up`)
f1-prefetch    # fill the arcade replay cache ahead of time
f1-eval        # regenerate the evaluation reports (registry, calibration, hygiene, projection, ...)
f1-pitwall     # attach the two PITWALL windows to an arcade already running
```

The first four are what a new user actually runs.

`f1-prefetch` exists because the first launch of any given race builds its replay telemetry, which takes minutes. It runs the same preparation the arcade menu runs, for a whole season or a rounds spec, so the wait can be paid in advance rather than while somebody is waiting to watch:

```bash
f1-prefetch --year 2025                   # the whole calendar
f1-prefetch --year 2025 --rounds 1,3,5-8  # commas and ranges
f1-prefetch --year 2025 --with-radio      # also fetch the team radio the agents read
```

Rounds already cached are skipped without being loaded, so re-running it costs one filesystem check per round.

`f1-eval` and `f1-pitwall` are developer tools rather than end-user surfaces. The first writes versioned markdown and JSON reports under `documents/eval_reports/` (`f1-eval registry`, `f1-eval calibration`, `f1-eval all`, ...); the second opens the PITWALL windows against an arcade process that is already running, which is how the UI is developed without restarting the replay.

First boot triggers a one-time download of the cached models and reference data into `~/.f1-strat/`. Subsequent runs are offline.

## 2. Clone the repo for development

To edit the code, run the notebooks or contribute back:

```bash
git clone https://github.com/VforVitorio/F1-StratLab.git
cd F1-StratLab
uv sync --all-extras
```

`uv sync` reads `pyproject.toml`, resolves the lockfile and pulls the CUDA-routed PyTorch wheel automatically **on Windows**. Everything else, Linux and macOS included, resolves to the CPU wheel: CI runners and CPU-only Linux boxes were downloading about 5 GB of unused CUDA libraries, so the markers were narrowed deliberately (`pyproject.toml`, #251). A Linux GPU box opts back in by editing those markers.

Run the simulation against a saved race:

```bash
uv run scripts/run_simulation_cli.py Sakhir NOR McLaren --no-llm
```

Drop `--no-llm` once an LLM provider is configured (LM Studio at `http://localhost:1234/v1` or `OPENAI_API_KEY` in `.env`).

## 3. Docker

For a reproducible all-in-one setup, see [Setup and deployment](#/setup) for the Docker compose recipe that boots the FastAPI backend and the React web app in one command. Qdrant runs on-disk inside the backend process rather than as its own container, so there is nothing extra to start.

## Where to next

- New to the architecture? Start at [Architecture overview](#/architecture).
- Want to see the agents in action? Open [Arcade quick start](#/arcade-quick-start).
- Looking for an API to call from external code? Jump to [Multi-agent system](#/agents-api).
- Curious about the numbers in the thesis? See [Thesis results](#/thesis).

## FAQ

### Do I need a GPU?

No, but it helps. `uv sync` pulls the CUDA-routed wheel **on Windows only**; Linux and macOS get the CPU build, so out of the box the stack runs on CPU almost everywhere. A GPU mainly accelerates Whisper radio transcription and the TCN tire model, the benchmark latencies on the [thesis results](#/thesis) page (Whisper 233.9 ms, NLP pipeline 42.1 ms) are GPU figures; on CPU it is slower but fully functional.

### Why is the first run slow?

The first boot triggers a one-time download of the cached models and reference data into `~/.f1-strat/`; subsequent runs are offline. The simulation also pre-warms Whisper and the agents before lap 1, so a cold start takes a while, pass `--no-llm` for a fast headless run.

### Which LLM providers are supported?

OpenAI and LM Studio, the system is provider-agnostic and does not depend on a single vendor. Set `F1_LLM_PROVIDER=openai` to use the OpenAI API; the default is a local LM Studio server at `http://localhost:1234/v1`.

### Do I need an API key?

Only for the LLM synthesis layer. Run with `--no-llm` and the ML models plus Monte Carlo simulation still produce a recommendation with no key required. LM Studio needs no key; OpenAI needs `OPENAI_API_KEY` in `.env`.
