# Getting started

**F1 StratLab is an open-source (Apache-2.0) multi-agent AI system for real-time Formula 1 race strategy**, combining seven ML models, six LangGraph sub-agents and one orchestrator. This page covers installation — for what it does and how it is wired, see the [architecture overview](#/architecture).

Three ways to get F1 StratLab running on your machine, from fastest to deepest.

<p align="center">
  <img src="/assets/demo/cli-demo.gif" alt="F1 StratLab CLI wizard in action" width="760"/>
  <br/>
  <sub>The <code>f1-strat</code> interactive wizard: pick the GP, drivers and provider, then watch the live inference panel.</sub>
</p>

## 1. Install the latest wheel

The quickest path. Installs the latest release into your current environment without cloning the repo.

```bash
uv pip install https://github.com/VforVitorio/F1-StratLab/releases/download/v__DOCS_VERSION__/f1_strat_manager-__DOCS_VERSION__-py3-none-any.whl
```

After install you have four console entry points:

```bash
f1-strat       # interactive launcher (recommended starting point)
f1-sim         # headless CLI simulation against a saved race
f1-arcade      # three-window PySide6 + pyglet experience
f1-streamlit   # Streamlit dashboards
```

First boot triggers a one-time download of the cached models and reference data into `~/.f1-strat/`. Subsequent runs are offline.

## 2. Clone the repo for development

If you want to edit the code, run the notebooks or contribute back:

```bash
git clone https://github.com/VforVitorio/F1-StratLab.git
cd F1-StratLab
uv sync --all-extras
```

`uv sync` reads `pyproject.toml`, resolves the lockfile and pulls the CUDA-routed PyTorch wheel automatically on Windows and Linux (CPU build on macOS).

Run the simulation against a saved race:

```bash
uv run scripts/run_simulation_cli.py Bahrain NOR McLaren --no-llm
```

Drop `--no-llm` once you have an LLM provider configured (LM Studio at `http://localhost:1234/v1` or `OPENAI_API_KEY` in `.env`).

## 3. Docker

For a reproducible all-in-one setup, see [Setup and deployment](#/setup) for the Docker compose recipe that boots the FastAPI backend, the Streamlit frontend and the Qdrant store in one command.

## Where to next

- New to the architecture? Start at [Architecture overview](#/architecture).
- Want to see the agents in action? Open [Arcade quick start](#/arcade-quick-start).
- Looking for an API to call from your own code? Jump to [Multi-agent system](#/agents-api).
- Curious about the numbers in the thesis? See [Thesis results](#/thesis).

## FAQ

### Do I need a GPU?

No, but it helps. `uv sync` pulls the CUDA-routed PyTorch wheel on Windows and Linux (a CPU build on macOS), so the stack runs on CPU. A GPU mainly accelerates Whisper radio transcription and the TCN tire model — the benchmark latencies on the [thesis results](#/thesis) page (Whisper 233.9 ms, NLP pipeline 42.1 ms) are GPU figures; on CPU it is slower but fully functional.

### Why is the first run slow?

The first boot triggers a one-time download of the cached models and reference data into `~/.f1-strat/`; subsequent runs are offline. The simulation also pre-warms Whisper and the agents before lap 1, so a cold start takes a while — pass `--no-llm` for a fast headless run.

### Which LLM providers are supported?

OpenAI and LM Studio — the system is provider-agnostic and does not depend on a single vendor. Set `F1_LLM_PROVIDER=openai` to use the OpenAI API; the default is a local LM Studio server at `http://localhost:1234/v1`.

### Do I need an API key?

Only for the LLM synthesis layer. Run with `--no-llm` and the ML models plus Monte Carlo simulation still produce a recommendation with no key required. With LM Studio you need no key; with OpenAI, put `OPENAI_API_KEY` in your `.env`.
