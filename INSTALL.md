# Install Guide — F1 StratLab

Three install paths, one per surface, each a **single command** once the
prerequisites are on the machine.

---

## Prerequisites

- Python **3.10 or 3.11** (the project pins `>=3.10,<3.13` in
  `pyproject.toml`).
- `OPENAI_API_KEY` in a `.env` at the repo root (or exported in the
  shell). The Arcade and CLI paths use OpenAI `gpt-4.1-mini` by default;
  set `F1_LLM_PROVIDER=lmstudio` to route to a local LM Studio server on
  `http://localhost:1234` instead.
- For Streamlit Docker flow: **Docker Desktop** (Windows/Mac) or
  `docker + compose` plugin (Linux).
- For Arcade: a working OpenGL graphics stack (any modern laptop
  qualifies; arcade auto-detects).
- For CLI / Arcade wheel install: [`uv`](https://docs.astral.sh/uv/)
  (recommended) or plain `pip`. `uv` resolves the CUDA-specific PyTorch
  wheel automatically via the `[tool.uv.sources]` table in
  `pyproject.toml`.

---

## CLI — headless strategy replay with Rich live panels

```bash
uv tool install "git+https://github.com/VforVitorio/F1_Strat_Manager.git"
f1-strat
```

`uv tool install` drops two global binaries: `f1-strat` (interactive
wizard with ASCII banner + arrow-key pickers for race / driver / laps /
provider / head-to-head rival) and `f1-sim` (the headless argparse
form). The wizard auto-resolves the team from
`laps_featured_2025.parquet`, shells out to `f1-sim` under the hood and
turns Ctrl+C into a clean italic *Interrupted.* notice.

Prefer the scripted form for demos and CI:

```bash
f1-sim Suzuka VER "Red Bull Racing" --year 2025
```

`--no-llm` runs the ML-only path (no OpenAI spend). See
`python -m scripts.run_simulation_cli --help` for every flag.

Already installed from a source checkout? `uv sync && uv run f1-strat`
(or `uv run f1-sim ...`) works too.

---

## Arcade — 3-window race replay + live dashboard + telemetry

```bash
uv tool install "git+https://github.com/VforVitorio/F1_Strat_Manager.git"
f1-arcade --viewer --year 2025 --round 3 --driver VER --team "Red Bull Racing" --driver2 LEC --strategy
```

Three windows spawn from that one command:

1. Arcade replay (pyglet) — track · leaderboard · weather · driver info
2. Strategy Dashboard (PySide6) — orchestrator + 6 agent cards + charts
3. Live Telemetry (PySide6) — 2×2 grid Delta / Speed / Brake / Throttle

**Docker is NOT recommended for Arcade**: pyglet + Qt need a host OpenGL
context and a native display. Cross-platform X forwarding from a
container is fragile on Windows / Mac and has no benefit over a local
install. Use `uv tool install` and run on the host.

See [`docs/arcade-quick-start.md`](docs/arcade-quick-start.md) for the
controls legend, troubleshooting and window tour.

---

## Streamlit — post-race analysis UI (backend + frontend)

```bash
git clone https://github.com/VforVitorio/F1_Strat_Manager.git
cd F1_Strat_Manager
docker compose up
```

Opens:

- Frontend at `http://localhost:8501`
- FastAPI backend at `http://localhost:8000`

Both containers mount `./src/telemetry` so edits reload without a
rebuild. `.env` at repo root is picked up by the backend image.

If you prefer a local (non-Docker) Streamlit run:

```bash
uv sync
uv run f1-streamlit
```

`f1-streamlit` is a wrapper around `python -m streamlit run
src/telemetry/frontend/app/main.py` that forwards any extra flag
(`--server.port`, `--server.headless`, etc.).

---

## Data bootstrap

All three surfaces read from `data/`:

- `data/processed/laps_featured_<year>.parquet` — featured lap data
- `data/raw/<year>/<Location>/` — per-race FastF1 pickle cache
- `data/processed/race_radios/<year>/<slug>/` — OpenF1 radio corpus +
  `rcm.parquet` for Race Control messages
- `data/tire_compounds_by_race.json` — canonical per-year GP calendar
  and compound allocation

The CLI and Arcade call `ensure_radio_corpus()` and FastF1's cache on
first run; a warm cache is zero-cost. For a production deploy without a
repo clone the data tree would be downloaded from the TFG's Hugging Face
mirror on first run (tracked under `project_cli_distribution_plan.md`,
deferred past the first release).

---

## Verification commands

After install, a quick sanity:

```bash
# CLI path — runs one lap with no LLM spend
f1-sim Melbourne VER "Red Bull Racing" --year 2025 --no-llm --laps 1-1

# Arcade path — opens the replay with strategy pipeline warmup
f1-arcade --viewer --year 2025 --round 3 --driver VER --team "Red Bull Racing" --strategy

# DRS zones audit (cross-check against FIA 2025 Event Notes)
python scripts/verify_drs_zones.py --year 2025 --summary
```

---

## Uninstall

```bash
uv tool uninstall f1-strat-manager
docker compose down      # from the repo root for the Streamlit stack
```
