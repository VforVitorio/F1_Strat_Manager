# Meet the author

Hi — I'm **Victor Vega Sobral** (a.k.a. VforVitorio), a graduate in
Intelligent Systems Engineering from UIE Campus Coruña and AI & Data
Intern at NTT DATA Spain. F1 StratLab was my Final-Degree Project,
defended in June 2026 with a 10/10 grade and Distinction (Matricula de
Honor) and unanimously recommended by the tribunal for publication as
a research article: an open-source multi-agent AI for real-time
Formula 1 race strategy, built end-to-end from telemetry ingestion and
ML modelling to a LangGraph orchestrator and three operator surfaces.
I built it because I wanted to see how far a single thesis could push
a digital twin of an F1 race — and because nobody else was going to do
it for me.

## Where to find me

- **GitHub** — [@VforVitorio](https://github.com/VforVitorio)
- **Project landing** — [f1stratlab.com](https://f1stratlab.com/)
- **DeepWiki** — [F1 StratLab on DeepWiki](https://deepwiki.com/VforVitorio/F1-StratLab)
- **LinkedIn** — [victorvegasobral](https://www.linkedin.com/in/victorvegasobral/)
- **Hugging Face dataset** — [f1-strategy-dataset](https://huggingface.co/datasets/VforVitorio/f1-strategy-dataset)
- **Portfolio** — [victorvegasobral.com](https://victorvegasobral.com)

## About the project

F1 StratLab is a multi-agent system that turns a live (or replayed)
Formula 1 race into actionable strategy. Seven ML models cover the
quantitative core — lap-time delta, tire degradation with MC Dropout,
overtake probability, safety-car prior, pit duration and undercut
success — and feed six LangGraph sub-agents (pace, tire, race
situation, pit strategy, radio, RAG) which a single orchestrator
(N31) fuses into a
Pydantic-typed decision per lap: action, pace target, risk level,
and a plan for the next pit window.

The same engine drives three operator surfaces: a Streamlit dashboard
for analysts (backed by a FastAPI/MCP API for programmatic access and
chat tool-calling), a CLI for headless replays, and a three-window
arcade (race replay, strategy dashboard, live telemetry) built in
PySide6 + pyglet for the demo experience. The whole stack is open
under Apache-2.0 and shipped as wheels and GitHub releases through
release-please automation.

This documentation site is the engineering companion to the thesis
memoria — every notebook, model, agent and surface is wired into the
graph view so you can navigate by topic, by tag, or by cross-reference.

## Acknowledgements

- **Academic** — UIE Campus Coruña, Intelligent Systems Engineering
  faculty and thesis advisors.
- **Open data community** — [FastF1](https://docs.fastf1.dev) and
  [OpenF1](https://openf1.org) provide the telemetry, lap and timing
  data this entire project depends on.
- **Open-source libraries** — LangGraph, LightGBM, XGBoost, PyTorch,
  Pydantic, FastAPI, Streamlit, PySide6, pyglet, Qdrant, and the
  Hugging Face ecosystem.
- **Reference work** — TUMFTM race-simulation for the pit-delta
  framing, plus the wider F1 analytics community whose public
  notebooks shaped the early modelling decisions.

No copyright infringement intended. Formula 1, F1, and related marks
are trademarks of Formula One Licensing B.V. and are used here for
reference only. This project is not affiliated with, endorsed by, or
in any way officially connected to Formula 1, the FIA, or any F1 team.
