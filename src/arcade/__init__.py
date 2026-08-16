"""F1 StratLab: Arcade race replay UI.

A pyglet window that renders the race on a real 2D circuit with strategic
overlays mirroring the Rich CLI, and hosts the TCP broadcast its follower
windows read.

**It does NOT consume the backend's SSE stream, and has not for a long time.**
The strategy pipeline runs IN THIS PROCESS (`strategy_pipeline.py`, delegating
to `src/strategy/inference/engine.py::run_lap`), so the arcade has no runtime
dependency on FastAPI at all - which is the whole reason `f1-arcade` needs no
backend. The leftover SSE constants in `config.py` are dead; see the note there.
"""
