"""F1 StratLab: Arcade race replay UI.

A pyglet window that renders the race on a real 2D circuit with strategic
overlays mirroring the Rich CLI, and hosts the TCP broadcast its follower
windows read.

**It does NOT consume the backend's SSE stream, and has not for a long time.**
The strategy pipeline runs IN THIS PROCESS (`strategy_pipeline.py`, delegating
to `src/strategy/inference/engine.py::run_lap`), so the arcade has no runtime
dependency on FastAPI at all - which is the whole reason `f1-arcade` needs no
backend.

The backend's own `/api/v1/strategy/simulate` still exists and still streams;
what is gone is any path from it to here. `src/arcade/strategy.py` says the
same thing from the other side, that the two share an engine rather than a
transport. This paragraph used to end by pointing at dead SSE constants in
`config.py` and a note explaining them: both were deleted, so the sentence was
citing something that is not there.
"""
