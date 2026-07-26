# src/vision: Computer vision experiments (archived)

**Status: archived.** The computer-vision direction (broadcast frame
analysis with YOLO + OpenCV) was abandoned during the early phases of the
project in favour of timing-data-only inputs from FastF1 / OpenF1. The
files in this folder are kept so the git history of the experiments stays
self-contained, but no module here is imported by the active pipeline or
the multi-agent system.

---

## Files

| File | Description |
|---|---|
| [`gap_calculation.py`](gap_calculation.py) | Jupytext export of the broadcast-frame gap-calculation prototype. Loads a YOLOv8 checkpoint, runs detection on each frame, and infers inter-car distances from bounding-box geometry plus track scale. Uses absolute paths from the original development machine, not runnable as-is, kept as a design reference for the YOLO pipeline |
| `__init__.py` | Empty package marker |

---

## Why the direction was dropped

- Broadcast feeds are *delayed* and *cropped* unpredictably, so the gap
  estimates from frame geometry have systematic biases that timing data
  does not.
- The OpenF1 `/v1/intervals` endpoint and the FastF1 `Time` column give
  the same information at higher precision and with no inference cost.
- Team identification by livery (the original justification for the
  YOLO model) is also available directly from `DriverNumber` → `Team`
  lookups without any vision step.

The successor data path is documented in
[`src/data_extraction/`](../data_extraction/README.md) (OpenF1 +
FastF1 extractors). The on-track gap that the agents consume comes from
`src/simulation/race_state_manager.py` via the `Time` column.
