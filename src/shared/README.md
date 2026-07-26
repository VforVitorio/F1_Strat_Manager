# src/shared: Legacy data-extraction wrappers (archived)

**Status: archived.** Early iteration of the data-extraction layer that
predates the cleaner `src/data_extraction/` reorganisation. The modules
here are not imported by any active pipeline and exist only because the
original notebooks referenced them by name; deleting them now would break
the historical jupytext exports under `src/strategy/` and `src/vision/`.

The canonical extractors live in
[`src/data_extraction/`](../data_extraction/README.md). When in doubt,
use that folder.

---

## Layout

```
src/shared/
└── data_extraction/
    ├── fastf1_extractor.py
    ├── openf1_extractor.py
    ├── data_augmentation.py
    └── video_processor.py
```

---

## Files

| File | Description | Successor |
|---|---|---|
| `data_extraction/fastf1_extractor.py` | `extract_f1_data(year, gp, session_type)`, initial FastF1 wrapper scoped to the 2023 Spanish GP. Writes laps + pit-stops + weather parquets under `f1_cache/` | [`src/data_extraction/fastf1/session_extractor.py`](../data_extraction/fastf1/session_extractor.py) and `scripts/download_data.py` |
| `data_extraction/openf1_extractor.py` | First-pass OpenF1 helper from the early intervals experiment | [`src/data_extraction/openf1/`](../data_extraction/openf1/) (intervals + radio dataset builders) |
| `data_extraction/data_augmentation.py` | Albumentations augmentation pipeline for the YOLO car-team image dataset (vision direction, archived) | [`src/data_extraction/legacy/image_augmentation.py`](../data_extraction/legacy/image_augmentation.py) |
| `data_extraction/video_processor.py` | Frame-extraction helper for the YOLO experiments | [`src/data_extraction/legacy/video_downloader.py`](../data_extraction/legacy/video_downloader.py) |
| `__init__.py` | Empty package marker |

---

## Why two parallel folders exist

The reorganisation into `src/data_extraction/{openf1, fastf1, legacy}/`
happened **after** several notebooks had already pinned imports to
`from src.shared.data_extraction.* import *`. Migrating those imports
without breaking the notebooks would require touching every cell that
loads the helpers, so the safer move was to keep the old layout
side-by-side and let the new code use the cleaner one.

A future `src/` cleanup pass can delete `src/shared/` entirely once the
last notebook reference is gone (search for `from src.shared` to find the
remaining call sites).
