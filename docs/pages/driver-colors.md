# Driver Colors: Year-Aware System

## Location

`src/telemetry/webapp/src/lib/drivers.ts`

> **This page used to document a Python module that no longer exists.** The
> year-aware palette was born in the Streamlit frontend at
> `frontend/components/common/driver_colors.py`, and that whole tree was
> deleted when the React web app replaced Streamlit in v2.0.0 (#551). The
> *design* survived the move intact — it was ported one for one — so this page
> now describes the TypeScript implementation that is actually running.

**It is still not shared with the backend.** `src/telemetry/backend/core/driver_colors.py`
is a separate, older implementation: a single flat `DRIVER_COLORS` dict labelled
"F1 2024 Driver Colors", with its own (different) hex values and no `year`
parameter at all. It predates the 2025 driver-swap handling described below — it
still maps `HAM` to a Mercedes-silver hex and has no entry for `ANT`, `BOR`,
`HAD`, or the 2025 `SAI`-to-Williams move. Used by
`backend/api/v1/endpoints/comparison.py` and `backend/services/telemetry_service.py`.
Treat the two as independent palettes kept in sync by hand: known tech debt, not
design intent.

## Purpose

F1 driver lineups change every season. A driver may switch teams between years,
so the colour associated with a driver abbreviation must be season-specific.
This module provides a year-aware palette covering the 2023-2025 seasons.

## Design

### Team base colours

Each team has two hex constants, one for the primary driver and one for the
secondary:

```ts
const RED_BULL = '#3671C6'     // primary (e.g. VER)
const RED_BULL_2 = '#1B3D8E'   // secondary
```

### Per-year lineups

`DRIVER_COLORS_BY_YEAR` maps a season to a `{ driverCode: hex }` record, so the
same code can resolve to different teams in different years.

### Flat fallback

`DRIVER_COLORS_FLAT = DRIVER_COLORS_BY_YEAR[2025]` serves any caller that does
not pass a year, and `DEFAULT_DRIVER_COLOR` (`#A259F7`) is returned for a code
the palette does not know.

## Public API

### `getDriverColor(code: string, year?: number): string`

Team colour for a driver in a given season. The code is case-insensitive. With
no `year`, or with a year the palette has no lineup for, it falls back to the
flat 2025 record; an unknown code returns `DEFAULT_DRIVER_COLOR`.

### `DEFAULT_DRIVER_COLOR`

The purple used for anything unrecognised, exported so callers can compare
against it rather than hardcoding the hex a second time.

## Usage

```ts
import { getDriverColor } from '@/lib/drivers'

getDriverColor('HAM', 2024)  // Mercedes teal
getDriverColor('HAM', 2025)  // Ferrari red — same driver, different team
getDriverColor('VER')        // flat 2025 fallback
```

## Where it is used

It lives in `src/lib/` rather than a feature folder because several features
consume it, so shared code stays app-level and the features stay decoupled
siblings:

- `features/dashboard/components/`, lap and channel chart line colours
- `features/comparison/`, toolbar and replay channel colours
- `features/chat/charts/`, colours for chat tool-result charts
- `components/radio/RadioBrowser.tsx`, per-driver radio tagging

Backend (static module, `backend/core/driver_colors.py`):

- `backend/api/v1/endpoints/comparison.py`, comparison endpoint colour assignment
- `backend/services/telemetry_service.py`, telemetry data colour tagging
