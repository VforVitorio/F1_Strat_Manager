# Streamlit Frontend

## Overview

The frontend is a multi-page Streamlit application at `src/telemetry/frontend/`. It communicates with the FastAPI backend via HTTP and renders telemetry visualizations, strategy recommendations, driver comparisons, and a chat interface.

Entry point: `frontend/app/main.py`.

<p align="center">
  <video src="/assets/demo/streamlit-demo.mp4" poster="/assets/demo/streamlit-demo-poster.jpg" width="760" autoplay loop muted playsinline preload="metadata" aria-label="F1 StratLab Streamlit app in action"></video>
  <br/>
  <sub>The Streamlit strategy advisor and chat, running against the FastAPI backend.</sub>
</p>

## Page map

| Page | File | Description |
|---|---|---|
| Dashboard | `pages/dashboard.py` | Telemetry charts for selected session/driver |
| Comparison | `pages/comparison.py` | Side-by-side telemetry for two drivers |
| Race Analysis | `pages/race_analysis.py` | Tire, gap, and radio analysis tabs |
| Strategy | `pages/strategy.py` | N25–N31 strategy advisor with agent tabs |
| Chat | `pages/chat.py` | LM Studio chat interface |
| Model Lab | `pages/model_lab.py` | Interactive ML model exploration |

## Directory structure

```
frontend/
  app/
    main.py            -- Streamlit entry point, page routing
    setup_path.py      -- sys.path configuration
    styles.py          -- GLOBAL_CSS constant
    track_data.py      -- Track metadata
  pages/               -- One file per page
  components/
    chatbot/           -- Chat UI components
    common/            -- Shared: driver_colors, chart_styles, loading
    comparison/        -- Driver comparison charts
    dashboard/         -- Dashboard-specific CSS and selectors
    layout/            -- Navbar
    race_analysis/     -- Tire charts, gap charts, radio panel
    strategy/          -- Strategy card, scenario chart, agent tabs
    streamlit_audio_viz/ -- Custom React component for audio visualization
    telemetry/         -- Individual telemetry graphs
    voice/             -- Voice input UI
  services/
    chat_service.py     -- Chat HTTP client
    strategy_service.py -- Strategy endpoints HTTP client
    telemetry_service.py
    voice_api.py
  utils/
    audio_utils.py
    chat_navigation.py
    chat_state.py
    data_loaders.py
    race_processing.py
    race_viz.py
    report_storage.py
    time_formatters.py
  shared/
    img/                -- Static image assets
  config.py             -- BACKEND_URL, API_BASE_URL from env vars
```

## Navigation

The app launches directly into the Dashboard. Page routing uses `st.session_state['current_page']`. The navbar (`components/layout/navbar.py`) sets this value. Pages are rendered conditionally in `main.py`.

## Strategy page

The Strategy page (`pages/strategy.py`) is the primary interface for the N25–N31 agent system:

1. **Selectors**: Year (hardcoded 2025), GP, Driver, Lap range, Analysis lap, Risk tolerance
2. **Run button**: calls `StrategyService.get_recommend()` which hits `/api/v1/strategy/recommend`
3. **Results**:
   - `render_strategy_card()` — recommendation card with action, confidence, reasoning
   - `render_scenario_chart()` — bar chart comparing MC scenario scores
   - `render_agent_tabs()` — tabbed detail view for each sub-agent output

## Chat tool-result rendering

The Chat page (`pages/chat.py`) consumes the MCP tool results streamed by `/api/v1/chat/tool-message-stream` (and the JSON variant at `/api/v1/chat/tool-message`). Each tool result carries a `display_type` hint (see `TOOL_DISPLAY_MAP` in [Backend API](#/backend-api)) that tells the frontend how to render the payload.

Dispatch lives in `components/chatbot/tool_result_renderer.py`:

```python
_RENDERERS = {
    "metrics":        _render_metrics,
    "strategy_card":  _render_strategy_card,
    "table":          _render_table,
    "text":           _render_text,
    "chart":          _render_chart,
}
```

### Inline Plotly charts (Phase 2 MCP telemetry tools)

The four telemetry tools (`get_lap_times`, `get_telemetry`, `compare_drivers`, `get_race_data`) are mapped to `DisplayType.CHART` and render as inline Plotly figures inside the chat bubble. `_render_chart(data, tool_name)` calls `build_figure(tool_name, data)` from `components/chatbot/chart_builders.py`, which dispatches to one of:

| Tool | Builder | Figure |
|---|---|---|
| `get_lap_times` | `build_lap_times_figure` | Lap-time line per driver |
| `get_telemetry` | `build_telemetry_figure` | Speed / throttle / brake traces for one lap |
| `compare_drivers` | `build_compare_drivers_figure` | Fastest-lap side-by-side comparison |
| `get_race_data` | `build_race_data_figure` | Full race overview (positions / gaps) |

Builders are Streamlit-free: they take the raw tool payload, pull per-driver colors via `get_driver_color` from `components/common/driver_colors.py` (see [Driver colors](#/driver-colors)), and apply the shared dark theme through `_apply_base_layout()`. The renderer wraps the figure with `apply_telemetry_chart_styles()` so the chart inherits the purple-outlined chat bubble look, and falls back to `_render_text` if the builder returns `None`.

## Services layer

All HTTP calls go through service classes in `services/`. Each method returns a `(success: bool, data: dict | None, error: str | None)` triple so callers handle results consistently.

```python
class StrategyService:
    @staticmethod
    def get_pace(lap_state) -> Tuple[bool, Optional[Dict], Optional[str]]: ...
    @staticmethod
    def get_tire(lap_state) -> Tuple[bool, Optional[Dict], Optional[str]]: ...
    @staticmethod
    def get_recommend(...) -> Tuple[bool, Optional[Dict], Optional[str]]: ...
```

Timeout is set to 300 seconds because first-call model loading (RoBERTa + SetFit + BERT-large + BGE-M3) is slow.

## Configuration

| Variable | Default | Description |
|---|---|---|
| `BACKEND_URL` | `http://localhost:8000` | FastAPI backend base URL |
| `FRONTEND_URL` | `http://localhost:8501` | Frontend self-reference |

## Appendix — CSS fixes

### Scroll fix on Plotly charts

Streamlit wraps Plotly charts in `div.stElementContainer` elements that can develop unwanted horizontal scrollbars. Located in `src/telemetry/frontend/components/common/chart_styles.py`, `apply_telemetry_chart_styles()` injects CSS that hides the scrollbar on the parent wrapper without clipping chart content:

```css
div.stElementContainer:has(div[data-testid="stPlotlyChart"]) {
    scrollbar-width: none !important;
    -ms-overflow-style: none !important;
}
div.stElementContainer:has(div[data-testid="stPlotlyChart"])::-webkit-scrollbar {
    display: none !important;
}
```

The same function applies the visual treatment to all Plotly charts:

```css
div[data-testid="stPlotlyChart"] {
    outline: 2px solid #a78bfa !important;
    outline-offset: -2px !important;
    border-radius: 12px !important;
    background-color: #181633 !important;
    box-shadow: 0 4px 12px rgba(167, 139, 250, 0.2) !important;
}
```

`outline` is used instead of `border` because outlines do not affect the box model and cannot cause layout shifts or trigger scrollbars.
