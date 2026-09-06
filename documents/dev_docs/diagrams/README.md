# draw.io diagram sources

Editable `.drawio` sources for the project's architecture diagrams. Open them with [diagrams.net](https://app.diagrams.net/) or the VS Code extension.

## Status, refreshed 2026-07-26

All of these were last edited on 2026-05-13, before two releases landed: v2.0.0 retired the Streamlit frontend and the voice surface, and v2.1.0 rewrote the Monte Carlo decision layer. They have now been brought back in line.

| Diagram | State |
|---|---|
| `system_architecture` | updated 2026-08-26 (#1090): Surface 2's windows 2 and 3 said `Dashboard (Qt)` and `Telemetry (Qt)` over a description reading `pyglet Arcade + PySide6 dashboard`. They are the two PITWALL webview windows in one process. Surface 3 is the React web app, not Streamlit |
| `backend_api` | updated: the `/voice` router and its two endpoints removed |
| `chat_mcp_flow` | updated: voice input path removed, the chat UI is the React tab |
| `docker_deployment` | updated: `f1_webapp` on nginx, host 8501 to container 80 |
| `webapp_structure` | **new**: the React app's feature folders and how they reach the backend |
| `frontend_pages_streamlit_legacy` | **renamed and marked retired**: it is the Streamlit page tree, kept as a record of a surface that no longer exists |
| `arcade_3window_architecture_qt_legacy` | **renamed and marked retired**: it is the PySide6 pair PITWALL replaced in sprint 7. Not relabelled into the new topology, because that is not a relabel - the DATA window is not shaped like the Qt telemetry window. The live picture is the Mermaid graph in `docs/pages/multi-agent.md` |
| `multi_agent_architecture` | updated 2026-08-02: N25's box no longer says "LangGraph ReAct", because pace's scaffold was formally retired in #781, N25 is now shown as a direct XGBoost call, same as the sub-agents header line |
| `multi_agent_flow` | updated 2026-08-26: its routing, draw count and synthesis model match the orchestrator, but its Layer 2 box did not. The four candidates are `STAY_OUT` / `PIT_NOW` / `UNDERCUT` / `OVERCUT` (`strategy_orchestrator.py:272`), not one STAY OUT and three PIT NOW compound variants, and the score is `payoff(project_positions(rivals))` per draw (`:1308-1323`), not a sample from the sub-agent distributions |
| `strategy_pipeline_flow` | updated 2026-08-26: its Layer 3 said `gpt-4.1-mini`, which is what the SUB-AGENTS use, while the synthesis it labels is the orchestrator's own `gpt-5.4-mini` (`_shared_defaults.py`, `orchestrator_model()`); and its pace MAE of 0.392 s is the notebook-era value the metrics registry marks `canonical: false`, superseded by 0.4104 s |
| `subprocess_launch_sequence` | updated 2026-08-26 (#1090): the actor columns already said PITWALL while four messages still described the Qt pair. Step 6b spawned `src.arcade.telemetry`, a module that does not exist, and step 14 sent a second TCP tick to it; `app.py:501` opens ONE subprocess and `src/pitwall/__main__.py` builds ONE `ArcadeStreamClient` for both windows, so both arrows now start at the PITWALL column. Step 7 is `PitwallHost(ArcadeStreamClient)`, and step 15's "tabs" went with the reasoning panel in #1020 |
| `tcp_broadcast_dataflow` | updated 2026-08-26 (#1090): the live subscriber box read `QMainWindow` / `QThread` / `pyqtSignal` beside a second box correctly marked RETIRED. It is one pywebview process reading one socket for both windows. The 2x2 grid is ECharts |
| `data_pipeline` | current on the surfaces; note the FastF1 cache is one directory now, not two |
| `agents/` | one per sub-agent (N25 to N30) plus the `StrategyRecommendation` schema; `N25_pace_agent` updated 2026-08-02 to drop the ReAct-tool boxes it never actually used (#781) |

Verified 2026-08-26: all 20 files parse as XML, and no label, tab name or XML comment carries an em dash.

The three that still named the retired Qt surface were redrawn in #1090, and the rule they broke is
now checked rather than asserted: `tests/surfaces/test_diagrams_no_retired_surface.py` parses the
`value=` attributes of every diagram outside the two `*_legacy.drawio` files and fails on
`PySide6`, `QApplication`, `QMainWindow`, `QThread`, `pyqtSignal` or `src.arcade.telemetry`. A box
whose own label carries the word RETIRED is exempt, because drawing the dead surface beside the
live one is how `tcp_broadcast_dataflow` shows what was replaced. It also fails a diagram that
yields no labels at all, which is what a compressed draw.io save looks like from the outside.

The check is the parse half only. Whether a diagram means what the code does is still a reading
against the source, which is how the four defects above were found and how the drift in
`multi_agent_flow` was found running the opposite way.

**How this audit is made.** Labels are parsed out of the `value=` attributes and the tab names, never grepped from the raw XML. A grep counts matches inside style attributes and colour names, which both over-reports (a file whose only hit is a colour) and under-reports (`docker_deployment`, whose defect was a container named `f1_telemetry_frontend`, containing neither search word). Each figure is then checked against the code it describes rather than against a sibling diagram, because the drift runs in both directions.

## These are not what the docs site renders

The site at docs.f1stratlab.com renders **Mermaid**, written directly into `docs/pages/*.md` as fenced code blocks. It has no draw.io renderer, so a diagram here reaches a reader only if someone exports it to SVG and embeds the image.

That split is deliberate. Mermaid is text: it diffs in review, it cannot drift out of sync with the page it sits in, and it renders live. draw.io is where a diagram goes when it needs a layout Mermaid cannot express, or when it is meant to be edited visually.

If you fix one of these, consider whether the diagram also belongs in `docs/pages/` as Mermaid. Several describe things the docs site has no diagram for.

## Also here

`site/diagrams/` at the repo root holds byte-identical copies of the pre-2026-07-26 versions. It is **not tracked by git**, being the build output of an older docs site. Do not edit it and do not treat it as a second source.
