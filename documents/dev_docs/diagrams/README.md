# draw.io diagram sources

Editable `.drawio` sources for the project's architecture diagrams. Open them with [diagrams.net](https://app.diagrams.net/) or the VS Code extension.

## Status, refreshed 2026-07-26

All of these were last edited on 2026-05-13, before two releases landed: v2.0.0 retired the Streamlit frontend and the voice surface, and v2.1.0 rewrote the Monte Carlo decision layer. They have now been brought back in line.

| Diagram | State |
|---|---|
| `system_architecture` | updated: Surface 3 is the React web app, not Streamlit |
| `backend_api` | updated: the `/voice` router and its two endpoints removed |
| `chat_mcp_flow` | updated: voice input path removed, the chat UI is the React tab |
| `docker_deployment` | updated: `f1_webapp` on nginx, host 8501 to container 80 |
| `webapp_structure` | **new**: the React app's feature folders and how they reach the backend |
| `frontend_pages_streamlit_legacy` | **renamed and marked retired**: it is the Streamlit page tree, kept as a record of a surface that no longer exists |
| `arcade_3window_architecture_qt_legacy` | **renamed and marked retired**: it is the PySide6 pair PITWALL replaced in sprint 7. Not relabelled into the new topology, because that is not a relabel - the DATA window is not shaped like the Qt telemetry window. The live picture is the Mermaid graph in `docs/pages/multi-agent.md` |
| `multi_agent_architecture` | updated 2026-08-02: N25's box no longer says "LangGraph ReAct", because pace's scaffold was formally retired in #781, N25 is now shown as a direct XGBoost call, same as the sub-agents header line |
| `multi_agent_flow` | current: its four checkable claims (500 Monte Carlo draws, the 0.30 safety-car routing threshold, `ThreadPoolExecutor` for the parallel layer, `gpt-5.4-mini` for the synthesis) each match the orchestrator source |
| `strategy_pipeline_flow` | updated 2026-08-26: its Layer 3 said `gpt-4.1-mini`, which is what the SUB-AGENTS use, while the synthesis it labels is the orchestrator's own `gpt-5.4-mini` (`strategy_orchestrator.py:139`); and its pace MAE of 0.392 s is the notebook-era value the metrics registry marks `canonical: false`, superseded by 0.4104 s |
| `subprocess_launch_sequence` | updated sprint 7: the sequence is unchanged, the follower it launches is `python -m src.pitwall` and the two windows share one reader |
| `tcp_broadcast_dataflow` | updated sprint 7: the subscriber is PITWALL, and the 2x2 grid is ECharts |
| `data_pipeline` | current on the surfaces; note the FastF1 cache is one directory now, not two |
| `agents/` | one per sub-agent (N25 to N30) plus the `StrategyRecommendation` schema; `N25_pace_agent` updated 2026-08-02 to drop the ReAct-tool boxes it never actually used (#781) |

Verified 2026-08-26: all 20 files parse as XML, no label outside the two marked legacy names a retired surface, and no label, tab name or XML comment carries an em dash.

**How this audit is made.** Labels are parsed out of the `value=` attributes and the tab names, never grepped from the raw XML. A grep counts matches inside style attributes and colour names, which both over-reports (a file whose only hit is a colour) and under-reports (`docker_deployment`, whose defect was a container named `f1_telemetry_frontend`, containing neither search word). Each figure is then checked against the code it describes rather than against a sibling diagram, because the drift runs in both directions.

## These are not what the docs site renders

The site at docs.f1stratlab.com renders **Mermaid**, written directly into `docs/pages/*.md` as fenced code blocks. It has no draw.io renderer, so a diagram here reaches a reader only if someone exports it to SVG and embeds the image.

That split is deliberate. Mermaid is text: it diffs in review, it cannot drift out of sync with the page it sits in, and it renders live. draw.io is where a diagram goes when it needs a layout Mermaid cannot express, or when it is meant to be edited visually.

If you fix one of these, consider whether the diagram also belongs in `docs/pages/` as Mermaid. Several describe things the docs site has no diagram for.

## Also here

`site/diagrams/` at the repo root holds byte-identical copies of the pre-2026-07-26 versions. It is **not tracked by git**, being the build output of an older docs site. Do not edit it and do not treat it as a second source.
