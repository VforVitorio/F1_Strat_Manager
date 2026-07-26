# draw.io diagram sources

Editable `.drawio` sources for the project's architecture diagrams. Open them with [diagrams.net](https://app.diagrams.net/) or the VS Code extension.

## Status, audited 2026-07-26

These were last edited on **2026-05-13**. Two major releases have landed since: v2.0.0 retired the Streamlit frontend and replaced it with a React web app, and v2.1.0 rewrote the Monte Carlo decision layer to score in projected track position. Six of the twelve still draw surfaces that no longer exist.

| Diagram | Retired-surface references | Read it as |
|---|---|---|
| `backend_api.drawio` | 12 | **stale** |
| `chat_mcp_flow.drawio` | 13 | **stale** |
| `frontend_pages.drawio` | 7 | **stale**, it draws the Streamlit page tree |
| `system_architecture.drawio` | 4 | **stale** |
| `data_pipeline.drawio` | 1 | **check**, and note the FastF1 cache is one directory now, not two |
| `docker_deployment.drawio` | 1 | **check** |
| `arcade_3window_architecture.drawio` | 0 | current |
| `multi_agent_architecture.drawio` | 0 | current |
| `multi_agent_flow.drawio` | 0 | current, but predates the projection redesign |
| `strategy_pipeline_flow.drawio` | 0 | predates the shared inference engine |
| `subprocess_launch_sequence.drawio` | 0 | current |
| `tcp_broadcast_dataflow.drawio` | 0 | current |

The count is a grep for Streamlit and voice, so it is a smell rather than a verdict. A zero does not certify a diagram as accurate; it only means it does not name a surface that was deleted.

## These are not what the docs site renders

The site at docs.f1stratlab.com renders **Mermaid**, written directly into `docs/pages/*.md` as fenced code blocks. It has no draw.io renderer, so a diagram here reaches a reader only if someone exports it to SVG and embeds the image.

That split is deliberate. Mermaid is text: it diffs in review, it cannot drift out of sync with the page it sits in, and it renders live. draw.io is where a diagram goes when it needs a layout Mermaid cannot express, or when it is meant to be edited visually.

**If you fix one of the stale files above, consider whether the diagram belongs in `docs/pages/` as Mermaid instead.** Several of them describe things the docs site currently has no diagram for at all.

## Also here

`agents/` holds one diagram per sub-agent (N25 to N30) plus the `StrategyRecommendation` schema.

`site/diagrams/` at the repo root holds byte-identical copies of these files. It is **not tracked by git**, being the build output of an older docs site. Do not edit it and do not treat it as a second source.
