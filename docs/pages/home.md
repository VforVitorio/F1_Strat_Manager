# Welcome to F1 StratLab

> Open-source multi-agent system that fuses seven machine-learning models, six LangGraph sub-agents and one strategy orchestrator into a single Formula 1 strategy recommender. Shipped under Apache-2.0.

This is the canonical technical reference for the F1 StratLab codebase. It is hand-curated and complements two sibling resources: the public landing at [f1stratlab.com](https://f1stratlab.com/) tells the project story for non-technical visitors, and the auto-generated [DeepWiki](https://deepwiki.com/VforVitorio/F1-StratLab) gives a notebook-per-notebook tour of the source tree. The pages here focus on the narratives those two cannot: **how the layers connect, why the contracts look the way they do, and what to do when something breaks**.

## The system at a glance

```mermaid
graph TD
    subgraph Sources["Race data"]
        FF[FastF1 sessions]
        OF1[OpenF1 radios]
        RCM[Race Control messages]
    end

    subgraph Engine["Replay engine"]
        RRE[RaceReplayEngine]
        RSM[RaceStateManager]
    end

    subgraph Models["ML core · 7 models"]
        M1[Pace XGBoost · N06]
        M2[Tire TCN · N07-N10]
        M3[Overtake LightGBM · N12]
        M4[Safety-car LightGBM · N14]
        M5[Pit duration HistGBT · N15]
        M6[Undercut LightGBM · N16]
        M7[Circuit clusters · N03]
    end

    subgraph Agents["LangGraph sub-agents · 6"]
        A1[Pace · N25]
        A2[Tire · N26]
        A3[Race Situation · N27]
        A4[Pit Strategy · N28]
        A5[Radio · N29]
        A6[RAG]
    end

    ORCH[N31 Strategy Orchestrator]

    subgraph Surfaces["Operator surfaces"]
        CLI[Headless CLI]
        ARC[Arcade dashboard]
        STR[React web app]
    end

    Sources --> RRE
    RRE --> RSM
    RSM --> ORCH
    Models -.consumed by.-> Agents
    Agents -->|structured outputs| ORCH
    ORCH -->|recommendation| CLI
    ORCH -->|recommendation| ARC
    ORCH -->|recommendation| STR
```

Three layers carry the system from raw telemetry to a strategy call: a machine-learning core, a multi-agent reasoning layer, and three operator surfaces. Each layer lives behind a documented contract, so any one can be swapped without disturbing the others.

## What lives where

The narratives on this site stop at the contract level. For per-file deep-dives, every function in `src/agents/`, every notebook from N06 to N34, every helper in `src/arcade/`, jump to the [F1 StratLab DeepWiki](https://deepwiki.com/VforVitorio/F1-StratLab). It is regenerated on every push to `main`.

## Project status

| Component | Version | Status |
|---|---|---|
| Multi-agent orchestrator (N31) | v1.0.0 | shipped |
| Arcade three-window MVP | v1.0.0 | shipped |
| Benchmark suite (Chapter 5) | v1.1.0 | shipped |
| Release automation (`release-please`) | v1.1.0 | shipped |
| Current release | v__DOCS_VERSION__ | shipped |

F1 StratLab was defended as a Final Degree Project in June 2026, graded 10/10 with Distinction (Matricula de Honor) and unanimously recommended by the tribunal for publication as a research article. Development continues past the defence: the React front-end migration is complete (v2.0.0, the React web app replaced Streamlit as the post-race UI) alongside ongoing correctness hardening across the multi-agent pipeline. See the [project changelog](https://github.com/VforVitorio/F1-StratLab/blob/main/CHANGELOG.md) for the full history.
