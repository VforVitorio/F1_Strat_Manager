# Roadmap

> Full project timeline from the first legacy commit through the planned future. Shipped versions are solid entries; planned milestones carry a "Planned" badge; the side-repo ecosystem is treated separately at the bottom.

<style>
/* ── Roadmap timeline ── */
.rl-wrap {
  margin: 40px 0 0;
  padding: 0;
  list-style: none;
}

/* Vertical spine */
.rl-wrap { position: relative; }
.rl-wrap::before {
  content: "";
  position: absolute;
  left: 11px;
  top: 8px;
  bottom: 8px;
  width: 2px;
  background: linear-gradient(
    to bottom,
    var(--purple-600) 0%,
    rgba(108, 92, 231, 0.4) 65%,
    rgba(108, 92, 231, 0.12) 100%
  );
  border-radius: 2px;
}

.rl-item {
  position: relative;
  padding: 0 0 36px 44px;
  margin: 0;
}
.rl-item:last-child { padding-bottom: 0; }

/* Dot on spine */
.rl-dot {
  position: absolute;
  left: 3px;
  top: 6px;
  width: 18px;
  height: 18px;
  border-radius: 50%;
  border: 2px solid var(--purple-600);
  background: var(--bg-0);
  box-shadow: 0 0 0 3px rgba(108, 92, 231, 0.18);
  display: flex;
  align-items: center;
  justify-content: center;
  flex-shrink: 0;
}
.rl-dot.done { background: var(--purple-600); }
.rl-dot.done::after {
  content: "";
  width: 6px;
  height: 6px;
  border-radius: 50%;
  background: #fff;
}
.rl-dot.planned {
  border-color: var(--purple-400);
  border-style: dashed;
  background: var(--bg-1);
}
.rl-dot.side {
  border-color: rgba(255,255,255,0.2);
  border-style: dashed;
  background: var(--bg-1);
}

/* Card body */
.rl-card {
  background: var(--bg-3);
  border: 1px solid var(--divider);
  border-radius: var(--radius-md);
  padding: 16px 20px;
  transition: border-color 0.15s ease, background 0.15s ease;
}
.rl-card:hover { border-color: rgba(108,92,231,0.35); background: var(--bg-4); }
.rl-card.planned {
  background: rgba(108,92,231,0.04);
  border-color: rgba(108,92,231,0.2);
  border-style: solid;
}
.rl-card.side {
  background: rgba(255,255,255,0.02);
  border-color: rgba(255,255,255,0.08);
}

/* Card header row */
.rl-header {
  display: flex;
  align-items: center;
  flex-wrap: wrap;
  gap: 10px;
  margin-bottom: 6px;
}
.rl-version {
  font-family: var(--font-mono);
  font-size: 13px;
  font-weight: 600;
  color: var(--purple-300);
  background: rgba(108,92,231,0.12);
  border: 1px solid rgba(108,92,231,0.22);
  padding: 2px 9px;
  border-radius: var(--radius-pill);
  white-space: nowrap;
}
.rl-date {
  font-family: var(--font-mono);
  font-size: 12px;
  color: var(--fg-4);
  white-space: nowrap;
}
.rl-badge {
  font-size: 10px;
  font-weight: 700;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  padding: 2px 8px;
  border-radius: var(--radius-pill);
  white-space: nowrap;
}
.rl-badge.done-badge {
  background: rgba(67,255,100,0.12);
  color: var(--success);
  border: 1px solid rgba(67,255,100,0.22);
}
.rl-badge.planned-badge {
  background: rgba(108,92,231,0.14);
  color: var(--purple-300);
  border: 1px solid rgba(108,92,231,0.28);
}
.rl-badge.side-badge {
  background: rgba(255,255,255,0.05);
  color: var(--fg-3);
  border: 1px solid rgba(255,255,255,0.1);
}

/* Title + summary */
.rl-title {
  font-family: var(--font-display);
  font-size: 16px;
  font-weight: 600;
  color: var(--fg-1);
  margin: 0 0 4px;
}
.rl-summary {
  font-size: 14px;
  line-height: 1.6;
  color: var(--fg-2);
  margin: 0;
}

/* Metric pills inside a card */
.rl-metrics {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
  margin-top: 10px;
}
.rl-metric {
  font-family: var(--font-mono);
  font-size: 11px;
  color: var(--fg-3);
  background: rgba(255,255,255,0.04);
  border: 1px solid var(--hairline);
  padding: 2px 8px;
  border-radius: var(--radius-pill);
}

/* Section dividers inside the list */
.rl-section-label {
  font-size: 11px;
  font-weight: 700;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  color: var(--fg-4);
  margin: 36px 0 20px 44px;
}

/* Side-repo grid */
.rl-side-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));
  gap: 14px;
  margin: 8px 0 0;
}
.rl-side-card {
  background: rgba(255,255,255,0.025);
  border: 1px dashed rgba(255,255,255,0.12);
  border-radius: var(--radius-md);
  padding: 14px 16px;
}
.rl-side-card:hover { border-color: rgba(255,255,255,0.22); background: rgba(255,255,255,0.04); }
.rl-repo {
  font-family: var(--font-mono);
  font-size: 12px;
  font-weight: 600;
  color: var(--fg-2);
  margin: 0 0 4px;
}
.rl-repo-desc {
  font-size: 13px;
  line-height: 1.5;
  color: var(--fg-3);
  margin: 0;
}
</style>

## Timeline

<p class="rl-section-label">Legacy phase</p>

<ul class="rl-wrap">

<li class="rl-item">
  <div class="rl-dot done"></div>
  <div class="rl-card">
    <div class="rl-header">
      <span class="rl-version"><span class="sr-only">Version: </span>v0.1 to v0.5</span>
      <span class="rl-date"><span class="sr-only">Date: </span>May 2025</span>
      <span class="rl-badge done-badge"><span class="sr-only">Status: </span>Shipped</span>
    </div>
    <p class="rl-title">Legacy integration and project setup</p>
    <p class="rl-summary">Third-year coursework iteration. Integrated the F1_Telemetry_Manager submodule, established the modular repository layout (<code>src/</code>, <code>notebooks/</code>, <code>data/</code>, <code>legacy/</code>), and verified seven FastAPI endpoint categories. A <code>legacy_version</code> branch preserves this work; active TFG development starts at v0.6.</p>
  </div>
</li>

</ul>

<p class="rl-section-label">Data and models</p>

<ul class="rl-wrap">

<li class="rl-item">
  <div class="rl-dot done"></div>
  <div class="rl-card">
    <div class="rl-header">
      <span class="rl-version"><span class="sr-only">Version: </span>v0.6.0</span>
      <span class="rl-date"><span class="sr-only">Date: </span>2026-02-12</span>
      <span class="rl-badge done-badge"><span class="sr-only">Status: </span>Shipped</span>
    </div>
    <p class="rl-title">Data engineering pipeline</p>
    <p class="rl-summary">End-to-end pipeline from raw FastF1 telemetry to a clean, feature-rich dataset. Circuit clustering (K-Means k=4, 25 circuits), 48-column feature set over ~45,000 clean racing laps, and 2025 saved as a held-out test set. Dataset published to Hugging Face Hub.</p>
    <div class="rl-metrics">
      <span class="rl-metric">4 clusters</span>
      <span class="rl-metric">~45k laps</span>
      <span class="rl-metric">2023-2025</span>
    </div>
  </div>
</li>

<li class="rl-item">
  <div class="rl-dot done"></div>
  <div class="rl-card">
    <div class="rl-header">
      <span class="rl-version"><span class="sr-only">Version: </span>v0.7.0</span>
      <span class="rl-date"><span class="sr-only">Date: </span>2026-03-05</span>
      <span class="rl-badge done-badge"><span class="sr-only">Status: </span>Shipped</span>
    </div>
    <p class="rl-title">ML foundation: lap time and tire degradation</p>
    <p class="rl-summary">First two production ML models. XGBoost delta-lap-time predictor (N06) and a Temporal Convolutional Network for tire degradation with MC Dropout uncertainty (N07-N10, per-compound fine-tuning on SOFT / MEDIUM / HARD).</p>
    <div class="rl-metrics">
      <span class="rl-metric">Pace MAE 0.4104 s</span>
      <span class="rl-metric">Tire MAE 0.708 s</span>
      <span class="rl-metric">MC Dropout N=50</span>
    </div>
  </div>
</li>

<li class="rl-item">
  <div class="rl-dot done"></div>
  <div class="rl-card">
    <div class="rl-header">
      <span class="rl-version"><span class="sr-only">Version: </span>v0.8.0 + v0.8.1</span>
      <span class="rl-date"><span class="sr-only">Date: </span>2026-03-13</span>
      <span class="rl-badge done-badge"><span class="sr-only">Status: </span>Shipped</span>
    </div>
    <p class="rl-title">Extended ML models: overtake, safety car, pit stop, undercut</p>
    <p class="rl-summary">Four additional predictors. LightGBM overtake classifier on 28,494 labeled pairs (N11-N12), soft safety-car prior (N13-N14), HistGBT quantile regression for pit duration (N15), and LightGBM undercut success scorer (N16). The Causal TCN alternative (N12B) archived as a documented negative result.</p>
    <div class="rl-metrics">
      <span class="rl-metric">Overtake AUC-PR 0.5491</span>
      <span class="rl-metric">SC lift 1.67x</span>
      <span class="rl-metric">Pit MAE 0.487 s</span>
      <span class="rl-metric">Undercut AUC-ROC 0.7708</span>
    </div>
  </div>
</li>

<li class="rl-item">
  <div class="rl-dot done"></div>
  <div class="rl-card">
    <div class="rl-header">
      <span class="rl-version"><span class="sr-only">Version: </span>v0.8.2</span>
      <span class="rl-date"><span class="sr-only">Date: </span>2026-03-22</span>
      <span class="rl-badge done-badge"><span class="sr-only">Status: </span>Shipped</span>
    </div>
    <p class="rl-title">NLP radio processing pipeline</p>
    <p class="rl-summary">Full team-radio NLP stack. Whisper ASR (N18), RoBERTa sentiment (N20, 0.84 accuracy), SetFit intent classification (N21, 5 classes), BERT-large NER for F1 entities (N22), deterministic RCM parser (N23). Unified inference entry point in N24 at GPU P95 latency of 59.4 ms.</p>
    <div class="rl-metrics">
      <span class="rl-metric">Sentiment 0.84</span>
      <span class="rl-metric">P95 59.4 ms GPU</span>
    </div>
  </div>
</li>

</ul>

<p class="rl-section-label">Agents and distribution</p>

<ul class="rl-wrap">

<li class="rl-item">
  <div class="rl-dot done"></div>
  <div class="rl-card">
    <div class="rl-header">
      <span class="rl-version"><span class="sr-only">Version: </span>v0.9.0</span>
      <span class="rl-date"><span class="sr-only">Date: </span>2026-03-17</span>
      <span class="rl-badge done-badge"><span class="sr-only">Status: </span>Shipped</span>
    </div>
    <p class="rl-title">src/ extraction, CLI simulation, radio corpus</p>
    <p class="rl-summary">Seven agent entry points extracted to importable <code>src/agents/</code> modules. Headless CLI simulation (<code>f1-sim</code>) with Rich Live rendering and no-LLM mode. OpenF1 radio corpus pipeline (529 MP3s, 48 parquets) with Whisper JSON cache. Lazy Hugging Face data download on first run.</p>
  </div>
</li>

<li class="rl-item">
  <div class="rl-dot done"></div>
  <div class="rl-card">
    <div class="rl-header">
      <span class="rl-version"><span class="sr-only">Version: </span>v0.10.0</span>
      <span class="rl-date"><span class="sr-only">Date: </span>2026-03-22</span>
      <span class="rl-badge done-badge"><span class="sr-only">Status: </span>Shipped</span>
    </div>
    <p class="rl-title">Multi-agent system: N25 to N31</p>
    <p class="rl-summary">Six LangGraph ReAct sub-agents coordinated by the N31 Strategy Orchestrator's three-layer pipeline: MoE-style dynamic routing, Monte Carlo simulation over 500 samples ranking four strategy candidates by risk-adjusted expected outcome, and LLM synthesis producing a 14-field <code>StrategyRecommendation</code>. Bahrain 2025 end-to-end demo.</p>
    <div class="rl-metrics">
      <span class="rl-metric">6 agents + 1 orchestrator</span>
      <span class="rl-metric">MC N=500</span>
      <span class="rl-metric">MoE routing</span>
    </div>
  </div>
</li>

<li class="rl-item">
  <div class="rl-dot done"></div>
  <div class="rl-card">
    <div class="rl-header">
      <span class="rl-version"><span class="sr-only">Version: </span>v0.11.0</span>
      <span class="rl-date"><span class="sr-only">Date: </span>2026-03-30</span>
      <span class="rl-badge done-badge"><span class="sr-only">Status: </span>Shipped</span>
    </div>
    <p class="rl-title">RAG system over FIA regulations</p>
    <p class="rl-summary">Retrieval-augmented generation grounding strategic decisions in FIA Sporting Regulations 2023-2025. BGE-M3 embeddings (1024-dim), Qdrant local vector store, 2,279 indexed chunks. Exposed as the <code>query_rag_tool</code> LangChain tool imported by the orchestrator.</p>
    <div class="rl-metrics">
      <span class="rl-metric">2,279 chunks</span>
      <span class="rl-metric">BGE-M3 1024d</span>
      <span class="rl-metric">Scores 0.62-0.76</span>
    </div>
  </div>
</li>

<li class="rl-item">
  <div class="rl-dot done"></div>
  <div class="rl-card">
    <div class="rl-header">
      <span class="rl-version"><span class="sr-only">Version: </span>v0.1.1</span>
      <span class="rl-date"><span class="sr-only">Date: </span>2026-04-09</span>
      <span class="rl-badge done-badge"><span class="sr-only">Status: </span>Shipped</span>
    </div>
    <p class="rl-title">R1: CLI wheel release</p>
    <p class="rl-summary">First distribution artifact. Wheel <code>f1_strat_manager-0.1.1-py3-none-any.whl</code> on GitHub Releases. Both <code>f1-strat</code> and <code>f1-sim</code> entry points verified. Installable via <code>uv tool install git+&lt;repo&gt;</code>.</p>
  </div>
</li>

<li class="rl-item">
  <div class="rl-dot done"></div>
  <div class="rl-card">
    <div class="rl-header">
      <span class="rl-version"><span class="sr-only">Version: </span>v0.12.0</span>
      <span class="rl-date"><span class="sr-only">Date: </span>2026-04-15</span>
      <span class="rl-badge done-badge"><span class="sr-only">Status: </span>Shipped</span>
    </div>
    <p class="rl-title">Interfaces and distribution: FastAPI, Streamlit, Arcade</p>
    <p class="rl-summary">Wired the multi-agent system into the FastAPI backend and FastMCP. Streamlit pages for race analysis and live strategy cards with inline Plotly charts. Arcade MVP: three windows from one command (pyglet 2D replay, PySide6 strategy dashboard, live telemetry grid). Voice chat rewritten with Whisper ASR and edge-tts.</p>
    <div class="rl-metrics">
      <span class="rl-metric">3 surfaces</span>
      <span class="rl-metric">7 MCP tools</span>
      <span class="rl-metric">TCP 10 Hz stream</span>
    </div>
  </div>
</li>

</ul>

<p class="rl-section-label">Stable release</p>

<ul class="rl-wrap">

<li class="rl-item">
  <div class="rl-dot done"></div>
  <div class="rl-card">
    <div class="rl-header">
      <span class="rl-version"><span class="sr-only">Version: </span>v1.0.0</span>
      <span class="rl-date"><span class="sr-only">Date: </span>2026-04-20</span>
      <span class="rl-badge done-badge"><span class="sr-only">Status: </span>Shipped</span>
    </div>
    <p class="rl-title">Final release: code freeze and three-surface distribution</p>
    <p class="rl-summary">First stable release. Consolidates the Arcade MVP, the full seven-model ML stack, and N25-N31 with FIA RAG into a single tagged production release. Three install paths: CLI wheel, <code>f1-arcade</code> via <code>uv tool install</code>, and Streamlit + FastAPI Docker Compose.</p>
  </div>
</li>

<li class="rl-item">
  <div class="rl-dot done"></div>
  <div class="rl-card">
    <div class="rl-header">
      <span class="rl-version"><span class="sr-only">Version: </span>v1.1.0</span>
      <span class="rl-date"><span class="sr-only">Date: </span>2026-05-11</span>
      <span class="rl-badge done-badge"><span class="sr-only">Status: </span>Shipped</span>
    </div>
    <p class="rl-title">Benchmark suite and thesis evaluation</p>
    <p class="rl-summary">Four standalone benchmark scripts covering pace, Whisper latency, six sub-agent latency, and the NLP pipeline on CPU and GPU. N33 precision-recall sweeps for overtake, safety car, and undercut models. N30B RAG benchmark comparing BGE-M3 variants over 15 ground-truth queries. Full English localization of strategy notebooks.</p>
  </div>
</li>

<li class="rl-item">
  <div class="rl-dot done"></div>
  <div class="rl-card">
    <div class="rl-header">
      <span class="rl-version"><span class="sr-only">Version: </span>v1.2.0 to v1.3.1</span>
      <span class="rl-date"><span class="sr-only">Date: </span>2026-05-12</span>
      <span class="rl-badge done-badge"><span class="sr-only">Status: </span>Shipped</span>
    </div>
    <p class="rl-title">Documentation site launch and custom domain</p>
    <p class="rl-summary">React docs site (plain <code>React.createElement</code> scripts, no JSX, no Babel, no build step) published under <code>docs.f1stratlab.com</code>. Full architecture pages, agent API reference, arcade quick-start, CI/CD narrative, and graph-based page discovery. Five drawio architecture diagrams. Brand theme aligned with the F1 StratLab purple palette.</p>
  </div>
</li>

<li class="rl-item">
  <div class="rl-dot done"></div>
  <div class="rl-card">
    <div class="rl-header">
      <span class="rl-version"><span class="sr-only">Version: </span>v2.0.0</span>
      <span class="rl-date"><span class="sr-only">Date: </span>2026-07-21</span>
      <span class="rl-badge done-badge"><span class="sr-only">Status: </span>Shipped</span>
    </div>
    <p class="rl-title">Modern frontend: the post-race UI moves to a React web app</p>
    <p class="rl-summary">The Streamlit post-race interface is fully replaced by a React web app (React 19, Vite, TypeScript, Tailwind, ECharts, TanStack Router/Query, Zustand) across six surfaces: dashboard, comparison (60fps replay), model lab, strategy, race analysis, and a streaming chat that renders each tool result inline. The FastAPI backend, the six agents, the orchestrator and the models are unchanged: a presentation-layer swap, not a rebuild.</p>
  </div>
</li>

</ul>

<p class="rl-section-label">Planned milestones</p>

<ul class="rl-wrap">

<li class="rl-item">
  <div class="rl-dot planned"></div>
  <div class="rl-card planned">
    <div class="rl-header">
      <span class="rl-version"><span class="sr-only">Version: </span>v2.5.0</span>
      <span class="rl-badge planned-badge"><span class="sr-only">Status: </span>Planned</span>
    </div>
    <p class="rl-title">Arcade, modernized: a web-native trackside frontend</p>
    <p class="rl-summary">Bring the web app's modern frontend to part of the live Arcade experience, running alongside the PySide6 / pyglet 2D replay rather than replacing it. The strategy and telemetry surfaces move to a web-native view; the <code>lap_state</code> contract and the agents stay unchanged.</p>
  </div>
</li>

<li class="rl-item">
  <div class="rl-dot planned"></div>
  <div class="rl-card planned">
    <div class="rl-header">
      <span class="rl-version"><span class="sr-only">Version: </span>v2.8.0</span>
      <span class="rl-badge planned-badge"><span class="sr-only">Status: </span>Planned</span>
    </div>
    <p class="rl-title">Rival Agent: anticipate each rival's next move</p>
    <p class="rl-summary">A new, additive LangGraph node that predicts each nearby rival's next strategic move (pit window, compound, undercut / overcut) and feeds it to the orchestrator. Recommendations move from reactive to anticipatory. The six existing agents are untouched. Validated by ablation against real 2024-2025 pit-stop outcomes.</p>
  </div>
</li>

<li class="rl-item">
  <div class="rl-dot planned"></div>
  <div class="rl-card planned">
    <div class="rl-header">
      <span class="rl-version"><span class="sr-only">Version: </span>v3.0.0</span>
      <span class="rl-badge planned-badge"><span class="sr-only">Status: </span>Planned</span>
    </div>
    <p class="rl-title">Live race inference and 2026 regulation adaptation</p>
    <p class="rl-summary">Real-time OpenF1 WebSocket ingestion. The <code>lap_state</code> contract is unchanged, so agents and the orchestrator do not change. Adaptation to the 2026 technical and sporting regulation: re-cluster circuits, re-label compounds, add drift monitoring.</p>
  </div>
</li>

</ul>

---

## Side repos

These four projects ship independently of the core release train. Each is a dedicated public repository under the `f1stratlab` GitHub organisation. They share the F1 StratLab domain and Hugging Face org but have their own versioning and release cadence.

<div class="rl-side-grid">

<div class="rl-side-card">
  <p class="rl-repo">gridmind</p>
  <p class="rl-repo-desc">Unsloth LoRA fine-tune of a Gemma-family LLM on an F1 text corpus for F1-specific strategy reasoning.</p>
</div>

<div class="rl-side-card">
  <p class="rl-repo">radiogate</p>
  <p class="rl-repo-desc">Large-scale F1 team-radio NLP corpus with auto-labelling and a novel deception and bluffing signal detector.</p>
</div>

<div class="rl-side-card">
  <p class="rl-repo">pitlab</p>
  <p class="rl-repo-desc">Button-driven MLOps studio: download, merge, inspect, retrain. Clustering-aware and progressive per-GP.</p>
</div>

<div class="rl-side-card">
  <p class="rl-repo">box-bot</p>
  <p class="rl-repo-desc">Automated X account narrating the orchestrator live during a Grand Prix.</p>
</div>

</div>
