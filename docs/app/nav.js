// =========================================================
// Pages config — single source of truth for nav, search, graph
// =========================================================

window.PAGES = [
  // ------- WELCOME -------
  {
    slug: "home",
    title: "Welcome",
    section: "Welcome",
    file: "pages/home.md",
    icon: "home",
    custom: "home",
    description: "Open-source multi-agent system for real-time F1 race strategy.",
    eyebrow: "F1 StratLab · Documentation",
    tags: ["overview"],
  },
  {
    slug: "getting-started",
    title: "Getting started",
    section: "Welcome",
    file: "pages/getting-started.md",
    description: "Three ways to install — wheel, repo clone, Docker.",
    eyebrow: "Install",
    tags: ["install", "cli", "uv"],
  },
  {
    slug: "meet-the-author",
    title: "Meet the author",
    section: "Welcome",
    file: "pages/meet-the-author.md",
    description: "Who built F1 StratLab and how to reach them.",
    eyebrow: "About",
    tags: ["overview"],
  },

  // ------- ARCHITECTURE -------
  {
    slug: "architecture",
    title: "How it is wired",
    section: "Architecture",
    file: "pages/architecture.md",
    description: "End-to-end tour of the codebase, layer by layer.",
    eyebrow: "Tour",
    tags: ["overview", "langgraph", "agents", "pipeline"],
  },
  {
    slug: "multi-agent",
    title: "Multi-agent system",
    section: "Architecture",
    file: "pages/multi-agent.md",
    description: "N25–N31 agents, MoE routing, MC simulation, LLM synthesis.",
    eyebrow: "N25–N31",
    tags: ["agents", "langgraph", "orchestrator", "ml", "monte-carlo", "rag", "xgboost", "lightgbm", "pydantic"],
  },
  {
    slug: "simulation",
    title: "Race replay engine",
    section: "Architecture",
    file: "pages/simulation.md",
    description: "RaceReplayEngine, RaceStateManager, the lap_state schema.",
    eyebrow: "src/simulation",
    tags: ["data", "fastf1", "telemetry"],
  },
  {
    slug: "agents-api",
    title: "Agents API reference",
    section: "Architecture",
    file: "pages/agents-api.md",
    description: "Entry points, output schemas, request/response models.",
    eyebrow: "Reference",
    tags: ["agents", "api", "langgraph", "ml", "rag", "xgboost", "lightgbm", "pydantic"],
  },

  // ------- SURFACES -------
  {
    slug: "backend-api",
    title: "Backend API",
    section: "Surfaces",
    file: "pages/backend-api.md",
    description: "FastAPI routers, telemetry, chat, voice, strategy.",
    eyebrow: "FastAPI",
    tags: ["api", "fastapi", "telemetry", "chat", "voice", "mcp", "rag", "pydantic"],
  },
  {
    slug: "streamlit",
    title: "Streamlit frontend",
    section: "Surfaces",
    file: "pages/streamlit.md",
    description: "Multi-page Streamlit app: dashboard, strategy, chat.",
    eyebrow: "Web app",
    tags: ["frontend", "ui", "telemetry", "chat", "voice", "mcp"],
  },
  {
    slug: "driver-colors",
    title: "Driver colors",
    section: "Surfaces",
    file: "pages/driver-colors.md",
    description: "Year-aware color palette covering 2023–2025 seasons.",
    eyebrow: "Palette",
    tags: ["frontend", "data"],
  },

  // ------- ARCADE -------
  {
    slug: "arcade-quick-start",
    title: "Arcade quick start",
    section: "Arcade",
    file: "pages/arcade-quick-start.md",
    description: "One-command launch of the three-window arcade MVP.",
    eyebrow: "Quick start",
    tags: ["arcade", "pyside6", "install"],
  },
  {
    slug: "arcade-dashboard",
    title: "Dashboard architecture",
    section: "Arcade",
    file: "pages/arcade-dashboard.md",
    description: "PySide6 package layout, wire protocol, thread model.",
    eyebrow: "PySide6",
    tags: ["arcade", "pyside6", "ui", "telemetry", "threading"],
  },
  {
    slug: "arcade-strategy-pipeline",
    title: "Strategy pipeline",
    section: "Arcade",
    file: "pages/arcade-strategy-pipeline.md",
    description: "Why the arcade duplicates the N31 orchestrator body.",
    eyebrow: "Internals",
    tags: ["arcade", "orchestrator", "agents", "pipeline", "monte-carlo", "threading"],
  },

  // ------- OPERATIONS -------
  {
    slug: "setup",
    title: "Setup & deployment",
    section: "Operations",
    file: "pages/setup.md",
    description: "Docker, local dev, environment variables.",
    eyebrow: "Deploy",
    tags: ["install", "docker", "uv"],
  },
  {
    slug: "thesis",
    title: "Thesis results",
    section: "Operations",
    file: "pages/thesis.md",
    description: "Numeric headline metrics for chapter 5 of the TFG.",
    eyebrow: "Chapter 5",
    tags: ["ml", "evaluation"],
  },

  // ------- DEVELOPMENT -------
  {
    slug: "development",
    title: "Development overview",
    section: "Development",
    file: "pages/development.md",
    description: "Workflows, conventional commits, branch naming.",
    eyebrow: "Overview",
    tags: ["dev", "git", "release"],
  },
  {
    slug: "docs-maintenance",
    title: "Docs maintenance",
    section: "Development",
    file: "pages/docs-maintenance.md",
    description: "How this React docs site is built and deployed.",
    eyebrow: "React",
    tags: ["dev", "github-actions"],
  },
  {
    slug: "ci-cd",
    title: "CI/CD pipeline",
    section: "Development",
    file: "pages/ci-cd.md",
    description: "GitHub Actions, release-please automation, Dependabot.",
    eyebrow: "Actions",
    tags: ["dev", "git", "release", "github-actions", "uv"],
  },
  {
    slug: "tags",
    title: "Tags index",
    section: "Development",
    file: "pages/tags.md",
    description: "Every tag declared by the docs site, grouped by theme.",
    eyebrow: "Index",
    tags: ["overview"],
  },
  {
    slug: "roadmap",
    title: "Roadmap",
    section: "Development",
    file: "pages/roadmap.md",
    description: "Full project timeline: every shipped version plus planned milestones and the side-repo ecosystem.",
    eyebrow: "Timeline",
    tags: ["overview", "release"],
  },
  {
    slug: "changelog",
    title: "Changelog",
    section: "Development",
    file: "pages/changelog.md",
    description: "Mirrored release history from the repo root CHANGELOG.md.",
    eyebrow: "Releases",
    tags: ["release"],
  },
];

// Tag colour map (used as a soft white-ish neutral with subtle hue hints).
// Keep neutral so they read as secondary nodes in the graph.
window.TAG_COLOR = "#cfc6e6";
window.TAG_LABELS = {
  agents: "agents",
  langgraph: "langgraph",
  orchestrator: "orchestrator",
  arcade: "arcade",
  pyside6: "pyside6",
  api: "api",
  fastapi: "fastapi",
  frontend: "frontend",
  ui: "ui",
  data: "data",
  fastf1: "fastf1",
  ml: "ml",
  evaluation: "evaluation",
  install: "install",
  docker: "docker",
  cli: "cli",
  dev: "dev",
  git: "git",
  overview: "overview",
  telemetry: "telemetry",
  rag: "rag",
  mcp: "mcp",
  chat: "chat",
  voice: "voice",
  pydantic: "pydantic",
  pipeline: "pipeline",
  threading: "threading",
  "monte-carlo": "monte-carlo",
  xgboost: "xgboost",
  lightgbm: "lightgbm",
  uv: "uv",
  release: "release",
  "github-actions": "github-actions",
};

// Section order
window.SECTIONS = [
  "Welcome",
  "Architecture",
  "Surfaces",
  "Arcade",
  "Operations",
  "Development",
];

// Section color (for graph nodes)
window.SECTION_COLORS = {
  Welcome:      "#a29bfe",
  Architecture: "#6c5ce7",
  Surfaces:     "#3385ff",
  Arcade:       "#ff6b9d",
  Operations:   "#43ff64",
  Development:  "#ffbd33",
};

// Build slug -> page map
window.PAGE_MAP = window.PAGES.reduce((acc, p) => { acc[p.slug] = p; return acc; }, {});

// Page cache (slug -> markdown text)
window.PAGE_CACHE = {};

// Loader
window.loadPage = async function loadPage(slug) {
  if (window.PAGE_CACHE[slug]) return window.PAGE_CACHE[slug];
  const page = window.PAGE_MAP[slug];
  if (!page) return null;
  // Absolute path so the fetch works from prerendered routes like /architecture/
  // (a relative "pages/..." would resolve against the subpath and 404).
  const res = await fetch("/" + page.file + "?v=2");
  if (!res.ok) return null;
  const text = await res.text();
  window.PAGE_CACHE[slug] = text;
  return text;
};

// Eager-load all pages in background so search + graph have content
window.loadAllPages = async function () {
  const all = window.PAGES.map(p => window.loadPage(p.slug));
  await Promise.all(all);
};

// Build node + edge lists. Page nodes come from window.PAGES; tag nodes come
// from window.PAGES[].tags (a tag is included only if it touches >= 2 pages).
// Page→page edges are derived from #/slug cross-references inside the markdown;
// page→tag edges from the tag list. Returns:
//   { pageNodes, tagNodes, edges }
window.buildGraph = function () {
  const slugs = window.PAGES.map(p => p.slug);
  const slugSet = new Set(slugs);

  // 1. page→page edges from markdown
  const pageEdges = [];
  const seen = new Set();
  for (const page of window.PAGES) {
    const md = window.PAGE_CACHE[page.slug] || "";
    const re = /#\/([a-z0-9-]+)/g;
    let m;
    while ((m = re.exec(md)) !== null) {
      const tgt = m[1];
      if (tgt === page.slug) continue;
      if (!slugSet.has(tgt)) continue;
      const key = page.slug + "→" + tgt;
      if (seen.has(key)) continue;
      seen.add(key);
      pageEdges.push({ from: page.slug, to: tgt, kind: "page" });
    }
  }

  // 2. count tag usage; keep only tags shared by >= 2 pages
  const tagCount = {};
  for (const p of window.PAGES) {
    for (const t of (p.tags || [])) {
      tagCount[t] = (tagCount[t] || 0) + 1;
    }
  }
  const keptTags = Object.keys(tagCount).filter(t => tagCount[t] >= 2);

  // 3. tag nodes
  const tagNodes = keptTags.map(t => ({
    slug: "tag:" + t,
    title: "#" + (window.TAG_LABELS[t] || t),
    section: "Tag",
    isTag: true,
    tag: t,
  }));

  // 4. page→tag edges
  const tagEdges = [];
  const keptSet = new Set(keptTags);
  for (const p of window.PAGES) {
    for (const t of (p.tags || [])) {
      if (!keptSet.has(t)) continue;
      tagEdges.push({ from: p.slug, to: "tag:" + t, kind: "tag" });
    }
  }

  return {
    nodes: window.PAGES.concat(tagNodes),
    pageNodes: window.PAGES.slice(),
    tagNodes,
    edges: pageEdges.concat(tagEdges),
    pageEdges,
    tagEdges,
  };
};
