// =========================================================
// GraphView — Obsidian-style force-directed graph
//   - Two node kinds: PAGE (coloured by section) and TAG (neutral, smaller, # label)
//   - Two edge kinds: page→page (purple) and page→tag (faint)
//   - Hover -> highlight neighbours, show tooltip
//   - Click -> navigate (page) or focus + filter (tag)
//   - Modes: 'overlay' (fullscreen) or 'mini' (embedded teaser)
// =========================================================

// var (not const): shared global scope across app scripts, no modules — see markdown.js.
var { useEffect, useRef, useState } = React;

// ---------- Canvas palette ----------
// The graph draws to a 2D canvas, so no CSS reaches it and every colour has to be a literal here.
// Two sets, keyed by theme, read fresh on each frame: the canvas redraws under
// requestAnimationFrame, so a theme change is picked up on the next frame with no invalidation
// step of its own.
//
// Two entries are not simple inversions. `nodeRing` is a ring drawn in the PAGE colour to punch
// the node away from the background, so it follows the page rather than the ink. `labelShadow` is
// a glow behind the label in the page colour for the same reason: on dark it is a black halo, on
// light it has to be a white one, and left alone it would smudge every label.
var GRAPH_PALETTE = {
  dark: {
    edgeHot:    "rgba(162,155,254,0.9)",
    edgeWarm:   "rgba(162,155,254,0.45)",
    edgeDimmed: "rgba(255,255,255,0.03)",
    edgeTag:    "rgba(207,198,230,0.10)",
    edge:       "rgba(255,255,255,0.10)",
    tagFill:    "rgba(207,198,230,0.85)",
    tagFillDim: "rgba(207,198,230,0.16)",
    tagRing:    "rgba(255,255,255,0.5)",
    nodeRing:   "rgba(8,8,12,0.85)",
    labelShadow:"rgba(8,8,12,0.95)",
    labelHot:   "#ffffff",
    labelTag:   "rgba(225,219,250,0.78)",
    label:      "rgba(255,255,255,0.72)",
  },
  light: {
    edgeHot:    "rgba(90,72,212,0.9)",
    edgeWarm:   "rgba(90,72,212,0.5)",
    edgeDimmed: "rgba(20,18,31,0.05)",
    edgeTag:    "rgba(20,18,31,0.14)",
    edge:       "rgba(20,18,31,0.16)",
    tagFill:    "rgba(91,84,112,0.85)",
    tagFillDim: "rgba(91,84,112,0.20)",
    tagRing:    "rgba(20,18,31,0.35)",
    nodeRing:   "rgba(245,245,250,0.9)",
    labelShadow:"rgba(245,245,250,0.95)",
    labelHot:   "#14121f",
    labelTag:   "rgba(20,18,31,0.72)",
    label:      "rgba(20,18,31,0.75)",
  },
};

function graphPalette() {
  return GRAPH_PALETTE[document.documentElement.getAttribute("data-theme") === "light" ? "light" : "dark"];
}

function buildSimulation(graph, w, h) {
  const all = graph.nodes;
  // Initial positions: spread evenly across the canvas using a jittered grid.
  // Stratified placement keeps the warm-up gentle (no nodes start on top of each other).
  const cols = Math.ceil(Math.sqrt(all.length * (w / h)));
  const rows = Math.ceil(all.length / cols);
  const cellW = w / cols;
  const cellH = h / rows;
  const nodes = all.map((p, i) => {
    const c = i % cols;
    const r = Math.floor(i / cols);
    const jx = (Math.random() - 0.5) * cellW * 0.6;
    const jy = (Math.random() - 0.5) * cellH * 0.6;
    return {
      id: p.isTag ? p.slug : p.slug,
      title: p.title,
      section: p.section,
      isTag: !!p.isTag,
      tag: p.tag || null,
      page: p,
      x: cellW * (c + 0.5) + jx,
      y: cellH * (r + 0.5) + jy,
      vx: 0, vy: 0,
      degree: 0,
    };
  });
  const byId = nodes.reduce((acc, n) => { acc[n.id] = n; return acc; }, {});
  const edges = graph.edges
    .map(e => ({ source: byId[e.from], target: byId[e.to], kind: e.kind }))
    .filter(e => e.source && e.target);
  for (const e of edges) {
    e.source.degree += 1;
    e.target.degree += 1;
  }
  return { nodes, edges, byId };
}

function tick(sim, w, h, params) {
  const { repulsion, linkDistance, linkDistanceTag, linkStrength, linkStrengthTag, gravity, damping } = params;
  const { nodes, edges } = sim;

  // Reset
  for (const n of nodes) { n.fx = 0; n.fy = 0; }

  // Repulsion — pages repel everything; tags repel each other less so they cluster around their pages
  for (let i = 0; i < nodes.length; i++) {
    const a = nodes[i];
    const aRep = a.isTag ? repulsion * 0.55 : repulsion;
    for (let j = i + 1; j < nodes.length; j++) {
      const b = nodes[j];
      const bRep = b.isTag ? repulsion * 0.55 : repulsion;
      const pair = (aRep + bRep) * 0.5;
      let dx = a.x - b.x, dy = a.y - b.y;
      let d2 = dx * dx + dy * dy;
      if (d2 < 1) { d2 = 1; dx = (Math.random() - 0.5) * 2; dy = (Math.random() - 0.5) * 2; }
      const d = Math.sqrt(d2);
      const f = pair / d2;
      const fx = (dx / d) * f, fy = (dy / d) * f;
      a.fx += fx; a.fy += fy;
      b.fx -= fx; b.fy -= fy;
    }
  }

  // Links
  for (const e of edges) {
    const isTag = e.kind === "tag";
    const targetLen = isTag ? linkDistanceTag : linkDistance;
    const k = isTag ? linkStrengthTag : linkStrength;
    const a = e.source, b = e.target;
    const dx = b.x - a.x, dy = b.y - a.y;
    const d = Math.sqrt(dx * dx + dy * dy) || 1;
    const diff = (d - targetLen) * k;
    const fx = (dx / d) * diff, fy = (dy / d) * diff;
    a.fx += fx; a.fy += fy;
    b.fx -= fx; b.fy -= fy;
  }

  // Gravity to center
  for (const n of nodes) {
    n.fx += (w / 2 - n.x) * gravity;
    n.fy += (h / 2 - n.y) * gravity;
  }

  // Integrate with force + velocity clamping (prevents runaway bouncing
  // when two nodes happen to start close together).
  const maxForce = params.maxForce || 28;
  const maxV = params.maxV || 4;
  for (const n of nodes) {
    if (n.fixed) continue;
    // Clamp force magnitude
    const fm = Math.hypot(n.fx, n.fy);
    if (fm > maxForce) {
      const k = maxForce / fm;
      n.fx *= k; n.fy *= k;
    }
    n.vx = (n.vx + n.fx) * damping;
    n.vy = (n.vy + n.fy) * damping;
    // Clamp velocity magnitude
    const vm = Math.hypot(n.vx, n.vy);
    if (vm > maxV) {
      const k = maxV / vm;
      n.vx *= k; n.vy *= k;
    }
    n.x += n.vx;
    n.y += n.vy;
    const pad = n.isTag ? 22 : 30;
    n.x = Math.max(pad, Math.min(w - pad, n.x));
    n.y = Math.max(pad, Math.min(h - pad, n.y));
  }
}

function nodeRadius(n, mode) {
  if (n.isTag) {
    return mode === "mini" ? 3.2 : 4.2;
  }
  const base = mode === "mini" ? 5.5 : 7;
  return base + Math.min(n.degree, 6) * (mode === "mini" ? 1.1 : 1.4);
}

function GraphView({ mode = "overlay", onClose, onNav, currentSlug, initialTag }) {
  const canvasRef = useRef(null);
  const wrapRef = useRef(null);
  const stateRef = useRef({ sim: null, hover: null, dragging: null, animation: null, focusedTag: initialTag || null });
  const [tooltip, setTooltip] = useState(null);
  const [, force] = useState(0);

  useEffect(() => {
    const canvas = canvasRef.current;
    const wrap = wrapRef.current;
    if (!canvas || !wrap) return;

    let w, h;
    function resize() {
      const r = wrap.getBoundingClientRect();
      w = r.width;
      h = r.height;
      const dpr = window.devicePixelRatio || 1;
      canvas.width = w * dpr;
      canvas.height = h * dpr;
      canvas.style.width = w + "px";
      canvas.style.height = h + "px";
      const ctx = canvas.getContext("2d");
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    }
    resize();

    const graph = window.buildGraph();
    const sim = buildSimulation(graph, w, h);
    stateRef.current.sim = sim;

    // Adjacency for highlight
    const adjacency = {};
    for (const e of sim.edges) {
      (adjacency[e.source.id] = adjacency[e.source.id] || new Set()).add(e.target.id);
      (adjacency[e.target.id] = adjacency[e.target.id] || new Set()).add(e.source.id);
    }

    // pin the current page in the centre on overlay mode
    if (currentSlug && sim.byId[currentSlug] && mode === "overlay") {
      const n = sim.byId[currentSlug];
      n.x = w / 2; n.y = h / 2; n.fixed = true;
    }

    // Mobile viewports get a tighter layout so the entire graph fits in view
    // instead of rendering only the central cluster behind a virtual zoom.
    // We crank gravity, drop repulsion and shorten link distances; the result
    // is a more compact graph that lands well inside a phone canvas.
    const isMobile = (typeof window !== "undefined") && window.innerWidth < 768;
    let params;
    if (mode === "mini") {
      params = isMobile
        ? { repulsion: 1200, linkDistance: 60, linkDistanceTag: 32, linkStrength: 0.07, linkStrengthTag: 0.09, gravity: 0.045, damping: 0.84 }
        : { repulsion: 2400, linkDistance: 105, linkDistanceTag: 55, linkStrength: 0.05, linkStrengthTag: 0.07, gravity: 0.018, damping: 0.84 };
    } else {
      params = isMobile
        ? { repulsion: 2400, linkDistance: 95, linkDistanceTag: 52, linkStrength: 0.06, linkStrengthTag: 0.09, gravity: 0.035, damping: 0.86 }
        : { repulsion: 4800, linkDistance: 170, linkDistanceTag: 90, linkStrength: 0.04, linkStrengthTag: 0.07, gravity: 0.011, damping: 0.86 };
    }

    // Warm-up: settle the layout silently before the first paint so the user
    // doesn't see the violent rearrangement from random initial positions.
    const warmupSteps = mode === "mini" ? 320 : 420;
    for (let s = 0; s < warmupSteps; s++) tick(sim, w, h, params);
    // Zero velocities so the first frame doesn't carry leftover motion.
    for (const n of sim.nodes) { n.vx = 0; n.vy = 0; }

    function draw(t) {
      // After warm-up the layout is already mostly stable; one tick per frame
      // is enough for the gentle drift / drag interactions.
      tick(sim, w, h, params);

      const ctx = canvas.getContext("2d");
      ctx.clearRect(0, 0, w, h);

      // Read per frame rather than per mount, so a theme change lands on the next frame without
      // needing to invalidate or remount anything. clearRect leaves the canvas transparent, so
      // the page colour already shows through correctly in both themes.
      const P = graphPalette();

      const hoverId = stateRef.current.hover;
      const hoverSet = hoverId ? adjacency[hoverId] || new Set() : null;
      const focusedTag = stateRef.current.focusedTag;
      const focusedTagId = focusedTag ? "tag:" + focusedTag : null;
      const focusedSet = focusedTagId ? adjacency[focusedTagId] || new Set() : null;

      // Edges
      for (const e of sim.edges) {
        const isTag = e.kind === "tag";
        const isHi = hoverId && (e.source.id === hoverId || e.target.id === hoverId);
        const isCurrent = currentSlug && (e.source.id === currentSlug || e.target.id === currentSlug);
        const isFocused = focusedTagId && (e.source.id === focusedTagId || e.target.id === focusedTagId);
        let stroke;
        let lw = isTag ? 0.8 : 1.1;
        if (isHi || isFocused) {
          stroke = P.edgeHot;
          lw += 0.4;
        } else if (isCurrent) {
          stroke = P.edgeWarm;
        } else if (isTag) {
          stroke = hoverId ? P.edgeDimmed : P.edgeTag;
        } else {
          stroke = hoverId ? P.edgeDimmed : P.edge;
        }
        ctx.strokeStyle = stroke;
        ctx.lineWidth = lw;
        ctx.beginPath();
        ctx.moveTo(e.source.x, e.source.y);
        ctx.lineTo(e.target.x, e.target.y);
        ctx.stroke();
      }

      // Nodes
      for (const n of sim.nodes) {
        const color = n.isTag ? window.tagColor() : (window.SECTION_COLORS[n.section] || P.edgeHot);
        const isHover = n.id === hoverId;
        const isNeighbour = hoverSet && hoverSet.has(n.id);
        const isCurrent = n.id === currentSlug;
        const isFocused = focusedSet && focusedSet.has(n.id) || (focusedTagId && n.id === focusedTagId);
        const dim = (hoverId && !isHover && !isNeighbour) || (focusedTagId && !isFocused);
        const r = nodeRadius(n, mode) * (isHover ? 1.4 : (isCurrent ? 1.2 : 1));

        // Halo for current/hover
        if ((isHover || isCurrent) && !n.isTag) {
          ctx.beginPath();
          ctx.arc(n.x, n.y, r * 2.4, 0, Math.PI * 2);
          const grd = ctx.createRadialGradient(n.x, n.y, r, n.x, n.y, r * 2.4);
          grd.addColorStop(0, color + "66");
          grd.addColorStop(1, color + "00");
          ctx.fillStyle = grd;
          ctx.fill();
        }

        // Draw node — tags as small open ring, pages as filled
        if (n.isTag) {
          ctx.beginPath();
          ctx.arc(n.x, n.y, r, 0, Math.PI * 2);
          ctx.fillStyle = dim ? P.tagFillDim : P.tagFill;
          ctx.fill();
          ctx.strokeStyle = dim ? "rgba(0,0,0,0)" : P.tagRing;
          ctx.lineWidth = 1;
          ctx.stroke();
        } else {
          ctx.beginPath();
          ctx.arc(n.x, n.y, r, 0, Math.PI * 2);
          ctx.fillStyle = dim ? color + "44" : color;
          ctx.fill();
          ctx.strokeStyle = dim ? "rgba(0,0,0,0)" : P.nodeRing;
          ctx.lineWidth = 1.5;
          ctx.stroke();
        }

        // Labels:
        //   - pages: in overlay always; in mini show for non-dim, font shrinks
        //   - tags: always show (small, italic-ish, soft white)
        const showLabel = !dim && (
          n.isTag
            ? true
            : (mode === "overlay" || isHover || isCurrent || n.degree >= 2)
        );
        if (showLabel) {
          ctx.shadowColor = P.labelShadow;
          ctx.shadowBlur = 4;
          ctx.textAlign = "center";
          ctx.textBaseline = "top";
          if (n.isTag) {
            ctx.font = (mode === "mini" ? "500 9.5px " : "500 10.5px ") + "JetBrains Mono, ui-monospace, monospace";
            ctx.fillStyle = isHover ? P.labelHot : P.labelTag;
            ctx.fillText(n.title, n.x, n.y + r + 4);
          } else {
            ctx.font = (mode === "mini" ? "500 10.5px " : "500 11.5px ") + "Inter, system-ui, sans-serif";
            ctx.fillStyle = (isHover || isCurrent) ? P.labelHot : P.label;
            ctx.fillText(n.title, n.x, n.y + r + 6);
          }
          ctx.shadowBlur = 0;
        }
      }
      stateRef.current.animation = requestAnimationFrame(() => draw(t + 16));
    }
    stateRef.current.animation = requestAnimationFrame(() => draw(0));

    function getMouse(e) {
      const r = canvas.getBoundingClientRect();
      return { x: e.clientX - r.left, y: e.clientY - r.top };
    }
    function findNode(p) {
      for (let i = sim.nodes.length - 1; i >= 0; i--) {
        const n = sim.nodes[i];
        const dx = p.x - n.x, dy = p.y - n.y;
        const r = nodeRadius(n, mode) + 6;
        if (dx * dx + dy * dy <= r * r) return n;
      }
      return null;
    }

    function onMove(e) {
      const p = getMouse(e);
      if (stateRef.current.dragging) {
        const d = stateRef.current.dragging;
        d.x = p.x; d.y = p.y; d.vx = 0; d.vy = 0; d.fixed = true;
        return;
      }
      const n = findNode(p);
      const prev = stateRef.current.hover;
      stateRef.current.hover = n ? n.id : null;
      if (n) {
        canvas.style.cursor = "pointer";
        if (mode === "overlay") {
          const r = wrap.getBoundingClientRect();
          if (n.isTag) {
            const count = (sim.edges.filter(e => e.kind === "tag" && (e.source.id === n.id || e.target.id === n.id))).length;
            setTooltip({ x: e.clientX - r.left + 12, y: e.clientY - r.top + 12, title: n.title, meta: "tag · " + count + " pages" });
          } else {
            setTooltip({ x: e.clientX - r.left + 12, y: e.clientY - r.top + 12, title: n.title, meta: n.section + "  ·  /" + n.id });
          }
        }
      } else {
        canvas.style.cursor = "default";
        if (tooltip) setTooltip(null);
      }
      if (prev !== stateRef.current.hover) force(x => x + 1);
    }
    function onLeave() {
      stateRef.current.hover = null;
      stateRef.current.dragging = null;
      canvas.style.cursor = "default";
      setTooltip(null);
    }
    function onDown(e) {
      const p = getMouse(e);
      const n = findNode(p);
      if (n) stateRef.current.dragging = n;
    }
    function onUp() {
      if (stateRef.current.dragging && stateRef.current.dragging.id !== currentSlug) {
        stateRef.current.dragging.fixed = false;
      }
      stateRef.current.dragging = null;
    }
    function onClick(e) {
      const p = getMouse(e);
      const n = findNode(p);
      if (!n) {
        // clear tag focus on empty-canvas click
        if (stateRef.current.focusedTag) {
          stateRef.current.focusedTag = null;
          force(x => x + 1);
        }
        return;
      }
      if (n.isTag) {
        // toggle focus on tag
        stateRef.current.focusedTag = stateRef.current.focusedTag === n.tag ? null : n.tag;
        force(x => x + 1);
        return;
      }
      if (onNav) onNav(n.id);
    }

    canvas.addEventListener("mousemove", onMove);
    canvas.addEventListener("mouseleave", onLeave);
    canvas.addEventListener("mousedown", onDown);
    window.addEventListener("mouseup", onUp);
    canvas.addEventListener("click", onClick);

    const ro = new ResizeObserver(() => { resize(); });
    ro.observe(wrap);

    return () => {
      cancelAnimationFrame(stateRef.current.animation);
      canvas.removeEventListener("mousemove", onMove);
      canvas.removeEventListener("mouseleave", onLeave);
      canvas.removeEventListener("mousedown", onDown);
      window.removeEventListener("mouseup", onUp);
      canvas.removeEventListener("click", onClick);
      ro.disconnect();
    };
  }, [mode, currentSlug, initialTag]);

  // ESC closes overlay
  useEffect(() => {
    if (mode !== "overlay") return;
    function onKey(e) { if (e.key === "Escape" && onClose) onClose(); }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [mode, onClose]);

  if (mode === "mini") {
    return React.createElement("div", {
      className: "mini-graph",
      ref: wrapRef,
      onClick: () => onNav && onNav("__graph__"),
    },
      React.createElement("canvas", { className: "mini-graph-canvas", ref: canvasRef }),
      React.createElement("div", { className: "mini-graph-overlay" },
        React.createElement("div", { className: "mini-graph-tag" },
          "Knowledge graph · ",
          window.PAGES.length, " pages · ",
          window.buildGraph().tagNodes.length, " tags",
        ),
        React.createElement("div", { className: "mini-graph-cta" },
          "Open full graph ",
          React.createElement(window.Icon, { name: "arrow-right" }),
        ),
      ),
    );
  }

  const graphCounts = window.buildGraph();
  return React.createElement("div", { className: "graph-overlay" },
    React.createElement("div", { className: "graph-header" },
      React.createElement(window.Icon, { name: "graph", width: 16, height: 16, style: { color: "var(--accent-text)" } }),
      React.createElement("div", null,
        React.createElement("div", { className: "graph-title" }, "Knowledge graph"),
        React.createElement("div", { className: "graph-subtitle" },
          graphCounts.pageNodes.length, " pages · ",
          graphCounts.tagNodes.length, " tags · ",
          graphCounts.pageEdges.length, " cross-references · click a tag to focus, click empty space to clear",
        ),
      ),
      React.createElement("button", {
        className: "graph-close",
        onClick: onClose,
      },
        React.createElement(window.Icon, { name: "x", width: 13, height: 13 }),
        React.createElement("span", null, "Close (Esc)"),
      ),
    ),
    React.createElement("div", { className: "graph-canvas-wrap", ref: wrapRef },
      React.createElement("canvas", { className: "graph-canvas", ref: canvasRef }),
      tooltip && React.createElement("div", {
        className: "graph-tooltip",
        style: { left: tooltip.x, top: tooltip.y },
      },
        React.createElement("div", null, tooltip.title),
        React.createElement("div", { className: "gt-meta" }, tooltip.meta),
      ),
      React.createElement("div", { className: "graph-legend" },
        React.createElement("div", { style: { fontWeight: 600, color: "var(--fg-1)", marginBottom: 6 } }, "Legend"),
        window.SECTIONS.map(s =>
          React.createElement("div", { key: s, className: "graph-legend-row" },
            React.createElement("span", { className: "graph-legend-dot", style: { background: window.SECTION_COLORS[s], color: window.SECTION_COLORS[s] } }),
            React.createElement("span", null, s),
          )
        ),
        React.createElement("div", { className: "graph-legend-row", style: { marginTop: 6, paddingTop: 6, borderTop: "1px solid var(--hairline)" } },
          React.createElement("span", { className: "graph-legend-dot", style: { background: window.tagColor(), color: window.tagColor(), width: 7, height: 7, boxShadow: "none", opacity: 0.85 } }),
          React.createElement("span", { style: { fontFamily: "var(--font-mono)" } }, "#tag"),
        ),
      ),
    ),
  );
}

window.GraphView = GraphView;
