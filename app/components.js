// =========================================================
// Components: TopNav, Sidebar, TOC, Search, Footer
// =========================================================

// var (not const): shared global scope across app scripts, no modules — see markdown.js.
var { useState, useEffect, useRef, useMemo } = React;

// ---------- Icons (inline SVG, lucide-style) ----------
function Icon({ name, ...props }) {
  const common = { width: 14, height: 14, viewBox: "0 0 24 24", fill: "none",
                   stroke: "currentColor", strokeWidth: 2, strokeLinecap: "round", strokeLinejoin: "round", ...props };
  switch (name) {
    case "search":  return React.createElement("svg", common, React.createElement("circle", {cx:11,cy:11,r:7}), React.createElement("path", {d:"m21 21-4.3-4.3"}));
    case "github":  return React.createElement("svg", common, React.createElement("path", {d:"M9 19c-5 1.5-5-2.5-7-3m14 6v-3.87a3.37 3.37 0 0 0-.94-2.61c3.14-.35 6.44-1.54 6.44-7A5.44 5.44 0 0 0 20 4.77 5.07 5.07 0 0 0 19.91 1S18.73.65 16 2.48a13.38 13.38 0 0 0-7 0C6.27.65 5.09 1 5.09 1A5.07 5.07 0 0 0 5 4.77a5.44 5.44 0 0 0-1.5 3.78c0 5.42 3.3 6.61 6.44 7A3.37 3.37 0 0 0 9 18.13V22"}));
    case "graph":   return React.createElement("svg", common, React.createElement("circle",{cx:5,cy:6,r:3}), React.createElement("circle",{cx:19,cy:6,r:3}), React.createElement("circle",{cx:12,cy:18,r:3}), React.createElement("path", {d:"M7.5 8 11 16M16.5 8 13 16"}));
    case "book":    return React.createElement("svg", common, React.createElement("path", {d:"M4 19.5A2.5 2.5 0 0 1 6.5 17H20"}), React.createElement("path", {d:"M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z"}));
    case "arrow-right": return React.createElement("svg", common, React.createElement("path", {d:"M5 12h14M13 5l7 7-7 7"}));
    case "arrow-left":  return React.createElement("svg", common, React.createElement("path", {d:"M19 12H5M12 19l-7-7 7-7"}));
    case "menu":   return React.createElement("svg", common, React.createElement("path", {d:"M3 12h18M3 6h18M3 18h18"}));
    case "x":      return React.createElement("svg", common, React.createElement("path", {d:"M18 6 6 18M6 6l12 12"}));
    case "globe":   return React.createElement("svg", common, React.createElement("circle",{cx:12,cy:12,r:10}), React.createElement("path",{d:"M2 12h20"}), React.createElement("path",{d:"M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z"}));
    case "external": return React.createElement("svg", common, React.createElement("path", {d:"M7 17 17 7M7 7h10v10"}));
    case "package":  return React.createElement("svg", common, React.createElement("path", {d:"m16.5 9.4-9-5.19M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16zM3.27 6.96 12 12.01l8.73-5.05M12 22.08V12"}));
    case "sun":     return React.createElement("svg", common, React.createElement("circle",{cx:12,cy:12,r:4}), React.createElement("path",{d:"M12 2v2M12 20v2M4.93 4.93l1.41 1.41M17.66 17.66l1.41 1.41M2 12h2M20 12h2M6.34 17.66l-1.41 1.41M19.07 4.93l-1.41 1.41"}));
    case "moon":    return React.createElement("svg", common, React.createElement("path", {d:"M12 3a6 6 0 0 0 9 9 9 9 0 1 1-9-9Z"}));
    default: return null;
  }
}
window.Icon = Icon;

// ---------- Search ----------
function Search({ onPick }) {
  const [q, setQ] = useState("");
  const [active, setActive] = useState(0);
  const [focused, setFocused] = useState(false);
  const inputRef = useRef(null);

  useEffect(() => {
    function onKey(e) {
      const isCmdK = (e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "k";
      const isSlash = e.key === "/" && !["INPUT", "TEXTAREA"].includes(document.activeElement?.tagName);
      if (isCmdK || isSlash) {
        e.preventDefault();
        inputRef.current?.focus();
        inputRef.current?.select();
      }
      if (e.key === "Escape" && focused) {
        inputRef.current?.blur();
        setFocused(false);
      }
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [focused]);

  const tagSuggestions = useMemo(() => {
    const raw = q.trim().toLowerCase();
    const term = raw.replace(/^#/, "");
    if (!term) return [];
    const counts = {};
    for (const p of window.PAGES) {
      for (const t of (p.tags || [])) {
        counts[t] = (counts[t] || 0) + 1;
      }
    }
    const tags = Object.keys(counts)
      .filter(t => {
        const label = (window.TAG_LABELS[t] || t).toLowerCase();
        return t.toLowerCase().includes(term) || label.includes(term);
      })
      .sort((a, b) => {
        const aLabel = (window.TAG_LABELS[a] || a).toLowerCase();
        const bLabel = (window.TAG_LABELS[b] || b).toLowerCase();
        const aStarts = a.startsWith(term) || aLabel.startsWith(term);
        const bStarts = b.startsWith(term) || bLabel.startsWith(term);
        if (aStarts !== bStarts) return aStarts ? -1 : 1;
        return counts[b] - counts[a];
      });
    return tags.slice(0, 6).map(t => ({ tag: t, count: counts[t] }));
  }, [q]);

  const results = useMemo(() => {
    const term = q.trim().toLowerCase();
    if (!term) return [];
    const termNoHash = term.replace(/^#/, "");
    const out = [];
    for (const p of window.PAGES) {
      const md = window.PAGE_CACHE[p.slug] || "";
      const titleHit = p.title.toLowerCase().includes(term);
      const sectionHit = p.section.toLowerCase().includes(term);
      const descHit = (p.description || "").toLowerCase().includes(term);
      const tags = (p.tags || []);
      // Tag match: when user types #foo, only direct tag matches count; otherwise allow contains.
      const tagHit = term.startsWith("#")
        ? tags.some(t => t.toLowerCase() === termNoHash || (window.TAG_LABELS[t] || t).toLowerCase() === termNoHash)
        : tags.some(t => t.toLowerCase().includes(term) || (window.TAG_LABELS[t] || t).toLowerCase().includes(term));
      let bodyHit = -1;
      if (md && !term.startsWith("#")) {
        bodyHit = md.toLowerCase().indexOf(term);
      }
      if (!titleHit && !sectionHit && !descHit && !tagHit && bodyHit < 0) continue;
      let snippet = p.description || "";
      if (tagHit) {
        snippet = "Tags: " + tags.map(t => "#" + (window.TAG_LABELS[t] || t)).join("  ");
      } else if (bodyHit >= 0) {
        const start = Math.max(0, bodyHit - 40);
        const end = Math.min(md.length, bodyHit + term.length + 80);
        snippet = (start > 0 ? "…" : "") + md.slice(start, end).replace(/[#`*_>]/g, "") + (end < md.length ? "…" : "");
      }
      // Score: title=4, tag-direct=3.5, section=2, desc=1, body=0.5
      const score = (titleHit ? 4 : 0) + (tagHit ? (term.startsWith("#") ? 4 : 3) : 0) + (sectionHit ? 2 : 0) + (descHit ? 1 : 0) + (bodyHit >= 0 ? 0.5 : 0);
      out.push({ page: p, snippet, score });
    }
    out.sort((a, b) => b.score - a.score);
    return out.slice(0, 8);
  }, [q]);

  const showResults = focused && q.trim().length > 0;

  function highlight(s) {
    const term = q.trim();
    if (!term) return s;
    const idx = s.toLowerCase().indexOf(term.toLowerCase());
    if (idx < 0) return s;
    return [
      s.slice(0, idx),
      React.createElement("mark", { key: "m" }, s.slice(idx, idx + term.length)),
      s.slice(idx + term.length),
    ];
  }

  function pick(p) {
    onPick(p.slug);
    setQ("");
    inputRef.current?.blur();
    setFocused(false);
  }

  function onKeyDown(e) {
    if (!showResults) return;
    if (e.key === "ArrowDown") { e.preventDefault(); setActive(a => Math.min(a + 1, results.length - 1)); }
    if (e.key === "ArrowUp")   { e.preventDefault(); setActive(a => Math.max(a - 1, 0)); }
    if (e.key === "Enter")     { e.preventDefault(); const r = results[active]; if (r) pick(r.page); }
  }

  return React.createElement("div", { className: "search-wrap" },
    React.createElement(Icon, { name: "search", className: "search-icon" }),
    React.createElement("input", {
      ref: inputRef,
      className: "search-input",
      placeholder: "Search docs or #tag…",
      value: q,
      role: "combobox",
      "aria-label": "Search documentation",
      "aria-autocomplete": "list",
      "aria-expanded": showResults && results.length > 0,
      "aria-controls": "search-listbox",
      "aria-activedescendant": showResults && results[active] ? "search-opt-" + active : undefined,
      onChange: e => { setQ(e.target.value); setActive(0); },
      onFocus: () => setFocused(true),
      onBlur: () => setTimeout(() => setFocused(false), 120),
      onKeyDown,
    }),
    !q && React.createElement("span", { className: "search-kbd" }, "⌘K"),
    showResults && React.createElement("div", { className: "search-results" },
      tagSuggestions.length > 0 && React.createElement("div", { className: "search-tag-section" },
        React.createElement("div", { className: "search-section-label" }, "Tags"),
        React.createElement("div", { className: "search-tag-row" },
          tagSuggestions.map(t =>
            React.createElement("button", {
              key: t.tag,
              className: "search-tag-suggest",
              onMouseDown: e => {
                e.preventDefault();
                setQ("#" + t.tag);
                setActive(0);
              },
              title: t.count + " page" + (t.count === 1 ? "" : "s"),
            },
              React.createElement("span", { className: "stt-hash" }, "#"),
              (window.TAG_LABELS[t.tag] || t.tag),
              React.createElement("span", { className: "stt-count" }, t.count),
            )
          )
        )
      ),
      results.length === 0
        ? (tagSuggestions.length === 0 && React.createElement("div", { className: "search-empty" }, "No matches"))
        : React.createElement("div", { id: "search-listbox", role: "listbox", "aria-label": "Search results" },
            results.map((r, i) =>
              React.createElement("a", {
                key: r.page.slug,
                id: "search-opt-" + i,
                role: "option",
                "aria-selected": i === active,
                className: "search-result" + (i === active ? " active" : ""),
                onMouseDown: e => { e.preventDefault(); pick(r.page); },
                onMouseEnter: () => setActive(i),
              },
                React.createElement("div", { className: "search-result-title" }, highlight(r.page.title)),
                React.createElement("div", { className: "search-result-meta" }, r.page.section + "  ·  /" + r.page.slug),
                React.createElement("div", { className: "search-result-snippet" }, highlight(r.snippet)),
              )
            )
          )
    ),
  );
}
window.Search = Search;


// ---------- Theme ----------
// The switch, and the circular reveal that plays when it is pressed.
//
// The technique is theme-toggle.rdsx.dev's, the same one the webapp ships. The animation lives
// here rather than in CSS because the circle's origin is wherever the button happens to be, which
// only JavaScript knows.
//
// The FOUC guard in index.html has already stamped data-theme on <html> before this file loads,
// so `readTheme` reads the DOM rather than recomputing from storage: one source of truth.
var THEME_STORAGE_KEY = "f1sl.docs.theme";
// Kept in step with the keyframe in tokens.css. Two places, because a keyframe on a
// pseudo-element outside the document tree cannot read a custom property.
var THEME_TRANSITION_MS = 480;

function readTheme() {
  return document.documentElement.getAttribute("data-theme") === "light" ? "light" : "dark";
}

// The mobile browser chrome follows the page, and the meta tag is the only way to say so.
function paintBrowserChrome(theme) {
  var meta = document.querySelector('meta[name="theme-color"]');
  if (meta) meta.setAttribute("content", theme === "light" ? "#f5f5fa" : "#0c0d14");
}

function applyTheme(theme) {
  document.documentElement.setAttribute("data-theme", theme);
  paintBrowserChrome(theme);
  try { localStorage.setItem(THEME_STORAGE_KEY, theme); } catch (e) {}
  // Mermaid renders to SVG with colours baked in at render time, and the canvas graph keeps its
  // own palette, so both have to be told. Anything else on the page follows the tokens for free.
  window.dispatchEvent(new CustomEvent("f1sl:themechange", { detail: { theme: theme } }));
}

// Expands a circle of the new theme from (x, y) out to the farthest corner, so it finishes
// covering the viewport exactly as the animation ends.
function revealTheme(x, y) {
  var endRadius = Math.hypot(Math.max(x, innerWidth - x), Math.max(y, innerHeight - y));
  document.documentElement.animate(
    { clipPath: ["circle(0px at " + x + "px " + y + "px)",
                 "circle(" + endRadius + "px at " + x + "px " + y + "px)"] },
    { duration: THEME_TRANSITION_MS, easing: "ease-in-out",
      pseudoElement: "::view-transition-new(root)" }
  );
}

function ThemeToggle() {
  const [theme, setTheme] = useState(readTheme);

  function onClick(e) {
    const next = theme === "light" ? "dark" : "light";
    const canAnimate = typeof document.startViewTransition === "function"
      && !matchMedia("(prefers-reduced-motion: reduce)").matches;

    if (!canAnimate) { applyTheme(next); setTheme(next); return; }

    const r = e.currentTarget.getBoundingClientRect();
    const x = r.left + r.width / 2;
    const y = r.top + r.height / 2;

    // applyTheme must run synchronously inside this callback: the API snapshots the old frame
    // before it and the new frame right after it returns, so a React state update alone would
    // land after the snapshot and the transition would capture no change at all.
    const t = document.startViewTransition(function () {
      applyTheme(next);
      setTheme(next);
    });
    t.ready.then(function () { revealTheme(x, y); }, function () {
      // `ready` rejects when the browser skips the transition. The theme has already flipped, so
      // there is nothing to undo and the reveal is simply dropped.
    });
  }

  // The icon shows the theme you are about to switch TO, not the one you are in, so the control
  // reads as an action rather than as a status light.
  return React.createElement("button", {
    className: "nav-pill",
    onClick: onClick,
    title: theme === "light" ? "Switch to dark theme" : "Switch to light theme",
    "aria-label": theme === "light" ? "Switch to dark theme" : "Switch to light theme",
  }, React.createElement(Icon, { name: theme === "light" ? "moon" : "sun" }));
}
window.ThemeToggle = ThemeToggle;

// ---------- TopNav ----------
function TopNav({ onNav, onOpenGraph, onToggleSidebar, sidebarOpen }) {
  return React.createElement("nav", { className: "topnav" },
    React.createElement("div", { className: "topnav-inner" },
      React.createElement("div", { style: { display: "flex", alignItems: "center", gap: 10 } },
        React.createElement("button", {
          className: "mobile-toggle",
          onClick: onToggleSidebar,
          "aria-label": "Toggle navigation menu",
          "aria-expanded": !!sidebarOpen,
          "aria-controls": "site-sidebar",
        }, React.createElement(Icon, { name: "menu" })),
        React.createElement("a", {
          className: "brand",
          href: "#/home",
          onClick: e => { e.preventDefault(); onNav("home"); },
        },
          React.createElement("span", { className: "brand-mark" }),
          React.createElement("span", { className: "brand-name" }, "F1 StratLab"),
          React.createElement("span", { className: "brand-tag" }, "docs"),
        ),
      ),
      React.createElement(Search, { onPick: onNav }),
      React.createElement("div", { className: "nav-right" },
        React.createElement(ThemeToggle, null),
        React.createElement("button", {
          className: "nav-pill",
          onClick: onOpenGraph,
          title: "Open graph view",
          "aria-label": "Open graph view",
        },
          React.createElement(Icon, { name: "graph" }),
          React.createElement("span", null, "Graph"),
        ),
        React.createElement("a", {
          className: "nav-pill",
          href: "https://f1stratlab.com/",
          target: "_blank", rel: "noopener noreferrer",
          title: "Public landing site",
          "aria-label": "Landing site (opens in new tab)",
        },
          React.createElement(Icon, { name: "globe" }),
          React.createElement("span", null, "Landing"),
        ),
        React.createElement("a", {
          className: "nav-pill",
          href: "https://deepwiki.com/VforVitorio/F1-StratLab",
          target: "_blank", rel: "noopener noreferrer",
          "aria-label": "DeepWiki (opens in new tab)",
        },
          React.createElement(Icon, { name: "book" }),
          React.createElement("span", null, "DeepWiki"),
        ),
        React.createElement("a", {
          className: "nav-pill primary",
          href: "https://github.com/VforVitorio/F1-StratLab",
          target: "_blank", rel: "noopener noreferrer",
          "aria-label": "GitHub repository (opens in new tab)",
        },
          React.createElement(Icon, { name: "github" }),
          React.createElement("span", null, "GitHub"),
        ),
      ),
    ),
  );
}
window.TopNav = TopNav;

// ---------- Sidebar ----------
function Sidebar({ activeSlug, onNav, open, onClose }) {
  const grouped = useMemo(() => {
    const g = {};
    for (const p of window.PAGES) {
      (g[p.section] = g[p.section] || []).push(p);
    }
    return g;
  }, []);

  // Lock body scroll while the off-canvas sidebar is open on small screens.
  useEffect(() => {
    if (!open) return;
    const prev = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    function onKey(e) { if (e.key === "Escape" && onClose) onClose(); }
    window.addEventListener("keydown", onKey);
    return () => {
      document.body.style.overflow = prev;
      window.removeEventListener("keydown", onKey);
    };
  }, [open, onClose]);

  return React.createElement(React.Fragment, null,
    open && React.createElement("div", {
      className: "sidebar-backdrop",
      onClick: () => onClose && onClose(),
      "aria-hidden": "true",
    }),
    React.createElement("aside", { id: "site-sidebar", "aria-label": "Site navigation", className: "sidebar" + (open ? " open" : "") },
      window.SECTIONS.map(sec =>
        React.createElement("div", { key: sec, className: "sidebar-section" },
          React.createElement("div", { className: "sidebar-section-title" }, sec),
          (grouped[sec] || []).map(p =>
            React.createElement("a", {
              key: p.slug,
              className: "sidebar-link" + (activeSlug === p.slug ? " active" : ""),
              href: "#/" + p.slug,
              onClick: e => { e.preventDefault(); onNav(p.slug); if (onClose) onClose(); },
            },
              React.createElement("span", null, p.title),
            )
          )
        )
      ),
      React.createElement("div", { className: "sidebar-footer" },
        React.createElement("strong", null, "Companion resources"),
        React.createElement("div", { style: { marginTop: 6 } },
          React.createElement("a", { href: "https://f1stratlab.com/", target: "_blank", rel: "noopener noreferrer" }, "f1stratlab.com"),
          " · ",
          React.createElement("a", { href: "https://deepwiki.com/VforVitorio/F1-StratLab", target: "_blank", rel: "noopener noreferrer" }, "DeepWiki"),
        ),
        React.createElement("div", { style: { marginTop: 6 } },
          React.createElement("a", { href: "https://huggingface.co/datasets/VforVitorio/f1-strategy-dataset", target: "_blank", rel: "noopener noreferrer" }, "HF dataset"),
        ),
      ),
    ),
  );
}
window.Sidebar = Sidebar;

// ---------- TOC (right) ----------
function TOC({ items }) {
  const [active, setActive] = useState("");
  useEffect(() => {
    if (!items || items.length === 0) return;
    const headings = items
      .map(it => document.getElementById(it.id))
      .filter(Boolean);

    function onScroll() {
      const top = window.scrollY + 100;
      let cur = "";
      for (const h of headings) {
        if (h.offsetTop <= top) cur = h.id; else break;
      }
      setActive(cur);
    }
    window.addEventListener("scroll", onScroll, { passive: true });
    onScroll();
    return () => window.removeEventListener("scroll", onScroll);
  }, [items]);

  if (!items || items.length === 0) return null;

  return React.createElement("aside", { className: "toc", "aria-label": "On this page" },
    React.createElement("div", { className: "toc-title" }, "On this page"),
    React.createElement("ul", { className: "toc-list" },
      items.map(it =>
        React.createElement("li", { key: it.id },
          React.createElement("a", {
            className: "toc-item" + (it.level === 3 ? " h3" : "") + (active === it.id ? " active" : ""),
            href: "#" + it.id,
            onClick: e => {
              e.preventDefault();
              const el = document.getElementById(it.id);
              if (el) {
                const y = el.getBoundingClientRect().top + window.scrollY - 80;
                window.scrollTo({ top: y, behavior: "smooth" });
              }
            },
          }, it.text),
        )
      )
    )
  );
}
window.TOC = TOC;

// ---------- Tag chips (article header) ----------
function TagChips({ page, onOpenGraph }) {
  if (!page.tags || page.tags.length === 0) return null;
  return React.createElement("div", { className: "tag-chips" },
    page.tags.map(t =>
      React.createElement("button", {
        key: t,
        className: "tag-chip",
        onClick: () => onOpenGraph && onOpenGraph(t),
        title: "Show this tag in the graph",
      }, "#", (window.TAG_LABELS[t] || t))
    )
  );
}
window.TagChips = TagChips;

// ---------- Breadcrumb ----------
function Breadcrumb({ page, onNav }) {
  return React.createElement("div", { className: "breadcrumb" },
    React.createElement("a", {
      href: "#/home",
      onClick: e => { e.preventDefault(); onNav("home"); },
    }, "docs"),
    React.createElement("span", { className: "breadcrumb-sep" }, "/"),
    React.createElement("span", null, page.section),
    React.createElement("span", { className: "breadcrumb-sep" }, "/"),
    React.createElement("span", { style: { color: "var(--accent-text)" } }, page.title),
  );
}
window.Breadcrumb = Breadcrumb;

// ---------- Page footer (prev/next) ----------
function PageFooter({ slug, onNav }) {
  const idx = window.PAGES.findIndex(p => p.slug === slug);
  const prev = idx > 0 ? window.PAGES[idx - 1] : null;
  const next = idx >= 0 && idx < window.PAGES.length - 1 ? window.PAGES[idx + 1] : null;
  if (!prev && !next) return null;
  return React.createElement("div", { className: "page-footer" },
    prev
      ? React.createElement("a", {
          className: "page-footer-link",
          href: "#/" + prev.slug,
          onClick: e => { e.preventDefault(); onNav(prev.slug); },
        },
          React.createElement("div", { className: "page-footer-label" }, "← Previous"),
          React.createElement("div", { className: "page-footer-title" }, prev.title),
        )
      : React.createElement("div", { className: "page-footer-link page-footer-empty" }),
    next
      ? React.createElement("a", {
          className: "page-footer-link next",
          href: "#/" + next.slug,
          onClick: e => { e.preventDefault(); onNav(next.slug); },
        },
          React.createElement("div", { className: "page-footer-label" }, "Next →"),
          React.createElement("div", { className: "page-footer-title" }, next.title),
        )
      : React.createElement("div", { className: "page-footer-link page-footer-empty" }),
  );
}
window.PageFooter = PageFooter;

// ---------- Docs footer ----------
function DocsFooter({ onNav }) {
  return React.createElement(React.Fragment, null,
    React.createElement("footer", { className: "docs-footer" },
      React.createElement("div", { className: "docs-footer-inner" },
        React.createElement("div", null,
          React.createElement("div", { className: "brand" },
            React.createElement("span", { className: "brand-mark" }),
            React.createElement("span", { className: "brand-name" }, "F1 StratLab"),
            React.createElement("span", { className: "brand-tag" }, "docs"),
          ),
          React.createElement("p", { className: "docs-footer-brand-copy" },
            "Open-source multi-agent AI for real-time F1 race strategy. Built end-to-end as a Final-Degree Project on Intelligent Systems Engineering. Apache-2.0.",
          ),
        ),
        React.createElement("div", { className: "docs-footer-col" },
          React.createElement("div", { className: "docs-footer-col-title" }, "Docs"),
          React.createElement("a", { href: "#/home", onClick: e => { e.preventDefault(); onNav("home"); } }, "Welcome"),
          React.createElement("a", { href: "#/getting-started", onClick: e => { e.preventDefault(); onNav("getting-started"); } }, "Getting started"),
          React.createElement("a", { href: "#/architecture", onClick: e => { e.preventDefault(); onNav("architecture"); } }, "Architecture"),
          React.createElement("a", { href: "#/agents-api", onClick: e => { e.preventDefault(); onNav("agents-api"); } }, "Agents API"),
        ),
        React.createElement("div", { className: "docs-footer-col" },
          React.createElement("div", { className: "docs-footer-col-title" }, "Project"),
          React.createElement("a", { href: "https://github.com/VforVitorio/F1-StratLab", target: "_blank", rel: "noopener noreferrer" }, "GitHub"),
          React.createElement("a", { href: "https://deepwiki.com/VforVitorio/F1-StratLab", target: "_blank", rel: "noopener noreferrer" }, "DeepWiki"),
          React.createElement("a", { href: "https://f1stratlab.com/", target: "_blank", rel: "noopener noreferrer" }, "Landing site"),
          React.createElement("a", { href: "https://huggingface.co/datasets/VforVitorio/f1-strategy-dataset", target: "_blank", rel: "noopener noreferrer" }, "HF dataset"),
        ),
        React.createElement("div", { className: "docs-footer-col" },
          React.createElement("div", { className: "docs-footer-col-title" }, "Release"),
          React.createElement("a", { href: "https://github.com/VforVitorio/F1-StratLab/releases", target: "_blank", rel: "noopener noreferrer" }, "Releases"),
          React.createElement("a", { href: "https://github.com/VforVitorio/F1-StratLab/blob/main/CHANGELOG.md", target: "_blank", rel: "noopener noreferrer" }, "Changelog"),
          React.createElement("a", { href: "https://github.com/VforVitorio/F1-StratLab/blob/main/LICENSE", target: "_blank", rel: "noopener noreferrer" }, "Apache-2.0 License"),
        ),
        React.createElement("div", { className: "docs-footer-col" },
          React.createElement("div", { className: "docs-footer-col-title" }, "Connect"),
          React.createElement("a", { href: "#/meet-the-author", onClick: e => { e.preventDefault(); onNav("meet-the-author"); } }, "Meet the author"),
          React.createElement("a", { href: "https://github.com/VforVitorio", target: "_blank", rel: "noopener noreferrer" }, "GitHub profile"),
          React.createElement("a", { href: "https://www.linkedin.com/in/victorvegasobral/", target: "_blank", rel: "noopener noreferrer" }, "LinkedIn"),
          React.createElement("a", { href: "https://huggingface.co/datasets/VforVitorio/f1-strategy-dataset", target: "_blank", rel: "noopener noreferrer" }, "HF dataset"),
          React.createElement("a", { href: "https://victorvegasobral.com", target: "_blank", rel: "noopener noreferrer" }, "Portfolio"),
        ),
      ),
      React.createElement("div", { className: "docs-footer-legal" },
        React.createElement("span", null, "© 2026 · VforVitorio · F1 StratLab"),
        React.createElement("span", null, "v2.6.1 · React · deployed to gh-pages by GitHub Actions"),
      ),
    ),
  );
}
window.DocsFooter = DocsFooter;
