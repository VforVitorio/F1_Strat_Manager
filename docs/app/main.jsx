// =========================================================
// App entry — router, layout, top-level state
// =========================================================

const { useState, useEffect, useCallback } = React;

function parseHash() {
  const h = location.hash;
  // Special: graph
  if (h === "#/graph") return { slug: null, openGraph: true, heading: null };
  const m = h.match(/^#\/([a-z0-9-]+)(?:#(.+))?$/);
  if (m) return { slug: m[1], openGraph: false, heading: m[2] || null };
  // No hash route: fall back to the real path so prerendered URLs like
  // /architecture/ render the right page on a direct visit (not home).
  const p = location.pathname.match(/^\/([a-z0-9-]+)\/?$/);
  if (p && window.PAGE_MAP && window.PAGE_MAP[p[1]]) {
    return { slug: p[1], openGraph: false, heading: null };
  }
  return { slug: "home", openGraph: false, heading: null };
}

function App() {
  const [route, setRoute] = useState(parseHash());
  const [tocItems, setTocItems] = useState([]);
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [graphOpen, setGraphOpen] = useState(false);
  const [graphTag, setGraphTag] = useState(null);
  const [pagesReady, setPagesReady] = useState(false);

  const openGraph = useCallback((tag) => {
    setGraphTag(typeof tag === "string" ? tag : null);
    setGraphOpen(true);
  }, []);

  // Hash listener
  useEffect(() => {
    function onHash() {
      const r = parseHash();
      setRoute(r);
      if (r.openGraph) setGraphOpen(true);
    }
    window.addEventListener("hashchange", onHash);
    onHash();
    return () => window.removeEventListener("hashchange", onHash);
  }, []);

  // Eager-load all pages so search has content & the graph has real page edges
  useEffect(() => {
    window.loadAllPages().then(() => setPagesReady(true));
  }, []);

  // Nav helper
  const navTo = useCallback((slug) => {
    if (slug === "__graph__") {
      setGraphOpen(true);
      return;
    }
    if (!window.PAGE_MAP[slug]) return;
    if (location.hash !== "#/" + slug) {
      location.hash = "/" + slug;
    } else {
      // already there — just scroll up
      window.scrollTo({ top: 0, behavior: "auto" });
    }
    setSidebarOpen(false);
  }, []);

  // Global click delegate — intercept internal #/... links so React doesn't blow away scroll state
  useEffect(() => {
    function onClick(e) {
      const a = e.target.closest && e.target.closest("a[data-internal]");
      if (!a) return;
      const href = a.getAttribute("href");
      if (!href || !href.startsWith("#/")) return;
      // Allow new-tab modifier
      if (e.metaKey || e.ctrlKey || e.shiftKey || e.button === 1) return;
      e.preventDefault();
      const m = href.match(/^#\/([a-z0-9-]+)(?:#(.+))?$/);
      if (!m) return;
      const slug = m[1];
      const heading = m[2];
      if (slug && window.PAGE_MAP[slug]) {
        if (heading) {
          location.hash = "/" + slug + "#" + heading;
        } else {
          location.hash = "/" + slug;
        }
      }
    }
    document.addEventListener("click", onClick);
    return () => document.removeEventListener("click", onClick);
  }, []);

  const slug = route.slug || "home";
  const page = window.PAGE_MAP[slug] || window.PAGE_MAP.home;
  const isHome = slug === "home";

  // Update document title
  useEffect(() => {
    document.title = isHome
      ? "F1 StratLab · Documentation"
      : page.title + " · F1 StratLab docs";
  }, [slug]);

  return React.createElement(React.Fragment, null,
    React.createElement(window.TopNav, {
      onNav: navTo,
      onOpenGraph: () => openGraph(),
      onToggleSidebar: () => setSidebarOpen(o => !o),
    }),
    React.createElement("div", { className: "shell" + (isHome ? " shell-no-toc" : "") },
      React.createElement(window.Sidebar, {
        activeSlug: slug,
        onNav: navTo,
        open: sidebarOpen,
        onClose: () => setSidebarOpen(false),
      }),
      React.createElement("main", { className: "content" + (isHome ? " content-wide" : "") },
        isHome
          ? React.createElement(window.HomePage, {
              onNav: navTo,
              onOpenGraph: () => openGraph(),
              pagesReady,
            })
          : React.createElement(React.Fragment, null,
              React.createElement(window.Breadcrumb, { page, onNav: navTo }),
              React.createElement(window.TagChips, { page, onOpenGraph: openGraph }),
              React.createElement(window.MarkdownArticle, {
                slug,
                onTOC: setTocItems,
              }),
              React.createElement(window.PageFooter, {
                slug,
                onNav: navTo,
              }),
            ),
      ),
      !isHome && React.createElement(window.TOC, { items: tocItems }),
    ),
    React.createElement(window.DocsFooter, { onNav: navTo }),
    graphOpen && React.createElement(window.GraphView, {
      mode: "overlay",
      currentSlug: slug,
      initialTag: graphTag,
      onClose: () => { setGraphOpen(false); setGraphTag(null); },
      onNav: (s) => { setGraphOpen(false); setGraphTag(null); navTo(s); },
    }),
  );
}

const root = ReactDOM.createRoot(document.getElementById("root"));
root.render(React.createElement(App));
