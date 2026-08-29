// =========================================================
// Markdown rendering
//   - marked.js for parsing
//   - mermaid for code blocks lang=mermaid
//   - Prism (loaded in index.html) for syntax highlighting
// =========================================================

// var (not const): these app scripts load in shared global scope without a
// module system, so the React-hook destructuring repeats across files. var
// tolerates the redeclaration; const would throw once Babel's transform is gone.
var { useEffect, useRef, useState } = React;

// Configure marked
function configureMarked() {
  if (!window.marked) return;
  const renderer = new marked.Renderer();

  // Code blocks — wrap in our chrome
  renderer.code = function (code, language) {
    if (language === "mermaid") {
      const id = "mmd-" + Math.random().toString(36).slice(2, 9);
      return `<div class="mermaid-block" data-mermaid-id="${id}"><div class="mermaid-src" style="display:none">${escapeHTML(code)}</div></div>`;
    }
    const lang = (language || "plain").toLowerCase();
    const langLabel = lang === "plain" || lang === "text" ? "" : lang;
    const escaped = highlightCode(code, lang);
    return `<div class="code-block">
      <div class="code-block-chrome">
        <div class="code-block-dots">
          <span class="code-block-dot"></span><span class="code-block-dot"></span><span class="code-block-dot"></span>
        </div>
        <div class="code-block-lang">${langLabel}</div>
        <button class="code-copy" type="button" data-copy>Copy</button>
      </div>
      <pre><code class="language-${lang}">${escaped}</code></pre>
    </div>`;
  };

  // Internal anchors for headings — h1 is the page title and gets no anchor.
  renderer.heading = function (text, level, raw) {
    const id = slugify(raw);
    if (level === 1) return `<h1 id="${id}">${text}</h1>`;
    return `<h${level} id="${id}">${text}<a class="heading-anchor" href="#${currentHashBase()}#${id}" data-heading-anchor="${id}" aria-label="link to this section">#</a></h${level}>`;
  };

  // Tables already produced by marked; just pass through.
  // Links: rewrite #/slug to keep them inside our router; external links open new tab.
  renderer.link = function (href, title, text) {
    const titleAttr = title ? ` title="${title}"` : "";
    if (!href) return `<a${titleAttr}>${text}</a>`;
    if (href.startsWith("#/")) {
      return `<a href="${href}"${titleAttr} data-internal>${text}</a>`;
    }
    if (href.startsWith("http") && !href.includes(location.host)) {
      return `<a href="${href}"${titleAttr} target="_blank" rel="noopener noreferrer">${text} <svg aria-hidden="true" focusable="false" style="display:inline;vertical-align:-1px;width:10px;height:10px;opacity:0.6" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><path d="M7 17L17 7M17 7H7M17 7v10"/></svg><span class="sr-only"> (opens in new tab)</span></a>`;
    }
    return `<a href="${href}"${titleAttr}>${text}</a>`;
  };

  marked.setOptions({ renderer, gfm: true, breaks: false });
}

function currentHashBase() {
  // returns the current page slug portion of the hash, e.g. "/architecture"
  const h = location.hash;
  const m = h.match(/^#\/([a-z0-9-]+)/);
  return m ? "/" + m[1] : "/home";
}

function escapeHTML(s) {
  return s.replace(/[&<>"']/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;","\"":"&quot;","'":"&#39;"}[c]));
}

function slugify(s) {
  return s.toLowerCase()
    .replace(/<[^>]+>/g, "")
    .replace(/[^a-z0-9\s-]/g, "")
    .replace(/\s+/g, "-")
    .replace(/-+/g, "-")
    .replace(/^-+|-+$/g, "")
    .slice(0, 80);
}

function highlightCode(code, lang) {
  const escaped = escapeHTML(code);
  if (window.Prism && Prism.languages[lang]) {
    try {
      return Prism.highlight(code, Prism.languages[lang], lang);
    } catch (e) {
      return escaped;
    }
  }
  return escaped;
}

// ---------- Mermaid ----------
// Mermaid bakes its colours into the SVG at render time, so unlike everything else on this page
// it cannot follow a token swap. Three things have to happen on a theme change, and missing any
// one of them leaves diagrams in the previous theme:
//
//   1. mermaid.initialize() has to run again with the new variables. It is a once-per-page call
//      by design, so the latch below records WHICH theme it was initialised for rather than a
//      plain boolean.
//   2. Every block's `rendered` flag has to be cleared, or the render loop skips it.
//   3. The source has to still exist. It does not: the loop replaces the block's innerHTML with
//      the finished SVG, which deletes the .mermaid-src child it read the source from. So the
//      source is cached onto the element the first time it is seen.
const MERMAID_PALETTE = {
  dark: {
    darkMode: true,
    background: "transparent",
    primaryColor: "#1e2139",
    primaryTextColor: "#ffffff",
    primaryBorderColor: "#a29bfe",
    lineColor: "#a29bfe",
    secondaryColor: "#23234a",
    tertiaryColor: "#111827",
    mainBkg: "#1e2139",
    secondBkg: "#23234a",
    textColor: "#e9e7ff",
    nodeBorder: "rgba(162,155,254,0.5)",
    clusterBkg: "rgba(108,92,231,0.08)",
    clusterBorder: "rgba(108,92,231,0.32)",
    edgeLabelBackground: "#111827",
    labelBoxBkgColor: "#111827",
    labelBoxBorderColor: "#a29bfe",
    labelTextColor: "#ffffff",
    actorBkg: "#1e2139",
    actorBorder: "#a29bfe",
    actorTextColor: "#ffffff",
    actorLineColor: "#a29bfe",
    noteBkgColor: "#23234a",
    noteBorderColor: "#a29bfe",
    noteTextColor: "#ffffff",
    sequenceNumberColor: "#0c0d14",
    activationBkgColor: "#6c5ce7",
    activationBorderColor: "#a29bfe",
  },
  // The light values mirror the light token ramp rather than lightening the dark ones: the node
  // fill is the card colour, text is the ink colour, and every line and border takes purple-700,
  // because purple-300 is 2.23:1 on a light ground and would leave the edges barely visible.
  light: {
    darkMode: false,
    background: "transparent",
    primaryColor: "#ffffff",
    primaryTextColor: "#14121f",
    primaryBorderColor: "#5a48d4",
    lineColor: "#5a48d4",
    secondaryColor: "#eceaf9",
    tertiaryColor: "#e9eaf4",
    mainBkg: "#ffffff",
    secondBkg: "#eceaf9",
    textColor: "#14121f",
    nodeBorder: "rgba(90,72,212,0.55)",
    clusterBkg: "rgba(108,92,231,0.06)",
    clusterBorder: "rgba(108,92,231,0.28)",
    edgeLabelBackground: "#f5f5fa",
    labelBoxBkgColor: "#ffffff",
    labelBoxBorderColor: "#5a48d4",
    labelTextColor: "#14121f",
    actorBkg: "#ffffff",
    actorBorder: "#5a48d4",
    actorTextColor: "#14121f",
    actorLineColor: "#5a48d4",
    noteBkgColor: "#eceaf9",
    noteBorderColor: "#5a48d4",
    noteTextColor: "#14121f",
    sequenceNumberColor: "#ffffff",
    activationBkgColor: "#6c5ce7",
    activationBorderColor: "#5a48d4",
  },
};

/** Which theme mermaid was last initialised for, or null before the first call. */
let mermaidTheme = null;

function currentTheme() {
  return document.documentElement.getAttribute("data-theme") === "light" ? "light" : "dark";
}

function ensureMermaid() {
  const theme = currentTheme();
  if (mermaidTheme === theme || !window.mermaid) return;
  mermaid.initialize({
    startOnLoad: false,
    theme: "base",
    securityLevel: "loose",
    fontFamily: "Inter, system-ui, sans-serif",
    themeVariables: MERMAID_PALETTE[theme],
  });
  mermaidTheme = theme;
}

async function renderMermaidBlocks(root) {
  if (!window.mermaid) return;
  ensureMermaid();
  const blocks = root.querySelectorAll(".mermaid-block");
  for (const b of blocks) {
    if (b.dataset.rendered === "1") continue;
    // The source is cached on first sight because the assignment below deletes the element it
    // came from. Without this, a re-render after a theme change finds nothing to render.
    const src = b.querySelector(".mermaid-src");
    if (src && b.dataset.mermaidSrc === undefined) b.dataset.mermaidSrc = src.textContent;
    const code = b.dataset.mermaidSrc;
    if (code === undefined) continue;
    const id = b.dataset.mermaidId || ("m" + Math.random().toString(36).slice(2, 9));
    try {
      const { svg } = await mermaid.render(id, code);
      b.innerHTML = svg;
      b.dataset.rendered = "1";
    } catch (e) {
      console.warn("mermaid render error", e);
      b.innerHTML = `<div class="mermaid-error">mermaid render error: ${escapeHTML(String(e.message || e))}</div>`;
    }
  }
}


/**
 * Re-render every diagram on the page in the new theme.
 *
 * Mermaid's own state has to be reset first (`ensureMermaid` reads the attribute and reinitialises
 * because the recorded theme no longer matches), then each block is unmarked so the render loop
 * stops skipping it.
 */
window.addEventListener("f1sl:themechange", function () {
  document.querySelectorAll(".mermaid-block").forEach(function (b) {
    if (b.dataset.mermaidSrc !== undefined) delete b.dataset.rendered;
  });
  renderMermaidBlocks(document);
});

// React component — renders markdown for a given slug
function MarkdownArticle({ slug, onTOC }) {
  const [html, setHtml] = useState("");
  const [loading, setLoading] = useState(true);
  const containerRef = useRef(null);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    window.loadPage(slug).then(md => {
      if (cancelled) return;
      if (md == null) {
        setHtml('<p style="color:var(--danger)">Page not found.</p>');
        setLoading(false);
        return;
      }
      configureMarked();
      const out = marked.parse(md);
      setHtml(out);
      setLoading(false);
    });
    return () => { cancelled = true; };
  }, [slug]);

  // After HTML is set: render mermaid, attach copy buttons, internal links, build TOC.
  useEffect(() => {
    if (!containerRef.current || !html) return;
    const root = containerRef.current;
    renderMermaidBlocks(root);

    // Reduced motion: the demo videos autoplay/loop, so for users who opt out
    // of motion we pause them and expose controls (WCAG 2.2.2 / 2.3.3).
    if (window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches) {
      root.querySelectorAll("video[autoplay]").forEach(v => {
        v.removeAttribute("autoplay");
        v.setAttribute("controls", "");
        v.pause();
      });
    }

    // Copy buttons
    root.querySelectorAll("[data-copy]").forEach(btn => {
      btn.addEventListener("click", () => {
        const pre = btn.closest(".code-block")?.querySelector("pre code");
        if (!pre) return;
        navigator.clipboard.writeText(pre.textContent || "").then(() => {
          btn.textContent = "Copied";
          setTimeout(() => { btn.textContent = "Copy"; }, 1400);
        });
      });
    });

    // Internal links — handled via global delegate in main.jsx; nothing to do here.

    // Build TOC: collect h2/h3
    if (onTOC) {
      const items = [];
      root.querySelectorAll("h2, h3").forEach(h => {
        if (!h.id) return;
        items.push({
          id: h.id,
          text: h.textContent.replace(/#$/, "").trim(),
          level: h.tagName === "H2" ? 2 : 3,
        });
      });
      onTOC(items);
    }

    // Scroll to in-page anchor if present after slug
    const m = location.hash.match(/^#\/[a-z0-9-]+#(.+)$/);
    if (m) {
      const target = root.querySelector("#" + CSS.escape(m[1]));
      if (target) {
        setTimeout(() => target.scrollIntoView({ block: "start", behavior: "auto" }), 50);
      }
    } else {
      window.scrollTo({ top: 0, behavior: "auto" });
    }
  }, [html]);

  if (loading) {
    return React.createElement("div", { className: "article-enter" },
      React.createElement("div", { className: "skel", style: { height: 36, width: "60%", marginBottom: 18 } }),
      React.createElement("div", { className: "skel", style: { height: 14, width: "85%", marginBottom: 12 } }),
      React.createElement("div", { className: "skel", style: { height: 14, width: "70%", marginBottom: 12 } }),
      React.createElement("div", { className: "skel", style: { height: 14, width: "80%", marginBottom: 12 } }),
    );
  }

  return React.createElement("div", {
    className: "article article-enter",
    ref: containerRef,
    dangerouslySetInnerHTML: { __html: html },
  });
}

window.MarkdownArticle = MarkdownArticle;
window.mdHelpers = { renderMermaidBlocks, configureMarked, slugify };
