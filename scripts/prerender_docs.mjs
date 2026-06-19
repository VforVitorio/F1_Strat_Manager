// =========================================================
// Deploy-time prerender for the docs site.
//
// WHY: the docs site is a React + in-browser-Babel SPA with hash routing, so a
// non-JS crawler (every major AI crawler: GPTBot, ClaudeBot, PerplexityBot…)
// receives only an empty <div id="root"></div>. The page content already exists
// as static Markdown under docs/pages/. This script renders each page to real,
// crawlable HTML at a real URL (/<slug>/index.html) with per-page <head> meta and
// JSON-LD, while keeping the SPA for interactive navigation (it replaces the
// prerendered content once JS runs). It also emits a real-URL sitemap, an
// llms-full.txt, and a 404.html SPA fallback.
//
// USAGE: node scripts/prerender_docs.mjs <docsDir> <outDir>
//   docsDir = source docs/ (read nav.js + pages/*.md + index.html shell)
//   outDir  = staged site (already a copy of docs/); generated files land here
//
// --- WHERE TO CHANGE IF THINGS MOVE ---
//   * Page list / descriptions: docs/app/nav.js (window.PAGES) — single source of truth.
//   * Global entity JSON-LD: docs/index.html <head> (this script adds per-page nodes).
//   * SPA path-awareness: docs/app/main.jsx parseRoute() must accept /<slug>/ paths.
// =========================================================

import { readFileSync, writeFileSync, mkdirSync, existsSync } from "node:fs";
import { join } from "node:path";
import { execFileSync } from "node:child_process";
import { marked } from "marked";

const SITE = "https://docs.f1stratlab.com";

const [, , docsDir = "docs", outDir = "_site"] = process.argv;

// ---- nav.js → PAGES -------------------------------------------------------
// nav.js assigns everything onto `window.*`, so evaluate it against a shim and
// read the result back. Functions inside reference fetch/location but are never
// called here, so the shim needs no DOM.
function loadPages(navPath) {
  const code = readFileSync(navPath, "utf8");
  const window = {};
  new Function("window", code)(window);
  if (!Array.isArray(window.PAGES)) throw new Error("nav.js did not define window.PAGES");
  return window.PAGES;
}

// ---- Markdown → HTML ------------------------------------------------------
marked.setOptions({ gfm: true, breaks: false });

function renderMarkdown(md) {
  return marked.parse(md);
}

// Rewrite in-app hash links (#/slug, #/slug#heading) to real crawlable paths
// (/slug/, /slug/#heading) so the static internal link graph points at real
// URLs. Only rewrites slugs that are real pages; leaves #/graph and anchors alone.
function rewriteInternalLinks(html, validSlugs) {
  return html.replace(/href="#\/([a-z0-9-]+)(#[^"]*)?"/g, (m, slug, hash) =>
    validSlugs.has(slug) ? `href="/${slug}/${hash || ""}"` : m
  );
}

// ---- per-page <head> ------------------------------------------------------
function canonicalFor(slug) {
  return slug === "home" ? `${SITE}/` : `${SITE}/${slug}/`;
}

function titleFor(page) {
  return page.slug === "home"
    ? "F1 StratLab · Documentation"
    : `${page.title} · F1 StratLab docs`;
}

// TechArticle + BreadcrumbList for the page. The global WebSite/SoftwareSourceCode/
// Person graph already ships in the shell <head>; these reference it by @id.
function perPageJsonLd(page) {
  const url = canonicalFor(page.slug);
  const techArticle = {
    "@context": "https://schema.org",
    "@type": "TechArticle",
    headline: page.title,
    name: page.title,
    description: page.description,
    url,
    inLanguage: "en",
    isPartOf: { "@id": `${SITE}/#website` },
    about: { "@id": `${SITE}/#software` },
    author: { "@id": `${SITE}/#author` },
    speakable: {
      "@type": "SpeakableSpecification",
      cssSelector: ["h1", ".prerender-content > p"],
    },
  };
  const breadcrumb = {
    "@context": "https://schema.org",
    "@type": "BreadcrumbList",
    itemListElement: [
      { "@type": "ListItem", position: 1, name: "Home", item: `${SITE}/` },
      { "@type": "ListItem", position: 2, name: page.title, item: url },
    ],
  };
  return (
    `<script type="application/ld+json">\n${JSON.stringify(techArticle, null, 2)}\n</script>\n` +
    `<script type="application/ld+json">\n${JSON.stringify(breadcrumb, null, 2)}\n</script>\n`
  );
}

// ---- shell templating -----------------------------------------------------
// Replace the shell's global meta with per-page values and inject content. The
// SPA replaces #root once JS runs; crawlers keep the prerendered HTML.
function buildPage(shell, page, contentHtml) {
  const url = canonicalFor(page.slug);
  const title = titleFor(page);
  const desc = page.description || "";

  let html = shell
    .replace(/<title>[\s\S]*?<\/title>/, `<title>${escapeHtml(title)}</title>`)
    .replace(/(<link rel="canonical" href=")[^"]*(")/, `$1${url}$2`)
    .replace(/(<meta property="og:url" content=")[^"]*(")/, `$1${url}$2`)
    .replace(/(<meta name="twitter:url" content=")[^"]*(")/, `$1${url}$2`)
    .replace(/(<meta property="og:title" content=")[^"]*(")/, `$1${escapeAttr(title)}$2`)
    .replace(/(<meta name="twitter:title" content=")[^"]*(")/, `$1${escapeAttr(title)}$2`);

  if (desc) {
    html = html
      .replace(/(<meta name="description" content=")[^"]*(")/, `$1${escapeAttr(desc)}$2`)
      .replace(/(<meta property="og:description" content=")[^"]*(")/, `$1${escapeAttr(desc)}$2`)
      .replace(/(<meta name="twitter:description" content=")[^"]*(")/, `$1${escapeAttr(desc)}$2`);
  }

  // per-page JSON-LD just before </head>
  html = html.replace("</head>", `${perPageJsonLd(page)}</head>`);

  // inject prerendered content into the root container (SPA overwrites it on load)
  const block = `<div id="root"><main class="prerender-content">\n${contentHtml}\n</main></div>`;
  html = html.replace(/<div id="root">\s*<\/div>/, block);

  return absolutizeAssets(html);
}

// The shell references CSS/JS/favicon with relative paths (styles/, app/, assets/),
// which break when the page is served from a subpath like /architecture/. Make
// them root-absolute so every prerendered route loads the same assets.
function absolutizeAssets(html) {
  return html.replace(/(href|src)="(styles\/|assets\/|app\/)/g, '$1="/$2');
}

function escapeHtml(s) {
  return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}
function escapeAttr(s) {
  return escapeHtml(s).replace(/"/g, "&quot;");
}

// ---- sitemap / llms-full / 404 -------------------------------------------
function gitLastmod(filePath) {
  try {
    const out = execFileSync("git", ["log", "-1", "--format=%cs", "--", filePath], {
      encoding: "utf8",
    }).trim();
    return out || todayIso();
  } catch {
    return todayIso();
  }
}
function todayIso() {
  return new Date().toISOString().slice(0, 10);
}

function buildSitemap(pages, docsDir) {
  const urls = pages
    .map((p) => {
      const lastmod = gitLastmod(join(docsDir, p.file));
      const priority = p.slug === "home" ? "1.0" : "0.8";
      return `  <url>\n    <loc>${canonicalFor(p.slug)}</loc>\n    <lastmod>${lastmod}</lastmod>\n    <changefreq>monthly</changefreq>\n    <priority>${priority}</priority>\n  </url>`;
    })
    .join("\n");
  return `<?xml version="1.0" encoding="UTF-8"?>\n<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n${urls}\n</urlset>\n`;
}

function buildLlmsFull(pages, docsDir) {
  const header =
    "# F1 StratLab — full documentation\n\n" +
    "> Concatenated body of every documentation page, for AI systems that want the complete text in one fetch.\n\n";
  const bodies = pages
    .map((p) => {
      const md = readFileSync(join(docsDir, p.file), "utf8");
      return `\n\n---\n\n# ${p.title} (${canonicalFor(p.slug)})\n\n${md.trim()}`;
    })
    .join("\n");
  return header + bodies + "\n";
}

// ---- main -----------------------------------------------------------------
function run() {
  const pages = loadPages(join(docsDir, "app", "nav.js"));
  const shell = readFileSync(join(docsDir, "index.html"), "utf8");
  const validSlugs = new Set(pages.map((p) => p.slug));

  let written = 0;
  for (const page of pages) {
    const mdPath = join(docsDir, page.file);
    if (!existsSync(mdPath)) {
      console.warn(`skip ${page.slug}: missing ${page.file}`);
      continue;
    }
    const md = readFileSync(mdPath, "utf8");
    const contentHtml = rewriteInternalLinks(renderMarkdown(md), validSlugs);
    const pageHtml = buildPage(shell, page, contentHtml);

    const dir = page.slug === "home" ? outDir : join(outDir, page.slug);
    mkdirSync(dir, { recursive: true });
    writeFileSync(join(dir, "index.html"), pageHtml, "utf8");

    // sanity: the prerendered HTML must actually carry body content
    if (!/class="prerender-content"/.test(pageHtml) || contentHtml.length < 20) {
      throw new Error(`prerender produced empty content for ${page.slug}`);
    }
    written++;
  }

  writeFileSync(join(outDir, "sitemap.xml"), buildSitemap(pages, docsDir), "utf8");
  writeFileSync(join(outDir, "llms-full.txt"), buildLlmsFull(pages, docsDir), "utf8");
  // 404 fallback: serve the SPA shell (with absolute assets) so unknown paths still boot the app.
  writeFileSync(join(outDir, "404.html"), absolutizeAssets(shell), "utf8");

  console.log(`prerendered ${written}/${pages.length} pages → ${outDir}`);
  console.log(`wrote sitemap.xml (${pages.length} urls), llms-full.txt, 404.html`);
  if (written !== pages.length) throw new Error("not all pages prerendered");
}

run();
