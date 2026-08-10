/**
 * Generate the docs site's changelog page from the canonical CHANGELOG.md.
 *
 * The page used to be a hand-refreshed copy, and it rotted exactly the way a
 * hand-refreshed copy does: the site sat on **1.10.5** while the repo had
 * shipped **2.5.1**, a whole major version of releases missing from the only
 * public record of them. Nobody forgot on purpose - release-please writes
 * `CHANGELOG.md` on every merge to `main` and there was no step that carried
 * that across.
 *
 * So the page is generated now, from the file release-please owns:
 *
 *   node scripts/sync_docs_changelog.mjs
 *
 * `.github/workflows/docs.yml` runs it before staging, and lists
 * `CHANGELOG.md` in its trigger paths, so a release publishes its own entry.
 * The committed copy is regenerated too, so the repo and the site agree and a
 * local preview is not a lie.
 *
 * --- WHERE TO CHANGE IF THE DATE FORMAT CHANGES ---
 * Here, and only here. The transformation deliberately does NOT live in the
 * renderer: `docs/app/markdown.js` renders in the browser and
 * `scripts/prerender_docs.mjs` renders the same markdown in Node, so a
 * render-time rule would be two implementations of one decision - this
 * repo's dominant defect. Rewriting the source once means both agree by
 * construction.
 */

import { readFileSync, writeFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const SOURCE = resolve(ROOT, "CHANGELOG.md");
const TARGET = resolve(ROOT, "docs", "pages", "changelog.md");

/**
 * What the site adds on top of the canonical file.
 *
 * It says GENERATED rather than "manually-refreshed mirror", because the old
 * wording was an instruction nobody followed and a promise the page could not
 * keep.
 */
const FRONT_MATTER = `> **Note:** this page is generated from
> [CHANGELOG.md](https://github.com/VforVitorio/F1-StratLab/blob/main/CHANGELOG.md)
> by \`scripts/sync_docs_changelog.mjs\`, which the docs workflow runs on every
> deploy. Do not edit it by hand — edit the canonical file, or the generator.

> Maintained by [release-please](https://github.com/googleapis/release-please)
> on each merge to \`main\` — see [CI/CD pipeline](#/ci-cd) for the full flow.
> Dates are shown day-month-year.

`;

/**
 * Release headings only: `## [2.5.1](compare-url) (2026-07-28)`.
 *
 * Anchored to the end of a heading line so it can never touch a date inside a
 * commit subject, which is prose the generator has no business rewriting.
 * The month is kept two-digit rather than spelled out: the column stays
 * aligned down a thousand lines, and it needs no locale.
 */
const RELEASE_HEADING_DATE = /^(#{2,3} .*)\((\d{4})-(\d{2})-(\d{2})\)\s*$/gm;

export function toDayMonthYear(markdown) {
  return markdown.replace(
    RELEASE_HEADING_DATE,
    (_match, heading, year, month, day) => `${heading}(${day}-${month}-${year})`,
  );
}

function main() {
  const canonical = readFileSync(SOURCE, "utf-8");
  const page = FRONT_MATTER + toDayMonthYear(canonical);
  writeFileSync(TARGET, page, "utf-8");

  const releases = (page.match(/^#{2,3} \[\d+\.\d+\.\d+\]/gm) || []).length;
  const latest = page.match(/^#{2,3} \[(\d+\.\d+\.\d+)\][^\n]*\((\d{2}-\d{2}-\d{4})\)/m);
  console.log(
    `sync_docs_changelog: ${releases} releases, newest ${latest ? latest[1] : "?"} ` +
      `(${latest ? latest[2] : "?"}) -> docs/pages/changelog.md`,
  );
}

main();
