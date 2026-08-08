/**
 * Serve a built bundle over http, from a map read once at startup.
 *
 * Shared by the screenshot tool and the smoke test. Two reasons it exists
 * at all, both learned the hard way:
 *
 * - chromium refuses `<script type="module">` over `file://` (CORS treats
 *   it as origin `null`), so a bundle opened from disk loads nothing and
 *   renders a blank page with two console errors. pywebview reaches the
 *   same files through the OS webview, which does allow it.
 * - nothing here turns a request into a path. The first version joined
 *   the URL onto the root and CodeQL called it a path injection, then the
 *   `stat`-per-entry version was a file-system race. Reading the tree
 *   into memory removes both questions instead of guarding them, and a
 *   built bundle is a few hundred kilobytes.
 */
import { readFileSync, readdirSync } from "node:fs";
import { createServer } from "node:http";
import { extname, join } from "node:path";

const MIME = {
  ".html": "text/html",
  ".js": "text/javascript",
  ".css": "text/css",
  ".json": "application/json",
};

export function serveDist(root) {
  const files = new Map();
  // `withFileTypes` answers directory-or-file from the directory read
  // itself. Asking again with `stat` is a second look at something that
  // can change in between.
  const walk = (dir, prefix) => {
    for (const entry of readdirSync(dir, { withFileTypes: true })) {
      const full = join(dir, entry.name);
      const url = `${prefix}/${entry.name}`;
      if (entry.isDirectory()) walk(full, url);
      else files.set(url, { body: readFileSync(full), type: MIME[extname(entry.name)] });
    }
  };
  walk(root, "");

  const server = createServer((req, res) => {
    const file = files.get(req.url.split("?")[0]);
    if (!file) {
      res.statusCode = 404;
      res.end();
      return;
    }
    res.setHeader("Content-Type", file.type ?? "application/octet-stream");
    res.end(file.body);
  });
  return new Promise((ready) => server.listen(0, "127.0.0.1", () => ready(server)));
}
