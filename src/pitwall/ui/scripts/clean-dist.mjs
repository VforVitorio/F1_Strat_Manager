/**
 * Empty `dist/` before a build, verify it, and FAIL LOUDLY if it is still there.
 *
 * **Two mechanisms already tried to do this and both failed in SILENCE**, which
 * is why this file exists and why it checks its own work rather than trusting a
 * call that returned.
 *
 * 1. Vite's `emptyOutDir: true` is set in `vite.config.ts` and does not empty it.
 * 2. `fs.rmSync(..., { force: true })` returns normally and deletes nothing on
 *    this tree, measured down to a single file, with `force` removed so an
 *    error could not be swallowed: `existsSync` said true before AND after, and
 *    nothing was thrown. A filter driver on the path is the likely culprit;
 *    the cause matters less than the fact that the return value is worthless.
 *
 * What the silence cost: builds accumulate, and every orphan chunk ships inside
 * the wheel. Measured at 83 assets and 11 MB against the 6 and 1.4 MB a clean
 * build produces: seven copies of a 1.3 MB chunk with exactly one of them live.
 * `tests/infra/test_wheel_ships_the_ui.py` is the guard that caught it.
 *
 * So: try Node, verify; fall back to the platform's own delete, verify again;
 * and only then stop the build with something a human can act on. A build that
 * halts is recoverable in seconds. A build that grows the wheel is not
 * noticed until someone weighs it.
 */
import { spawnSync } from "node:child_process";
import { existsSync, readdirSync, rmSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const DIST = resolve(dirname(fileURLToPath(import.meta.url)), "..", "dist");

/** Entries still in `dist/`, or 0 when it is gone. Never throws. */
function remaining() {
  if (!existsSync(DIST)) return 0;
  try {
    return readdirSync(DIST).length;
  } catch {
    return -1;
  }
}

if (remaining() !== 0) {
  try {
    rmSync(DIST, { recursive: true, force: true, maxRetries: 3, retryDelay: 100 });
  } catch (error) {
    console.error(`clean-dist: node rm failed (${error.code ?? error.message}), trying the shell`);
  }
}

if (remaining() !== 0) {
  // The platform's own delete. Reached routinely on this machine, not as a
  // last resort, so it is not treated as an error path.
  const [command, args] =
    process.platform === "win32"
      ? ["cmd", ["/c", "rmdir", "/s", "/q", DIST]]
      : ["rm", ["-rf", DIST]];
  spawnSync(command, args, { stdio: "ignore" });
}

if (remaining() !== 0) {
  console.error(
    `clean-dist: dist/ still holds ${remaining()} entr(ies) after two attempts to remove it.\n` +
      "  The build would ADD to it rather than replace it, and every stale chunk\n" +
      "  would ship inside the wheel. Most often a running window holds the bundle\n" +
      "  open - PITWALL renders in the platform webview, and an orphaned renderer\n" +
      "  outlives the window that spawned it:\n" +
      '    powershell "Get-Process msedgewebview2 -EA SilentlyContinue | Stop-Process -Force"\n' +
      `  Then remove it by hand: rm -rf "${DIST}"`,
  );
  process.exit(1);
}
