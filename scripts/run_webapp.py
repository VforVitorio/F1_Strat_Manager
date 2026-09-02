"""``f1-webapp`` console script wrapper.

Exposed via ``[project.scripts]`` in ``pyproject.toml`` so an installed
checkout gives the user a single-command launcher for the post-race web
app (FastAPI backend + React SPA) alongside ``f1-sim`` (CLI) and
``f1-arcade`` (race replay + the two PITWALL windows).

Running this module delegates to ``docker compose up`` at the repo root:
compose is the canonical way to serve the web app (nginx serves the built
SPA on :8501 and reverse-proxies /api to the backend on :8000). Extra CLI
arguments are forwarded verbatim (``--build``, ``-d``, etc.) so existing
compose knobs keep working.

--- WHERE TO CHANGE IF THE WEB APP DEPLOYMENT CHANGES ---
The compose file lives at the repo root (``docker-compose.yml``); the
webapp service and ports are defined there and in
``src/telemetry/docker-compose.yml``. If the web app ever gains a native
(no-Docker) serving mode, this wrapper is the place to route it.
"""

from __future__ import annotations

import shlex
import shutil
import subprocess
import sys
from pathlib import Path

WEBAPP_URL = "http://localhost:8501"
BACKEND_URL = "http://localhost:8000"

USAGE = f"""f1-webapp: launch the F1 StratLab web app (FastAPI backend + React SPA).

Runs `docker compose up` from the repo root and serves:
  web app  {WEBAPP_URL}
  backend  {BACKEND_URL}

Requires Docker and a source checkout (git clone --recurse-submodules).
Extra arguments are forwarded to `docker compose up` verbatim
(e.g. `f1-webapp --build`, `f1-webapp -d`). Stop with Ctrl+C or
`docker compose down` from the repo root.
"""


def main() -> int:
    """Launch the web app stack with ``docker compose up``.

    Resolves the compose file relative to this file so the wrapper works
    from any working directory of a source checkout. Fails with a clear
    message when Docker or the compose file is missing (a wheel install
    without the repo cannot run the web app: the containers build from
    the source tree). Propagates the child process's exit code.
    """
    if "--help" in sys.argv[1:] or "-h" in sys.argv[1:]:
        print(USAGE)
        return 0

    repo_root = Path(__file__).resolve().parent.parent
    compose_file = repo_root / "docker-compose.yml"
    if not compose_file.exists():
        print(
            f"f1-webapp: cannot find {compose_file} — the web app runs from a "
            "source checkout (git clone --recurse-submodules), not from an "
            "installed wheel.",
            file=sys.stderr,
        )
        return 2

    if shutil.which("docker") is None:
        print(
            "f1-webapp: docker not found on PATH — install Docker Desktop "
            "(or Docker Engine) to run the web app stack.",
            file=sys.stderr,
        )
        return 2

    cmd = ["docker", "compose", "up", *sys.argv[1:]]
    print(f"$ {' '.join(shlex.quote(arg) for arg in cmd)}", file=sys.stderr)
    print(f"web app: {WEBAPP_URL}  ·  backend: {BACKEND_URL}", file=sys.stderr)
    return subprocess.call(cmd, cwd=repo_root)


if __name__ == "__main__":
    sys.exit(main())
