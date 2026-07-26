"""Audit DRS zones per GP: sanity check against the FIA 2025 circuit docs.

Uses the same ``_extract_reference_lap`` pattern the arcade's
``SessionLoader`` relies on: fastest qualifying lap telemetry, ``DRS``
column, ``add_distance()`` for the distance axis.

For each DRS activation zone the script reports:

- Start / end distance in metres from the start-finish line
- Zone length in metres
- Nearest corner number at start and end (``after T3 → before T5``) via
  ``session.get_circuit_info()`` so the zone reads the same way the FIA
  Race Director's Event Notes describe them
- Start / end (X, Y) coordinates and boundary speeds

The FIA publishes a per-GP PDF with zones as "detection at X m before
Turn Y, activation from Z m after Turn W". Cross-reference the corner
tags and the metre offsets below with that PDF: they should match
within ~30 m (sampling resolution of FastF1 telemetry).

Anomalies to expect:

- Short telemetry blips under ``--min-length`` are skipped (default 80
  m) because FastF1 sometimes registers a ~20 m DRS test flick near
  pit-out that is not a real activation zone.
- A circuit reported as ``BROKEN`` usually means FastF1 could not
  produce a clean flying lap for Q (red flag / no valid time on the
  day the session was cached). Re-run with the race session by
  swapping ``get_session(..., "Q")`` → ``"R"`` for that round.

Usage:
    python scripts/verify_drs_zones.py --year 2025
    python scripts/verify_drs_zones.py --year 2025 --round 3
    python scripts/verify_drs_zones.py --year 2025 --json drs_audit.json
    python scripts/verify_drs_zones.py --year 2025 --summary --min-length 100
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import fastf1  # noqa: E402


def extract_zones(tel, corners_df=None, min_length_m: float = 80.0) -> list[dict]:
    """Walk the telemetry row-by-row, emit one dict per DRS-ON run.

    FastF1 publishes ``DRS`` as a status byte; values ``>= 10`` mean the
    flap is open. A "zone" is any maximal run of open samples. Zones
    shorter than ``min_length_m`` are dropped as telemetry blips: real
    F1 DRS activation zones are always several hundred metres long.

    When ``corners_df`` is provided (from ``session.get_circuit_info()``)
    each zone is tagged with the nearest corner at its start and end
    coordinates so the output reads like the FIA Event Notes.
    """
    drs = tel["DRS"].to_numpy().astype(float)
    dist = tel["Distance"].to_numpy().astype(float)
    x = tel["X"].to_numpy().astype(float)
    y = tel["Y"].to_numpy().astype(float)
    speed = tel["Speed"].to_numpy().astype(float)

    zones: list[dict] = []
    active = drs >= 10
    n = active.size
    i = 0
    while i < n:
        if not active[i]:
            i += 1
            continue
        j = i
        while j < n and active[j]:
            j += 1
        length = float(dist[j - 1] - dist[i])
        # Reject sub-zone blips (pit-out test flicks and similar noise).
        if length < min_length_m:
            i = j
            continue
        start_corner = _nearest_corner(x[i], y[i], corners_df)
        end_corner = _nearest_corner(x[j - 1], y[j - 1], corners_df)
        zones.append(
            {
                "zone_start_m": round(float(dist[i]), 1),
                "zone_end_m": round(float(dist[j - 1]), 1),
                "length_m": round(length, 1),
                "start_corner": start_corner,
                "end_corner": end_corner,
                "start_xy": (int(x[i]), int(y[i])),
                "end_xy": (int(x[j - 1]), int(y[j - 1])),
                "start_speed_kph": round(float(speed[i]), 1),
                "end_speed_kph": round(float(speed[j - 1]), 1),
                "samples": int(j - i),
            }
        )
        i = j
    return zones


def _nearest_corner(x: float, y: float, corners_df) -> str:
    """Return ``"T3"`` (closest corner number) for a given point, or ``"—"``
    when circuit info is unavailable. Uses Euclidean distance in the
    FastF1 position coordinate space (units do not matter for argmin)."""
    if corners_df is None or corners_df.empty:
        return "—"
    try:
        dx = corners_df["X"].to_numpy() - float(x)
        dy = corners_df["Y"].to_numpy() - float(y)
        idx = int(np.argmin(dx * dx + dy * dy))
        num = int(corners_df.iloc[idx]["Number"])
        return f"T{num}"
    except Exception:
        return "—"


def audit_one(year: int, round_: int, min_length_m: float) -> dict:
    """Load quali, fastest lap, telemetry + add_distance, then
    extract zones with corner tags. Exceptions become error dicts so a
    single bad session does not halt the batch."""
    try:
        session = fastf1.get_session(year, round_, "Q")
        session.load(telemetry=True, laps=True, weather=False, messages=False)
        lap = session.laps.pick_fastest()
        if lap is None or lap.empty:
            return {"round": round_, "error": "no fastest lap in quali"}
        tel = lap.get_telemetry().add_distance()
        if "DRS" not in tel.columns:
            return {
                "round": round_,
                "gp": session.event.get("Location", "?"),
                "error": "DRS column missing",
            }
        corners_df = None
        try:
            ci = session.get_circuit_info()
            corners_df = ci.corners
        except Exception:
            # Circuit info is only available on recent FastF1 versions /
            # for circuits FastF1 has the corner database for: graceful
            # degrade, the zone tags will read "—".
            pass
        zones = extract_zones(tel, corners_df=corners_df, min_length_m=min_length_m)
        lap_length_m = round(float(tel["Distance"].iloc[-1]), 1)
        return {
            "round": round_,
            "gp": session.event.get("Location", "?"),
            "lap_length_m": lap_length_m,
            "n_zones": len(zones),
            "zones": zones,
        }
    except Exception as exc:
        return {"round": round_, "error": f"{type(exc).__name__}: {exc}"}


def _classify(row: dict) -> str:
    if "error" in row:
        return "ERROR"
    n = row.get("n_zones", 0)
    if n == 0:
        return "BROKEN"
    if 1 <= n <= 4:
        return "OK"
    return "SUSPICIOUS"


def _format_row(row: dict) -> str:
    """Human-readable per-GP block for cross-checking against FIA docs."""
    header = f"R{row['round']:02d}  "
    if "error" in row:
        return header + f"ERROR: {row['error']}"
    body = f"{row['gp']:<22s}  lap={row['lap_length_m']}m  zones={row['n_zones']}"
    lines = [header + body]
    for i, z in enumerate(row["zones"], 1):
        lines.append(
            f"    Zone {i}: {z['zone_start_m']:>6.1f}m → {z['zone_end_m']:>6.1f}m  "
            f"({z['length_m']:>5.1f}m) · "
            f"{z['start_corner']} → {z['end_corner']} · "
            f"spd {z['start_speed_kph']:>5.1f} → {z['end_speed_kph']:>5.1f} kph"
        )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit DRS zones per GP.")
    parser.add_argument("--year", type=int, default=2025)
    parser.add_argument(
        "--round",
        type=int,
        default=0,
        help="Single round to audit (default: all rounds in the year).",
    )
    parser.add_argument(
        "--json",
        type=str,
        default="",
        help="If set, dump the full audit as JSON to this path.",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Skip the per-GP detail block and print only the summary table.",
    )
    parser.add_argument(
        "--min-length",
        type=float,
        default=80.0,
        help="Drop DRS runs shorter than this (metres). Default 80m filters "
        "FastF1 pit-out flicks / noise without rejecting real zones.",
    )
    args = parser.parse_args()

    fastf1.Cache.enable_cache(_REPO_ROOT / "data" / "cache" / "fastf1")

    rounds = [args.round] if args.round else list(range(1, 25))
    rows: list[dict] = []
    for r in rounds:
        row = audit_one(args.year, r, args.min_length)
        rows.append(row)
        if not args.summary:
            print(_format_row(row))
            print()

    print("=== Summary ===")
    for row in rows:
        status = _classify(row)
        tag = f"[{status:10s}]"
        if "error" in row:
            print(f"  R{row['round']:02d} {tag} — {row['error']}")
        else:
            corners = (
                " · ".join(f"{z['start_corner']}→{z['end_corner']}" for z in row["zones"])
                or "no zones"
            )
            print(
                f"  R{row['round']:02d} {tag}  {row['gp']:<22s} zones={row['n_zones']}  [{corners}]"
            )
    broken = [r for r in rows if _classify(r) in ("BROKEN", "ERROR")]
    sus = [r for r in rows if _classify(r) == "SUSPICIOUS"]
    print(f"\n{len(rows)} rounds audited · {len(broken)} broken/error · {len(sus)} suspicious")

    if args.json:
        out_path = Path(args.json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
        print(f"\nJSON dump written to {out_path}")

    return 1 if broken else 0


if __name__ == "__main__":
    sys.exit(main())
