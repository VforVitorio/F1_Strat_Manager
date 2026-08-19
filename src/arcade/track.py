"""Circuit geometry + rendering for the Arcade race replay.

Ported from Tom Shaw's f1-race-replay reference (see
`c:/tmp/arcade_analysis/01_track_rendering.md`). The heavy lifting
(reference-lap interpolation, shoelace-corrected normals, DRS zone index
remap, world-to-screen with rotation about the bbox centre) mirrors that
source but expects raw (non-rotated) telemetry from `SessionLoader`. We
rotate ONCE inside this class: the previous round's bug was rotating both
here and in the loader, which collapsed the outline into a pseudo-circle.
"""

from __future__ import annotations

import logging
from typing import Final

import numpy as np

import arcade
from src.arcade.config import (
    DRS_COLOR,
    DRS_OPEN_CODES,
    DRS_WIDTH,
    FINISH_CHEQUER_SEGMENTS,
    FINISH_CHEQUER_WIDTH,
    TRACK_EDGE_COLOR,
    TRACK_EDGE_WIDTH,
    TRACK_INTERP_EDGE,
    TRACK_INTERP_REF,
    TRACK_PADDING,
    TRACK_WIDTH_WORLD,
)

logger = logging.getLogger(__name__)

# The open-code set lives in `config.py` now: the wire publishes a decoded
# `drs_open` for PITWALL's DRS lane, so two subsystems read this fact and a second
# copy of it is the twin this repo pays for most often.
_DRS_OUTWARD_OFFSET: Final[float] = 0.9  # fraction of track_width beyond the outer edge


class Track:
    """Static circuit geometry cached once; per-frame work is scale + transform."""

    def __init__(
        self,
        ref_x: np.ndarray,
        ref_y: np.ndarray,
        drs_flags: np.ndarray,
        *,
        rotation_deg: float = 0.0,
        track_width: float = TRACK_WIDTH_WORLD,
        interp_ref: int = TRACK_INTERP_REF,
        interp_edge: int = TRACK_INTERP_EDGE,
    ) -> None:
        self._rotation_rad = float(np.deg2rad(rotation_deg))
        self._track_width = float(track_width)
        self._interp_edge = int(interp_edge)

        ref_x = np.asarray(ref_x, dtype=float)
        ref_y = np.asarray(ref_y, dtype=float)

        if ref_x.size < 4:
            logger.warning("Track: reference lap too short, geometry will be empty")
            self._has_geometry = False
            self._screen_inner: np.ndarray = np.zeros((0, 2))
            self._screen_outer: np.ndarray = np.zeros((0, 2))
            self._screen_drs_segments: list[np.ndarray] = []
            self._screen_finish: tuple[tuple[float, float], tuple[float, float]] | None = None
            self._scale = 1.0
            self._tx = 0.0
            self._ty = 0.0
            self._world_cx = 0.0
            self._world_cy = 0.0
            return

        ref_xs = self._resample(ref_x, ref_y, interp_ref)
        normals = self._compute_normals(ref_xs[:, 0], ref_xs[:, 1])
        half_w = self._track_width / 2.0
        inner = ref_xs - normals * half_w
        outer = ref_xs + normals * half_w
        drs_line = ref_xs + normals * (half_w + self._track_width * _DRS_OUTWARD_OFFSET)

        self._world_inner = self._resample(inner[:, 0], inner[:, 1], interp_edge)
        self._world_outer = self._resample(outer[:, 0], outer[:, 1], interp_edge)
        self._world_drs = self._resample(drs_line[:, 0], drs_line[:, 1], interp_edge)
        self._world_ref = ref_xs

        if self._rotation_rad != 0.0:
            pivot = self._centre(np.vstack([self._world_inner, self._world_outer]))
            self._world_inner = self._rotate(self._world_inner, pivot, self._rotation_rad)
            self._world_outer = self._rotate(self._world_outer, pivot, self._rotation_rad)
            self._world_drs = self._rotate(self._world_drs, pivot, self._rotation_rad)
            self._world_ref = self._rotate(self._world_ref, pivot, self._rotation_rad)
            self._world_pivot = pivot
        else:
            self._world_pivot = self._centre(np.vstack([self._world_inner, self._world_outer]))

        all_pts = np.vstack([self._world_inner, self._world_outer])
        self._world_cx = float((all_pts[:, 0].min() + all_pts[:, 0].max()) / 2.0)
        self._world_cy = float((all_pts[:, 1].min() + all_pts[:, 1].max()) / 2.0)
        self._world_w = float(all_pts[:, 0].max() - all_pts[:, 0].min())
        self._world_h = float(all_pts[:, 1].max() - all_pts[:, 1].min())

        self._drs_segments_world = self._detect_drs_zones(
            np.asarray(drs_flags, dtype=float), interp_edge
        )

        self._has_geometry = True
        self._scale = 1.0
        self._tx = 0.0
        self._ty = 0.0
        self._screen_inner = np.zeros_like(self._world_inner)
        self._screen_outer = np.zeros_like(self._world_outer)
        self._screen_drs_segments = []
        self._screen_finish = None

    # --- Public API ------------------------------------------------------

    def update_scaling(
        self,
        width: int,
        height: int,
        *,
        margin_left: int,
        margin_right: int,
        margin_bottom: int,
        margin_top: int,
        padding: float = TRACK_PADDING,
    ) -> None:
        """Fit the rotated track into the reserved viewport and rebuild screen polylines."""
        if not self._has_geometry:
            return

        inner_w = max(1.0, width - margin_left - margin_right)
        inner_h = max(1.0, height - margin_bottom - margin_top)
        usable_w = inner_w * (1.0 - 2.0 * padding)
        usable_h = inner_h * (1.0 - 2.0 * padding)

        scale_x = usable_w / max(1e-6, self._world_w)
        scale_y = usable_h / max(1e-6, self._world_h)
        self._scale = float(min(scale_x, scale_y))

        screen_cx = margin_left + inner_w / 2.0
        screen_cy = margin_bottom + inner_h / 2.0
        self._tx = float(screen_cx - self._scale * self._world_cx)
        self._ty = float(screen_cy - self._scale * self._world_cy)

        self._screen_inner = self._project_poly(self._world_inner)
        self._screen_outer = self._project_poly(self._world_outer)
        self._screen_drs_segments = [self._project_poly(seg) for seg in self._drs_segments_world]
        if len(self._screen_inner) > 0 and len(self._screen_outer) > 0:
            self._screen_finish = (
                tuple(self._screen_inner[0]),
                tuple(self._screen_outer[0]),
            )

    def project(self, wx: float, wy: float) -> tuple[float, float]:
        """World (raw FastF1 X/Y) → screen pixel coords. Applies rotation + scale."""
        if not self._has_geometry:
            return 0.0, 0.0
        if self._rotation_rad != 0.0:
            rx, ry = self._rotate_point(wx, wy, self._world_pivot, self._rotation_rad)
        else:
            rx, ry = wx, wy
        return float(self._scale * rx + self._tx), float(self._scale * ry + self._ty)

    def draw(
        self,
        *,
        edge_color: tuple[int, int, int] = TRACK_EDGE_COLOR,
        edge_width: int = TRACK_EDGE_WIDTH,
        drs_color: tuple[int, int, int] = DRS_COLOR,
        drs_width: int = DRS_WIDTH,
        show_finish_line: bool = True,
        show_drs: bool = True,
    ) -> None:
        """Render inner + outer edges, DRS overlays, and finish chequer."""
        if not self._has_geometry:
            return
        if len(self._screen_inner) >= 2:
            arcade.draw_line_strip([tuple(p) for p in self._screen_inner], edge_color, edge_width)
        if len(self._screen_outer) >= 2:
            arcade.draw_line_strip([tuple(p) for p in self._screen_outer], edge_color, edge_width)
        if show_drs:
            for seg in self._screen_drs_segments:
                if len(seg) >= 2:
                    arcade.draw_line_strip([tuple(p) for p in seg], drs_color, drs_width)
        if show_finish_line and self._screen_finish is not None:
            self._draw_finish_line(self._screen_finish, edge_width)

    # --- Internals -------------------------------------------------------

    @staticmethod
    def _resample(xs: np.ndarray, ys: np.ndarray, n: int) -> np.ndarray:
        if xs.size < 2:
            return np.column_stack([xs, ys])
        t = np.linspace(0.0, 1.0, len(xs))
        t_new = np.linspace(0.0, 1.0, n)
        xr = np.interp(t_new, t, xs)
        yr = np.interp(t_new, t, ys)
        return np.column_stack([xr, yr])

    @staticmethod
    def _compute_normals(xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
        dx = np.gradient(xs)
        dy = np.gradient(ys)
        mag = np.hypot(dx, dy)
        mag[mag < 1e-12] = 1.0
        nx = -dy / mag
        ny = dx / mag
        # Shoelace signed area — flip sign so normals consistently point outward.
        signed_area = 0.5 * float(np.sum(xs * np.roll(ys, -1) - np.roll(xs, -1) * ys))
        if signed_area > 0:
            nx = -nx
            ny = -ny
        return np.column_stack([nx, ny])

    def _detect_drs_zones(self, drs_flags: np.ndarray, edge_len: int) -> list[np.ndarray]:
        """Project the raw-lap DRS-active mask onto the resampled edge polyline.

        The reference draws DRS by slicing the unsampled outer polyline with
        raw telemetry indices. We resample the edges to `edge_len`, so we map
        each edge point to its nearest raw sample (same parametric t in
        [0, 1]) and scan for contiguous active runs there. Gives a clean
        start/end without the fragmentation the earlier index-rescale
        produced."""
        if drs_flags.size < 2 or edge_len < 2:
            return []
        active_raw = np.isin(np.round(drs_flags).astype(int), tuple(DRS_OPEN_CODES))
        if not active_raw.any():
            return []
        raw_len = len(drs_flags)
        t_edge = np.arange(edge_len) / (edge_len - 1)
        raw_idx = np.clip(np.round(t_edge * (raw_len - 1)).astype(int), 0, raw_len - 1)
        active_edge = active_raw[raw_idx].copy()
        # Morphological close: bridge gaps up to 30 edge samples (~1.5% of lap).
        # Without this, nearest-neighbor projection can alternate True/False at
        # raw/edge resolution mismatches, producing visible pinhole gaps.
        gap_fill = 30
        false_runs: list[tuple[int, int]] = []
        i = 0
        while i < edge_len:
            if not active_edge[i]:
                j = i
                while j < edge_len and not active_edge[j]:
                    j += 1
                false_runs.append((i, j))
                i = j
            else:
                i += 1
        for start, end in false_runs:
            run_len = end - start
            if 0 < start and end < edge_len and run_len <= gap_fill:
                active_edge[start:end] = True
        segments: list[np.ndarray] = []
        i = 0
        while i < edge_len:
            if active_edge[i]:
                j = i
                while j < edge_len and active_edge[j]:
                    j += 1
                if j - i >= 2:
                    segments.append(self._world_drs[i:j])
                i = j
            else:
                i += 1
        return segments

    def _project_poly(self, poly: np.ndarray) -> np.ndarray:
        if poly.size == 0:
            return poly
        return poly * self._scale + np.array([self._tx, self._ty])

    @staticmethod
    def _centre(pts: np.ndarray) -> tuple[float, float]:
        return (
            float((pts[:, 0].min() + pts[:, 0].max()) / 2.0),
            float((pts[:, 1].min() + pts[:, 1].max()) / 2.0),
        )

    @staticmethod
    def _rotate(pts: np.ndarray, pivot: tuple[float, float], rad: float) -> np.ndarray:
        cos_a = np.cos(rad)
        sin_a = np.sin(rad)
        px, py = pivot
        shifted = pts - np.array([px, py])
        rotated = np.column_stack(
            [
                shifted[:, 0] * cos_a - shifted[:, 1] * sin_a,
                shifted[:, 0] * sin_a + shifted[:, 1] * cos_a,
            ]
        )
        return rotated + np.array([px, py])

    @staticmethod
    def _rotate_point(
        x: float, y: float, pivot: tuple[float, float], rad: float
    ) -> tuple[float, float]:
        cos_a = np.cos(rad)
        sin_a = np.sin(rad)
        px, py = pivot
        dx = x - px
        dy = y - py
        return (dx * cos_a - dy * sin_a + px, dx * sin_a + dy * cos_a + py)

    def _draw_finish_line(
        self,
        endpoints: tuple[tuple[float, float], tuple[float, float]],
        width: int,
    ) -> None:
        (ix, iy), (ox, oy) = endpoints
        dx = ox - ix
        dy = oy - iy
        segs = FINISH_CHEQUER_SEGMENTS
        for s in range(segs):
            t0 = s / segs
            t1 = (s + 1) / segs
            x0, y0 = ix + dx * t0, iy + dy * t0
            x1, y1 = ix + dx * t1, iy + dy * t1
            color = (255, 255, 255) if s % 2 == 0 else (20, 20, 20)
            arcade.draw_line(x0, y0, x1, y1, color, FINISH_CHEQUER_WIDTH)
