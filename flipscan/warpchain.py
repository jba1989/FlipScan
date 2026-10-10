"""A page's geometry as one chain of transforms, rendered with ONE resample.

Every straightening pass (outline quad, deskew, text-line field, keystone)
only *estimates* a step; nothing is warped until the end. Steps map output
pixels back to their input (inverse mapping), so the chain composes by
calling them last-to-first and the source frame is resampled exactly once:
sharper text than four chained cubic warps, and no pass ever sees another
pass's padding — the validity mask says where real pixels end.

Steps hold only their parameters (a matrix, a few coefficients), never an
image, so chains are cheap to keep and copy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import cv2
import numpy as np

from .workres import WORK_LONG_EDGE, paper_fill

Size = tuple[int, int]     # (width, height)
GRID = 4                   # inverse maps are evaluated every GRID px, then upsampled


class Step(Protocol):
    def back(self, xs: np.ndarray, ys: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Output coordinates -> input coordinates."""
        ...


@dataclass(frozen=True)
class Homography:
    """A projective step; `forward` maps input to output pixels."""
    forward: np.ndarray

    def back(self, xs: np.ndarray, ys: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        inv = np.linalg.inv(self.forward)
        w = inv[2, 0] * xs + inv[2, 1] * ys + inv[2, 2]
        return ((inv[0, 0] * xs + inv[0, 1] * ys + inv[0, 2]) / w,
                (inv[1, 0] * xs + inv[1, 1] * ys + inv[1, 2]) / w)


@dataclass(frozen=True)
class Chain:
    """Source frame -> page output: ordered steps plus the output size.

    `clips` are (step count, size) boundaries: a point is only real page if,
    walked back to that stage, it lies inside that stage's canvas. The crop
    stage is one — the desk beyond the page quad must stay out even though a
    later rotation's corners reach it on the frame."""
    steps: tuple[Step, ...]
    size: Size
    clips: tuple[tuple[int, Size], ...] = ()

    @classmethod
    def identity(cls, img: np.ndarray) -> Chain:
        return cls((), (img.shape[1], img.shape[0]))

    def then(self, step: Step, size: Size) -> Chain:
        return Chain(self.steps + (step,), size, self.clips)

    def clipped(self) -> Chain:
        """Mark the current canvas as the page boundary."""
        return Chain(self.steps, self.size, self.clips + ((len(self.steps), self.size),))

    def back(self, xs: np.ndarray, ys: np.ndarray
             ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Source coordinates, and each point's distance inside every clip
        (negative = outside). A distance, not a flag, so it upsamples into
        a clean edge instead of grid-sized stairs."""
        margin = np.full(np.shape(xs), np.inf)
        clips = dict(self.clips)
        for n in range(len(self.steps), -1, -1):
            if n in clips:
                cw, ch = clips[n]
                margin = np.minimum.reduce([margin, xs + 0.5, cw - 0.5 - xs,
                                            ys + 0.5, ch - 0.5 - ys])
            if n:
                xs, ys = self.steps[n - 1].back(xs, ys)
        return xs, ys, margin

    def maps(self, scale: float = 1.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """float32 source maps for an output rendered at `scale`, plus the
        inside-every-clip mask."""
        w = max(1, round(self.size[0] * scale))
        h = max(1, round(self.size[1] * scale))
        gw, gh = max(2, w // GRID + 1), max(2, h // GRID + 1)
        gx, gy = np.meshgrid(np.linspace(0, w - 1, gw), np.linspace(0, h - 1, gh))
        sx, sy, margin = self.back(gx / scale, gy / scale)
        margin = np.minimum(margin, 1e6)               # no clips: all inside
        # every step is smooth, so a coarse grid upsampled is exact enough —
        # but corner-aligned like the linspace grid (cv2.resize aligns pixel
        # centres and would shift and stretch the maps by up to a cell)
        ux = np.broadcast_to(np.linspace(0, gw - 1, w, dtype=np.float32), (h, w))
        uy = np.broadcast_to(np.linspace(0, gh - 1, h, dtype=np.float32)[:, None], (h, w))
        def up(a: np.ndarray) -> np.ndarray:
            return cv2.remap(a.astype(np.float32), ux, uy, cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_REPLICATE)
        return up(sx), up(sy), up(margin) > 0

    def render(self, src: np.ndarray, scale: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
        """(image, valid) — valid marks output pixels of real page; the rest
        (frame edge, beyond a clip) is white paper."""
        if not self.steps and scale == 1.0:
            return src, np.ones(src.shape[:2], bool)
        map_x, map_y, valid = self.maps(scale)
        h, w = src.shape[:2]
        valid &= (map_x >= 0) & (map_x <= w - 1) & (map_y >= 0) & (map_y <= h - 1)
        interp = cv2.INTER_CUBIC if scale >= 1.0 else cv2.INTER_LINEAR
        out = cv2.remap(src, map_x, map_y, interp,
                        borderMode=cv2.BORDER_CONSTANT, borderValue=paper_fill(src))
        out[~valid] = paper_fill(src)
        return out, valid

    def preview(self, src: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
        """Work-resolution render for the estimators: (image, valid, scale)."""
        scale = min(1.0, WORK_LONG_EDGE / max(self.size))
        img, valid = self.render(src, scale)
        return img, valid, scale


def translation(dx: float, dy: float) -> Homography:
    return Homography(np.array([[1.0, 0, -dx], [0, 1.0, -dy], [0, 0, 1.0]]))
