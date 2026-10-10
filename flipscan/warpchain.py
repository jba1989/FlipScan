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
from typing import Callable, Protocol

import cv2
import numpy as np

from .workres import paper_fill, work_scale

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
class Clip:
    """A page boundary: coordinates pass through, but a point is only real
    page if it lies inside `size` here. A crop ends in one, so the desk
    beyond the page quad stays out even when a later rotation's corners
    reach it on the frame."""
    size: Size

    def back(self, xs: np.ndarray, ys: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return xs, ys

    def margin(self, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
        """Distance inside the boundary (negative = outside)."""
        cw, ch = self.size
        return np.minimum.reduce([xs + 0.5, cw - 0.5 - xs, ys + 0.5, ch - 0.5 - ys])


@dataclass(frozen=True)
class Preview:
    """What every estimator sees: the chain so far rendered in grey at work
    resolution, where real page is, and how that maps to the chain's size."""
    gray: np.ndarray
    valid: np.ndarray
    scale: float
    size: Size


Estimate = tuple[Step, Size]
Estimator = Callable[[Preview], "Estimate | None"]


@dataclass(frozen=True)
class Chain:
    """Source frame -> page output: ordered steps plus the output size."""
    steps: tuple[Step, ...]
    size: Size

    @classmethod
    def identity(cls, img: np.ndarray) -> Chain:
        return cls((), (img.shape[1], img.shape[0]))

    def then(self, step: Step, size: Size) -> Chain:
        return Chain(self.steps + (step,), size)

    def crop(self, step: Step, size: Size) -> Chain:
        """A step whose output canvas is the page: clipped to it."""
        return self.then(step, size).then(Clip(size), size)

    def back(self, xs: np.ndarray, ys: np.ndarray
             ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Source coordinates, and each point's distance inside every clip
        (negative = outside). A distance, not a flag, so it upsamples into
        a clean edge instead of grid-sized stairs."""
        margin = np.full(np.shape(xs), 1e6)
        for step in reversed(self.steps):
            if isinstance(step, Clip):
                margin = np.minimum(margin, step.margin(xs, ys))
            xs, ys = step.back(xs, ys)
        return xs, ys, margin

    def maps(self, scale: float = 1.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """float32 source maps for an output rendered at `scale`, plus the
        inside-every-clip mask."""
        w = max(1, round(self.size[0] * scale))
        h = max(1, round(self.size[1] * scale))
        gw, gh = max(2, w // GRID + 1), max(2, h // GRID + 1)
        gx, gy = np.meshgrid(np.linspace(0, w - 1, gw), np.linspace(0, h - 1, gh))
        grid = np.dstack(self.back(gx / scale, gy / scale)).astype(np.float32)
        # every step is smooth, so a coarse grid upsampled is exact enough —
        # but corner-aligned like the linspace grid (cv2.resize aligns pixel
        # centres and would shift and stretch the maps by up to a cell)
        ux = np.tile(np.linspace(0, gw - 1, w, dtype=np.float32), (h, 1))
        uy = np.tile(np.linspace(0, gh - 1, h, dtype=np.float32)[:, None], (1, w))
        full = cv2.remap(grid, ux, uy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        return full[..., 0], full[..., 1], full[..., 2] > 0

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
        if not valid.all():                            # clipped-off desk -> paper
            mask = valid if out.ndim == 2 else valid[..., None]
            np.copyto(out, np.asarray(paper_fill(src), out.dtype), where=~mask)
        return out, valid

    def preview(self, gray_src: np.ndarray) -> Preview:
        """Work-resolution render of a grey source for the estimators."""
        scale = work_scale(self.size)
        img, valid = self.render(gray_src, scale)
        return Preview(img, valid, scale, self.size)

    def estimate(self, gray_src: np.ndarray, estimators: list[Estimator]) -> Chain:
        """Extend the chain with each estimator's step in turn. Every
        estimator sees a fresh preview of the SOURCE through the chain so
        far — never a re-warp of the previous pass's output; the preview is
        only re-rendered when a step was actually added."""
        chain, preview = self, self.preview(gray_src)
        for estimator in estimators:
            found = estimator(preview)
            if found is not None:
                chain = chain.then(*found)
                preview = chain.preview(gray_src)
        return chain


def translation(dx: float, dy: float) -> Homography:
    return Homography(np.array([[1.0, 0, -dx], [0, 1.0, -dy], [0, 0, 1.0]]))


def to_gray(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img


def apply(img: np.ndarray, estimators: list[Estimator]) -> tuple[np.ndarray, np.ndarray]:
    """Run estimators on a ready page image and render once: (image, valid).
    The image itself comes back when nothing needed fixing."""
    return Chain.identity(img).estimate(to_gray(img), estimators).render(img)
