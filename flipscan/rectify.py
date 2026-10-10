"""Content-based page straightening: rectify from the printed text lines.

A book page's outline is a poor guide — the desk often shares its colour, the
page bows near the spine, and the corners hide under thumbs — so a quad warp
alone leaves text tilted and curled. The text itself is the reliable ruler:
every printed line should end up horizontal and straight.

Passes, each gated so a page without enough evidence is left untouched
(the invoshot lesson: an uncertain observation must degrade to a no-op):

1. ``estimate_skew``: whole-page rotation by projection-profile search.
2. ``fit_text_field``: residual curl / keystone. Text-line centerlines are
   fit jointly to one smooth vertical displacement field.
3. ``keystone.keystone_step``: page verticals upright.

Each pass only estimates a step of a warpchain.Chain from a fresh preview of
the source; ``straighten_warp`` returns the chain and the frame is resampled
once (warpchain.py).
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .keystone import keystone_step
from .warpchain import Chain, Homography, Size
from .workres import ink_mask, work_gray

MAX_SKEW_DEG = 10.0        # beyond this it's a mis-framed shot, not skew
MIN_SKEW_DEG = 0.3         # smaller rotations are not worth a resample
MIN_SKEW_GAIN = 1.15       # best profile must beat the unrotated one by 15%
MIN_LINES = 5              # fewer text lines can't pin down a 2-D field
MAX_DISPLACEMENT = 0.06    # of page height; more means the fit went wild
X_DEGREE, Y_DEGREE = 4, 2  # displacement field d(x, y) polynomial orders


def rotation_step(angle: float, size: Size) -> tuple[Homography, Size]:
    """Rotate about the centre (cv2 sign convention), growing the canvas so
    no corner is cut."""
    w, h = size
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    cos, sin = abs(m[0, 0]), abs(m[0, 1])
    nw, nh = int(round(h * sin + w * cos)), int(round(h * cos + w * sin))
    m[0, 2] += nw / 2 - w / 2
    m[1, 2] += nh / 2 - h / 2
    return Homography(np.vstack([m, [0.0, 0.0, 1.0]])), (nw, nh)


def _profile_score(ys: np.ndarray, xs: np.ndarray, angle: float) -> float:
    """Sharpness of the row profile of the ink pixels rotated by `angle`:
    text rows alternate with blank leading, so the profile is spikiest when
    the lines run exactly horizontal. Rotates coordinates, not the image."""
    t = np.radians(angle)
    rows = np.round(ys * np.cos(t) - xs * np.sin(t)).astype(np.int64)
    profile = np.pad(np.bincount(rows - rows.min()), 1).astype(np.float64)
    return float(np.square(np.diff(profile)).sum())


def _best_angle(ys: np.ndarray, xs: np.ndarray,
                angles: np.ndarray) -> tuple[float, float]:
    scores = [_profile_score(ys, xs, a) for a in angles]
    i = int(np.argmax(scores))
    return float(angles[i]), scores[i]


def estimate_skew(color: np.ndarray) -> float:
    """Rotation (degrees, cv2 convention) that makes the text lines horizontal,
    or 0.0 when the page has no confident line structure."""
    gray, _ = work_gray(color)
    ink = ink_mask(gray)
    if np.count_nonzero(ink) < 0.005 * ink.size:
        return 0.0
    ys, xs = (v.astype(np.float64) for v in np.nonzero(ink))
    best, _ = _best_angle(ys, xs, np.arange(-MAX_SKEW_DEG, MAX_SKEW_DEG + 1e-6, 0.5))
    best, score = _best_angle(ys, xs, np.arange(best - 0.4, best + 0.41, 0.1))
    if score < MIN_SKEW_GAIN * _profile_score(ys, xs, 0.0) or abs(best) < MIN_SKEW_DEG:
        return 0.0
    return best


def _char_height(ink: np.ndarray) -> float | None:
    _n, _lab, stats, _c = cv2.connectedComponentsWithStats(ink)
    hs = stats[1:, cv2.CC_STAT_HEIGHT]
    hs = hs[(hs >= 4) & (hs < 0.05 * ink.shape[0])]
    return float(np.median(hs)) if len(hs) >= 30 else None


def _line_samples(ink: np.ndarray) -> list[np.ndarray]:
    """Centerline samples (x, y) of each long text line on the work image."""
    w = ink.shape[1]
    ch = _char_height(ink)
    if ch is None:
        return []
    k = max(3, int(round(1.6 * ch)))
    blobs = cv2.morphologyEx(ink, cv2.MORPH_CLOSE,
                             cv2.getStructuringElement(cv2.MORPH_RECT, (k, 1)))
    blobs = cv2.morphologyEx(blobs, cv2.MORPH_OPEN,
                             cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(2, int(ch * 0.3)))))
    n, lab, stats, _c = cv2.connectedComponentsWithStats(blobs)
    step = max(2, int(ch))
    lines = []
    for i in range(1, n):
        x, y, bw, bh = stats[i, :4]
        if bw < 0.25 * w or bh > 3.0 * ch or bw < 6 * bh:
            continue
        starts = np.arange(0, bw, step)
        hit = np.logical_or.reduceat(lab[y:y + bh, x:x + bw] == i, starts, axis=1)
        has = hit.any(axis=0)
        if has.sum() >= 8:
            first = hit.argmax(axis=0)
            last = bh - 1 - hit[::-1].argmax(axis=0)
            pts = np.c_[x + starts + step / 2, y + (first + last) / 2]
            lines.append(pts[has].astype(np.float64))
    return lines


def _design(xn: np.ndarray, yn: np.ndarray) -> np.ndarray:
    """Displacement-field basis x^j y^k with j >= 1: pure functions of y only
    shift whole rows and are absorbed by each line's own offset."""
    return np.stack([xn ** j * yn ** k
                     for j in range(1, X_DEGREE + 1) for k in range(Y_DEGREE + 1)], axis=-1)


Bounds = tuple[float, float, float, float]


def fit_displacement(lines: list[np.ndarray], w: int, h: int
                     ) -> tuple[np.ndarray, Bounds, float, float] | None:
    """Jointly fit y = t_i + d(x, y) over all lines. Returns (coef, bounds,
    rms_flat, rms_fit) or None. d is zero along the page's centre column."""
    if len(lines) < MIN_LINES:
        return None
    pts = np.concatenate(lines)
    ids = np.concatenate([np.full(len(l), i) for i, l in enumerate(lines)])
    xn, yn = (pts[:, 0] - w / 2) / w, pts[:, 1] / h
    onehot = np.eye(len(lines))[ids]
    a = np.hstack([onehot, _design(xn, yn)])
    sol, *_ = np.linalg.lstsq(a, pts[:, 1], rcond=None)
    rms_fit = float(np.sqrt(np.mean((a @ sol - pts[:, 1]) ** 2)))
    means = np.array([l[:, 1].mean() for l in lines])
    rms_flat = float(np.sqrt(np.mean((pts[:, 1] - means[ids]) ** 2)))
    bounds = (xn.min(), xn.max(), yn.min(), yn.max())
    return sol[len(lines):], bounds, rms_flat, rms_fit


def _field(coef: np.ndarray, bounds: Bounds, xs: np.ndarray, ys: np.ndarray, w: int, h: int) -> np.ndarray:
    """Evaluate d (pixels) with inputs clamped to the fitted region, so the
    polynomial never extrapolates into the margins."""
    x0, x1, y0, y1 = bounds
    xn = np.clip((xs - w / 2) / w, x0, x1)
    yn = np.clip(ys / h, y0, y1)
    return _design(xn, yn) @ coef


@dataclass(frozen=True)
class FieldStep:
    """Vertical displacement field, fit on a preview at `scale`: an output
    pixel (x, y) reads its input at (x, y_src) with y_src = y + d(x, y_src)."""
    coef: np.ndarray
    bounds: Bounds
    work: Size                                         # preview (w, h)
    scale: float

    def back(self, xs: np.ndarray, ys: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        sw, sh = self.work
        xw, yw = xs * self.scale, ys * self.scale
        src = yw
        for _ in range(3):                             # fixed-point inversion
            src = yw + _field(self.coef, self.bounds, xw, src, sw, sh)
        return xs, src / self.scale


def fit_text_field(preview: np.ndarray, scale: float) -> FieldStep | None:
    """Field that flattens curled / keystoned text lines, or None unless the
    fit is confident and clearly helps."""
    gray, _ = work_gray(preview)
    sh, sw = gray.shape
    fit = fit_displacement(_line_samples(ink_mask(gray)), sw, sh)
    if fit is None:
        return None
    coef, bounds, rms_flat, rms_fit = fit
    if rms_flat < 1.0 or rms_fit > 0.6 * rms_flat:
        return None                                    # already flat / no gain
    gy, gx = np.mgrid[0:sh:4, 0:sw:4].astype(np.float64)
    if np.abs(_field(coef, bounds, gx, gy, sw, sh)).max() > MAX_DISPLACEMENT * sh:
        return None
    return FieldStep(coef, bounds, (sw, sh), scale)


def straighten_warp(src: np.ndarray, base: Chain) -> Chain:
    """Extend `base` with deskew, text-line flattening and keystone steps.
    Each estimator sees a fresh preview of the SOURCE through the chain so
    far — never a re-warp of the previous pass's output."""
    chain = base
    preview, _valid, _scale = chain.preview(src)
    angle = estimate_skew(preview)
    if angle:
        chain = chain.then(*rotation_step(angle, chain.size))
    preview, _valid, scale = chain.preview(src)
    field = fit_text_field(preview, scale)
    if field is not None:
        chain = chain.then(field, chain.size)
    preview, valid, scale = chain.preview(src)
    found = keystone_step(preview, valid, scale, chain.size)
    if found is not None:
        chain = chain.then(*found)
    return chain


def _apply(color: np.ndarray, chain: Chain) -> np.ndarray:
    return chain.render(color)[0]


def straighten(color: np.ndarray) -> np.ndarray:
    """All passes on a ready-cropped page image; the input itself comes back
    when nothing needed fixing."""
    return _apply(color, straighten_warp(color, Chain.identity(color)))


def deskew(color: np.ndarray) -> np.ndarray:
    chain = Chain.identity(color)
    angle = estimate_skew(color)
    return _apply(color, chain.then(*rotation_step(angle, chain.size)) if angle else chain)


def dewarp_text_lines(color: np.ndarray) -> np.ndarray:
    chain = Chain.identity(color)
    preview, _valid, scale = chain.preview(color)
    field = fit_text_field(preview, scale)
    return _apply(color, chain.then(field, chain.size) if field is not None else chain)
