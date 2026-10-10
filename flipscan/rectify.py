"""Content-based page straightening: rectify from the printed text lines.

A book page's outline is a poor guide — the desk often shares its colour, the
page bows near the spine, and the corners hide under thumbs — so a quad warp
alone leaves text tilted and curled. The text itself is the reliable ruler:
every printed line should end up horizontal and straight.

Two passes, each gated so a page without enough evidence is left untouched
(the invoshot lesson: an uncertain observation must degrade to a no-op):

1. ``estimate_skew``: whole-page rotation by projection-profile search.
2. ``dewarp_text_lines``: residual curl / keystone. Text-line centerlines are
   fit jointly to one smooth vertical displacement field, then remapped flat.
"""

from __future__ import annotations

import cv2
import numpy as np

from .keystone import correct_keystone
from .workres import ink_mask, paper_fill, work_gray

MAX_SKEW_DEG = 10.0        # beyond this it's a mis-framed shot, not skew
MIN_SKEW_DEG = 0.3         # smaller rotations are not worth a resample
MIN_SKEW_GAIN = 1.15       # best profile must beat the unrotated one by 15%
MIN_LINES = 5              # fewer text lines can't pin down a 2-D field
MAX_DISPLACEMENT = 0.06    # of page height; more means the fit went wild
X_DEGREE, Y_DEGREE = 4, 2  # displacement field d(x, y) polynomial orders


def _rotate(img: np.ndarray, angle: float) -> np.ndarray:
    """Rotate about the centre, growing the canvas so no corner is cut."""
    h, w = img.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    cos, sin = abs(m[0, 0]), abs(m[0, 1])
    nw, nh = int(round(h * sin + w * cos)), int(round(h * cos + w * sin))
    m[0, 2] += nw / 2 - w / 2
    m[1, 2] += nh / 2 - h / 2
    return cv2.warpAffine(img, m, (nw, nh), flags=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=paper_fill(img))


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


def deskew(color: np.ndarray) -> np.ndarray:
    angle = estimate_skew(color)
    return color if angle == 0.0 else _rotate(color, angle)


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


def dewarp_text_lines(color: np.ndarray) -> np.ndarray:
    """Flatten curled / keystoned text so every line runs straight; the input
    is returned untouched unless the fit is confident and clearly helps."""
    gray, scale = work_gray(color)
    sh, sw = gray.shape
    fit = fit_displacement(_line_samples(ink_mask(gray)), sw, sh)
    if fit is None:
        return color
    coef, bounds, rms_flat, rms_fit = fit
    if rms_flat < 1.0 or rms_fit > 0.6 * rms_flat:
        return color                                   # already flat / no gain

    # coarse grid in work coords, upsampled: the field is smooth by design
    gh, gw = max(2, sh // 4), max(2, sw // 4)
    ys, xs = np.mgrid[0:gh, 0:gw].astype(np.float64)
    xs *= (sw - 1) / (gw - 1)
    ys *= (sh - 1) / (gh - 1)
    d = _field(coef, bounds, xs, ys, sw, sh)
    if np.abs(d).max() > MAX_DISPLACEMENT * sh:
        return color
    src = ys + d
    for _ in range(2):                                 # invert y_out = y_src - d
        src = ys + _field(coef, bounds, xs, src, sw, sh)
    h, w = color.shape[:2]
    map_y = cv2.resize((src / scale).astype(np.float32), (w, h),
                       interpolation=cv2.INTER_CUBIC)
    map_x = np.broadcast_to(np.arange(w, dtype=np.float32)[None, :], (h, w)).copy()
    return cv2.remap(color, map_x, map_y, cv2.INTER_CUBIC,
                     borderMode=cv2.BORDER_CONSTANT, borderValue=paper_fill(color))


def straighten(color: np.ndarray) -> np.ndarray:
    """Deskew, flatten the residual curl, then square up the trapezoid
    (keystone.py). Safe on any page image."""
    return correct_keystone(dewarp_text_lines(deskew(color)))
