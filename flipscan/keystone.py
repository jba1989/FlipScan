"""Keystone correction: make the page's vertical edges vertical again.

After rectify.py the text lines run horizontal, but a camera tilted toward
the top or bottom of the book still leaves a trapezoid: the page and chart
edges lean in toward a vanishing point. Long near-vertical segments (page
edges, chart frames, boxed text) are voted into that point with RANSAC, and
one homography sends it to infinity — verticals become parallel and upright,
horizontals stay horizontal, and the foreshortened rows regain their height.

Gated like the rest of the straightening: too little or too narrow evidence,
or an implausibly strong correction, leaves the image untouched. Like every
pass it only estimates a warpchain step; the resample happens once.
"""

from __future__ import annotations

import cv2
import numpy as np

from .warpchain import Chain, Homography, Size
from .workres import work_gray

MAX_LEAN_DEG = 15.0        # segments leaning further are not page verticals
MIN_SEGMENT = 0.08         # of page height
INLIER_DEG = 0.8           # segment must point at the vanishing point this well
INLIER_COS = np.cos(np.radians(INLIER_DEG))
MIN_EVIDENCE = 0.6         # total inlier length, in page heights
MIN_SPREAD = 0.35          # inliers must span this much of the width
MAX_SCALE_RATIO = 1.3      # top vs bottom row scale; beyond it, distrust
MAX_SHEAR = np.tan(np.radians(8.0))
MAX_SHEAR_SPREAD = np.tan(np.radians(1.5))  # verticals must agree this well
RANSAC_ITERS = 500


def vertical_segments(gray: np.ndarray) -> np.ndarray:
    """Long near-vertical line segments as rows of x1, y1, x2, y2."""
    found = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD).detect(gray)[0]
    if found is None:
        return np.zeros((0, 4))
    seg = found.reshape(-1, 4).astype(np.float64)
    dx, dy = seg[:, 2] - seg[:, 0], seg[:, 3] - seg[:, 1]
    lean = np.degrees(np.arctan2(np.abs(dx), np.abs(dy)))
    keep = (np.hypot(dx, dy) > MIN_SEGMENT * gray.shape[0]) & (lean < MAX_LEAN_DEG)
    return seg[keep]


def drop_padding_edges(seg: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Discard segments along the edge of the padding that earlier steps
    exposed (rotation canvas, an outline quad past the frame): those edges
    are straight, long, and lean by exactly the previous step's angle —
    perfect fake verticals. `valid` marks pixels that came from the frame."""
    if not len(seg) or valid.all():
        return seg
    pad = cv2.dilate((~valid).astype(np.uint8), np.ones((9, 9), np.uint8))
    h, w = valid.shape
    ts = np.linspace(0.1, 0.9, 9)[None, :]
    xs = np.clip((seg[:, [0]] + ts * (seg[:, [2]] - seg[:, [0]])).astype(int), 0, w - 1)
    ys = np.clip((seg[:, [1]] + ts * (seg[:, [3]] - seg[:, [1]])).astype(int), 0, h - 1)
    return seg[pad[ys, xs].mean(axis=1) < 0.5]


def _inliers(vp: np.ndarray, mids: np.ndarray, dirs: np.ndarray) -> np.ndarray:
    """Segments whose direction points at vp (homogeneous; vp[2] may be 0)."""
    to = vp[None, :2] - mids * vp[2]
    to /= np.linalg.norm(to, axis=1, keepdims=True) + 1e-12
    return np.abs((to * dirs).sum(axis=1)) > INLIER_COS


def vanishing_point(seg: np.ndarray, w: int, h: int) -> np.ndarray | None:
    """Vertical vanishing point (homogeneous, centred coords), or None when
    the segments don't agree on one."""
    if len(seg) < 3:
        return None
    centre = np.array([w / 2, h / 2])
    p1 = np.c_[seg[:, :2] - centre, np.ones(len(seg))]
    p2 = np.c_[seg[:, 2:] - centre, np.ones(len(seg))]
    lines = np.cross(p1, p2)
    lines /= np.linalg.norm(lines[:, :2], axis=1, keepdims=True)
    length = np.hypot(*(seg[:, 2:] - seg[:, :2]).T)
    mids = (p1[:, :2] + p2[:, :2]) / 2
    dirs = (p2 - p1)[:, :2] / length[:, None]

    rng = np.random.default_rng(0)
    best_score, best_mask = 0.0, None
    for _ in range(RANSAC_ITERS):
        i, j = rng.choice(len(seg), 2, replace=False)
        vp = np.cross(lines[i], lines[j])
        if np.linalg.norm(vp) < 1e-12:
            continue
        mask = _inliers(vp / np.linalg.norm(vp), mids, dirs)
        score = float(length[mask].sum())
        if score > best_score:
            best_score, best_mask = score, mask
    if best_mask is None or best_mask.sum() < 3:
        return None
    if best_score < MIN_EVIDENCE * h or np.ptp(mids[best_mask, 0]) < MIN_SPREAD * w:
        return None
    # least-squares point closest to all inlier lines, weighted by length
    return np.linalg.svd(lines[best_mask] * length[best_mask, None])[2][-1]


def keystone_homography(vp: np.ndarray, h: int) -> np.ndarray | None:
    """Homography (centred coords) sending vp to vertical infinity while
    keeping horizontal lines horizontal; None if implausibly strong."""
    vx, vy, vw = vp
    if abs(vy) < 1e-12:
        return None
    shear, persp = -vx / vy, -vw / vy
    top, bottom = 1 + persp * (-h / 2), 1 + persp * (h / 2)
    if top <= 0 or bottom <= 0 or abs(shear) > MAX_SHEAR:
        return None
    if max(top / bottom, bottom / top) > MAX_SCALE_RATIO:
        return None
    return np.array([[1.0, shear, 0.0], [0.0, 1.0, 0.0], [0.0, persp, 1.0]])


def shear_homography(seg: np.ndarray, h: int) -> np.ndarray | None:
    """Fallback when no vanishing point holds up: if the verticals at least
    agree on one common lean (a warp that sheared the page, e.g. from a bad
    outline quad), shear them upright. None unless the lean is consistent."""
    if len(seg) < 3:
        return None
    d = seg[:, 2:] - seg[:, :2]
    d *= np.where(d[:, 1] < 0, -1.0, 1.0)[:, None]      # all pointing down
    length = np.hypot(d[:, 0], d[:, 1])
    slope = d[:, 0] / d[:, 1]                           # dx per dy
    if length.sum() < MIN_EVIDENCE * h:
        return None
    order = np.argsort(slope)
    cum = np.cumsum(length[order])
    median = slope[order][np.searchsorted(cum, cum[-1] / 2)]
    spread = np.average(np.abs(slope - median), weights=length)
    if spread > MAX_SHEAR_SPREAD or abs(median) > MAX_SHEAR:
        return None
    return np.array([[1.0, -median, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])


def keystone_step(preview: np.ndarray, valid: np.ndarray, scale: float,
                  size: Size) -> tuple[Homography, Size] | None:
    """Step that uprights the page's verticals, estimated on a preview
    rendered at `scale` of a page of `size`; None unless the vanishing point
    is well supported and the correction plausible."""
    gray, _ = work_gray(preview)
    sh, sw = gray.shape
    seg = drop_padding_edges(vertical_segments(gray), valid)
    vp = vanishing_point(seg, sw, sh)
    hc = keystone_homography(vp, sh) if vp is not None else None
    if hc is None:
        hc = shear_homography(seg, sh)
    if hc is None:
        return None
    shear, persp = hc[0, 1], hc[2, 1]
    if abs(shear) < 0.005 and abs(persp * sh) < 0.01:
        return None                                    # already square

    w, h = size
    to_c = np.array([[scale, 0, -sw / 2], [0, scale, -sh / 2], [0, 0, 1.0]])
    m = np.linalg.inv(to_c) @ hc @ to_c                # full-res pixel coords
    corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], np.float64).reshape(-1, 1, 2)
    out = cv2.perspectiveTransform(corners, m).reshape(-1, 2)
    x0, y0 = out.min(axis=0)
    x1, y1 = out.max(axis=0)
    fit = min(1.0, np.sqrt((w * h) / max((x1 - x0) * (y1 - y0), 1.0)) * 1.15)
    place = np.array([[fit, 0, -x0 * fit], [0, fit, -y0 * fit], [0, 0, 1.0]])
    out_size = (int(round((x1 - x0) * fit)), int(round((y1 - y0) * fit)))
    return Homography(place @ m), out_size


def correct_keystone(color: np.ndarray) -> np.ndarray:
    """keystone_step applied to a ready page image (the input itself comes
    back when there is nothing to fix)."""
    chain = Chain.identity(color)
    preview, valid, scale = chain.preview(color)
    found = keystone_step(preview, valid, scale, chain.size)
    return chain.then(*found).render(color)[0] if found else color
