"""Keystone correction: page verticals upright after a tilted-camera shot."""

import cv2
import numpy as np

from flipscan import keystone, workres


def _page(w=900, h=1200):
    """Bright page with a printed frame (two long verticals) and text rows."""
    img = np.full((h, w, 3), 235, np.uint8)
    cv2.rectangle(img, (120, 150), (780, 1050), (40, 40, 40), 3)
    for y in range(200, 1000, 40):
        cv2.rectangle(img, (160, y), (740, y + 16), (60, 60, 60), -1)
    return img


def _warp(img, dst):
    h, w = img.shape[:2]
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    m = cv2.getPerspectiveTransform(src, np.float32(dst))
    return cv2.warpPerspective(img, m, (w, h), borderValue=(90, 110, 130))


def _mean_lean(img):
    """Length-weighted lean (degrees) of the long near-vertical segments."""
    gray, _ = workres.work_gray(img)
    seg = keystone._drop_fill_edges(keystone.vertical_segments(gray), gray)
    assert len(seg), "no verticals found"
    d = seg[:, 2:] - seg[:, :2]
    length = np.hypot(d[:, 0], d[:, 1])
    lean = np.degrees(np.arctan2(np.abs(d[:, 0]), np.abs(d[:, 1])))
    return float((lean * length).sum() / length.sum())


def test_trapezoid_is_squared():
    # camera tilted toward the top: the top edge shrinks inward
    page = _warp(_page(), [[90, 0], [810, 0], [900, 1200], [0, 1200]])
    assert _mean_lean(page) > 2.0
    out = keystone.correct_keystone(page)
    assert out is not page
    assert _mean_lean(out) < 0.5


def test_sheared_page_is_uprighted():
    page = _warp(_page(), [[60, 0], [960, 0], [900, 1200], [0, 1200]])
    out = keystone.correct_keystone(page)
    assert _mean_lean(out) < 0.5


def test_square_page_is_untouched():
    page = _page()
    assert keystone.correct_keystone(page) is page


def test_page_without_verticals_is_untouched():
    page = np.full((1200, 900, 3), 235, np.uint8)
    for y in range(200, 1000, 40):
        cv2.rectangle(page, (160, y), (740, y + 16), (60, 60, 60), -1)
    assert keystone.correct_keystone(page) is page


def test_implausible_correction_is_refused():
    # a vanishing point this close would mean a >1.3x row-scale change
    vp = np.array([0.0, -1.0, 1.0 / 900])
    assert keystone.keystone_homography(vp / np.linalg.norm(vp), 1200) is None
