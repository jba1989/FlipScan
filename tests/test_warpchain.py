"""Composed page geometry: one resample for a whole chain of steps."""

import cv2
import numpy as np

from flipscan.imaging import tighten_to_text
from flipscan.rectify import rotation_step
from flipscan.warpchain import Chain, Homography, translation


def _img(w=400, h=300):
    rng = np.random.default_rng(1)
    img = cv2.GaussianBlur(rng.integers(0, 255, (h, w, 3), dtype=np.uint8), (0, 0), 3)
    return cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)


def test_identity_chain_returns_the_source_itself():
    img = _img()
    out, valid = Chain.identity(img).render(img)
    assert out is img and valid.all()


def test_composed_homographies_match_one_warp():
    img = _img()
    a = np.array([[1.0, 0.05, 10], [0.02, 1.0, -5], [0.0, 0.0002, 1.0]])
    b = np.array([[0.9, 0.0, 20], [0.0, 1.1, 0], [0.0, 0.0, 1.0]])
    chain = Chain.identity(img).then(Homography(a), (400, 300)).then(Homography(b), (400, 300))
    out, valid = chain.render(img)
    ref = cv2.warpPerspective(img, b @ a, (400, 300), flags=cv2.INTER_CUBIC)
    inner = cv2.erode(valid.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
    assert np.abs(out.astype(int) - ref)[inner].mean() < 2.0


def test_translation_is_a_crop():
    img = _img()
    out, valid = Chain.identity(img).then(translation(50, 30), (100, 80)).render(img)
    assert np.abs(out.astype(int) - img[30:110, 50:150]).max() <= 2 and valid.all()


def test_clip_keeps_pixels_beyond_the_crop_out():
    # rotating a crop must not pull in the frame around it
    img = _img()
    crop = Chain.identity(img).crop(translation(100, 75), (200, 150))
    out, valid = crop.then(*rotation_step(20.0, crop.size)).render(img)
    assert not valid[0, 0] and (out[0, 0] == 255).all()   # corner: padding
    assert valid[out.shape[0] // 2, out.shape[1] // 2]


def test_tighten_measures_the_edge_from_real_content():
    # a white pad (from a warp) wider than the 3% edge band sits outside the
    # stacked-page sliver; with the content extent the sliver still goes
    page = np.full((1000, 860, 3), 235, np.uint8)
    page[:, :100] = 255                                   # padding
    for y in range(300, 700, 28):
        cv2.rectangle(page, (110, y), (160, y + 14), (30, 30, 30), -1)    # sliver
    for y in range(120, 860, 28):
        x = 300
        while x < 800:
            cv2.rectangle(page, (x, y), (x + 14, y + 14), (30, 30, 30), -1)
            x += 20
    assert tighten_to_text(page, "left").shape == page.shape           # blind
    valid = np.ones(page.shape[:2], bool)
    valid[:, :100] = False
    assert tighten_to_text(page, "left", valid).shape[1] < 860 - 150
