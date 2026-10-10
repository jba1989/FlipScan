"""Content-based straightening: deskew and text-line dewarp."""

import cv2
import numpy as np

from flipscan import rectify, workres


def _page(curl=0.0, w=900, h=1200):
    """Fake printed page: rows of word-like strokes. `curl` bows every line
    (pixels of sag at the edges, like paper bending toward the spine)."""
    img = np.full((h, w, 3), 240, np.uint8)
    rng = np.random.default_rng(0)
    for y in range(120, h - 120, 40):
        x = 80
        while x < w - 100:
            bw = int(rng.integers(14, 24))
            dy = int(curl * ((x - w / 2) / (w / 2)) ** 2)
            cv2.rectangle(img, (x, y + dy), (x + bw, y + dy + 18), (30, 30, 30), -1)
            x += bw + 6
    return img


def _rotate(img, angle):
    h, w = img.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(img, m, (w, h), borderValue=(240, 240, 240))


def _line_bow(img):
    """Mean vertical spread of each detected text line's centerline."""
    gray, _ = workres.work_gray(img)
    lines = rectify._line_samples(workres.ink_mask(gray))
    assert lines, "no text lines found"
    return float(np.mean([np.ptp(l[:, 1]) for l in lines]))


def test_estimate_skew_recovers_rotation():
    angle = rectify.estimate_skew(_rotate(_page(), 4.0))
    assert abs(angle + 4.0) < 0.5           # undo the 4 degree tilt


def test_straight_page_has_no_skew():
    assert rectify.estimate_skew(_page()) == 0.0


def test_blank_page_is_untouched():
    blank = np.full((1200, 900, 3), 240, np.uint8)
    assert rectify.straighten(blank) is blank


def test_flat_page_is_not_resampled():
    page = _page()
    assert rectify.dewarp_text_lines(page) is page


def test_dewarp_flattens_curled_lines():
    curled = _page(curl=30)
    flat = rectify.dewarp_text_lines(curled)
    assert flat is not curled and flat.shape == curled.shape
    assert _line_bow(flat) < 0.4 * _line_bow(curled)


def test_straighten_handles_tilt_and_curl_together():
    page = _rotate(_page(curl=25), 3.0)
    out = rectify.straighten(page)
    # the tilted input has no measurable lines at all; deskew-only is the bar
    assert _line_bow(out) < 0.5 * _line_bow(rectify.deskew(page))


def test_vertical_text_is_left_alone():
    # 直書: columns of stacked characters have no horizontal lines to fit, so
    # neither pass may invent a rotation or a warp
    page = np.full((1200, 900, 3), 240, np.uint8)
    for x in range(100, 800, 45):
        for y in range(100, 1080, 26):
            cv2.rectangle(page, (x, y), (x + 20, y + 20), (30, 30, 30), -1)
    assert rectify.estimate_skew(page) == 0.0
    assert rectify.dewarp_text_lines(page) is page
