"""Two-page spread detection, splitting, and text-block tightening."""

import cv2
import numpy as np

from flipscan.imaging import split_spread, tighten_to_text


def _text_lines(img, x0, x1, y0, y1, step=28):
    """Fake printed text: rows of short dark strokes (dense Canny edges)."""
    rng = np.random.default_rng(0)
    for y in range(y0, y1, step):
        x = x0
        while x < x1 - 20:
            w = int(rng.integers(10, 18))
            cv2.rectangle(img, (x, y), (x + w, y + 14), (30, 30, 30), -1)
            x += w + 6


def _spread(gutter_dx=0):
    """1920x1080 frame: dark desk, bright open book, text on both pages, a
    shaded fold. gutter_dx tilts the fold (top x - bottom x)."""
    img = np.full((1080, 1920, 3), (60, 90, 120), np.uint8)
    cv2.rectangle(img, (260, 60), (1660, 1020), (235, 235, 235), -1)
    # a real fold is a soft shadow, not a crisp line (no Canny edges of its own)
    shade = np.zeros((1080, 1920), np.float32)
    g_top, g_bot = 960 + gutter_dx // 2, 960 - gutter_dx // 2
    cv2.line(shade, (g_top, 60), (g_bot, 1020), 1.0, 30)
    shade = cv2.GaussianBlur(shade, (0, 0), 12)
    img = (img * (1 - 0.35 * shade[..., None])).astype(np.uint8)
    _text_lines(img, 340, 880, 140, 940)
    _text_lines(img, 1040, 1580, 140, 940)
    return img


def test_split_spread_finds_the_fold_and_two_page_quads():
    r = split_spread(_spread())
    assert r is not None
    left, right = np.array(r["left"]), np.array(r["right"])
    # the shared edge is the fold, near x = 960/1920 = 0.5
    assert abs(left[1][0] - 0.5) < 0.02 and abs(left[2][0] - 0.5) < 0.02
    assert np.allclose(left[1], right[0]) and np.allclose(left[2], right[3])
    assert left[0][0] < 0.2 and right[1][0] > 0.8


def test_split_spread_follows_a_tilted_fold():
    r = split_spread(_spread(gutter_dx=60))       # camera not square to the book
    assert r is not None
    top_x, bot_x = r["left"][1][0] * 1920, r["left"][2][0] * 1920
    assert top_x - bot_x > 30


def test_single_page_is_not_split():
    img = np.full((1080, 1920, 3), (60, 90, 120), np.uint8)
    cv2.rectangle(img, (620, 60), (1300, 1020), (235, 235, 235), -1)
    _text_lines(img, 680, 1240, 140, 940)
    assert split_spread(img) is None


def test_mid_turn_frame_is_not_split():
    # text on one side only, the other page blank (lifted/blurred mid-turn)
    img = _spread()
    cv2.rectangle(img, (1000, 100), (1640, 1000), (235, 235, 235), -1)
    assert split_spread(img) is None


def _page_with_stacked_sliver():
    page = np.full((1000, 760, 3), 235, np.uint8)
    _text_lines(page, 10, 70, 300, 700)            # sliver of the page underneath
    _text_lines(page, 200, 700, 120, 860)          # the real page
    cv2.rectangle(page, (120, 940), (165, 965), (30, 30, 30), -1)  # "008" corner
    return page


def test_tighten_drops_stacked_pages_beside_the_text_block():
    out = tighten_to_text(_page_with_stacked_sliver())
    assert out.shape[1] <= 760 - 70                # sliver cut away


def test_tighten_keeps_the_page_number_in_the_outer_corner():
    # page numbers sit in the bottom OUTER corner — the same side the stacked
    # pages show up on — so the cut goes right after the sliver, not at the text
    page = _page_with_stacked_sliver()
    out = tighten_to_text(page)
    x_off = page.shape[1] - out.shape[1]           # only the left side is cut
    assert x_off < 120 and out.shape[0] == page.shape[0]


def test_tighten_keeps_own_text_split_by_a_gutter():
    # a chart's axis labels / a short column leave a gap inside the page's
    # OWN text block — a wide run past that gap is not a stacked page
    page = np.full((1000, 760, 3), 235, np.uint8)
    _text_lines(page, 40, 460, 120, 860)
    _text_lines(page, 540, 720, 120, 860)
    assert tighten_to_text(page).shape == page.shape


def test_tighten_ignores_a_speck_in_the_middle_of_the_page():
    page = np.full((1000, 760, 3), 235, np.uint8)
    _text_lines(page, 40, 420, 120, 860)
    _text_lines(page, 470, 500, 400, 440)          # stray mark mid-page
    _text_lines(page, 560, 720, 120, 860)          # rest of this page's text
    assert tighten_to_text(page).shape == page.shape


def test_tighten_never_cuts_at_the_spine():
    # a right-hand page's stacked pages lie on its RIGHT; the same sliver
    # pattern on its left (spine side) is its own text — keep it
    page = _page_with_stacked_sliver()
    assert tighten_to_text(page, side="right").shape == page.shape
    assert tighten_to_text(page, side="left").shape[1] <= 760 - 70


def test_tighten_keeps_page_without_confident_block():
    blank = np.full((1000, 760, 3), 235, np.uint8)
    assert tighten_to_text(blank).shape == blank.shape


# ---------------- select: pages from spreads

def _ws_with_spread(tmp_path, direction="forward"):
    from flipscan.workspace import Workspace
    ws = Workspace.create(tmp_path / "book", videos=[])
    ws.manifest["videos"] = [{"id": "v0", "direction": direction}]
    ws.frames_dir("v0").mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(ws.frames_dir("v0") / "f000001.jpg"), _spread())
    ws.manifest["pages"] = [{"id": "p0000", "video": "v0", "canonical": "v0_f000001",
                             "status": "ok", "md": "pages/p0000.md"}]
    return ws


def _cfg(on=True):
    from flipscan.config import DEFAULTS
    return {**DEFAULTS, "preprocess": {**DEFAULTS["preprocess"], "split_spreads": on}}


def test_select_splits_a_spread_into_two_pages(tmp_path):
    from flipscan.stages.select import split_spreads
    ws = _ws_with_spread(tmp_path)
    assert split_spreads(ws, _cfg(), log=lambda m: None) == 1
    left, right = ws.manifest["pages"]
    assert (left["id"], left["side"]) == ("p0000", "left")
    assert (right["id"], right["side"]) == ("p0000r", "right")
    assert left["spread_order"] < right["spread_order"]
    # the whole-spread transcription no longer matches either crop
    assert "md" not in left and "md" not in right


def test_select_split_is_idempotent_and_keeps_its_transcriptions(tmp_path):
    from flipscan.stages.select import split_spreads
    ws = _ws_with_spread(tmp_path)
    split_spreads(ws, _cfg(), log=lambda m: None)
    ws.manifest["pages"][0]["md"] = "pages/p0000.md"     # transcribed half
    split_spreads(ws, _cfg(), log=lambda m: None)         # re-run
    assert [p["id"] for p in ws.manifest["pages"]] == ["p0000", "p0000r"]
    assert ws.manifest["pages"][0]["md"] == "pages/p0000.md"


def test_turning_split_off_restores_one_page(tmp_path):
    from flipscan.stages.select import split_spreads
    ws = _ws_with_spread(tmp_path)
    split_spreads(ws, _cfg(), log=lambda m: None)
    split_spreads(ws, _cfg(on=False), log=lambda m: None)
    (page,) = ws.manifest["pages"]
    assert "side" not in page and "spread_quad" not in page


def test_reverse_video_orders_right_half_first(tmp_path):
    from flipscan.stages.select import split_spreads
    from flipscan.stages.transcribe import _frame_no
    ws = _ws_with_spread(tmp_path, direction="reverse")
    split_spreads(ws, _cfg(), log=lambda m: None)
    left, right = ws.manifest["pages"]
    assert _frame_no(right) < _frame_no(left)


def test_preprocess_writes_each_half(tmp_path):
    from flipscan.stages.preprocess import preprocess_page
    from flipscan.stages.select import split_spreads
    ws = _ws_with_spread(tmp_path)
    split_spreads(ws, _cfg(), log=lambda m: None)
    for p in ws.manifest["pages"]:
        preprocess_page(ws, p, _cfg())
        img = cv2.imread(str(ws.root / p["color"]))
        assert img.shape[0] > img.shape[1]                 # portrait page, not spread


def test_preprocess_pads_spread_quads_like_single_pages(tmp_path):
    # page numbers sit at the very edge of the detected page; the quad is
    # padded (quad_pad) exactly as the single-page path does
    from flipscan.stages.preprocess import preprocess_page
    from flipscan.stages.select import split_spreads
    ws = _ws_with_spread(tmp_path)
    split_spreads(ws, _cfg(), log=lambda m: None)
    left = ws.manifest["pages"][0]
    q = np.array(left["spread_quad"]) * [1920, 1080]
    quad_h = (np.linalg.norm(q[3] - q[0]) + np.linalg.norm(q[2] - q[1])) / 2
    preprocess_page(ws, left, _cfg())
    out_h = cv2.imread(str(ws.root / left["color"])).shape[0]
    assert out_h > quad_h * 1.03
