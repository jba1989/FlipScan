"""Stage 8: figures — crop LLM-reported regions from the corrected color frames.

The LLM bbox is approximate: find the figure block it points at (or, failing
that, expand it and snap to content) on the full-res corrected frame, then
clean the crop up for reading. Cropped images are inserted at their
[[region-N]] placeholders in the page markdown.
"""

from __future__ import annotations

import re
import string
from pathlib import Path

import cv2
import numpy as np

from ..imaging import figure_block_bbox, order_quad, sharpness
from ..workspace import Workspace

EXPAND = 0.075          # grow the LLM bbox by 7.5% per side before snapping
SNAP_MARGIN = 8         # px kept around detected content
MIN_FIGURE_SHARPNESS = 40.0


def file_ref(path) -> str | None:
    """Content fingerprint of a page image — crop coordinates are only valid
    against the exact image they were drawn on (preprocess can reframe a
    page, silently shifting every normalized coordinate)."""
    import hashlib
    from pathlib import Path
    p = Path(path)
    if not p.exists():
        return None
    return hashlib.sha1(p.read_bytes()).hexdigest()[:16]


def crop_from_region(color: np.ndarray, region: dict) -> np.ndarray | None:
    """Reproduce a stored crop exactly: perspective-correct quad_norm when the
    user skewed the corners, else a straight bbox_norm slice."""
    from .preprocess import correct_page

    h, w = color.shape[:2]
    if region.get("quad_norm") and len(region["quad_norm"]) == 4:
        quad = order_quad(np.clip(np.array(region["quad_norm"], dtype=np.float64), 0, 1))
        crop = correct_page(color, quad)
    elif region.get("bbox_norm"):
        x0, y0, x1, y1 = region["bbox_norm"]
        crop = color[int(y0 * h):int(y1 * h), int(x0 * w):int(x1 * w)]
    else:
        return None
    if crop.shape[0] < 10 or crop.shape[1] < 10:
        return None
    return crop


def snap_bbox(gray: np.ndarray, x0: int, y0: int, x1: int, y1: int) -> tuple[int, int, int, int]:
    """Tighten a bbox to the content inside it (anything deviating from the
    page background, estimated from the crop's border pixels)."""
    crop = gray[y0:y1, x0:x1]
    if crop.size == 0:
        return x0, y0, x1, y1
    border = np.concatenate([crop[0], crop[-1], crop[:, 0], crop[:, -1]])
    bg = float(np.median(border))
    mask = np.abs(crop.astype(np.int16) - bg) > 25
    rows, cols = np.any(mask, axis=1), np.any(mask, axis=0)
    if not rows.any() or not cols.any():
        return x0, y0, x1, y1
    ry, cx = np.where(rows)[0], np.where(cols)[0]
    return (
        max(0, x0 + int(cx[0]) - SNAP_MARGIN),
        max(0, y0 + int(ry[0]) - SNAP_MARGIN),
        min(gray.shape[1], x0 + int(cx[-1]) + 1 + SNAP_MARGIN),
        min(gray.shape[0], y0 + int(ry[-1]) + 1 + SNAP_MARGIN),
    )


def enhance_figure(crop: np.ndarray, upscale: float = 2.0,
                   max_long_edge: int = 2400) -> np.ndarray:
    """Make a video-frame crop read like a scan: per-channel levels (the paper
    goes white, the grey/blue cast of room light goes away, colored fills
    keep their hue), a Lanczos upscale, and an unsharp mask on lightness
    only so red/green candles don't get color fringes.

    The white point comes from the crop's border, which is paper for a
    figure set on a page; a full-bleed dark photo has no paper there, so its
    levels are left alone rather than blown out."""
    border = np.concatenate([crop[:3].reshape(-1, 3), crop[-3:].reshape(-1, 3),
                             crop[:, :3].reshape(-1, 3), crop[:, -3:].reshape(-1, 3)])
    out = crop
    if float(np.median(border)) > 150:
        lo = np.minimum(np.percentile(crop.reshape(-1, 3), 1, axis=0), 60.0)
        hi = np.percentile(border, 90, axis=0)
        span = np.maximum(hi - lo, 40.0)      # flat crops: don't blow up noise
        out = np.clip((crop.astype(np.float32) - lo) * (255.0 / span), 0, 255)
        out = out.astype(np.uint8)
    h, w = out.shape[:2]
    scale = min(upscale, max_long_edge / max(h, w))
    if scale != 1.0:
        out = cv2.resize(out, (round(w * scale), round(h * scale)),
                         interpolation=cv2.INTER_LANCZOS4 if scale > 1 else cv2.INTER_AREA)
    lab = cv2.cvtColor(out, cv2.COLOR_BGR2LAB)
    lum = lab[:, :, 0]
    blur = cv2.GaussianBlur(lum, (0, 0), 1.2 * max(scale, 1.0))
    lab[:, :, 0] = cv2.addWeighted(lum, 1.6, blur, -0.6, 0)
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def write_figure(path, crop: np.ndarray, cfg: dict) -> None:
    """Write the figure file for an automatic crop, per [figures] config.

    PNG keeps the faint gridlines and small print that lossy formats and
    denoising smear; the upscale is what costs bytes, so when the file would
    exceed max_kb the scale steps down (to none, then below 1x for a huge
    crop) until it fits."""
    fcfg = cfg.get("figures", {})
    params = [cv2.IMWRITE_PNG_COMPRESSION, 9]
    budget = int(fcfg.get("max_kb", 1000)) * 1024
    if not fcfg.get("enhance", True):
        cv2.imwrite(str(path), crop, params)
        return
    scale = float(fcfg.get("upscale", 2.0))
    while True:
        out = enhance_figure(crop, scale)
        ok, buf = cv2.imencode(".png", out, params)
        if not ok:
            raise RuntimeError(f"PNG encode failed for {path}")
        if len(buf) <= budget or min(out.shape[:2]) < 200:
            break
        scale *= 0.85
    Path(path).write_bytes(buf.tobytes())


def is_whole_page(box: tuple[int, int, int, int], w: int, h: int) -> bool:
    """A "figure" covering basically the whole page is a model miss (a
    chapter-title page, a full-page tint), not something to crop out."""
    bw, bh = (box[2] - box[0]) / w, (box[3] - box[1]) / h
    return bw * bh > 0.85 or (bw > 0.93 and bh > 0.93)


def auto_crop_box(color: np.ndarray, gray: np.ndarray,
                  bbox_norm: list[float]) -> tuple[int, int, int, int]:
    """Pixel box for a model-reported region: the figure block it points at,
    else the old expand-and-snap around the model's box."""
    h, w = gray.shape
    block = figure_block_bbox(color, bbox_norm)
    if block is not None:
        x0, y0, x1, y1 = block
        return (max(0, x0 - SNAP_MARGIN), max(0, y0 - SNAP_MARGIN),
                min(w, x1 + SNAP_MARGIN), min(h, y1 + SNAP_MARGIN))
    bx0, by0, bx1, by1 = bbox_norm
    dx, dy = (bx1 - bx0) * EXPAND, (by1 - by0) * EXPAND
    return snap_bbox(gray, int(max(0.0, bx0 - dx) * w), int(max(0.0, by0 - dy) * h),
                     int(min(1.0, bx1 + dx) * w), int(min(1.0, by1 + dy) * h))


def run(ws: Workspace, cfg: dict, log=print) -> None:
    fig_dir = ws.dir("figures")
    total = 0
    for page in ws.manifest["pages"]:
        regions = page.get("regions") or []
        if (not regions or not page.get("md") or not page.get("color")
                or page.get("status") in ("duplicate", "deleted")):
            if (not regions and page.get("figures")
                    and page.get("status") not in ("duplicate", "deleted")):
                # no regions on this page (any more) — a leftover figures
                # list points at crops of an OLDER image; drop it
                page["figures"] = []
            continue
        color = cv2.imread(str(ws.root / page["color"]))
        if color is None:
            continue
        gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        md_path = ws.root / page["md"]
        md = md_path.read_text(encoding="utf-8")
        page_figs = []

        for i, region in enumerate(regions):
            if region.get("deleted"):
                continue
            letter = string.ascii_lowercase[i % 26]
            name = f"{page['id']}_{letter}.png"
            rel = f"figures/{name}"
            if region.get("user_crop") or region.get("auto_refined"):
                # a human placed (or accepted) this crop — keep it in the
                # figure list and markdown, and regenerate the file from the
                # stored geometry if it went missing — but ONLY when the page
                # image is the one the crop was drawn on; regenerating against
                # a reframed image cuts out the wrong part of the page
                if not (fig_dir / name).exists():
                    # a re-acquired close-up is a standalone photo; it cannot be
                    # re-derived from the page image, so never try to crop one
                    if region.get("own_image"):
                        region["stale_crop"] = True
                        continue
                    ref = region.get("color_ref")
                    if not ref or ref != file_ref(ws.root / page["color"]):
                        region["stale_crop"] = True
                        continue
                    crop = crop_from_region(color, region)
                    if crop is None:
                        continue
                    cv2.imwrite(str(fig_dir / name), crop)
                region.pop("stale_crop", None)
                page_figs.append(rel)
                img_md = f"![{region.get('caption') or ''}]({rel})"
                placeholder = f"[[region-{i}]]"
                if placeholder in md:
                    md = md.replace(placeholder, img_md)
                elif rel not in md:
                    md = md.rstrip() + f"\n\n{img_md}\n"
                continue
            x0, y0, x1, y1 = auto_crop_box(color, gray, region["bbox_norm"])
            if x1 - x0 < 20 or y1 - y0 < 20 or is_whole_page((x0, y0, x1, y1), w, h):
                continue
            crop = color[y0:y1, x0:x1]
            write_figure(fig_dir / name, crop, cfg)
            page_figs.append(rel)
            total += 1

            # judged on the raw crop: sharpening would pass every blurry one
            if sharpness(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)) < MIN_FIGURE_SHARPNESS:
                page["figure_quality"] = True
                if page["status"] == "ok":
                    page["status"] = "suspect"

            img_md = f"![{region.get('caption') or ''}]({rel})"
            placeholder = f"[[region-{i}]]"
            if placeholder in md:
                md = md.replace(placeholder, img_md)
            elif rel not in md:  # don't stack a second copy on re-runs
                md = md.rstrip() + f"\n\n{img_md}\n"

        # hygiene: older stage versions appended the same image again on every
        # run — keep only the first reference to each figure file
        seen_imgs: set[str] = set()
        kept_lines = []
        for ln in md.splitlines():
            im = re.match(r"!\[[^\]]*\]\((figures/[^)]+)\)\s*$", ln.strip())
            if im:
                if im.group(1) in seen_imgs:
                    continue
                seen_imgs.add(im.group(1))
            kept_lines.append(ln)
        md = re.sub(r"\n{3,}", "\n\n", "\n".join(kept_lines))

        md_path.write_text(md, encoding="utf-8")
        page["figures"] = page_figs
    ws.save()
    ws.stage_done("figures", figure_count=total)
    log(f"  {total} figures cropped -> figures/")
