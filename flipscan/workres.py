"""Shared bits of the page-straightening passes (rectify.py, keystone.py)."""

from __future__ import annotations

import cv2
import numpy as np

WORK_LONG_EDGE = 1200      # analysis resolution; warps still run at full res
PAPER = 255                # fill for canvas a warp exposes; keystone filters it


def work_gray(color: np.ndarray) -> tuple[np.ndarray, float]:
    """Grayscale copy downscaled to the analysis resolution, and its scale."""
    gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY) if color.ndim == 3 else color
    scale = min(1.0, WORK_LONG_EDGE / max(gray.shape[:2]))
    if scale < 1.0:
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return gray, scale


def paper_fill(img: np.ndarray) -> int | tuple[int, ...]:
    """borderValue that pads with white paper instead of black streaks."""
    return (PAPER,) * img.shape[2] if img.ndim == 3 else PAPER


def ink_mask(gray: np.ndarray) -> np.ndarray:
    """Binary mask of printed strokes, robust to uneven lighting."""
    return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                 cv2.THRESH_BINARY_INV, 31, 15)
