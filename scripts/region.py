"""Anatomical region from image content — heuristic baseline until a model replaces it.

Verified on the training set (GE Lunar Prodigy exports):
  * spine scans are 300 px wide, hip scans 280 px wide (one prosthesis case: 248);
  * on a hip scan the femoral shaft occupies the lower part of the frame;
    right hip -> shaft in the left half of the image, left hip -> right half.
    Caveat: in ~8 % of studies the scanner displays the left hip mirrored, so both
    hips of a study land on the same side. The side is therefore a best-effort guess
    (correct for 92/100 training studies); the region itself (spine vs hip) is exact.
"""
from __future__ import annotations

import numpy as np

SPINE = "spine"
HIP_R = "hip_R"
HIP_L = "hip_L"
UNKNOWN = "unknown"

_BONE_THRESHOLD = 100  # uint8 intensity above which a pixel is treated as bone


def hip_side(pixels: np.ndarray) -> str:
    """Side of a single-hip frame from the horizontal position of the femoral shaft."""
    bottom = pixels[int(pixels.shape[0] * 0.75):, :]
    cols = (bottom > _BONE_THRESHOLD).sum(axis=0)
    if cols.sum() == 0:
        return UNKNOWN
    w = pixels.shape[1]
    cx = (cols * np.arange(w)).sum() / cols.sum()
    return HIP_R if cx < w / 2 else HIP_L


def detect_region(pixels: np.ndarray) -> str:
    """spine / hip_R / hip_L / unknown."""
    w = pixels.shape[1]
    if w >= 295:
        return SPINE
    if w >= 240:
        return hip_side(pixels)
    return UNKNOWN


def region_group(region: str) -> str:
    """Coarse group used for per-region models: 'spine' | 'hip' | 'unknown'."""
    return SPINE if region == SPINE else ("hip" if region.startswith("hip") else UNKNOWN)
