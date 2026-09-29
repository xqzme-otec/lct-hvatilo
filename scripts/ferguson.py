"""Scoliosis angle by the Ferguson method from vertebra centers.

Ferguson: take the centers of the two end vertebrae of a curve and of its apex vertebra;
the angle between the lines lower-end -> apex and apex -> upper-end is the curve angle
(the angle between their perpendiculars is the same, so perpendiculars are not needed).

Only centers are used, so it works with the axis-aligned vertebra boxes of our annotation
(Cobb needs endplate tilt, which the annotation does not have on 90 of 99 images).
The ends are chosen automatically: every pair of vertebrae with at least one between them
is tried, the apex is the vertebra farthest from the chord, the largest angle wins.

Input vertebrae are dicts as produced by `spine_from_result` (scripts/train_spine_pose.py):
{"level", "top": [x, y, conf], "bottom": [x, y, conf], "box": [x1, y1, x2, y2] (optional)}.
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np

# Not Cobb's 10°: Ferguson reads lower, and on our images it is ~6° on healthy spines from
# point noise alone. 8° on the doctor's points vs `scoliosis_v2`: sensitivity 0.83, specificity 0.76
# (AUC 0.855). Re-check on out-of-fold model predictions before relying on it.
SCOLIOSIS_LIMIT_DEG = 8.0


def vertebra_center(v: dict, kpt_conf: float = 0.5) -> Optional[tuple[float, float]]:
    """Midpoint of the upper and lower keypoints; box center if a keypoint is unreliable."""
    (tx, ty, tc), (bx, by, bc) = v["top"], v["bottom"]
    if tc >= kpt_conf and bc >= kpt_conf:
        return (tx + bx) / 2, (ty + by) / 2
    if v.get("box") is not None:
        x1, y1, x2, y2 = v["box"]
        return (x1 + x2) / 2, (y1 + y2) / 2
    return None


def _turn_deg(a: np.ndarray, k: np.ndarray, b: np.ndarray) -> float:
    """Signed angle between a->k and k->b; positive = apex to the right of the chord (image x grows right)."""
    u, w = k - a, b - k
    return math.degrees(math.atan2(u[1] * w[0] - u[0] * w[1], u @ w))


def ferguson(vertebrae: list[dict], kpt_conf: float = 0.5) -> dict:
    """Largest Ferguson angle over the visible spine.

    Returns {"angle_deg": signed angle or nan, "abs_deg", "lower", "apex", "upper": levels
    of the end and apex vertebrae (None if fewer than 3 usable vertebrae), "scoliosis": bool}.
    """
    pts = [(v.get("level"), vertebra_center(v, kpt_conf)) for v in vertebrae]
    pts = sorted([(lvl, c) for lvl, c in pts if c is not None], key=lambda p: -p[1][1])  # bottom first
    best = {"angle_deg": float("nan"), "abs_deg": float("nan"),
            "lower": None, "apex": None, "upper": None, "scoliosis": False}
    if len(pts) < 3:
        return best
    c = np.array([p[1] for p in pts], dtype=float)
    for i in range(len(c) - 2):
        for j in range(i + 2, len(c)):
            chord = c[j] - c[i]
            n = np.linalg.norm(chord)
            if n < 1:
                continue
            between = c[i + 1:j]
            dist = np.abs(chord[0] * (between[:, 1] - c[i, 1]) - chord[1] * (between[:, 0] - c[i, 0])) / n
            k = i + 1 + int(np.argmax(dist))
            ang = _turn_deg(c[i], c[k], c[j])
            if not abs(ang) <= abs(best["angle_deg"]):  # also replaces the initial nan
                best.update(angle_deg=ang, abs_deg=abs(ang),
                            lower=pts[i][0], apex=pts[k][0], upper=pts[j][0])
    best["scoliosis"] = bool(best["abs_deg"] > SCOLIOSIS_LIMIT_DEG)
    return best
