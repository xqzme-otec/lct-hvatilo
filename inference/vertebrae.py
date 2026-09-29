import logging
import math
from logging import Logger

import numpy as np
from ultralytics import YOLO
from PIL import Image

from scripts.ferguson import ferguson

logger = Logger(__name__)
logger.setLevel(logging.DEBUG)

LEVELS_BOTTOM_UP = ["L5", "L4", "L3", "L2", "L1", "Th12", "Th11", "Th10"]
AXIS_LIMIT_DEG = 5.0
DUP_OVERLAP = 0.4      # two boxes covering > 40 % of the smaller one = the same vertebra boxed twice
MIDDLE_OVERLAP = 0.3   # a box covered > 30 % by both neighbours straddles two vertebrae


def fit_axis_deg(pts: np.ndarray) -> float:
    """Signed angle of the least-squares line x = a*y + b to the Y axis (positive = bottom to the right)."""
    if len(pts) < 2 or np.ptp(pts[:, 1]) < 1:
        return float("nan")
    a = np.polyfit(pts[:, 1], pts[:, 0], 1)[0]
    return math.degrees(math.atan(a))


def overlap(a: np.ndarray, b: np.ndarray) -> float:
    """Share of the smaller box covered by the other one."""
    iw = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    return iw * ih / max(min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1])), 1e-9)


def suppress_duplicates(boxes: np.ndarray, dup: float = DUP_OVERLAP, mid: float = MIDDLE_OVERLAP) -> list[int]:
    """Indices of boxes to keep after vertebra-specific clean-up of YOLO output.

    1. The model often boxes one vertebra twice with IoU just under the NMS threshold. Boxes whose
       overlap (share of the smaller box) exceeds `dup` are the same vertebra: keep the one whose
       height is closest to the median vertebra height on this image. Model confidence picks the
       right box of such a pair only half the time, typical height about 70 %.
    2. A box covered by both its upper and lower neighbour by more than `mid` straddles two
       vertebrae: drop that middle box.
    Correct neighbouring vertebrae overlap by at most ~0.38 on the training set (median 0.04).
    """
    if len(boxes) == 0:
        return []
    h = boxes[:, 3] - boxes[:, 1]
    med = np.median(h)
    keep: list[int] = []
    for i in np.argsort(np.abs(h / med - 1)):  # most typical height first
        if all(overlap(boxes[i], boxes[j]) <= dup for j in keep):
            keep.append(int(i))
    changed = True
    while changed and len(keep) >= 3:
        changed = False
        order = sorted(keep, key=lambda i: boxes[i, 1] + boxes[i, 3])
        for a, b, c in zip(order, order[1:], order[2:]):
            if overlap(boxes[b], boxes[a]) > mid and overlap(boxes[b], boxes[c]) > mid:
                keep.remove(b)
                changed = True
                break
    return keep


def spine_from_result(res, conf: float, kpt_conf: float, dedup: bool = True) -> dict:
    """Vertebrae (bottom-up, with levels) and the fitted axis angle from one ultralytics result."""
    boxes = res.boxes.xyxy.cpu().numpy()
    scores = res.boxes.conf.cpu().numpy()
    kps = res.keypoints.data.cpu().numpy()  # (n, 2, 3): x, y, conf
    keep = scores >= conf
    boxes, scores, kps = boxes[keep], scores[keep], kps[keep]
    if dedup:
        keep = suppress_duplicates(boxes)
        boxes, scores, kps = boxes[keep], scores[keep], kps[keep]
    order = np.argsort(-(boxes[:, 1] + boxes[:, 3]))  # bottom first
    verts = []
    for rank, i in enumerate(order):
        (tx, ty, tc), (bx, by, bc) = kps[i]
        verts.append({"level": LEVELS_BOTTOM_UP[min(rank, len(LEVELS_BOTTOM_UP) - 1)],
                      "score": float(scores[i]), "box": boxes[i].round(1).tolist(),
                      "top": [float(tx), float(ty), float(tc)], "bottom": [float(bx), float(by), float(bc)],
                      "tilt_deg": math.degrees(math.atan2(bx - tx, by - ty))})
    pts = np.array([p[:2] for v in verts for p in (v["top"], v["bottom"]) if p[2] >= kpt_conf]).reshape(-1, 2)
    return {"vertebrae": verts, "n_vertebrae": len(verts), "axis_deg": fit_axis_deg(pts)}


def infer_vertebrae(image_path):
    image = np.array(Image.open(image_path).convert("L"))
    if image.ndim == 3:
        image = image[..., 0]
    if image.dtype != np.uint8:
        image = (image / max(image.max(), 1) * 255).astype(np.uint8)
    model_input = np.stack([image] * 3, axis=-1)


    model = YOLO("/app/models/art_paste_strong.pt")
    res = model.predict(model_input, imgsz=640, conf=0.05, verbose=False)[0]

    sp = spine_from_result(res, conf=0.3, kpt_conf=0.5)
    ferguson_result = ferguson(sp["vertebrae"])

    logger.debug("Количество позвонков:", sp["n_vertebrae"], "Угол оси:", round(sp["axis_deg"], 2), "°")
    for v in sp["vertebrae"]:
        logger.debug(v["level"], v["box"], "Топ:", v["top"][:2], "Бот:", v["bottom"][:2])
    logger.debug(ferguson_result)

    # Формируем структуру данных для JSON
    response_data = {
        "n_vertebrae": sp["n_vertebrae"],
        "axis_deg": round(sp["axis_deg"], 2),
        "ferguson": ferguson_result,
        "vertebrae": []
    }

    for v in sp["vertebrae"]:
        # v["box"], v["top"], v["bottom"] могут быть массивами,
        # NpEncoder автоматически преобразует их в списки (lists)
        response_data["vertebrae"].append({
            "level": v["level"],
            "box": v["box"],
            "top": v["top"][:2],
            "bottom": v["bottom"][:2]
        })
