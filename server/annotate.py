#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


# ----------------------- вспомогательные -----------------------

def _clip_box(box, w, h):
    x1, y1, x2, y2 = [int(round(float(v))) for v in box]
    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(0, min(x2, w - 1))
    y2 = max(0, min(y2, h - 1))
    return x1, y1, x2, y2


def _put_label(img, text, org, color=(255, 255, 255)):
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale, thickness = 0.5, 1
    x, y = int(org[0]), int(org[1])
    (tw, th), base = cv2.getTextSize(text, font, scale, thickness)
    cv2.rectangle(img, (x, y - th - base), (x + tw, y + base), (0, 0, 0), -1)
    cv2.putText(img, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)


def _draw_filled_box(img, box, color, alpha=0.20, thickness=2):
    h, w = img.shape[:2]
    x1, y1, x2, y2 = _clip_box(box, w, h)
    if x2 <= x1 or y2 <= y1:
        return
    overlay = img.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)


def _draw_point(img, pt, color, radius=4, label=None):
    x, y = int(round(float(pt[0]))), int(round(float(pt[1])))
    cv2.circle(img, (x, y), radius, color, -1)
    cv2.circle(img, (x, y), radius + 1, (255, 255, 255), 1)
    if label:
        _put_label(img, str(label), (x + 5, y - 5), color)


# ----------------------- основная функция -----------------------

def annotate_image(image, result):
    """
    Наносит разметку из dict `result` (один элемент из data["results"])
    на изображение `image` (numpy BGR) и возвращает новое изображение.

    Поддерживает:
      - ok=true: vertebrae_w_rib (box + keypoints), vertebrae (box, top, bottom),
                 ferguson (текст), artifact (боксы)
      - ok=false: текст с ошибкой и регионом

    :param image: np.ndarray (H, W, 3), BGR
    :param result: dict
    :return: np.ndarray — размеченная копия изображения
    """
    if image is None:
        raise ValueError("image is None")
    if not isinstance(result, dict):
        raise TypeError("result must be a dict")

    img = image.copy()

    # --- ветка ошибки ---
    if not result.get("ok", False):
        error = result.get("error", "")
        region = result.get("region", "")
        text = f"FAILED: {error} | region={region}"
        _put_label(img, text, (10, 30), (0, 0, 255))
        return img

    # --- vertebrae_w_rib ---
    vwr = result.get("vertebrae_w_rib")
    if vwr:
        for det in vwr.get("detections", []):
            box = det.get("box")
            if box:
                _draw_filled_box(img, box, (0, 165, 255), alpha=0.18, thickness=2)
            for i, kp in enumerate(det.get("keypoints", [])):
                _draw_point(img, kp, (0, 165, 255), radius=4, label=str(i))

    # --- vertebrae ---
    vertebrae = result.get("vertebrae")
    if vertebrae:
        for vert in vertebrae.get("vertebrae", []):
            box = vert.get("box")
            level = vert.get("level", "")

            if box:
                _draw_filled_box(img, box, (0, 255, 0), alpha=0.16, thickness=2)
                _put_label(img, level, (int(box[0]), int(box[1]) - 5), (0, 255, 0))

            top = vert.get("top")
            bottom = vert.get("bottom")

            if top:
                _draw_point(img, top, (0, 0, 255), radius=4, label="top")
            if bottom:
                _draw_point(img, bottom, (255, 0, 0), radius=4, label="bot")
            if top and bottom:
                cv2.line(
                    img,
                    (int(top[0]), int(top[1])),
                    (int(bottom[0]), int(bottom[1])),
                    (255, 255, 0), 2,
                )

        f = vertebrae.get("ferguson")
        if f:
            text = (
                f"Ferguson: {f.get('angle_deg')} deg | "
                f"lower={f.get('lower')} apex={f.get('apex')} upper={f.get('upper')} "
                f"scoliosis={f.get('scoliosis')}"
            )
            _put_label(img, text, (10, 30), (255, 255, 255))

    # --- artifact ---
    artifact = result.get("artifact")
    if artifact and artifact.get("has_artifact"):
        boxes = artifact.get("artifact_boxes", {})
        if isinstance(boxes, dict):
            for name, bxs in boxes.items():
                if not isinstance(bxs, list):
                    bxs = [bxs]
                for b in bxs:
                    if isinstance(b, dict) and "box" in b:
                        b = b["box"]
                    _draw_filled_box(img, b, (0, 0, 255), alpha=0.25, thickness=2)
                    _put_label(img, str(name), (int(b[0]), int(b[1]) - 5), (0, 0, 255))
        elif isinstance(boxes, list):
            for b in boxes:
                _draw_filled_box(img, b, (0, 0, 255), alpha=0.25, thickness=2)

    return img


# ----------------------- CLI-обвязка -----------------------

def _resolve_image(file_field, images_root):
    p = Path(file_field)
    root = Path(images_root)
    candidates = []
    if p.is_absolute():
        candidates.append(p)
    candidates.append(root / p)
    candidates.append(root / p.name)
    for c in candidates:
        if c.exists():
            return c
    for c in root.rglob(p.name):
        return c
    raise FileNotFoundError(file_field)


def main():
    parser = argparse.ArgumentParser(
        description="Наносит боксы/точки из JSON на изображения."
    )
    parser.add_argument("--json", required=True, help="Путь к JSON-файлу")
    parser.add_argument("--images-root", default=".",
                        help="Корень, где лежат изображения")
    parser.add_argument("--out-dir", default="annotated",
                        help="Папка для сохранения размеченных изображений")
    parser.add_argument("--show", action="store_true",
                        help="Показывать изображения после разметки")
    args = parser.parse_args()

    with open(args.json, "r", encoding="utf-8") as f:
        data = json.load(f)

    results = data.get("results") or data.get("source", {}).get("results") or []
    out_dir = Path(args.out_dir)

    for res in results:
        file_field = res.get("file")
        if not file_field:
            continue

        try:
            img_path = _resolve_image(file_field, args.images_root)
        except FileNotFoundError:
            print(f"[WARN] Не найдено изображение: {file_field}")
            continue

        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is None:
            print(f"[WARN] Не удалось прочитать: {img_path}")
            continue

        # >>> вот здесь вызов отдельной функции <<<
        annotated = annotate_image(img, res)

        rel = Path(file_field)
        if rel.is_absolute():
            rel = Path(rel.name)
        out_path = out_dir / rel
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_path), annotated)
        print(f"[OK] {img_path} -> {out_path}")

        if args.show:
            cv2.imshow("annotated", annotated)
            cv2.waitKey(0)

    if args.show:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()