import logging
from logging import Logger
from pathlib import Path
from ultralytics import YOLO

logger = Logger(__name__)
logger.setLevel(logging.DEBUG)

def infer_vertebrae_with_rib(image_path: Path):
    path = Path(__file__).resolve().parent.parent / "models" / "vertebrae_with_rib.pt"
    model = YOLO(str(path))

    r = model.predict(image_path, conf=0.3, verbose=False)[0]
    has = len(r.boxes) > 0
    verdict = "accept" if has else "review"

    logger.debug(f"image={image_path}")
    logger.debug(f"verdict={verdict}")

    response_data = {
        "image": str(image_path),
        "verdict": verdict,
        "detections": []
    }

    if has:
        logger.debug(f"n_boxes={len(r.boxes)}")

        boxes_list = r.boxes.xyxy.tolist()
        kpts_list = r.keypoints.xy.tolist() if r.keypoints is not None else [[]] * len(boxes_list)

        for i, (box, kpts) in enumerate(zip(boxes_list, kpts_list)):
            logger.debug(f"  box {i}: {box}")
            logger.debug(f"  kpts: {kpts}")

            # Добавляем каждый объект в список детекций
            response_data["detections"].append({
                "box_index": i,
                "box": box,
                "keypoints": kpts
            })

    return response_data