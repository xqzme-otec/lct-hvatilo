import numpy as np

from ultralytics import YOLO
from PIL import Image


def inference_artifact(image_path):
    image = np.array(Image.open(image_path).convert("L"))
    if image.ndim == 3:
        image = image[..., 0]
    if image.dtype != np.uint8:
        image = (image / max(image.max(), 1) * 255).astype(np.uint8)
    model_input = np.stack([image] * 3, axis=-1)

    model = YOLO("/app/models/art_paste_strong.pt")
    r = model.predict(model_input, imgsz=640, conf=0.4, verbose=False)[0]
    # boxes = r.boxes.xyxy.cpu().numpy()  # рамки [x1, y1, x2, y2] в пикселях снимка
    conf = r.boxes.conf.cpu().numpy()  # уверенность каждой рамки
    return {
        "has_artifact": len(conf) > 0
    }