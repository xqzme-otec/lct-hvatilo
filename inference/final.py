from pathlib import Path
import numpy as np

from ultralytics import YOLO
from scripts.dicom_io import load_dicom


img = load_dicom("dicom_examples/CR000000_ПОП.dcm")      # переводит снимок в uint8 так же, как при обучении


def inference_final(image_path: Path):
    model_path = Path(__file__).resolve().parent.parent / "models" / "final.pt"
    model = YOLO(model_path)

    r = model.predict(np.stack([img.pixels] * 3, -1), imgsz=640, conf=0.4, verbose=False)[0]

    boxes = r.boxes.xyxy.cpu().numpy()  # рамки [x1, y1, x2, y2] в пикселях снимка
    conf = r.boxes.conf.cpu().numpy()  # уверенность каждой рамки
    has_artifact = len(conf) > 0  # есть хотя бы одна рамка с уверенностью от 0.4 — артефакт есть