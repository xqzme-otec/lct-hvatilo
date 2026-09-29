import logging
import tempfile
import time
import zipfile
import numpy as np
from PIL import Image
from pathlib import Path
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.concurrency import run_in_threadpool

from inference.artifact import inference_artifact
from inference.vertebrae import infer_vertebrae
from inference.vertebrae_with_rib import infer_vertebrae_with_rib
from scripts.region import SPINE, detect_region

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Inference Service")

def safe_extract_zip(zip_path: Path, target: Path) -> None:
    target = target.resolve()
    with zipfile.ZipFile(zip_path) as zf:
        for member in zf.namelist():
            member_path = (target / member).resolve()
            if not member_path.is_relative_to(target):
                raise HTTPException(400, f"Unsafe path in archive: {member}")
        zf.extractall(target)

def _run_inference_sync(file_paths: list[Path]) -> list[dict]:
    """Синхронный запуск инференса по всем файлам."""
    results = []
    for p in file_paths:
        file_started = time.perf_counter()
        
        region_started = time.perf_counter()
        anatomical_region = detect_region(np.array(Image.open(p)))
        region_detect_ms = (time.perf_counter() - region_started) * 1000
        if anatomical_region == SPINE:
            timings = {
                "region_detect_ms": round(region_detect_ms, 2),
            }

            model_results = {}

            models = (
                ("vertebrae_w_rib", infer_vertebrae_with_rib),
                ("vertebrae", infer_vertebrae),
                ("artifact", inference_artifact),
            )

            for model_name, model_func in models:
                model_started = time.perf_counter()
                try:
                    model_results[model_name] = model_func(p)
                except Exception:
                    logger.exception("Inference failed for %s (%s)", p, model_name)
                    model_results[model_name] = None
                finally:
                    timings[f"{model_name}_ms"] = round(
                        (time.perf_counter() - model_started) * 1000, 2
                    )

            results.append({
                "file": str(p),
                "ok": True,
                "region": anatomical_region,
                "vertebrae_w_rib": model_results.get("vertebrae_w_rib"),
                "vertebrae": model_results.get("vertebrae"),
                "artifact": model_results.get("artifact"),
                "timings": timings,
            })
        else:
            results.append({
                "file": str(p),
                "ok": False,
                "error": "Unsupported body region",
                "region": anatomical_region,
                "timings": {
                    "region_detect_ms": round(region_detect_ms, 2),
                    "total_ms": round((time.perf_counter() - file_started) * 1000, 2),
                },
            })
    return results

@app.post("/infer")
async def infer_archive(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(400, "No filename provided")

    contents = await file.read()
    if not contents:
        raise HTTPException(400, "Empty file")

    with tempfile.TemporaryDirectory(prefix="inference_") as tmp:
        tmp_path = Path(tmp)
        zip_path = tmp_path / "dicom_images_files.zip"
        zip_path.write_bytes(contents)

        extract_dir = tmp_path / "extracted"
        extract_dir.mkdir()

        try:
            safe_extract_zip(zip_path, extract_dir)
        except zipfile.BadZipFile:
            raise HTTPException(400, "Invalid zip archive")

        file_paths = [p for p in extract_dir.rglob('*') if p.is_file()]

        if not file_paths:
            raise HTTPException(422, "No files found in archive")

        # Запускаем инференс в threadpool
        results = await run_in_threadpool(_run_inference_sync, file_paths)

        # Преобразуем абсолютные пути в относительные
        formatted_results = []
        for r in results:
            formatted_results.append({
                **r,
                "file": str(Path(r["file"]).relative_to(extract_dir)),
            })

        return {"results": formatted_results}