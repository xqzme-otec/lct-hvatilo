import logging
import tempfile
import zipfile
from pathlib import Path
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.concurrency import run_in_threadpool

from inference.vertebrae_with_rib import infer_vertebrae_with_rib

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
        try:
            pred = infer_vertebrae_with_rib(p)
            results.append({"file": str(p), "ok": True, "result": pred})
        except Exception as e:
            logger.exception("Inference failed for %s", p)
            results.append({"file": str(p), "ok": False, "error": str(e)})
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