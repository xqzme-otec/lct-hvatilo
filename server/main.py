import logging
import os
import tempfile
import zipfile
import asyncio
import contextlib
from contextlib import asynccontextmanager
from pathlib import Path
from typing import TypeVar, overload
import httpx

from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.concurrency import run_in_threadpool

from hot_env import reload_if_changed, load_env
from dicom_io import find_dicom_files

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if not load_env(force=False):
    logger.warning("No .env file found at startup, will rely on os.environ")

T = TypeVar("T")


@overload
def env_optional(key: str) -> str | None: ...


@overload
def env_optional(key: str, default: T) -> T: ...


def env_optional(key: str, default: T | None = None) -> T | None:
    reload_if_changed()
    value = os.environ.get(key)
    return default if value is None else value


def env(key: str) -> str:
    reload_if_changed()
    value = os.environ.get(key)
    if not value:
        raise RuntimeError(f"Missing required env var: {key}")
    return value


@asynccontextmanager
async def lifespan(app: FastAPI):
    async def watch_and_reset():
        while True:
            try:
                if reload_if_changed():
                    logger.info("Environment reloaded")
            except Exception:
                logger.exception("watch_and_reset failed")
            await asyncio.sleep(5)

    watcher = asyncio.create_task(watch_and_reset())
    try:
        yield
    finally:
        watcher.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await watcher


app = FastAPI(lifespan=lifespan)

templates = Jinja2Templates(directory="server/templates")

ALLOWED_EXTENSIONS = [".zip"]
MAX_SIZE = 50 * 1024 * 1024
MAX_FILES = 10

INFERENCE_URL = os.environ.get("INFERENCE_URL", "http://inference:8001/infer")

@app.get("/", response_class=HTMLResponse)
async def upload_form(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="upload.html",
        context={"allowed": ALLOWED_EXTENSIONS},
    )


@app.get("/test")
async def root():
    return {"message": "Hello DEXA (Web Server)"}


def safe_extract_zip(zip_path: Path, target: Path) -> None:
    """Безопасная распаковка zip с защитой от path traversal (zip-slip)."""
    target = target.resolve()
    with zipfile.ZipFile(zip_path) as zf:
        for member in zf.namelist():
            member_path = (target / member).resolve()
            if not member_path.is_relative_to(target):
                raise HTTPException(400, f"Unsafe path in archive: {member}")
        zf.extractall(target)


def _create_dicom_zip(dicom_files: list[Path], base_dir: Path, output_zip: Path) -> None:
    """Упаковывает только найденные DICOM-файлы в новый ZIP для отправки в инференс."""
    with zipfile.ZipFile(output_zip, 'w', zipfile.ZIP_DEFLATED) as zf:
        for file_path in dicom_files:
            # Сохраняем относительный путь внутри архива
            arcname = file_path.relative_to(base_dir)
            zf.write(file_path, arcname)


@app.post("/upload")
async def upload_archive(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(400, "No filename provided")

    filename = file.filename.lower()
    if not any(filename.endswith(ext) for ext in ALLOWED_EXTENSIONS):
        raise HTTPException(400, f"Unsupported archive type: {file.filename}")

    contents = await file.read()
    if len(contents) > MAX_SIZE:
        raise HTTPException(413, "File too large")

    with tempfile.TemporaryDirectory(prefix="dicom_upload_") as tmp:
        tmp_path = Path(tmp)
        zip_path = tmp_path / "archive.zip"
        zip_path.write_bytes(contents)

        extract_dir = tmp_path / "extracted"
        extract_dir.mkdir()

        try:
            safe_extract_zip(zip_path, extract_dir)
        except zipfile.BadZipFile:
            raise HTTPException(400, "Invalid zip archive")

        try:
            dicom_files = await run_in_threadpool(find_dicom_files, extract_dir)
        except Exception:
            logger.exception("find_dicom_files failed")
            raise HTTPException(500, "Failed to scan archive for DICOM files")

        if not dicom_files:
            raise HTTPException(422, "No DICOM files found in archive")
        if len(dicom_files) > MAX_FILES:
            raise HTTPException(413, f"Too many DICOM files: {len(dicom_files)}")

        dicom_zip_path = tmp_path / "dicom_files.zip"
        await run_in_threadpool(_create_dicom_zip, dicom_files, extract_dir, dicom_zip_path)

        try:
            async with httpx.AsyncClient(timeout=120.0) as client:  # Таймаут увеличен для инференса
                with open(dicom_zip_path, "rb") as f:
                    files = {"file": ("dicom_files.zip", f, "application/zip")}
                    response = await client.post(INFERENCE_URL, files=files)

                response.raise_for_status()
                inference_results = response.json()

        except httpx.RequestError as exc:
            logger.error(f"Ошибка соединения с сервисом инференса: {exc}")
            raise HTTPException(503, "Сервис инференса недоступен")
        except httpx.HTTPStatusError as exc:
            logger.error(f"Сервис инференса вернул ошибку: {exc.response.text}")
            raise HTTPException(exc.response.status_code, f"Ошибка инференса: {exc.response.text}")

        return {
            "filename": file.filename,
            "dicom_count": len(dicom_files),
            "results": inference_results.get("results", []),
        }