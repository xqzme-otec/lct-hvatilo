import logging
import os
import zipfile
import io
import asyncio
import contextlib

from contextlib import asynccontextmanager
from typing import TypeVar, overload
from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from hot_env import reload_if_changed, load_env

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


@app.get("/", response_class=HTMLResponse)
async def upload_form(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="upload.html",
        context={"allowed": ALLOWED_EXTENSIONS},
    )


@app.get("/test")
async def root():
    return {"message": "Hello DEXA"}


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

    try:
        with zipfile.ZipFile(io.BytesIO(contents)) as zf:
            names = zf.namelist()
    except zipfile.BadZipFile:
        raise HTTPException(400, "Invalid zip archive")

    return {"filename": file.filename, "size": len(contents), "entries": names}


@app.post("/upload-multiple")
async def upload_multiple(files: list[UploadFile] = File(...)):
    return [{"filename": f.filename, "content_type": f.content_type} for f in files]