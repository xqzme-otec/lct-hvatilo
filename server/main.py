import logging
import os
from contextlib import asynccontextmanager
from typing import TypeVar, overload

from fastapi import FastAPI

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
    value = os.environ.get(key)
    if value is None:
        return default
    return value

def env(key: str) -> str:
    reload_if_changed()
    value = os.environ.get(key)
    if not value:
        raise RuntimeError(f"Missing required env var: {key}")
    return value


@asynccontextmanager
async def lifespan(app: FastAPI):
    import asyncio

    async def watch_and_reset():
        while True:
            try:
                if reload_if_changed():
                    ...
            except Exception:
                logger.exception("watch_and_reset failed")
            await asyncio.sleep(5)

    watcher = asyncio.create_task(watch_and_reset())
    yield
    watcher.cancel()

app = FastAPI(lifespan=lifespan)


@app.get("/")
async def root():
    return {"message": "Hello DEXA"}


@app.post("/submit")
async def root():
    return {"message": "Submit DEXA"}


