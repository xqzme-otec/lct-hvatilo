import asyncio
import hashlib
import logging
import os
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ENV_PATH = Path(os.environ.get("VAULT_ENV_PATH", "/vault-secrets/.env"))
MAX_WAIT_SECONDS = 30
WATCH_INTERVAL_SECONDS = 5

def _file_hash() -> str | None:
    if not ENV_PATH.exists():
        return None
    try:
        return hashlib.sha256(ENV_PATH.read_bytes()).hexdigest()
    except OSError:
        return None


def _parse_env_file(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            result[key.strip()] = value.strip().strip('"').strip("'")
    return result


def wait_for_env_file(timeout: int = MAX_WAIT_SECONDS) -> None:
    """Блокирующее ожидание появления .env (для entrypoint до старта uvicorn)."""
    start = time.time()
    while not ENV_PATH.exists():
        if time.time() - start > timeout:
            raise RuntimeError(
                f"Vault Agent did not render {ENV_PATH} within {timeout}s"
            )
        logger.info("Waiting for %s ...", ENV_PATH)
        time.sleep(0.5)
    time.sleep(1.5)


def load_env(force: bool = False) -> bool:
    if not ENV_PATH.exists():
        return False

    try:
        values = _parse_env_file(ENV_PATH)
    except OSError as e:
        logger.warning("Failed to read %s: %s", ENV_PATH, e)
        return False

    for key, value in values.items():
        if force or key not in os.environ:
            os.environ[key] = value

    global _last_hash
    _last_hash = _file_hash()
    return True


def reload_if_changed() -> bool:
    current = _file_hash()
    if current is None or current == _last_hash:
        return False
    logger.info("Detected change in %s, reloading secrets", ENV_PATH)
    return load_env(force=True)


async def watch_env_file(interval: int = WATCH_INTERVAL_SECONDS) -> None:
    while True:
        try:
            reload_if_changed()
        except Exception:
            logger.exception("env watcher failed")
        await asyncio.sleep(interval)
