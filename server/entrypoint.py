import logging
import sys

from hot_env import wait_for_env_file, load_env

logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
    try:
        # Ожидание инжекта от vault
        wait_for_env_file()
        load_env(force=True)
    except RuntimeError as e:
        print(f"FATAL: {e}", file=sys.stderr)
        sys.exit(1)

    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, log_level="info")