import json
import logging
from pathlib import Path
from datetime import datetime, timezone

_LOG = logging.getLogger("resume_tailoring")
_configured = False


def configure(log_file: str) -> None:
    global _configured
    if _configured:
        return
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(log_file, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(message)s"))
    _LOG.addHandler(handler)
    _LOG.setLevel(logging.INFO)
    _configured = True


def reset() -> None:
    """Скидає конфігурацію логера. Використовується в тестах, щоб
    кожен тест писав у свій tmp_path."""
    global _configured
    for handler in list(_LOG.handlers):
        handler.close()
        _LOG.removeHandler(handler)
    _configured = False


def log(layer: str, request_id: str, **kwargs) -> None:
    record = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "layer": layer,
        "request_id": request_id,
        **kwargs,
    }
    _LOG.info(json.dumps(record, ensure_ascii=False))