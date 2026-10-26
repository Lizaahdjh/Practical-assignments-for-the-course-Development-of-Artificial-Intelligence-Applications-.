"""Кеш відповідей: однакове питання — одна оплата.

Ключ охоплює все, що впливає на відповідь: провайдер, модель, версія
prompt, messages, temperature, max_tokens. Не входить: client,
draft_id, request_id, час, fresh.

Сховище — словник у пам'яті + файл `state/cache.json` (переживає
перезапуск, атомарний запис). TTL — `CACHE_TTL_SECONDS`.
"""

import hashlib
import json
import os
import tempfile
import threading
import time
from pathlib import Path

from dotenv import load_dotenv

from .providers.base import Request, Result

load_dotenv()

ENABLED = os.getenv("CACHE_ENABLED", "1") == "1"
TTL_SECONDS = float(os.getenv("CACHE_TTL_SECONDS", "3600"))
MAX_ENTRIES = int(os.getenv("CACHE_MAX_ENTRIES", "500"))
CACHE_FILE = os.getenv("CACHE_FILE", "state/cache.json")

_lock = threading.Lock()

# Лічильники (у пам'яті, не на диску).
_stats = {"hits": 0, "misses": 0, "bypassed": 0}

# Словник: key → {"stored_at": float, "result": dict}
_store: dict[str, dict] = {}
_loaded = False


def settings() -> dict:
    return {
        "enabled": ENABLED,
        "ttl_seconds": TTL_SECONDS,
        "max_entries": MAX_ENTRIES,
        "file": CACHE_FILE or None,
    }


def _path() -> Path | None:
    return Path(CACHE_FILE) if CACHE_FILE else None


def _load() -> None:
    """Завантажити кеш із диска один раз."""
    global _loaded
    if _loaded:
        return
    _loaded = True
    p = _path()
    if not p or not p.exists():
        return
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            for k, v in data.items():
                if isinstance(v, dict) and "stored_at" in v and "result" in v:
                    _store[k] = v
    except (OSError, ValueError):
        # Битий файл — починаємо з нуля
        _store.clear()


def _save() -> None:
    """Атомарний запис усього кешу на диск."""
    p = _path()
    if not p:
        return
    p.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=p.parent, prefix=".cache-", suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(_store, f, ensure_ascii=False)
        os.replace(tmp_name, p)
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass


def _canonical(request: Request, prompt_version: str) -> dict:
    """Канонічне подання запиту для ключа."""
    return {
        "provider": request.metadata.get("provider") or os.getenv("LLM_PROVIDER", "openai"),
        "model": request.model or os.getenv("LLM_MODEL") or "",
        "prompt_version": prompt_version,
        "messages": request.messages,
        "temperature": round(float(request.temperature), 6),
        "max_tokens": int(request.max_tokens),
    }


def make_key(request: Request, prompt_version: str) -> str:
    """sha256 від канонічного JSON."""
    blob = json.dumps(_canonical(request, prompt_version),
                      sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _evict_if_needed() -> None:
    """Прибрати найстаріші записи, якщо перевищено MAX_ENTRIES."""
    if len(_store) <= MAX_ENTRIES:
        return
    # Сортуємо за часом запису, видаляємо найстаріші.
    ordered = sorted(_store.items(), key=lambda kv: kv[1].get("stored_at", 0))
    to_remove = len(_store) - MAX_ENTRIES
    for k, _ in ordered[:to_remove]:
        _store.pop(k, None)


def get(key: str) -> Result | None:
    """Результат за ключем, якщо не прострочений; інакше None."""
    if not ENABLED:
        return None
    _load()
    now = time.time()
    with _lock:
        entry = _store.get(key)
        if not entry:
            _stats["misses"] += 1
            return None
        stored_at = float(entry.get("stored_at", 0))
        if TTL_SECONDS > 0 and now - stored_at > TTL_SECONDS:
            _store.pop(key, None)
            _stats["misses"] += 1
            return None
        _stats["hits"] += 1
    r = entry.get("result") or {}
    return Result(
        text=r.get("text", ""),
        provider=r.get("provider", ""),
        model=r.get("model", ""),
        usage=r.get("usage"),
        finish_reason=r.get("finish_reason"),
        elapsed=r.get("elapsed"),
        cost=r.get("cost"),
        cost_estimated=bool(r.get("cost_estimated", False)),
        cached=True,
    )


def put(key: str, result: Result) -> None:
    """Зберегти результат. Коли кеш вимкнено — нічого не робити."""
    if not ENABLED:
        return
    _load()
    entry = {
        "stored_at": time.time(),
        "result": {
            "text": result.text,
            "provider": result.provider,
            "model": result.model,
            "usage": result.usage,
            "finish_reason": result.finish_reason,
            "elapsed": result.elapsed,
            "cost": result.cost,
            "cost_estimated": result.cost_estimated,
        },
    }
    with _lock:
        _store[key] = entry
        _evict_if_needed()
        _save()


def bypassed() -> None:
    """Відмітити, що кеш обійдено прапорцем fresh."""
    if ENABLED:
        with _lock:
            _stats["bypassed"] += 1


def stats() -> dict:
    if not ENABLED:
        return {"enabled": False}
    _load()
    with _lock:
        return {
            "hits": _stats["hits"],
            "misses": _stats["misses"],
            "bypassed": _stats["bypassed"],
            "entries": len(_store),
        }