"""Обмеження частоти звертань до власного API.

Рішення: ковзне вікно на 60 секунд на клієнта. `retry_after` — скільки
секунд до звільнення найстарішої мітки.
"""

import os
import threading
import time

from dotenv import load_dotenv

load_dotenv()

PER_MINUTE = int(os.getenv("RATE_LIMIT_PER_MINUTE", "10"))
WINDOW = 60.0

_lock = threading.Lock()
_calls: dict[str, list[float]] = {}


class RateLimited(Exception):
    def __init__(self, retry_after: float) -> None:
        super().__init__(f"ліміт вичерпано, повторіть через {retry_after:.0f} с")
        self.retry_after = retry_after


def settings() -> dict:
    return {"per_minute": PER_MINUTE}


def acquire(client: str) -> None:
    """Зарахувати звертання або підняти RateLimited."""
    if not client:
        client = "anonymous"
    now = time.monotonic()
    with _lock:
        times = _calls.setdefault(client, [])
        # Викидаємо старі
        cutoff = now - WINDOW
        times[:] = [t for t in times if t > cutoff]
        if len(times) >= PER_MINUTE:
            # Найстаріша мітка звільниться через:
            retry_after = max(0.1, times[0] + WINDOW - now)
            raise RateLimited(retry_after)
        times.append(now)