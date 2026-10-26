"""Обмеження частоти звертань до власного API.

Ковзне вікно 60 с на клієнта. Відмови логуються як
`rate_limit.rejected` з клієнтом і `retry_after`.
"""

import logging
import os
import threading
import time

from dotenv import load_dotenv

load_dotenv()

log = logging.getLogger("app.limits")

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
    if not client:
        client = "anonymous"
    now = time.monotonic()
    with _lock:
        times = _calls.setdefault(client, [])
        cutoff = now - WINDOW
        times[:] = [t for t in times if t > cutoff]
        if len(times) >= PER_MINUTE:
            retry_after = max(0.1, times[0] + WINDOW - now)
            log.warning("rate_limit.rejected", extra={
                "client": client, "retry_after": round(retry_after, 1),
                "per_minute": PER_MINUTE,
            })
            raise RateLimited(retry_after)
        times.append(now)