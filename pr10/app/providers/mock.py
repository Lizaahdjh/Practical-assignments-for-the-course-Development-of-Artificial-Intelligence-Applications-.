"""Підставний провайдер: без мережі, зі збоями на вимогу.

Рішення:

* текст — `reply_example` зі зразка, знайденого за `metadata["request_id"]`;
  якщо звернення невідоме — сталий рядок;
* затримки: `MOCK_FIRST_DELAY_MS` перед першим фрагментом, `MOCK_DELAY_MS`
  між іншими;
* збої: `MOCK_FAILURE_RATE` — частка; `MOCK_FAILURE_KIND`:
  - `timeout` — мовчимо довше за таймаут (чесно перевіряє таймаут шлюзу);
  - `rate_limit` — ProviderError з `retry_after=2`;
  - `unavailable` — ProviderError(retryable=True);
  - `auth` — ProviderError(retryable=False);
  - `mid_stream` — кілька фрагментів, потім ProviderError;
* поважає скасування: якщо `request.metadata.get("_cancelled")()` — виходимо.
"""

import os
import random
import time
from typing import Iterator

from dotenv import load_dotenv

from .base import Chunk, ProviderError, Request, Result

load_dotenv()

FIRST_DELAY_MS = int(os.getenv("MOCK_FIRST_DELAY_MS", "600"))
DELAY_MS = int(os.getenv("MOCK_DELAY_MS", "40"))
FAILURE_RATE = float(os.getenv("MOCK_FAILURE_RATE", "0"))
FAILURE_KIND = os.getenv("MOCK_FAILURE_KIND", "unavailable")


def _text_for(request: Request) -> str:
    """Текст за `request_id` зі зразків, інакше — заглушка."""
    rid = (request.metadata or {}).get("request_id")
    try:
        from .. import samples
        sample = samples.get(rid) if rid else None
        if sample and sample.get("reply_example"):
            return sample["reply_example"]
    except Exception:
        pass
    return ("Добрий день! Дякуємо за звернення. "
            "Наш оператор відповість вам найближчим часом.")


def _maybe_fail() -> str | None:
    """Повертає вид збою, якщо цього разу маємо збоїти, інакше None."""
    if not FAILURE_KIND or FAILURE_RATE <= 0:
        return None
    if random.random() < FAILURE_RATE:
        return FAILURE_KIND
    return None


def _raise_kind(kind: str) -> None:
    if kind == "timeout":
        # Мовчимо довше за будь-який розумний таймаут; шлюз перерве
        # очікування, але нам треба прокинутися вчасно, щоб не тримати
        # потік вічно.
        time.sleep(120)
        raise ProviderError("timeout", "Провайдер не відповів.", retryable=True)
    if kind == "rate_limit":
        raise ProviderError("rate_limit", "Перевищено ліміт запитів.",
                            retryable=True, retry_after=2.0)
    if kind == "unavailable":
        raise ProviderError("unavailable", "Провайдер недоступний.",
                            retryable=True)
    if kind == "auth":
        raise ProviderError("auth", "Помилка доступу.", retryable=False)
    if kind == "bad_request":
        raise ProviderError("bad_request", "Провайдер не прийняв запит.",
                            retryable=False)
    # Невідомий вид — трактуємо як unavailable, але не повторюємо
    raise ProviderError("unknown", "Невідомий збій.", retryable=False)


class MockProvider:
    """Провайдер без моделі: канонічні тексти, затримки, збої на вимогу."""

    name = "mock"
    model = "mock-1"

    def complete(self, request: Request) -> Result:
        """Відповідь цілком — з тими самими затримками й збоями."""
        started = time.perf_counter()
        fail = _maybe_fail()
        if fail == "timeout":
            time.sleep(max(0.001, 60))
            raise ProviderError("timeout", "Провайдер не відповів.", retryable=True)
        if fail and fail != "mid_stream":
            _raise_kind(fail)
        time.sleep(FIRST_DELAY_MS / 1000)
        text = _text_for(request)
        if fail == "mid_stream":
            raise ProviderError("unavailable", "Обрив під час відповіді.",
                                retryable=True)
        return Result(text=text, provider=self.name, model=self.model,
                      usage=None, finish_reason="stop",
                      elapsed=time.perf_counter() - started)

    def stream(self, request: Request) -> Iterator[Chunk]:
        """Фрагменти з затримками. Поважає скасування."""
        fail = _maybe_fail()

        if fail == "timeout":
            time.sleep(120)
            raise ProviderError("timeout", "Провайдер не відповів.", retryable=True)
        if fail and fail not in ("mid_stream",):
            _raise_kind(fail)

        text = _text_for(request)
        chunks = list(text)
        # Затримка перед першим фрагментом
        time.sleep(FIRST_DELAY_MS / 1000)

        is_cancelled = (request.metadata or {}).get("_cancelled")
        for i, ch in enumerate(chunks):
            if is_cancelled and is_cancelled():
                raise ProviderError("cancelled", "Перервано.", retryable=False)
            if fail == "mid_stream" and i >= min(5, len(chunks) - 1):
                raise ProviderError("unavailable", "Обрив під час відповіді.",
                                    retryable=True)
            yield Chunk(text=ch)
            time.sleep(DELAY_MS / 1000)

        yield Chunk(text="", finish_reason="stop")