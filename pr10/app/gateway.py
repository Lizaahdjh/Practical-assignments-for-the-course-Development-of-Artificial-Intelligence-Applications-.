"""Шлюз до моделі: єдиний шлях застосунку до провайдера.

Рішення:

* `make_provider(name)` — `openai` / `mock` / інша назва моделі (та сама
  адреса й ключ, інша модель);
* `generate` — цикл спроб із таймаутами, backoff+jitter, обмеженням
  одночасності (Semaphore), скасуванням, fallback;
* таймаути:
  - `LLM_TIMEOUT` — SDK;
  - `LLM_IDLE_TIMEOUT` — між фрагментами в streaming;
  - `LLM_TOTAL_TIMEOUT` — весь запит із паузами;
* повторюємо лише `retryable=True`; `retry_after` від провайдера
  використовуємо, але якщо пауза перевищує залишок бюджету — припиняємо;
* fallback — лише після всіх спроб основного, з `on_status({"fallback": True})`;
* скасування — `is_cancelled()` між фрагментами й між спробами,
  `ProviderError("cancelled")`;
* повідомляємо через `on_status`: `{"attempt": n, "provider": name,
  "fallback": bool, "detail": "..."}`.
"""

import logging
import os
import random
import threading
import time
from typing import Callable

from dotenv import load_dotenv

from .providers.base import Provider, ProviderError, Request, Result
from .providers.mock import MockProvider
from .providers.openai_compat import OpenAICompatProvider

load_dotenv()
logger = logging.getLogger(__name__)

PROVIDER = os.getenv("LLM_PROVIDER", "openai")
FALLBACK = os.getenv("LLM_FALLBACK", "none")

TIMEOUT = float(os.getenv("LLM_TIMEOUT", "20"))
IDLE_TIMEOUT = float(os.getenv("LLM_IDLE_TIMEOUT", "8"))
TOTAL_TIMEOUT = float(os.getenv("LLM_TOTAL_TIMEOUT", "45"))
MAX_ATTEMPTS = int(os.getenv("LLM_MAX_ATTEMPTS", "3"))
BACKOFF_BASE = float(os.getenv("LLM_BACKOFF_BASE", "0.5"))
BACKOFF_MAX = float(os.getenv("LLM_BACKOFF_MAX", "8"))
MAX_CONCURRENT = int(os.getenv("LLM_MAX_CONCURRENT", "2"))

MODEL = os.getenv("LLM_MODEL")
TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.3"))
MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "600"))

BASE_URL = os.getenv("LLM_BASE_URL", "")
API_KEY = os.getenv("LLM_API_KEY", "")

_semaphore = threading.Semaphore(MAX_CONCURRENT)


def settings() -> dict:
    return {
        "provider": PROVIDER,
        "fallback": FALLBACK,
        "model": MODEL,
        "timeout": TIMEOUT,
        "idle_timeout": IDLE_TIMEOUT,
        "total_timeout": TOTAL_TIMEOUT,
        "max_attempts": MAX_ATTEMPTS,
        "max_concurrent": MAX_CONCURRENT,
    }


def make_provider(name: str) -> Provider:
    """`openai` / `mock` / інша назва моделі з тією самою адресою."""
    if name == "mock":
        return MockProvider()
    if name in ("openai", "", "none"):
        return OpenAICompatProvider(BASE_URL, API_KEY, MODEL or "", TIMEOUT)
    # Інша назва — та сама адреса й ключ, інша модель (для fallback)
    return OpenAICompatProvider(BASE_URL, API_KEY, name, TIMEOUT)


def _backoff(attempt: int) -> float:
    """Експоненційна пауза з jitter, обмежена BACKOFF_MAX."""
    base = min(BACKOFF_MAX, BACKOFF_BASE * (2 ** attempt))
    return base + random.uniform(0, base * 0.2)


def _try_once(provider: Provider, request: Request,
              on_delta: Callable[[str], None],
              on_status: Callable[[dict], None],
              is_cancelled: Callable[[], bool],
              deadline: float) -> Result:
    """Одна спроба з таймаутами спроби, бездіяльності й загальним."""
    started = time.perf_counter()
    text_parts: list[str] = []
    usage = None
    finish_reason = None
    first_token_at = None
    last_chunk_at = time.perf_counter()

    try:
        stream = provider.stream(request)
    except ProviderError:
        raise
    except Exception as e:
        raise ProviderError("unknown", "Провайдер збоїв.", retryable=False,
                            detail=f"{type(e).__name__}: {e}") from e

    try:
        while True:
            if is_cancelled():
                raise ProviderError("cancelled", "Перервано.", retryable=False)

            now = time.perf_counter()
            if now > deadline:
                raise ProviderError("timeout", "Перевищено загальний час.",
                                    retryable=False, detail="total")
            if now - last_chunk_at > IDLE_TIMEOUT:
                raise ProviderError("timeout", "Провайдер мовчить.",
                                    retryable=True, detail="idle")

            try:
                chunk = next(stream)
            except StopIteration:
                break
            except ProviderError:
                raise
            except Exception as e:
                raise ProviderError("unknown", "Помилка потоку.",
                                    retryable=False,
                                    detail=f"{type(e).__name__}: {e}") from e

            last_chunk_at = time.perf_counter()
            if chunk.usage:
                usage = chunk.usage
            if chunk.finish_reason:
                finish_reason = chunk.finish_reason
            if chunk.text:
                if first_token_at is None:
                    first_token_at = time.perf_counter() - started
                text_parts.append(chunk.text)
                on_delta(chunk.text)
    finally:
        try:
            stream.close()
        except Exception:
            pass

    text = "".join(text_parts)
    elapsed = {
        "first_token": round(first_token_at, 3) if first_token_at else None,
        "total": round(time.perf_counter() - started, 3),
    }
    return Result(
        text=text,
        provider=provider.name,
        model=getattr(provider, "model", MODEL or ""),
        usage=usage,
        finish_reason=finish_reason,
        elapsed=elapsed["total"],
    )


def _generate_with(provider_name: str, request: Request,
                   on_delta: Callable[[str], None],
                   on_status: Callable[[dict], None],
                   is_cancelled: Callable[[], bool],
                   started_at: float) -> Result:
    """Цикл спроб для одного провайдера."""
    provider = make_provider(provider_name)
    deadline = started_at + TOTAL_TIMEOUT
    last_error: ProviderError | None = None

    for attempt in range(1, MAX_ATTEMPTS + 1):
        if is_cancelled():
            raise ProviderError("cancelled", "Перервано.", retryable=False)
        if time.perf_counter() > deadline:
            raise ProviderError("timeout", "Вичерпано загальний час.",
                                retryable=False, detail="total")

        on_status({
            "attempt": attempt,
            "provider": provider.name,
            "model": getattr(provider, "model", None),
            "fallback": False,
            "detail": f"спроба {attempt} з {MAX_ATTEMPTS}",
        })

        try:
            return _try_once(provider, request, on_delta, on_status,
                             is_cancelled, deadline)
        except ProviderError as e:
            last_error = e
            logger.warning("Спроба %d: %s (%s) retryable=%s",
                           attempt, e.kind, e.message, e.retryable)
            if e.kind == "cancelled":
                raise
            if not e.retryable:
                raise
            if attempt >= MAX_ATTEMPTS:
                break

            # Пауза перед наступною спробою
            delay = _backoff(attempt - 1)
            if e.retry_after:
                delay = max(delay, e.retry_after)
            if time.perf_counter() + delay > deadline:
                raise ProviderError("timeout", "Не вистачає часу на повтор.",
                                    retryable=False, detail="budget")
            logger.info("Пауза %.2f с перед спробою %d", delay, attempt + 1)
            time.sleep(delay)

    raise last_error or ProviderError("unknown", "Усі спроби вичерпано.",
                                      retryable=False)


def generate(request: Request, on_delta: Callable[[str], None],
             on_status: Callable[[dict], None],
             is_cancelled: Callable[[], bool]) -> Result:
    """Згенерувати з таймаутами, спробами, одночасністю й fallback."""
    started_at = time.perf_counter()

    # Обмеження одночасності
    acquired = _semaphore.acquire(timeout=max(1.0, TOTAL_TIMEOUT))
    if not acquired:
        raise ProviderError("unavailable",
                            "Занадто багато одночасних запитів.",
                            retryable=True, detail="semaphore")

    try:
        try:
            return _generate_with(PROVIDER, request, on_delta, on_status,
                                  is_cancelled, started_at)
        except ProviderError as first_error:
            if first_error.kind == "cancelled":
                raise
            if FALLBACK in ("none", "", PROVIDER):
                raise
            # Fallback
            on_status({"fallback": True, "provider": FALLBACK,
                       "detail": f"перехід на {FALLBACK} після {first_error.kind}"})
            logger.warning("Fallback на %s після %s", FALLBACK, first_error.kind)
            try:
                return _generate_with(FALLBACK, request, on_delta, on_status,
                                      is_cancelled, started_at)
            except ProviderError as fallback_error:
                raise fallback_error
    finally:
        _semaphore.release()