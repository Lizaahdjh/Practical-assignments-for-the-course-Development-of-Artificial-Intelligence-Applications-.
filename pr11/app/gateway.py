"""Шлюз до моделі: єдиний шлях застосунку до провайдера.

ПР11:
  • кеш — перед першою спробою (якщо не fresh); запис — після вдалої;
  • бюджет — check() перед кожною спробою, record() після кожної
    (включно з невдалими: токени вже витрачено);
  • вартість — costs.estimate() після кожної спроби → Result.cost;
  • журнал — llm.attempt з усіма полями.

ВАЖЛИВО: `usage` від провайдера має дійти до `costs.estimate` — тоді
`cost_estimated=False`, і бюджет рахується точно.
"""

import logging
import os
import random
import threading
import time
from typing import Callable

from dotenv import load_dotenv

from . import budget, cache, costs
from .providers.base import Provider, ProviderError, Request, Result
from .providers.mock import MockProvider
from .providers.openai_compat import OpenAICompatProvider

load_dotenv()
log = logging.getLogger("app.gateway")

PROVIDER = os.getenv("LLM_PROVIDER", "openai")
FALLBACK = os.getenv("LLM_FALLBACK", "none")

TIMEOUT = float(os.getenv("LLM_TIMEOUT", "60"))
IDLE_TIMEOUT = float(os.getenv("LLM_IDLE_TIMEOUT", "30"))
TOTAL_TIMEOUT = float(os.getenv("LLM_TOTAL_TIMEOUT", "120"))
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
    if name == "mock":
        return MockProvider()
    if name in ("openai", "", "none"):
        return OpenAICompatProvider(BASE_URL, API_KEY, MODEL or "", TIMEOUT)
    return OpenAICompatProvider(BASE_URL, API_KEY, name, TIMEOUT)


def _backoff(attempt: int) -> float:
    base = min(BACKOFF_MAX, BACKOFF_BASE * (2 ** (attempt - 1)))
    return base + random.uniform(0, base * 0.2)


def _log_attempt(meta: dict, attempt: int, provider: str, model: str | None,
                 status: str, *, kind: str | None = None,
                 duration_ms: float | None = None,
                 queue_ms: float | None = None,
                 first_token_ms: float | None = None,
                 usage: dict | None = None,
                 cost: float | None = None,
                 cost_estimated: bool = False,
                 fallback: bool = False,
                 cached: bool = False,
                 detail: str | None = None) -> None:
    extra = {
        "draft_id": meta.get("draft_id"),
        "request": meta.get("request_id"),
        "client": meta.get("client"),
        "attempt": attempt,
        "provider": provider,
        "model": model,
        "prompt_version": meta.get("prompt_version"),
        "status": status,
        "fallback": fallback,
        "cached": cached,
    }
    if kind:
        extra["kind"] = kind
    if duration_ms is not None:
        extra["duration_ms"] = round(duration_ms, 1)
    if queue_ms is not None:
        extra["queue_ms"] = round(queue_ms, 1)
    if first_token_ms is not None:
        extra["first_token_ms"] = round(first_token_ms, 1)
    if usage:
        extra["prompt_tokens"] = usage.get("prompt_tokens")
        extra["completion_tokens"] = usage.get("completion_tokens")
        details = usage.get("prompt_tokens_details") or {}
        if isinstance(details, dict) and details.get("cached_tokens") is not None:
            extra["cached_tokens"] = details.get("cached_tokens")
    if cost is not None:
        extra["cost"] = cost
        extra["cost_estimated"] = cost_estimated
    if detail:
        extra["detail"] = detail

    level = logging.INFO
    if status == "error":
        if kind in ("cancelled",):
            level = logging.INFO
        elif kind == "budget":
            level = logging.ERROR
        elif kind in ("timeout", "rate_limit", "unavailable"):
            level = logging.WARNING
        else:
            level = logging.ERROR
    if fallback:
        level = logging.WARNING
    log.log(level, "llm.attempt", extra=extra)


def _try_once(provider: Provider, request: Request,
              on_delta: Callable[[str], None],
              is_cancelled: Callable[[], bool],
              deadline: float) -> tuple[Result, float | None]:
    """Одна спроба. Повертає (Result, first_token_at_seconds)."""
    started = time.perf_counter()
    text_parts: list[str] = []
    usage = None
    finish = None
    first_token_at = None
    last_chunk_at = time.perf_counter()

    try:
        stream = provider.stream(request)
    except ProviderError:
        raise
    except Exception as e:
        raise ProviderError("unknown", "Провайдер збоїв.",
                            retryable=False,
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
                finish = chunk.finish_reason
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
    return Result(
        text=text,
        provider=provider.name,
        model=getattr(provider, "model", MODEL or ""),
        usage=usage,
        finish_reason=finish,
        elapsed=time.perf_counter() - started,
    ), first_token_at


def _generate_with(provider_name: str, request: Request,
                   on_delta, on_status, is_cancelled,
                   started_at: float, meta: dict,
                   queue_ms: float, is_fallback: bool) -> Result:
    provider = make_provider(provider_name)
    deadline = started_at + TOTAL_TIMEOUT
    last_error: ProviderError | None = None

    for attempt in range(1, MAX_ATTEMPTS + 1):
        if is_cancelled():
            raise ProviderError("cancelled", "Перервано.", retryable=False)
        if time.perf_counter() > deadline:
            raise ProviderError("timeout", "Вичерпано загальний час.",
                                retryable=False, detail="total")

        try:
            budget.check()
        except budget.BudgetExceeded as e:
            _log_attempt(meta, attempt, provider.name,
                         getattr(provider, "model", None), "error",
                         kind="budget", fallback=is_fallback,
                         detail=f"resets_at={e.resets_at}")
            raise ProviderError("budget", str(e), retryable=False,
                                detail=f"resets_at={e.resets_at}") from e

        on_status({
            "attempt": attempt,
            "provider": provider.name,
            "model": getattr(provider, "model", None),
            "fallback": is_fallback,
            "detail": f"спроба {attempt} з {MAX_ATTEMPTS}",
        })

        attempt_started = time.perf_counter()
        try:
            result, first_token_at = _try_once(
                provider, request, on_delta, is_cancelled, deadline,
            )
        except ProviderError as e:
            duration_ms = (time.perf_counter() - attempt_started) * 1000

            # Оцінка вартості частково витрачених токенів (обрив, timeout)
            partial_len = 0
            if e.detail and "partial_text_len=" in e.detail:
                try:
                    partial_len = int(
                        e.detail.split("partial_text_len=")[1].split()[0])
                except (ValueError, IndexError):
                    partial_len = 0
            partial_cost = None
            partial_est = False
            if partial_len > 0:
                partial_cost, partial_est = costs.estimate(
                    None, provider=provider.name,
                    model=getattr(provider, "model", None),
                    text=" " * partial_len,
                )
                if partial_cost:
                    budget.record(partial_cost)

            _log_attempt(meta, attempt, provider.name,
                         getattr(provider, "model", None), "error",
                         kind=e.kind, duration_ms=duration_ms,
                         queue_ms=queue_ms, fallback=is_fallback,
                         cost=partial_cost,
                         cost_estimated=partial_est,
                         detail=e.detail)
            last_error = e

            if e.kind == "cancelled":
                raise
            if not e.retryable or attempt >= MAX_ATTEMPTS:
                raise
            delay = _backoff(attempt)
            if e.retry_after:
                delay = max(delay, e.retry_after)
            if time.perf_counter() + delay > deadline:
                raise ProviderError("timeout", "Не вистачає часу на повтор.",
                                    retryable=False, detail="budget")
            time.sleep(delay)
            continue

        duration_ms = (time.perf_counter() - attempt_started) * 1000
        cost, est = costs.estimate(
            result.usage,
            provider=result.provider, model=result.model, text=result.text,
        )
        result.cost = cost
        result.cost_estimated = est
        if cost:
            budget.record(cost)
        _log_attempt(meta, attempt, result.provider, result.model, "ok",
                     duration_ms=duration_ms,
                     queue_ms=queue_ms,
                     first_token_ms=(first_token_at * 1000)
                                    if first_token_at else None,
                     usage=result.usage, cost=cost,
                     cost_estimated=est, fallback=is_fallback)
        return result

    raise last_error or ProviderError("unknown", "Усі спроби вичерпано.",
                                      retryable=False)

def generate(request: Request, on_delta: Callable[[str], None],
             on_status: Callable[[dict], None],
             is_cancelled: Callable[[], bool]) -> Result:
    """Кеш → бюджет → провайдер → вартість → кеш."""
    meta = request.metadata or {}
    prompt_version = meta.get("prompt_version") or ""
    fresh = bool(meta.get("fresh", False))

    key = None
    if not fresh:
        key = cache.make_key(request, prompt_version)
        cached = cache.get(key)
        if cached is not None:
            _log_attempt(meta, 0, cached.provider, cached.model, "ok",
                         cached=True, cost=0.0, cost_estimated=False)
            if cached.text:
                on_delta(cached.text)
            return cached
    else:
        cache.bypassed()

    started_at = time.perf_counter()
    acquired = _semaphore.acquire(timeout=max(1.0, TOTAL_TIMEOUT))
    if not acquired:
        raise ProviderError("unavailable", "Забагато одночасних.",
                            retryable=True, detail="semaphore")
    queue_ms = (time.perf_counter() - started_at) * 1000
    try:
        try:
            result = _generate_with(PROVIDER, request, on_delta, on_status,
                                    is_cancelled, started_at, meta,
                                    queue_ms, is_fallback=False)
        except ProviderError as first:
            if first.kind == "cancelled":
                raise
            if FALLBACK in ("none", "", PROVIDER):
                raise
            on_status({"fallback": True, "provider": FALLBACK,
                       "detail": f"перехід на {FALLBACK} після {first.kind}"})
            result = _generate_with(FALLBACK, request, on_delta, on_status,
                                    is_cancelled, started_at, meta,
                                    queue_ms, is_fallback=True)

        if result.finish_reason in (None, "stop"):
            if key is None:
                key = cache.make_key(request, prompt_version)
            cache.put(key, result)
        return result
    finally:
        _semaphore.release()