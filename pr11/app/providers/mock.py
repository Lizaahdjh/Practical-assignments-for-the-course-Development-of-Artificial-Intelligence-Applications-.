"""Підставний провайдер: без мережі, зі збоями на вимогу.

Повертає правдоподібний `usage` за кількістю символів у запиті й
відповіді (~3.5 символи/токен для української). Це потрібно, щоб
бюджет, вартість і кеш можна було перевірити без ключа.
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

# Скільки секунд «мовчати» при timeout; шлюз перерве раніше.
_TIMEOUT_SLEEP = float(os.getenv("LLM_TOTAL_TIMEOUT", "45")) + 5
CHARS_PER_TOKEN = 3.5


def _text_for(request: Request) -> str:
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


def _usage_for(request: Request, text: str) -> dict:
    prompt_chars = sum(len(m.get("content", "")) for m in request.messages)
    prompt_tokens = max(1, round(prompt_chars / CHARS_PER_TOKEN))
    completion_tokens = max(1, round(len(text) / CHARS_PER_TOKEN))
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }


def _maybe_fail() -> str | None:
    if not FAILURE_KIND or FAILURE_RATE <= 0:
        return None
    if random.random() < FAILURE_RATE:
        return FAILURE_KIND
    return None


def _raise_kind(kind: str) -> None:
    if kind == "timeout":
        time.sleep(_TIMEOUT_SLEEP)
        raise ProviderError("timeout", "Провайдер не відповів.",
                            retryable=True)
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
    raise ProviderError("unknown", "Невідомий збій.", retryable=False)


class MockProvider:
    name = "mock"
    model = "mock-1"

    def complete(self, request: Request) -> Result:
        started = time.perf_counter()
        fail = _maybe_fail()
        if fail == "timeout":
            time.sleep(_TIMEOUT_SLEEP)
            raise ProviderError("timeout", "Провайдер не відповів.",
                                retryable=True)
        if fail and fail != "mid_stream":
            _raise_kind(fail)
        time.sleep(FIRST_DELAY_MS / 1000)
        text = _text_for(request)
        if fail == "mid_stream":
            partial = text[: len(text) // 2]
            raise ProviderError(
                "unavailable", "Обрив під час відповіді.",
                retryable=True,
                detail=f"partial_text_len={len(partial)}",
            )
        return Result(
            text=text,
            provider=self.name,
            model=self.model,
            usage=_usage_for(request, text),
            finish_reason="stop",
            elapsed=time.perf_counter() - started,
        )

    def stream(self, request: Request) -> Iterator[Chunk]:
        fail = _maybe_fail()

        if fail == "timeout":
            time.sleep(_TIMEOUT_SLEEP)
            raise ProviderError("timeout", "Провайдер не відповів.",
                                retryable=True)
        if fail and fail != "mid_stream":
            _raise_kind(fail)

        text = _text_for(request)
        chunks = list(text)
        time.sleep(FIRST_DELAY_MS / 1000)

        is_cancelled = (request.metadata or {}).get("_cancelled")
        collected: list[str] = []
        mid_at = min(5, max(1, len(chunks) // 3))

        for i, ch in enumerate(chunks):
            if is_cancelled and is_cancelled():
                raise ProviderError("cancelled", "Перервано.", retryable=False)
            if fail == "mid_stream" and i >= mid_at:
                raise ProviderError(
                    "unavailable", "Обрив під час відповіді.",
                    retryable=True,
                    detail=f"partial_text_len={len(''.join(collected))}",
                )
            collected.append(ch)
            yield Chunk(text=ch)
            time.sleep(DELAY_MS / 1000)

        full = "".join(collected)
        yield Chunk(text="", usage=_usage_for(request, full),
                    finish_reason="stop")