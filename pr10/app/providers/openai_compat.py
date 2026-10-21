"""Справжній провайдер: OpenAI-сумісний API.

Єдиний модуль застосунку, який імпортує SDK `openai`. Перетворює наш
`Request` на виклик API, а відповідь і збої API — на `Result`, `Chunk`
і `ProviderError`. Повтори, таймаути, fallback — справа шлюзу.

Рішення:

* `max_retries=0` у клієнті — повтори лише в шлюзі, де видно спроби;
* мапінг винятків SDK → `ProviderError.kind`:
  `APITimeoutError` → timeout; `RateLimitError` → rate_limit
  (з `Retry-After`); `APIConnectionError` → unavailable;
  `InternalServerError` / 5xx → unavailable; `AuthenticationError` → auth;
  `BadRequestError` → bad_request; решта → unknown;
* `usage` приходить лише в останньому фрагменті з
  `stream_options={"include_usage": True}`, інакше — `None`;
* при скасуванні споживач перестає читати — закриваємо потік
  (`stream.close()`);
* `detail` для журналу — тип винятку; `message` для людини — короткий
  текст; ніяких ключів і повного запиту.
"""

import logging
import time
from typing import Iterator

import openai
from openai import OpenAI

from .base import Chunk, ProviderError, Request, Result

logger = logging.getLogger(__name__)


def _classify_error(e: Exception) -> ProviderError:
    """Звести виняток SDK до ProviderError."""
    if isinstance(e, openai.APITimeoutError):
        return ProviderError("timeout", "Провайдер не відповів вчасно.",
                             retryable=True, detail=type(e).__name__)
    if isinstance(e, openai.RateLimitError):
        retry_after = None
        resp = getattr(e, "response", None)
        if resp is not None:
            h = resp.headers.get("retry-after")
            if h:
                try:
                    retry_after = float(h)
                except (TypeError, ValueError):
                    retry_after = None
        return ProviderError("rate_limit", "Перевищено ліміт запитів.",
                             retryable=True, retry_after=retry_after,
                             detail=type(e).__name__)
    if isinstance(e, openai.AuthenticationError):
        return ProviderError("auth", "Помилка доступу до провайдера.",
                             retryable=False, detail=type(e).__name__)
    if isinstance(e, openai.BadRequestError):
        return ProviderError("bad_request", "Провайдер не прийняв запит.",
                             retryable=False, detail=type(e).__name__)
    if isinstance(e, openai.APIConnectionError):
        return ProviderError("unavailable", "Немає з'єднання з провайдером.",
                             retryable=True, detail=type(e).__name__)
    if isinstance(e, openai.APIStatusError):
        code = getattr(e, "status_code", 500)
        kind = "unavailable" if code >= 500 else "bad_request"
        return ProviderError(kind, f"Провайдер повернув статус {code}.",
                             retryable=(code >= 500),
                             detail=f"{type(e).__name__} ({code})")
    return ProviderError("unknown", "Невідомий збій провайдера.",
                         retryable=False, detail=f"{type(e).__name__}: {e}")


class OpenAICompatProvider:
    """Провайдер поверх OpenAI-сумісного API."""

    name = "openai"

    def __init__(self, base_url: str, api_key: str, model: str, timeout: float) -> None:
        if not (base_url and api_key and model):
            raise ProviderError("auth", "Не задано доступ до провайдера.",
                                retryable=False,
                                detail="відсутні LLM_BASE_URL / LLM_API_KEY / LLM_MODEL")
        self.base_url = base_url
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self._client = OpenAI(base_url=base_url, api_key=api_key,
                              timeout=timeout, max_retries=0)

    def _kwargs(self, request: Request) -> dict:
        return {
            "model": request.model or self.model,
            "messages": request.messages,
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
        }

    def complete(self, request: Request) -> Result:
        """Отримати відповідь цілком."""
        started = time.perf_counter()
        try:
            resp = self._client.chat.completions.create(**self._kwargs(request))
        except Exception as e:
            raise _classify_error(e) from e

        elapsed = time.perf_counter() - started
        choice = resp.choices[0]
        usage = getattr(resp, "usage", None)
        return Result(
            text=choice.message.content or "",
            provider=self.name,
            model=resp.model or (request.model or self.model),
            usage={
                "prompt_tokens": getattr(usage, "prompt_tokens", 0) if usage else 0,
                "completion_tokens": getattr(usage, "completion_tokens", 0) if usage else 0,
            } if usage else None,
            finish_reason=choice.finish_reason,
            elapsed=elapsed,
        )

    def stream(self, request: Request) -> Iterator[Chunk]:
        """Віддавати фрагменти. При закритті генератора — закриваємо потік."""
        kwargs = self._kwargs(request)
        kwargs["stream"] = True
        kwargs["stream_options"] = {"include_usage": True}

        try:
            stream = self._client.chat.completions.create(**kwargs)
        except Exception as e:
            raise _classify_error(e) from e

        try:
            for chunk in stream:
                choice = chunk.choices[0] if chunk.choices else None
                if choice is None:
                    # Останній чанк із usage
                    usage = getattr(chunk, "usage", None)
                    if usage:
                        yield Chunk(text="", usage={
                            "prompt_tokens": usage.prompt_tokens,
                            "completion_tokens": usage.completion_tokens,
                        })
                    continue
                delta = choice.delta
                text = delta.content or "" if delta else ""
                finish = choice.finish_reason
                yield Chunk(text=text, finish_reason=finish)
        except Exception as e:
            raise _classify_error(e) from e
        finally:
            try:
                stream.close()
            except Exception:
                pass