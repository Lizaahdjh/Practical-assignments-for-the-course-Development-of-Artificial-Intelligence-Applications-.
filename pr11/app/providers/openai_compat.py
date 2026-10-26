"""Справжній провайдер: OpenAI-сумісний API.

Єдиний модуль, який імпортує SDK `openai`. Зводить винятки SDK до
`ProviderError`. `usage` приходить лише в останньому фрагменті з
`stream_options={"include_usage": True}`; якщо провайдер не приймає
цей параметр — повторюємо виклик без нього.

ВАЖЛИВО (ПР11): `usage` має дійти до шлюзу — інакше вартість стає
оцінкою. Тому віддаємо `Chunk(text="", usage=...)`:
  • коли SDK віддає чанк без `choices` (окремий чанк з usage — Gemini,
    OpenAI);
  • коли `usage` є на чанку з `finish_reason` (деякі сумісні API).
"""

import logging
import time
from typing import Iterator

import openai
from openai import OpenAI

from .base import Chunk, ProviderError, Request, Result

logger = logging.getLogger(__name__)


def _classify_error(e: Exception) -> ProviderError:
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
                    pass
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


def _usage_dict(usage) -> dict | None:
    """SDK-об'єкт usage → dict, зі збереженням cached_tokens."""
    if usage is None:
        return None
    result: dict = {}
    pt = getattr(usage, "prompt_tokens", None)
    ct = getattr(usage, "completion_tokens", None)
    tt = getattr(usage, "total_tokens", None)
    if pt is not None:
        result["prompt_tokens"] = pt
    if ct is not None:
        result["completion_tokens"] = ct
    if tt is not None:
        result["total_tokens"] = tt
    details = getattr(usage, "prompt_tokens_details", None)
    if details is not None:
        cached = getattr(details, "cached_tokens", None)
        if cached is not None:
            result["prompt_tokens_details"] = {"cached_tokens": cached}
    return result or None


class OpenAICompatProvider:
    name = "openai"

    def __init__(self, base_url: str, api_key: str, model: str,
                 timeout: float) -> None:
        if not (base_url and api_key and model):
            raise ProviderError(
                "auth", "Не задано доступ до провайдера.",
                retryable=False,
                detail="відсутні LLM_BASE_URL / LLM_API_KEY / LLM_MODEL",
            )
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
        started = time.perf_counter()
        try:
            resp = self._client.chat.completions.create(**self._kwargs(request))
        except Exception as e:
            raise _classify_error(e) from e
        choice = resp.choices[0]
        return Result(
            text=choice.message.content or "",
            provider=self.name,
            model=resp.model or (request.model or self.model),
            usage=_usage_dict(getattr(resp, "usage", None)),
            finish_reason=choice.finish_reason,
            elapsed=time.perf_counter() - started,
        )

    def stream(self, request: Request) -> Iterator[Chunk]:
        kwargs = self._kwargs(request)
        kwargs["stream"] = True
        kwargs["stream_options"] = {"include_usage": True}

        try:
            stream = self._client.chat.completions.create(**kwargs)
        except openai.BadRequestError:
            # stream_options не підтримується — пробуємо без нього
            kwargs.pop("stream_options", None)
            try:
                stream = self._client.chat.completions.create(**kwargs)
            except Exception as e:
                raise _classify_error(e) from e
        except Exception as e:
            raise _classify_error(e) from e

        try:
            for chunk in stream:
                # Чанк без choices — це чанк з usage (Gemini, OpenAI)
                if not getattr(chunk, "choices", None):
                    u = getattr(chunk, "usage", None)
                    if u is not None:
                        yield Chunk(text="", usage=_usage_dict(u))
                    continue

                choice = chunk.choices[0]
                delta = getattr(choice, "delta", None)
                text = (delta.content or "") if delta else ""
                finish = getattr(choice, "finish_reason", None)

                # Деякі сумісні API віддають usage на чанку з finish_reason
                u = getattr(chunk, "usage", None)
                usage_dict = _usage_dict(u) if u is not None else None

                if text or finish or usage_dict:
                    yield Chunk(text=text, usage=usage_dict,
                                finish_reason=finish)
        except Exception as e:
            raise _classify_error(e) from e
        finally:
            try:
                stream.close()
            except Exception:
                pass