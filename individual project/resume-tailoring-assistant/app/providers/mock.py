import asyncio
import json
import random
from app.providers.base import Provider, ProviderError, ProviderRequest, ProviderResult


class MockProvider(Provider):
    """Детермінований провайдер: повертає валідний структурований вихід
    за схемою, відтворює збої за налаштуванням.

    Результат залежить від входу (детерміновано), щоб тест міг довести,
    що запит пройшов крізь оркестрацію, а не повернувся напряму.
    """

    def __init__(self, delay_ms: int = 0, failure_rate: float = 0.0, failure_kind: str = "timeout"):
        self.delay_ms = delay_ms
        self.failure_rate = failure_rate
        self.failure_kind = failure_kind

    async def complete(self, request: ProviderRequest) -> ProviderResult:
        if self.delay_ms:
            await asyncio.sleep(self.delay_ms / 1000)

        if self.failure_rate > 0 and random.random() < self.failure_rate:
            self._raise_failure()

        # Беремо текст останнього повідомлення користувача як вхід,
        # щоб результат залежав від реального запиту.
        user_text = ""
        for msg in reversed(request.messages):
            if msg.get("role") == "user":
                user_text = msg.get("content", "")
                break

        structured = self._build_structured(user_text)
        return ProviderResult(
            text=json.dumps(structured, ensure_ascii=False),
            structured=structured,
            model=request.model,
            tokens_in=0,
            tokens_out=0,
            cost_usd=0.0,
        )

    def _build_structured(self, user_text: str) -> dict:
        # Детермінований, але залежний від входу бал:
        # однаковий вхід -> однаковий вихід; різний вхід -> різний вихід.
        # Використовуємо суму ord, щоб уникнути залежності від PYTHONHASHSEED.
        base = 30 + (sum(ord(c) for c in user_text) % 41)  # 30..70
        if "django" in user_text.lower():
            base = min(base + 30, 95)
        elif "python" in user_text.lower():
            base = min(base + 10, 90)

        return {
            "match_score": base,
            "strengths": ["Python", "Django", "REST API"],
            "gaps": ["PostgreSQL"],
            "style_suggestions": [
                {"original": "робив різні речі", "suggestion": "розробляв REST API на Flask, 2 роки"}
            ],
            "interview_questions": [
                "Як ви проєктували REST API?",
                "Який досвід з PostgreSQL?",
                "Як тестуєте Django-проєкти?",
            ],
        }

    def _raise_failure(self) -> None:
        kind = self.failure_kind
        if kind == "timeout":
            raise ProviderError("timeout", "mock: provider timed out", retryable=True)
        if kind == "rate_limit":
            raise ProviderError("rate_limit", "mock: 429 too many requests", retryable=True)
        if kind == "unavailable":
            raise ProviderError("provider_unavailable", "mock: service unavailable", retryable=True)
        if kind == "invalid_json":
            raise ProviderError("provider_unavailable", "mock: invalid JSON in response", retryable=False)
        if kind == "mid_stream":
            raise ProviderError("provider_unavailable", "mock: stream interrupted", retryable=False)
        raise ProviderError("internal", f"mock: unknown failure kind {kind!r}")