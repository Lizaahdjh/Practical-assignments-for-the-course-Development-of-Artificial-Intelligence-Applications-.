import asyncio
import json
import random
from app.providers.base import Provider, ProviderError, ProviderRequest, ProviderResult


class MockProvider(Provider):
    """Детермінований провайдер: повертає валідний структурований вихід
    за схемою, відтворює збої за налаштуванням."""

    def __init__(self, delay_ms: int = 0, failure_rate: float = 0.0, failure_kind: str = "timeout"):
        self.delay_ms = delay_ms
        self.failure_rate = failure_rate
        self.failure_kind = failure_kind

    async def complete(self, request: ProviderRequest) -> ProviderResult:
        if self.delay_ms:
            await asyncio.sleep(self.delay_ms / 1000)

        if self.failure_rate > 0 and random.random() < self.failure_rate:
            self._raise_failure()

        structured = self._build_structured()
        return ProviderResult(
            text=json.dumps(structured, ensure_ascii=False),
            structured=structured,
            model=request.model,
            tokens_in=0,
            tokens_out=0,
            cost_usd=0.0,
        )

    def _build_structured(self) -> dict:
        # Детермінована відповідь, що відповідає схемі з docs/api.md.
        # Оркестрація (ПР15) має шанс перевірити її на валідність.
        return {
            "match_score": 72,
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