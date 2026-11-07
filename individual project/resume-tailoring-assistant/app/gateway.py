from app.providers import MockProvider, OpenAICompatProvider, Provider, ProviderRequest, ProviderResult
from app.settings import Settings


def build_provider(settings: Settings) -> Provider:
    if settings.llm_provider == "mock":
        return MockProvider(
            delay_ms=settings.mock_delay_ms,
            failure_rate=settings.mock_failure_rate,
            failure_kind=settings.mock_failure_kind,
        )
    if settings.llm_provider == "openai_compat":
        return OpenAICompatProvider(
            api_key=settings.llm_api_key,
            model=settings.llm_model,
            timeout_s=settings.llm_timeout_s,
        )
    # local — заготовка; повертаємо mock, щоб каркас працював,
    # але фіксуємо це в журналі.
    return MockProvider()


class Gateway:
    def __init__(self, provider: Provider):
        self.provider = provider

    async def complete(self, request: ProviderRequest) -> ProviderResult:
        return await self.provider.complete(request)