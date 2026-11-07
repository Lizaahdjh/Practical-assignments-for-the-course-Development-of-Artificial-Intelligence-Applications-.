from app.providers.base import Provider, ProviderError, ProviderRequest, ProviderResult


class OpenAICompatProvider(Provider):
    """Заготовка ПР15: реальний виклик Gemini через OpenAI-сумісний endpoint."""

    def __init__(self, api_key: str, model: str, timeout_s: float):
        self.api_key = api_key
        self.model = model
        self.timeout_s = timeout_s

    async def complete(self, request: ProviderRequest) -> ProviderResult:
        raise ProviderError(
            kind="provider_unavailable",
            message="openai_compat provider is not implemented yet (planned for ПР15)",
            retryable=False,
        )