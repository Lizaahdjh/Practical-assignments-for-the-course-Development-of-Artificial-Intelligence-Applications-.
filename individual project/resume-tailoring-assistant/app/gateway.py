from app.providers import MockProvider, OpenAICompatProvider, Provider, ProviderRequest, ProviderResult
from app.settings import Settings
from app.budget import BudgetTracker, BudgetExceeded
from app.providers.base import ProviderError
# Якщо ви ще не створили recorded.py, закоментуйте імпорт нижче
from app.providers.recorded import RecordedProvider

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
            base_url=settings.llm_base_url,
        )
    
    if settings.llm_provider == "recorded":
        return RecordedProvider()
    
    # local — заготовка; повертаємо mock, щоб каркас працював
    return MockProvider()


class Gateway:
    def __init__(self, provider: Provider, budget: BudgetTracker | None = None):
        self.provider = provider
        self.budget = budget

    async def complete(self, request: ProviderRequest) -> ProviderResult:
        if self.budget:
            try:
                # Оцінюємо вартість як 0 (Gemini free tier)
                self.budget.check_and_record(estimated_cost_usd=0.0)
            except BudgetExceeded as e:
                raise ProviderError("budget", str(e), retryable=False)
        
        return await self.provider.complete(request)