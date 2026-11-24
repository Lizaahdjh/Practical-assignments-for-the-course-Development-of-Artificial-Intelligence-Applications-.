import json
from pathlib import Path
from app.providers.base import Provider, ProviderRequest, ProviderResult, ProviderError

class RecordedProvider(Provider):
    """Заглушка для записаного провайдера (ПР15)."""
    async def complete(self, request: ProviderRequest) -> ProviderResult:
        raise ProviderError("provider_unavailable", "Recorded provider is not implemented yet", retryable=False)
    
    def __init__(self, recordings_dir: str = "tests/recordings"):
        self.recordings_dir = Path(recordings_dir)

    async def complete(self, request: ProviderRequest) -> ProviderResult:
        # Для простоти беремо просто перший файл у папці (в реальності — хеш запиту)
        files = list(self.recordings_dir.glob("*.json"))
        if not files:
            raise ProviderError("provider_unavailable", "Немає записаних відповідей", retryable=False)
            
        data = json.loads(files[0].read_text(encoding="utf-8"))
        return ProviderResult(
            text=data["text"],
            structured=data.get("structured"),
            model=data["model"],
            tokens_in=data.get("tokens_in", 0),
            tokens_out=data.get("tokens_out", 0),
            cost_usd=0.0
        )