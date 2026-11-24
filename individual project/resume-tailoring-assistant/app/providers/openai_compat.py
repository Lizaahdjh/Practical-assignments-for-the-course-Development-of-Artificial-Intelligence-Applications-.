import httpx
from app.providers.base import Provider, ProviderError, ProviderRequest, ProviderResult

class OpenAICompatProvider(Provider):
    """Реальний виклик LLM через OpenAI-сумісний endpoint (Gemini, OpenAI)."""

    def __init__(self, api_key: str, model: str, timeout_s: float, base_url: str = "https://generativelanguage.googleapis.com/v1beta/openai/"):
        self.api_key = api_key
        self.model = model
        self.timeout_s = timeout_s
        self.base_url = base_url.rstrip("/")

    async def complete(self, request: ProviderRequest) -> ProviderResult:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        
        payload = {
            "model": self.model,
            "messages": request.messages,
            "temperature": 0.0,
        }

        if request.response_schema:
            payload["response_format"] = {"type": "json_object"}

        async with httpx.AsyncClient(timeout=self.timeout_s) as client:
            try:
                response = await client.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=payload,
                )
                
                if response.status_code == 401:
                    raise ProviderError("auth", "Недійсний API ключ", retryable=False)
                elif response.status_code == 429:
                    raise ProviderError("rate_limit", "Перевищено ліміт запитів", retryable=True)
                elif response.status_code >= 500:
                    raise ProviderError("provider_unavailable", "Помилка сервера провайдера", retryable=True)
                
                response.raise_for_status()
                data = response.json()

                text = data["choices"][0]["message"]["content"]
                usage = data.get("usage", {})

                return ProviderResult(
                    text=text,
                    structured=None,
                    model=data.get("model", self.model),
                    tokens_in=usage.get("prompt_tokens", 0),
                    tokens_out=usage.get("completion_tokens", 0),
                    cost_usd=0.0,
                )

            except httpx.TimeoutException:
                raise ProviderError("timeout", "Час очікування провайдера вичерпано", retryable=True)
            except httpx.RequestError as e:
                raise ProviderError("provider_unavailable", f"Помилка з'єднання: {e}", retryable=True)