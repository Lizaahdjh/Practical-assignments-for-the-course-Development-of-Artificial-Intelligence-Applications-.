from typing import Any
from pydantic import BaseModel


class ProviderRequest(BaseModel):
    messages: list[dict[str, str]]
    model: str
    response_schema: dict[str, Any] | None = None
    timeout_s: float = 15.0


class ProviderResult(BaseModel):
    text: str
    structured: dict[str, Any] | None = None
    model: str
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0


class ProviderError(Exception):
    def __init__(self, kind: str, message: str, retryable: bool = False):
        super().__init__(message)
        self.kind = kind
        self.message = message
        self.retryable = retryable


class Provider:
    async def complete(self, request: ProviderRequest) -> ProviderResult:
        raise NotImplementedError