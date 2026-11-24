from enum import Enum
from pydantic import BaseModel, Field, field_validator


class ErrorKind(str, Enum):
    invalid_input = "invalid_input"
    not_found = "not_found"
    provider_unavailable = "provider_unavailable"
    timeout = "timeout"
    rate_limit = "rate_limit"
    internal = "internal"
    invalid_output = "invalid_output"
    budget = "budget"
    auth = "auth"


class RequestStatus(str, Enum):
    """Статуси запиту в сховищі. Enum замість рядка — щоб не було міграції
    пізніше і щоб неможливо було записати довільний статус."""
    ok = "ok"
    error = "error"
    pending = "pending"


class ErrorBody(BaseModel):
    kind: ErrorKind
    message: str
    retryable: bool = False
    request_id: str


class ErrorEnvelope(BaseModel):
    error: ErrorBody


class AnalyzeRequest(BaseModel):
    vacancy_text: str = Field(min_length=10, max_length=4000)
    resume_text: str = Field(min_length=10, max_length=20000)
    resume_filename: str = Field(default="resume.pdf")

    @field_validator("vacancy_text", "resume_text")
    @classmethod
    def _strip(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("text must not be blank")
        return v


class StyleSuggestion(BaseModel):
    original: str
    suggestion: str


class AnalyzeResult(BaseModel):
    request_id: str
    match_score: int = Field(ge=0, le=100)
    strengths: list[str] = Field(min_length=1, max_length=5)
    gaps: list[str] = Field(default_factory=list, max_length=5)
    style_suggestions: list[StyleSuggestion] = Field(default_factory=list)
    interview_questions: list[str] = Field(min_length=3, max_length=3)


class StoredRequest(BaseModel):
    """Сутність сховища: те, що повертає Storage.get_request()."""
    request_id: str
    created_at: str
    status: RequestStatus
    result: dict | None = None


class HealthResponse(BaseModel):
    status: str
    version: str
    provider: str
    model: str
    mode: str
    storage: str
    extra: dict = Field(default_factory=dict)
    uptime_s: float