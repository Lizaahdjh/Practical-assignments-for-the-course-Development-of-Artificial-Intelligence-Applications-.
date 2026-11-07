from app.providers import ProviderRequest, ProviderResult
from app.gateway import Gateway

RESPONSE_SCHEMA = {
    "type": "object",
    "required": ["match_score", "strengths", "gaps", "style_suggestions", "interview_questions"],
    "properties": {
        "match_score": {"type": "integer", "minimum": 0, "maximum": 100},
        "strengths": {"type": "array", "items": {"type": "string"}, "minItems": 1, "maxItems": 5},
        "gaps": {"type": "array", "items": {"type": "string"}, "minItems": 0, "maxItems": 5},
        "style_suggestions": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["original", "suggestion"],
                "properties": {"original": {"type": "string"}, "suggestion": {"type": "string"}},
            },
        },
        "interview_questions": {"type": "array", "items": {"type": "string"}, "minItems": 3, "maxItems": 3},
    },
}


class Orchestrator:
    """Заготовка ПР15: тут буде збирання контексту (RAG), шаблон інструкції,
    пост-перевірка на галюцинації та маскування PII. У ПР14 — лише прохід
    до провайдера зі схемою структурованого виходу."""

    def __init__(self, gateway: Gateway, model: str):
        self.gateway = gateway
        self.model = model

    async def run(self, vacancy_text: str, resume_text: str) -> ProviderResult:
        request = ProviderRequest(
            messages=[
                {"role": "system", "content": "You are a resume tailoring assistant. Reply with JSON only."},
                {"role": "user", "content": f"VACANCY:\n{vacancy_text}\n\nRESUME:\n{resume_text}"},
            ],
            model=self.model,
            response_schema=RESPONSE_SCHEMA,
        )
        return await self.gateway.complete(request)