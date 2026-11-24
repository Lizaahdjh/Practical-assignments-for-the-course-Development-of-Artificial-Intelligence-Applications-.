from app.gateway import Gateway
from app.models import AnalyzeRequest, AnalyzeResult, RequestStatus, StyleSuggestion
from app.orchestration import Orchestrator
from app.providers import ProviderError
from app.settings import Settings
from app.storage import Storage
from app.telemetry import log


class Service:
    def __init__(self, settings: Settings, gateway: Gateway, storage: Storage):
        self.settings = settings
        self.storage = storage
        self.orchestrator = Orchestrator(gateway, settings.llm_model)

    async def analyze(self, request_id: str, payload: AnalyzeRequest) -> AnalyzeResult:
        log("service", request_id, event="start", provider=self.settings.llm_provider)

        try:
            result = await self.orchestrator.run(payload.vacancy_text, payload.resume_text)
        except ProviderError:
            log("service", request_id, event="provider_error")
            raise

        data = result.structured or {}
        analyze_result = AnalyzeResult(
            request_id=request_id,
            match_score=int(data.get("match_score", 0)),
            strengths=list(data.get("strengths", [])) or ["N/A"],
            gaps=list(data.get("gaps", [])),
            style_suggestions=[StyleSuggestion(**s) for s in data.get("style_suggestions", [])],
            interview_questions=list(data.get("interview_questions", []))[:3],
        )
        self.storage.save_request(request_id, status=RequestStatus.ok, result=analyze_result.model_dump())
        log("service", request_id, event="done", score=analyze_result.match_score)
        return analyze_result