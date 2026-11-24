import asyncio

from app.gateway import Gateway
from app.models import AnalyzeRequest, RequestStatus
from app.providers import MockProvider
from app.service import Service
from app.settings import get_settings
from app.storage import Storage


def test_service_direct_without_http(mock_env, tmp_path):
    settings = get_settings()
    storage = Storage(str(tmp_path / "svc.db"))
    gateway = Gateway(MockProvider())
    service = Service(settings, gateway, storage)

    payload = AnalyzeRequest(
        vacancy_text="Python backend developer. Django, PostgreSQL.",
        resume_text="Олена К. Python developer, 3 роки. Django, DRF.",
    )
    result = asyncio.run(service.analyze("req-direct-1", payload))
    assert 0 <= result.match_score <= 100
    assert len(result.interview_questions) == 3

    saved = storage.get_request("req-direct-1")
    assert saved is not None
    assert saved.status == RequestStatus.ok
    assert saved.result is not None
    assert saved.result["match_score"] == result.match_score