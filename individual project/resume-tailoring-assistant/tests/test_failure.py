def test_provider_failure_returns_controlled_error(monkeypatch, tmp_path):
    monkeypatch.setenv("LLM_PROVIDER", "mock")
    monkeypatch.setenv("MOCK_FAILURE_RATE", "1.0")
    monkeypatch.setenv("MOCK_FAILURE_KIND", "timeout")
    monkeypatch.setenv("DB_PATH", str(tmp_path / "fail.db"))
    monkeypatch.setenv("LOG_FILE", str(tmp_path / "fail.log"))

    from app.settings import get_settings
    get_settings.cache_clear()

    from fastapi.testclient import TestClient
    from app.main import app
    with TestClient(app) as client:
        r = client.post(
            "/api/analyze",
            json={"vacancy_text": "Python developer", "resume_text": "Олена. Python, 2 роки."},
        )
    assert r.status_code == 504
    body = r.json()
    assert body["error"]["kind"] == "timeout"
    assert body["error"]["retryable"] is True
    assert body["error"]["request_id"]

    get_settings.cache_clear()