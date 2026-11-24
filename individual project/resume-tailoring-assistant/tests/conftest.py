import os
import tempfile

import pytest


@pytest.fixture
def mock_env(monkeypatch, tmp_path):
    monkeypatch.setenv("LLM_PROVIDER", "mock")
    monkeypatch.setenv("MOCK_DELAY_MS", "0")
    monkeypatch.setenv("MOCK_FAILURE_RATE", "0.0")
    monkeypatch.setenv("DB_PATH", str(tmp_path / "app.db"))
    monkeypatch.setenv("LOG_FILE", str(tmp_path / "app.log"))
    
    # Скидаємо кеш налаштувань
    from app.settings import get_settings
    get_settings.cache_clear()
    
    # Скидаємо конфігурацію логера — критично для тестів журналу
    from app.telemetry import reset as reset_telemetry
    reset_telemetry()
    
    yield
    
    get_settings.cache_clear()
    reset_telemetry()


@pytest.fixture
def client(mock_env):
    from fastapi.testclient import TestClient
    from app.main import app
    with TestClient(app) as c:
        yield c