import subprocess
from functools import lru_cache
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


def _git_short_hash() -> str | None:
    """Повертає короткий git hash, якщо git доступний і ми в репозиторії."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=2, check=False,
        )
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return None


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    llm_provider: str = Field(default="mock")
    llm_model: str = Field(default="gemini-1.5-flash")
    llm_api_key: str = Field(default="")
    llm_timeout_s: float = Field(default=15.0)

    mock_delay_ms: int = Field(default=0)
    mock_failure_rate: float = Field(default=0.0)
    mock_failure_kind: str = Field(default="timeout")

    db_path: str = Field(default="data/app.db")
    log_file: str = Field(default="data/app.log")
    app_version: str = Field(default="")

    @field_validator("llm_provider")
    @classmethod
    def _provider_allowed(cls, v: str) -> str:
        allowed = {"mock", "openai_compat", "local"}
        if v not in allowed:
            raise ValueError(f"LLM_PROVIDER must be one of {allowed}, got {v!r}")
        return v

    @field_validator("llm_api_key")
    @classmethod
    def _key_required_for_real_provider(cls, v: str, info) -> str:
        provider = info.data.get("llm_provider")
        if provider and provider != "mock" and not v:
            raise ValueError(f"LLM_API_KEY is required when LLM_PROVIDER={provider!r}")
        return v

    @field_validator("app_version")
    @classmethod
    def _version_fallback(cls, v: str) -> str:
        # APP_VERSION із .env має приоритет; інакше — git hash; інакше — "dev".
        if v:
            return v
        return _git_short_hash() or "dev"


@lru_cache
def get_settings() -> Settings:
    return Settings()