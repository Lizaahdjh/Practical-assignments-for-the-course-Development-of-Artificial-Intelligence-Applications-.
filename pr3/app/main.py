"""Веб-рівень застосунку: сторінка зі зверненням і JSON-ендпоінт."""

import logging
from pathlib import Path

from fastapi import FastAPI, HTTPException, status
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from . import llm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Помічник служби підтримки — ПР3")

INDEX_PAGE = Path(__file__).parent / "templates" / "index.html"
CONTEXT_FILE = Path(__file__).parent.parent / "context.md"


class Question(BaseModel):
    """Модель звернення користувача."""

    question: str = Field(
        ...,
        min_length=1,
        max_length=4000,
        description="Текст запитання клієнта",
    )


def load_context() -> str:
    """Прочитати правила організації з файлу context.md."""
    if not CONTEXT_FILE.exists():
        raise RuntimeError(f"Файл контексту {CONTEXT_FILE} не знайдено.")
    return CONTEXT_FILE.read_text(encoding="utf-8")


@app.on_event("startup")
def _warn_if_context_missing() -> None:
    """Попередити на старті, якщо context.md відсутній."""
    if not CONTEXT_FILE.exists():
        logger.warning("Увага: файл контексту %s не знайдено.", CONTEXT_FILE)
    if not llm.API_KEY or not llm.BASE_URL:
        logger.warning("Увага: LLM_API_KEY або LLM_BASE_URL не задані в .env.")


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    """Віддати HTML-сторінку з інтерфейсом."""
    return INDEX_PAGE.read_text(encoding="utf-8")


@app.post("/api/ask")
def api_ask(payload: Question):
    """JSON-ендпоінт обробки запитання користувача."""
    try:
        context = load_context()
        result = llm.ask(payload.question, context)
        return result

    except llm.LLMError as e:
        logger.error(
            "LLMError: %s (status=%d, retryable=%s)",
            e.message, e.status_code, e.is_retryable,
        )
        # Для 5xx не показуємо користувачеві деталі, тільки загальне
        detail = e.message
        if e.status_code >= 500:
            detail = "Сервіс тимчасово не може відповісти. Спробуйте пізніше."
        raise HTTPException(status_code=e.status_code, detail=detail) from e

    except RuntimeError as e:
        logger.exception("Проблема з конфігурацією застосунку: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Застосунок налаштовано некоректно. Зверніться до адміністратора.",
        ) from e

    except Exception as e:
        logger.exception("Непередбачена помилка сервера: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Внутрішня помилка сервера.",
        ) from e