"""Веб-рівень застосунку: сторінка діалогу і JSON-ендпоінт.

Цей файл не має знати ані про провайдера моделі, ані про те, як
складається запит, ані про те, як виглядає схема відповіді, — усе це
лишається в `app/llm.py` і `app/schema.py`. Тут вирішується інше: що
застосунок приймає від сторінки, що віддає їй і з яким HTTP-статусом.

Запуск із папки pr4:

    uvicorn app.main:app --reload

Далі відкрийте http://127.0.0.1:8000
"""

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

app = FastAPI(title="Помічник служби підтримки — ПР4")

INDEX_PAGE = Path(__file__).parent / "templates" / "index.html"
CONTEXT_FILE = Path(__file__).parent.parent / "context.md"


class Turn(BaseModel):
    """Одна репліка розмови: `user` — клієнт, `assistant` — помічник."""

    role: str
    content: str


class ChatRequest(BaseModel):
    """Те, що надсилає сторінка: нове повідомлення й розмову до нього."""

    message: str = Field(..., min_length=1, max_length=4000)
    history: list[Turn] = []


def load_context() -> str:
    """Прочитати правила організації з файлу context.md."""
    if not CONTEXT_FILE.exists():
        raise RuntimeError(f"Файл контексту {CONTEXT_FILE} не знайдено.")
    return CONTEXT_FILE.read_text(encoding="utf-8")


@app.on_event("startup")
def _warn_if_configuration_missing() -> None:
    """Попередити на старті, якщо немає context.md або .env."""
    if not CONTEXT_FILE.exists():
        logger.warning("Увага: файл контексту %s не знайдено.", CONTEXT_FILE)
    if not llm.API_KEY or not llm.BASE_URL:
        logger.warning("Увага: LLM_API_KEY або LLM_BASE_URL не задані в .env.")


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    """Віддати сторінку діалогу."""
    return INDEX_PAGE.read_text(encoding="utf-8")


@app.post("/api/chat")
def api_chat(payload: ChatRequest):
    """Повернути структуровану відповідь помічника у форматі JSON.

    Сторінка очікує обʼєкт із полями `result` (перевірена відповідь моделі;
    текст для клієнта — у `result.reply`), `model`, `elapsed` і `usage`.

    Обробка збоїв перенесена з ПР3 і доповнена новим видом:
    `LLMSchemaError` — модель відповіла, але невалідно.
    """
    try:
        context = load_context()
        history = [turn.model_dump() for turn in payload.history]
        result = llm.ask(payload.message, history, context)
        return result

    except llm.LLMSchemaError as e:
        # Окремий вид збою — логуємо з деталями, клієнту показуємо
        # зрозумілу фразу. Деталі — тільки в журнал.
        logger.error(
            "SchemaError: %s | detail=%s",
            e.message, getattr(e, "detail", ""),
        )
        raise HTTPException(
            status_code=502,
            detail="Помічник тимчасово не може обробити відповідь. "
                   "Спробуйте, будь ласка, ще раз або переформулюйте.",
        ) from e

    except llm.LLMError as e:
        logger.error(
            "LLMError: %s (status=%d, retryable=%s)",
            e.message, e.status_code, e.is_retryable,
        )
        detail = e.message
        if e.status_code >= 500:
            detail = "Сервіс тимчасово не може відповісти. Спробуйте пізніше."
        raise HTTPException(status_code=e.status_code, detail=detail) from e

    except RuntimeError as e:
        logger.exception("Проблема з конфігурацією застосунку: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Застосунок налаштовано некоректно. "
                   "Зверніться до адміністратора.",
        ) from e

    except Exception as e:
        logger.exception("Непередбачена помилка сервера: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Внутрішня помилка сервера.",
        ) from e