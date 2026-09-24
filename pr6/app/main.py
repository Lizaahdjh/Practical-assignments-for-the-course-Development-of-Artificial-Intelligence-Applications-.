"""Веб-рівень застосунку: сторінка помічника і JSON-ендпоінти.

Цей файл не знає ані як шукаються фрагменти, ані як збирається контекст,
ані якою моделлю й за якою інструкцією отримано відповідь — усе це
лишається в `app/retrieval.py`, `app/llm.py`, `app/schema.py` і
поєднується в `app/rag.py`. Тут вирішується інше: що застосунок приймає
від сторінки, що віддає їй і з яким HTTP-статусом.

Індекс будується заздалегідь командою `python ingest.py` (з папки pr6),
а тут лише читається при старті — як у ПР5.

Запуск із папки pr6:

    uvicorn app.main:app --reload

Далі відкрийте http://127.0.0.1:8000
"""

import logging
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from . import index, keyword, llm, rag

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Помічник за базою знань — ПР6")

INDEX_PAGE = Path(__file__).parent / "templates" / "index.html"


class AskRequest(BaseModel):
    """Те, що надсилає сторінка.

    `filters` — умови на метадані, які обрав користувач, наприклад
    `{"product": "Вега S"}`; порожній словник — без фільтрів. Те, що
    користувач обирати не має (аудиторія документа), сюди не входить —
    це рішення коду, а не сторінки.
    """

    question: str
    filters: dict = {}


def hit_to_dict(hit: index.Hit) -> dict:
    return {
        "score": hit.score,
        "text": hit.chunk.text,
        "source": hit.chunk.source,
        "metadata": hit.chunk.metadata,
    }


def answer_to_dict(result: rag.Answer) -> dict:
    """Перетворити результат конвеєра на те, що піде на сторінку."""
    return {
        "answer": result.text,
        "found": result.found,
        "sources": [
            {"ref": s.ref, "score": s.score, "source": s.chunk.source,
             "metadata": s.chunk.metadata, "text": s.chunk.text}
            for s in result.sources
        ],
        "retrieved": [hit_to_dict(h) for h in result.retrieved],
        "model": result.model,
        "elapsed": result.elapsed,
        "usage": result.usage,
    }


@app.on_event("startup")
def load_indexes() -> None:
    """Прочитати збудований індекс і зібрати індекс за словами з тих
    самих фрагментів — як у ПР5.

    Без індексу застосунок усе одно стартує: сторінка має відкритися й
    пояснити, що робити.
    """
    app.state.index = None
    app.state.keyword_index = None
    try:
        app.state.index = index.load()
    except Exception as exc:  # noqa: BLE001 — старт не має падати без індексу
        print(f"Індекс не завантажено: {type(exc).__name__}: {exc}")
        return
    app.state.keyword_index = keyword.build(app.state.index.chunks)


@app.get("/", response_class=HTMLResponse)
def page() -> str:
    """Віддати сторінку помічника."""
    return INDEX_PAGE.read_text(encoding="utf-8")


@app.get("/api/status")
def api_status() -> dict:
    """Стан індексу: чи збудовано, скільки фрагментів, якою моделлю."""
    idx = app.state.index
    if idx is None:
        return {"ready": False, "hint": "індекс не збудовано — виконайте python ingest.py"}
    sources = {chunk.source for chunk in idx.chunks}
    return {
        "ready": True,
        "chunks": len(idx),
        "documents": len(sources),
        "model": idx.model_name,
    }


@app.post("/api/ask")
def api_ask(payload: AskRequest) -> dict:
    """Відповісти на питання й повернути відповідь із джерелами.

    Сторінка очікує обʼєкт із полями `answer`, `found`, `sources`,
    `retrieved`, `model`, `elapsed`, `usage` — див. `answer_to_dict`.

    Обробка збоїв:

    * порожнє питання — 400;
    * відсутній індекс — 503;
    * збої моделі (таймаут, ліміт, автентифікація, schema) — з ПР3–ПР4;
    * несподівана помилка — 500.
    """
    question = (payload.question or "").strip()
    if not question:
        raise HTTPException(status_code=400, detail="Питання не може бути порожнім.")
    if len(question) > 4000:
        raise HTTPException(status_code=400, detail="Питання занадто довге.")

    filters = dict(payload.filters or {})
    filters = {k: v for k, v in filters.items() if v not in (None, "", [])}

    if app.state.index is None:
        raise HTTPException(
            status_code=503,
            detail="Індекс не збудовано. Виконайте `python ingest.py` "
                   "і перезапустіть застосунок.",
        )

    try:
        result = rag.answer(
            question,
            app.state.index,
            app.state.keyword_index,
            filters=filters or None,
        )
        return answer_to_dict(result)
    except llm.LLMSchemaError as e:
        logger.error("SchemaError: %s | detail=%s",
                     e.message, getattr(e, "detail", ""))
        raise HTTPException(
            status_code=502,
            detail="Помічник тимчасово не може обробити відповідь. "
                   "Спробуйте ще раз.",
        ) from e
    except llm.LLMError as e:
        logger.error("LLMError: %s (status=%d)", e.message, e.status_code)
        detail = e.message
        if e.status_code >= 500:
            detail = "Сервіс тимчасово не може відповісти. Спробуйте пізніше."
        raise HTTPException(status_code=e.status_code, detail=detail) from e
    except Exception as e:
        logger.exception("Непередбачена помилка: %s", e)
        raise HTTPException(
            status_code=500, detail="Внутрішня помилка сервера."
        ) from e