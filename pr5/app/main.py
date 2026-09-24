"""Веб-рівень застосунку: сторінка пошуку і JSON-ендпоінти.

Цей файл не знає ані якою моделлю отримано вектори, ані як влаштований
індекс, ані як ранжує пошук за словами — усе це лишається в модулях
`app/embeddings.py`, `app/index.py`, `app/keyword.py`. Тут вирішується
інше: що застосунок приймає від сторінки, що віддає їй і з яким
HTTP-статусом.

Індекс будується заздалегідь командою `python ingest.py` (з папки pr5),
а тут лише читається при старті.

Запуск із папки pr5:

    uvicorn app.main:app --reload

Далі відкрийте http://127.0.0.1:8000
"""

import time
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from . import embeddings, index, keyword

app = FastAPI(title="Пошук у базі знань — ПР5")

INDEX_PAGE = Path(__file__).parent / "templates" / "index.html"

# Дозволені режими й поля фільтрів — щоб невідомий режим чи фільтр
# давали зрозумілу помилку, а не тихо ігнорувалися.
ALLOWED_MODES = {"semantic", "keyword", "both"}
ALLOWED_FILTER_FIELDS = {"category", "product", "audience", "status"}


class SearchRequest(BaseModel):
    """Те, що надсилає сторінка.

    `mode` — `semantic`, `keyword` або `both`. `filters` — умови на
    метадані фрагментів, наприклад `{"category": "інструкція"}`; порожній
    словник означає «без фільтрів». `threshold` — поріг схожості для
    семантичного пошуку; `None` — узяти значення з конфігурації.
    """

    query: str
    mode: str = "both"
    top_k: int = index.DEFAULT_TOP_K
    filters: dict = {}
    threshold: float | None = None


def hit_to_dict(hit: index.Hit) -> dict:
    """Перетворити влучення на те, що піде на сторінку."""
    return {
        "score": hit.score,
        "text": hit.chunk.text,
        "source": hit.chunk.source,
        "metadata": hit.chunk.metadata,
    }


@app.on_event("startup")
def load_indexes() -> None:
    """Прочитати збудований індекс і зібрати індекс за словами з тих
    самих фрагментів.

    Якщо індексу на диску немає, застосунок усе одно стартує: сторінка
    має відкритися й пояснити, що робити. `app.state.index` лишається
    `None`, а `/api/status` поверне `ready: false`.
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
    """Віддати сторінку пошуку."""
    return INDEX_PAGE.read_text(encoding="utf-8")


@app.get("/api/status")
def api_status() -> dict:
    """Стан індексу: чи збудовано, скільки фрагментів, якою моделлю."""
    idx = app.state.index
    if idx is None:
        return {
            "ready": False,
            "hint": "індекс не збудовано — виконайте python ingest.py "
                    "і перезапустіть застосунок",
        }
    sources = {chunk.source for chunk in idx.chunks}
    return {
        "ready": True,
        "chunks": len(idx),
        "documents": len(sources),
        "model": idx.model_name,
    }


@app.post("/api/search")
def api_search(payload: SearchRequest) -> dict:
    """Виконати пошук і повернути влучення обох способів.

    Сторінка очікує обʼєкт із полями `semantic` і `keyword` (список
    влучень або `null`, якщо спосіб не запитували) та `elapsed` — час
    кожного способу в секундах.

    Обробка збоїв:

    * порожній запит — 400;
    * невідомий режим — 400;
    * невідоме поле фільтра — 400;
    * `top_k` поза межами 1–50 — 400;
    * індекс не збудовано — 503 (з підказкою в detail);
    * несподівана помилка — 500.
    """
    # --- Валідація входу ---
    query = (payload.query or "").strip()
    if not query:
        raise HTTPException(status_code=400, detail="Запит не може бути порожнім.")

    if payload.mode not in ALLOWED_MODES:
        raise HTTPException(
            status_code=400,
            detail=f"Невідомий режим: {payload.mode!r}. "
                   f"Дозволені: {', '.join(sorted(ALLOWED_MODES))}.",
        )

    if payload.top_k < 1 or payload.top_k > 50:
        raise HTTPException(
            status_code=400,
            detail="top_k має бути в межах від 1 до 50.",
        )

    filters = dict(payload.filters or {})
    filters = {k: v for k, v in filters.items() if v not in (None, "", [])}
    if filters:
        unknown = set(filters) - ALLOWED_FILTER_FIELDS
        if unknown:
            raise HTTPException(
                status_code=400,
                detail=f"Невідомі поля фільтра: {', '.join(sorted(unknown))}. "
                       f"Дозволені: {', '.join(sorted(ALLOWED_FILTER_FIELDS))}.",
            )

    # Захисна межа безпеки: якщо користувач не звузив аудиторію —
    # показуємо лише клієнтські документи. Внутрішня інструкція
    # оператора не повинна зринати у відповіді клієнтові.
    # Це рішення коду, а не користувача.
    filters.setdefault("audience", "клієнти")

    # --- Індекс ---
    idx = app.state.index
    if payload.mode in ("semantic", "both") and idx is None:
        raise HTTPException(
            status_code=503,
            detail="Індекс не збудовано. Виконайте `python ingest.py` "
                   "і перезапустіть застосунок.",
        )
    if payload.mode in ("keyword", "both") and app.state.keyword_index is None:
        raise HTTPException(
            status_code=503,
            detail="Індекс не збудовано. Виконайте `python ingest.py` "
                   "і перезапустіть застосунок.",
        )

    result: dict = {
        "query": query,
        "semantic": None,
        "keyword": None,
        "elapsed": {},
    }

    # --- Семантичний ---
    if payload.mode in ("semantic", "both"):
        started = time.perf_counter()
        try:
            vector = embeddings.embed_query(query)
            hits = index.search(
                idx,
                vector,
                top_k=payload.top_k,
                filters=filters,
                threshold=payload.threshold if payload.threshold is not None
                else index.SIMILARITY_THRESHOLD,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(
                status_code=500,
                detail=f"Помилка семантичного пошуку: {type(exc).__name__}.",
            ) from exc
        result["elapsed"]["semantic"] = time.perf_counter() - started
        result["semantic"] = [hit_to_dict(h) for h in hits]

    # --- Keyword ---
    if payload.mode in ("keyword", "both"):
        started = time.perf_counter()
        try:
            hits = keyword.search(
                app.state.keyword_index,
                query,
                top_k=payload.top_k,
                filters=filters,
            )
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(
                status_code=500,
                detail=f"Помилка пошуку за ключовими словами: {type(exc).__name__}.",
            ) from exc
        result["elapsed"]["keyword"] = time.perf_counter() - started
        result["keyword"] = [hit_to_dict(h) for h in hits]

    return result