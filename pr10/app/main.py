"""Веб-рівень застосунку: сторінка помічника оператора і JSON-ендпоінти.

Цей файл не знає ані який провайдер відповідає, ані скільки було спроб,
ані в якому потоці йде генерація — усе це в `app/gateway.py`,
`app/providers/` і `app/jobs.py`. Тут вирішується інше: що застосунок
приймає від сторінки, що віддає їй і з яким HTTP-статусом, і як події
фонової роботи стають потоком server-sent events.

Запуск із папки pr10:

    uvicorn app.main:app --reload

Далі відкрийте http://127.0.0.1:8000
"""

import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Iterator

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel

from . import gateway, jobs, limits, samples

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Помічник оператора — ПР10")

INDEX_PAGE = Path(__file__).parent / "templates" / "index.html"


class DraftIn(BaseModel):
    """Замовлення чернетки зі сторінки.

    `request_id` — яке звернення зі зразків. `client` — ідентифікатор
    клієнта сторінки для обмеження частоти; у справжньому застосунку
    його взяв би із сесії веб-рівень, а не сторінка.
    """

    request_id: str
    client: str = "anonymous"


def draft_to_dict(draft: jobs.Draft) -> dict:
    """Знімок чернетки для сторінки — без журналу подій."""
    data = asdict(draft)
    data["events"] = len(draft.events)
    return data


@app.get("/", response_class=HTMLResponse)
def page() -> str:
    """Віддати сторінку."""
    return INDEX_PAGE.read_text(encoding="utf-8")


@app.get("/api/health")
def api_health() -> dict:
    """Чинні налаштування — щоб сторінка показала, хто відповідає і які
    обмеження діють. Ключа тут немає."""
    return {"gateway": gateway.settings(), "limits": limits.settings()}


@app.get("/api/requests")
def api_requests() -> list[dict]:
    """Зразки звернень для вибору на сторінці."""
    return [samples.public(s) for s in samples.all_samples()]


@app.post("/api/drafts", status_code=202)
def api_create(payload: DraftIn) -> dict:
    """Замовити чернетку. Повертає її стан одразу, не чекаючи на
    генерацію; далі сторінка слухає події.

    Обробка збоїв:

    * невідоме звернення — 404;
    * вичерпаний ліміт частоти — 429 з `Retry-After`;
    * збій запуску фонової роботи — 500 зі зрозумілим текстом;
    * збій провайдера сюди не доходить — він стається пізніше, у фоні,
      і стає подією `error`.
    """
    sample = samples.get(payload.request_id)
    if sample is None:
        raise HTTPException(
            status_code=404,
            detail=f"Звернення {payload.request_id!r} не знайдено.",
        )

    try:
        limits.acquire(payload.client)
    except limits.RateLimited as e:
        raise HTTPException(
            status_code=429,
            detail=f"Занадто багато запитів. Повторіть через {e.retry_after:.0f} с.",
            headers={"Retry-After": str(int(round(e.retry_after)))},
        ) from e

    try:
        draft = jobs.start(sample, payload.client)
        return draft_to_dict(draft)
    except Exception as e:
        logger.exception("Помилка запуску чернетки: %s", e)
        raise HTTPException(
            status_code=500,
            detail="Не вдалося запустити генерацію.",
        ) from e


@app.get("/api/drafts")
def api_drafts() -> list[dict]:
    """Останні чернетки, нові першими."""
    return [draft_to_dict(d) for d in jobs.list_drafts()]


@app.get("/api/drafts/{draft_id}")
def api_draft(draft_id: str) -> dict:
    """Знімок однієї чернетки. Невідомий номер — 404."""
    draft = jobs.get(draft_id)
    if draft is None:
        raise HTTPException(
            status_code=404,
            detail=f"Чернетки {draft_id!r} не знайдено.",
        )
    return draft_to_dict(draft)


@app.get("/api/drafts/{draft_id}/events")
def api_events(draft_id: str, request: Request) -> StreamingResponse:
    """Потік подій чернетки у форматі server-sent events.

    Браузер при обриві зʼєднання перепідключається сам і надсилає
    заголовок `Last-Event-ID` — звідси продовжуємо, не повторюючи
    віддане. Порожній коментар раз на кілька секунд тримає зʼєднання
    живим; якщо клієнт відключився, спроба записати в нього завершує
    генератор.
    """
    draft = jobs.get(draft_id)
    if draft is None:
        raise HTTPException(
            status_code=404,
            detail=f"Чернетки {draft_id!r} не знайдено.",
        )

    try:
        since = int(request.headers.get("last-event-id") or 0)
    except ValueError:
        since = 0

    def stream() -> Iterator[str]:
        for event in jobs.subscribe(draft_id, since):
            if event is None:
                yield ": ping\n\n"
                continue
            data = json.dumps(event["data"], ensure_ascii=False)
            yield f"id: {event['id']}\nevent: {event['event']}\ndata: {data}\n\n"

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/api/drafts/{draft_id}/cancel")
def api_cancel(draft_id: str) -> dict:
    """Скасувати генерацію.

    * невідома чернетка — 404;
    * чернетка вже завершена — 400;
    * інакше — 200 зі станом чернетки (статус `cancelled`).
    """
    draft = jobs.get(draft_id)
    if draft is None:
        raise HTTPException(
            status_code=404,
            detail=f"Чернетки {draft_id!r} не знайдено.",
        )
    try:
        return draft_to_dict(jobs.cancel(draft_id))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e