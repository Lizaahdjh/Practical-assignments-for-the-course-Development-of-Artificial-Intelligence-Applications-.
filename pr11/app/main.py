"""Веб-рівень застосунку: сторінка помічника оператора і JSON-ендпоінти.

ПР11:
  • middleware request_context — X-Request-ID, http.request у журнал;
  • обробники помилок: RateLimited → 429, BudgetExceeded → 429,
    HTTPException → з request_id у тілі, X-Request-ID у заголовку,
    RequestValidationError → стандартний detail + request_id;
  • приглушено httpx/httpcore/openai (див. telemetry.setup()) —
    щоб журнал не забивався технічними записами.

Запуск:
    uvicorn app.main:app --reload
"""

import datetime
import json
import logging
import re
import time
from dataclasses import asdict
from pathlib import Path
from typing import Iterator

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel

from . import (budget, cache, costs, drafting, gateway, jobs, limits,
               samples, telemetry)

telemetry.setup()
log = logging.getLogger("app.http")

app = FastAPI(title="Помічник оператора — ПР11")

INDEX_PAGE = Path(__file__).parent / "templates" / "index.html"

_REQUEST_ID = re.compile(r"^[A-Za-z0-9._-]{4,64}$")


class DraftIn(BaseModel):
    request_id: str
    client: str = "anonymous"
    fresh: bool = False


# ---------------------------------------------------------------- middleware

@app.middleware("http")
async def request_context(request: Request, call_next):
    incoming = request.headers.get("x-request-id", "")
    request_id = (incoming if _REQUEST_ID.match(incoming)
                  else telemetry.new_request_id())
    token = telemetry.request_id_var.set(request_id)
    started = time.perf_counter()
    status = 500
    try:
        response = await call_next(request)
        status = response.status_code
        response.headers["X-Request-ID"] = request_id
        return response
    except Exception:
        log.exception("http.unhandled",
                      extra={"method": request.method,
                             "path": request.url.path})
        raise
    finally:
        log.info("http.request", extra={
            "method": request.method,
            "path": request.url.path,
            "status": status,
            "duration_ms": round((time.perf_counter() - started) * 1000, 1),
        })
        telemetry.request_id_var.reset(token)


# ---------------------------------------------------------------- помилки

def _error_body(kind: str, message: str, request_id: str | None,
                extra: dict | None = None) -> dict:
    body = {"error": {"kind": kind, "message": message,
                      "request_id": request_id}}
    if extra:
        body["error"].update(extra)
    return body


def _error_response(status_code: int, body: dict,
                    headers: dict | None = None) -> JSONResponse:
    rid = telemetry.request_id_var.get()
    h = {"X-Request-ID": rid} if rid else {}
    if headers:
        h.update(headers)
    return JSONResponse(status_code=status_code, content=body, headers=h)


@app.exception_handler(limits.RateLimited)
async def _rate_limited_handler(request: Request, exc: limits.RateLimited):
    rid = telemetry.request_id_var.get()
    seconds = max(1, int(round(exc.retry_after)))
    return _error_response(
        429,
        _error_body("rate_limit",
                    f"Занадто багато запитів. Повторіть через {seconds} с.",
                    rid, {"retry_after": seconds}),
        headers={"Retry-After": str(seconds)},
    )


@app.exception_handler(budget.BudgetExceeded)
async def _budget_handler(request: Request, exc: budget.BudgetExceeded):
    rid = telemetry.request_id_var.get()
    try:
        reset = datetime.datetime.fromisoformat(exc.resets_at)
        if reset.tzinfo is not None:
            now = datetime.datetime.now(reset.tzinfo)
            seconds = max(1, int((reset - now).total_seconds()))
        else:
            seconds = max(1, int((reset - datetime.datetime.now()).total_seconds()))
    except (ValueError, TypeError):
        seconds = 3600
    return _error_response(
        429,
        _error_body(
            "budget",
            f"Денний ліміт витрат вичерпано "
            f"(витрачено {exc.spent:.4f} $ з {exc.limit:.4f} $). "
            f"Відновиться {exc.resets_at}.",
            rid,
            {"spent": round(exc.spent, 6), "limit": exc.limit,
             "resets_at": exc.resets_at},
        ),
        headers={"Retry-After": str(seconds)},
    )


@app.exception_handler(RequestValidationError)
async def _validation_handler(request: Request, exc: RequestValidationError):
    rid = telemetry.request_id_var.get()
    headers = {"X-Request-ID": rid} if rid else {}
    return JSONResponse(status_code=422,
                        content={"detail": exc.errors(),
                                 "request_id": rid},
                        headers=headers)


@app.exception_handler(HTTPException)
async def _http_handler(request: Request, exc: HTTPException):
    rid = telemetry.request_id_var.get()
    detail = exc.detail
    if isinstance(detail, dict) and "error" in detail:
        return _error_response(exc.status_code, detail,
                               headers=exc.headers or {})
    kind = {404: "not_found", 400: "bad_request",
            429: "rate_limit"}.get(exc.status_code, "http")
    return _error_response(
        exc.status_code,
        _error_body(kind, str(detail), rid),
        headers=exc.headers or {},
    )


# ---------------------------------------------------------------- допоміжне

def draft_to_dict(draft: jobs.Draft) -> dict:
    data = asdict(draft)
    data["events"] = len(draft.events)
    return data


def _or_pending(fn) -> dict:
    try:
        return fn()
    except NotImplementedError as exc:
        return {"pending": str(exc)}


# ---------------------------------------------------------------- ендпоінти

@app.get("/", response_class=HTMLResponse)
def page() -> str:
    return INDEX_PAGE.read_text(encoding="utf-8")


@app.get("/api/health")
def api_health() -> dict:
    return {
        "gateway": gateway.settings(),
        "limits": limits.settings(),
        "cache": cache.settings(),
        "budget": budget.settings(),
        "prompt_version": getattr(drafting, "PROMPT_VERSION", None),
        "log_file": telemetry.settings()["log_file"],
    }


@app.get("/api/usage")
def api_usage() -> dict:
    return {
        "budget": _or_pending(budget.status),
        "cache": _or_pending(cache.stats),
        "pricing": costs.settings(),
    }


@app.get("/api/requests")
def api_requests() -> list[dict]:
    return [samples.public(s) for s in samples.all_samples()]


@app.post("/api/drafts", status_code=202)
def api_create(payload: DraftIn) -> dict:
    sample = samples.get(payload.request_id)
    if sample is None:
        raise HTTPException(
            status_code=404,
            detail=f"Звернення {payload.request_id!r} не знайдено.",
        )
    limits.acquire(payload.client)
    try:
        return draft_to_dict(jobs.start(sample, payload.client,
                                        fresh=payload.fresh))
    except HTTPException:
        raise
    except Exception as exc:
        log.exception("api_create: не вдалося запустити чернетку")
        raise HTTPException(
            status_code=500,
            detail="Не вдалося запустити генерацію чернетки.",
        ) from exc


@app.get("/api/drafts")
def api_drafts() -> list[dict]:
    return [draft_to_dict(d) for d in jobs.list_drafts()]


@app.get("/api/drafts/{draft_id}")
def api_draft(draft_id: str) -> dict:
    draft = jobs.get(draft_id)
    if draft is None:
        raise HTTPException(status_code=404,
                            detail=f"Чернетки {draft_id!r} не знайдено.")
    return draft_to_dict(draft)


@app.get("/api/drafts/{draft_id}/events")
def api_events(draft_id: str, request: Request) -> StreamingResponse:
    draft = jobs.get(draft_id)
    if draft is None:
        raise HTTPException(status_code=404,
                            detail=f"Чернетки {draft_id!r} не знайдено.")
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
            yield (f"id: {event['id']}\nevent: {event['event']}\n"
                   f"data: {data}\n\n")

    return StreamingResponse(
        stream(), media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/api/drafts/{draft_id}/cancel")
def api_cancel(draft_id: str) -> dict:
    draft = jobs.get(draft_id)
    if draft is None:
        raise HTTPException(status_code=404,
                            detail=f"Чернетки {draft_id!r} не знайдено.")
    try:
        return draft_to_dict(jobs.cancel(draft_id))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc