"""Фонові задачі: чернетка генерується в окремому потоці.

ПР11:
  • fresh — обхід кешу;
  • request_id у фоні через contextvars.copy_context();
  • cached / cost / cost_estimated у Draft і події done;
  • журнал: draft.created, draft.done, draft.failed, draft.cancelled.

Скасування — Event, прив'язаний до draft_id; перевіряється шлюзом.
cancel() НЕ змінює статус — це робить runner, щоб не було гонки.
"""

import contextvars
import datetime
import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Iterator

log = logging.getLogger("app.jobs")

TERMINAL = {"done", "failed", "cancelled"}


@dataclass
class Draft:
    draft_id: str
    request_id: str
    client: str
    status: str = "queued"
    text: str = ""
    attempts: int = 0
    provider: str | None = None
    model: str | None = None
    fallback: bool = False
    cached: bool = False
    cost: float | None = None
    cost_estimated: bool = False
    error: dict | None = None
    usage: dict | None = None
    elapsed: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    created_at: str = ""
    updated_at: str = ""
    events: list[dict] = field(default_factory=list)


_drafts: dict[str, Draft] = {}
_order: list[str] = []
_lock = threading.Lock()
_changed = threading.Condition(_lock)
_cancel_events: dict[str, threading.Event] = {}


def _now() -> str:
    return datetime.datetime.now().isoformat(timespec="seconds")


def new_draft(request_id: str, client: str) -> Draft:
    draft = Draft(draft_id=uuid.uuid4().hex[:8], request_id=request_id,
                  client=client, created_at=_now(), updated_at=_now())
    with _lock:
        _drafts[draft.draft_id] = draft
        _order.insert(0, draft.draft_id)
    return draft


def get(draft_id: str) -> Draft | None:
    with _lock:
        return _drafts.get(draft_id)


def list_drafts(limit: int = 50) -> list[Draft]:
    with _lock:
        return [_drafts[i] for i in _order[:limit]]


def emit(draft: Draft, event: str, data: dict | None = None) -> None:
    with _changed:
        draft.events.append({"id": len(draft.events) + 1, "event": event,
                             "data": data or {}})
        draft.updated_at = _now()
        _changed.notify_all()


def subscribe(draft_id: str, since: int = 0,
              heartbeat: float = 15.0) -> Iterator[dict | None]:
    while True:
        with _changed:
            draft = _drafts.get(draft_id)
            if draft is None:
                return
            pending = draft.events[since:]
            if not pending:
                if draft.status in TERMINAL:
                    return
                _changed.wait(timeout=heartbeat)
                pending = draft.events[since:]
        if not pending:
            yield None
            continue
        for event in pending:
            yield event
        since += len(pending)


def start(sample: dict, client: str, fresh: bool = False) -> Draft:
    from . import drafting, gateway

    draft = new_draft(sample["id"], client)

    log.info("draft.created", extra={
        "draft_id": draft.draft_id,
        "request": sample["id"],
        "client": client,
        "fresh": fresh,
        "request_length": len(sample.get("text", "")),
        "facts_count": len(sample.get("facts", [])),
    })

    request = drafting.build_request(sample)
    request.metadata.update({
        "draft_id": draft.draft_id,
        "client": client,
        "fresh": fresh,
        "request_id": sample["id"],
    })

    cancel_event = threading.Event()
    _cancel_events[draft.draft_id] = cancel_event

    def runner() -> None:
        try:
            with _lock:
                draft.status = "running"
                draft.updated_at = _now()
            emit(draft, "status", {"status": "running", "attempt": 1})

            first_delta_at = {"t": None}
            started = time.perf_counter()

            def on_delta(text: str) -> None:
                if first_delta_at["t"] is None:
                    first_delta_at["t"] = time.perf_counter() - started
                with _lock:
                    draft.text += text
                emit(draft, "delta", {"text": text})

            def on_status(info: dict) -> None:
                if info.get("attempt") and info["attempt"] > 1:
                    with _lock:
                        draft.text = ""
                with _lock:
                    if info.get("attempt"):
                        draft.attempts = info["attempt"]
                    if info.get("provider"):
                        draft.provider = info["provider"]
                    if info.get("model"):
                        draft.model = info["model"]
                    if info.get("fallback"):
                        draft.fallback = True
                emit(draft, "status", info)

            def is_cancelled() -> bool:
                return cancel_event.is_set()

            try:
                result = gateway.generate(request, on_delta, on_status,
                                          is_cancelled)
            except Exception as e:
                kind = getattr(e, "kind", "unknown")
                message = getattr(e, "message", str(e))
                retryable = getattr(e, "retryable", False)
                if cancel_event.is_set() or kind == "cancelled":
                    with _lock:
                        draft.status = "cancelled"
                    emit(draft, "cancelled", {})
                    log.info("draft.cancelled", extra={
                        "draft_id": draft.draft_id, "request": sample["id"],
                        "client": client,
                    })
                    return
                with _lock:
                    draft.status = "failed"
                    draft.error = {"kind": kind, "message": message,
                                   "retryable": retryable}
                emit(draft, "error", draft.error)
                log.error("draft.failed", extra={
                    "draft_id": draft.draft_id, "request": sample["id"],
                    "client": client, "kind": kind,
                })
                return

            if cancel_event.is_set():
                with _lock:
                    draft.status = "cancelled"
                emit(draft, "cancelled", {})
                log.info("draft.cancelled", extra={
                    "draft_id": draft.draft_id, "request": sample["id"],
                    "client": client,
                })
                return

            warnings = drafting.check(result.text, sample)

            with _lock:
                draft.status = "done"
                draft.text = result.text or draft.text
                draft.attempts = draft.attempts or 1
                draft.provider = result.provider
                draft.model = result.model
                draft.usage = result.usage
                draft.cached = bool(getattr(result, "cached", False))
                draft.cost = getattr(result, "cost", None)
                draft.cost_estimated = bool(getattr(result, "cost_estimated", False))
                draft.warnings = warnings
                draft.elapsed = {
                    "first_token": round(first_delta_at["t"], 3)
                                   if first_delta_at["t"] else None,
                    "total": round(time.perf_counter() - started, 3),
                }
            emit(draft, "done", {
                "text": draft.text,
                "provider": draft.provider,
                "model": draft.model,
                "usage": draft.usage,
                "elapsed": draft.elapsed,
                "attempts": draft.attempts,
                "fallback": draft.fallback,
                "cached": draft.cached,
                "cost": draft.cost,
                "cost_estimated": draft.cost_estimated,
                "warnings": draft.warnings,
            })
            log.info("draft.done", extra={
                "draft_id": draft.draft_id, "request": sample["id"],
                "client": client, "attempts": draft.attempts,
                "provider": draft.provider, "model": draft.model,
                "cached": draft.cached,
                "cost": draft.cost, "cost_estimated": draft.cost_estimated,
                "duration_ms": round(draft.elapsed["total"] * 1000, 1),
            })
        except Exception:
            log.exception("Несподіваний збій фонової роботи %s",
                          draft.draft_id)
            with _lock:
                if draft.status not in TERMINAL:
                    draft.status = "failed"
                    draft.error = {"kind": "unknown",
                                   "message": "Внутрішній збій фонової роботи.",
                                   "retryable": False}
            emit(draft, "error", draft.error or {})
        finally:
            _cancel_events.pop(draft.draft_id, None)

    ctx = contextvars.copy_context()
    threading.Thread(target=ctx.run, args=(runner,), daemon=True,
                     name=f"draft-{draft.draft_id}").start()
    return draft


def cancel(draft_id: str) -> Draft:
    """Попросити фонову роботу зупинитися. Статус змінить runner."""
    draft = get(draft_id)
    if draft is None:
        raise ValueError(f"чернетки {draft_id} не знайдено")
    if draft.status in TERMINAL:
        raise ValueError(f"чернетка вже {draft.status}, скасувати не можна")
    ev = _cancel_events.get(draft_id)
    if ev:
        ev.set()
    return draft