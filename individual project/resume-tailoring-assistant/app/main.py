import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse

from app.gateway import build_provider, Gateway
from app.models import (
    AnalyzeRequest,
    AnalyzeResult,
    ErrorBody,
    ErrorEnvelope,
    ErrorKind,
    HealthResponse,
)
from app.providers import ProviderError
from app.service import Service
from app.settings import get_settings
from app.storage import Storage
from app.telemetry import configure as configure_telemetry, log

START_TS = time.time()


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    configure_telemetry(settings.log_file)
    app.state.settings = settings
    app.state.storage = Storage(settings.db_path)
    app.state.gateway = Gateway(build_provider(settings))
    app.state.service = Service(settings, app.state.gateway, app.state.storage)
    yield


app = FastAPI(title="Resume Tailoring Assistant", lifespan=lifespan)


def error_response(kind: ErrorKind, message: str, request_id: str, status: int, retryable: bool = False) -> JSONResponse:
    body = ErrorEnvelope(error=ErrorBody(kind=kind, message=message, retryable=retryable, request_id=request_id))
    return JSONResponse(status_code=status, content=body.model_dump())


@app.middleware("http")
async def add_request_id(request: Request, call_next):
    request_id = request.headers.get("x-request-id") or str(uuid.uuid4())
    request.state.request_id = request_id
    response = await call_next(request)
    response.headers["x-request-id"] = request_id
    return response


@app.get("/api/health", response_model=HealthResponse)
async def health():
    settings = get_settings()
    storage = app.state.storage
    return HealthResponse(
        status="ok",
        version=settings.app_version,
        provider=settings.llm_provider,
        model=settings.llm_model,
        mode="mock" if settings.llm_provider == "mock" else "real",
        storage="ok" if storage.ping() else "down",
        extra={"index": "not_loaded"},  # ПР16
        uptime_s=round(time.time() - START_TS, 1),
    )


@app.post("/api/analyze", response_model=AnalyzeResult)
async def analyze(payload: AnalyzeRequest, request: Request):
    request_id = request.state.request_id
    log("controller", request_id, event="received")
    try:
        return await app.state.service.analyze(request_id, payload)
    except ProviderError as e:
        kind = ErrorKind(e.kind) if e.kind in ErrorKind.__members__ else ErrorKind.provider_unavailable
        status = 504 if kind == ErrorKind.timeout else (429 if kind == ErrorKind.rate_limit else 503)
        log("controller", request_id, event="provider_error", kind=e.kind)
        return error_response(kind, e.message, request_id, status, retryable=e.retryable)


@app.get("/api/requests/{request_id}")
async def get_request(request_id: str):
    row = app.state.storage.get_request(request_id)
    if row is None:
        return error_response(ErrorKind.not_found, "request not found", request_id, 404)
    return row


@app.get("/", response_class=HTMLResponse)
async def index():
    return """
    <!doctype html><html><head><meta charset="utf-8"><title>Resume Tailoring</title></head>
    <body style="font-family:sans-serif;max-width:720px;margin:2em auto">
      <h1>Resume Tailoring Assistant (ПР14)</h1>
      <p>Мінімальний інтерфейс для перевірки каркасу. У ПР15 з'явиться повноцінний UI.</p>
      <form id="f">
        <label>Вакансія<br><textarea id="v" rows="4" cols="60">Python backend developer. Django, PostgreSQL.</textarea></label><br>
        <label>Резюме<br><textarea id="r" rows="4" cols="60">Олена К. Python developer, 3 роки. Django, DRF.</textarea></label><br>
        <button type="submit">Аналізувати</button>
      </form>
      <pre id="out"></pre>
      <script>
        const f = document.getElementById('f');
        f.addEventListener('submit', async (e) => {
          e.preventDefault();
          const body = { vacancy_text: v.value, resume_text: r.value };
          const res = await fetch('/api/analyze', { method:'POST', headers:{'content-type':'application/json'}, body: JSON.stringify(body) });
          const data = await res.json();
          document.getElementById('out').textContent = JSON.stringify(data, null, 2);
        });
      </script>
    </body></html>
    """