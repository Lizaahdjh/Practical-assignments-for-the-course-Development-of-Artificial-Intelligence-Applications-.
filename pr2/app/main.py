"""Веб-рівень застосунку: сторінка із завантаженням файлу та JSON-ендпоінт."""

import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, Optional

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile, status
from fastapi.responses import HTMLResponse

from . import detector

_last_request_time: Dict[str, float] = {}
RATE_LIMIT_SECONDS = 3.0  # Інтервал захисту від частого виклику


@asynccontextmanager
async def lifespan(app: FastAPI):
    detector.load_model()
    yield


app = FastAPI(title="Object Detection API — PR2", lifespan=lifespan)

INDEX_PAGE = Path(__file__).parent / "templates" / "index.html"


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return INDEX_PAGE.read_text(encoding="utf-8")


@app.post("/api/detect")
async def api_detect(
    request: Request,
    image: UploadFile = File(...),
    confidence: Optional[float] = Form(None),
):
    client_ip = request.client.host if request.client else "127.0.0.1"
    current_time = time.time()

    if client_ip in _last_request_time:
        elapsed = current_time - _last_request_time[client_ip]
        if elapsed < RATE_LIMIT_SECONDS:
            wait_time = round(RATE_LIMIT_SECONDS - elapsed, 1)
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"Rate limit exceeded. Please wait {wait_time}s / Зачекайте {wait_time}с.",
            )

    _last_request_time[client_ip] = current_time
    content = await image.read()

    conf_value = confidence if confidence is not None else detector.DEFAULT_CONFIDENCE

    if not (0.0 <= conf_value <= 1.0):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Confidence threshold must be between 0.0 and 1.0 / Поріг повинен бути в межах від 0.0 до 1.0",
        )

    try:
        return detector.detect(content, confidence=conf_value)
    except detector.InvalidImageError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except (detector.ModelInferenceError, detector.DetectionError) as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))