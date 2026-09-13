"""Веб-сервер FastAPI з локалізацією."""

from pathlib import Path
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from . import weather

app = FastAPI(title="Weather App — Glassmorphism")

BASE_DIR = Path(__file__).parent
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")

INDEX_PAGE = BASE_DIR / "templates" / "index.html"


@app.get("/", response_class=HTMLResponse)
async def index() -> str:
    return INDEX_PAGE.read_text(encoding="utf-8")


@app.get("/api/weather", response_model=weather.WeatherResponse)
async def api_weather(
    city: str = Query(..., description="Назва міста / City name"),
    lang: str = Query("uk", description="Мова відповіді / Language code ('uk' або 'en')")
):
    try:
        return await weather.get_current_weather(city, lang=lang)
    except weather.WeatherError as exc:
        msg = exc.message_en if lang == "en" else exc.message_uk
        raise HTTPException(status_code=exc.status_code, detail=msg)
    except Exception:
        err_msg = "Internal server error" if lang == "en" else "Внутрішня помилка сервера"
        raise HTTPException(status_code=500, detail=err_msg)