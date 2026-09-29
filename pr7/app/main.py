"""Веб-рівень застосунку: сторінка розбору рахунків і JSON-ендпоінти.

Цей файл не знає ані як готується зображення, ані якою моделлю й за
якою інструкцією вилучено поля, ані якими правилами їх перевірено — усе
це в `app/images.py`, `app/llm.py`, `app/schema.py`, `app/rules.py` і
поєднується в `app/extraction.py`. Тут вирішується інше: що застосунок
приймає від сторінки, що віддає їй і з яким HTTP-статусом.

Запуск із папки pr7:

    uvicorn app.main:app --reload

Далі відкрийте http://127.0.0.1:8000
"""

import logging
from dataclasses import asdict
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from . import extraction, images, llm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Розбір рахунків — ПР7")

INDEX_PAGE = Path(__file__).parent / "templates" / "index.html"
SAMPLES_DIR = Path(__file__).resolve().parent.parent / "samples"
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}

# Зразки віддаються як статичні файли, щоб сторінка могла їх показати й
# надіслати на розбір так само, як завантажений користувачем файл.
app.mount("/samples", StaticFiles(directory=SAMPLES_DIR), name="samples")


def result_to_dict(result: extraction.Result) -> dict:
    """Перетворити результат конвеєра на те, що піде на сторінку."""
    return {
        "decision": result.decision,
        "reasons": result.reasons,
        "document": result.document,
        "issues": [asdict(issue) for issue in result.issues],
        "image": result.image,
        "model": result.model,
        "elapsed": result.elapsed,
        "usage": result.usage,
    }


@app.get("/", response_class=HTMLResponse)
def page() -> str:
    """Віддати сторінку."""
    return INDEX_PAGE.read_text(encoding="utf-8")


@app.get("/api/samples")
def api_samples() -> dict:
    """Перелік зразків за папками: `clean`, `degraded` і, якщо є, `own`."""
    groups = {}
    for folder in sorted(p for p in SAMPLES_DIR.iterdir() if p.is_dir()):
        files = sorted(f.name for f in folder.iterdir() if f.suffix.lower() in IMAGE_SUFFIXES)
        if files:
            groups[folder.name] = files
    return groups


@app.post("/api/extract")
async def api_extract(image: UploadFile = File(...)) -> dict:
    """Розібрати документ і повернути поля, проблеми й рішення.

    Сторінка очікує обʼєкт із полями `decision`, `reasons`, `document`,
    `issues`, `image`, `model`, `elapsed`, `usage` — див. `result_to_dict`.

    Обробка збоїв:

    * порожній файл / не зображення / завеликий файл — 400/422;
    * замале зображення — 422 (треба перезняти);
    * збої моделі (таймаут, ліміт, авторизація, schema) — з ПР3–ПР6;
    * несподівана помилка — 500.
    """
    try:
        content = await image.read()
    except Exception as e:
        logger.exception("Не вдалося прочитати файл: %s", e)
        raise HTTPException(status_code=400, detail="Не вдалося прочитати файл.") from e

    if not content:
        raise HTTPException(status_code=400, detail="Порожній файл.")

    try:
        result = extraction.process(content)
        return result_to_dict(result)
    except images.ImageError as e:
        # Файл не можна обробити: не зображення, завеликий, замалий.
        # 422 — семантично правильніше за 400: файл отримано, але
        # обробити його не вийшло.
        raise HTTPException(status_code=422, detail=str(e)) from e
    except llm.LLMSchemaError as e:
        logger.error("SchemaError: %s | detail=%s",
                     e.message, getattr(e, "detail", ""))
        raise HTTPException(
            status_code=502,
            detail="Модель повернула невалідну відповідь. Спробуйте ще раз.",
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