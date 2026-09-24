"""Контракт відповіді моделі: що саме вона має повернути і як це
перевіряється.

Рішення:

* схема описана pydantic-моделлю `ModelReply` — з неї генерується
  JSON Schema для `response_format` і нею ж перевіряється те, що
  повернула модель;
* поля: `answer` (текст клієнту), `sources` (номери джерел з контексту),
  `found` (чи знайдено відповідь у фрагментах);
* `sources` — список цілих чисел; порожній список допустимий лише коли
  `found=false`; це перевіряє код у `rag.answer`, а не схема — схема
  відповідає лише за форму;
* `answer` — обовʼязковий і непорожній: модель не може «промовчати».
"""

import json
import re
from typing import Any, Dict, List

from pydantic import BaseModel, Field, ValidationError, field_validator


class ModelReply(BaseModel):
    """Те, що модель має повернути за інструкцією."""

    answer: str = Field(
        ...,
        min_length=1,
        max_length=4000,
        description="Текст відповіді для клієнта.",
    )
    sources: List[int] = Field(
        default_factory=list,
        description="Номери фрагментів контексту, на які спирається "
                    "відповідь. Порожній список — якщо не спирається ні на що.",
    )
    found: bool = Field(
        ...,
        description="True, якщо відповідь знайдено у наданих фрагментах; "
                    "False — якщо ні.",
    )

    @field_validator("sources")
    @classmethod
    def _check_sources(cls, v: List[int]) -> List[int]:
        # Усі номери — цілі додатні; дублікати прибираємо
        cleaned = sorted({int(x) for x in v if int(x) > 0})
        return cleaned


def output_schema() -> Dict[str, Any]:
    """JSON Schema для `response_format`."""
    return ModelReply.model_json_schema()


class SchemaError(Exception):
    """Відповідь моделі не пройшла перевірку за схемою."""

    def __init__(self, detail: str, raw: str = ""):
        super().__init__(detail)
        self.detail = detail
        self.raw = raw


_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


def _extract_json(raw: str) -> str:
    """Прибрати можливу огорожу ```json ... ``` і текст довкола."""
    text = (raw or "").strip()
    text = _FENCE_RE.sub("", text).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        text = text[start:end + 1]
    return text


def validate(raw: str) -> Dict[str, Any]:
    """Перевірити сиру відповідь моделі й повернути дані, яким можна
    довіряти структурно."""
    if not raw or not raw.strip():
        raise SchemaError("порожня відповідь моделі", raw=raw or "")

    candidate = _extract_json(raw)

    try:
        data = json.loads(candidate)
    except json.JSONDecodeError as e:
        raise SchemaError(f"відповідь не є JSON: {e.msg}", raw=raw) from e

    if not isinstance(data, dict):
        raise SchemaError("відповідь не є JSON-обʼєктом", raw=raw)

    try:
        model = ModelReply.model_validate(data)
    except ValidationError as e:
        first = e.errors()[0] if e.errors() else {"msg": "невідома помилка"}
        loc = ".".join(str(p) for p in first.get("loc", [])) or "?"
        raise SchemaError(f"поле «{loc}»: {first.get('msg', 'невалідне')}",
                          raw=raw) from e

    return model.model_dump()