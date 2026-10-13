"""Контракт кроку розбору: що модель має витягти зі звернення клієнта
і як це перевіряється.

Рішення:

* `kind` — enum із `return` / `cancel` / `status` / `other`; `other` —
  «не розумію, про що звернення»;
* `order_id` — рядок цифр або `null`; `null` — «клієнт не назвав»;
* `item` — рядок, як назвав клієнт (вільна форма);
* `reason` — enum `not_suitable` / `defect` / `wrong_item` / `damaged`
  або `null`; заповнюється лише для `kind="return"`;
* `additionalProperties: false` — модель не може додати зайвих полів;
* `validate` перевіряє форму, не зміст. Чи існує замовлення і чиє воно —
  перевірка змісту на наступному кроці.
"""

import json
import re
from enum import Enum
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field, ValidationError, field_validator


class Kind(str, Enum):
    RETURN = "return"
    CANCEL = "cancel"
    STATUS = "status"
    OTHER = "other"


class Reason(str, Enum):
    NOT_SUITABLE = "not_suitable"
    DEFECT = "defect"
    WRONG_ITEM = "wrong_item"
    DAMAGED = "damaged"


_ORDER_RE = re.compile(r"^\d{3,8}$")


class Parsed(BaseModel):
    """Результат розбору звернення."""

    kind: Kind = Field(..., description="Вид звернення")
    order_id: Optional[str] = Field(
        None, description="Номер замовлення, якщо клієнт його назвав"
    )
    item: Optional[str] = Field(
        None, description="Назва товару, як клієнт його назвав"
    )
    reason: Optional[Reason] = Field(
        None, description="Причина повернення, якщо це повернення"
    )

    @field_validator("order_id")
    @classmethod
    def _check_order(cls, v: Optional[str]) -> Optional[str]:
        if v is None or v == "":
            return None
        if not _ORDER_RE.match(v):
            raise ValueError("номер замовлення має бути 3–8 цифр")
        return v

    @field_validator("item")
    @classmethod
    def _clean_item(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        v = v.strip()
        return v or None

    @field_validator("reason")
    @classmethod
    def _clean_reason(cls, v: Optional[Reason]) -> Optional[Reason]:
        return v


def request_schema() -> Dict[str, Any]:
    """JSON Schema результату розбору для `response_format`."""
    return Parsed.model_json_schema()


class SchemaError(Exception):
    def __init__(self, detail: str, raw: str = ""):
        super().__init__(detail)
        self.detail = detail
        self.raw = raw


_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


def _extract_json(raw: str) -> str:
    text = (raw or "").strip()
    text = _FENCE_RE.sub("", text).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        text = text[start:end + 1]
    return text


def validate(raw: str) -> Dict[str, Any]:
    """Перевірити сиру відповідь моделі на кроці розбору."""
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
        parsed = Parsed.model_validate(data)
    except ValidationError as e:
        first = e.errors()[0] if e.errors() else {"msg": "невідома помилка"}
        loc = ".".join(str(p) for p in first.get("loc", [])) or "?"
        raise SchemaError(
            f"поле «{loc}»: {first.get('msg', 'невалідне')}", raw=raw
        ) from e

    result = parsed.model_dump(mode="json")

    # Семантична перевірка кодом: причина має бути лише для return
    if result["kind"] != "return" and result["reason"] is not None:
        result["reason"] = None

    return result