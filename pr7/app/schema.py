"""Контракт відповіді моделі: які поля вона має повернути й як це
перевіряється.

Рішення:

* схема описана pydantic-моделлю `InvoiceExtraction` — з неї генерується
  JSON Schema для `response_format` і нею ж перевіряється те, що
  повернула модель;
* суми — рядки з двома знаками після крапки (`"12570.00"`); конвертація
  в Decimal — на боці правил. Це уникає `float`-помилок;
* IBAN — без пробілів, верхній регістр;
* дати — ISO `РРРР-ММ-ДД`;
* `document_type` — enum `рахунок`/`видаткова накладна`/`інше`;
* відсутнє значення — `null` (однакова позначка для «немає в документі»
  і «не видно»; різниця лише в нотатці `expected.json`);
* `items` — список об'єктів з `name`, `unit`, `quantity`, `price`,
  `amount`;
* обов'язкові поля в схемі — тільки ті, без яких документ не має сенсу.
"""

import json
import re
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, ValidationError, field_validator


class DocumentType(str, Enum):
    INVOICE = "рахунок"
    WAYBILL = "видаткова накладна"
    OTHER = "інше"


# Регулярки для нормалізації в коді (модель уже має віддавати нормалізоване,
# але про всяк випадок підчищаємо)
_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_MONEY_RE = re.compile(r"^\d+(\.\d{1,2})?$")
_IBAN_RE = re.compile(r"^UA\d{27}$")


class Party(BaseModel):
    """Сторона документа: постачальник або покупець."""

    name: Optional[str] = None
    code: Optional[str] = None       # ЄДРПОУ (8) або РНОКПП (10)
    iban: Optional[str] = None        # лише для постачальника


class Item(BaseModel):
    """Одна позиція таблиці."""

    name: Optional[str] = None
    unit: Optional[str] = None
    quantity: Optional[str] = None    # рядок, як надруковано
    price: Optional[str] = None       # рядок, як надруковано
    amount: Optional[str] = None      # рядок, як надруковано


class InvoiceExtraction(BaseModel):
    """Поля, які модель вилучає з рахунку."""

    document_type: DocumentType = Field(
        ..., description="Тип документа: рахунок, накладна або інше."
    )
    number: Optional[str] = None
    date: Optional[str] = None
    valid_until: Optional[str] = None
    supplier: Party = Field(default_factory=Party)
    buyer: Party = Field(default_factory=Party)
    items: List[Item] = Field(default_factory=list)
    total_without_vat: Optional[str] = None
    vat: Optional[str] = None
    total: Optional[str] = None

    @field_validator("date", "valid_until")
    @classmethod
    def _check_date(cls, v: Optional[str]) -> Optional[str]:
        if v is None or v == "":
            return None
        if not _DATE_RE.match(v):
            raise ValueError("дата має бути у форматі РРРР-ММ-ДД")
        return v

    @field_validator("total_without_vat", "vat", "total")
    @classmethod
    def _check_money(cls, v):
        if v is None or v == "":
            return None
        # Приймаємо як рядок, так і число
        s = str(v).replace(" ", "").replace(",", ".")
        if not _MONEY_RE.match(s):
            raise ValueError("сума має бути числом або рядком виду '123.45'")
        # Завжди повертаємо рядок із двома знаками
        try:
            return f"{float(s):.2f}"
        except ValueError:
            raise ValueError(f"не вдалося привести {v!r} до числа")

    @field_validator("supplier", "buyer")
    @classmethod
    def _clean_party(cls, v: Party) -> Party:
        if v.iban:
            v.iban = v.iban.replace(" ", "").upper()
        if v.code:
            v.code = v.code.strip()
        return v

    @field_validator("items")
    @classmethod
    def _clean_items(cls, items: List[Item]) -> List[Item]:
        return items

class Item(BaseModel):
    name: Optional[str] = None
    unit: Optional[str] = None
    quantity: Optional[str] = None
    price: Optional[str] = None
    amount: Optional[str] = None

    @field_validator("quantity", "price", "amount")
    @classmethod
    def _to_str(cls, v):
        if v is None or v == "":
            return None
        s = str(v).replace(" ", "").replace(",", ".")
        if not _MONEY_RE.match(s):
            raise ValueError("має бути число або рядок '123.45'")
        return f"{float(s):.2f}"

def output_schema() -> Dict[str, Any]:
    """JSON Schema для `response_format`."""
    return InvoiceExtraction.model_json_schema()


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
    """Перевірити сиру відповідь моделі за схемою."""
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
        model = InvoiceExtraction.model_validate(data)
    except ValidationError as e:
        first = e.errors()[0] if e.errors() else {"msg": "невідома помилка"}
        loc = ".".join(str(p) for p in first.get("loc", [])) or "?"
        raise SchemaError(f"поле «{loc}»: {first.get('msg', 'невалідне')}",
                          raw=raw) from e

    return model.model_dump(mode="json")