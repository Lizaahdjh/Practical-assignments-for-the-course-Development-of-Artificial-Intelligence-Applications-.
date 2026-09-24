"""Контракт відповіді помічника: що саме модель має повернути і як це
перевіряється.

Окремий модуль навмисно. Схема — це домовленість між моделлю і рештою
застосунку, і вона не залежить від провайдера: змінивши модель або спосіб
виклику, схему ви не змінюєте. Веб-рівень і сторінка працюють лише з тим,
що пройшло перевірку тут.

Рішення, закладені в схему:

* схема описана pydantic-моделлю `AssistantReply` — з неї генерується
  і JSON Schema для `response_format`, і перевірка;
* тема звернення — суворо з переліку (`Topic`) — за нею звернення
  передається потрібному відділу;
* `reply`, `topic`, `grounded_in_rules`, `needs_clarification`,
  `escalate_to_human` — обовʼязкові;
* `order_number` — необовʼязковий (його може не бути в розмові), але
  якщо є — рівно 6 цифр (формат із `context.md`);
* код сам вирішує «передати оператору»: `escalate_to_human` — це
  підказка від моделі, а фінальне рішення ухвалює застосунок (див. `ask`);
* номер замовлення додатково перевіряється кодом: якщо модель його
  назвала, він має зустрічатися в тексті розмови.
"""

import json
import logging
import re
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, ValidationError, field_validator

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Перелік тем — фіксований, бо за ним звернення передається відділу
# ---------------------------------------------------------------------------
class Topic(str, Enum):
    ORDER = "order"               # замовлення, скасування, зміна складу
    DELIVERY = "delivery"         # доставка, терміни, вартість
    PAYMENT = "payment"           # оплата, способи, ліміти
    RETURN = "return"             # повернення товару
    WARRANTY = "warranty"         # гарантія, несправності
    SUPPORT = "support"           # графік роботи підтримки
    OTHER = "other"               # поза переліком


ORDER_NUMBER_RE = re.compile(r"^\d{6}$")


# ---------------------------------------------------------------------------
# Схема відповіді помічника
# ---------------------------------------------------------------------------
class AssistantReply(BaseModel):
    """Те, що решті застосунку потрібно від відповіді моделі."""

    reply: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="Текст відповіді для клієнта — те, що показує сторінка.",
    )
    topic: Topic = Field(
        ...,
        description="Тема звернення з фіксованого переліку — за нею "
                    "звернення передається потрібному відділу.",
    )
    grounded_in_rules: bool = Field(
        ...,
        description="True, якщо відповідь спирається на правила з context.md; "
                    "False — якщо правила про це мовчать.",
    )
    needs_clarification: bool = Field(
        ...,
        description="True, якщо звернення неоднозначне й потрібне уточнення.",
    )
    escalate_to_human: bool = Field(
        ...,
        description="True, якщо розмову варто передати оператору. "
                    "Це підказка моделі — остаточне рішення ухвалює застосунок.",
    )
    order_number: Optional[str] = Field(
        None,
        description="Номер замовлення, якщо клієнт його назвав. Рівно 6 цифр.",
    )

    @field_validator("order_number")
    @classmethod
    def _check_order_number(cls, v: Optional[str]) -> Optional[str]:
        if v is None or v == "":
            return None
        if not ORDER_NUMBER_RE.match(v):
            raise ValueError("номер замовлення має складатися з 6 цифр")
        return v


# ---------------------------------------------------------------------------
# JSON Schema для response_format
# ---------------------------------------------------------------------------
def output_schema() -> Dict[str, Any]:
    """Повернути JSON Schema відповіді помічника.

    Та сама схема передається моделі як опис очікуваного результату
    (через `response_format`) і використовується для перевірки того,
    що повернулося. У нас — генерується з pydantic-моделі.
    """
    return AssistantReply.model_json_schema()


# ---------------------------------------------------------------------------
# Помилка валідації — окремий клас, щоб `llm.ask` міг її розпізнати
# ---------------------------------------------------------------------------
class SchemaError(Exception):
    """Відповідь моделі не пройшла перевірку за схемою.

    Поле `detail` — коротке пояснення, яке можна передати моделі
    в повторному запиті або показати в журналі.
    """

    def __init__(self, detail: str, raw: str = ""):
        super().__init__(detail)
        self.detail = detail
        self.raw = raw


# ---------------------------------------------------------------------------
# Допоміжне: витягнути JSON з можливої «огорожі» моделі
# ---------------------------------------------------------------------------
def _extract_json(raw: str) -> str:
    """Повернути текст, який має бути JSON-обʼєктом.

    Модель іноді обгортає JSON у ```json ... ``` або додає текст
    довкола. Тут ми це прибираємо — але якщо JSON усе одно не
    розбереться, `validate` підійме `SchemaError`.
    """
    text = (raw or "").strip()
    if text.startswith("```"):
        # ```json\n{...}\n```
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
        text = text.strip()
    # Якщо є зовнішні фігурні дужки — лишити тільки їх
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        text = text[start:end + 1]
    return text


# ---------------------------------------------------------------------------
# Перевірка сирої відповіді
# ---------------------------------------------------------------------------
def validate(raw: str) -> Dict[str, Any]:
    """Перевірити сиру відповідь моделі й повернути дані, яким можна
    довіряти структурно.

    На вході — текст, який повернула модель. На виході — словник, що
    відповідає схемі. Якщо текст не є JSON або не проходить схему —
    підіймає `SchemaError` із поясненням, що саме не так.
    """
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
        model = AssistantReply.model_validate(data)
    except ValidationError as e:
        # Стислий, але інформативний опис — перша помилка + перелік полів
        first = e.errors()[0] if e.errors() else {"msg": "невідома помилка"}
        loc = ".".join(str(p) for p in first.get("loc", [])) or "?"
        msg = f"поле «{loc}»: {first.get('msg', 'невалідне')}"
        raise SchemaError(msg, raw=raw) from e

    return model.model_dump(mode="json")


# ---------------------------------------------------------------------------
# Додаткова перевірка кодом: номер замовлення справді з розмови
# ---------------------------------------------------------------------------
def verify_order_number(result: Dict[str, Any], history: List[Dict[str, str]],
                        message: str) -> Optional[str]:
    """Перевірити, чи названий номер замовлення зустрічається в розмові.

    Схема гарантує формат (6 цифр), але не гарантує, що модель його
    не вигадала. Повертає номер, якщо він є в історії або поточному
    повідомленні; інакше — None. У разі None `llm.ask` очищає поле,
    щоб вигаданий номер не потрапив до картки.
    """
    number = result.get("order_number")
    if not number:
        return None
    haystack_parts = [message or ""]
    for turn in history or []:
        haystack_parts.append(turn.get("content", "") or "")
    haystack = " ".join(haystack_parts)
    return number if number in haystack else None