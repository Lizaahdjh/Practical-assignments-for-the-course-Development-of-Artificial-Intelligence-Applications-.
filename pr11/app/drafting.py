"""Що саме просимо в моделі: чернетка відповіді клієнтові.

Версія інструкції — `PROMPT_VERSION`; хеш — `_PROMPT_HASH`. Обидва
йдуть у `Request.metadata` і в ключ кешу.

У журнал із запиту потрапляють лише: довжина тексту, кількість
фактів, хеш тексту, версія prompt. Ні тексту клієнта, ні фактів із
іменами й номерами, ні повного prompt.
"""

import hashlib
import re

from .providers.base import Request

PROMPT_VERSION = "drafts-v1"

SYSTEM_PROMPT = """Ти — оператор підтримки інтернет-магазину «Сузірʼя».
Склади ввічливу відповідь клієнтові українською мовою.

ПРАВИЛА:
1. Спирайся ЛИШЕ на факти нижче. Не додавай нічого, чого в них немає.
2. Якщо фактів недостатньо — напиши, що потрібно уточнити.
3. Звертайся до клієнта на ім'я.
4. Довжина — 2–5 речень.
5. Не згадуй внутрішні примітки, повні номери карток, ID клієнта.
6. Не вигадуй сум, дат, строків, яких немає у фактах.
"""

_PROMPT_HASH = hashlib.sha256(SYSTEM_PROMPT.encode("utf-8")).hexdigest()[:8]

_FORBIDDEN = ["internal_note", "card_last4", "customer_id",
              "Клієнт двічі", "не називати"]


def build_request(sample: dict) -> Request:
    """Скласти запит до моделі зі зразка звернення."""
    facts_text = "\n".join(f"• {f}" for f in sample.get("facts", []))
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": (
            f"Клієнт: {sample.get('customer', '')}.\n"
            f"Текст звернення (цитата клієнта):\n«{sample.get('text', '')}»\n\n"
            f"Факти, які встановлено:\n{facts_text}\n\n"
            "Склади відповідь клієнтові."
        )},
    ]
    text = sample.get("text", "")
    metadata = {
        "request_id": sample.get("id"),
        "prompt_version": PROMPT_VERSION,
        "prompt_hash": _PROMPT_HASH,
        "request_length": len(text),
        "facts_count": len(sample.get("facts", [])),
        "text_hash": hashlib.sha256(text.encode("utf-8")).hexdigest()[:8],
    }
    return Request(
        messages=messages,
        max_tokens=600,
        temperature=0.3,
        model=None,
        metadata=metadata,
    )

_NEGATIVE_MARKERS = (
    "не поширюється", "не підлягає", "не стосується",
    "скасувати не можна", "не вийде", "не приймається",
    "не поширюються", "не поширюється",
)

def check(text: str, sample: dict) -> list[str]:
    """Перевірити чернетку, повернути список попереджень."""
    warnings: list[str] = []
    if not text or not text.strip():
        return ["Чернетка порожня."]
    if len(text) > 2000:
        warnings.append(f"Чернетка задовга ({len(text)} символів).")
    if len(text) < 40:
        warnings.append("Чернетка коротка — перевірте, чи все сказано.")
    low = text.lower()
    for bad in _FORBIDDEN:
        if bad.lower() in low:
            warnings.append(f"У чернетці згадано «{bad}» — перевірте.")
    for fact in sample.get("facts", []):
        fact_low = fact.lower()
        # Негативні факти: номер може бути лише для пояснення, чому НЕ буде дії
        if any(marker in fact_low for marker in _NEGATIVE_MARKERS):
            continue
        for num in re.findall(r"[A-Z]{0,3}-?\d{3,}", fact):
            if len(num) >= 5 and num not in text:
                warnings.append(f"У чернетці немає номера «{num}» із фактів.")
    seen, unique = set(), []
    for w in warnings:
        if w not in seen:
            unique.append(w)
            seen.add(w)
    return unique