"""Що саме просимо в моделі: чернетка відповіді клієнтові.

Рішення:

* системна інструкція: від імені оператора, українською, коротко;
  спиратися ЛИШЕ на факти; фактів немає — прямо сказати;
* факти подаються списком, текст клієнта — окремо;
* `check` — перевірка кодом: не порожня, довжина в межах, ключові
  факти з `facts` присутні, заборонені підрядки (`internal_note`,
  `card_last4`) відсутні.
"""

from .providers.base import Request

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


def build_request(sample: dict) -> Request:
    """Скласти запит до моделі."""
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
    return Request(
        messages=messages,
        max_tokens=600,
        temperature=0.3,
        model=None,
        metadata={"request_id": sample.get("id")},
    )


_FORBIDDEN = ["internal_note", "card_last4", "customer_id",
              "Клієнт двічі", "не називати"]


def check(text: str, sample: dict) -> list[str]:
    """Перевірити чернетку. Повернути список попереджень."""
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
    # Ключові факти: номери заявок, накладних тощо
    facts = sample.get("facts", [])
    import re
    for fact in facts:
        for num in re.findall(r"[A-Z]{0,3}-?\d{3,}", fact):
            # Перевіряємо лише «важливі» номери (5+ цифр або з префіксом)
            if len(num) >= 5 and num not in text:
                warnings.append(f"У чернетці немає номера «{num}» із фактів.")
    # Дедуплікація
    seen = set()
    unique = []
    for w in warnings:
        if w not in seen:
            unique.append(w)
            seen.add(w)
    return unique