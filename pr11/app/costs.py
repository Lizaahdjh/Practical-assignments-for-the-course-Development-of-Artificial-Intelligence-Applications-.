"""Вартість виклику: токени → гроші.

Рахуємо за `usage` від провайдера. Якщо `usage` немає — оцінка за
текстом, позначена як оцінка (`cost_estimated=True`), а не вигадане
точне число.

Поля `usage` (OpenAI-сумісний API):
    prompt_tokens, completion_tokens, total_tokens
    prompt_tokens_details.cached_tokens  — кешовані провайдером вхідні

Підставний провайдер за замовчуванням рахується тими самими цінами
(MOCK_BILLABLE=1), щоб бюджет і порівняння з кешем можна було
перевірити без ключа.
"""

import os

from dotenv import load_dotenv

load_dotenv()

# Ціни за мільйон токенів, у доларах.
PRICE_INPUT = float(os.getenv("LLM_PRICE_INPUT_PER_1M", "0.30"))
PRICE_OUTPUT = float(os.getenv("LLM_PRICE_OUTPUT_PER_1M", "2.50"))
_cached = os.getenv("LLM_PRICE_CACHED_INPUT_PER_1M", "")
PRICE_CACHED_INPUT = float(_cached) if _cached else PRICE_INPUT * 0.25
MOCK_BILLABLE = os.getenv("MOCK_BILLABLE", "1") == "1"

# Скільки символів на токен для оцінки без usage (укр. ~3.5, англ. ~4).
CHARS_PER_TOKEN = 3.5


def settings() -> dict:
    return {
        "input_per_1m": PRICE_INPUT,
        "output_per_1m": PRICE_OUTPUT,
        "cached_input_per_1m": PRICE_CACHED_INPUT,
        "mock_billable": MOCK_BILLABLE,
        "currency": "USD",
    }


def _usage_ints(usage: dict) -> tuple[int, int, int]:
    """Витягти (prompt, completion, cached) з usage; відсутні — 0."""
    prompt = int(usage.get("prompt_tokens") or 0)
    completion = int(usage.get("completion_tokens") or 0)
    cached = 0
    details = usage.get("prompt_tokens_details") or {}
    if isinstance(details, dict):
        cached = int(details.get("cached_tokens") or 0)
    cached = min(cached, prompt)  # не більше за весь вхід
    return prompt, completion, cached


def estimate(usage: dict | None, *, provider: str | None = None,
             model: str | None = None,
             text: str | None = None,
             prompt_text: str | None = None) -> tuple[float | None, bool]:
    """Оцінити вартість одного звертання.

    Повертає (cost_usd | None, is_estimate: bool).
    """
    # Підставний провайдер може бути безкоштовним — але за замовчуванням
    # рахуємо, щоб бюджет можна було перевірити.
    billable = True
    if provider == "mock" and not MOCK_BILLABLE:
        billable = False

    if usage:
        prompt, completion, cached = _usage_ints(usage)
        if provider == "mock" and not billable:
            return 0.0, False
        cost = (
            (prompt - cached) * PRICE_INPUT
            + cached * PRICE_CACHED_INPUT
            + completion * PRICE_OUTPUT
        ) / 1_000_000
        return round(cost, 6), False

    # usage немає — оцінка за текстом
    if text is None and prompt_text is None:
        return None, True

    est_in_tokens = round(len(prompt_text) / CHARS_PER_TOKEN) if prompt_text else 0
    est_out_tokens = round(len(text) / CHARS_PER_TOKEN) if text else 0
    cost = (
        est_in_tokens * PRICE_INPUT + est_out_tokens * PRICE_OUTPUT
    ) / 1_000_000
    if provider == "mock" and not billable:
        return 0.0, True
    return round(cost, 6), True