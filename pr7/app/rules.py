"""Програмні правила: перевірка змісту вилучених даних.

Рішення:

* усі суми — `Decimal`, ніколи `float`;
* IBAN — mod 97 за ISO 13616; український IBAN = `UA` + 27 символів
  (2 контрольні + 6 код банку + 19 номер рахунку);
* ЄДРПОУ (8 цифр) — контрольна сума за модулем 11 із різними вагами
  залежно від діапазону коду;
* РНОКПП (10 цифр) — контрольна сума за модулем 11;
* арифметика: quantity × price = amount у рядку; сума рядків = total_without_vat;
  vat = 20% від total_without_vat (або 0 для неплатника); total = total_without_vat + vat;
  допуск 0.01 грн;
* дати: date ≤ today; valid_until ≥ date (якщо є);
* покупець = `reference/company.json` — порівняння **після нормалізації**
  апострофів (’ ‘ ` ʼ → '), пробілів і регістру;
* постачальник відомий і IBAN збігається з довідником;
* обов'язкові поля: `document_type`, `number`, `date`, `supplier.name`,
  `supplier.code`, `supplier.iban`, `buyer.name`, `buyer.code`,
  `items` (непорожній), `total`;
* severity: усе — `error`; окремо `warning` не використовуємо (простіше
  для рішення; у проді розділили б).
"""

import json
from dataclasses import dataclass
from datetime import date
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Optional

REFERENCE_DIR = Path(__file__).resolve().parent.parent / "reference"

TOLERANCE = Decimal("0.01")


@dataclass
class Issue:
    field: str
    rule: str
    message: str
    severity: str = "error"


def load_reference() -> tuple[dict, list[dict]]:
    company = json.loads((REFERENCE_DIR / "company.json").read_text(encoding="utf-8"))
    suppliers = json.loads((REFERENCE_DIR / "suppliers.json").read_text(encoding="utf-8"))
    return company, suppliers


# ---------------------------------------------------------------------------
# Нормалізація тексту
# ---------------------------------------------------------------------------
def _normalize_name(s: Optional[str]) -> str:
    """Привести апострофи, пробіли й регістр до одного вигляду.

    Потрібно для порівняння назв: у документах трапляються різні символи
    апострофа (U+2019 ’ , U+2018 ‘ , U+0060 ` , U+02BC ʼ), а також різні
    пробіли. Без нормалізації правильний документ дає хибну помилку
    «Рахунок виставлено не «Сузірʼя Рітейл»».
    """
    if not s:
        return ""
    # Усі варіанти апострофа → стандартний ASCII '
    for ch in ("\u2019", "\u2018", "\u0060", "\u02BC", "\u2032"):
        s = s.replace(ch, "'")
    # Кілька пробілів → один; обрізаємо
    s = " ".join(s.split())
    return s.strip().lower()


# ---------------------------------------------------------------------------
# IBAN
# ---------------------------------------------------------------------------
def _iban_mod97(iban: str) -> int:
    """Повернути залишок mod 97 для IBAN (ISO 13616)."""
    rearranged = iban[4:] + iban[:4]
    digits = []
    for ch in rearranged:
        if ch.isdigit():
            digits.append(ch)
        else:
            digits.append(str(ord(ch) - ord("A") + 10))
    return int("".join(digits)) % 97


def valid_iban(iban: Optional[str]) -> bool:
    if not iban:
        return False
    iban = iban.replace(" ", "").upper()
    if len(iban) != 29 or not iban.startswith("UA"):
        return False
    try:
        return _iban_mod97(iban) == 1
    except (ValueError, TypeError):
        return False


# ---------------------------------------------------------------------------
# ЄДРПОУ (8 цифр, модуль 11, ваги за діапазоном) і РНОКПП (10 цифр)
# ---------------------------------------------------------------------------
def valid_edrpou(code: Optional[str]) -> bool:
    """Перевірити 8-значний код ЄДРПОУ за контрольною сумою."""
    if not code or not code.isdigit() or len(code) != 8:
        return False
    digits = [int(d) for d in code]
    if digits[0] <= 3:
        weights = [1, 2, 3, 4, 5, 6, 7]
    elif digits[0] <= 6:
        weights = [7, 1, 2, 3, 4, 5, 6]
    else:
        weights = [3, 1, 2, 3, 4, 5, 6]
    total = sum(d * w for d, w in zip(digits[:7], weights))
    remainder = total % 11
    control = remainder if remainder < 10 else 0
    return control == digits[7]


def valid_rnokpp(code: Optional[str]) -> bool:
    """Перевірити 10-значний РНОКПП за контрольною сумою."""
    if not code or not code.isdigit() or len(code) != 10:
        return False
    digits = [int(d) for d in code]
    weights = [-1, 5, 7, 9, 4, 6, 10, 5, 7]
    total = sum(d * w for d, w in zip(digits[:9], weights))
    control = (total % 11) % 10
    return control == digits[9]


def valid_party_code(code: Optional[str]) -> bool:
    """ЄДРПОУ (8) або РНОКПП (10)."""
    if not code:
        return False
    code = code.strip()
    if len(code) == 8:
        return valid_edrpou(code)
    if len(code) == 10:
        return valid_rnokpp(code)
    return False


# ---------------------------------------------------------------------------
# Гроші
# ---------------------------------------------------------------------------
def _dec(value) -> Optional[Decimal]:
    if value is None or value == "":
        return None
    try:
        return Decimal(str(value).replace(" ", "").replace(",", "."))
    except (InvalidOperation, ValueError):
        return None


# ---------------------------------------------------------------------------
# Основна перевірка
# ---------------------------------------------------------------------------
def check(document: dict) -> list[Issue]:
    """Застосувати правила до вилученого документа."""
    issues: list[Issue] = []
    company, suppliers = load_reference()

    # --- Тип документа ---
    doc_type = document.get("document_type")
    if doc_type != "рахунок":
        issues.append(Issue(
            field="document_type",
            rule="document_type",
            message=f"Це не рахунок на оплату (тип: {doc_type!r}).",
        ))

    # --- Обов'язкові поля ---
    if not document.get("number"):
        issues.append(Issue("number", "missing_required", "Номер рахунку відсутній."))
    if not document.get("date"):
        issues.append(Issue("date", "missing_required", "Дата рахунку відсутня."))
    if not document.get("total"):
        issues.append(Issue("total", "missing_required", "Сума до сплати відсутня."))

    supplier = document.get("supplier") or {}
    buyer = document.get("buyer") or {}
    items = document.get("items") or []

    if not supplier.get("name"):
        issues.append(Issue("supplier.name", "missing_required", "Назва постачальника відсутня."))
    if not supplier.get("code"):
        issues.append(Issue("supplier.code", "missing_required", "Код постачальника відсутній."))
    if not supplier.get("iban"):
        issues.append(Issue("supplier.iban", "missing_required", "IBAN постачальника відсутній."))
    if not buyer.get("name"):
        issues.append(Issue("buyer.name", "missing_required", "Назва покупця відсутня."))
    if not buyer.get("code"):
        issues.append(Issue("buyer.code", "missing_required", "Код покупця відсутній."))
    if not items:
        issues.append(Issue("items", "missing_required", "Позиції відсутні."))

    # --- IBAN ---
    iban = (supplier.get("iban") or "").replace(" ", "").upper()
    if iban:
        if not valid_iban(iban):
            issues.append(Issue(
                "supplier.iban", "iban_checksum",
                f"IBAN не проходить контрольну суму: {iban}",
            ))
        else:
            known_ibans = {s["iban"].replace(" ", "").upper() for s in suppliers}
            if iban not in known_ibans:
                issues.append(Issue(
                    "supplier.iban", "iban_registry",
                    f"IBAN {iban} не збігається з жодним постачальником із довідника.",
                ))

    # --- Коди сторін ---
    if supplier.get("code") and not valid_party_code(supplier["code"]):
        issues.append(Issue(
            "supplier.code", "code_checksum",
            f"Код постачальника {supplier['code']} не проходить контрольну суму.",
        ))
    if buyer.get("code") and not valid_party_code(buyer["code"]):
        issues.append(Issue(
            "buyer.code", "code_checksum",
            f"Код покупця {buyer['code']} не проходить контрольну суму.",
        ))

    # --- Покупець — ми (з нормалізацією апострофів) ---
    if buyer.get("name") and _normalize_name(buyer["name"]) != _normalize_name(company["name"]):
        issues.append(Issue(
            "buyer.name", "buyer",
            f"Рахунок виставлено не «Сузірʼя Рітейл»: {buyer['name']!r}",
        ))
    if buyer.get("code") and buyer["code"] != company["code"]:
        issues.append(Issue(
            "buyer.code", "buyer",
            f"Код покупця {buyer['code']} не збігається з нашим {company['code']}.",
        ))

    # --- Дати ---
    today = date.today()
    inv_date = document.get("date")
    valid_until = document.get("valid_until")
    if inv_date:
        try:
            d = date.fromisoformat(inv_date)
            if d > today:
                issues.append(Issue(
                    "date", "date_future", f"Дата рахунку {inv_date} у майбутньому.",
                ))
        except ValueError:
            issues.append(Issue("date", "date_format",
                                f"Дата {inv_date!r} не у форматі РРРР-ММ-ДД."))
    if inv_date and valid_until:
        try:
            d1 = date.fromisoformat(inv_date)
            d2 = date.fromisoformat(valid_until)
            if d2 < d1:
                issues.append(Issue(
                    "valid_until", "date_order",
                    f"Строк дії {valid_until} раніше за дату рахунку {inv_date}.",
                ))
        except ValueError:
            pass

    # --- Арифметика позицій ---
    line_sum = Decimal("0")
    for i, item in enumerate(items):
        q = _dec(item.get("quantity"))
        p = _dec(item.get("price"))
        a = _dec(item.get("amount"))
        if q is None or p is None or a is None:
            continue
        expected = (q * p).quantize(TOLERANCE)
        if abs(expected - a) > TOLERANCE:
            issues.append(Issue(
                f"items.{i}.amount", "arithmetic_line",
                f"Рядок {i + 1}: {q} × {p} = {expected}, а надруковано {a}.",
            ))
        line_sum += a

    # --- Підсумки ---
    total_wo_vat = _dec(document.get("total_without_vat"))
    vat = _dec(document.get("vat"))
    total = _dec(document.get("total"))

    if total_wo_vat is not None and line_sum and abs(line_sum - total_wo_vat) > TOLERANCE:
        issues.append(Issue(
            "total_without_vat", "arithmetic_sum",
            f"Сума рядків {line_sum}, а «разом без ПДВ» — {total_wo_vat}.",
        ))

    # ПДВ: для неплатника (vat=0 і total == total_without_vat) перевірку
    # пропускаємо — це ФОП на єдиному податку, а не помилка.
    if total_wo_vat is not None and vat is not None and total is not None:
        is_non_payer = vat == 0 and total == total_wo_vat

        if not is_non_payer:
            expected_vat = (total_wo_vat * Decimal("0.20")).quantize(TOLERANCE)
            if abs(expected_vat - vat) > TOLERANCE:
                issues.append(Issue(
                    "vat", "arithmetic_vat",
                    f"ПДВ має бути 20 % від {total_wo_vat}: {expected_vat}, "
                    f"а надруковано {vat}.",
                ))

    if total_wo_vat is not None and vat is not None and total is not None:
        expected_total = (total_wo_vat + vat).quantize(TOLERANCE)
        if abs(expected_total - total) > TOLERANCE:
            issues.append(Issue(
                "total", "arithmetic_total",
                f"{total_wo_vat} + {vat} = {expected_total}, а «до сплати» — {total}.",
            ))

    return issues