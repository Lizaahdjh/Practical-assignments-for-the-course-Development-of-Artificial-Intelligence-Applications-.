"""Конвеєр обробки документа: від байтів файлу до рішення.

Рішення:

* `reject` — не рахунок (`document_type != "рахунок"`) або модель нічого
  не витягла (`document is None` після помилки);
* `auto` — тільки якщо **жодного** `error` немає і **всі критичні поля**
  присутні (`supplier.iban`, `supplier.code`, `total`, `buyer.name`);
* `review` — усе інше: є проблеми, є пропущені критичні поля;
* причини — людські: перелічити, що саме перевірити.
"""

import time
from dataclasses import dataclass, field

from . import images, llm, rules
from .rules import Issue


# Поля, без яких auto заборонено (навіть якщо issue не спрацював)
CRITICAL_PATHS = (
    "supplier.iban",
    "supplier.code",
    "buyer.name",
    "total",
)


@dataclass
class Result:
    decision: str
    reasons: list[str] = field(default_factory=list)
    document: dict | None = None
    issues: list[Issue] = field(default_factory=list)
    image: dict = field(default_factory=dict)
    model: str | None = None
    elapsed: dict = field(default_factory=dict)
    usage: dict | None = None


def _has_path(document: dict, path: str) -> bool:
    parts = path.split(".")
    current = document
    for p in parts:
        if not isinstance(current, dict):
            return False
        current = current.get(p)
    return current not in (None, "", [])


def decide(document: dict, issues: list[Issue]) -> tuple[str, list[str]]:
    """Вирішити долю документа за вилученими полями й знайденими проблемами."""
    if document is None:
        return "reject", ["Не вдалося вилучити поля з документа."]

    doc_type = document.get("document_type")
    if doc_type != "рахунок":
        return "reject", [
            f"Це не рахунок на оплату, а {doc_type!r}. "
            f"Документ не можна ставити в реєстр платежів."
        ]

    reasons: list[str] = []

    # 1. Будь-яка помилка → review
    errors = [i for i in issues if i.severity == "error"]
    if errors:
        reasons.append(
            f"Знайдено {len(errors)} проблем(и), які потребують перевірки людиною:"
        )
        for e in errors[:5]:
            reasons.append(f"  • {e.field}: {e.message}")
        if len(errors) > 5:
            reasons.append(f"  … та ще {len(errors) - 5}.")

    # 2. Пропущені критичні поля
    missing_critical = [p for p in CRITICAL_PATHS if not _has_path(document, p)]
    if missing_critical:
        reasons.append(
            "Відсутні критичні поля: " + ", ".join(missing_critical)
        )

    if reasons:
        return "review", reasons

    # 3. Усе чисто
    return "auto", ["Усі перевірки пройдено, критичні поля на місці."]


def process(content: bytes) -> Result:
    """Обробити файл: підготувати → вилучити → перевірити → вирішити."""
    elapsed: dict = {}

    # 1. Підготовка зображення
    t0 = time.perf_counter()
    prepared = images.prepare(content)
    elapsed["prepare"] = round(time.perf_counter() - t0, 3)

    image_info = {
        "original": prepared.original,
        "sent": prepared.sent,
    }

    # 2. Вилучення
    t1 = time.perf_counter()
    extraction = llm.extract(prepared)
    elapsed["extraction"] = round(time.perf_counter() - t1, 3)

    document = extraction["document"]

    # 3. Перевірки
    t2 = time.perf_counter()
    issues = rules.check(document)
    elapsed["checks"] = round(time.perf_counter() - t2, 4)

    # 4. Рішення
    decision, reasons = decide(document, issues)

    return Result(
        decision=decision,
        reasons=reasons,
        document=document,
        issues=issues,
        image=image_info,
        model=extraction["model"],
        elapsed=elapsed,
        usage=extraction["usage"],
    )