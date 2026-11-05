"""Модель-суддя: оцінка вільного тексту за рубрикою з бінарних критеріїв.

Готове: `RUBRIC`, контракт `Verdict`, `verdict_schema`, `parse_verdict`.
Реалізовано: `build_messages`, `evaluate` (з fallback на `json_object`,
якщо провайдер не приймає `json_schema`).
"""

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

JUDGE_BASE_URL = os.getenv("JUDGE_BASE_URL") or os.getenv("LLM_BASE_URL", "")
JUDGE_API_KEY = os.getenv("JUDGE_API_KEY") or os.getenv("LLM_API_KEY", "")
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "")
JUDGE_TIMEOUT = float(os.getenv("JUDGE_TIMEOUT", "30"))


RUBRIC = {
    "tone": "Текст ввічливий і діловий: звертання до клієнта, без повчань, без надмірних "
            "вибачень, без канцеляриту; його можна надіслати клієнтові без правок тону.",
    "next_step": "Клієнтові зрозуміло, що буде далі або що зробити йому: куди принести, "
                 "скільки чекати, що надійде окремо.",
    "grounded": "Кожне твердження про замовлення, строки, суми, адреси й умови є серед наданих "
                "фактів; нічого не додано від себе й нічого з фактів не перекручено.",
    "addresses_question": "Відповідь стосується саме того, про що запитав клієнт, а не "
                          "сусіднього питання.",
    "no_extra_promises": "Немає обіцянок, винятків, компенсацій, знижок і строків, яких немає у "
                         "фактах.",
    "refuses_properly": "Якщо це відмова — названо причину з фактів, без вигаданих альтернатив, "
                        "без звинувачень і без «як виняток».",
    "stays_on_topic": "Текст лишається в межах підтримки магазину і не виконує стороннє "
                      "прохання клієнта (вірш, переклад, порада поза темою).",
    "asks_clarification": "Коли фактів бракує або вони не стосуються питання, текст чесно каже, "
                          "що оператор уточнить, і не вигадує подробиць.",
}


@dataclass
class CriterionResult:
    id: str
    passed: bool
    reason: str = ""


@dataclass
class Verdict:
    criteria: list[CriterionResult] = field(default_factory=list)
    overall: bool = False
    model: str | None = None
    usage: dict | None = None
    raw: str | None = None


def settings() -> dict:
    return {
        "base_url": JUDGE_BASE_URL,
        "model": JUDGE_MODEL,
        "configured": bool(JUDGE_BASE_URL and JUDGE_API_KEY and JUDGE_MODEL),
    }


def verdict_schema() -> dict:
    return {
        "type": "object",
        "properties": {
            "criteria": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string"},
                        "passed": {"type": "boolean"},
                        "reason": {"type": "string"},
                    },
                    "required": ["id", "passed", "reason"],
                },
            },
            "overall": {"type": "boolean"},
        },
        "required": ["criteria", "overall"],
    }


def parse_verdict(raw: str, expected_ids: list[str]) -> Verdict:
    try:
        data = json.loads(raw)
    except ValueError as exc:
        raise ValueError(f"відповідь судді — не JSON: {exc}") from exc
    if not isinstance(data, dict) or "criteria" not in data or "overall" not in data:
        raise ValueError("у відповіді судді немає полів criteria і overall")
    if not isinstance(data["overall"], bool):
        raise ValueError("overall має бути true або false")
    criteria: list[CriterionResult] = []
    for item in data["criteria"]:
        if not isinstance(item, dict) or not isinstance(item.get("id"), str) \
                or not isinstance(item.get("passed"), bool):
            raise ValueError(f"критерій не за схемою: {item!r}")
        criteria.append(CriterionResult(item["id"], item["passed"], str(item.get("reason", ""))))
    got = {c.id for c in criteria}
    missing = [cid for cid in expected_ids if cid not in got]
    if missing:
        raise ValueError("суддя не оцінив критерії: " + ", ".join(missing))
    unknown = sorted(got - set(expected_ids))
    if unknown:
        raise ValueError("суддя оцінив критерії, яких не просили: " + ", ".join(unknown))
    return Verdict(criteria=criteria, overall=data["overall"], raw=raw)


def build_messages(case: dict, output: dict, criteria: list[str]) -> list[dict]:
    rubric_lines = []
    for cid in criteria:
        rubric_lines.append(f"- {cid}: {RUBRIC.get(cid, '(критерій не описано)')}")
    system = (
        "Ти оцінюєш відповідь служби підтримки інтернет-магазину «Сузірʼя» "
        "за бінарними критеріями. Для КОЖНОГО критерію поверни passed (true/false) "
        "і коротку причину. Не додавай критеріїв, яких не просили. Не оцінюй довжину "
        "як перевагу; оцінюй зміст. Спирайся ЛИШЕ на надані факти.\n\n"
        "Критерії:\n" + "\n".join(rubric_lines) + "\n\n"
        "Поверни JSON за схемою: "
        '{"criteria": [{"id": "...", "passed": true|false, "reason": "..."}], "overall": true|false}'
    )
    inp = case.get("input") or {}
    facts = inp.get("facts") or []
    facts_text = "\n".join(f"- {f}" for f in facts) if facts else "(фактів немає)"
    user = (
        f"Звернення клієнта:\n{inp.get('text', '')}\n\n"
        f"Факти, які встановлено:\n{facts_text}\n\n"
        f"Відповідь системи:\n{output.get('text', '')}"
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def evaluate(case: dict, output: dict, criteria: list[str] | None = None) -> Verdict:
    """Оцінити відповідь суддею за критеріями (типово — вся рубрика)."""
    if not (JUDGE_BASE_URL and JUDGE_API_KEY and JUDGE_MODEL):
        raise RuntimeError(
            "суддя не налаштований: задайте JUDGE_BASE_URL, JUDGE_API_KEY, JUDGE_MODEL у .env"
        )
    criteria = list(criteria or list(RUBRIC))
    messages = build_messages(case, output, criteria)

    import openai
    client = openai.OpenAI(base_url=JUDGE_BASE_URL, api_key=JUDGE_API_KEY,
                           timeout=JUDGE_TIMEOUT, max_retries=0)

    schema_format = {"type": "json_schema",
                     "json_schema": {"name": "verdict", "schema": verdict_schema()}}

    def call(response_format):
        completion = client.chat.completions.create(
            model=JUDGE_MODEL, messages=messages, temperature=0,
            max_tokens=800, response_format=response_format,
        )
        return completion.choices[0].message.content or "", completion

    last_error: str | None = None
    for attempt in (1, 2):
        try:
            try:
                raw, completion = call(schema_format)
            except openai.BadRequestError:
                raw, completion = call({"type": "json_object"})
        except Exception as exc:  # noqa: BLE001
            last_error = f"збій виклику судді ({type(exc).__name__}): {exc}"
            continue
        try:
            verdict = parse_verdict(raw, criteria)
            verdict.model = JUDGE_MODEL
            usage = getattr(completion, "usage", None)
            if usage is not None:
                verdict.usage = {
                    "prompt_tokens": getattr(usage, "prompt_tokens", None),
                    "completion_tokens": getattr(usage, "completion_tokens", None),
                }
            return verdict
        except ValueError as exc:
            last_error = f"відповідь судді не за схемою (спроба {attempt}): {exc}"
    raise RuntimeError(last_error or "суддя не відповів")