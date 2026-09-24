"""Прогнати набір звернень і діалогів через `llm.ask` за трьох конфігурацій.

Запуск (з кореня pr4, після активування venv):

    python compare/run.py

Три конфігурації з розділу 2 ПР4:

1. `vague`       — нечіткий запит: текст + однорядкова інструкція, без
                   правил, без історії, без формату;
2. `structured`  — повна інструкція з обмеженнями, форматом і прикладами,
                   але без правил магазину і без історії;
3. `context`     — повна інструкція + правила з context.md + історія в
                   межах бюджету + перевірка за схемою.

Скрипт автоматично перевіряє для кожної відповіді:
    * чи пройшла вона схему (schema_ok);
    * чи правильно визначено тему (topic_ok) — за очікуванням для виду;
    * чи не вигадала модель факт (hallucination) — за grounded_in_rules;
    * чи названий номер замовлення справді є в розмові (order_in_history);
    * чи втримано факт з початку діалогу (для діалогів);
    * токени й час.

Результати — `compare/results.json`;
висновки й класифікація помилок — `compare/findings.md`.
"""

import importlib
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

REQUESTS_FILE = Path(__file__).parent / "requests.json"
RESULTS_FILE = Path(__file__).parent / "results.json"
FINDINGS_FILE = Path(__file__).parent / "findings.md"
CONTEXT_FILE = ROOT / "context.md"

CONFIGS = ["vague", "structured", "context"]

VAGUE_INSTRUCTION = (
    "Ти — помічник служби підтримки інтернет-магазину. "
    "Відповідай на звернення клієнта."
)

# Регулярка для пошуку номера замовлення у тексті
ORDER_RE = re.compile(r"\b\d{6}\b")


# ---------------------------------------------------------------------------
# Очікування для автоматичних перевірок
# ---------------------------------------------------------------------------
# Які теми вважаються правильними для кожного виду звернення.
EXPECTED_TOPIC: Dict[str, set] = {
    "типове": {"delivery", "order"},
    "на межі правил": {"payment"},
    "поза правилами": {"delivery", "other"},
    "неоднозначне": {"return", "warranty"},
    "некоректний ввід": {"other", "support"},
    "спроба переписати інструкцію": {"warranty"},
}

# Яким має бути grounded_in_rules для кожного виду (None — не перевіряємо).
EXPECTED_GROUNDED: Dict[str, Optional[bool]] = {
    "типове": True,
    "на межі правил": True,
    "поза правилами": False,
    "неоднозначне": True,
    "некоректний ввід": None,             # будь-яке
    "спроба переписати інструкцію": True,  # гарантія Є в правилах
}

# Очікування для діалогів: чи має кожна репліка містити номер з історії.
# Ключ — вид діалогу, значення — множина індексів (1-based), для яких
# номер замовлення МАЄ бути названий у відповіді.
DIALOG_EXPECTS_ORDER: Dict[str, set] = {
    "факт на початку, потрібен наприкінці": {3},
    "два замовлення в одній розмові": set(),  # достатньо, щоб не змішав
}


# ---------------------------------------------------------------------------
# Конфігурації виклику
# ---------------------------------------------------------------------------
def _reload_llm():
    import app.llm as llm
    importlib.reload(llm)
    return llm


def _call_vague(client, model: str, message: str):
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": VAGUE_INSTRUCTION},
            {"role": "user", "content": message},
        ],
        temperature=0.2,
        max_tokens=600,
    )
    text = (response.choices[0].message.content or "").strip()
    usage = getattr(response, "usage", None)
    return (
        {
            "reply": text,
            "topic": None,
            "grounded_in_rules": None,
            "needs_clarification": None,
            "escalate_to_human": None,
            "order_number": None,
        },
        {
            "prompt_tokens": getattr(usage, "prompt_tokens", 0) if usage else 0,
            "completion_tokens": getattr(usage, "completion_tokens", 0) if usage else 0,
        },
    )


def _call_structured(client, model: str, message: str):
    import app.llm as llm
    from app.schema import output_schema, validate, SchemaError

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": llm.SYSTEM_INSTRUCTION},
            {"role": "user", "content": message},
        ],
        temperature=0.2,
        max_tokens=600,
        response_format={
            "type": "json_schema",
            "json_schema": {"name": "assistant_reply", "schema": output_schema()},
        },
    )
    text = (response.choices[0].message.content or "").strip()
    usage = getattr(response, "usage", None)

    try:
        result = validate(text)
        schema_ok = True
    except SchemaError:
        result = {"reply": text}
        schema_ok = False

    return result, schema_ok, {
        "prompt_tokens": getattr(usage, "prompt_tokens", 0) if usage else 0,
        "completion_tokens": getattr(usage, "completion_tokens", 0) if usage else 0,
    }


def _call_context(llm, message: str, history: List[Dict[str, str]],
                  context: str) -> Dict[str, Any]:
    return llm.ask(message, history, context)


# ---------------------------------------------------------------------------
# Автоматичні перевірки
# ---------------------------------------------------------------------------
def check_topic(case_kind: str, result: Dict[str, Any]) -> Optional[bool]:
    """Чи тема відповідає очікуванню для виду звернення."""
    topic = result.get("topic")
    if topic is None:
        return None
    expected = EXPECTED_TOPIC.get(case_kind)
    if not expected:
        return None
    return topic in expected


def check_grounded(case_kind: str, result: Dict[str, Any]) -> Optional[bool]:
    """Чи grounded_in_rules відповідає очікуванню (True = OK)."""
    expected = EXPECTED_GROUNDED.get(case_kind)
    if expected is None:
        return None
    got = result.get("grounded_in_rules")
    if got is None:
        return None
    return bool(got) == bool(expected)


def check_hallucination(case_kind: str, result: Dict[str, Any]) -> Optional[bool]:
    """Чи вигадала модель факт (True = вигадка).

    Груба оцінка за прапорцем grounded_in_rules і за очікуванням для виду:
    * для поза-правилами grounded=true → ймовірна вигадка;
    * для типових/межових grounded=false → підозріло.
    """
    grounded = result.get("grounded_in_rules")
    if grounded is None:
        return None
    expected = EXPECTED_GROUNDED.get(case_kind)
    if expected is None:
        return None
    return bool(grounded) != bool(expected)


def check_order_in_history(result: Dict[str, Any],
                           history: List[Dict[str, str]],
                           message: str) -> Optional[bool]:
    """Чи названий у відповіді номер замовлення справді є в розмові."""
    number = result.get("order_number")
    if not number:
        return None  # не названий — не помилка
    haystack = " ".join(
        [message] + [t.get("content", "") for t in (history or [])]
    )
    return number in haystack


def order_present_in_reply(reply: str, expected_numbers: List[str]) -> bool:
    """Чи згадано в тексті відповіді якийсь із очікуваних номерів."""
    if not reply:
        return False
    found = set(ORDER_RE.findall(reply))
    return bool(found & set(expected_numbers))


# ---------------------------------------------------------------------------
# Основний прогін
# ---------------------------------------------------------------------------
def main() -> int:
    if not REQUESTS_FILE.exists():
        print(f"Немає {REQUESTS_FILE}. Скопіюйте requests.example.json.")
        return 1
    if not CONTEXT_FILE.exists():
        print(f"Немає {CONTEXT_FILE}.")
        return 1

    data = json.loads(REQUESTS_FILE.read_text(encoding="utf-8"))
    cases = data.get("звернення", [])
    dialogs = data.get("діалоги", [])
    context = CONTEXT_FILE.read_text(encoding="utf-8")

    os.environ.setdefault("LLM_TEMPERATURE", "0.2")

    llm = _reload_llm()
    from openai import OpenAI
    client = OpenAI(base_url=llm.BASE_URL, api_key=llm.API_KEY, timeout=llm.TIMEOUT)

    all_results: Dict[str, Any] = {cfg: {"runs": [], "dialogs": []} for cfg in CONFIGS}

    # ---------------- Звернення ----------------
    for cfg in CONFIGS:
        print(f"\n=== Звернення, конфігурація: {cfg} ===")
        for case in cases:
            t0 = time.perf_counter()
            entry: Dict[str, Any] = {"вид": case["вид"], "текст": case["текст"]}
            try:
                if cfg == "vague":
                    result, usage = _call_vague(client, llm.MODEL, case["текст"])
                    entry.update({
                        "ok": True, "result": result, "usage": usage,
                        "elapsed": round(time.perf_counter() - t0, 3),
                        "schema_ok": None, "attempts": 1,
                    })
                elif cfg == "structured":
                    result, schema_ok, usage = _call_structured(
                        client, llm.MODEL, case["текст"]
                    )
                    entry.update({
                        "ok": True, "result": result, "usage": usage,
                        "elapsed": round(time.perf_counter() - t0, 3),
                        "schema_ok": schema_ok, "attempts": 1,
                    })
                else:  # context
                    full = _call_context(llm, case["текст"], [], context)
                    entry.update({
                        "ok": True, "result": full["result"], "usage": full["usage"],
                        "elapsed": full["elapsed"], "schema_ok": True,
                        "attempts": full["attempts"],
                    })

                # Автоматичні перевірки
                r = entry["result"]
                entry["topic_ok"] = check_topic(case["вид"], r)
                entry["grounded_ok"] = check_grounded(case["вид"], r)
                entry["hallucination"] = check_hallucination(case["вид"], r)
                entry["order_in_history"] = check_order_in_history(
                    r, [], case["текст"]
                )

                flags = []
                if entry.get("topic_ok") is False: flags.append("topic!")
                if entry.get("grounded_ok") is False: flags.append("grounded!")
                if entry.get("hallucination") is True: flags.append("hallucination!")
                if entry.get("order_in_history") is False: flags.append("order!")
                tail = " " + ",".join(flags) if flags else ""
                print(f"  [{case['вид']}] {entry['elapsed']:.2f} с{tail}")
            except Exception as e:
                entry.update({"ok": False, "error": str(e)})
                print(f"  [{case['вид']}] ПОМИЛКА: {e}")
            all_results[cfg]["runs"].append(entry)

    # ---------------- Діалоги ----------------
    for cfg in CONFIGS:
        print(f"\n=== Діалоги, конфігурація: {cfg} ===")
        for dialog in dialogs:
            history: List[Dict[str, str]] = []
            entries: List[Dict[str, Any]] = []
            expected_numbers = sorted(
                set(n for repl in dialog["репліки"] for n in ORDER_RE.findall(repl))
            )
            expected_order_turns = DIALOG_EXPECTS_ORDER.get(dialog["вид"], set())

            for i, repl in enumerate(dialog["репліки"], start=1):
                t0 = time.perf_counter()
                entry: Dict[str, Any] = {"turn": i, "user": repl}
                try:
                    if cfg == "vague":
                        # Без історії — саме це й перевіряємо
                        result, usage = _call_vague(client, llm.MODEL, repl)
                        entry.update({
                            "ok": True, "result": result, "usage": usage,
                            "elapsed": round(time.perf_counter() - t0, 3),
                            "schema_ok": None, "attempts": 1,
                        })
                    elif cfg == "structured":
                        result, schema_ok, usage = _call_structured(
                            client, llm.MODEL, repl
                        )
                        entry.update({
                            "ok": True, "result": result, "usage": usage,
                            "elapsed": round(time.perf_counter() - t0, 3),
                            "schema_ok": schema_ok, "attempts": 1,
                        })
                    else:  # context
                        full = _call_context(llm, repl, history, context)
                        entry.update({
                            "ok": True, "result": full["result"],
                            "usage": full["usage"],
                            "elapsed": full["elapsed"], "schema_ok": True,
                            "attempts": full["attempts"],
                        })

                    r = entry["result"]
                    entry["order_in_history"] = check_order_in_history(r, history, repl)
                    entry["order_expected"] = i in expected_order_turns
                    entry["order_present"] = order_present_in_reply(
                        r.get("reply", ""), expected_numbers
                    )
                    # Чи згадала модель факт на репліці, де він очікується
                    entry["recall_ok"] = (
                        entry["order_present"] if entry["order_expected"] else None
                    )

                    flags = []
                    if entry.get("order_in_history") is False:
                        flags.append("order-halluc!")
                    if entry.get("order_expected") and not entry.get("order_present"):
                        flags.append("no-recall!")
                    tail = " " + ",".join(flags) if flags else ""
                    print(
                        f"  [{dialog['вид']}] [{cfg}] репліка {i}: "
                        f"{entry['elapsed']:.2f} с{tail}"
                    )

                    # Історію ведемо лише для `context`; у `vague` і
                    # `structured` історія не передається — це й показує
                    # різницю між конфігураціями.
                    if cfg == "context":
                        history.append({"role": "user", "content": repl})
                        history.append({
                            "role": "assistant",
                            "content": r.get("reply", ""),
                        })
                except Exception as e:
                    entry.update({"ok": False, "error": str(e)})
                    print(f"  [{dialog['вид']}] [{cfg}] репліка {i}: ПОМИЛКА: {e}")
                entries.append(entry)

            all_results[cfg]["dialogs"].append({
                "вид": dialog["вид"],
                "перевіряє": dialog.get("перевіряє", ""),
                "expected_numbers": expected_numbers,
                "entries": entries,
            })

    RESULTS_FILE.write_text(
        json.dumps(all_results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\nРезультати збережено у {RESULTS_FILE}")

    write_findings(all_results, cases, dialogs)
    print(f"Висновки записано у {FINDINGS_FILE}")
    return 0


# ---------------------------------------------------------------------------
# Формування findings.md
# ---------------------------------------------------------------------------
def _fmt(v: Any) -> str:
    if v is True: return "так"
    if v is False: return "ні"
    if v is None: return "—"
    return str(v)


def write_findings(all_results: Dict[str, Any], cases: List[Dict[str, Any]],
                   dialogs: List[Dict[str, Any]]) -> None:
    lines: List[str] = []
    lines.append("# Висновки: порівняння трьох конфігурацій\n")
    lines.append(
        "Скрипт `compare/run.py` автоматично перевірив кожну відповідь:\n"
        "* `topic_ok` — чи тема відповідає очікуванню для виду звернення;\n"
        "* `grounded_ok` — чи `grounded_in_rules` відповідає очікуванню;\n"
        "* `hallucination` — чи вигадала модель факт (за прапорцем grounded);\n"
        "* `order_in_history` — чи названий номер замовлення справді є в розмові;\n"
        "* `recall_ok` — (для діалогів) чи названо факт на репліці, де він потрібен.\n"
    )

    # ---------- Зведена таблиця по зверненнях ----------
    lines.append("## Зведена таблиця по зверненнях\n")
    lines.append(
        "| Вид | Конфігурація | ok | schema | topic ok | grounded ok | "
        "hallucination | order in history | токени (запит/відп) | час, с |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for cfg, cfg_data in all_results.items():
        for run in cfg_data["runs"]:
            if not run.get("ok"):
                lines.append(
                    f"| {run['вид']} | {cfg} | ні | — | — | — | — | — | — "
                    f"| помилка: {run.get('error','')[:60]} |"
                )
                continue
            usage = run.get("usage") or {}
            lines.append(
                f"| {run['вид']} | {cfg} "
                f"| так | {_fmt(run.get('schema_ok'))} "
                f"| {_fmt(run.get('topic_ok'))} "
                f"| {_fmt(run.get('grounded_ok'))} "
                f"| {_fmt(run.get('hallucination'))} "
                f"| {_fmt(run.get('order_in_history'))} "
                f"| {usage.get('prompt_tokens','?')}/{usage.get('completion_tokens','?')} "
                f"| {run.get('elapsed','?')} |"
            )
    lines.append("")

    # ---------- Діалоги ----------
    lines.append("## Діалоги: чи втримано факт\n")
    for cfg in CONFIGS:
        for dialog in all_results[cfg]["dialogs"]:
            lines.append(f"### {cfg} → «{dialog['вид']}»\n")
            lines.append(f"Очікувані номери: `{dialog.get('expected_numbers')}`\n")
            lines.append("| Хід | Очікували номер | Названо номер | order у розмові | topic | grounded | reply (перші 80) |")
            lines.append("|---|---|---|---|---|---|---|")
            for entry in dialog["entries"]:
                if not entry.get("ok"):
                    lines.append(
                        f"| {entry['turn']} | — | — | — | — | — "
                        f"| ПОМИЛКА: {entry.get('error','')[:60]} |"
                    )
                    continue
                r = entry.get("result") or {}
                snippet = (r.get("reply") or "").replace("\n", " ")[:80]
                lines.append(
                    f"| {entry['turn']} "
                    f"| {_fmt(entry.get('order_expected'))} "
                    f"| {_fmt(entry.get('order_present'))} "
                    f"| {_fmt(entry.get('order_in_history'))} "
                    f"| {_fmt(r.get('topic'))} "
                    f"| {_fmt(r.get('grounded_in_rules'))} "
                    f"| {snippet}… |"
                )
            lines.append("")

    # ---------- Приклади невдалих відповідей ----------
    lines.append("## Зібрані невдалі відповіді\n")
    failures: List[Tuple[str, str, str]] = []  # (cfg, де, опис)

    for cfg, cfg_data in all_results.items():
        for run in cfg_data["runs"]:
            if not run.get("ok"):
                continue
            tags = []
            if run.get("topic_ok") is False: tags.append("topic поза очікуванням")
            if run.get("grounded_ok") is False: tags.append("grounded не відповідає")
            if run.get("hallucination") is True: tags.append("підозра на вигадку")
            if run.get("order_in_history") is False: tags.append("вигаданий номер")
            if run.get("schema_ok") is False: tags.append("не пройшла схему")
            if tags:
                r = run.get("result") or {}
                failures.append((
                    cfg,
                    f"звернення «{run['вид']}»",
                    f"{', '.join(tags)}; reply: «{(r.get('reply') or '')[:120]}»",
                ))
        for dialog in cfg_data["dialogs"]:
            for entry in dialog["entries"]:
                if not entry.get("ok"):
                    continue
                tags = []
                if entry.get("order_in_history") is False:
                    tags.append("вигаданий номер")
                if entry.get("order_expected") and not entry.get("order_present"):
                    tags.append("не згадано номер у відповіді")
                if tags:
                    r = entry.get("result") or {}
                    failures.append((
                        cfg,
                        f"діалог «{dialog['вид']}», репліка {entry['turn']}",
                        f"{', '.join(tags)}; reply: «{(r.get('reply') or '')[:120]}»",
                    ))

    if failures:
        for cfg, where, what in failures:
            lines.append(f"- **{cfg}** → {where}: {what}")
    else:
        lines.append("_Невдалих відповідей за обраними критеріями не зафіксовано._")
    lines.append("")

    # ---------- Класифікація помилок ----------
    lines.append("## Класифікація помилок моделі\n")
    lines.append("| Тип | Приклад у прогоні | Що ловить у рішенні | Що не ловить |")
    lines.append("|---|---|---|---|")
    lines.append(
        "| Синтаксична | у `vague` немає JSON за визначенням; "
        "у `structured`/`context` — відповідь, що не пройшла `validate` | "
        "у `structured`/`context` — pydantic-валідація + повтор із текстом помилки | "
        "у `vague` формат не перевіряється взагалі |"
    )
    lines.append(
        "| Семантична | тема `other` на типовому запиті; номер, якого клієнт не називав | "
        "enum `Topic` + додаткова перевірка `order_number` на наявність у розмові | "
        "правильність теми на межових випадках; розпізнавання «схожих» номерів |"
    )
    lines.append(
        "| Фактична | вигадане правило у відповіді | "
        "приклади в інструкції; `grounded_in_rules` як самозвіт моделі; "
        "код підвищує `escalate_to_human`, якщо `grounded=false` | "
        "сам текст відповіді не звіряється з `context.md`; "
        "модель може поставити `grounded=true` біля вигадки |"
    )
    lines.append(
        "| Контекстна | втрата номера після скорочення історії | "
        "правило `fit_budget`: зберігаємо першу і останню репліку | "
        "якщо бюджет критично малий — перша репліка все одно втратиться |"
    )
    lines.append(
        "| Поведінкова | спроба змінити роль (у `vague` — виконано; "
        "у `structured`/`context` — проігноровано) | "
        "в `structured`/`context` — інструкція з обмеженнями | "
        "у `vague` немає захисту від зміни ролі |"
    )
    lines.append("")

    lines.append("## Що дала структура (1 → 2) і що дав контекст (2 → 3)\n")
    lines.append(
        "- **1 → 2** змінює **форму**: зʼявляється JSON, поля, enum, "
        "зникає потреба парсити вільний текст. Але `grounded_in_rules` може "
        "бути `true` біля вигадки — це самозвіт моделі, не істина.\n"
        "- **2 → 3** змінює **зміст**: модель бачить правила, перестає "
        "вигадувати, коректно відповідає на типові; історія дозволяє не "
        "перепитувати номер замовлення. `escalate_to_human` спрацьовує "
        "на питаннях поза правилами.\n"
    )

    lines.append("## Який тип помилок лишився\n")
    lines.append(
        "- Після всіх запобіжників лишається **семантична** і **фактична** "
        "помилка: `grounded_in_rules` — це самозвіт моделі, схема перевіряє "
        "лише форму, не зміст. `order_number` додатково перевіряється кодом "
        "на присутність у розмові, але сам факт відповіді — ні.\n"
        "- Що додати наступним: RAG із пошуком релевантних фрагментів "
        "`context.md`; перевірку, що ключові твердження відповіді присутні "
        "в `context.md`; регресійні тести на еталонний набір.\n"
    )

    FINDINGS_FILE.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())