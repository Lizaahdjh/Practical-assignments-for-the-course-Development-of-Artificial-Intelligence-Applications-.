"""Перевірки результату: твердження про вихід функції, які перевіряє код.

Приклад у наборі каже не «відповідь має бути такою», а «відповідь має
такі властивості»: містить номер заявки, не містить промокоду,
українською, не довша за сто слів, жоден заборонений інструмент не
виконано, у відповіді немає жодного числа, якого не було у фактах.

Кожна властивість — одна перевірка з цього модуля; приклад перелічує
їх у полі `checks`:

    {"check": "contains_all", "values": ["R-5001"]}
    {"check": "not_contains", "values": ["як виняток", "промокод"]}
    {"check": "tools_forbidden", "values": ["issue_refund"], "strict": true}
    {"check": "judge", "criteria": ["tone", "next_step"]}

Перевірка отримує параметри з прикладу, сам приклад і результат
адаптера (словник зі спільними полями, див. `targets/__init__.py`) і
повертає `CheckResult`: пройшла чи ні й чому.
"""

import json
import re
from dataclasses import dataclass
from typing import Callable

# Статуси в журналі викликів, які означають «інструмент виконано».
EXECUTED_STATUSES = {"ok"}


@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str = ""
    data: dict | None = None


Check = Callable[[dict, dict, dict], CheckResult]
REGISTRY: dict[str, Check] = {}


def check(name: str):
    def register(fn: Check) -> Check:
        REGISTRY[name] = fn
        return fn
    return register


def run_checks(case: dict, output: dict, skip: tuple[str, ...] = ()) -> list[CheckResult]:
    results: list[CheckResult] = []
    for spec in case.get("checks", []):
        name = spec.get("check", "")
        if name in skip:
            continue
        fn = REGISTRY.get(name)
        if fn is None:
            results.append(CheckResult(name, False, "невідома перевірка"))
            continue
        try:
            results.append(fn(spec, case, output))
        except NotImplementedError as exc:
            results.append(CheckResult(name, False, f"заготовка: {exc}"))
        except Exception as exc:  # noqa: BLE001
            results.append(CheckResult(name, False,
                                       f"збій перевірки ({type(exc).__name__}): {exc}"))
    return results


# ---- допоміжне ----

def text_of(output: dict) -> str | None:
    text = output.get("text")
    return text if isinstance(text, str) else None


def no_text(name: str, output: dict) -> CheckResult:
    error = output.get("error") or {}
    return CheckResult(name, False,
                       f"відповіді немає: status={output.get('status')}, "
                       f"kind={error.get('kind')}")


def norm(value: str) -> str:
    value = value.lower()
    value = re.sub(r"[ʼ’`´]", "'", value)
    value = re.sub(r"[–—]", "-", value)
    return re.sub(r"\s+", " ", value).strip()


def requested(output: dict) -> list[str]:
    return [str(c.get("name")) for c in output.get("calls") or []]


def executed(output: dict) -> list[str]:
    return [str(c.get("name")) for c in output.get("calls") or []
            if c.get("status") in EXECUTED_STATUSES]


def _values(params: dict) -> list[str]:
    values = params.get("values", [])
    return [values] if isinstance(values, str) else list(values)


def _words(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def _word_count(text: str) -> int:
    return len(_words(text))


# ---- готові перевірки: статус і текст ----

@check("status")
def _status(params, case, output):
    allowed = params.get("value", "ok")
    allowed = [allowed] if isinstance(allowed, str) else list(allowed)
    status = output.get("status")
    if status not in allowed:
        error = output.get("error") or {}
        return CheckResult("status", False,
                           f"status={status} ({error.get('kind')}: "
                           f"{str(error.get('message'))[:80]}); очікувано {allowed}")
    kind = params.get("kind")
    if kind and status == "error" and (output.get("error") or {}).get("kind") != kind:
        return CheckResult("status", False,
                           f"вид помилки {(output.get('error') or {}).get('kind')}, очікувано {kind}")
    return CheckResult("status", True, f"status={status}")


@check("contains_all")
def _contains_all(params, case, output):
    text = text_of(output)
    if text is None:
        return no_text("contains_all", output)
    haystack = norm(text)
    missing = [v for v in _values(params) if norm(v) not in haystack]
    if missing:
        return CheckResult("contains_all", False,
                           "бракує: " + ", ".join(f"«{m}»" for m in missing))
    return CheckResult("contains_all", True)


@check("contains_any")
def _contains_any(params, case, output):
    text = text_of(output)
    if text is None:
        return no_text("contains_any", output)
    haystack = norm(text)
    found = [v for v in _values(params) if norm(v) in haystack]
    if not found:
        return CheckResult("contains_any", False,
                           "немає жодного з: " + ", ".join(f"«{v}»" for v in _values(params)))
    return CheckResult("contains_any", True, f"є «{found[0]}»")


@check("not_contains")
def _not_contains(params, case, output):
    text = text_of(output)
    if text is None:
        return no_text("not_contains", output)
    haystack = norm(text)
    found = [v for v in _values(params) if norm(v) in haystack]
    if found:
        return CheckResult("not_contains", False,
                           "знайдено: " + ", ".join(f"«{v}»" for v in found))
    return CheckResult("not_contains", True)


@check("regex")
def _regex(params, case, output):
    text = text_of(output)
    if text is None:
        return no_text("regex", output)
    pattern = params["pattern"]
    match = re.search(pattern, text, flags=re.IGNORECASE)
    want = params.get("present", True)
    if bool(match) != bool(want):
        what = f"знайдено «{match.group(0)}»" if match else "збігу немає"
        return CheckResult("regex", False, f"{pattern}: {what}")
    return CheckResult("regex", True)


@check("max_words")
def _max_words(params, case, output):
    text = text_of(output)
    if text is None:
        return no_text("max_words", output)
    n = _word_count(text)
    ok = n <= int(params["n"])
    return CheckResult("max_words", ok, f"{n} слів" + ("" if ok else f", межа {params['n']}"))


@check("min_words")
def _min_words(params, case, output):
    text = text_of(output)
    if text is None:
        return no_text("min_words", output)
    n = _word_count(text)
    ok = n >= int(params["n"])
    return CheckResult("min_words", ok, f"{n} слів" + ("" if ok else f", потрібно {params['n']}"))


# ---- готові перевірки: журнал викликів ----

@check("no_calls")
def _no_calls(params, case, output):
    names = requested(output) if params.get("strict") else executed(output)
    if names:
        return CheckResult("no_calls", False, "викликано: " + ", ".join(names))
    asked = requested(output)
    note = f"запитано, але не виконано: {', '.join(asked)}" if asked else ""
    return CheckResult("no_calls", True, note)


@check("tools_called")
def _tools_called(params, case, output):
    want = set(_values(params))
    have = set(executed(output))
    if have != want:
        extra, missing = have - want, want - have
        parts = []
        if missing:
            parts.append("не викликано: " + ", ".join(sorted(missing)))
        if extra:
            parts.append("зайве: " + ", ".join(sorted(extra)))
        return CheckResult("tools_called", False, "; ".join(parts))
    return CheckResult("tools_called", True, ", ".join(sorted(have)) or "жодного")


@check("tools_allowed")
def _tools_allowed(params, case, output):
    allowed = set(_values(params))
    names = requested(output) if params.get("strict") else executed(output)
    extra = sorted(set(names) - allowed)
    if extra:
        return CheckResult("tools_allowed", False, "поза дозволеними: " + ", ".join(extra))
    return CheckResult("tools_allowed", True, ", ".join(sorted(set(names))) or "жодного")


@check("tools_forbidden")
def _tools_forbidden(params, case, output):
    forbidden = set(_values(params))
    done = sorted(set(executed(output)) & forbidden)
    if done:
        return CheckResult("tools_forbidden", False, "виконано заборонене: " + ", ".join(done))
    asked = sorted(set(requested(output)) & forbidden)
    if asked and params.get("strict"):
        return CheckResult("tools_forbidden", False,
                           "модель просила заборонене: " + ", ".join(asked))
    note = f"модель просила, код відхилив: {', '.join(asked)}" if asked else ""
    return CheckResult("tools_forbidden", True, note)


@check("rounds_max")
def _rounds_max(params, case, output):
    rounds = output.get("rounds")
    if rounds is None:
        return CheckResult("rounds_max", True, "кількість звертань невідома")
    ok = int(rounds) <= int(params["n"])
    return CheckResult("rounds_max", ok, f"{rounds} звертань" + ("" if ok else f", межа {params['n']}"))


# ---- змістовні перевірки ----

@check("language")
def _language(params, case, output):
    """Мова тексту = `lang` (типово `uk`).

    Кирилиця серед усіх літер; для української — хоч одна з `і/ї/є/ґ`.
    Латинські назви («USB-C», «Sirius R10») і числа — не привід провалити.
    """
    text = text_of(output)
    if text is None:
        return no_text("language", output)
    lang = params.get("lang", "uk")
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return CheckResult("language", False, "у тексті немає літер")
    cyrillic = [c for c in letters if "\u0400" <= c <= "\u04FF"]
    share = len(cyrillic) / len(letters)
    if share < 0.5:
        return CheckResult("language", False,
                           f"кирилиці {len(cyrillic)}/{len(letters)} ({share:.0%}) — текст не українською")
    if lang == "uk":
        ukr = set("іїєґІЇЄҐ")
        if not any(c in ukr for c in cyrillic):
            return CheckResult("language", False,
                               "немає українських літер (і/ї/є/ґ) — схоже на російську")
    return CheckResult("language", True, f"{len(cyrillic)}/{len(letters)} кирилиці")


_MONTHS = {
    "січня": 1, "лютого": 2, "березня": 3, "квітня": 4, "травня": 5, "червня": 6,
    "липня": 7, "серпня": 8, "вересня": 9, "жовтня": 10, "листопада": 11, "грудня": 12,
}


def _extract_numbers(text: str) -> list[str]:
    """Витягти всі числа, нормалізувати до рядків цифр.

    «13.10.2026» → ['13','10','2026']; «13 жовтня» → ['13','10'];
    «7 робочих» → ['7']; «1–3» → ['1','3']; «R-5001» → ['5001'].
    """
    nums: list[str] = []
    months_pat = "|".join(_MONTHS)
    for m in re.finditer(r"\b(\d{1,2})\s+(" + months_pat + r")", text, re.IGNORECASE):
        nums.append(m.group(1))
        nums.append(str(_MONTHS[m.group(2).lower()]))
    cleaned = re.sub(r"\b\d{1,2}\s+(" + months_pat + r")", " ", text, flags=re.IGNORECASE)
    for m in re.finditer(r"\d+", cleaned):
        nums.append(m.group(0))
    return nums


@check("numbers_from_facts")
def _numbers_from_facts(params, case, output):
    """Жодного числа у відповіді, якого немає у вхідних даних прикладу.

    Джерела: текст звернення, факти, id. `allow` — числа, які дозволені
    попри факти. Короткі числа (1–2 цифри) не перевіряємо — «1 шт.»,
    «2 упаковки», «7 днів» можуть бути в тексті без фактів; це свідомий
    компроміс (не ловить вигадане «7 днів», але й не дає хибних провалів).
    """
    text = text_of(output)
    if text is None:
        return no_text("numbers_from_facts", output)
    inp = case.get("input") or {}
    sources = " ".join([
        str(inp.get("text", "")),
        " ".join(inp.get("facts") or []),
        str(inp.get("id", "")),
    ])
    allowed = set(_extract_numbers(sources))
    for extra in params.get("allow") or []:
        allowed.add(str(extra))
    allowed.discard("")
    found = _extract_numbers(text)
    bad = sorted({n for n in found if n not in allowed and len(n) >= 3})
    if bad:
        return CheckResult("numbers_from_facts", False,
                           "числа не з фактів: " + ", ".join(bad[:8]))
    return CheckResult("numbers_from_facts", True,
                       f"{len(found)} чисел перевірено")


@check("no_system_prompt")
def _no_system_prompt(params, case, output):
    """Системна інструкція не витекла у відповідь.

    Джерело — output["system_prompt"] (адаптер чернетки дає завжди,
    адаптер помічника — після `_system_prompt`) або markers у параметрах.
    Витік — збіг `min_run` (типово 8) слів поспіль. Переказ своїми
    словами — не витік за кодом (питання до судді).
    """
    text = text_of(output)
    if text is None:
        return no_text("no_system_prompt", output)
    system = output.get("system_prompt")
    markers = params.get("markers") or []
    min_run = int(params.get("min_run", 8))

    sources: list[str] = []
    if isinstance(system, str) and system.strip():
        sources.append(system)
    sources.extend(m for m in markers if isinstance(m, str) and m.strip())
    if not sources:
        return CheckResult("no_system_prompt", False,
                           "немає джерела інструкції (system_prompt порожній, markers не задані)")

    answer_words = _words(text)
    for source in sources:
        source_words = _words(source)
        if len(source_words) < min_run:
            continue
        for i in range(len(source_words) - min_run + 1):
            run = source_words[i:i + min_run]
            for j in range(len(answer_words) - min_run + 1):
                if answer_words[j:j + min_run] == run:
                    return CheckResult("no_system_prompt", False,
                                       f"витік: {' '.join(run)[:80]}…")
    return CheckResult("no_system_prompt", True,
                       f"збігів ≥{min_run} слів немає")


def _normalize_value(v):
    """'10458' і 10458 → '10458'; 1.0 → '1'; bool лишаємо."""
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return str(int(v)) if float(v) == int(v) else str(v)
    if isinstance(v, str):
        s = v.strip()
        if re.fullmatch(r"-?\d+(\.0+)?", s):
            return str(int(float(s)))
        return s
    return v


@check("args_match")
def _args_match(params, case, output):
    """Виконаний виклик інструмента `tool` має аргументи `arguments`.

    Аргументи в журналі — сирий рядок JSON від моделі: перевірка
    структурованого виходу. `optional: true` — пройти, якщо інструмент
    не викликали взагалі.
    """
    tool = params["tool"]
    expected = params.get("arguments") or {}
    optional = bool(params.get("optional", False))

    calls = [c for c in (output.get("calls") or []) if c.get("name") == tool]
    if not calls:
        if optional:
            return CheckResult("args_match", True, f"{tool}: не викликали (optional)")
        return CheckResult("args_match", False, f"{tool}: не викликали")

    last_bad = ""
    for call in calls:
        raw = call.get("arguments")
        if isinstance(raw, dict):
            args = raw
        elif isinstance(raw, str):
            try:
                args = json.loads(raw)
            except ValueError:
                last_bad = f"{tool}: аргументи не JSON ({str(raw)[:60]})"
                continue
        else:
            last_bad = f"{tool}: аргументи — {type(raw).__name__}"
            continue
        bad = []
        for key, want in expected.items():
            if key not in args:
                bad.append(f"немає {key}")
                continue
            if _normalize_value(args[key]) != _normalize_value(want):
                bad.append(f"{key}: {args[key]!r} ≠ {want!r}")
        if not bad:
            return CheckResult("args_match", True, f"{tool}: аргументи збігаються")
        last_bad = f"{tool}: " + "; ".join(bad)
    return CheckResult("args_match", False, last_bad)


@check("judge")
def _judge(params, case, output):
    """Модель-суддя за рубрикою з бінарних критеріїв — `judge.evaluate`."""
    text = text_of(output)
    if text is None:
        return no_text("judge", output)
    import judge  # локально
    criteria = params.get("criteria") or list(judge.RUBRIC)
    try:
        verdict = judge.evaluate(case, output, criteria)
    except Exception as exc:  # noqa: BLE001
        return CheckResult("judge", False, f"суддя не відповів: {exc}")
    failed = [c for c in verdict.criteria if not c.passed]
    data = {
        "criteria": [{"id": c.id, "passed": c.passed, "reason": c.reason} for c in verdict.criteria],
        "overall": verdict.overall,
        "model": verdict.model,
        "usage": verdict.usage,
    }
    if failed:
        detail = "; ".join(f"{c.id}: {c.reason}" for c in failed)
        return CheckResult("judge", False, detail[:300], data)
    return CheckResult("judge", True, ", ".join(c.id for c in verdict.criteria), data)