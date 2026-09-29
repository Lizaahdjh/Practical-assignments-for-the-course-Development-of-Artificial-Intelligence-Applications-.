"""Прогнати всі зразки через конвеєр і порівняти з expected.json.

Запуск (з кореня pr7):
    python eval/run.py
"""

import json
import sys
import time
from collections import defaultdict
from decimal import Decimal, InvalidOperation
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

SAMPLES = ROOT / "samples"
EXPECTED_FILE = SAMPLES / "expected.json"
RESULTS_FILE = Path(__file__).parent / "results.json"
FINDINGS_FILE = Path(__file__).parent / "findings.md"

CRITICAL_PATHS = ("supplier.iban", "supplier.code", "total")


def _norm(v):
    """Нормалізувати значення для порівняння."""
    if v is None:
        return None
    if isinstance(v, str):
        s = v.strip().replace("ʼ", "'").replace("’", "'").replace("`", "'")
        # IBAN без пробілів
        if s.upper().startswith("UA") and len(s.replace(" ", "")) == 29:
            return s.replace(" ", "").upper()
        # Суми
        try:
            return Decimal(s.replace(" ", "").replace(",", "."))
        except (InvalidOperation, ValueError):
            pass
        return s
    return v


def _flatten(obj, prefix="", out=None):
    if out is None:
        out = []
    for k, v in (obj or {}).items():
        path = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            _flatten(v, path, out)
        elif isinstance(v, list) and v and isinstance(v[0], dict):
            for i, item in enumerate(v):
                _flatten(item, f"{path}.{i}", out)
        else:
            out.append((path, v))
    return out


def _compare(expected, actual):
    """Порівняти поле за полем після нормалізації."""
    exp = dict(_flatten(expected))
    act = dict(_flatten(actual))
    keys = set(exp) | set(act)
    results = {}
    for k in keys:
        e = _norm(exp.get(k))
        a = _norm(act.get(k))
        if e is None and a is None:
            results[k] = "правильно"
        elif e is None and a is not None:
            results[k] = "вигадка"
        elif e is not None and a is None:
            results[k] = "пропуск"
        elif e == a:
            results[k] = "правильно"
        else:
            results[k] = "помилка"
    return results


def _matches_checks(expected_checks, issues):
    """Чи всі очікувані правила спрацювали."""
    fired = {i.rule for i in issues}
    return [c for c in expected_checks if c in fired]


def main() -> int:
    if not EXPECTED_FILE.exists():
        print(f"Немає {EXPECTED_FILE}.")
        return 1

    from app import extraction

    expected = json.loads(EXPECTED_FILE.read_text(encoding="utf-8"))

    results = []
    by_variant = defaultdict(lambda: {"total": 0, "ok": 0, "err": 0,
                                       "skip": 0, "invent": 0,
                                       "caught": 0, "silent": 0,
                                       "decision_ok": 0,
                                       "tokens": 0})

    for path, exp in expected.items():
        entry = {
            "file": path,
            "variant": exp.get("variant"),
            "note": exp.get("note", ""),
        }
        full = SAMPLES / path
        if not full.exists():
            entry["error"] = f"Немає файлу {full}"
            results.append(entry)
            continue

        try:
            t0 = time.perf_counter()
            res = extraction.process(full.read_bytes())
            elapsed = time.perf_counter() - t0
            entry.update({
                "decision": res.decision,
                "expected_decision": exp.get("expected_decision"),
                "issues_rules": sorted({i.rule for i in res.issues}),
                "expected_checks": exp.get("expected_checks", []),
                "elapsed": round(elapsed, 2),
                "usage": res.usage,
            })
            field_results = _compare(exp.get("fields") or {}, res.document or {})
            entry["field_results"] = field_results
            entry["decision_ok"] = (res.decision == exp.get("expected_decision"))
            entry["checks_caught"] = _matches_checks(exp.get("expected_checks", []), res.issues)

            # Статистика по варіанту
            v = by_variant[exp["variant"]]
            v["total"] += len(field_results)
            for k, verdict in field_results.items():
                if verdict == "правильно":
                    v["ok"] += 1
                elif verdict == "помилка":
                    v["err"] += 1
                elif verdict == "пропуск":
                    v["skip"] += 1
                elif verdict == "вигадка":
                    v["invent"] += 1
                # «правильно чи спіймано» — тільки для помилок і вигадок
                if verdict in ("помилка", "вигадка"):
                    caught = any(
                        i.field == k or i.field.startswith(k + ".")
                        for i in res.issues
                    )
                    if caught:
                        v["caught"] += 1
                    else:
                        v["silent"] += 1
            if entry["decision_ok"]:
                v["decision_ok"] += 1
            if res.usage:
                v["tokens"] += res.usage.get("total_tokens", 0)

            print(f"[{exp['variant']:8s}] {path} → {res.decision} "
                  f"(очік. {exp.get('expected_decision')}) "
                  f"{'✓' if entry['decision_ok'] else '✗'}")
        except Exception as e:
            entry.update({"error": f"{type(e).__name__}: {e}"})
            print(f"[{exp['variant']:8s}] {path} ПОМИЛКА: {e}")
        results.append(entry)

    RESULTS_FILE.write_text(
        json.dumps({"by_variant": dict(by_variant), "results": results},
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\nРезультати: {RESULTS_FILE}")

    write_findings(by_variant, results)
    print(f"Висновки: {FINDINGS_FILE}")
    return 0


def write_findings(by_variant, results) -> None:
    lines = ["# Висновки: якісні проти погіршених\n"]
    lines.append("## Зведення по варіантах\n")
    lines.append("| Варіант | Файлів | Полів OK | Помилок | Пропусків | Вигадок "
                 "| Спіймано | Тихих | Рішення OK | Токени |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for variant, v in sorted(by_variant.items()):
        lines.append(
            f"| {variant} | {v['total']} | {v['ok']} | {v['err']} | {v['skip']} "
            f"| {v['invent']} | {v['caught']} | {v['silent']} "
            f"| {v['decision_ok']} | {v['tokens']} |"
        )
    lines.append("")

    lines.append("## Деталі по файлах\n")
    lines.append("| Файл | Рішення | Очік. | Збіг | Проблем |")
    lines.append("|---|---|---|---|---|")
    for r in results:
        if "error" in r:
            lines.append(f"| {r['file']} | — | — | ПОМИЛКА | {r['error']} |")
            continue
        lines.append(
            f"| {r['file']} | {r['decision']} | {r['expected_decision']} "
            f"| {'✓' if r['decision_ok'] else '✗'} "
            f"| {', '.join(r['issues_rules']) or '—'} |"
        )
    lines.append("")

    lines.append("## Що написати від руки\n")
    lines.append("- Який вид погіршення найшкідливіший?\n")
    lines.append("- Скільки тихих помилок? Що з ними робити?\n")
    lines.append("- Що зробила модель із помилкою постачальника в rahunok-05?\n")
    lines.append("- Що показав rahunok-08 із «новими реквізитами»?\n")
    lines.append("- Що змінила одна зміна (IMAGE_MAX_SIDE тощо)?\n")

    FINDINGS_FILE.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())