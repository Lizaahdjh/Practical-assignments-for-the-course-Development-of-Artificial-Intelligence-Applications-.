"""Прогнати набір питань через RAG-конвеєр і класифікувати помилки.

Запуск (з кореня pr6):
    python eval/run.py
"""

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

QUESTIONS_FILE = Path(__file__).parent / "questions.json"
RESULTS_FILE = Path(__file__).parent / "results.json"
FINDINGS_FILE = Path(__file__).parent / "findings.md"


def _position_in_context(retrieved: list, expected_source: str) -> int | None:
    for i, hit in enumerate(retrieved, start=1):
        if hit.get("source") == expected_source:
            return i
    return None


def main() -> int:
    if not QUESTIONS_FILE.exists():
        print(f"Немає {QUESTIONS_FILE}. Скопіюйте questions.example.json.")
        return 1

    from app import index, keyword, rag

    idx = index.load()
    kw = keyword.build(idx.chunks)

    data = json.loads(QUESTIONS_FILE.read_text(encoding="utf-8"))
    questions = data["питання"]

    results = []
    stats = {
        "total": 0,
        "expected_in_context": 0,
        "found_answer": 0,
        "correct_refusal": 0,
        "wrong_refusal": 0,
        "wrong_answer": 0,
        "false_source": 0,
    }

    for q in questions:
        stats["total"] += 1
        entry = {
            "вид": q["вид"],
            "питання": q["питання"],
            "очікувані_документи": q.get("очікувані_документи", []),
            "очікується_відповідь": q.get("очікується_відповідь", True),
        }
        try:
            t0 = time.perf_counter()
            ans = rag.answer(q["питання"], idx, kw, filters=q.get("фільтр"))
            elapsed = time.perf_counter() - t0

            retrieved_sources = [h.chunk.source for h in ans.retrieved]
            cited_refs = {s.ref for s in ans.sources}
            cited_sources = [s.chunk.source for s in ans.sources]

            entry.update({
                "ok": True,
                "answer": ans.text,
                "found": ans.found,
                "sources": cited_sources,
                "retrieved_sources": retrieved_sources,
                "elapsed": round(elapsed, 3),
                "usage": ans.usage,
            })

            # Класифікація
            expected_docs = set(q.get("очікувані_документи", []))
            retrieved_set = set(retrieved_sources)
            cited_set = set(cited_sources)

            # Чи очікуваний документ у контексті
            hit_in_ctx = bool(expected_docs & retrieved_set) if expected_docs else None
            entry["expected_in_context"] = hit_in_ctx
            if hit_in_ctx:
                stats["expected_in_context"] += 1

            # Чи правильна відповідь
            expected_answer = q.get("очікується_відповідь", True)
            if expected_answer and ans.found:
                stats["found_answer"] += 1
            elif expected_answer and not ans.found:
                stats["wrong_refusal"] += 1
                entry["error_type"] = "модель: безпідставна відмова" \
                    if hit_in_ctx else "пошук: не знайдено"
            elif not expected_answer and not ans.found:
                stats["correct_refusal"] += 1
            elif not expected_answer and ans.found:
                stats["wrong_answer"] += 1
                entry["error_type"] = "модель: відповіла там, де не мала"

            # Хибні посилання: чи всі процитовані джерела серед retrieved
            if not cited_set.issubset(retrieved_set):
                stats["false_source"] += 1
                entry["error_type"] = "модель: хибне посилання"

            print(f"[{q['вид']:28s}] found={ans.found} "
                  f"cited={len(cited_refs)} retrieved={len(retrieved_sources)}")
        except Exception as e:
            entry.update({"ok": False, "error": f"{type(e).__name__}: {e}"})
            print(f"[{q['вид']:28s}] ПОМИЛКА: {e}")

        results.append(entry)

    RESULTS_FILE.write_text(
        json.dumps({"stats": stats, "results": results},
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\nРезультати: {RESULTS_FILE}")

    write_findings(stats, results)
    print(f"Висновки: {FINDINGS_FILE}")
    return 0


def write_findings(stats: dict, results: list) -> None:
    lines = ["# Висновки: RAG-помічник за базою знань\n"]
    total = stats["total"]
    lines.append(f"Набір: {total} питань.\n")

    lines.append("## Зведені метрики\n")
    lines.append("| Метрика | Значення |")
    lines.append("|---|---|")
    lines.append(f"| Питань з очікуваним документом у контексті | "
                 f"{stats['expected_in_context']}/{total} |")
    lines.append(f"| Правильних відповідей | {stats['found_answer']}/{total} |")
    lines.append(f"| Правильних відмов | {stats['correct_refusal']}/{total} |")
    lines.append(f"| Хибних відмов | {stats['wrong_refusal']}/{total} |")
    lines.append(f"| Хибних відповідей | {stats['wrong_answer']}/{total} |")
    lines.append(f"| Хибних посилань на джерела | {stats['false_source']}/{total} |")
    lines.append("")

    lines.append("## Деталі по питаннях\n")
    lines.append("| Вид | Питання | Очік. відп. | found | У контексті | Джерел |")
    lines.append("|---|---|---|---|---|---|")
    for r in results:
        if not r.get("ok"):
            lines.append(f"| {r['вид']} | {r['питання'][:50]} | — | — | — | помилка |")
            continue
        lines.append(
            f"| {r['вид']} | {r['питання'][:50]} "
            f"| {r['очікується_відповідь']} "
            f"| {r['found']} "
            f"| {r.get('expected_in_context', '—')} "
            f"| {len(r['sources'])} |"
        )
    lines.append("")

    lines.append("## Що написати від руки\n")
    lines.append("- Скільки невдач на боці пошуку й скільки на боці моделі?\n")
    lines.append("- На яких питаннях поле `found` розійшлося з вашою перевіркою?\n")
    lines.append("- Що змінила ваша одна зміна (крок 4)?\n")
    lines.append("- Що ви взяли б у застосунок для реальних клієнтів?\n")

    FINDINGS_FILE.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())