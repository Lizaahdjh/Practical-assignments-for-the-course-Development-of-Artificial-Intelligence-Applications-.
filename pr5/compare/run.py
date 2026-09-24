"""Прогнати набір запитів через семантичний і keyword-пошук.

Запуск (з кореня pr5, після активування venv і `python ingest.py`):

    python compare/run.py

Читає `compare/queries.json`, проганяє кожен запит обома способами,
рахує hit@1 і hit@3 (позицію очікуваного документа у видачі), збирає
час і формує `compare/findings.md`.
"""

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

QUERIES_FILE = Path(__file__).parent / "queries.json"
RESULTS_FILE = Path(__file__).parent / "results.json"
FINDINGS_FILE = Path(__file__).parent / "findings.md"

TOP_K = 5


def _position(hits: list, source: str) -> int | None:
    """Позиція очікуваного документа у видачі (1-based) або None."""
    if not source:
        return None
    for i, h in enumerate(hits, start=1):
        if h["source"] == source:
            return i
    return None


def main() -> int:
    if not QUERIES_FILE.exists():
        print(f"Немає {QUERIES_FILE}. Скопіюйте queries.example.json.")
        return 1

    data = json.loads(QUERIES_FILE.read_text(encoding="utf-8"))
    queries = data["запити"]

    # Індекси будуємо один раз
    from app import embeddings, index, keyword
    idx = index.load()
    kw_idx = keyword.build(idx.chunks)

    results = {"top_k": TOP_K, "queries": []}
    hit1_sem = hit3_sem = hit1_kw = hit3_kw = 0
    answerable = 0

    for q in queries:
        entry = {
            "вид": q["вид"],
            "запит": q["запит"],
            "очікуваний_документ": q.get("очікуваний_документ", ""),
            "фільтр": q.get("фільтр"),
        }

        filters = q.get("фільтр") or None
        if filters:
            filters = dict(filters)
            filters.setdefault("audience", "клієнти")
        else:
            filters = {"audience": "клієнти"}
        expected = q.get("очікуваний_документ") or ""

        # Семантичний
        try:
            t0 = time.perf_counter()
            vector = embeddings.embed_query(q["запит"])
            sem_hits = index.search(idx, vector, top_k=TOP_K, filters=filters)
            sem_elapsed = time.perf_counter() - t0
            sem_list = [{"source": h.chunk.source, "score": h.score} for h in sem_hits]
            entry["semantic"] = {
                "hits": sem_list,
                "elapsed": round(sem_elapsed, 4),
                "position": _position(sem_list, expected),
                "top_score": sem_list[0]["score"] if sem_list else None,
            }
        except Exception as e:
            entry["semantic"] = {"error": f"{type(e).__name__}: {e}"}

        # Keyword
        try:
            t0 = time.perf_counter()
            kw_hits = keyword.search(kw_idx, q["запит"], top_k=TOP_K, filters=filters)
            kw_elapsed = time.perf_counter() - t0
            kw_list = [{"source": h.chunk.source, "score": h.score} for h in kw_hits]
            entry["keyword"] = {
                "hits": kw_list,
                "elapsed": round(kw_elapsed, 4),
                "position": _position(kw_list, expected),
                "top_score": kw_list[0]["score"] if kw_list else None,
            }
        except Exception as e:
            entry["keyword"] = {"error": f"{type(e).__name__}: {e}"}

        if expected:
            answerable += 1
            if entry["semantic"].get("position") == 1:
                hit1_sem += 1
            if entry["semantic"].get("position") in (1, 2, 3):
                hit3_sem += 1
            if entry["keyword"].get("position") == 1:
                hit1_kw += 1
            if entry["keyword"].get("position") in (1, 2, 3):
                hit3_kw += 1

        results["queries"].append(entry)
        s_p = entry["semantic"].get("position")
        k_p = entry["keyword"].get("position")
        print(f"[{q['вид']:28s}] sem@{s_p} kw@{k_p}")

    results["summary"] = {
        "answerable": answerable,
        "semantic": {"hit@1": hit1_sem, "hit@3": hit3_sem},
        "keyword": {"hit@1": hit1_kw, "hit@3": hit3_kw},
    }

    previous = None
    if RESULTS_FILE.exists():
        try:
            previous = json.loads(RESULTS_FILE.read_text(encoding="utf-8"))
        except Exception:
            previous = None

    RESULTS_FILE.write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\nРезультати: {RESULTS_FILE}")

    write_findings(results, previous)
    print(f"Висновки: {FINDINGS_FILE}")
    return 0


def write_findings(results: dict, previous: dict | None) -> None:
    lines = ["# Висновки: порівняння семантичного і keyword-пошуку\n"]
    s = results["summary"]

    lines.append(
        f"Набір: {s['answerable']} запитів із відомим документом, "
        f"top_k={results['top_k']}.\n"
    )
    lines.append("## Зведені метрики\n")
    lines.append("| Спосіб | hit@1 | hit@3 |")
    lines.append("|---|---|---|")
    lines.append(
        f"| Семантичний | {s['semantic']['hit@1']}/{s['answerable']} "
        f"| {s['semantic']['hit@3']}/{s['answerable']} |"
    )
    lines.append(
        f"| Keyword | {s['keyword']['hit@1']}/{s['answerable']} "
        f"| {s['keyword']['hit@3']}/{s['answerable']} |"
    )
    lines.append("")

    lines.append("## Деталі по запитах\n")
    lines.append(
        "| Вид | Запит | Очік. | Sem поз. | Sem top "
        "| Kw поз. | Kw top | Час sem, мс | Час kw, мс |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|")

    for q in results["queries"]:
        sem = q.get("semantic", {})
        kw = q.get("keyword", {})
        sem_top = sem.get("top_score")
        kw_top = kw.get("top_score")
        sem_top_str = f"{sem_top:.3f}" if sem_top is not None else "—"
        kw_top_str = f"{kw_top:.2f}" if kw_top is not None else "—"
        lines.append(
            f"| {q['вид']} | {q['запит'][:60]} | {q['очікуваний_документ']} "
            f"| {sem.get('position', '—')} | {sem_top_str} "
            f"| {kw.get('position', '—')} | {kw_top_str} "
            f"| {sem.get('elapsed', 0) * 1000:.0f} "
            f"| {kw.get('elapsed', 0) * 1000:.1f} |"
        )
    lines.append("")

    if previous and previous.get("summary"):
        lines.append("## Порівняння з попереднім прогоном\n")
        ps = previous["summary"]
        lines.append(
            f"- Попередній набір: {ps['answerable']} запитів, "
            f"top_k={previous.get('top_k')}."
        )
        lines.append(
            f"- Семантичний: hit@1 {ps['semantic']['hit@1']} → "
            f"{s['semantic']['hit@1']}, hit@3 "
            f"{ps['semantic']['hit@3']} → {s['semantic']['hit@3']}."
        )
        lines.append(
            f"- Keyword: hit@1 {ps['keyword']['hit@1']} → "
            f"{s['keyword']['hit@1']}, hit@3 "
            f"{ps['keyword']['hit@3']} → {s['keyword']['hit@3']}."
        )
        lines.append("")

    lines.append("## Що написати від руки\n")
    lines.append(
        "- На яких видах запитів програє семантичний пошук, "
        "а на яких — keyword?\n"
    )
    lines.append(
        "- Який поріг схожості ви б поставили і скільки правильних "
        "відповідей він відсік би?\n"
    )
    lines.append("- Що показала зміна `CHUNK_SIZE` (крок 5)?\n")
    lines.append(
        "- Що ви взяли б у застосунок: один спосіб, другий чи обидва?\n"
    )

    FINDINGS_FILE.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())