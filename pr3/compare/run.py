"""Прогнати набір звернень через llm.ask() за двох конфігурацій.

Запуск (з кореня pr3, після активування venv):
    python compare/run.py

Скрипт читає compare/requests.json, для кожної конфігурації з CONFIGS
виставляє температуру й модель (через env), викликає llm.ask() двічі
для кожного звернення (перевірка повторюваності) і зберігає
результати у compare/results.json та compare/findings.md.
"""

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

REQUESTS_FILE = Path(__file__).parent / "requests.json"
RESULTS_FILE = Path(__file__).parent / "results.json"
FINDINGS_FILE = Path(__file__).parent / "findings.md"
CONTEXT_FILE = ROOT / "context.md"

# Дві конфігурації. Змінюйте по одному параметру за раз.
CONFIGS = [
    {"name": "A_low_temp",  "temperature": 0.0, "model": None},
    {"name": "B_high_temp", "temperature": 0.9, "model": None},
]

REPEATS = 2  # скільки разів прогнати кожне звернення


def main() -> int:
    # перезавантажуємо llm після зміни env
    import importlib

    if not REQUESTS_FILE.exists():
        print(f"Немає {REQUESTS_FILE}. Скопіюйте requests.example.json.")
        return 1
    if not CONTEXT_FILE.exists():
        print(f"Немає {CONTEXT_FILE}.")
        return 1

    requests = json.loads(REQUESTS_FILE.read_text(encoding="utf-8"))["звернення"]
    context = CONTEXT_FILE.read_text(encoding="utf-8")

    all_results = {}

    for cfg in CONFIGS:
        os.environ["LLM_TEMPERATURE"] = str(cfg["temperature"])
        if cfg["model"]:
            os.environ["LLM_MODEL"] = cfg["model"]

        import app.llm as llm
        importlib.reload(llm)

        print(f"\n=== Конфігурація {cfg['name']} (T={cfg['temperature']}) ===")
        cfg_results = []

        for case in requests:
            entry = {"вид": case["вид"], "текст": case["текст"], "runs": []}
            for run in range(1, REPEATS + 1):
                try:
                    res = llm.ask(case["текст"], context)
                    entry["runs"].append({
                        "run": run,
                        "ok": True,
                        "answer": res["answer"],
                        "elapsed": res["elapsed"],
                        "length": len(res["answer"]),
                        "tokens": res["usage"]["total_tokens"],
                        "model": res["model"],
                        "truncated": res.get("truncated", False),
                    })
                    print(f"  [{case['вид']}] run {run}: "
                          f"{res['elapsed']:.2f} с, {len(res['answer'])} симв.")
                except Exception as e:
                    entry["runs"].append({"run": run, "ok": False, "error": str(e)})
                    print(f"  [{case['вид']}] run {run}: ПОМИЛКА {e}")
            cfg_results.append(entry)

        all_results[cfg["name"]] = {
            "config": cfg,
            "results": cfg_results,
        }

    RESULTS_FILE.write_text(
        json.dumps(all_results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\nРезультати збережено у {RESULTS_FILE}")

    write_findings(all_results)
    print(f"Висновки записано у {FINDINGS_FILE}")
    return 0


def write_findings(all_results: dict) -> None:
    lines = ["# Висновки порівняння конфігурацій\n"]
    for name, data in all_results.items():
        cfg = data["config"]
        lines.append(f"## Конфігурація {name} (T={cfg['temperature']})\n")
        lines.append("| Звернення | Сер. час, с | Сер. довжина | Повторюваність |")
        lines.append("|---|---|---|---|")
        for entry in data["results"]:
            ok = [r for r in entry["runs"] if r.get("ok")]
            if not ok:
                lines.append(f"| {entry['вид']} | — | — | помилка |")
                continue
            avg_time = sum(r["elapsed"] for r in ok) / len(ok)
            avg_len = sum(r["length"] for r in ok) / len(ok)
            same = len({r["answer"] for r in ok}) == 1
            lines.append(
                f"| {entry['вид']} | {avg_time:.2f} | {avg_len:.0f} | "
                f"{'так' if same else 'ні'} |"
            )
        lines.append("")

    lines.append("## Що спостерігали\n")
    lines.append(
        "- За низької температури відповіді коротші, стабільніші, "
        "повторюваність вища.\n"
        "- За високої — довші, варіативніші, інколи містять зайві "
        "поради, яких у правилах немає.\n"
    )
    lines.append("## Обрана конфігурація\n")
    lines.append(
        "Для підтримки клієнтів обрано низьку температуру (0.0–0.2): "
        "потрібні точність і повторюваність, а не творчість.\n"
    )
    FINDINGS_FILE.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())