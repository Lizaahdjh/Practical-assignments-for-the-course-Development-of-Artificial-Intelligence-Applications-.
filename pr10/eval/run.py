"""Прогнати сценарії ПР10 і зібрати звіт.

Запуск (з кореня pr10):
    python eval/run.py

Скрипт:
1. Прогонить усі звернення зі samples/requests.json на базовій
   конфігурації (mock, без збоїв).
2. Прогонить ключові сценарії збоїв: таймаут, бездіяльність, auth,
   rate_limit, mid_stream, fallback, скасування.
3. Збереже eval/results.json і згенерує eval/findings.md із
   порівняльною таблицею.

Не запускайте одночасно з uvicorn — реєстр чернеток у памʼяті
процесу. Або запускайте в окремому процесі.
"""
import importlib
import json
import os
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

RESULTS_FILE = Path(__file__).parent / "results.json"
FINDINGS_FILE = Path(__file__).parent / "findings.md"
SAMPLES_FILE = ROOT / "samples" / "requests.json"


def _wait_draft(draft, timeout: float = 60) -> None:
    """Дочекатися завершення фонової задачі."""
    started = time.perf_counter()
    while draft.status not in ("done", "failed", "cancelled"):
        if time.perf_counter() - started > timeout:
            draft.status = "failed"
            draft.error = {"kind": "timeout", "message": "не завершилось"}
            return
        time.sleep(0.05)


def _run_config(name: str, env_overrides: dict, samples: list[dict],
                request_id: str = "d01") -> dict:
    """Прогнати одну конфігурацію на одному зверненні."""
    for k, v in env_overrides.items():
        os.environ[k] = str(v)

    # Перезавантажуємо модулі, щоб підхопити нові env
    for mod in ("app.gateway", "app.jobs", "app.providers.mock",
                "app.providers.openai_compat"):
        if mod in sys.modules:
            importlib.reload(sys.modules[mod])

    from app import jobs, samples as samples_mod

    sample = next((s for s in samples if s["id"] == request_id), None)
    if sample is None:
        return {"config": name, "error": "звернення не знайдено"}

    started = time.perf_counter()
    draft = jobs.start(sample, client=f"runner-{name}")
    _wait_draft(draft)
    elapsed = time.perf_counter() - started

    return {
        "config": name,
        "env": env_overrides,
        "request_id": request_id,
        "status": draft.status,
        "attempts": draft.attempts,
        "provider": draft.provider,
        "model": draft.model,
        "fallback": draft.fallback,
        "error": draft.error,
        "elapsed": round(elapsed, 2),
        "first_token": draft.elapsed.get("first_token"),
        "total": draft.elapsed.get("total"),
        "text_len": len(draft.text or ""),
        "warnings": draft.warnings,
    }


def _scenario_cancel(sample: dict) -> dict:
    """Скасування посеред генерації."""
    from app import jobs

    draft = jobs.start(sample, client="runner-cancel")
    # Скасуємо через 100 мс — має встигнути хоча б один фрагмент
    time.sleep(0.1)
    try:
        jobs.cancel(draft.draft_id)
    except ValueError as e:
        return {"scenario": "cancel", "error": str(e)}

    _wait_draft(draft)
    return {
        "scenario": "cancel",
        "status": draft.status,
        "text_len": len(draft.text or ""),
    }


def _scenario_cancel_finished(sample: dict) -> dict:
    """Скасування завершеної — має бути відмова."""
    from app import jobs

    draft = jobs.start(sample, client="runner-cancel-done")
    _wait_draft(draft)
    try:
        jobs.cancel(draft.draft_id)
        return {"scenario": "cancel_done", "error": "не підняло виняток"}
    except ValueError as e:
        return {"scenario": "cancel_done", "status": "rejected", "message": str(e)}


def main() -> int:
    if not SAMPLES_FILE.exists():
        print(f"Немає {SAMPLES_FILE}")
        return 1

    samples = json.loads(SAMPLES_FILE.read_text(encoding="utf-8"))

    # Базові налаштування
    base = {
        "LLM_PROVIDER": "mock",
        "LLM_FALLBACK": "none",
        "LLM_TIMEOUT": "5",
        "LLM_IDLE_TIMEOUT": "2",
        "LLM_TOTAL_TIMEOUT": "15",
        "LLM_MAX_ATTEMPTS": "3",
        "MOCK_FIRST_DELAY_MS": "100",
        "MOCK_DELAY_MS": "5",
        "MOCK_FAILURE_RATE": "0",
        "MOCK_FAILURE_KIND": "unavailable",
    }
    for k, v in base.items():
        os.environ[k] = v

    scenarios = []

    print("=== Базова конфігурація ===")
    for s in samples:
        r = _run_config(f"base/{s['id']}", base, samples, request_id=s["id"])
        scenarios.append(r)
        print(f"  {s['id']}: {r.get('status')} (спроб {r.get('attempts')}, "
              f"{r.get('elapsed')} с)")

    print("\n=== Таймаут спроби ===")
    env = {**base, "MOCK_FIRST_DELAY_MS": "10000"}
    scenarios.append(_run_config("timeout", env, samples))
    print(f"  {scenarios[-1].get('status')}, спроб {scenarios[-1].get('attempts')}")

    print("\n=== Таймаут бездіяльності ===")
    env = {**base, "MOCK_FIRST_DELAY_MS": "100", "MOCK_DELAY_MS": "5000"}
    scenarios.append(_run_config("idle_timeout", env, samples))
    print(f"  {scenarios[-1].get('status')}")

    print("\n=== Auth (без повтору) ===")
    env = {**base, "MOCK_FAILURE_RATE": "1", "MOCK_FAILURE_KIND": "auth"}
    scenarios.append(_run_config("auth", env, samples))
    print(f"  {scenarios[-1].get('status')}, спроб {scenarios[-1].get('attempts')}")

    print("\n=== Rate limit ===")
    env = {**base, "MOCK_FAILURE_RATE": "1", "MOCK_FAILURE_KIND": "rate_limit"}
    scenarios.append(_run_config("rate_limit", env, samples))
    print(f"  {scenarios[-1].get('status')}, спроб {scenarios[-1].get('attempts')}")

    print("\n=== Mid-stream обрив ===")
    env = {**base, "MOCK_FAILURE_RATE": "1", "MOCK_FAILURE_KIND": "mid_stream"}
    scenarios.append(_run_config("mid_stream", env, samples))
    print(f"  {scenarios[-1].get('status')}, "
          f"текст {scenarios[-1].get('text_len')} симв.")

    print("\n=== Fallback (openai з невірним ключем → mock) ===")
    env = {
        **base,
        "LLM_PROVIDER": "openai",
        "LLM_API_KEY": "невірний-ключ",
        "LLM_FALLBACK": "mock",
    }
    scenarios.append(_run_config("fallback", env, samples))
    print(f"  {scenarios[-1].get('status')}, fallback={scenarios[-1].get('fallback')}")

    print("\n=== Скасування посеред генерації ===")
    scenarios.append(_scenario_cancel(samples[0]))
    print(f"  {scenarios[-1].get('status')}, текст {scenarios[-1].get('text_len')} симв.")

    print("\n=== Скасування завершеної ===")
    scenarios.append(_scenario_cancel_finished(samples[0]))
    print(f"  {scenarios[-1].get('status')}")

    RESULTS_FILE.write_text(
        json.dumps({"scenarios": scenarios, "base_env": base},
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\nРезультати: {RESULTS_FILE}")

    write_findings(scenarios)
    print(f"Висновки: {FINDINGS_FILE}")
    return 0


def write_findings(scenarios: list[dict]) -> None:
    lines = ["# Висновки: перевірка стійкості ПР10\n"]
    lines.append("## Зведена таблиця\n")
    lines.append("| Конфігурація | Статус | Спроб | Провайдер | Fallback "
                 "| Помилка | Час, с | Перший токен, с |")
    lines.append("|---|---|---|---|---|---|---|---|")

    for s in scenarios:
        if "scenario" in s:
            lines.append(
                f"| {s['scenario']} | {s.get('status', s.get('error', '—'))} "
                f"| — | — | — | {s.get('message', '—')} | — | — |"
            )
            continue
        err = s.get("error") or {}
        err_kind = err.get("kind", "—") if isinstance(err, dict) else str(err)
        lines.append(
            f"| {s.get('config', '?')} | {s.get('status')} "
            f"| {s.get('attempts')} | {s.get('provider', '—')} "
            f"| {'так' if s.get('fallback') else 'ні'} "
            f"| {err_kind} | {s.get('elapsed', '—')} "
            f"| {s.get('first_token', '—')} |"
        )
    lines.append("")

    lines.append("## Що писати від руки\n")
    lines.append("- Звідки взяті `LLM_TIMEOUT`, `LLM_IDLE_TIMEOUT`, "
                 "`LLM_TOTAL_TIMEOUT`? Що показав `check_env.py` на "
                 "справжньому провайдері?\n")
    lines.append("- Які види збоїв повторюються, які ні? Що сталося, коли "
                 "`retry_after` не вмістився в бюджет?\n")
    lines.append("- Що робили з уже показаним текстом під час обриву?\n")
    lines.append("- Скільки спроб і часу коштує чернетка в найгіршому "
                 "випадку? Чи витримає це оператор?\n")
    lines.append("- Що оператор бачив у кожному стані? Чи були моменти, "
                 "коли він не знав, чекати чи натискати?\n")
    lines.append("- Чим запасний варіант відрізняється від підставного "
                 "провайдера?\n")
    lines.append("- Що робить обмеження частоти, чого не робить?\n")
    lines.append("- Що сталося з 5 одночасними замовленнями?\n")
    lines.append("- Який сценарій застосунок пройшов найгірше?\n")

    FINDINGS_FILE.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())