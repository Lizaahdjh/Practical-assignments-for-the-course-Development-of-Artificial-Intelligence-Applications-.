"""Сховище стану звернень: зберегти, прочитати, перелічити.

Рішення:

* JSON-файл на звернення: `RUNS_DIR/{run_id}.json`;
* запис атомарний: спершу в `{run_id}.json.tmp`, потім `os.replace` —
  збій посередині не лишає зіпсованого файлу;
* `list_runs` читає всі файли й сортує за `updated_at` (нові першими);
* паралелізм — відповідальність workflow: `resume` перевіряє статус
  і робить дозволений перехід; сховище лише зберігає.
"""

import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

from .state import Run, Step

load_dotenv()

RUNS_DIR = Path(os.getenv("RUNS_DIR", "runs"))


def _run_path(run_id: str) -> Path:
    return RUNS_DIR / f"{run_id}.json"


def _run_from_dict(data: dict) -> Run:
    steps = [Step(**s) for s in data.get("steps", [])]
    data = {**data, "steps": steps}
    return Run(**data)


def save(run: Run) -> None:
    """Зберегти стан звернення, замінивши попередній. Атомарно."""
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    payload = asdict(run)
    tmp = _run_path(run.run_id).with_suffix(".json.tmp")
    tmp.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    os.replace(tmp, _run_path(run.run_id))


def load(run_id: str) -> Optional[Run]:
    """Прочитати стан звернення; `None`, якщо такого немає."""
    path = _run_path(run_id)
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return _run_from_dict(data)


def list_runs(status: Optional[str] = None) -> list[Run]:
    """Усі звернення, за потреби — лише з певним статусом; нові першими."""
    if not RUNS_DIR.exists():
        return []
    runs: list[Run] = []
    for path in RUNS_DIR.glob("*.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            run = _run_from_dict(data)
        except Exception:
            continue
        if status is None or run.status == status:
            runs.append(run)
    runs.sort(key=lambda r: r.updated_at or "", reverse=True)
    return runs