"""Денний ліміт витрат: скільки застосунок може витратити за добу.

Стан на диску в `state/budget.json`: {"day": "YYYY-MM-DD", "spent": float}.
Переживає перезапуск. Межа доби — локальний час.

Дві лінії:
  • м'яка — на частці WARN_AT: журнал отримує WARNING;
  • жорстка — після вичерпання: BudgetExceeded.

check() викликається перед кожним платним викликом;
record(cost) — після кожного (включно з невдалими).
"""

import datetime
import json
import logging
import os
import tempfile
import threading
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("app.budget")

DAILY_USD = float(os.getenv("BUDGET_DAILY_USD", "1.00"))
WARN_AT = float(os.getenv("BUDGET_WARN_AT", "0.8"))
BUDGET_FILE = os.getenv("BUDGET_FILE", "state/budget.json")

_lock = threading.Lock()
_warned_today = False

class BudgetExceeded(Exception):
    def __init__(self, spent: float, limit: float, resets_at: str) -> None:
        super().__init__(
            f"денний ліміт {limit:.4f} $ вичерпано "
            f"(витрачено {spent:.4f} $); відновиться {resets_at}"
        )
        self.spent = spent
        self.limit = limit
        self.resets_at = resets_at

def settings() -> dict:
    return {"daily_usd": DAILY_USD, "warn_at": WARN_AT, "file": BUDGET_FILE}


def _today() -> str:
    return datetime.date.today().isoformat()


def _resets_at() -> str:
    tomorrow = datetime.date.today() + datetime.timedelta(days=1)
    midnight = datetime.datetime.combine(tomorrow, datetime.time.min)
    return midnight.isoformat()


def _read() -> dict:
    p = Path(BUDGET_FILE)
    if not p.exists():
        return {"day": _today(), "spent": 0.0}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return {"day": _today(), "spent": 0.0}
        if data.get("day") != _today():
            return {"day": _today(), "spent": 0.0}
        return {"day": data["day"], "spent": float(data.get("spent", 0.0))}
    except (OSError, ValueError, TypeError):
        return {"day": _today(), "spent": 0.0}


def _write(state: dict) -> None:
    p = Path(BUDGET_FILE)
    p.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=p.parent, prefix=".budget-", suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(state, f)
        os.replace(tmp, p)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass


def check(expected_cost: float | None = None) -> None:
    """Підняти BudgetExceeded, якщо денний ліміт вичерпано.

    `expected_cost` — резерв (не використовуємо, але лишаємо в сигнатурі).
    """
    global _warned_today
    with _lock:
        state = _read()
        spent = state["spent"]
        if DAILY_USD > 0 and spent >= DAILY_USD:
            raise BudgetExceeded(spent=spent, limit=DAILY_USD, resets_at=_resets_at())
        # М'яка лінія — попередження
        if DAILY_USD > 0 and not _warned_today and spent >= DAILY_USD * WARN_AT:
            _warned_today = True
            logger.warning(
                "budget.warning",
                extra={"spent": round(spent, 6), "limit": DAILY_USD,
                       "threshold": round(DAILY_USD * WARN_AT, 6)},
            )


def record(cost: float | None) -> None:
    """Зарахувати вартість до витрат дня. None → 0."""
    if cost is None or cost <= 0:
        return
    with _lock:
        state = _read()
        state["spent"] = round(state["spent"] + float(cost), 6)
        _write(state)
        spent = state["spent"]
    # М'яка лінія після запису (якщо ще не попереджали)
    global _warned_today
    if DAILY_USD > 0 and not _warned_today and spent >= DAILY_USD * WARN_AT:
        _warned_today = True
        logger.warning(
            "budget.warning",
            extra={"spent": round(spent, 6), "limit": DAILY_USD,
                   "threshold": round(DAILY_USD * WARN_AT, 6)},
        )


def status() -> dict:
    with _lock:
        state = _read()
        spent = state["spent"]
    limit = DAILY_USD
    remaining = max(0.0, limit - spent) if limit > 0 else 0.0
    return {
        "day": _today(),
        "spent": round(spent, 6),
        "limit": limit,
        "remaining": round(remaining, 6),
        "warning": limit > 0 and spent >= limit * WARN_AT and spent < limit,
        "exceeded": limit > 0 and spent >= limit,
        "resets_at": _resets_at(),
    }