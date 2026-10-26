"""Журнал застосунку: структуровані записи й ідентифікатор запиту.

Тут дано готовим плумбінг, який не є предметом роботи:

* `request_id_var` — контекстна змінна з ідентифікатором поточного
  HTTP-запиту. Проміжний шар у `app/main.py` ставить її на вході й
  повертає значення клієнтові в заголовку `X-Request-ID`;
* `JsonFormatter` — кожен запис журналу стає одним рядком JSON: час,
  рівень, назва логера, подія (повідомлення) і всі поля, передані через
  `extra={...}`; ідентифікатор запиту підставляється з контексту, якщо
  його не передали явно;
* `setup()` — журнал у файл `LOG_FILE` (формат JSON lines) і коротко
  в консоль; рівень — `LOG_LEVEL`.

ПР11: `setup()` приглушує технічні логи сторонніх бібліотек
(`httpx`, `httpx2`, `httpcore`, `httpcore2`, `openai`, `urllib3`), щоб
у `logs/app.jsonl` були лише події застосунку. Різні версії SDK
реєструють різні імена логерів, тому перевіряємо ПІДРЯДОК у назві —
це працює навіть якщо SDK перейменує логер.

Як записувати:

    import logging
    log = logging.getLogger("app.gateway")
    log.info("llm.attempt", extra={"draft_id": d, "attempt": 1})

Подія — у повідомленні, подробиці — в `extra`; у рядок JSON вони
потрапляють як є. Не передавайте в `extra` ключі, які є в самого
`LogRecord` (`name`, `msg`, `args`, `levelname`, `message`, …).
"""

import contextvars
import datetime
import json
import logging
import os
import sys
import uuid
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

LOG_FILE = Path(os.getenv("LOG_FILE", "logs/app.jsonl"))
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

# Ідентифікатор поточного HTTP-запиту; ставить проміжний шар у main.py.
request_id_var: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "request_id", default=None)


def new_request_id() -> str:
    return uuid.uuid4().hex[:12]


def settings() -> dict:
    return {"log_file": LOG_FILE.as_posix(), "log_level": LOG_LEVEL}


# Поля, які є в кожному LogRecord; усе інше в record.__dict__ — це `extra`.
_STANDARD = set(
    logging.LogRecord("", 0, "", 0, "", (), None).__dict__
) | {"message", "asctime"}


def _extra(record: logging.LogRecord) -> dict:
    return {k: v for k, v in record.__dict__.items()
            if k not in _STANDARD and not k.startswith("_")}


class JsonFormatter(logging.Formatter):
    """Один запис — один рядок JSON."""

    def format(self, record: logging.LogRecord) -> str:
        data = {
            "ts": datetime.datetime.now(datetime.timezone.utc).isoformat(
                timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "event": record.getMessage(),
        }
        extra = _extra(record)
        request_id = extra.pop("request_id", None) or request_id_var.get()
        if request_id:
            data["request_id"] = request_id
        data.update(extra)
        if record.exc_info:
            data["exc"] = self.formatException(record.exc_info)
        return json.dumps(data, ensure_ascii=False, default=str)


class ConsoleFormatter(logging.Formatter):
    """Коротко для консолі: час, рівень, подія, поля."""

    def format(self, record: logging.LogRecord) -> str:
        stamp = datetime.datetime.now().strftime("%H:%M:%S")
        fields = " ".join(f"{k}={v}" for k, v in _extra(record).items())
        line = (f"{stamp} {record.levelname:<7} "
                f"{record.getMessage()}  {fields}").rstrip()
        if record.exc_info:
            line += "\n" + self.formatException(record.exc_info)
        return line


# Підрядки в назвах логерів, які треба приглушити. Різні версії SDK
# реєструють різні імена: httpx / httpx2, httpcore / httpcore2.
_NOISY_SUBSTRINGS = ("httpx", "httpcore", "urllib3", "openai._base_client",
                     "openai.resources", "openai.api_resources")


def _silence_noisy_loggers() -> None:
    """Приглушити технічні логи сторонніх бібліотек до WARNING.

    Ітеруємо всі зареєстровані логери й перевіряємо підрядок у назві —
    так працює навіть якщо SDK перейменує свій логер (наприклад,
    `httpx` → `httpx2`).
    """
    manager = logging.root.manager
    for name in list(manager.loggerDict.keys()):
        low = name.lower()
        if any(sub in low for sub in _NOISY_SUBSTRINGS):
            logging.getLogger(name).setLevel(logging.WARNING)


def setup() -> None:
    """Налаштувати кореневий логер: файл JSON lines + консоль.

    Повторний виклик (наприклад, під `--reload`) нічого не додає.
    """
    root = logging.getLogger()
    if getattr(root, "_pr11_configured", False):
        return
    root._pr11_configured = True  # type: ignore[attr-defined]
    root.setLevel(LOG_LEVEL)

    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
    file_handler.setFormatter(JsonFormatter())
    root.addHandler(file_handler)

    console = logging.StreamHandler(sys.stderr)
    console.setFormatter(ConsoleFormatter())
    root.addHandler(console)

    # Технічні логи сторонніх бібліотек — до WARNING, щоб у журналі
    # були лише події застосунку.
    _silence_noisy_loggers()

    # Журнал доступу uvicorn — текстовий і дублює запис `http.request`
    # з проміжного шару; вимикаємо.
    logging.getLogger("uvicorn.access").disabled = True