"""Перевірка середовища для ПР2."""

import importlib
import sys
import time

MIN_PYTHON = (3, 10)
PACKAGES = ("ultralytics", "fastapi", "uvicorn", "multipart")
WEIGHTS = "yolov8n.pt"


def check_python() -> bool:
    actual = sys.version_info[:2]
    ok = actual >= MIN_PYTHON
    need = ".".join(map(str, MIN_PYTHON))
    have = ".".join(map(str, actual))
    report(ok, f"Python {have}", "" if ok else f"потрібен Python {need} або новіший")
    return ok


def check_packages() -> bool:
    ok = True
    for name in PACKAGES:
        try:
            importlib.import_module(name)
        except ImportError:
            pkg_hint = "python-multipart" if name == "multipart" else name
            report(False, f"пакет {pkg_hint}", "не встановлено: pip install -r requirements.txt")
            ok = False
        else:
            pkg_label = "python-multipart" if name == "multipart" else name
            report(True, f"пакет {pkg_label}", "")
    return ok


def check_weights() -> bool:
    try:
        from ultralytics import YOLO
    except ImportError:
        report(False, f"ваги {WEIGHTS}", "перевірку пропущено: немає пакета ultralytics")
        return False

    print(f"       завантаження {WEIGHTS} — перший раз може тривати хвилину…")
    started = time.perf_counter()
    try:
        YOLO(WEIGHTS)
    except Exception as exc:
        report(False, f"ваги {WEIGHTS}", f"не вдалося завантажити ({type(exc).__name__}: {exc})")
        return False
    elapsed = time.perf_counter() - started
    report(True, f"ваги {WEIGHTS}", f"готові за {elapsed:.1f} с")
    return True


def report(ok: bool, what: str, hint: str) -> None:
    mark = "[ OK ]" if ok else "[ !! ]"
    tail = f" — {hint}" if hint else ""
    print(f"{mark} {what}{tail}")


def main() -> int:
    print("Перевірка середовища для ПР2\n")
    results = [check_python(), check_packages(), check_weights()]
    print()
    if all(results):
        print("Середовище готове до роботи.")
        return 0
    print("Є проблеми — усуньте позначені [ !! ] і запустіть перевірку ще раз.")
    return 1


if __name__ == "__main__":
    sys.exit(main())