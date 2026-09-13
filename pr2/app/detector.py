"""Модуль inference: єдине місце застосунку, яке знає про модель."""

import io
import time
from typing import Any, Dict, List, Optional
from PIL import Image
from ultralytics import YOLO

WEIGHTS = "yolov8n.pt"
DEFAULT_CONFIDENCE = 0.25

# Глобальна змінна для кешування моделі (синглтон)
_model_instance: Optional[YOLO] = None


class DetectionError(Exception):
    """Базовий виняток для помилок детекції."""

    pass


class InvalidImageError(DetectionError):
    """Виникає, коли переданий файл не є валидним зображенням або порожній."""

    pass


class ModelInferenceError(DetectionError):
    """Виникає при внутрішніх збоях виконання моделі."""

    pass


def load_model() -> YOLO:
    """Повернути готову до роботи модель.

    Використовує паттерн Singleton: завантажує ваги лише один раз у пам'ять,
    а при наступних викликах повертає вже існуючий екземпляр.
    """
    global _model_instance
    if _model_instance is None:
        _model_instance = YOLO(WEIGHTS)
    return _model_instance


def detect(
    image_bytes: bytes, confidence: float = DEFAULT_CONFIDENCE
) -> Dict[str, Any]:
    """Знайти обʼєкти на зображенні.

    Приймає байти зображення та поріг упевненості.
    Повертає структурований JSON-сумісний словник.
    """
    if not image_bytes or len(image_bytes) == 0:
        raise InvalidImageError("Передано порожній файл.")

    # Валідація зображення за допомогою Pillow
    try:
        img = Image.open(io.BytesIO(image_bytes))
        img.verify()  # Перевірка цілісності структури файлу
        # verify() закриває/псує об'єкт Image, тому для подальшої роботи відкриваємо заново
        img = Image.open(io.BytesIO(image_bytes))
    except Exception as exc:
        raise InvalidImageError(
            "Наданий файл не є коректним зображенням."
        ) from exc

    model = load_model()

    # Вимірюємо час виконання inference
    start_time = time.perf_counter()
    try:
        results = model.predict(source=img, conf=confidence, verbose=False)
    except Exception as exc:
        raise ModelInferenceError(f"Помилка під час інференсу: {exc}") from exc
    inference_time = time.perf_counter() - start_time

    objects: List[Dict[str, Any]] = []
    first_result = results[0]

    if first_result.boxes is not None:
        for box in first_result.boxes:
            class_id = int(box.cls[0].item())
            class_name = first_result.names.get(class_id, str(class_id))
            conf = float(box.conf[0].item())

            # Отримуємо bounding box у пікселях [xmin, ymin, xmax, ymax]
            coords = box.xyxy[0].tolist()

            objects.append(
                {
                    "class": class_name,
                    "confidence": round(conf, 4),
                    "bbox": [round(c, 2) for c in coords],
                }
            )

    return {
        "count": len(objects),
        "inference_time_seconds": round(inference_time, 4),
        "confidence_threshold": confidence,
        "objects": objects,
    }