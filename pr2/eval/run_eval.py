"""Скрипт оцінювання якості детектора / Evaluation script."""

import json
import sys
from collections import Counter
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from app import detector


def run_evaluation():
    eval_dir = Path(__file__).parent
    config_path = eval_dir / "expected.json"

    if not config_path.exists():
        print(f"❌ Config file not found / Файл конфігурації не знайдено: {config_path}")
        return

    print("⏳ Loading YOLO model / Завантаження моделі YOLO...")
    detector.load_model()
    print("✅ Model loaded / Модель завантажено.\n")

    config = json.loads(config_path.read_text(encoding="utf-8"))
    images_data = config.get("images", [])

    total_expected_count = 0
    total_found_count = 0
    total_correct_count = 0
    total_inference_time = 0.0

    print("=" * 70)
    print(f"🚀 RUNNING EVALUATION / ЗАПУСК ОЦІНЮВАННЯ ({len(images_data)} files)")
    print("=" * 70 + "\n")

    for idx, item in enumerate(images_data, 1):
        rel_path = item["file"]
        img_path = eval_dir / rel_path
        expected_classes = item.get("expected", [])
        note = item.get("note", "")

        print(f"[{idx}/{len(images_data)}] File / Файл: {rel_path}")
        if note:
            print(f"    Note / Примітка: {note}")

        if not img_path.exists():
            print(f"    ❌ File does not exist / Файл не знайдено: {img_path}\n")
            continue

        img_bytes = img_path.read_bytes()

        try:
            res = detector.detect(img_bytes, confidence=0.25)
        except Exception as e:
            print(f"    ❌ Detection error / Помилка детекції: {e}\n")
            continue

        found_objects = res.get("objects", [])
        found_classes = [obj["class"] for obj in found_objects]
        inf_time = res.get("inference_time_seconds", 0.0)

        total_inference_time += inf_time
        total_expected_count += len(expected_classes)
        total_found_count += len(found_classes)

        exp_counter = Counter(expected_classes)
        found_counter = Counter(found_classes)
        correct_counter = exp_counter & found_counter
        total_correct_count += sum(correct_counter.values())

        missing = list((exp_counter - found_counter).elements())
        extra = list((found_counter - exp_counter).elements())

        print(f"    Expected / Очікувалось ({len(expected_classes)}): {expected_classes}")
        print(f"    Detected / Знайдено    ({len(found_classes)}): {found_classes}")
        print(f"    Inference time / Час: {inf_time:.4f}s")

        if not missing and not extra:
            print("    Result / Результат: ✅ PERFECT / ІДЕАЛЬНО")
        else:
            details = []
            if missing: details.append(f"Missing/Пропущено: {missing}")
            if extra: details.append(f"Extra/Зайве: {extra}")
            print("    Result / Результат: ⚠️ MISMATCH -> " + "; ".join(details))

        print("-" * 70)

    avg_time = total_inference_time / len(images_data) if images_data else 0.0
    precision = (total_correct_count / total_found_count * 100) if total_found_count > 0 else 0.0
    recall = (total_correct_count / total_expected_count * 100) if total_expected_count > 0 else 0.0

    print("\n" + "=" * 70)
    print("📊 EVALUATION REPORT / ПІДСУМКОВИЙ ЗВІТ")
    print("=" * 70)
    print(f"Total test images / Тестових зображень: {len(images_data)}")
    print(f"Expected objects / Очікувалось об'єктів: {total_expected_count}")
    print(f"Detected objects / Виявлено об'єктів:   {total_found_count}")
    print(f"Correctly detected / Правильно виявлено: {total_correct_count}")
    print(f"Precision / Точність:                     {precision:.1f}%")
    print(f"Recall / Повнота:                        {recall:.1f}%")
    print(f"Avg inference time / Середній час:       {avg_time:.4f}s")
    print("=" * 70)


if __name__ == "__main__":
    run_evaluation()