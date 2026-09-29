"""Підготовка зображення до надсилання моделі.

Рішення:

* перевіряємо, що це зображення (Pillow відкриває); інакше ImageError;
* максимальний розмір файлу — `UPLOAD_MAX_MB` (10 МБ за замовчуванням);
* максимальна довга сторона — `IMAGE_MAX_SIDE` (1600 пікселів);
* мінімальна довга сторона — 400 пікселів: менше — цифри нерозбірливі,
  краще попросити перезняти;
* орієнтація з EXIF — застосовуємо `ImageOps.exif_transpose` (фото з
  телефона часто «боком» у пікселях);
* формат: PNG без втрат; якщо зменшуємо — все одно PNG (для тексту
  краще за JPEG);
* `PIL.Image.MAX_IMAGE_PIXELS` захищає від decompression bomb.
"""

import io
import os
from dataclasses import dataclass

from dotenv import load_dotenv
from PIL import Image, ImageOps, UnidentifiedImageError

load_dotenv()

UPLOAD_MAX_BYTES = int(float(os.getenv("UPLOAD_MAX_MB", "10")) * 1024 * 1024)
IMAGE_MAX_SIDE = int(os.getenv("IMAGE_MAX_SIDE", "1600"))
IMAGE_MIN_SIDE = int(os.getenv("IMAGE_MIN_SIDE", "400"))


class ImageError(Exception):
    """Файл не можна віддати моделі. Повідомлення — для користувача."""


@dataclass
class PreparedImage:
    """Зображення, готове до надсилання."""
    data: bytes
    mime: str
    original: dict
    sent: dict


def prepare(content: bytes) -> PreparedImage:
    """Перевірити байти й підготувати зображення для моделі."""
    if not content:
        raise ImageError("Порожній файл.")

    if len(content) > UPLOAD_MAX_BYTES:
        mb = len(content) / 1024 / 1024
        raise ImageError(
            f"Файл завеликий: {mb:.1f} МБ. Максимум — "
            f"{UPLOAD_MAX_BYTES / 1024 / 1024:.0f} МБ."
        )

    try:
        img = Image.open(io.BytesIO(content))
        img.load()  # force decode — тут і виявиться, чи це справді зображення
    except UnidentifiedImageError as e:
        raise ImageError("Це не зображення або формат не підтримується.") from e
    except OSError as e:
        raise ImageError(f"Не вдалося прочитати зображення: {e}") from e

    original_w, original_h = img.size
    original_bytes = len(content)

    # EXIF-орієнтація — фото з телефона часто «боком»
    img = ImageOps.exif_transpose(img)

    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")

    # Зменшення за довгою стороною
    w, h = img.size
    long_side = max(w, h)
    if long_side > IMAGE_MAX_SIDE:
        scale = IMAGE_MAX_SIDE / long_side
        img = img.resize(
            (max(1, int(w * scale)), max(1, int(h * scale))),
            Image.LANCZOS,
        )

    # Перевірка мінімального розміру — після зменшення
    w, h = img.size
    if max(w, h) < IMAGE_MIN_SIDE:
        raise ImageError(
            f"Зображення замале: {w}×{h}. Потрібно щонайменше "
            f"{IMAGE_MIN_SIDE} пікселів за довшою стороною. "
            f"Перезніміть документ."
        )

    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    data = buf.getvalue()

    return PreparedImage(
        data=data,
        mime="image/png",
        original={
            "width": original_w,
            "height": original_h,
            "bytes": original_bytes,
        },
        sent={
            "width": w,
            "height": h,
            "bytes": len(data),
        },
    )