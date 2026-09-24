"""Модуль ембедінгів: єдине місце застосунку, яке знає, яка модель
перетворює текст на вектор і як її викликати.

Рішення, закладені тут:

* модель — `intfloat/multilingual-e5-small` (багатомовна, 384 виміри,
  працює на CPU, ~120 МБ); назва читається з `.env`;
* модель створюється один раз (`_model` — кеш) і перевикористовується;
* префікси `query: ` і `passage: ` — вимога картки e5; тут вони
  застосовуються в окремих функціях `embed_query` і `embed_passages`;
* вектори нормалізуються одразу в `encode`, щоб косинус був скалярним
  добутком;
* фрагменти кодуються пакетами — для сотень фрагментів це помітно
  швидше за цикл по одному.
"""

import os

import numpy as np
from dotenv import load_dotenv

load_dotenv()

MODEL_NAME = os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-small")

# Чи потрібні префікси — визначаємо за назвою моделі. e5 вимагає їх;
# для інших моделей залишаємо порожніми. Якщо зміните модель — перевірте
# її картку на Hugging Face.
_NEEDS_PREFIX = "e5" in MODEL_NAME.lower()
QUERY_PREFIX = "query: " if _NEEDS_PREFIX else ""
PASSAGE_PREFIX = "passage: " if _NEEDS_PREFIX else ""

# Розмір пакета для кодування фрагментів — компроміс між швидкістю
# і пам'яттю. Для 384-вимірної моделі 32 цілком безпечно.
BATCH_SIZE = 32

_model = None


def get_model():
    """Повернути готову до роботи модель. Створюється один раз."""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def embed_passages(texts: list[str]) -> np.ndarray:
    """Перетворити тексти фрагментів на вектори.

    Повертає масив (N × d), нормалізований, готовий до косинусного
    порівняння через скалярний добуток.
    """
    if not texts:
        return np.zeros((0, 384), dtype=np.float32)
    model = get_model()
    prepared = [PASSAGE_PREFIX + t for t in texts]
    vectors = model.encode(
        prepared,
        batch_size=BATCH_SIZE,
        normalize_embeddings=True,
        show_progress_bar=False,
        convert_to_numpy=True,
    )
    return np.asarray(vectors, dtype=np.float32)


def embed_query(text: str) -> np.ndarray:
    """Перетворити запит користувача на вектор тієї самої розмірності.

    Окрема функція навмисно: у моделей із префіксами запит кодується не
    так, як фрагмент. Повертає 1-D вектор (d,).
    """
    text = (text or "").strip()
    if not text:
        raise ValueError("Запит не може бути порожнім.")
    model = get_model()
    vector = model.encode(
        [QUERY_PREFIX + text],
        normalize_embeddings=True,
        show_progress_bar=False,
        convert_to_numpy=True,
    )
    return np.asarray(vector[0], dtype=np.float32)