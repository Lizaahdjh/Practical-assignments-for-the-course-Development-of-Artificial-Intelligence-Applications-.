"""Векторний індекс: зберігання векторів фрагментів і пошук найближчих.

Рішення, закладені тут:

* схожість — косинус; вектори нормалізовані, тож це скалярний добуток
  матриці на вектор — одна операція на весь індекс;
* фільтри за метаданими застосовуються **до** ранжування: top-k
  рахується серед дозволених фрагментів;
* індекс зберігається як `vectors.npy` (float32) і `chunks.json`
  (фрагменти з метаданими) плюс `meta.json` із назвою моделі й версією;
* поріг схожості — з конфігурації; порожній означає «без порога».
"""

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

from .documents import Chunk

load_dotenv()

INDEX_DIR = Path(__file__).parent.parent / "index"
INDEX_VERSION = 1

DEFAULT_TOP_K = int(os.getenv("SEARCH_TOP_K", "5"))
_threshold = os.getenv("SIMILARITY_THRESHOLD", "").strip()
SIMILARITY_THRESHOLD: float | None = float(_threshold) if _threshold else None


@dataclass
class Hit:
    """Одне влучення пошуку: фрагмент і оцінка його схожості із запитом."""

    chunk: Chunk
    score: float


@dataclass
class SearchIndex:
    """Індекс у памʼяті."""

    chunks: list[Chunk]
    vectors: np.ndarray
    model_name: str
    extra: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.chunks)


# ---------------------------------------------------------------------------
# Побудова
# ---------------------------------------------------------------------------
def build(chunks: list[Chunk], vectors: np.ndarray, model_name: str) -> SearchIndex:
    """Зібрати індекс із фрагментів і їхніх векторів.

    Перевіряє, що векторів стільки ж, скільки фрагментів, і приводить
    вектори до float32. Нормалізація — на боці `embeddings`: сюди
    приходять уже нормалізовані.
    """
    vectors = np.asarray(vectors, dtype=np.float32)
    if vectors.ndim != 2:
        raise ValueError(f"Очікували 2-D масив векторів, отримали {vectors.shape}")
    if vectors.shape[0] != len(chunks):
        raise ValueError(
            f"Кількість векторів ({vectors.shape[0]}) не збігається "
            f"з кількістю фрагментів ({len(chunks)})"
        )
    return SearchIndex(chunks=list(chunks), vectors=vectors, model_name=model_name)


# ---------------------------------------------------------------------------
# Збереження й читання
# ---------------------------------------------------------------------------
def save(index_obj: SearchIndex, path: Path = INDEX_DIR) -> None:
    """Зберегти індекс на диск: вектори, фрагменти, метадані."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    np.save(path / "vectors.npy", index_obj.vectors)

    chunks_payload = [
        {"text": c.text, "source": c.source, "metadata": c.metadata}
        for c in index_obj.chunks
    ]
    (path / "chunks.json").write_text(
        json.dumps(chunks_payload, ensure_ascii=False),
        encoding="utf-8",
    )

    meta = {
        "version": INDEX_VERSION,
        "model_name": index_obj.model_name,
        "count": len(index_obj.chunks),
        "dim": int(index_obj.vectors.shape[1]) if index_obj.vectors.size else 0,
    }
    (path / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def load(path: Path = INDEX_DIR) -> SearchIndex:
    """Прочитати індекс із диска.

    Якщо індексу немає — підіймає `FileNotFoundError` із зрозумілим
    повідомленням: веб-рівень має сказати користувачеві «індекс не
    збудовано», а не впасти глибоко всередині.
    """
    path = Path(path)
    vectors_file = path / "vectors.npy"
    chunks_file = path / "chunks.json"
    meta_file = path / "meta.json"

    if not (vectors_file.exists() and chunks_file.exists() and meta_file.exists()):
        raise FileNotFoundError(
            "Індекс не знайдено. Виконайте `python ingest.py`, щоб його збудувати."
        )

    meta = json.loads(meta_file.read_text(encoding="utf-8"))
    if meta.get("version") != INDEX_VERSION:
        raise RuntimeError(
            f"Версія індексу {meta.get('version')} не підтримується "
            f"(очікується {INDEX_VERSION}). Перебудуйте: `python ingest.py`."
        )

    vectors = np.load(vectors_file)
    chunks_payload = json.loads(chunks_file.read_text(encoding="utf-8"))
    chunks = [
        Chunk(text=c["text"], source=c["source"], metadata=c.get("metadata", {}))
        for c in chunks_payload
    ]

    return SearchIndex(
        chunks=chunks,
        vectors=vectors,
        model_name=meta.get("model_name", ""),
        extra={"version": meta.get("version")},
    )


# ---------------------------------------------------------------------------
# Пошук
# ---------------------------------------------------------------------------
def _matches(metadata: dict, filters: dict) -> bool:
    """Чи проходить фрагмент усі умови фільтра.

    Значення метаданих і фільтра порівнюються як рядки (без регістру):
    у файлах — «клієнти», у формі — «клієнти». Порожні значення фільтра
    ігноруються.
    """
    for key, expected in filters.items():
        if expected in (None, "", []):
            continue
        actual = metadata.get(key)
        if actual is None:
            return False
        if str(actual).strip().lower() != str(expected).strip().lower():
            return False
    return True


def search(
    index_obj: SearchIndex,
    query_vector: np.ndarray,
    top_k: int = DEFAULT_TOP_K,
    filters: dict | None = None,
    threshold: float | None = SIMILARITY_THRESHOLD,
) -> list[Hit]:
    """Знайти фрагменти, найближчі до вектора запиту.

    Фільтри застосовуються до ранжування: серед дозволених фрагментів
    рахуються оцінки й беруться top_k. Так `top_k=5` справді повертає
    до 5 результатів, а не «5 узято, 1 лишився».
    """
    if index_obj is None or len(index_obj) == 0:
        return []

    vectors = index_obj.vectors
    chunks = index_obj.chunks

    # Кандидати — усі, хто проходить фільтри
    if filters:
        allowed = np.array(
            [_matches(c.metadata, filters) for c in chunks],
            dtype=bool,
        )
    else:
        allowed = np.ones(len(chunks), dtype=bool)

    if not allowed.any():
        return []

    # Косинус = скалярний добуток (вектори нормалізовані)
    scores = vectors @ query_vector.astype(np.float32)  # (N,)
    scores = np.where(allowed, scores, -np.inf)

    # Скільки брати
    k = max(1, int(top_k))
    k = min(k, int(allowed.sum()))

    # argpartition — O(N), потім сортуємо лише k відібраних
    if k < len(scores):
        idx_part = np.argpartition(-scores, k - 1)[:k]
        idx_sorted = idx_part[np.argsort(-scores[idx_part])]
    else:
        idx_sorted = np.argsort(-scores)

    hits: list[Hit] = []
    for i in idx_sorted:
        score = float(scores[i])
        if score == -np.inf:
            continue
        if threshold is not None and score < threshold:
            continue
        hits.append(Hit(chunk=chunks[int(i)], score=score))

    return hits