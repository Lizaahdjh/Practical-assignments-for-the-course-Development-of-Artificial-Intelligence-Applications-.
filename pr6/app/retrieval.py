"""Відбір фрагментів для моделі та збирання контексту.

Між пошуком із ПР5 і мовною моделлю. Саме тут ухвалюються рішення:

* завжди додаємо `audience=клієнти` і `status=чинний` — користувач не
  може цього обійти, це не фільтр сторінки, а умова в коді;
* викликаємо семантичний пошук з `SEARCH_TOP_K=8` — ширше, ніж
  віддаємо моделі, бо серед топ-5 може не бути потрібного;
* лишаємо `RAG_CONTEXT_CHUNKS=4` найкращих;
* якщо після порога нічого не лишилося — повертаємо порожній список,
  і `rag.answer` не викликає модель;
* дублікати (той самий документ + заголовок) не згортаємо — сусідні
  розділи однієї інструкції часто корисні;
* контекст: `[N] Title › Heading (updated):\n<текст>` — модель має
  бачити, звідки фрагмент, і могла послатися на номер;
* бюджет `RAG_CONTEXT_BUDGET` — з ПР4; якщо фрагменти не вміщаються,
  відкидаємо з кінця (там гірші за оцінкою).
"""

import os
from dataclasses import dataclass

from dotenv import load_dotenv

from .documents import Chunk
from .index import Hit, SearchIndex, SIMILARITY_THRESHOLD, search as vector_search
from .keyword import KeywordIndex

load_dotenv()

CONTEXT_CHUNKS = int(os.getenv("RAG_CONTEXT_CHUNKS", "4"))
CONTEXT_BUDGET = int(os.getenv("RAG_CONTEXT_BUDGET", "1500"))
SEARCH_TOP_K = int(os.getenv("SEARCH_TOP_K", "8"))

# Оцінка токенів: для української близько 3 символів на токен.
_CHARS_PER_TOKEN = 3.0

# Фільтри, які код накладає завжди — користувач їх не бачить і не може
# вимкнути. Це межа безпеки: внутрішнє й архівне не потрапляє до моделі.
MANDATORY_FILTERS = {
    "audience": "клієнти",
    "status": "чинний",
}


@dataclass
class Source:
    """Джерело, показане моделі й користувачеві."""
    ref: int
    chunk: Chunk
    score: float


def _estimate_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, int(len(text) / _CHARS_PER_TOKEN))


def retrieve(
    query: str,
    index: SearchIndex,
    keyword_index: KeywordIndex | None,
    filters: dict | None = None,
) -> list[Hit]:
    """Знайти фрагменти, які варто показати моделі.

    Фільтри користувача об'єднуються з обов'язковими; обов'язкові мають
    пріоритет.
    """
    combined: dict = dict(filters or {})
    for key, value in MANDATORY_FILTERS.items():
        combined[key] = value  # перезаписуємо, якщо користувач намагався

    # Семантичний пошук
    from . import embeddings
    vector = embeddings.embed_query(query)
    hits = vector_search(
        index,
        vector,
        top_k=SEARCH_TOP_K,
        filters=combined,
        threshold=SIMILARITY_THRESHOLD,
    )

    # Обрізаємо до CONTEXT_CHUNKS
    hits = hits[:CONTEXT_CHUNKS]
    return hits


def build_context(
    hits: list[Hit],
    budget: int = CONTEXT_BUDGET,
) -> tuple[str, list[Source]]:
    """Зібрати текст контексту й перелік джерел.

    Формат позначки: `[N] Title › Heading (updated)`. Позначка
    допомагає моделі посилатися й людині розуміти, звідки фрагмент.
    Текст — дослівно.
    """
    if not hits:
        return "", []

    sources: list[Source] = []
    parts: list[str] = []
    used_tokens = 0

    for i, hit in enumerate(hits, start=1):
        meta = hit.chunk.metadata
        title = meta.get("title") or hit.chunk.source
        heading = meta.get("heading")
        updated = meta.get("updated")

        header_bits = [f"[{i}] {title}"]
        if heading:
            header_bits.append(f"› {heading}")
        if updated:
            header_bits.append(f"({updated})")
        header = " ".join(header_bits)

        body = f"{header}\n{hit.chunk.text}"

        est = _estimate_tokens(body)
        if used_tokens + est > budget:
            # Бюджет вичерпано — далі не додаємо
            break

        parts.append(body)
        sources.append(Source(ref=i, chunk=hit.chunk, score=hit.score))
        used_tokens += est

    context = "\n\n---\n\n".join(parts)
    return context, sources