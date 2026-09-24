"""Пошук за ключовими словами — точка порівняння для семантичного.

Рішення:

* токенізація однакова для запиту й фрагментів;
* регістр знижується, розділові знаки відкидаються, апостроф
  зберігається;
* морфологію не зводимо: для української готового стемера в
  `rank_bm25` немає; це свідомий компроміс;
* ранжування — BM25 із `rank_bm25`;
* фільтри — ті самі, що й у векторного пошуку, **до** ранжування;
* формат результату — той самий `Hit`, щоб сторінка показувала обидва
  способи поруч.
"""

import re
from dataclasses import dataclass, field

from .documents import Chunk
from .index import DEFAULT_TOP_K, Hit, _matches


@dataclass
class KeywordIndex:
    """Індекс для пошуку за словами: фрагменти + BM25."""

    chunks: list[Chunk]
    bm25: object = None
    chunk_ids: list[int] = field(default_factory=list)
    extra: dict = field(default_factory=dict)


# Літери (включно з кирилицею), цифри, дефіс і апостроф — усе інше
# розділювач. Дефіс потрібен для артикулів (`OR-X2-BLK`), апостроф —
# для слів на кшталт «звʼязок».
_TOKEN_RE = re.compile(r"[0-9a-zA-Zа-яА-ЯіїєґІЇЄҐ'’\-]+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    """Розбити текст на слова для індексування й для запиту.

    Однакова для обох — інакше збігів не буде. Усі токени в нижньому
    регістрі. Порожні токени відкидаються.
    """
    if not text:
        return []
    return [t.lower() for t in _TOKEN_RE.findall(text) if t.strip()]


def build(chunks: list[Chunk]) -> KeywordIndex:
    """Зібрати індекс за словами з тих самих фрагментів, що й векторний."""
    from rank_bm25 import BM25Okapi

    if not chunks:
        return KeywordIndex(chunks=[], bm25=None, chunk_ids=[])

    tokenized = [tokenize(c.text) for c in chunks]
    # BM25Okapi не любить повністю порожніх документів — замінюємо
    tokenized = [toks if toks else ["_порожньо_"] for toks in tokenized]

    bm25 = BM25Okapi(tokenized)
    return KeywordIndex(
        chunks=list(chunks),
        bm25=bm25,
        chunk_ids=list(range(len(chunks))),
    )


def search(
    index_obj: KeywordIndex,
    query: str,
    top_k: int = DEFAULT_TOP_K,
    filters: dict | None = None,
) -> list[Hit]:
    """Знайти фрагменти за словами запиту.

    Фільтри застосовуються **до** ранжування: BM25 рахує оцінки лише
    для фрагментів, що пройшли фільтр. Так `top_k=5` справді повертає
    до 5 результатів.
    """
    if index_obj is None or index_obj.bm25 is None or not index_obj.chunks:
        return []

    tokens = tokenize(query)
    if not tokens:
        return []

    # BM25 рахує оцінки для всіх одразу
    all_scores = index_obj.bm25.get_scores(tokens)

    # Фільтри — до ранжування
    hits: list[Hit] = []
    for i, chunk in enumerate(index_obj.chunks):
        if filters and not _matches(chunk.metadata, filters):
            continue
        score = float(all_scores[i])
        if score <= 0:
            # BM25 дає 0 для фрагментів без спільних слів — це «не знайдено»
            continue
        hits.append(Hit(chunk=chunk, score=score))

    hits.sort(key=lambda h: h.score, reverse=True)
    k = max(1, int(top_k))
    return hits[:k]