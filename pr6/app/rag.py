"""Конвеєр відповіді за базою знань.

Рішення:

* якщо `retrieve` не повернув нічого — модель НЕ викликаємо; повертаємо
  готову відмову. Це детерміновано, дешево й не залежить від настрою
  моделі;
* після відповіді моделі перевіряємо кодом:
  - усі `ref` у `sources` існують серед показаних фрагментів;
  - якщо `found=true`, а `sources` порожній — перезаписуємо на `false`;
  - неіснуючі номери — викидаємо з `sources`;
* у `sources` (що показуємо клієнту) — лише ті фрагменти, на які
  послалася модель;
* у `retrieved` (для відладки) — усі, які бачила модель;
* `elapsed` — окремо `retrieval` і `generation`.
"""

import time
from dataclasses import dataclass, field

from . import llm
from .index import Hit, SearchIndex
from .keyword import KeywordIndex
from .retrieval import Source, build_context, retrieve


@dataclass
class Answer:
    text: str
    found: bool
    sources: list[Source] = field(default_factory=list)
    retrieved: list[Hit] = field(default_factory=list)
    model: str | None = None
    elapsed: dict = field(default_factory=dict)
    usage: dict | None = None


_FALLBACK_TEXT = (
    "На жаль, у базі знань магазину немає інформації, яка відповідає "
    "на це питання. Спробуйте переформулювати або зверніться до "
    "оператора підтримки."
)


def answer(
    question: str,
    index: SearchIndex,
    keyword_index: KeywordIndex | None,
    filters: dict | None = None,
) -> Answer:
    """Повний конвеєр: знайти → відібрати → зібрати → спитати → перевірити."""
    # 1. Пошук і відбір
    t0 = time.perf_counter()
    hits = retrieve(question, index, keyword_index, filters=filters)
    retrieval_elapsed = time.perf_counter() - t0

    # 2. Якщо нічого — не викликаємо модель
    if not hits:
        return Answer(
            text=_FALLBACK_TEXT,
            found=False,
            sources=[],
            retrieved=[],
            model=None,
            elapsed={"retrieval": round(retrieval_elapsed, 3), "generation": 0.0},
            usage=None,
        )

    # 3. Контекст
    context, sources = build_context(hits)
    if not context:
        # Бюджет виявився замалим — теж відмова
        return Answer(
            text=_FALLBACK_TEXT,
            found=False,
            sources=[],
            retrieved=hits,
            model=None,
            elapsed={"retrieval": round(retrieval_elapsed, 3), "generation": 0.0},
            usage=None,
        )

    # 4. Модель
    t1 = time.perf_counter()
    try:
        result = llm.ask(question, context)
    except llm.LLMError:
        raise
    generation_elapsed = time.perf_counter() - t1

    reply = result["reply"]
    answer_text: str = reply.get("answer", "").strip()
    found_flag: bool = bool(reply.get("found"))
    raw_refs: list[int] = list(reply.get("sources") or [])

    # 5. Перевірка кодом
    valid_refs = {s.ref for s in sources}
    # Неіснуючі номери викидаємо
    checked_refs = [r for r in raw_refs if r in valid_refs]
    if len(checked_refs) != len(raw_refs):
        print(f"[rag] Модель послалася на неіснуючі джерела: "
              f"{set(raw_refs) - valid_refs}")

    # «found=true» без жодного джерела — підозріло, перезаписуємо
    if found_flag and not checked_refs:
        print("[rag] found=true, але джерел немає — перезаписую на false")
        found_flag = False
        if not answer_text:
            answer_text = _FALLBACK_TEXT

    # «found=false», але з джерелами — суперечність, лишаємо як є,
    # але джерела відкидаємо: якщо не знайшов, посилатися нема на що
    if not found_flag:
        checked_refs = []

    # 6. Sources для клієнта — лише процитовані
    cited: list[Source] = [s for s in sources if s.ref in checked_refs]

    return Answer(
        text=answer_text or _FALLBACK_TEXT,
        found=found_flag,
        sources=cited,
        retrieved=hits,
        model=result["model"],
        elapsed={
            "retrieval": round(retrieval_elapsed, 3),
            "generation": round(generation_elapsed, 3),
        },
        usage=result.get("usage"),
    )