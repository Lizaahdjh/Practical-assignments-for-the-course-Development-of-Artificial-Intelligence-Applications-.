"""Колекція документів: читання, метадані, поділ на фрагменти.

Це єдине місце, яке знає, як влаштовані файли в `docs/`: де в них
метадані, як розмічено текст, за якими межами його ділити. Решта
застосунку працює з готовими фрагментами (`Chunk`) і не читає файлів.
"""

import os
import re
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

DOCS_DIR = Path(__file__).parent.parent / "docs"

# Параметри поділу — з конфігурації. CHUNK_SIZE у символах:
# для e5-small максимум 512 токенів, 600 символів ≈ 200 токенів —
# із запасом. Перекриття рятує думку, розрізану межею.
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "600"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))


@dataclass
class Chunk:
    """Фрагмент документа — одиниця індексування й пошуку.

    `text` — те, що перетворюється на вектор і показується в результатах.
    `source` — імʼя файлу, з якого взято фрагмент.
    `metadata` — поля з блоку метаданих файлу (title, category, product,
    audience, updated, status) плюс `heading` і `chunk_index`, які додає
    `split`. Фільтри пошуку працюють саме з цим словником.
    """

    text: str
    source: str
    metadata: dict = field(default_factory=dict)


def parse_front_matter(raw: str) -> tuple[dict, str]:
    """Відокремити блок метаданих від тексту документа.

    Блок — рядки `ключ: значення` між двома рядками `---` на початку
    файлу. Повертає словник метаданих і решту тексту. Порожні значення
    (`product:` без нічого) стають порожнім рядком. Якщо блоку немає —
    порожній словник і текст як є.
    """
    lines = raw.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}, raw
    metadata: dict = {}
    for i, line in enumerate(lines[1:], start=1):
        if line.strip() == "---":
            body = "\n".join(lines[i + 1:]).lstrip("\n")
            return metadata, body
        if ":" in line:
            key, _, value = line.partition(":")
            metadata[key.strip()] = value.strip()
    return {}, raw


def load_documents(docs_dir: Path = DOCS_DIR) -> list[tuple[str, dict, str]]:
    """Прочитати всі документи колекції.

    Повертає список трійок (імʼя файлу, метадані, текст) для кожного
    `*.md` у папці, крім `README.md` — він описує колекцію, а не є її
    частиною. Порядок — за іменем файлу, щоб індекс будувався однаково
    від запуску до запуску.
    """
    documents = []
    for path in sorted(docs_dir.glob("*.md")):
        if path.name.lower() == "readme.md":
            continue
        metadata, body = parse_front_matter(path.read_text(encoding="utf-8"))
        documents.append((path.name, metadata, body))
    return documents


# Заголовки рівня ## — природні межі розділів. ### — підрозділи
# лишаються в тексті розділу; не ріжемо за ними, щоб не дробити занадто.
_HEADING_RE = re.compile(r"^##\s+(.+?)\s*$", re.MULTILINE)


def split(text: str, source: str, metadata: dict) -> list[Chunk]:
    """Поділити текст документа на фрагменти.

    Правило поділу (свідоме рішення):

    1. Документ ріжеться за заголовками `##` — природні межі розділів.
       Уривок «Утримуйте кнопку 10 секунд» без заголовка незрозумілий
       ні моделі, ні людині.
    2. Якщо розділ довший за `CHUNK_SIZE` — ріжеться далі вікном із
       перекриттям `CHUNK_OVERLAP`.
    3. У текст фрагмента додається назва документа й заголовок розділу:
       модель бачить контекст, а користувач — джерело.
    4. У метадані фрагмента додається `heading` і `chunk_index`.
    """
    text = text.strip()
    if not text:
        return []

    title = metadata.get("title") or source

    # Розбити за заголовками ## (заголовок зберігається окремо)
    sections: list[tuple[str, str]] = []
    matches = list(_HEADING_RE.finditer(text))

    if not matches:
        # Документ без заголовків — одна секція
        sections = [("", text)]
    else:
        # Текст до першого ##
        pre = text[:matches[0].start()].strip()
        if pre:
            sections.append(("", pre))
        # Кожен ## і текст до наступного ##
        for i, match in enumerate(matches):
            heading = match.group(1).strip()
            start = match.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            body = text[start:end].strip()
            if body:
                sections.append((heading, body))

    chunks: list[Chunk] = []
    chunk_index = 0

    for heading, section in sections:
        if len(section) <= CHUNK_SIZE:
            pieces = [section]
        else:
            pieces = []
            step = max(1, CHUNK_SIZE - CHUNK_OVERLAP)
            start = 0
            while start < len(section):
                piece = section[start:start + CHUNK_SIZE].strip()
                if piece:
                    pieces.append(piece)
                if start + CHUNK_SIZE >= len(section):
                    break
                start += step

        for piece in pieces:
            # Пропускаємо занадто короткі фрагменти — заголовки без
            # змісту або обривки. 40 символів — евристика: коротше
            # майже ніколи не буває корисною відповіддю, але займає
            # місце в top-k і розводить оцінки.
            if len(piece) < 40:
                continue

            # Контекст у тексті фрагмента
            header = title
            if heading:
                header = f"{title} — {heading}"
            chunk_text = f"{header}\n\n{piece}"

            chunk_metadata = dict(metadata)
            if heading:
                chunk_metadata["heading"] = heading
            chunk_metadata["chunk_index"] = chunk_index

            chunks.append(Chunk(
                text=chunk_text,
                source=source,
                metadata=chunk_metadata,
            ))
            chunk_index += 1

    return chunks


def load_chunks(docs_dir: Path = DOCS_DIR) -> list[Chunk]:
    """Прочитати колекцію й повернути всі її фрагменти."""
    chunks: list[Chunk] = []
    for source, metadata, body in load_documents(docs_dir):
        chunks.extend(split(body, source, metadata))
    return chunks