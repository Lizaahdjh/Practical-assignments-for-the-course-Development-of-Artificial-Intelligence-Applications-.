# Resume Tailoring Assistant

Розумний помічник для адаптації резюме під конкретну вакансію.

## Для кого
Шукач роботи, який подає одне резюме на багато вакансій і не розуміє, чому його відсіюють ATS-фільтри.

## Що робить
Порівнює текст резюме (PDF/DOCX) із текстом вакансії, обчислює відсоток відповідності, показує сильні сторони й прогалини, генерує поради щодо стилю та 3 можливі питання на співбесіді.

## Структура

resume-tailoring-assistant/
├── .env.example
├── requirements.txt
├── README.md
├── docs/
│   ├── architecture.md
│   ├── api.md
│   └── adr/
│       ├── 001-sync-vs-async.md
│       ├── 002-storage.md
│       └── 003-provider.md
├── app/
│   ├── __init__.py
│   ├── main.py
│   ├── settings.py
│   ├── models.py
│   ├── service.py
│   ├── orchestration.py
│   ├── gateway.py
│   ├── storage.py
│   ├── telemetry.py
│   └── providers/
│       ├── __init__.py
│       ├── base.py
│       ├── mock.py
│       └── openai_compat.py
└── tests/
    ├── __init__.py
    ├── conftest.py
    ├── test_api.py
    ├── test_service.py
    └── test_failure.py

## Стек
- Python 3.10+
- Streamlit (UI)
- PyPDF2 / docx2txt (читання файлів)
- scikit-learn (TF-IDF + косинусна схожість)
- sentence-transformers + ChromaDB (RAG, локально)
- Google Gemini API (безкоштовний рівень) — генерація порад і питань

## Запуск
```bash
git clone <repo-url>
cd resume-tailoring-assistant
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
uvicorn app.main:app --reload

## Статус
ПР13 — постановка задачі. Код ще не написано.