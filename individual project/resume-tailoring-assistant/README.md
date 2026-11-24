# Resume Tailoring Assistant

Розумний помічник для адаптації резюме під конкретну вакансію.

## Для кого
Шукач роботи, який подає одне резюме на багато вакансій і не розуміє, чому його відсіюють ATS-фільтри.

## Що робить
Порівнює текст резюме (PDF/DOCX) із текстом вакансії, обчислює відсоток відповідності, показує сильні сторони й прогалини, генерує поради щодо стилю та 3 можливі питання на співбесіді.

## Структура проєкту

```text
resume-tailoring-assistant/
│
├── .env.example
├── requirements.txt
├── README.md
├── AI_USAGE.md
│
├── docs/
│   ├── architecture.md          # Рівні 1–3 архітектури
│   ├── api.md                   # API-контракти
│   ├── backlog.md               # План ПР14–ПР20
│   ├── passport.md              # Паспорт проєкту
│   │
│   └── adr/
│       ├── 001-sync-vs-async.md # Вибір між синхронною та асинхронною взаємодією
│       ├── 002-storage.md       # Архітектурне рішення щодо сховища
│       └── 003-provider.md      # Вибір AI-провайдера
│
├── app/
│   ├── main.py                  # Контролер: маршрути, health, HTML
│   ├── settings.py              # Конфігурація із .env
│   ├── models.py                # Pydantic-схеми контрактів
│   ├── service.py               # Прикладний сервіс
│   ├── orchestration.py         # Оркестрація AI (заготовка ПР15)
│   ├── gateway.py               # Шлюз: вибір провайдера
│   ├── storage.py               # SQLite-сховище
│   ├── telemetry.py             # JSON-логи з request_id
│   │
│   └── providers/
│       ├── base.py               # Контракт Provider
│       ├── mock.py               # Детермінований mock-провайдер
│       └── openai_compat.py      # Заготовка для ПР15
│
├── eval/
│   └── cases.json               # Eval-кейси (n01–n09)
│
└── tests/
    ├── conftest.py              # Спільні налаштування тестів
    ├── test_api.py              # Тести API
    ├── test_service.py          # Тести прикладного сервісу
    └── test_failure.py          # Тести обробки помилок
```

### Опис основних директорій

| Директорія | Призначення |
|---|---|
| `app/` | Основна логіка застосунку |
| `app/providers/` | Контракти та реалізації AI-провайдерів |
| `docs/` | Архітектурна документація, API-контракти та планування |
| `docs/adr/` | Архітектурні рішення (ADR) |
| `eval/` | Набір тестових сценаріїв для оцінювання |
| `tests/` | Автоматизовані тести застосунку |

### Основні файли

- `main.py` — точка входу, маршрути API, health-check та HTML.
- `settings.py` — завантаження конфігурації з `.env`.
- `models.py` — моделі даних і Pydantic-схеми.
- `service.py` — прикладна бізнес-логіка.
- `orchestration.py` — оркестрація взаємодії з AI.
- `gateway.py` — вибір відповідного AI-провайдера.
- `storage.py` — робота зі сховищем SQLite.
- `telemetry.py` — журналювання подій у форматі JSON із `request_id`.
- `providers/base.py` — загальний контракт для провайдерів.
- `providers/mock.py` — детермінована mock-реалізація.
- `providers/openai_compat.py` — заготовка сумісного OpenAI-провайдера.
- `eval/cases.json` — тестові eval-кейси `n01–n09`.
- `AI_USAGE.md` — документація використання штучного інтелекту.
- `requirements.txt` — залежності Python.
- `.env.example` — приклад змінних середовища.

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