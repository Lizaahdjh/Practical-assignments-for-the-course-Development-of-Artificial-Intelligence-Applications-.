def test_analyze_through_api_with_mock(client):
    payload = {
        "vacancy_text": "Python backend developer. Django, PostgreSQL.",
        "resume_text": "Олена К. Python developer, 3 роки. Django, DRF.",
    }
    r = client.post("/api/analyze", json=payload)
    assert r.status_code == 200, r.text
    body = r.json()
    # Бал залежить від входу (mock дивиться на текст) — просто перевіряємо межі
    assert 0 <= body["match_score"] <= 100
    assert 1 <= len(body["strengths"]) <= 5
    assert len(body["interview_questions"]) == 3
    # request_id присутній і в заголовку, і в тілі
    assert r.headers.get("x-request-id")
    assert body["request_id"]


def test_mock_depends_on_input(client):
    """Доводить, що результат не фіксований: різний вхід -> різний бал.
    Якщо б mock повертав константу, цей тест провалився б."""
    r1 = client.post(
        "/api/analyze",
        json={"vacancy_text": "Python backend developer.", "resume_text": "Django, DRF, 3 роки."},
    )
    r2 = client.post(
        "/api/analyze",
        json={"vacancy_text": "Java developer.", "resume_text": "Spring, Hibernate, 2 роки."},
    )
    assert r1.status_code == 200
    assert r2.status_code == 200
    assert r1.json()["match_score"] != r2.json()["match_score"]


def test_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert body["provider"] == "mock"
    assert body["mode"] == "mock"
    assert body["storage"] == "ok"
    assert "api_key" not in str(body).lower()