def test_analyze_through_api_with_mock(client):
    payload = {
        "vacancy_text": "Python backend developer. Django, PostgreSQL.",
        "resume_text": "Олена К. Python developer, 3 роки. Django, DRF.",
    }
    r = client.post("/api/analyze", json=payload)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["match_score"] == 72
    assert 1 <= len(body["strengths"]) <= 5
    assert len(body["interview_questions"]) == 3
    # request_id присутній і в заголовку, і в тілі
    assert r.headers.get("x-request-id")
    assert body["request_id"]


def test_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert body["provider"] == "mock"
    assert body["mode"] == "mock"
    assert body["storage"] == "ok"
    assert "api_key" not in str(body).lower()