import json
from pathlib import Path

from app.settings import get_settings


def test_log_contains_all_layers(client, mock_env):
    """Доводить, що запит пройшов крізь шари: у журналі є записи
    controller і service з одним і тим самим request_id."""
    payload = {
        "vacancy_text": "Python backend developer. Django, PostgreSQL.",
        "resume_text": "Олена К. Python developer, 3 роки. Django, DRF.",
    }
    r = client.post("/api/analyze", json=payload)
    assert r.status_code == 200
    req_id = r.json()["request_id"]

    log_path = Path(get_settings().log_file)
    assert log_path.exists(), f"log file not found: {log_path}"

    lines = log_path.read_text(encoding="utf-8").strip().split("\n")
    records = [json.loads(line) for line in lines if req_id in line]

    layers = {rec["layer"] for rec in records}
    assert "controller" in layers, f"no controller log; got {layers}"
    assert "service" in layers, f"no service log; got {layers}"
    assert all(rec["request_id"] == req_id for rec in records)