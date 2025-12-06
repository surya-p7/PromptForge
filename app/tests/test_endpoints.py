from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert "backend" in r.json()

def test_generate_mock():
    payload = {"text": "Hello world", "metadata": {}}
    r = client.post("/api/generate", json=payload)
    assert r.status_code == 200
    data = r.json()
    assert "text" in data
