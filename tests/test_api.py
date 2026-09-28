"""
/health needs no API key at all — pure endpoint logic, no agent call.
/chat is NOT tested here since it invokes the real agent (needs OPENAI_API_KEY);
that becomes an integration test once a key is available.
"""

from fastapi.testclient import TestClient

from src.api.main import app

client = TestClient(app)


def test_health_returns_ok():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_chat_rejects_empty_question():
    response = client.post("/chat", json={"question": "   "})
    assert response.status_code == 400