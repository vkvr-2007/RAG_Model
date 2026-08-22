from fastapi.testclient import TestClient
from app.main import app
from app.schemas import QueryResponse


class FakeService:
    async def query(self, query):
        return QueryResponse(answer="संदर्भ उत्तर", grounded=True, confidence=.9, sources=[], latency_ms=1)


def test_malformed_and_empty_requests():
    with TestClient(app) as client:
        client.app.state.rag = FakeService()
        assert client.post("/rag/query", json={}).status_code == 422
        assert client.post("/rag/query", json={"query":"   "}).status_code == 422
        assert client.post("/rag/query", json={"query": 12}).status_code == 422


def test_api_endpoint_response_shape():
    with TestClient(app) as client:
        client.app.state.rag = FakeService()
        response = client.post("/rag/query", json={"query":"test"})
    assert response.status_code == 200 and response.json()["grounded"] is True
