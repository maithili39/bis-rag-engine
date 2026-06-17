"""Tests for the FastAPI application."""

import sys
import os
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).parent.parent))
from api import app, _pipeline

# Ensure the app starts with a mocked pipeline for testing
@pytest.fixture(autouse=True)
def mock_pipeline(monkeypatch):
    class MockPipeline:
        def __init__(self):
            self.known_codes = {"IS 269: 1989", "IS 383: 1970"}
            
        def process_query(self, query: str, top_k: int):
            return ["IS 269: 1989"], "This is a mock rationale.", 0.042
            
        def get_pipeline_stats(self):
            return {"total_standards": 2, "embedding_model": "mock-model"}
            
    import api
    monkeypatch.setattr(api, "_pipeline", MockPipeline())
    yield


client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "pipeline_stats" in data
    assert data["pipeline_stats"]["total_standards"] == 2


def test_standards_endpoint():
    response = client.get("/standards")
    assert response.status_code == 200
    data = response.json()
    assert data["count"] == 2
    assert "IS 269: 1989" in data["standards"]


def test_query_valid_request():
    response = client.post(
        "/query",
        json={"query": "portland cement", "top_k": 3}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["query"] == "portland cement"
    assert len(data["recommended_standards"]) > 0
    assert data["rationale"] == "This is a mock rationale."
    assert "latency_seconds" in data


def test_query_too_short():
    response = client.post(
        "/query",
        json={"query": "abc", "top_k": 3}
    )
    assert response.status_code == 422


def test_x_process_time_header():
    response = client.get("/health")
    assert response.status_code == 200
    assert "X-Process-Time" in response.headers
