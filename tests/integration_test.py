"""
Integration tests for the RaaS application.
These tests cover the full request-response cycle through the API.
"""

import pytest
from fastapi.testclient import TestClient
from src.main import app


@pytest.fixture
def client():
    """Create a test client for the application."""
    return TestClient(app)


def test_reason_dryrun_integration(client):
    """Test the full reasoning pipeline in dryrun mode."""
    request_data = {
        "input": "What is the capital of France?",
        "context": {"domain": "geography"},
    }

    response = client.post("/api/v1/reason?mode=dryrun", json=request_data)

    assert response.status_code == 200
    data = response.json()

    assert "answer" in data
    assert "reasoning_trace" in data
    assert "faithfulness_score" in data
    assert "coherence_score" in data
    assert "risk_flag" in data
    assert "monitor_explanation" in data
    assert "metadata" in data

    assert 0.0 <= data["faithfulness_score"] <= 1.0
    assert 0.0 <= data["coherence_score"] <= 1.0
    assert isinstance(data["risk_flag"], bool)

    assert data["metadata"]["mode"] == "dryrun"

def test_reason_perturb_integration(client):
    """Test the full reasoning pipeline in perturb mode."""
    request_data = {
        "input": "Solve 2 + 2",
        "context": {"domain": "math"},
    }

    response = client.post("/api/v1/reason?mode=perturb", json=request_data)

    assert response.status_code == 200
    data = response.json()

    assert "answer" in data
    assert "perturbation" in data
    assert data["perturbation"] is not None
    assert "original_answer" in data["perturbation"]
    assert "perturbed_answers" in data["perturbation"]
    assert "causal_influence_score" in data["perturbation"]

    assert data["metadata"]["mode"] == "perturb"

def test_reason_live_integration(client):
    """Test the full reasoning pipeline in live mode."""
    request_data = {
        "input": "What is 1 + 1?",
        "context": {"domain": "math"},
    }

    response = client.post("/api/v1/reason?mode=live", json=request_data)

    assert response.status_code in [200, 409]
    data = response.json()

    if response.status_code == 200:
        assert "answer" in data
        assert data["metadata"]["mode"] == "live"
    else:
        assert data is not None

def test_reason_empty_input_integration(client):
    """Test the reasoning endpoint with empty input."""
    request_data = {
        "input": "",
    }

    response = client.post("/api/v1/reason?mode=dryrun", json=request_data)

    assert response.status_code == 422
