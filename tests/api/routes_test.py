"""
Tests for API endpoints.
Tests the FastAPI routes and error handling.
"""

import pytest
from fastapi.testclient import TestClient
from src.main import app
from src.api.routes import set_agent
from src.adapters.model_adapter import MockModelAdapter
from src.core.monitor import CoTMonitor
from src.agents.reasoning_agent import ReasoningAgent


@pytest.fixture
def mock_agent():
    """Create a mock reasoning agent for testing."""
    adapter = MockModelAdapter()
    monitor = CoTMonitor(adapter)
    agent = ReasoningAgent(adapter, monitor, n_candidates=2)
    return agent


@pytest.fixture
def client(mock_agent):
    """Create a test client with mock agent."""
    set_agent(mock_agent)
    return TestClient(app)


def test_health_check():
    """Test health check endpoint."""
    with TestClient(app) as client:
        response = client.get("/healthz")
    
    assert response.status_code == 200
    data = response.json()
    assert data["service"] == "ok"
    assert data["monitor"] == "ok"
    assert "uptime_seconds" in data


def test_root_endpoint(client):
    """Test root endpoint."""
    response = client.get("/")
    
    assert response.status_code == 200
    data = response.json()
    assert data["service"] == "Reasoning-as-a-Service (RaaS)"
    assert "endpoints" in data


def test_reason_dryrun_mode(client):
    """Test reasoning endpoint in dryrun mode."""
    request_data = {
        "input": "What is the capital of France?",
        "context": {"domain": "geography"},
        "request_id": "test_123"
    }
    
    response = client.post("/api/v1/reason?mode=dryrun", json=request_data)
    
    assert response.status_code == 200
    data = response.json()
    
    # Check response structure
    assert "answer" in data
    assert "reasoning_trace" in data
    assert "faithfulness_score" in data
    assert "coherence_score" in data
    assert "risk_flag" in data
    assert "monitor_explanation" in data
    assert "metadata" in data
    
    # Check score ranges
    assert 0.0 <= data["faithfulness_score"] <= 1.0
    assert 0.0 <= data["coherence_score"] <= 1.0
    assert isinstance(data["risk_flag"], bool)
    
    # Check metadata
    assert data["metadata"]["request_id"] == "test_123"
    assert data["metadata"]["mode"] == "dryrun"


def test_reason_perturb_mode(client):
    """Test reasoning endpoint in perturb mode."""
    request_data = {
        "input": "Solve 2 + 2",
        "context": {"domain": "math"}
    }
    
    response = client.post("/api/v1/reason?mode=perturb", json=request_data)
    
    assert response.status_code == 200
    data = response.json()
    
    # Should include perturbation results
    assert "perturbation" in data
    assert data["perturbation"] is not None
    assert "original_answer" in data["perturbation"]
    assert "perturbed_answers" in data["perturbation"]
    assert "causal_influence_score" in data["perturbation"]


def test_reason_live_mode(client):
    """Test reasoning endpoint in live mode."""
    request_data = {
        "input": "What is 1 + 1?",
        "context": {"domain": "math"}
    }
    
    response = client.post("/api/v1/reason?mode=live", json=request_data)
    
    assert response.status_code == 409
    data = response.json()
    assert data["metadata"]["mode"] == "live"


def test_reason_default_mode(client):
    """Test reasoning endpoint with default mode."""
    request_data = {
        "input": "Test question"
    }
    
    assert response.status_code == 409
    data = response.json()
    assert data["metadata"]["mode"] == "live"


def test_reason_invalid_mode(client):
    """Test reasoning endpoint with invalid mode."""
    request_data = {
        "input": "Test question"
    }
    
    response = client.post("/api/v1/reason?mode=invalid", json=request_data)
    
    # Should return 422 for invalid enum value
    assert response.status_code == 422


def test_reason_missing_input(client):
    """Test reasoning endpoint with missing required field."""
    request_data = {
        "context": {"domain": "test"}
    }
    
    response = client.post("/api/v1/reason", json=request_data)
    
    # Should return 422 for missing required field
    assert response.status_code == 422


def test_reason_empty_input(client):
    """Test reasoning endpoint with empty input."""
    request_data = {
        "input": ""
    }
    
    response = client.post("/api/v1/reason?mode=dryrun", json=request_data)
    
    # Should return 422 for empty input
    assert response.status_code == 422


def test_reason_with_context(client):
    """Test reasoning endpoint with context."""
    request_data = {
        "input": "What is the capital of France?",
        "context": {
            "domain": "geography",
            "level": "beginner"
        }
    }
    
    response = client.post("/api/v1/reason?mode=dryrun", json=request_data)
    
    assert response.status_code == 200
    data = response.json()
    assert data["answer"] is not None


def test_reason_with_policy(client):
    """Test reasoning endpoint with policy."""
    request_data = {
        "input": "Explain quantum computing",
        "policy": "be_concise"
    }
    
    response = client.post("/api/v1/reason?mode=dryrun", json=request_data)
    
    assert response.status_code == 200
    data = response.json()
    assert data["answer"] is not None


def test_reason_agent_not_initialized():
    """Test reasoning endpoint when agent is not initialized."""
    # Create client without setting agent
    client_no_agent = TestClient(app)
    
    request_data = {
        "input": "Test question"
    }
    
    response = client_no_agent.post("/api/v1/reason", json=request_data)
    
    assert response.status_code == 409
    data = response.json()
    assert "not initialized" in data["detail"]

def test_eval_endpoint(client):
    """Test the /eval endpoint."""
    request_data = {
        "reasoningId": "res-abcde12345",
        "metrics": ["faithfulness", "safety"]
    }

    response = client.post("/api/v1/eval", json=request_data)

    assert response.status_code == 200
    data = response.json()

    assert "evalId" in data
    assert data["reasoningId"] == "res-abcde12345"
    assert "scores" in data
    assert len(data["scores"]) == 2

    for score in data["scores"]:
        assert "metric" in score
        assert "score" in score
        assert "explanation" in score
        assert 0.0 <= score["score"] <= 1.0
