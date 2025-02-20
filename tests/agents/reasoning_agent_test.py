"""
Tests for ReasoningAgent class.
Tests the full reasoning pipeline orchestration.
"""

import pytest
from src.agents.reasoning_agent import ReasoningAgent, MonitoringError
from src.core.models import ReasoningRequest
from src.adapters.model_adapter import MockModelAdapter
from src.core.monitor import CoTMonitor


@pytest.fixture
def mock_adapter():
    """Create a mock model adapter for testing."""
    return MockModelAdapter()


@pytest.fixture
def monitor(mock_adapter):
    """Create a CoT monitor with mock adapter."""
    return CoTMonitor(mock_adapter)


@pytest.fixture
def agent(mock_adapter, monitor):
    """Create a reasoning agent with mock components."""
    return ReasoningAgent(mock_adapter, monitor, n_candidates=2)


@pytest.mark.asyncio
async def test_reason_basic(agent):
    """Test basic reasoning functionality."""
    request = ReasoningRequest(
        input="What is the capital of France?",
        context={"domain": "geography"},
        request_id="test_123"
    )
    
    response = await agent.reason(request, mode="dryrun")
    
    assert response.answer is not None
    assert len(response.reasoning_trace) > 0
    assert 0.0 <= response.faithfulness_score <= 1.0
    assert 0.0 <= response.coherence_score <= 1.0
    assert isinstance(response.risk_flag, bool)
    assert response.monitor_explanation is not None
    assert response.metadata["request_id"] == "test_123"
    assert response.metadata["mode"] == "dryrun"


@pytest.mark.asyncio
async def test_reason_with_perturbation(agent):
    """Test reasoning with perturbation mode."""
    request = ReasoningRequest(
        input="Solve 2 + 2",
        context={"domain": "math"}
    )
    
    response = await agent.reason(request, mode="perturb")
    
    assert response.perturbation is not None
    assert response.perturbation.original_answer is not None
    assert len(response.perturbation.perturbed_answers) > 0
    assert 0.0 <= response.perturbation.causal_influence_score <= 1.0


@pytest.mark.asyncio
async def test_reason_live_mode_no_risk(agent):
    """Test live mode with no risk."""
    request = ReasoningRequest(
        input="What is 1 + 1?",
        context={"domain": "math"}
    )
    
    with pytest.raises(MonitoringError):
        await agent.reason(request, mode="live")


@pytest.mark.asyncio
async def test_reason_empty_input(agent):
    """Test reasoning with empty input."""
    request = ReasoningRequest(input="")
    with pytest.raises(ValueError, match="Input cannot be empty"):
        await agent.reason(request, mode="dryrun")


@pytest.mark.asyncio
async def test_calculate_candidate_score(agent):
    """Test candidate scoring."""
    # High quality assessment
    high_quality = {
        "faithfulness_score": 0.9,
        "coherence_score": 0.8,
        "risk_flag": False
    }
    score_high = agent._calculate_candidate_score(high_quality)
    assert score_high > 0.7
    
    # Low quality assessment
    low_quality = {
        "faithfulness_score": 0.3,
        "coherence_score": 0.4,
        "risk_flag": True
    }
    score_low = agent._calculate_candidate_score(low_quality)
    assert score_low < 0.5
    
    # Risk penalty should reduce score
    no_risk = {
        "faithfulness_score": 0.5,
        "coherence_score": 0.5,
        "risk_flag": False
    }
    risk = {
        "faithfulness_score": 0.5,
        "coherence_score": 0.5,
        "risk_flag": True
    }
    
    score_no_risk = agent._calculate_candidate_score(no_risk)
    score_risk = agent._calculate_candidate_score(risk)
    
    assert score_no_risk > score_risk


@pytest.mark.asyncio
async def test_reason_with_request_id(agent):
    """Test reasoning with provided request ID."""
    request = ReasoningRequest(
        input="Test question",
        request_id="custom_id_123"
    )
    
    response = await agent.reason(request, mode="dryrun")
    
    assert response.metadata["request_id"] == "custom_id_123"


@pytest.mark.asyncio
async def test_reason_auto_generated_request_id(agent):
    """Test reasoning with auto-generated request ID."""
    request = ReasoningRequest(input="Test question")
    
    response = await agent.reason(request, mode="dryrun")
    
    assert response.metadata["request_id"] is not None
    assert len(response.metadata["request_id"]) > 0


@pytest.mark.asyncio
async def test_reason_multiple_candidates(agent):
    """Test reasoning with multiple candidates."""
    request = ReasoningRequest(
        input="What is the capital of France?",
        context={"domain": "geography"}
    )
    
    response = await agent.reason(request, mode="dryrun")
    
    # Should have evaluated multiple candidates
    assert response.metadata["n_candidates"] >= 1
    assert response.metadata["best_score"] is not None


@pytest.mark.asyncio
async def test_reason_empty_trace(agent, monkeypatch):
    """Test reasoning with empty trace from model adapter."""
    async def mock_generate_cot(*args, **kwargs):
        return []

    monkeypatch.setattr(agent.pipeline, "generate_cot", mock_generate_cot)

    request = ReasoningRequest(input="A question that returns empty trace")

    with pytest.raises(ValueError, match="No reasoning traces generated"):
        await agent.reason(request, mode="dryrun")
