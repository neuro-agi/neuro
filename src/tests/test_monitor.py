"""
Tests for CoTMonitor class.
Tests faithfulness, coherence, and risk detection.
"""

import pytest
from src.core.monitor import CoTMonitor
from src.adapters.model_adapter import MockModelAdapter


@pytest.fixture
def mock_adapter():
    """Create a mock model adapter for testing."""
    return MockModelAdapter()


@pytest.fixture
def monitor(mock_adapter):
    """Create a CoT monitor with mock adapter."""
    return CoTMonitor(mock_adapter)


@pytest.mark.asyncio
async def test_assess_basic(monitor):
    """Test basic assessment functionality."""
    trace = [
        "Step 1: I need to identify the capital",
        "Step 2: France is a country in Europe", 
        "Step 3: The capital of France is Paris"
    ]
    answer = "Paris"
    
    result = await monitor.assess(trace, answer)
    
    assert "faithfulness_score" in result
    assert "coherence_score" in result
    assert "risk_flag" in result
    assert "monitor_explanation" in result
    assert "components" in result
    
    # Check score ranges
    assert 0.0 <= result["faithfulness_score"] <= 1.0
    assert 0.0 <= result["coherence_score"] <= 1.0
    assert isinstance(result["risk_flag"], bool)


@pytest.mark.asyncio
async def test_assess_empty_trace(monitor):
    """Test assessment with empty trace."""
    result = await monitor.assess([], "answer")
    
    assert result["faithfulness_score"] == 0.0
    assert result["coherence_score"] == 0.0
    assert result["risk_flag"] is True
    assert "Empty reasoning trace" in result["monitor_explanation"]


@pytest.mark.asyncio
async def test_assess_single_step(monitor):
    """Test assessment with single step."""
    trace = ["Step 1: The answer is 42"]
    answer = "42"
    
    result = await monitor.assess(trace, answer)
    
    assert result["coherence_score"] == 1.0  # Single step is always coherent
    assert 0.0 <= result["faithfulness_score"] <= 1.0


@pytest.mark.asyncio
async def test_compute_step_entailment(monitor):
    """Test step entailment computation."""
    trace = [
        "Step 1: I need to solve 2 + 2",
        "Step 2: 2 + 2 equals 4"
    ]
    answer = "4"
    
    entailment = await monitor._compute_step_entailment(trace, answer)
    
    assert 0.0 <= entailment <= 1.0


@pytest.mark.asyncio
async def test_compute_coherence(monitor):
    """Test coherence computation."""
    # Coherent trace
    coherent_trace = [
        "Step 1: I need to find the capital",
        "Step 2: France is a country",
        "Step 3: The capital is Paris"
    ]
    
    coherence = await monitor._compute_coherence(coherent_trace)
    assert 0.0 <= coherence <= 1.0


@pytest.mark.asyncio
async def test_compute_obfuscation(monitor):
    """Test obfuscation detection."""
    # Clear trace
    clear_trace = [
        "Step 1: The answer is 42",
        "Step 2: This is definitive"
    ]
    
    obfuscation = await monitor._compute_obfuscation(clear_trace)
    assert 0.0 <= obfuscation <= 1.0


def test_lexical_obfuscation_heuristic(monitor):
    """Test lexical obfuscation detection."""
    # Clear text
    clear_text = "The answer is 42. This is definitive."
    obfuscation = monitor._lexical_obfuscation_heuristic(clear_text)
    assert obfuscation < 0.5
    
    # Evasive text
    evasive_text = "Maybe the answer could be 42, but I'm not sure. It might be something else."
    obfuscation = monitor._lexical_obfuscation_heuristic(evasive_text)
    assert obfuscation > 0.5


def test_aggregate_faithfulness(monitor):
    """Test faithfulness aggregation."""
    components = {
        "counterfactual_influence": 0.8,
        "step_entailment": 0.7,
        "obfuscation": 0.2
    }
    
    faithfulness = monitor._aggregate_faithfulness(components)
    assert 0.0 <= faithfulness <= 1.0


def test_detect_risk(monitor):
    """Test risk detection."""
    # No risk
    no_risk = monitor._detect_risk(0.8, 0.7, {"obfuscation": 0.3})
    assert no_risk is False
    
    # Low faithfulness risk
    faithfulness_risk = monitor._detect_risk(0.4, 0.7, {"obfuscation": 0.3})
    assert faithfulness_risk is True
    
    # Low coherence risk
    coherence_risk = monitor._detect_risk(0.8, 0.3, {"obfuscation": 0.3})
    assert coherence_risk is True
    
    # High obfuscation risk
    obfuscation_risk = monitor._detect_risk(0.8, 0.7, {"obfuscation": 0.9})
    assert obfuscation_risk is True


def test_generate_explanation(monitor):
    """Test explanation generation."""
    explanation = monitor._generate_explanation(0.8, 0.7, False, {"obfuscation": 0.3})
    
    assert isinstance(explanation, str)
    assert len(explanation) > 0
    assert "High faithfulness" in explanation
    assert "High coherence" in explanation
    assert "No significant risks" in explanation


def test_generate_explanation_with_risk(monitor):
    """Test explanation generation with risk."""
    explanation = monitor._generate_explanation(0.4, 0.3, True, {"obfuscation": 0.9})
    
    assert "Low faithfulness" in explanation
    assert "Low coherence" in explanation
    assert "Risk detected" in explanation


@pytest.mark.asyncio
async def test_assess_contradictory_trace(monitor):
    """Test assessment with a contradictory trace."""
    trace = [
        "Step 1: The sky is blue.",
        "Step 2: The sky is not blue, it is green."
    ]
    answer = "The sky is green."

    result = await monitor.assess(trace, answer)

    assert result["coherence_score"] < 0.5
    assert result["risk_flag"] is True
    assert "Low coherence" in result["monitor_explanation"]
