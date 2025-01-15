"""
Tests for model adapters.
Tests MockModelAdapter and basic adapter functionality.
"""

import pytest
from src.adapters.model_adapter import MockModelAdapter


@pytest.fixture
def mock_adapter():
    """Create a mock model adapter for testing."""
    return MockModelAdapter()


@pytest.mark.asyncio
async def test_mock_generate_basic(mock_adapter):
    """Test basic generation with mock adapter."""
    prompt = "What is the capital of France?"
    
    responses = await mock_adapter.generate(prompt, n=2)
    
    assert len(responses) == 2
    assert all(isinstance(response, str) for response in responses)
    assert all(len(response) > 0 for response in responses)


@pytest.mark.asyncio
async def test_mock_generate_deterministic(mock_adapter):
    """Test that mock adapter is deterministic."""
    prompt = "Test question"
    
    responses1 = await mock_adapter.generate(prompt, n=1)
    responses2 = await mock_adapter.generate(prompt, n=1)
    
    # Should be deterministic
    assert responses1[0] == responses2[0]


@pytest.mark.asyncio
async def test_mock_generate_different_prompts(mock_adapter):
    """Test that different prompts produce different responses."""
    prompt1 = "What is the capital of France?"
    prompt2 = "What is 2 + 2?"
    
    response1 = await mock_adapter.generate(prompt1, n=1)
    response2 = await mock_adapter.generate(prompt2, n=1)
    
    # Should be different
    assert response1[0] != response2[0]


@pytest.mark.asyncio
async def test_mock_score_text_entailment(mock_adapter):
    """Test entailment scoring with mock adapter."""
    premise = "France is a country in Europe"
    hypothesis = "France is in Europe"
    
    score = await mock_adapter.score_text_entailment(premise, hypothesis)
    
    assert 0.0 <= score <= 1.0
    assert isinstance(score, float)


@pytest.mark.asyncio
async def test_mock_score_text_entailment_high_overlap(mock_adapter):
    """Test entailment scoring with high word overlap."""
    premise = "The capital of France is Paris"
    hypothesis = "Paris is the capital of France"
    
    score = await mock_adapter.score_text_entailment(premise, hypothesis)
    
    # Should have high score due to word overlap
    assert score > 0.5


@pytest.mark.asyncio
async def test_mock_score_text_entailment_low_overlap(mock_adapter):
    """Test entailment scoring with low word overlap."""
    premise = "The capital of France is Paris"
    hypothesis = "The weather is sunny today"
    
    score = await mock_adapter.score_text_entailment(premise, hypothesis)
    
    # Should have low score due to no overlap
    assert score < 0.5


@pytest.mark.asyncio
async def test_mock_classify_obfuscation_clear(mock_adapter):
    """Test obfuscation classification for clear text."""
    text = "The answer is 42. This is definitive."
    
    score = await mock_adapter.classify_obfuscation(text)
    
    assert 0.0 <= score <= 1.0
    assert score < 0.5  # Should be low for clear text


@pytest.mark.asyncio
async def test_mock_classify_obfuscation_evasive(mock_adapter):
    """Test obfuscation classification for evasive text."""
    text = "Maybe the answer could be 42, but I'm not sure. It might be something else."
    
    score = await mock_adapter.classify_obfuscation(text)
    
    assert 0.0 <= score <= 1.0
    assert score > 0.5  # Should be high for evasive text


@pytest.mark.asyncio
async def test_mock_classify_obfuscation_self_referential(mock_adapter):
    """Test obfuscation classification for self-referential text."""
    text = "As an AI, I cannot provide a definitive answer. I am not able to determine this."
    
    score = await mock_adapter.classify_obfuscation(text)
    
    assert 0.0 <= score <= 1.0
    assert score > 0.5  # Should be high for self-referential text


@pytest.mark.asyncio
async def test_mock_generate_cot_format(mock_adapter):
    """Test that mock adapter generates CoT format."""
    prompt = "What is the capital of France?"
    
    responses = await mock_adapter.generate(prompt, n=1)
    response = responses[0]
    
    # Should contain step markers
    assert "Step" in response
    assert ":" in response


@pytest.mark.asyncio
async def test_mock_generate_math_prompt(mock_adapter):
    """Test mock adapter with math prompt."""
    prompt = "What is 2 + 2?"
    
    responses = await mock_adapter.generate(prompt, n=1)
    response = responses[0]
    
    # Should contain math-related content
    assert len(response) > 0
    assert "Step" in response


@pytest.mark.asyncio
async def test_mock_score_text_entailment_empty_hypothesis(mock_adapter):
    """Test entailment scoring with empty hypothesis."""
    premise = "Some premise"
    hypothesis = ""
    
    score = await mock_adapter.score_text_entailment(premise, hypothesis)
    
    assert score == 0.0


@pytest.mark.asyncio
async def test_mock_classify_obfuscation_empty_text(mock_adapter):
    """Test obfuscation classification with empty text."""
    text = ""
    
    score = await mock_adapter.classify_obfuscation(text)
    
    assert score == 0.0
