"""
Tests for ReasoningPipeline class.
Tests CoT generation, parsing, and perturbation experiments.
"""

import pytest
from src.core.pipeline import ReasoningPipeline
from src.adapters.model_adapter import MockModelAdapter


@pytest.fixture
def mock_adapter():
    """Create a mock model adapter for testing."""
    return MockModelAdapter()


@pytest.fixture
def pipeline(mock_adapter):
    """Create a reasoning pipeline with mock adapter."""
    return ReasoningPipeline(mock_adapter)


@pytest.mark.asyncio
async def test_generate_cot_basic(pipeline):
    """Test basic CoT generation."""
    prompt = "What is the capital of France?"
    context = {"domain": "geography"}
    
    traces = await pipeline.generate_cot(prompt, context, n_candidates=2)
    
    assert len(traces) == 2
    assert all(isinstance(trace, list) for trace in traces)
    assert all(len(trace) > 0 for trace in traces)


@pytest.mark.asyncio
async def test_generate_cot_no_context(pipeline):
    """Test CoT generation without context."""
    prompt = "Solve 2 + 2"
    
    traces = await pipeline.generate_cot(prompt, None, n_candidates=1)
    
    assert len(traces) == 1
    assert len(traces[0]) > 0


@pytest.mark.asyncio
async def test_parse_cot_steps():
    """Test CoT step parsing."""
    pipeline = ReasoningPipeline(MockModelAdapter())
    
    # Test numbered steps
    response = """Step 1: I need to identify the capital.
Step 2: France is a country in Europe.
Step 3: The capital is Paris."""
    
    steps = pipeline._parse_cot_steps(response)
    assert len(steps) == 3
    assert "identify the capital" in steps[0]
    assert "Paris" in steps[2]


@pytest.mark.asyncio
async def test_parse_cot_steps_fallback():
    """Test CoT parsing fallback for non-numbered responses."""
    pipeline = ReasoningPipeline(MockModelAdapter())
    
    # Test sentence splitting fallback
    response = "This is step one. This is step two. This is step three."
    
    steps = pipeline._parse_cot_steps(response)
    assert len(steps) >= 2


@pytest.mark.asyncio
async def test_finalize_answer(pipeline):
    """Test answer finalization."""
    trace = [
        "Step 1: I need to solve this problem",
        "Step 2: The answer is 42"
    ]
    
    answer = await pipeline.finalize_answer(trace)
    assert isinstance(answer, str)
    assert len(answer) > 0


@pytest.mark.asyncio
async def test_finalize_answer_empty_trace(pipeline):
    """Test answer finalization with empty trace."""
    answer = await pipeline.finalize_answer([])
    assert answer == ""


def test_perturb_trace(pipeline):
    """Test trace perturbation."""
    trace = ["Step 1", "Step 2", "Step 3", "Step 4"]
    removed_indices = [1, 3]
    
    perturbed = pipeline.perturb_trace(trace, removed_indices)
    
    assert len(perturbed) == 2
    assert "Step 1" in perturbed
    assert "Step 3" in perturbed
    assert "Step 2" not in perturbed
    assert "Step 4" not in perturbed


def test_perturb_trace_empty(pipeline):
    """Test perturbation of empty trace."""
    perturbed = pipeline.perturb_trace([], [0])
    assert perturbed == []


@pytest.mark.asyncio
async def test_run_perturbation_experiments(pipeline):
    """Test perturbation experiments."""
    trace = [
        "Step 1: Identify the problem",
        "Step 2: Analyze the data", 
        "Step 3: Calculate the result",
        "Step 4: Verify the answer"
    ]
    
    result = await pipeline.run_perturbation_experiments(trace, n_trials=5)
    
    assert result.original_answer is not None
    assert len(result.perturbed_answers) == 5
    assert 0.0 <= result.causal_influence_score <= 1.0
    
    # Check that each perturbation has required fields
    for perturb in result.perturbed_answers:
        assert "removed_steps" in perturb
        assert "new_answer" in perturb
        assert "changed" in perturb
        assert isinstance(perturb["changed"], bool)


@pytest.mark.asyncio
async def test_run_perturbation_experiments_empty_trace(pipeline):
    """Test perturbation experiments with empty trace."""
    result = await pipeline.run_perturbation_experiments([], n_trials=3)
    
    assert result.original_answer == ""
    assert result.perturbed_answers == []
    assert result.causal_influence_score == 0.0


@pytest.mark.asyncio
async def test_generate_cot_single_step_trace(pipeline, monkeypatch):
    """Test CoT generation with a single-step trace."""
    async def mock_generate(*args, **kwargs):
        return ["This is a single step trace."]

    monkeypatch.setattr(pipeline.model_adapter, "generate", mock_generate)

    prompt = "A question that returns a single step trace"
    traces = await pipeline.generate_cot(prompt, n_candidates=1)

    assert len(traces) == 1
    assert len(traces[0]) == 1
    assert traces[0][0] == "This is a single step trace."
