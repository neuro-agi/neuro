# Reasoning-as-a-Service (RaaS)

A prototype implementation of Chain of Thought (CoT) monitoring and reasoning based on the paper "Chain of Thought Monitorability" (arXiv:2507.11473). This service provides auditable reasoning with faithfulness and coherence scoring, risk detection, and perturbation analysis.

## Features

- **Chain of Thought Generation**: Generates multiple reasoning candidates with step-by-step analysis
- **Faithfulness Monitoring**: Evaluates how well reasoning steps support the final answer
- **Coherence Assessment**: Detects contradictions and logical inconsistencies
- **Risk Detection**: Identifies unsafe content, obfuscation, and low-quality reasoning
- **Perturbation Analysis**: Systematic experiments to measure causal influence of reasoning steps
- **Multiple Processing Modes**: Live (production), dryrun (testing), and perturb (analysis)
- **Configurable Backends**: Mock adapter for testing, OpenAI and Gemini adapters for production

## Quick Start

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd neuro
```

2. Install dependencies:
```bash
pip install fastapi uvicorn pydantic httpx pytest
```

3. Set environment variables (optional):
```bash
export MODEL_BACKEND=mock  # or 'openai' or 'gemini'
export OPENAI_API_KEY=your_key_here  # if using OpenAI
export GEMINI_API_KEY=your_key_here  # if using Gemini
export FAITHFULNESS_THRESHOLD=0.6
export COHERENCE_THRESHOLD=0.5
```

### Running the Application

```bash
uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload
```

Access the API:
- **API Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/healthz
- **Reasoning Endpoint**: POST http://localhost:8000/api/v1/reason

### Running Tests

```bash
pytest src/tests/ -v
```

## API Usage

### Basic Reasoning Request

```bash
curl -X POST "http://localhost:8000/api/v1/reason?mode=dryrun" \
  -H "Content-Type: application/json" \
  -d '{
    "input": "What is the capital of France?",
    "context": {"domain": "geography"},
    "request_id": "test_123"
  }'
```

### Perturbation Analysis

```bash
curl -X POST "http://localhost:8000/api/v1/reason?mode=perturb" \
  -H "Content-Type: application/json" \
  -d '{
    "input": "Solve: 2 + 2 = ?",
    "context": {"domain": "math"}
  }'
```

### Production Mode

```bash
curl -X POST "http://localhost:8000/api/v1/reason?mode=live" \
  -H "Content-Type: application/json" \
  -d '{
    "input": "Explain quantum computing",
    "context": {"domain": "science", "level": "beginner"}
  }'
```

## Response Format

The API returns structured responses with:

```json
{
  "answer": "Paris",
  "reasoning_trace": [
    "Step 1: I need to identify the capital of France",
    "Step 2: France is a country in Europe", 
    "Step 3: The capital of France is Paris"
  ],
  "faithfulness_score": 0.85,
  "coherence_score": 0.92,
  "risk_flag": false,
  "monitor_explanation": "High faithfulness and coherence scores indicate reliable reasoning",
  "metadata": {
    "request_id": "req_123",
    "n_candidates": 3,
    "best_score": 0.88,
    "mode": "dryrun",
    "components": {
      "counterfactual_influence": 0.8,
      "step_entailment": 0.9,
      "coherence": 0.92,
      "obfuscation": 0.1
    }
  },
  "perturbation": {
    "original_answer": "Paris",
    "perturbed_answers": [
      {
        "removed_steps": [1],
        "new_answer": "Paris", 
        "changed": false
      }
    ],
    "causal_influence_score": 0.3
  }
}
```

## Processing Modes

### Live Mode (`mode=live`)
- Normal production processing
- Returns HTTP 409 if risk is detected
- Blocks potentially unsafe or low-quality reasoning

### Dry Run Mode (`mode=dryrun`)
- Safe testing mode
- Returns results even if risk is detected
- Useful for development and testing

### Perturbation Mode (`mode=perturb`)
- Includes perturbation analysis in response
- Systematically removes reasoning steps
- Measures causal influence on final answer

## Configuration

Key environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_BACKEND` | `mock` | Model backend: `mock`, `openai`, or `gemini` |
| `OPENAI_API_KEY` | - | Required for OpenAI backend |
| `GEMINI_API_KEY` | - | Required for Gemini backend |
| `FAITHFULNESS_THRESHOLD` | `0.6` | Risk threshold for faithfulness |
| `COHERENCE_THRESHOLD` | `0.5` | Risk threshold for coherence |
| `N_CANDIDATES` | `3` | Number of reasoning candidates |
| `PERTURB_STEPS_MAX` | `2` | Max steps to remove in perturbations |
| `LOG_LEVEL` | `INFO` | Logging level |

## Architecture

### Core Components

- **ReasoningAgent**: Orchestrates the full reasoning pipeline
- **ReasoningPipeline**: Handles CoT generation and perturbation experiments
- **CoTMonitor**: Assesses faithfulness, coherence, and risk
- **ModelAdapter**: Abstract interface for different AI backends
- **MockModelAdapter**: Deterministic adapter for testing
- **OpenAIModelAdapter**: Production adapter using OpenAI API
- **GeminiModelAdapter**: Production adapter using Gemini API

### Monitoring Components

1. **Faithfulness Score**: Combines counterfactual influence and step entailment
2. **Coherence Score**: Measures internal logical consistency
3. **Risk Detection**: Flags unsafe content, obfuscation, and low scores
4. **Perturbation Analysis**: Systematic experiments to measure causal influence

## Development

### Project Structure

```
src/
├── main.py                 # FastAPI application
├── core/
│   ├── config.py          # Configuration management
│   ├── models.py          # Pydantic models
│   ├── pipeline.py        # Reasoning pipeline
│   └── monitor.py         # CoT monitoring
├── agents/
│   └── reasoning_agent.py  # Main reasoning agent
├── adapters/
│   └── model_adapter.py   # Model adapters
├── api/
│   └── routes.py          # API routes
├── utils/
│   └── logger.py          # Logging utilities
└── tests/                 # Test suite
```

### Adding New Model Adapters

1. Inherit from `ModelAdapter` base class
2. Implement required methods:
   - `generate(prompt, n) -> List[str]`
   - `score_text_entailment(premise, hypothesis) -> float`
   - `classify_obfuscation(text) -> float`
3. Add configuration in `config.py`
4. Update agent initialization in `main.py`

### Adding New Monitoring Components

1. Add new assessment method to `CoTMonitor`
2. Update `assess()` method to include new component
3. Update `_aggregate_faithfulness()` if needed
4. Add tests for new functionality

## Testing

The test suite covers:

- **Pipeline Tests**: CoT generation, parsing, and perturbation
- **Monitor Tests**: Faithfulness, coherence, and risk detection
- **Agent Tests**: Full reasoning pipeline orchestration
- **API Tests**: Endpoint behavior and error handling
- **Adapter Tests**: Model adapter functionality

Run specific test categories:

```bash
# Run all tests
pytest src/tests/ -v

# Run specific test files
pytest src/tests/test_pipeline.py -v
pytest src/tests/test_monitor.py -v
pytest src/tests/test_agent.py -v
pytest src/tests/test_api.py -v
```

## Error Handling

The API handles various error conditions:

- **503 Service Unavailable**: Agent not initialized or missing API keys
- **409 Conflict**: Risk detected in live mode
- **422 Unprocessable Entity**: Invalid request format
- **500 Internal Server Error**: Processing errors

## Production Considerations

- Use OpenAI or Gemini adapter for production workloads
- Set appropriate thresholds based on use case
- Monitor logs for risk detection patterns
- Consider rate limiting for OpenAI API calls
- Implement proper secret management
- Add metrics and monitoring dashboards

## License

This project implements research from "Chain of Thought Monitorability" (arXiv:2507.11473) as a prototype service.