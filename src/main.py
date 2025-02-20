"""
Reasoning-as-a-Service (RaaS) FastAPI application.
Main entry point with startup/shutdown events and health checks.
"""

import time
import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src.core.config import config
from src.core.models import HealthResponse
from src.adapters.model_adapter import MockModelAdapter, OpenAIModelAdapter
from src.adapters.gemini_adapter import GeminiModelAdapter
from src.core.monitor import CoTMonitor
from src.agents.reasoning_agent import ReasoningAgent
from src.api.routes import router, set_agent
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Global variables for app state
start_time = time.time()
model_adapter = None
monitor = None
agent = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    global model_adapter, monitor, agent
    
    # Startup
    logger.info("Starting Reasoning-as-a-Service application")
    
    try:
        # Initialize model adapter
        if config.model_backend == "mock":
            model_adapter = MockModelAdapter()
            logger.info("Initialized MockModelAdapter")
        elif config.model_backend == "openai":
            adapter_config = config.get_model_adapter_config()
            model_adapter = OpenAIModelAdapter(**adapter_config)
            logger.info("Initialized OpenAIModelAdapter")
        elif config.model_backend == "gemini":
            adapter_config = config.get_model_adapter_config()
            model_adapter = GeminiModelAdapter(**adapter_config)
            logger.info("Initialized GeminiModelAdapter")
        else:
            raise ValueError(f"Unknown model backend: {config.model_backend}")
        
        # Initialize monitor
        monitor_config = config.get_monitor_config()
        monitor = CoTMonitor(model_adapter, monitor_config)
        logger.info("Initialized CoTMonitor")
        
        # Initialize agent
        agent = ReasoningAgent(model_adapter, monitor, config.n_candidates)
        set_agent(agent)
        logger.info("Initialized ReasoningAgent")
        
        logger.info("Application startup completed successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize application: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down application")
    
    shutdown_tasks = []
    if model_adapter and hasattr(model_adapter, 'close'):
        shutdown_tasks.append(model_adapter.close())
    
    if shutdown_tasks:
        try:
            await asyncio.wait_for(asyncio.gather(*shutdown_tasks), timeout=10.0)
            logger.info("Graceful shutdown of components completed")
        except asyncio.TimeoutError:
            logger.warning("Shutdown timed out. Forcing exit.")
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
    
    logger.info("Application shutdown completed")


# Create FastAPI app
app = FastAPI(
    title="Reasoning-as-a-Service (RaaS)",
    description="A prototype implementation of Chain of Thought monitoring and reasoning",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(router)


@app.get("/healthz", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint.
    Returns service status and uptime.
    """
    uptime = time.time() - start_time
    
    # Check if components are initialized
    if agent is None or monitor is None or model_adapter is None:
        raise HTTPException(
            status_code=503, 
            detail="Service not ready - components not initialized"
        )
    
    # Check model adapter health (for OpenAI, this would check API key)
    if config.model_backend == "openai" and not config.openai_api_key:
        raise HTTPException(
            status_code=503,
            detail="Service not ready - OpenAI API key not configured"
        )
    if config.model_backend == "gemini" and not config.gemini_api_key:
        raise HTTPException(
            status_code=503,
            detail="Service not ready - Gemini API key not configured"
        )
    
    return HealthResponse(
        service="ok",
        monitor="ok",
        uptime_seconds=round(uptime, 2)
    )


@app.get("/")
async def root():
    """Root endpoint with basic service information."""
    return {
        "service": "Reasoning-as-a-Service (RaaS)",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "health": "/healthz",
            "reason": "/api/v1/reason",
            "docs": "/docs"
        }
    }


# README snippet as requested
"""
# Reasoning-as-a-Service (RaaS) - Quick Start

## Running the Application

1. Install dependencies:
   ```bash
   pip install fastapi uvicorn pydantic httpx pytest
   ```

2. Set environment variables (optional):
   ```bash
   export MODEL_BACKEND=mock  # or 'openai' or 'gemini'
   export OPENAI_API_KEY=your_key_here  # if using OpenAI
   export GEMINI_API_KEY=your_key_here  # if using Gemini
   export FAITHFULNESS_THRESHOLD=0.6
   export COHERENCE_THRESHOLD=0.5
   ```

3. Run the application:
   ```bash
   uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload
   ```

4. Access the API:
   - API Documentation: http://localhost:8000/docs
   - Health Check: http://localhost:8000/healthz
   - Reasoning Endpoint: POST http://localhost:8000/api/v1/reason

## Running Tests

```bash
pytest src/tests/ -v
```

## Example Usage

### Dry Run Mode (Safe Testing)
```bash
curl -X POST "http://localhost:8000/api/v1/reason?mode=dryrun" \
  -H "Content-Type: application/json" \
  -d '{
    "input": "What is the capital of France?",
    "context": {"domain": "geography"},
    "request_id": "test_123"
  }'
```

### Perturbation Mode (Analysis)
```bash
curl -X POST "http://localhost:8000/api/v1/reason?mode=perturb" \
  -H "Content-Type: application/json" \
  -d '{
    "input": "Solve: 2 + 2 = ?",
    "context": {"domain": "math"}
  }'
```

### Live Mode (Production)
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
- `answer`: Final reasoning result
- `reasoning_trace`: Step-by-step reasoning
- `faithfulness_score`: How well steps support the answer (0-1)
- `coherence_score`: Internal consistency of reasoning (0-1)
- `risk_flag`: Whether risks were detected
- `monitor_explanation`: Human-readable assessment
- `metadata`: Additional processing information
- `perturbation`: Perturbation analysis (perturb mode only)

## Configuration

Key environment variables:
- `MODEL_BACKEND`: "mock", "openai", or "gemini"
- `OPENAI_API_KEY`: Required for OpenAI backend
- `GEMINI_API_KEY`: Required for Gemini backend
- `FAITHFULNESS_THRESHOLD`: Risk threshold for faithfulness (default: 0.6)
- `COHERENCE_THRESHOLD`: Risk threshold for coherence (default: 0.5)
- `N_CANDIDATES`: Number of reasoning candidates (default: 3)
- `PERTURB_STEPS_MAX`: Max steps to remove in perturbations (default: 2)
"""


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)