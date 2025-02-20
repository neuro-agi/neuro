"""
Pydantic models for Reasoning-as-a-Service (RaaS) API.
Defines request/response schemas with validation.
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, validator


class ReasoningRequest(BaseModel):
    input: str = Field(..., description="The input question or problem to reason about")
    context: Optional[Dict[str, Any]] = Field(None, description="Additional context for reasoning")
    policy: Optional[str] = Field(None, description="Policy constraints for reasoning")
    request_id: Optional[str] = Field(None, description="Optional request ID for correlation")
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "input": "What is the capital of France?",
                "context": {"domain": "geography"},
                "policy": "be_concise",
                "request_id": "req_123"
            }
        }
    }


class PerturbationResult(BaseModel):
    original_answer: str = Field(..., description="The original answer before perturbation")
    perturbed_answers: List[Dict[str, Any]] = Field(..., description="List of perturbation results")
    causal_influence_score: float = Field(..., ge=0.0, le=1.0, description="Fraction of perturbations that changed the answer")
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "original_answer": "Paris",
                "perturbed_answers": [
                    {"removed_steps": [1], "new_answer": "Paris", "changed": False},
                    {"removed_steps": [2], "new_answer": "London", "changed": True}
                ],
                "causal_influence_score": 0.5
            }
        }
    }


class ErrorResponse(BaseModel):
    code: int = Field(..., description="Error code")
    message: str = Field(..., description="Error message")
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "code": 409,
                "message": "Risk detected: faithfulness score below threshold"
            }
        }
    }


class ReasoningResponse(BaseModel):
    answer: str = Field(..., description="The final answer")
    reasoning_trace: List[str] = Field(..., description="List of reasoning steps")
    faithfulness_score: float = Field(..., ge=0.0, le=1.0, description="Faithfulness score (0-1)")
    coherence_score: float = Field(..., ge=0.0, le=1.0, description="Coherence score (0-1)")
    risk_flag: bool = Field(..., description="Whether risk was detected")
    monitor_explanation: str = Field(..., description="Explanation of monitoring results")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    perturbation: Optional[PerturbationResult] = Field(None, description="Perturbation results (only in perturb mode)")
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "answer": "Paris",
                "reasoning_trace": [
                    "Step 1: I need to identify the capital of France",
                    "Step 2: France is a country in Europe",
                    "Step 3: The capital of France is Paris"
                ],
                "faithfulness_score": 0.85,
                "coherence_score": 0.92,
                "risk_flag": False,
                "monitor_explanation": "High faithfulness and coherence scores indicate reliable reasoning",
                "metadata": {"model": "gpt-3.5-turbo", "processing_time": 1.2}
            }
        }
    }

    @validator('faithfulness_score', 'coherence_score')
    def validate_scores(cls, v):
        if not 0.0 <= v <= 1.0:
            raise ValueError("Scores must be between 0.0 and 1.0")
        return v


class HealthResponse(BaseModel):
    service: str = Field(..., description="Service status")
    monitor: str = Field(..., description="Monitor status")
    uptime_seconds: float = Field(..., description="Service uptime in seconds")
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "service": "ok",
                "monitor": "ok",
                "uptime_seconds": 123.45
            }
        }
    }


class ReasoningEvent(BaseModel):
    event_id: str = Field(..., description="Unique event ID")
    reasoning_id: str = Field(..., description="Reasoning task ID")
    timestamp: str = Field(..., description="Event timestamp in ISO 8601 format")
    query: str = Field(..., description="The user query")
    model: str = Field(..., description="The model used for reasoning")
    latency_ms: float = Field(..., description="Processing latency in milliseconds")
    cost: float = Field(..., description="Estimated cost of the reasoning task")
    user_id: Optional[str] = Field(None, description="User ID associated with the request")


class ReasoningEventList(BaseModel):
    events: List[ReasoningEvent] = Field(..., description="List of reasoning events")
    has_more: bool = Field(..., description="Whether more events are available")


class ReasoningDetails(ReasoningResponse):
    pass
