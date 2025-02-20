"""
API routes for Reasoning-as-a-Service (RaaS).
Handles reasoning requests with different modes and error handling.
"""

from fastapi import APIRouter, HTTPException, Query
from typing import Literal, Optional
from src.core.models import (
    ReasoningRequest, 
    ReasoningResponse, 
    ErrorResponse, 
    HealthResponse,
    ReasoningEventList,
    ReasoningDetails
)
from src.agents.reasoning_agent import ReasoningAgent, MonitoringError
from src.core import events
from src.utils.logger import get_logger
import time

logger = get_logger(__name__)

# Create router
router = APIRouter(prefix="/api/v1")

# Global agent instance (will be initialized in main.py)
agent: ReasoningAgent = None
start_time = time.time()


def set_agent(reasoning_agent: ReasoningAgent):
    """Set the global agent instance."""
    global agent
    agent = reasoning_agent


@router.post("/reason", response_model=ReasoningResponse)
async def reason(
    request: ReasoningRequest,
    mode: Literal["live", "dryrun", "perturb"] = Query(default="live", description="Processing mode")
):
    """
    Process a reasoning request.
    
    Args:
        request: The reasoning request
        mode: Processing mode
            - live: Normal processing with risk detection (returns 409 on risk)
            - dryrun: Process but return results even with risk
            - perturb: Include perturbation experiments in response
    
    Returns:
        ReasoningResponse with reasoning results
        
    Raises:
        HTTPException: 409 if risk detected in live mode, 500 for other errors
    """
    if agent is None:
        raise HTTPException(status_code=503, detail="Reasoning agent not initialized")

    if not request.input:
        raise HTTPException(status_code=422, detail="Input must not be empty")
    
    logger.info(f"Received reasoning request in {mode} mode")
    
    try:
        response = await agent.reason(request, mode)
        logger.info(f"Successfully processed request in {mode} mode")
        return response
        
    except MonitoringError as e:
        logger.warning(f"Monitoring error in {mode} mode: {e}")
        if mode == "live":
            # Return 409 for risk in live mode
            error_response = ErrorResponse(
                code=409,
                message=str(e)
            )
            raise HTTPException(status_code=409, detail=error_response.dict())
        else:
            # In dryrun/perturb mode, return the response with risk flagged
            # This should not happen as the agent should handle this, but just in case
            logger.warning("Unexpected monitoring error in non-live mode")
            raise HTTPException(status_code=500, detail="Internal processing error")
            
    except Exception as e:
        logger.error(f"Unexpected error processing request: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Return service health status."""
    uptime = time.time() - start_time
    
    # Basic checks
    service_status = "ok"
    monitor_status = "ok"
    
    if agent is None:
        service_status = "degraded"
        monitor_status = "unavailable"
    
    # In a real system, you might add more checks here, e.g.,
    # - Check database connectivity
    # - Check model backend availability
    
    return HealthResponse(
        service=service_status,
        monitor=monitor_status,
        uptime_seconds=uptime
    )

@router.get("/monitor/events", response_model=ReasoningEventList)
async def list_reasoning_events(
    limit: int = Query(default=20, le=100),
    offset: int = Query(default=0),
    user_id: Optional[str] = Query(None, description="Filter events by user ID"),
    model: Optional[str] = Query(None, description="Filter events by model used")
):
    """
    Retrieve a paginated list of all reasoning events.
    """
    event_list = events.list_events(limit=limit, offset=offset, user_id=user_id, model=model)
    has_more = len(event_list) == limit
    return ReasoningEventList(events=event_list, has_more=has_more)


@router.get("/monitor/reasoning/{reasoningId}", response_model=ReasoningDetails)
async def get_reasoning_details(reasoningId: str):
    """
    Retrieve the detailed record for a single reasoning task.
    """
    response = events.get_reasoning_response(reasoningId)
    if response is None:
        raise HTTPException(status_code=404, detail="Reasoning ID not found")
    return response