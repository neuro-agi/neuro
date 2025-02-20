"""
In-memory storage for reasoning events and responses.
"""

from typing import List, Dict, Optional
from src.core.models import ReasoningEvent, ReasoningResponse

# In-memory storage for events and responses
reasoning_events: Dict[str, ReasoningEvent] = {}
reasoning_responses: Dict[str, ReasoningResponse] = {}


def record_event(event: ReasoningEvent, response: ReasoningResponse):
    """Record a new reasoning event and its corresponding response."""
    reasoning_events[event.reasoning_id] = event
    reasoning_responses[event.reasoning_id] = response


def list_events(
    limit: int = 20,
    offset: int = 0,
    user_id: Optional[str] = None,
    model: Optional[str] = None,
) -> List[ReasoningEvent]:
    """List all reasoning events with optional filtering."""
    filtered_events = list(reasoning_events.values())

    if user_id:
        filtered_events = [event for event in filtered_events if event.user_id == user_id]
    if model:
        filtered_events = [event for event in filtered_events if event.model == model]

    return filtered_events[offset : offset + limit]


def get_event_by_reasoning_id(reasoning_id: str) -> Optional[ReasoningEvent]:
    """Get a reasoning event by its reasoning ID."""
    return reasoning_events.get(reasoning_id)


def get_reasoning_response(reasoning_id: str) -> Optional[ReasoningResponse]:
    """Get a reasoning response by its reasoning ID."""
    return reasoning_responses.get(reasoning_id)
