"""
API routes for neuro-specific endpoints.
"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from src.core.database import get_db
from src.core.security import get_api_key
from src.core import models

router = APIRouter(prefix="/api/v1/neuro", dependencies=[Depends(get_api_key)])


@router.post("/eval", response_model=models.NeuroEvaluationResponse)
async def neuro_eval(evaluation: models.NeuroEvaluation, db: Session = Depends(get_db)):
    """
    Save an evaluation to the database.
    """
    db_eval = models.Evaluation(**evaluation.dict())
    db.add(db_eval)
    db.commit()
    db.refresh(db_eval)
    return db_eval


@router.post("/reason", response_model=models.NeuroReasonResponse)
async def neuro_reason(reason: models.NeuroReason, db: Session = Depends(get_db)):
    """
    Simulate a task and log the result to the database.
    """
    # Simulate a long-running task
    result = f"Result for query: {reason.query}"
    db_reason = models.ReasoningLog(query=reason.query, result=result)
    db.add(db_reason)
    db.commit()
    db.refresh(db_reason)
    return db_reason
