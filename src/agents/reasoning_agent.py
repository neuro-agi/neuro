"""
Reasoning agent that orchestrates the full reasoning pipeline.
Handles request processing, candidate generation, monitoring, and response formatting.
"""

import uuid
from typing import Optional, Dict, Any
from src.adapters.model_adapter import ModelAdapter
from src.core.models import ReasoningRequest, ReasoningResponse, PerturbationResult
from src.core.pipeline import ReasoningPipeline
from src.core.monitor import CoTMonitor
from src.utils.logger import get_logger

logger = get_logger(__name__)


class MonitoringError(Exception):
    """Exception raised when monitoring detects risk in live mode."""
    pass


class ReasoningAgent:
    """Main reasoning agent that orchestrates the reasoning pipeline."""
    
    def __init__(self, model_adapter: ModelAdapter, monitor: CoTMonitor, n_candidates: Optional[int] = None):
        """Initialize the reasoning agent."""
        self.model_adapter = model_adapter
        self.monitor = monitor
        self.pipeline = ReasoningPipeline(model_adapter)
        self.n_candidates = n_candidates or 3
        self.logger = get_logger(__name__)
    
    async def reason(self, request: ReasoningRequest, mode: str = "live") -> ReasoningResponse:
        """
        Process a reasoning request and return a response.
        
        Args:
            request: The reasoning request
            mode: Processing mode (live, dryrun, perturb)
            
        Returns:
            ReasoningResponse with results
            
        Raises:
            MonitoringError: If risk is detected in live mode
        """
        # Generate or use request ID
        request_id = request.request_id or str(uuid.uuid4())
        self.logger.info(f"Processing reasoning request {request_id} in {mode} mode")
        
        try:
            # 1. Generate candidate reasoning traces
            traces = await self.pipeline.generate_cot(
                request.input, 
                request.context, 
                self.n_candidates
            )
            
            if not traces:
                raise ValueError("No reasoning traces generated")
            
            # 2. Evaluate each candidate
            candidates = []
            for i, trace in enumerate(traces):
                self.logger.debug(f"Evaluating candidate {i+1} with {len(trace)} steps")
                
                # Generate final answer for this trace
                answer = await self.pipeline.finalize_answer(trace, request.context)
                
                # Assess the trace
                assessment = await self.monitor.assess(trace, answer, request.context)
                
                # Calculate composite score
                score = self._calculate_candidate_score(assessment)
                
                candidates.append({
                    'trace': trace,
                    'answer': answer,
                    'assessment': assessment,
                    'score': score
                })
            
            # 3. Select best candidate
            best_candidate = max(candidates, key=lambda c: c['score'])
            self.logger.info(f"Selected best candidate with score {best_candidate['score']:.3f}")
            
            # 4. Handle perturbation experiments if requested
            perturbation_result = None
            if mode == "perturb":
                self.logger.debug("Running perturbation experiments")
                perturbation_result = await self.pipeline.run_perturbation_experiments(
                    best_candidate['trace'], 
                    request.context
                )
            
            # 5. Check for risk in live mode
            if mode == "live" and best_candidate['assessment']['risk_flag']:
                self.logger.warning(f"Risk detected for request {request_id}")
                raise MonitoringError(
                    f"Risk detected: {best_candidate['assessment']['monitor_explanation']}"
                )
            
            # 6. Build response
            response = ReasoningResponse(
                answer=best_candidate['answer'],
                reasoning_trace=best_candidate['trace'],
                faithfulness_score=best_candidate['assessment']['faithfulness_score'],
                coherence_score=best_candidate['assessment']['coherence_score'],
                risk_flag=best_candidate['assessment']['risk_flag'],
                monitor_explanation=best_candidate['assessment']['monitor_explanation'],
                metadata={
                    'request_id': request_id,
                    'n_candidates': len(candidates),
                    'best_score': best_candidate['score'],
                    'mode': mode,
                    'components': best_candidate['assessment']['components']
                },
                perturbation=perturbation_result
            )
            
            self.logger.info(f"Successfully processed request {request_id}")
            return response
            
        except Exception as e:
            self.logger.error(f"Error processing request {request_id}: {e}")
            raise
    
    def _calculate_candidate_score(self, assessment: Dict[str, Any]) -> float:
        """Calculate composite score for a candidate."""
        faithfulness = assessment['faithfulness_score']
        coherence = assessment['coherence_score']
        risk_penalty = 0.3 if assessment['risk_flag'] else 0.0
        
        # Weighted combination with risk penalty
        score = (faithfulness * 0.6 + coherence * 0.4) - risk_penalty
        return max(0.0, min(1.0, score))