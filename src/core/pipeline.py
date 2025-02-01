"""
Reasoning pipeline for Chain of Thought generation and perturbation experiments.
Handles CoT generation, parsing, and systematic perturbation analysis.
"""

import re
import random
from typing import List, Dict, Any, Optional
from src.adapters.model_adapter import ModelAdapter
from src.core.models import PerturbationResult
from src.utils.logger import get_logger

logger = get_logger(logger_name=__name__)


class ReasoningPipeline:
    """Pipeline for generating and analyzing Chain of Thought reasoning."""
    
    # Prompt templates for consistent CoT generation
    COT_PROMPT_TEMPLATE = """You are a careful, explicit reasoner. When answering, enumerate your reasoning steps as 'Step 1: ...', 'Step 2: ...', etc. Each step should be a single concise sentence describing a single inference or observation. After steps, write a final 'Conclusion:' line with the final answer in one sentence.

Question: {input}
{context_section}"""

    def __init__(self, model_adapter: ModelAdapter, logger=None):
        """Initialize the reasoning pipeline."""
        self.model_adapter = model_adapter
        self.logger = logger or get_logger(logger_name=__name__)
    
    async def generate_cot(self, prompt: str, context: Optional[Dict[str, Any]] = None, n_candidates: int = 3) -> List[List[str]]:
        """
        Generate multiple Chain of Thought (CoT) reasoning traces for a given prompt.

        This method takes a prompt and context, then uses the configured model adapter
        to generate multiple candidate reasoning traces. Each trace is a sequence of
        explicit reasoning steps.

        Args:
            prompt: The input question or problem to reason about.
            context: Optional dictionary of key-value pairs providing additional context.
            n_candidates: The number of candidate traces to generate.

        Returns:
            A list of reasoning traces. Each trace is a list of strings, where each
            string is a single step in the reasoning process.
        """
        self.logger.debug(f"Generating {n_candidates} CoT candidates for prompt: {prompt[:100]}...")
        
        # Build the full prompt with context
        context_section = ""
        if context:
            context_items = [f"{k}: {v}" for k, v in context.items()]
            context_section = f"Context: {', '.join(context_items)}\n"
        
        full_prompt = self.COT_PROMPT_TEMPLATE.format(
            input=prompt,
            context_section=context_section
        )
        
        # Generate responses from model
        responses = await self.model_adapter.generate(full_prompt, n_candidates)
        
        # Parse each response into steps
        traces = []
        for i, response in enumerate(responses):
            try:
                steps = self._parse_cot_steps(response)
                traces.append(steps)
                self.logger.debug(f"Parsed {len(steps)} steps from candidate {i+1}")
            except Exception as e:
                self.logger.warning(f"Failed to parse CoT from candidate {i+1}: {e}")
                # Fallback: treat entire response as single step
                traces.append([response.strip()])
        
        return traces
    
    def _parse_cot_steps(self, response: str) -> List[str]:
        """Parse a CoT response into individual reasoning steps."""
        steps = []
        
        # Try to find numbered steps (Step 1:, Step 2:, etc.)
        step_pattern = r'Step\s+\d+:\s*([^\n]+)'
        matches = re.findall(step_pattern, response, re.IGNORECASE)
        
        if matches:
            steps.extend([match.strip() for match in matches])
        else:
            # Fallback: split on sentence boundaries
            sentences = re.split(r'[.!?]+', response)
            steps = [s.strip() for s in sentences if s.strip()]
        
        # Remove conclusion line if present
        steps = [step for step in steps if not step.lower().startswith('conclusion')]
        
        return steps if steps else [response.strip()]
    
    async def finalize_answer(self, trace: List[str], context: Optional[Dict[str, Any]] = None) -> str:
        """
        Generate a final answer from a reasoning trace.
        
        Args:
            trace: List of reasoning steps
            context: Additional context
            
        Returns:
            Final answer string
        """
        self.logger.debug(f"Finalizing answer from {len(trace)} steps")
        
        # If trace is empty, return empty string
        if not trace:
            return ""
        
        # Try to extract answer from last step if it looks like a conclusion
        last_step = trace[-1]
        if any(word in last_step.lower() for word in ['therefore', 'thus', 'so', 'answer is', 'result is']):
            return last_step
        
        # Otherwise, ask model to generate final answer
        trace_text = '\n'.join([f"Step {i+1}: {step}" for i, step in enumerate(trace)])
        
        prompt = f"""Based on the following reasoning steps, provide a concise final answer:

{trace_text}

Final Answer:"""
        
        try:
            response = await self.model_adapter.generate(prompt, 1)
            return response[0].strip()
        except Exception as e:
            self.logger.warning(f"Failed to generate final answer: {e}")
            return last_step  # Fallback to last step
    
    def perturb_trace(self, trace: List[str], removed_indices: List[int]) -> List[str]:
        """
        Create a perturbed trace by removing specified steps.
        
        Args:
            trace: Original reasoning trace
            removed_indices: Indices of steps to remove (0-based)
            
        Returns:
            New trace with specified steps removed
        """
        if not trace:
            return []
        
        # Create new trace excluding removed indices
        new_trace = []
        for i, step in enumerate(trace):
            if i not in removed_indices:
                new_trace.append(step)
        
        return new_trace
    
    async def run_perturbation_experiments(self, trace: List[str], context: Optional[Dict[str, Any]] = None, n_trials: int = 10) -> PerturbationResult:
        """
        Run systematic perturbation experiments on a reasoning trace.
        
        Args:
            trace: Original reasoning trace
            context: Additional context
            n_trials: Number of perturbation trials
            
        Returns:
            PerturbationResult with analysis
        """
        self.logger.debug(f"Running {n_trials} perturbation experiments on {len(trace)} steps")
        
        if not trace:
            return PerturbationResult(
                original_answer="",
                perturbed_answers=[],
                causal_influence_score=0.0
            )
        
        # Get original answer
        original_answer = await self.finalize_answer(trace, context)
        
        # Generate perturbation experiments
        perturbed_answers = []
        changed_count = 0
        
        for trial in range(n_trials):
            # Randomly select steps to remove (1 to min(3, len(trace)-1))
            max_remove = min(3, len(trace) - 1)
            num_to_remove = random.randint(1, max_remove)
            removed_indices = random.sample(range(len(trace)), num_to_remove)
            
            # Create perturbed trace
            perturbed_trace = self.perturb_trace(trace, removed_indices)
            
            # Get answer for perturbed trace
            try:
                perturbed_answer = await self.finalize_answer(perturbed_trace, context)
                changed = perturbed_answer.lower().strip() != original_answer.lower().strip()
                
                if changed:
                    changed_count += 1
                
                perturbed_answers.append({
                    "removed_steps": removed_indices,
                    "new_answer": perturbed_answer,
                    "changed": changed
                })
                
            except Exception as e:
                self.logger.warning(f"Failed to generate answer for perturbation {trial}: {e}")
                perturbed_answers.append({
                    "removed_steps": removed_indices,
                    "new_answer": "",
                    "changed": False
                })
        
        # Calculate causal influence score
        causal_influence_score = changed_count / len(perturbed_answers) if perturbed_answers else 0.0
        
        self.logger.debug(f"Causal influence score: {causal_influence_score:.3f} ({changed_count}/{len(perturbed_answers)} changed)")
        
        return PerturbationResult(
            original_answer=original_answer,
            perturbed_answers=perturbed_answers,
            causal_influence_score=causal_influence_score
        )