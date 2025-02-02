---
title: Reasoning Agent
description: "Understanding the core logic of the ReasoningAgent."
---

## The `ReasoningAgent` Class

The `ReasoningAgent` is the brain of the Neuro platform. It is responsible for the core logic of generating, refining, and validating each step in a chain of thought. It takes the current state from the `ReasoningPipeline` and produces the next thought, action, and result.

### Core Logic: The "Thought-Action-Result" Loop

The agent operates on a simple but powerful loop:

1.  **Thought**: Given the user's query and the history of previous steps, the agent first generates a `thought`. This is a piece of internal monologue that outlines the plan for the next step. For example: *"I need to find the population of the city I just identified."*
2.  **Action**: Based on the thought, the agent decides on an `action`. This could be an internal action (like synthesizing information) or an external one (like calling a search tool).
3.  **Result**: The `ReasoningPipeline` executes the action and returns a `result` to the agent. The agent then uses this result to inform the next cycle of the loop.

This cycle repeats until the agent determines that it has enough information to form a final answer.

### Key Responsibilities

- **Chain-of-Thought Generation**: The agent's primary job is to produce the step-by-step reasoning chain.
- **Refinement**: The agent can revise its previous thoughts or actions if it encounters new information that contradicts its current path.
- **Validation**: It performs basic checks to ensure its reasoning is logical and coherent.
- **Termination**: The agent decides when the reasoning is complete and signals to the pipeline to stop.

### Python Example (`src/agents/reasoning_agent.py`)

Below is a simplified implementation of the `ReasoningAgent`.

```python
class ReasoningAgent:
    def __init__(self, model: ModelAdapter):
        self.model = model

    def run(self, state: str) -> tuple[str, str, str, bool]:
        """Generates the next thought, action, and result."""
        
        # 1. Generate a thought based on the current state
        thought_prompt = self._create_thought_prompt(state)
        thought = self.model.generate(thought_prompt)

        # 2. Formulate an action based on the thought
        action_prompt = self._create_action_prompt(state, thought)
        action = self.model.generate(action_prompt)

        # 3. Execute the action (simplified for this example)
        # In a real scenario, this would involve tool use.
        result = self._execute_action(action)

        # 4. Check for termination condition
        is_final = self._check_if_final(result)

        return thought, action, result, is_final

    def _create_thought_prompt(self, state: str) -> str:
        # ... prompt engineering to encourage thinking
        pass

    def _create_action_prompt(self, state: str, thought: str) -> str:
        # ... prompt engineering to generate a concrete action
        pass

    def _execute_action(self, action: str) -> str:
        # ... placeholder for tool execution logic
        return f"Result of action: {action}"

    def _check_if_final(self, result: str) -> bool:
        # ... logic to determine if the answer is complete
        return "final answer" in result.lower()

```

### Customizing the Agent

The `ReasoningAgent` can be swapped out or modified to support different reasoning strategies. For example, you could create a specialized agent for mathematical reasoning or one that is optimized for a specific domain.

[See the Reasoning Pipeline →](./pipeline.md)
