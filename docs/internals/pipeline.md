---
title: Reasoning Pipeline
description: "An in-depth look at the internal mechanics of the ReasoningPipeline."
---

## The `ReasoningPipeline` Class

The `ReasoningPipeline` is the orchestrator of the entire chain-of-thought process in Neuro. It is responsible for managing the state of a reasoning task, interacting with the selected language model, and ensuring the final output is coherent and complete.

### Core Responsibilities

1.  **State Management**: The pipeline initializes and tracks the state of the reasoning chain, including the initial query, each intermediate step, and the final answer.
2.  **Model Interaction**: It sends requests to the configured language model (via the `ModelAdapter`) and processes the responses.
3.  **Agent Coordination**: It invokes the `ReasoningAgent` to generate, refine, and validate each step of the reasoning process.
4.  **Error Handling**: It manages retries, timeouts, and other failures that can occur during a long-running reasoning task.
5.  **Output Formatting**: It assembles the final, structured response that is returned by the API.

### Execution Flow

A typical execution flow through the `ReasoningPipeline` looks like this:

1.  **Initialization**: The pipeline is instantiated with a `query`, `model`, and other parameters from the API request.
2.  **Loop Start**: The pipeline enters a loop that continues until a final answer is reached or the maximum number of steps is exceeded.
3.  **Agent Invocation**: In each iteration, the pipeline calls the `ReasoningAgent` with the current state.
4.  **Thought Generation**: The agent generates a `thought`—a hypothesis or plan for the next step.
5.  **Action Formulation**: Based on the thought, the agent formulates an `action` to take (e.g., call a tool, search for information).
6.  **Result Retrieval**: The pipeline executes the action and retrieves a `result`.
7.  **State Update**: The `thought`, `action`, and `result` are appended to the reasoning chain, and the state is updated.
8.  **Termination Check**: The pipeline checks if the agent has produced a final answer. If so, the loop terminates.
9.  **Finalization**: The pipeline formats the complete chain and the final answer into the API response.

### Python Example (`src/core/pipeline.py`)

Here is a simplified representation of the `ReasoningPipeline` class:

```python
class ReasoningPipeline:
    def __init__(self, query: str, model_id: str = "default", max_steps: int = 5):
        self.query = query
        self.model = ModelAdapter.get_instance(model_id)
        self.agent = ReasoningAgent(self.model)
        self.max_steps = max_steps
        self.chain = []
        self.final_answer = None

    def execute(self) -> dict:
        """Runs the full reasoning pipeline from query to final answer."""
        for step in range(self.max_steps):
            current_state = self._get_current_state()
            thought, action, result, is_final = self.agent.run(current_state)

            self.chain.append({
                "step": step + 1,
                "thought": thought,
                "action": action,
                "result": result
            })

            if is_final:
                self.final_answer = result
                break
        
        return self._format_response()

    def _get_current_state(self) -> str:
        """Constructs the current state from the query and previous steps."""
        # ... logic to format the history for the agent
        pass

    def _format_response(self) -> dict:
        """Assembles the final API response."""
        # ... logic to structure the JSON output
        pass
```

### Extensibility

The pipeline is designed to be extensible. You can plug in new model adapters or modify the `ReasoningAgent` to change the core reasoning logic without altering the orchestration flow.

[See the Reasoning Agent →](./agent.md)
