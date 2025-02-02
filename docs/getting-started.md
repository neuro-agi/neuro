---
title: Getting Started
description: "Set up Neuro and make your first API call in minutes."
---

## Overview

This guide will walk you through the basics of setting up Neuro and making your first API call. We provide examples using both `cURL` and our Python SDK.

### Prerequisites

- A valid Neuro API key.
- Python 3.8+ (for the Python SDK).

## Installation

To run Neuro locally, clone the repository and install the required dependencies:

```bash
git clone https://github.com/neuro-ai/neuro.git
cd neuro
pip install -r requirements.txt
```

To use the Python SDK, install it via pip:

```bash
pip install neuro-sdk
```

## Making Your First API Call

Let's generate a simple reasoning chain. The goal is to answer the question: "What is the capital of France and what is its population?"

### Using cURL

You can call the `/reason` endpoint directly using `cURL`.

```bash
curl -X POST https://api.neuro.ai/v1/reason \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is the capital of France and what is its population?",
    "model": "default",
    "steps": 3
  }'
```

**Response:**

```json
{
  "reasoningId": "res-12345",
  "chain": [
    {
      "step": 1,
      "thought": "The user is asking for two pieces of information: the capital of France and its population. I should first identify the capital.",
      "action": "Search for the capital of France.",
      "result": "The capital of France is Paris."
    },
    {
      "step": 2,
      "thought": "Now that I know the capital is Paris, I need to find its population.",
      "action": "Search for the population of Paris.",
      "result": "The population of Paris in 2024 is approximately 2.1 million."
    },
    {
      "step": 3,
      "thought": "I have both pieces of information. I can now form the final answer.",
      "action": "Synthesize the final answer.",
      "result": "The capital of France is Paris, and its population is approximately 2.1 million."
    }
  ],
  "finalAnswer": "The capital of France is Paris, and its population is approximately 2.1 million."
}
```

### Using the Python SDK

Here is the same example using our official Python SDK.

```python
import os
from neuro_sdk import NeuroClient

# It's recommended to set your API key as an environment variable
client = NeuroClient(api_key=os.getenv("NEURO_API_KEY"))

response = client.reasoning.generate(
    query="What is the capital of France and what is its population?",
    model="default",
    steps=3
)

print(response.final_answer)
# Output: The capital of France is Paris, and its population is approximately 2.1 million.

# You can also inspect the full chain of thought
for step in response.chain:
    print(f"Step {step.step}: {step.thought}")
```

## Next Steps

- Explore the [Reasoning API →](../api/reasoning.md) in more detail.
- Learn about our [evaluation and monitoring →](../api/eval.md) endpoints.
- Dive into the [internal architecture →](../internals/pipeline.md) of Neuro.
