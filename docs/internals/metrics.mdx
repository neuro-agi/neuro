---
title: Evaluation Metrics
description: "A guide to the scoring algorithms and metrics used in Neuro."
---

## Metrics and Scoring

Neuro provides a robust evaluation framework to measure the quality and safety of reasoning chains. The scoring logic is housed in the `src/core/monitor.py` module and exposed via the `/eval` API endpoint.

### Faithfulness Score

**Faithfulness** is a measure of how factually accurate the reasoning chain is. It ensures that the model does not invent facts or contradict the information it was given.

**Scoring Algorithm:**

1.  **Claim Extraction**: The system first identifies all factual claims made in the reasoning chain.
2.  **Source Verification**: Each claim is cross-referenced against the initial prompt and any information retrieved via tools during the reasoning process.
3.  **Score Calculation**: A score is calculated based on the percentage of claims that can be verified. Unsupported claims lower the score.

```python
# Simplified logic from src/core/monitor.py

def calculate_faithfulness(reasoning_chain: list) -> float:
    claims = extract_claims(reasoning_chain)
    sources = get_sources(reasoning_chain)
    
    verified_claims = 0
    for claim in claims:
        if is_verified(claim, sources):
            verified_claims += 1
            
    return verified_claims / len(claims) if claims else 1.0
```

### Safety Score

**Safety** ensures that the output is free of harmful, biased, or inappropriate content. This is critical for building responsible AI applications.

**Scoring Algorithm:**

1.  **Content Classification**: The final answer and each step in the chain are passed through a series of content classifiers.
2.  **Flagging**: The classifiers flag content related to hate speech, violence, self-harm, and other categories.
3.  **Score Aggregation**: A safety score is aggregated based on the presence and severity of any flagged content. A single high-severity flag can result in a score of 0.

```python
# Simplified logic from src/core/monitor.py

def calculate_safety(reasoning_chain: list) -> float:
    final_answer = reasoning_chain[-1]["result"]
    
    # Each classifier returns a score from 0 (unsafe) to 1 (safe)
    scores = [
        hate_speech_classifier.score(final_answer),
        violence_classifier.score(final_answer),
        sh_classifier.score(final_answer)
    ]
    
    # Return the minimum score
    return min(scores)
```

### Adding Custom Metrics

The evaluation system is designed to be extensible. To add a new metric, you need to:

1.  **Create a new scoring function** in `src/core/monitor.py`.
    - The function should accept a `reasoning_chain` (a list of dicts) as input and return a float between 0 and 1.
2.  **Register the new metric** in the `METRICS` dictionary.
    - This makes the metric available to the `/eval` endpoint.

**Example: Adding a "Conciseness" Metric**

```python
# 1. Define the scoring function
def calculate_conciseness(reasoning_chain: list) -> float:
    """Scores the response based on how concise it is."""
    num_words = len(reasoning_chain[-1]["result"].split())
    
    if num_words > 500:
        return 0.2
    if num_words > 200:
        return 0.7
    return 1.0

# 2. Register the metric
METRICS = {
    "faithfulness": calculate_faithfulness,
    "safety": calculate_safety,
    "conciseness": calculate_conciseness, # Add the new metric
}
```

Now you can request the `conciseness` metric when calling the `/eval` endpoint.

[See the Evaluation API →](../api/eval.md)
