---
title: Evaluation API
description: "Score reasoning chains for faithfulness, safety, and other metrics."
---

## The `/eval` Endpoint

The `/eval` endpoint allows you to score a completed reasoning chain against various quality and safety metrics. This is crucial for building trust in your application and for filtering out low-quality or unsafe responses.

### Endpoint

`POST /v1/eval`

### Request Body

| Parameter     | Type   | Required | Description                                                                 |
|---------------|--------|----------|-----------------------------------------------------------------------------|
| `reasoningId` | string | Yes      | The ID of the reasoning chain to evaluate.                                  |
| `metrics`     | array  | Yes      | A list of metrics to score. Supported values: `faithfulness`, `safety`.     |

### Example Request

```bash
curl -X POST https://api.neuro.ai/v1/eval \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "reasoningId": "res-abcde12345",
    "metrics": ["faithfulness", "safety"]
  }'
```

### Response Body

The response provides a score for each requested metric, typically on a scale of 0 to 1.

```json
{
  "evalId": "eval-lmnop67890",
  "reasoningId": "res-abcde12345",
  "scores": [
    {
      "metric": "faithfulness",
      "score": 0.92,
      "explanation": "The reasoning chain correctly uses the provided information and does not introduce unsupported facts."
    },
    {
      "metric": "safety",
      "score": 0.99,
      "explanation": "The response is free of harmful or inappropriate content."
    }
  ]
}
```

## Available Metrics

### Faithfulness

**Faithfulness** measures how well the reasoning chain adheres to the facts provided in the initial prompt or retrieved during the reasoning process. A high faithfulness score means the model is not "hallucinating" or making up information.

### Safety

**Safety** measures whether the response is free from harmful, unethical, or inappropriate content. Neuro uses a combination of classifiers and content filters to generate this score.

### Custom Metrics

Neuro is designed to be extensible. You can define your own evaluation modules to score responses based on custom business logic.

[Learn more about custom metrics →](../internals/metrics.md)
