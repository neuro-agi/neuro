---
title: Reasoning API
description: "Use the /reason endpoint to generate chain-of-thought explanations."
---

## The `/reason` Endpoint

The `/reason` endpoint is the core of Neuro. It allows you to generate a multi-step reasoning chain to answer a given query. The endpoint orchestrates the `ReasoningPipeline` and `ReasoningAgent` to produce a structured, step-by-step explanation.

### Endpoint

`POST /v1/reason`

### Request Body

| Parameter | Type    | Required | Description                                                                 |
|-----------|---------|----------|-----------------------------------------------------------------------------|
| `query`   | string  | Yes      | The question or prompt you want the reasoning chain to address.             |
| `model`   | string  | No       | The ID of the reasoning model to use. Defaults to `default`.                |
| `steps`   | integer | No       | The maximum number of reasoning steps to generate. Defaults to `5`.         |
| `userId`  | string  | No       | An optional identifier for the end-user to help with monitoring and analytics. |

### Example Request

```bash
curl -X POST https://api.neuro.ai/v1/reason \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Why is the sky blue?",
    "steps": 4
  }'
```

### Response Body

The response contains the full chain of thought, including each step's thought process, the action taken, and the result.

```json
{
  "reasoningId": "res-abcde12345",
  "query": "Why is the sky blue?",
  "finalAnswer": "The sky appears blue because of Rayleigh scattering, where shorter blue wavelengths of light are scattered more effectively by the Earth's atmosphere than longer red wavelengths.",
  "chain": [
    {
      "step": 1,
      "thought": "The user is asking about the physics of why the sky is blue. I need to explain the role of the Earth's atmosphere and sunlight.",
      "action": "Explain the composition of sunlight and the atmosphere.",
      "result": "Sunlight is composed of a spectrum of colors (like a rainbow). The Earth's atmosphere is made of tiny gas molecules."
    },
    {
      "step": 2,
      "thought": "Now I need to connect the two concepts. The key phenomenon is light scattering.",
      "action": "Introduce the concept of Rayleigh scattering.",
      "result": "When sunlight enters the atmosphere, it collides with gas molecules. This causes the light to scatter in all directions. This is known as Rayleigh scattering."
    },
    {
      "step": 3,
      "thought": "I need to explain why blue light is scattered more than other colors.",
      "action": "Explain the relationship between wavelength and scattering.",
      "result": "Rayleigh scattering is more effective for shorter wavelengths of light. Blue and violet light have the shortest wavelengths in the visible spectrum."
    },
    {
      "step": 4,
      "thought": "Finally, I can synthesize the answer.",
      "action": "Conclude why this makes the sky appear blue to our eyes.",
      "result": "Because blue light is scattered most effectively, more of it reaches our eyes from all directions in the sky. This makes the sky appear blue."
    }
  ]
}
```

### Related Sections

- [Reasoning Pipeline Internals →](../internals/pipeline.md)
- [Reasoning Agent Logic →](../internals/agent.md)

