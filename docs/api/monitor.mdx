---
title: Monitor API
description: "Track and analyze the performance of your reasoning tasks."
---

## The `/monitor` Endpoints

The `/monitor` endpoints provide the tools to track the usage, performance, and cost of your reasoning tasks. This is essential for understanding how Neuro is being used and for identifying areas for improvement.

### List Reasoning Events

This endpoint retrieves a paginated list of all reasoning events, sorted by timestamp.

**Endpoint**

`GET /v1/monitor/events`

**Query Parameters**

| Parameter | Type    | Description                                                                 |
|-----------|---------|-----------------------------------------------------------------------------|
| `limit`   | integer | The number of events to return. Defaults to `20`. Max `100`.                |
| `offset`  | integer | The number of events to skip for pagination.                                |
| `userId`  | string  | Filter events by a specific user ID.                                        |
| `model`   | string  | Filter events by the model used.                                            |

**Example Request**

```bash
curl -X GET "https://api.neuro.ai/v1/monitor/events?limit=10&userId=user-123" \
  -H "Authorization: Bearer YOUR_API_KEY"
```

**Response**

```json
{
  "events": [
    {
      "eventId": "evt-xyz789",
      "reasoningId": "res-abcde12345",
      "timestamp": "2024-10-26T10:00:00Z",
      "query": "Why is the sky blue?",
      "model": "default",
      "latencyMs": 1250,
      "cost": 0.0015
    }
    // ... more events
  ],
  "hasMore": true
}
```

### Get Reasoning Details

This endpoint retrieves the detailed record for a single reasoning task by its ID.

**Endpoint**

`GET /v1/monitor/reasoning/{reasoningId}`

**Example Request**

```bash
curl -X GET https://api.neuro.ai/v1/monitor/reasoning/res-abcde12345 \
  -H "Authorization: Bearer YOUR_API_KEY"
```

**Response**

The response for this endpoint is the same as the one from a `POST /v1/reason` call, providing the full chain of thought and other metadata.

### Related Sections

- [Metrics Internals →](../internals/metrics.md)
- [Evaluation API →](./eval.md)

