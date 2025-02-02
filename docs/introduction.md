---
title: Introduction
description: "Neuro: Chain-of-Thought as a Service for advanced reasoning and explanation generation."
---

## What is Neuro?

Neuro is a "Chain-of-Thought as a Service" platform that provides developers with a robust, scalable backend for multi-step reasoning. It exposes a clean set of REST APIs to generate, refine, and evaluate complex reasoning chains, making it easy to integrate advanced AI-driven reasoning into any application.

The core mission of Neuro is to democratize access to powerful reasoning capabilities that are otherwise complex to build and maintain. By offering reasoning as a service, Neuro enables developers to focus on their application's core logic while leveraging a managed infrastructure for AI-driven insights.

### Key Features

- **Reasoning API**: A simple yet powerful endpoint (`/reason`) to generate complete chain-of-thought explanations.
- **Evaluation Engine**: Endpoints (`/eval`) to score reasoning outputs for faithfulness, safety, and coherence.
- **Monitoring Suite**: A set of tools (`/monitor`) to track the performance, accuracy, and cost of reasoning tasks.
- **Extensible Architecture**: Easily plug in different large language models (LLMs) or define custom evaluation metrics.
- **Scalable & Reliable**: Built for production workloads, ensuring high availability and low latency.

### How It Works

Neuro orchestrates a series of calls to language models through a `ReasoningPipeline`. This pipeline manages the state of a reasoning task, from the initial prompt to the final refined explanation. A `ReasoningAgent` sits at the heart of this process, generating, validating, and refining the steps in a chain of thought.

The entire system is exposed via a REST API, allowing you to:
1.  **Submit a query** to the `/reason` endpoint.
2.  **Receive a structured reasoning chain** as a response.
3.  **Evaluate the output** using the `/eval` endpoint.
4.  **Track all operations** via the `/monitor` endpoints.

[Get started now →](../getting-started.md)
