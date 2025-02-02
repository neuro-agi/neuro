---
title: Configuration
description: "Configuring Neuro using environment variables and model providers."
---

## Configuration System

Neuro uses a simple and powerful configuration system that relies on environment variables. All configuration is managed in the `src/core/config.py` module.

### Environment Variables

To configure Neuro, set the following environment variables before running the application. You can place them in a `.env` file for local development.

| Variable              | Description                                                                 |
|-----------------------|-----------------------------------------------------------------------------|
| `NEURO_API_KEY`       | Your secret key for authenticating with the Neuro API.                      |
| `LOG_LEVEL`           | The logging level to use. Can be `DEBUG`, `INFO`, `WARNING`, or `ERROR`.      |
| `DEFAULT_MODEL_ID`    | The ID of the default model to use for reasoning tasks.                     |
| `OPENAI_API_KEY`      | Your API key for OpenAI (if you are using an OpenAI model).                 |
| `ANTHROPIC_API_KEY`   | Your API key for Anthropic (if you are using a Claude model).               |
| `DATABASE_URL`        | The connection string for the database used for monitoring.                 |

### Model Provider Configuration

Neuro supports multiple large language model (LLM) providers through a `ModelAdapter` interface. The configuration for each provider is defined in `src/core/config.py`.

To add a new model, you need to:

1.  **Create a new model adapter** in `src/adapters/`.
2.  **Add the model configuration** to the `MODELS` dictionary in `src/core/config.py`.

#### Example: Adding a New OpenAI Model

Let's say you want to add support for OpenAI's `gpt-4-turbo`.

1.  **Ensure your `OpenAIAdapter` can handle it.** (This is likely already done).
2.  **Add the model to the `MODELS` dictionary.**

```python
# In src/core/config.py

from src.adapters.model_adapter import OpenAIAdapter, AnthropicAdapter

MODELS = {
    "default": {
        "adapter": OpenAIAdapter,
        "model_name": "gpt-4",
    },
    "claude-3-opus": {
        "adapter": AnthropicAdapter,
        "model_name": "claude-3-opus-20240229",
    },
    "gpt-4-turbo": { # Add the new model here
        "adapter": OpenAIAdapter,
        "model_name": "gpt-4-turbo-preview",
    },
}
```

Once this is done, you can specify `"model": "gpt-4-turbo"` when making calls to the `/reason` endpoint.

### Python Example (`src/core/config.py`)

Here is a simplified look at the configuration loader.

```python
import os
from dotenv import load_dotenv

# Load environment variables from a .env file
load_dotenv()

class Settings:
    # API and Logging
    NEURO_API_KEY: str = os.getenv("NEURO_API_KEY", "your-default-key")
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

    # Model Configuration
    DEFAULT_MODEL_ID: str = os.getenv("DEFAULT_MODEL_ID", "default")

    # Provider API Keys
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY")
    ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY")

    # Database
    DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///./neuro.db")

# Instantiate settings to be used across the application
settings = Settings()
```

This setup provides a single, consistent way to manage all configuration, making it easy to switch between development, staging, and production environments.
