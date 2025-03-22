"""
Configuration module for Reasoning-as-a-Service (RaaS) prototype.
Handles environment-driven settings with validation and sensible defaults.
"""

import os
import logging
from typing import Dict, Any
from pydantic import BaseModel, validator
from dotenv import load_dotenv

load_dotenv()

class Config(BaseModel):
    """Configuration settings loaded from environment variables."""
    
    # Model backend configuration
    model_backend: str = "gemini"  # "mock", "openai", or "gemini"
    openai_api_key: str = ""  # Required if model_backend is "openai"
    gemini_api_key: str = "AIzaSyB4JlCxpvCfD6Nqsp6_fMH4A7nkUB3-W0E"  # Required if model_backend is "gemini"     
    # Monitoring thresholds for evaluating reasoning chains
    faithfulness_threshold: float = 0.6  # For checking if the model's reasoning aligns with the source
    coherence_threshold: float = 0.5  # For checking if the model's reasoning is logical
    
    # Perturbation settings for testing robustness
    perturb_steps_max: int = 2  # Maximum number of steps for perturbation
    n_candidates: int = 3  # Number of candidates to generate for perturbation
    
    # Logging configuration
    log_level: str = "INFO"  # Logging level (e.g., "DEBUG", "INFO", "WARNING")
    database_url: str = os.getenv("DATABASE_URL", "postgresql://user:password@localhost/neuro")
    api_key: str = "test-key"

    model_config = {
        "env_file": ".env",
        "case_sensitive": False
    }
    
    @validator('model_backend')
    def validate_model_backend(cls, v):
        if v not in ['mock', 'openai', 'gemini']:
            raise ValueError("MODEL_BACKEND must be 'mock', 'openai', or 'gemini'")
        return v
    
    @validator('faithfulness_threshold', 'coherence_threshold')
    def validate_thresholds(cls, v):
        if not 0.0 <= v <= 1.0:
            raise ValueError("Thresholds must be between 0.0 and 1.0")
        return v
    
    @validator('perturb_steps_max', 'n_candidates')
    def validate_positive_integers(cls, v):
        if v < 1:
            raise ValueError("Must be positive integer")
        return v
    
    @validator('log_level')
    def validate_log_level(cls, v):
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if v.upper() not in valid_levels:
            raise ValueError(f"LOG_LEVEL must be one of {valid_levels}")
        return v.upper()
    
    def get_model_adapter_config(self) -> Dict[str, Any]:
        if self.model_backend == "openai":
            if not self.openai_api_key:
                raise ValueError("OPENAI_API_KEY is required when MODEL_BACKEND=openai")
            return {
                "api_key": self.openai_api_key,
                "model": "gpt-3.5-turbo"
            }
        if self.model_backend == "gemini":
            if not self.gemini_api_key:
                raise ValueError("GEMINI_API_KEY is required when MODEL_BACKEND=gemini")
            return {
                "api_key": self.gemini_api_key,
                "model": "gemini-2.5-flash-lite"
            }
        return {}
    
    def get_monitor_config(self) -> Dict[str, Any]:
        return {
            "faithfulness_threshold": self.faithfulness_threshold,
            "coherence_threshold": self.coherence_threshold,
        }


config = Config(
    model_backend=os.getenv("MODEL_BACKEND", "mock"),
    openai_api_key=os.getenv("OPENAI_API_KEY", ""),
    gemini_api_key=os.getenv("GEMINI_API_KEY", ""),
    faithfulness_threshold=float(os.getenv("FAITHFULNESS_THRESHOLD", "0.6")),
    coherence_threshold=float(os.getenv("COHERENCE_THRESHOLD", "0.5")),
    perturb_steps_max=int(os.getenv("PERTURB_STEPS_MAX", "2")),
    n_candidates=int(os.getenv("N_CANDIDATES", "3")),
    log_level=os.getenv("LOG_LEVEL", "INFO")
)

logging.basicConfig(
    level=getattr(logging, config.log_level),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
