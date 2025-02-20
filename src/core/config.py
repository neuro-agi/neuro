"""
Configuration module for Reasoning-as-a-Service (RaaS) prototype.
Handles environment-driven settings with validation and sensible defaults.
"""

import os
import logging
from typing import Dict, Any
from pydantic import BaseModel, field_validator


class Config(BaseModel):
    """Configuration settings loaded from environment variables."""
    
    # Model backend configuration
    model_backend: str = "mock"  # "mock", "openai", or "gemini"
    openai_api_key: str = ""  # Required if model_backend is "openai"
    gemini_api_key: str = ""  # Required if model_backend is "gemini"     
    # Monitoring thresholds for evaluating reasoning chains
    faithfulness_threshold: float = 0.6  # For checking if the model's reasoning aligns with the source
    coherence_threshold: float = 0.5  # For checking if the model's reasoning is logical
    
    # Perturbation settings for testing robustness
    perturb_steps_max: int = 2  # Maximum number of steps for perturbation
    n_candidates: int = 3  # Number of candidates to generate for perturbation
    
    # Logging configuration
    log_level: str = "INFO"  # Logging level (e.g., "DEBUG", "INFO", "WARNING")
    
    model_config = {
        "env_file": ".env",
        "case_sensitive": False
    }
    
    @field_validator('model_backend')
    @classmethod
    def validate_model_backend(cls, v):
        """Validate model backend is 'mock', 'openai', or 'gemini'."""
        if v not in ['mock', 'openai', 'gemini']:
            raise ValueError("MODEL_BACKEND must be 'mock', 'openai', or 'gemini'")
        return v
    
    @field_validator('faithfulness_threshold', 'coherence_threshold')
    @classmethod
    def validate_thresholds(cls, v):
        """Validate thresholds are between 0 and 1."""
        if not 0.0 <= v <= 1.0:
            raise ValueError("Thresholds must be between 0.0 and 1.0")
        return v
    
    @field_validator('perturb_steps_max', 'n_candidates')
    @classmethod
    def validate_positive_integers(cls, v):
        """Validate positive integer values."""
        if v < 1:
            raise ValueError("Must be positive integer")
        return v
    
    @field_validator('log_level')
    @classmethod
    def validate_log_level(cls, v):
        """Validate log level is valid."""
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if v.upper() not in valid_levels:
            raise ValueError(f"LOG_LEVEL must be one of {valid_levels}")
        return v.upper()
    
    def get_model_adapter_config(self) -> Dict[str, Any]:
        """
        Get configuration for model adapter initialization.
        
        Returns:
            A dictionary with the model adapter configuration.
        """
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
                "model": "gemini-pro"
            }
        return {}
    
    def get_monitor_config(self) -> Dict[str, Any]:
        """Get configuration for monitor initialization."""
        return {
            "faithfulness_threshold": self.faithfulness_threshold,
            "coherence_threshold": self.coherence_threshold,
        }


# Global config instance - load from environment variables
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

# Configure logging
logging.basicConfig(
    level=getattr(logging, config.log_level),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
