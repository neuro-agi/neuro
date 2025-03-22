"""
Security and authentication utilities.
"""

from fastapi import Security, HTTPException, status
from fastapi.security.api_key import APIKeyHeader
from src.core.config import config

API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

async def get_api_key(api_key: str = Security(api_key_header)):
    """
    Dependency to verify the API key.
    """
    if api_key == config.api_key:
        return api_key
    else:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Could not validate credentials",
        )
