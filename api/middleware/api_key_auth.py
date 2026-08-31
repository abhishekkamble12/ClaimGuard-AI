"""
ProofPilot — API Key Authentication Middleware & Dependency
-------------------------------------------------------------
Secures dispute scoring and audit retrieval endpoints from unauthorized access.
"""

import os
from fastapi import Header, HTTPException
from utils.logging_config import get_logger

logger = get_logger(__name__)


async def verify_api_key_dependency(
    x_api_key: str | None = Header(None, alias="X-API-Key"),
) -> str:
    """
    Validates the X-API-Key header against PROOFPILOT_API_KEY environment variable.
    """
    configured_key = os.getenv("PROOFPILOT_API_KEY")

    # If an API key is configured in the environment, enforce it strictly
    if configured_key:
        if not x_api_key:
            raise HTTPException(
                status_code=401,
                detail="Missing required X-API-Key authentication header.",
            )
        if x_api_key != configured_key:
            raise HTTPException(
                status_code=403,
                detail="Invalid API key provided in X-API-Key header.",
            )
        return x_api_key

    # In development mode when no key is explicitly configured, allow pass-through
    logger.debug("PROOFPILOT_API_KEY not configured in environment; running in open development mode.")
    return x_api_key or "dev_open_access"
