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

    # If the required API key is missing we *reject* every request.
    # This forces the deployment to fail fast rather than opening an unrestricted back‑door.
    if not configured_key:
        logger.error('PROOFPILOT_API_KEY is not defined – rejecting request')
        raise HTTPException(status_code=401, detail='API key required')
