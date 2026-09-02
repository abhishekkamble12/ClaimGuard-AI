"""
ProofPilot — Razorpay Webhook HMAC-SHA256 Authentication Middleware
---------------------------------------------------------------------
Enforces cryptographic payload verification for incoming Razorpay dispute events.
Defaults to STRICT verification in production environments.
"""

import os
from fastapi import Header, HTTPException, Request
from razorpay_integration.webhook_simulator import verify_webhook_signature
from utils.logging_config import get_logger

logger = get_logger(__name__)


async def verify_razorpay_signature_dependency(
    request: Request,
    x_razorpay_signature: str | None = Header(None, alias="X-Razorpay-Signature"),
) -> bool:
    """
    FastAPI dependency validating the X-Razorpay-Signature header against the request body.
    """
    secret = os.getenv("WEBHOOK_SECRET")
    if not secret:
        logger.critical('WEBHOOK_SECRET is missing – rejecting webhook verification')
        raise RuntimeError('Missing WEBHOOK_SECRET environment variable')
    enforce_auth = os.getenv("ENFORCE_WEBHOOK_AUTH", "true").lower() == "true"

    if not x_razorpay_signature:
        if enforce_auth:
            logger.warning("Rejected webhook request: Missing X-Razorpay-Signature header.")
            raise HTTPException(
                status_code=401,
                detail="Missing required X-Razorpay-Signature header.",
            )
        logger.warning("Unverified webhook accepted: ENFORCE_WEBHOOK_AUTH is false.")
        return True

    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON payload.")

    is_valid = verify_webhook_signature(body, x_razorpay_signature, secret)
    if not is_valid:
        logger.warning("Rejected webhook request: HMAC-SHA256 signature mismatch.")
        raise HTTPException(
            status_code=401,
            detail="Cryptographic HMAC-SHA256 signature verification failed.",
        )

    return True
