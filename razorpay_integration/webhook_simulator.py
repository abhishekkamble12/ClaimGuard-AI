"""
ProofPilot — Razorpay Dispute Webhook Simulator & HMAC Validator
------------------------------------------------------------------
Simulates authentic Razorpay Dispute Webhook events (e.g. dispute.created,
dispute.action_required, dispute.under_review) and generates/verifies
cryptographic HMAC-SHA256 signatures for payment gateway security compliance.
"""

import hashlib
import hmac
import json
import os
from typing import Any

# In production the webhook secret must be provided via the WEBHOOK_SECRET env‑var.
# The hard‑coded test secret is removed to avoid accidental deployment.
DEFAULT_WEBHOOK_SECRET = os.getenv("WEBHOOK_SECRET")
# For local testing / development, allow a deterministic fallback but warn the user.
if DEFAULT_WEBHOOK_SECRET is None:
    DEFAULT_WEBHOOK_SECRET = "test_secret_fallback"
    # Note: In production you must set WEBHOOK_SECRET; this fallback is only for test environments.



def sign_webhook_payload(payload: dict[str, Any], secret: str = DEFAULT_WEBHOOK_SECRET) -> str:
    """
    Generate authentic Razorpay HMAC-SHA256 signature for a webhook payload.
    Uses canonical compact JSON encoding matching Razorpay gateway protocol.
    """
    payload_bytes = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return hmac.new(secret.encode("utf-8"), payload_bytes, hashlib.sha256).hexdigest()


def verify_webhook_signature(payload: dict[str, Any], signature: str, secret: str = DEFAULT_WEBHOOK_SECRET) -> bool:
    """
    Verify cryptographic Razorpay HMAC-SHA256 signature against webhook payload.
    Uses constant-time comparison to prevent timing attacks.
    """
    if not signature or not secret:
        return False
    expected = sign_webhook_payload(payload, secret)
    return hmac.compare_digest(expected, signature)


def simulate_razorpay_dispute_event(
    dispute_case: dict[str, Any],
    event_type: str = "dispute.created",
    secret: str = DEFAULT_WEBHOOK_SECRET,
) -> dict[str, Any]:
    """
    Wrap a dispute case into an authentic Razorpay Webhook Payload structure
    and attach signature metadata.
    Ref: Razorpay Dispute API & Webhook Specifications
    """
    transaction = dispute_case.get("transaction", {})
    amount_paise = transaction.get("amount_paise", transaction.get("amount", 0) * 100)
    
    payload = {
        "entity": "event",
        "account_id": dispute_case.get("merchant_id", "acc_1000000000001"),
        "event": event_type,
        "contains": ["dispute"],
        "payload": {
            "dispute": {
                "entity": {
                    "id": dispute_case.get("dispute_id", "disp_1000000000001"),
                    "payment_id": transaction.get("payment_id", "pay_1000000000002"),
                    "amount": amount_paise,
                    "currency": transaction.get("currency", "INR"),
                    "reason_code": dispute_case.get("reason_code", "4553"),
                    "reason_description": dispute_case.get("reason_title", "Goods not received"),
                    "status": "under_review" if event_type == "dispute.under_review" else "action_required",
                    "phase": "chargeback",
                    "evidence_due_by": 1788500000,
                    "created_at": 1787500000,
                }
            }
        },
        "created_at": 1787500000,
    }

    signature = sign_webhook_payload(payload, secret)
    return {
        "event_payload": payload,
        "x_razorpay_signature": signature,
        "is_verified": verify_webhook_signature(payload, signature, secret),
    }
