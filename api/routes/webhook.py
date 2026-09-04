"""
ProofPilot — Razorpay Webhook Ingestion Router
------------------------------------------------
Accepts real or simulated Razorpay dispute webhook events, validates cryptographic
HMAC-SHA256 signature, parses nested dispute entities, and evaluates evidence risk.
"""

from typing import Any
from fastapi import APIRouter, Depends, HTTPException

from api.dependencies import get_all_reason_codes
from api.middleware.hmac_auth import verify_razorpay_signature_dependency
from api.schemas.webhook import RazorpayWebhookPayload
from scoring.scorer import score_dispute

router = APIRouter(prefix="/webhook", tags=["Razorpay Webhook"])

# Mapping reason codes to reason categories
REASON_CODE_TO_CATEGORY = {
    "4553": "goods_not_received",
    "4554": "product_not_as_described",
    "4513": "refund_not_processed",
    "4540": "unauthorized_fraud",
    "4521": "duplicate_charge",
    "U001": "upi_credit_failed",
    "U005": "upi_autopay_goods_not_received",
    "U008": "upi_fraudulent_collect",
}


@router.post("/dispute", dependencies=[Depends(verify_razorpay_signature_dependency)])
def handle_dispute_webhook(
    payload: RazorpayWebhookPayload,
    reason_codes: dict[str, Any] = Depends(get_all_reason_codes),
) -> dict[str, Any]:
    """
    Ingest and score Razorpay dispute webhook event.
    Automatically assigns category and routes to auto-draft or human review.
    """
    dispute_entity = payload.payload.get("dispute", {}).get("entity", {})
    if not dispute_entity:
        raise HTTPException(status_code=400, detail="Missing dispute.entity in webhook payload.")

    reason_code = str(dispute_entity.get("reason_code", "4553"))
    category = REASON_CODE_TO_CATEGORY.get(reason_code, "goods_not_received")

    amount_paise = dispute_entity.get("amount", 100000)
    amount_inr = float(amount_paise) / 100.0

    dispute_case = {
        "dispute_id": dispute_entity.get("id", "disp_unknown"),
        "merchant_id": payload.account_id,
        "merchant_name": "Razorpay Merchant",
        "network": "UPI" if reason_code.startswith("U") else "CARD",
        "reason_code": reason_code,
        "reason_category": category,
        "transaction": {
            "amount": amount_inr,
            "amount_paise": amount_paise,
            "currency": dispute_entity.get("currency", "INR"),
            "payment_id": dispute_entity.get("payment_id", "pay_unknown"),
            "payment_method": "upi" if reason_code.startswith("U") else "card",
        },
        "evidence_documents": {},  # Newly created dispute has no evidence uploaded yet
    }

    try:
        scoring_result = score_dispute(dispute_case, reason_codes, log_audit=True)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Scoring error: {str(exc)}")

    return {
        "status": "success",
        "event": payload.event,
        "account_id": payload.account_id,
        "dispute_id": scoring_result.get("dispute_id", dispute_case["dispute_id"]),
        "routing_decision": scoring_result.get("routing_decision", "human_review"),
        "risk_level": scoring_result.get("risk_level", "HIGH"),
        "completeness_score": scoring_result.get("completeness_score", 0.0),
        "completeness_pct": scoring_result.get("completeness_pct", "0%"),
        "confidence": scoring_result.get("confidence", 0.0),
        "win_probability": scoring_result.get("win_probability", 0.0),
        "win_probability_pct": scoring_result.get("win_probability_pct", "0%"),
        "economic_recommendation": scoring_result.get("economic_recommendation", {}),
        "expected_financial_value": scoring_result.get("expected_financial_value", {}),
        "missing_evidence": scoring_result.get("missing_evidence", []),
        "weak_evidence": scoring_result.get("weak_evidence", []),
        "gap_explanation": scoring_result.get("gap_explanation", []),
        "webhook_verified": True,
    }
