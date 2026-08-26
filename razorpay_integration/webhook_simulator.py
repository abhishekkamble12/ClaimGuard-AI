"""
ProofPilot — Razorpay Dispute Webhook Simulator
------------------------------------------------
Simulates Razorpay Dispute Webhook events (e.g. dispute.created, dispute.under_review).
"""

from typing import Any


def simulate_razorpay_dispute_event(dispute_case: dict[str, Any], event_type: str = "dispute.created") -> dict[str, Any]:
    """
    Wrap a dispute case into an authentic Razorpay Webhook Payload structure.
    Ref: Razorpay Dispute API Specs
    """
    transaction = dispute_case.get("transaction", {})
    amount_paise = transaction.get("amount_paise", transaction.get("amount", 0) * 100)
    
    return {
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
