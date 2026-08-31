"""
ProofPilot — Pydantic Schemas for Razorpay Webhook Events
-----------------------------------------------------------
Complies with Razorpay Dispute Webhook Specification.
Ref: https://razorpay.com/docs/api/disputes/
"""

from typing import Any
from pydantic import BaseModel, Field


class RazorpayDisputeEntity(BaseModel):
    id: str = Field(..., description="Razorpay Dispute ID, e.g. disp_1000000000001")
    payment_id: str = Field(..., description="Razorpay Payment ID, e.g. pay_1000000000002")
    amount: int = Field(..., description="Dispute amount in paise (e.g. 499900 for ₹4,999)")
    currency: str = Field("INR", description="Currency code")
    reason_code: str = Field(..., description="Network reason code, e.g. 4553, U001")
    reason_description: str = Field(..., description="Dispute reason description")
    status: str = Field("action_required", description="Dispute state: action_required, under_review, won, lost")
    phase: str = Field("chargeback", description="Dispute phase: chargeback, retrieval, pre_arbitration")
    evidence_due_by: int | None = Field(None, description="Unix timestamp for evidence submission deadline")
    created_at: int | None = Field(None, description="Unix timestamp of dispute creation")


class RazorpayWebhookPayload(BaseModel):
    entity: str = Field("event", description="Top-level entity type")
    account_id: str = Field(..., description="Merchant Razorpay Account ID, e.g. acc_1000000000001")
    event: str = Field(..., description="Event type: dispute.created, dispute.action_required, dispute.under_review")
    contains: list[str] = Field(["dispute"], description="Entity types contained in the payload")
    payload: dict[str, Any] = Field(..., description="Nested entity dictionary")
    created_at: int | None = Field(None, description="Unix timestamp of webhook event dispatch")
