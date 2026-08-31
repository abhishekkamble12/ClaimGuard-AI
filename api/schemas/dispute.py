"""
ProofPilot — Pydantic Schemas for Dispute Ingestion & Responses
"""

from typing import Any
from pydantic import BaseModel, Field
from api.schemas.scoring import (
    EconomicRecommendationSchema,
    EvidenceElementSchema,
    ExpectedFinancialValueSchema,
    GapExplanationSchema,
)


class TransactionMetadataSchema(BaseModel):
    amount: float = Field(..., description="Transaction amount in INR")
    amount_paise: int | None = Field(None, description="Transaction amount in Paise")
    currency: str = Field("INR", description="Currency string")
    payment_method: str = Field("card", description="Payment method: card, upi, wallet, netbanking")
    payment_date: str | None = Field(None, description="ISO date of payment")
    order_id: str | None = Field(None, description="Razorpay order ID")
    payment_id: str | None = Field(None, description="Razorpay payment ID")


class DisputeCaseSchema(BaseModel):
    dispute_id: str = Field(..., description="Unique dispute identifier")
    merchant_id: str | None = Field(None, description="Merchant account identifier")
    merchant_name: str = Field("Merchant", description="Merchant business name")
    network: str = Field("CARD", description="Payment network: CARD, UPI, AMEX")
    reason_code: str = Field(..., description="Network dispute reason code")
    reason_category: str = Field(..., description="Standard category: goods_not_received, etc.")
    transaction: TransactionMetadataSchema = Field(..., description="Transaction metadata")
    evidence_documents: dict[str, str] = Field(default_factory=dict, description="Uploaded evidence text or URLs")


class DisputeScoreRequest(BaseModel):
    dispute: DisputeCaseSchema
    api_key: str | None = Field(None, description="Optional Google Gemini API key for live reasoning")
    use_ground_truth: bool = Field(False, description="Whether to bypass NLP extraction with ground truth labels")


class DisputeScoringResponse(BaseModel):
    dispute_id: str
    reason_code: str
    reason_category: str
    reason_title: str
    completeness_score: float
    completeness_pct: str
    confidence: float
    confidence_pct: str
    risk_level: str
    routing_decision: str
    missing_evidence: list[str]
    weak_evidence: list[str]
    evidence_elements: dict[str, EvidenceElementSchema]
    gap_explanation: list[GapExplanationSchema]
    win_probability: float
    win_probability_pct: str
    expected_financial_value: ExpectedFinancialValueSchema
    economic_recommendation: EconomicRecommendationSchema
    ml_model_auc: float
    ml_feature_importances: dict[str, float]
    decision_trace: list[dict[str, Any]]


class BatchScoreRequest(BaseModel):
    disputes: list[DisputeCaseSchema]
    api_key: str | None = None


class BatchScoreResponse(BaseModel):
    total_processed: int
    successful_scores: int
    results: list[DisputeScoringResponse]
