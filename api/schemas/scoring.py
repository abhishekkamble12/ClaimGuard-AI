"""
ProofPilot — Pydantic Schemas for Scoring & ML Results
"""

from typing import Any
from pydantic import BaseModel, Field


class EvidenceElementSchema(BaseModel):
    status: str = Field(..., description="Evidence status: present, weak, missing, irrelevant")
    weight: float = Field(..., description="Reason-code checklist relative weight")
    score: float = Field(..., description="Raw score between 0.0 and 1.0")
    weighted_contribution: float = Field(..., description="Contribution to total readiness score")
    semantic_relevance: float = Field(0.0, description="TF-IDF cosine similarity to requirement specification")


class ExpectedFinancialValueSchema(BaseModel):
    amount_inr: float = Field(..., description="Disputed transaction amount in INR")
    win_probability: float = Field(..., description="Predicted win probability P(Win)")
    dispute_fee_inr: float = Field(500.0, description="Non-refundable dispute penalty fee in INR")
    expected_value_inr: float = Field(..., description="Monetary Expected Value EV in INR")
    is_positive_roi: bool = Field(..., description="Whether contesting has positive expected return")


class EconomicRecommendationSchema(BaseModel):
    action: str = Field(..., description="Recommended action: CONTEST or ACCEPT_LOSS")
    expected_gain_inr: float = Field(..., description="Projected gain or loss in INR")
    fee_saved_if_accepted: float = Field(0.0, description="Penalty fee saved if accepting loss")
    reason: str = Field(..., description="Human-readable financial rationale")


class GapExplanationSchema(BaseModel):
    evidence_id: str
    current_status: str
    potential_score_gain: float
    score_if_added: float
    projected_risk: str
    projected_route: str | None = None
    explanation: str | None = None
