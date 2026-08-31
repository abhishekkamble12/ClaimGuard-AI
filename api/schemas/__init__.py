"""
ProofPilot — Pydantic Schemas for API Serialization
"""

from api.schemas.dispute import (
    BatchScoreRequest,
    BatchScoreResponse,
    DisputeCaseSchema,
    DisputeScoreRequest,
    DisputeScoringResponse,
)
from api.schemas.scoring import (
    EconomicRecommendationSchema,
    EvidenceElementSchema,
    ExpectedFinancialValueSchema,
    GapExplanationSchema,
)
from api.schemas.webhook import RazorpayWebhookPayload

__all__ = [
    "RazorpayWebhookPayload",
    "DisputeCaseSchema",
    "DisputeScoreRequest",
    "DisputeScoringResponse",
    "BatchScoreRequest",
    "BatchScoreResponse",
    "EvidenceElementSchema",
    "EconomicRecommendationSchema",
    "ExpectedFinancialValueSchema",
    "GapExplanationSchema",
]
