"""
ProofPilot — ML Feature Engineering Pipeline
----------------------------------------------
Transforms raw dispute payloads, evidence status extractions, and transaction
metadata into a rich 22+ dimensional feature vector for ensemble win probability
prediction and risk modeling.
"""

import math
from typing import Any

import numpy as np

PAYMENT_METHODS = ["card", "upi", "wallet", "netbanking"]

REASON_CATEGORIES = [
    "goods_not_received",
    "product_not_as_described",
    "refund_not_processed",
    "unauthorized_fraud",
    "duplicate_charge",
    "upi_credit_failed",
    "upi_autopay_goods_not_received",
    "upi_fraudulent_collect",
]

# Historical base win rate priors for payment categories (Indian FinTech benchmark)
BASE_CATEGORY_WIN_RATES = {
    "goods_not_received": 0.58,
    "product_not_as_described": 0.42,
    "refund_not_processed": 0.65,
    "unauthorized_fraud": 0.35,
    "duplicate_charge": 0.72,
    "upi_credit_failed": 0.82,
    "upi_autopay_goods_not_received": 0.48,
    "upi_fraudulent_collect": 0.60,
}


def compute_evidence_entropy(evidence_elements: dict[str, Any]) -> float:
    """
    Calculate Shannon entropy of the evidence status distribution.
    Higher entropy indicates high uncertainty / mixed evidence signals.
    """
    if not evidence_elements:
        return 0.0

    counts: dict[str, int] = {}
    total = len(evidence_elements)
    for detail in evidence_elements.values():
        status = detail.get("status", "missing") if isinstance(detail, dict) else str(detail)
        counts[status] = counts.get(status, 0) + 1

    entropy = 0.0
    for count in counts.values():
        p = count / total
        if p > 0:
            entropy -= p * math.log2(p)
    return round(float(entropy), 4)


class DisputeFeatureExtractor:
    """
    Extracts structured, domain-grounded numerical and categorical features
    from dispute cases and scoring results.
    """

    def __init__(self, category_win_rates: dict[str, float] | None = None):
        self.category_win_rates = category_win_rates or BASE_CATEGORY_WIN_RATES

    def get_feature_names(self) -> list[str]:
        """Returns the complete ordered list of feature names."""
        base_names = [
            "Completeness Score",
            "Confidence Score",
            "Missing Count",
            "Weak Count",
            "Normalized Amount",
            "Log Amount",
            "Evidence Entropy",
            "Strongest Evidence Weight",
            "Weakest Gap Weight",
            "Has Critical Evidence",
            "Semantic Relevance Mean",
            "Semantic Relevance Min",
            "Category Base Win Rate",
        ]
        pm_names = [f"Method: {m}" for m in PAYMENT_METHODS]
        cat_names = [f"Category: {c}" for c in REASON_CATEGORIES]
        return base_names + pm_names + cat_names

    def extract_features(
        self,
        case: dict[str, Any],
        scoring_result: dict[str, Any] | None = None,
    ) -> list[float]:
        """
        Extract numerical feature vector for a dispute case.
        Handles both training dataset records and live scoring evaluation objects.
        """
        transaction = case.get("transaction", {})
        raw_amount = float(transaction.get("amount", 1000.0))

        # Core readiness & confidence
        if scoring_result:
            score = float(scoring_result.get("completeness_score", 0.5))
            confidence = float(scoring_result.get("confidence", 0.5))
            missing_evidence = scoring_result.get("missing_evidence", [])
            weak_evidence = scoring_result.get("weak_evidence", [])
            evidence_elements = scoring_result.get("evidence_elements", {})
        else:
            score = float(case.get("expected_completeness_score", 0.5))
            confidence = float(case.get("expected_confidence", 0.5))
            missing_evidence = case.get("missing_evidence", [])
            weak_evidence = case.get("weak_evidence", [])
            evidence_elements = case.get("ground_truth_evidence", {})

        missing_cnt = float(len(missing_evidence))
        weak_cnt = float(len(weak_evidence))

        # Amount transformations
        norm_amount = raw_amount / 10000.0
        log_amount = math.log1p(max(0.0, raw_amount)) / 10.0

        # Evidence quality signals
        if isinstance(evidence_elements, dict) and evidence_elements:
            entropy = compute_evidence_entropy(evidence_elements)
            
            present_weights = []
            gap_weights = []
            semantic_scores = []

            for eid, detail in evidence_elements.items():
                if isinstance(detail, dict):
                    status = detail.get("status", "missing")
                    weight = float(detail.get("weight", 0.2))
                    sem_rel = float(detail.get("semantic_relevance", 0.0))
                else:
                    status = str(detail)
                    weight = 0.2
                    sem_rel = 0.0

                if status == "present":
                    present_weights.append(weight)
                else:
                    gap_weights.append(weight)

                if sem_rel > 0:
                    semantic_scores.append(sem_rel)

            strongest_weight = max(present_weights) if present_weights else 0.0
            weakest_gap_weight = max(gap_weights) if gap_weights else 0.0
            has_critical = 1.0 if strongest_weight >= 0.25 else 0.0
            sem_mean = float(np.mean(semantic_scores)) if semantic_scores else (score * 0.5)
            sem_min = float(np.min(semantic_scores)) if semantic_scores else 0.0
        else:
            entropy = 0.5
            strongest_weight = score * 0.3
            weakest_gap_weight = (1.0 - score) * 0.3
            has_critical = 1.0 if score >= 0.7 else 0.0
            sem_mean = score * 0.4
            sem_min = 0.0

        category = case.get("reason_category", "goods_not_received")
        base_win_rate = self.category_win_rates.get(category, 0.50)

        # Categorical one-hot encoding
        pm = transaction.get("payment_method", "card")
        pm_encoded = [1.0 if pm == m else 0.0 for m in PAYMENT_METHODS]

        cat_encoded = [1.0 if category == c else 0.0 for c in REASON_CATEGORIES]

        features = [
            round(score, 4),
            round(confidence, 4),
            round(missing_cnt, 2),
            round(weak_cnt, 2),
            round(norm_amount, 4),
            round(log_amount, 4),
            round(entropy, 4),
            round(strongest_weight, 4),
            round(weakest_gap_weight, 4),
            has_critical,
            round(sem_mean, 4),
            round(sem_min, 4),
            round(base_win_rate, 4),
        ] + pm_encoded + cat_encoded

        return features
