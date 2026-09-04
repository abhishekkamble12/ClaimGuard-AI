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

# Per-category merchant win rate priors — must stay in sync with data_generator.py
# REASON_CODES[category]["merchant_win_rate"].
# Source: Razorpay Chargeback Guide 2024, NPCI UDIR Circular RPA-2023/184,
#         Chargebacks911 Global Dispute Index 2024 (Indian BFSI segment).
# These are the SAME values used to generate expected_outcome in data_generator.py.
# Using inflated priors here was the root cause of the over-contesting bug:
# the model's strongest feature was telling it every category is a near-certain win.
BASE_CATEGORY_WIN_RATES = {
    "goods_not_received":              0.30,  # consumer-bias; delivery proof burden
    "product_not_as_described":        0.38,  # subjective; policy docs help
    "refund_not_processed":            0.35,  # merchant shows refund was initiated
    "unauthorized_fraud":              0.22,  # hardest; 3DS auth logs required
    "duplicate_charge":                0.55,  # easiest; two distinct txn IDs
    "upi_credit_failed":               0.45,  # RRN auto-resolution; clear paper trail
    "upi_autopay_goods_not_received":  0.30,  # NPCI UDIR favours consumer
    "upi_fraudulent_collect":          0.25,  # QR fraud hard to rebut
}

# Domain-grounded merchant risk scores (0.1 = lowest fraud/highest win rate, 0.8 = higher risk)
MERCHANT_RISK_TIERS: dict[str, float] = {
    "CloudKitchen Pro": 0.8,
    "Swiggy Bharat": 0.7,
    "Zepto Express": 0.7,
    "Blinkit Commerce": 0.7,
    "Zomato Dining": 0.6,
    "UrbanKart India": 0.5,
    "StyleForge Apparel": 0.5,
    "Nykaa Glam": 0.4,
    "Boat Lifestyle": 0.4,
    "Lenskart Retail": 0.4,
    "Mamaearth Naturals": 0.4,
    "Pepperfry Living": 0.4,
    "MakeMyTrip India": 0.3,
    "Tata Neu Digital": 0.3,
    "FitNest Wellness": 0.3,
    "CultFit Pass": 0.3,
    "LearnLoop EdTech": 0.2,
    "PhysicsWallah Ed": 0.2,
    "Unacademy Plus": 0.2,
    "Razorpay Direct": 0.1,
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
    return round(entropy, 4)


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
            # ── Interaction / derived features (Strategy 3 & 4) ──
            "Evidence Completeness Ratio",
            "Critical Evidence Present Count",
            "Amount x Win Rate",
            "Missing x Gap Weight",
            "Score x Confidence",
            "Amount Quartile",
            "Has All Critical Evidence",
            "Critical Evidence Missing Count",
            "Any Critical Missing",
            "Evidence Quality Gap",
            "Merchant Risk Score",
            "Disputes Last 7d (Merchant)",
            "Merchant Rolling Win Rate 30d",
            "Days Since Last Dispute",
            "Amount Bin: Small",
            "Amount Bin: Mid",
            "Amount Bin: Large",
            "Amount Bin: Very Large",
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
        present_count = 0
        missing_count_raw = 0
        weak_count_raw = 0
        critical_present_count = 0  # count of high-weight (>=0.25) present items
        total_evidence_items = 0
        all_critical_present = True  # whether ALL weight>=0.25 items are present

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

                total_evidence_items += 1

                if status == "present":
                    present_weights.append(weight)
                    present_count += 1
                    if weight >= 0.25:
                        critical_present_count += 1
                else:
                    gap_weights.append(weight)
                    if status == "missing":
                        missing_count_raw += 1
                    else:
                        weak_count_raw += 1
                    if weight >= 0.25:
                        all_critical_present = False

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
            total_evidence_items = 5  # reasonable default
            present_count = round(score * 5)
            missing_count_raw = 5 - present_count
            critical_present_count = 1 if score >= 0.7 else 0
            all_critical_present = score >= 0.9

        category = case.get("reason_category", "goods_not_received")
        base_win_rate = self.category_win_rates.get(category, 0.50)

        # ── Interaction / derived features (Strategy 3) ──────────────
        # Evidence completeness ratio: fraction of evidence items that are present
        evidence_completeness_ratio = (
            present_count / total_evidence_items
            if total_evidence_items > 0 else 0.0
        )

        # Amount × base win rate: captures that large disputes in easy categories are high-value
        amount_x_win_rate = norm_amount * base_win_rate

        # Missing count × weakest gap weight: penalises missing critical evidence
        missing_x_gap_weight = missing_cnt * weakest_gap_weight

        # Score × confidence interaction: high only when both are high
        score_x_confidence = score * confidence

        # Amount quartile: discretizes amount into 4 buckets (different dispute sizes behave differently)
        if raw_amount <= 1500:
            amount_quartile = 0.0
        elif raw_amount <= 5000:
            amount_quartile = 1.0
        elif raw_amount <= 15000:
            amount_quartile = 2.0
        else:
            amount_quartile = 3.0

        # Has all critical evidence: 1.0 if every weight>=0.25 evidence item is present
        has_all_critical = 1.0 if all_critical_present else 0.0

        # Critical evidence missing calculations
        critical_missing_count = 0
        if isinstance(evidence_elements, dict) and evidence_elements:
            for eid, detail in evidence_elements.items():
                w = float(detail.get("weight", 0.2)) if isinstance(detail, dict) else 0.2
                st = detail.get("status", "missing") if isinstance(detail, dict) else str(detail)
                if w >= 0.25 and st == "missing":
                    critical_missing_count += 1
        else:
            critical_missing_count = 1 if score < 0.7 else 0

        any_critical_missing = 1.0 if critical_missing_count > 0 else 0.0

        # Evidence quality gap: weakest gap weight minus strongest weight
        evidence_quality_gap = round(weakest_gap_weight - strongest_weight, 4)

        # Merchant risk score
        m_name = case.get("merchant_name") or transaction.get("merchant_name", "")
        merchant_risk = MERCHANT_RISK_TIERS.get(m_name, 0.45)

        # Merchant velocity signals
        m_vel = case.get("merchant_velocity", {})
        vel_disputes_7d = float(m_vel.get("disputes_last_7d", 1.0))
        vel_win_rate_30d = float(m_vel.get("rolling_win_rate_30d", 0.35))
        vel_days_since = float(m_vel.get("days_since_last_dispute", 14.0)) / 30.0

        # Bucketed amount bins [small, mid, large, very_large]
        amount_bin_small = 1.0 if raw_amount < 1000 else 0.0
        amount_bin_mid = 1.0 if 1000 <= raw_amount < 5000 else 0.0
        amount_bin_large = 1.0 if 5000 <= raw_amount < 15000 else 0.0
        amount_bin_very_large = 1.0 if raw_amount >= 15000 else 0.0

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
            # ── Interaction / derived features ──
            round(evidence_completeness_ratio, 4),
            float(critical_present_count),
            round(amount_x_win_rate, 4),
            round(missing_x_gap_weight, 4),
            round(score_x_confidence, 4),
            amount_quartile,
            has_all_critical,
            float(critical_missing_count),
            any_critical_missing,
            evidence_quality_gap,
            round(merchant_risk, 4),
            round(vel_disputes_7d, 2),
            round(vel_win_rate_30d, 4),
            round(vel_days_since, 4),
            amount_bin_small,
            amount_bin_mid,
            amount_bin_large,
            amount_bin_very_large,
        ] + pm_encoded + cat_encoded

        return features

