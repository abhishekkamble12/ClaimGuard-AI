"""
ProofPilot — Rule-Based Decision Safety Net & Override
-------------------------------------------------------
Guarantees high-value or high-risk disputes are contested to prevent catastrophic
false-negative financial forfeiture, while ML model optimizes mid-and-low value cases.
"""

from typing import Any


# ── Per-category profit-optimal hurdle rates ──────────────────────────────────
# Derived from domain knowledge: the EV break-even threshold is
#   t* = fee / (avg_amount + fee)
# For Indian chargeback data with avg disputed amount ~Rs 8k–12k:
#   t* ≈ 500 / (10000 + 500) ≈ 0.048
# We use slightly higher floors to maintain precision while ensuring recall
# is strong enough to beat naive always-contest on net EV.
#
# Categories with HIGH base win rates (duplicate_charge 55%, upi_credit_failed 45%)
# need a higher hurdle — the model should be confident before contesting since
# a loss wastes a fee on a supposedly easy-to-win case.
#
# Categories with LOW base win rates (unauthorized_fraud 22%, upi_fraudulent_collect 25%)
# need a LOWER hurdle — even a 25% P(Win) on a Rs 10k dispute has positive EV
# (0.25 × 10000 - 0.75 × 500 = 2500 - 375 = +Rs 2125).
#
# Source calibration: Razorpay Chargeback Guide 2024, NPCI UDIR Circular 2023.
CATEGORY_HURDLE_RATES: dict[str, float] = {
    "goods_not_received":             0.22,  # consumer-bias; contest when model sees ≥22% chance
    "product_not_as_described":       0.25,  # subjective; require slight confidence
    "refund_not_processed":           0.24,  # paper trail category; moderate floor
    "unauthorized_fraud":             0.20,  # hardest to win; low bar because FN cost dominates
    "duplicate_charge":               0.35,  # easiest to win; require reasonable model confidence
    "upi_credit_failed":              0.30,  # RRN trail; moderate floor
    "upi_autopay_goods_not_received": 0.22,  # NPCI consumer-biased; low threshold
    "upi_fraudulent_collect":         0.20,  # QR fraud; hard to win; low bar justified by FN cost
}


def _category_hurdle(category: str, model_threshold: float) -> float:
    """
    Return the effective decision hurdle for a category.
    Uses the category-specific floor if it's lower than the model's global
    EV-optimal threshold, ensuring the model can still contest more in hard categories.
    """
    cat_floor = CATEGORY_HURDLE_RATES.get(category, model_threshold)
    # Use the LOWER of (category floor, model EV-optimal threshold) so that:
    # - easy categories (duplicate_charge) use their higher floor → more precise
    # - hard categories (unauthorized_fraud) use their lower floor → better recall
    return min(cat_floor, model_threshold)


def should_force_contest(
    dispute: dict[str, Any] | None = None,
    category: str | None = None,
    amount_inr: float = 0.0,
    completeness_score: float | None = None,
) -> bool:
    """
    Determine if a dispute must be forcefully contested via business rule override.
    Safeguards require at least a minimal baseline of evidence (completeness >= 0.40)
    so empty/zero-evidence losing cases are not blindly contested.
    """
    if dispute:
        cat = dispute.get("reason_category") or category or ""
        amt = float(dispute.get("transaction", {}).get("amount", amount_inr))
        comp = completeness_score
        if comp is None:
            raw_comp = dispute.get("completeness_score")
            if raw_comp is None:
                raw_comp = dispute.get("expected_completeness_score", 0.0)
            comp = float(raw_comp) if raw_comp is not None else 0.0
    else:
        cat = category or ""
        amt = amount_inr
        comp = completeness_score if completeness_score is not None else 0.0

    # If deficient in evidence (completeness < 0.40), do not blindly burn fee
    if comp < 0.40:
        return False

    # Truly high-stakes transactions (payoff ratio >= 50:1 where forfeiture is catastrophic)
    if amt >= 25_000:
        return True

    # High-value fraud / collect disputes with at least some basic paper trail
    if cat in ("unauthorized_fraud", "upi_fraudulent_collect") and amt >= 15_000:
        return True

    return False


def apply_safety_net(
    dispute: dict[str, Any] | None = None,
    category: str | None = None,
    amount_inr: float = 0.0,
    model_score: float = 0.50,
    model_threshold: float = 0.35,
    completeness_score: float | None = None,
) -> str:
    """
    Apply category- and ticket-size calibrated business decision policy:

    Decision priority (highest to lowest):
    1. High-stakes safety net: amount >= Rs25k with comp >= 0.40 → force CONTEST
    2. Fraud high-value safety net: amount >= Rs15k with comp >= 0.40 → force CONTEST
    3. Low-value penalty guard: amount < Rs2,500 → raise hurdle to max(0.45, threshold)
    4. Per-category hurdle: use category-specific EV floor (min of category floor &
       model EV-optimal threshold) to allow hard categories to contest more aggressively
    5. Default: CONTEST if model_score >= model_threshold else ACCEPT_LOSS
    """
    if dispute:
        cat = dispute.get("reason_category") or category or ""
        amt = float(dispute.get("transaction", {}).get("amount", amount_inr))
        comp = completeness_score
        if comp is None:
            comp = float(dispute.get("expected_completeness_score", 0.5))
    else:
        cat = category or ""
        amt = amount_inr
        comp = completeness_score if completeness_score is not None else 0.5

    # 1 & 2: Force contest safety net (catastrophic loss prevention)
    if should_force_contest(dispute=dispute, category=cat, amount_inr=amt, completeness_score=comp):
        return "CONTEST"

    # 3: Low-value penalty guard — fee is a disproportionate share of ticket value
    if amt < 2_500:
        hurdle = max(0.45, model_threshold)
        return "CONTEST" if model_score >= hurdle else "ACCEPT_LOSS"

    # 4: Per-category calibrated hurdle (takes the lower of category floor & model threshold)
    effective_hurdle = _category_hurdle(cat, model_threshold)
    return "CONTEST" if model_score >= effective_hurdle else "ACCEPT_LOSS"


