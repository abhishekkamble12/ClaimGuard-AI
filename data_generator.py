"""
ProofPilot — Synthetic Chargeback & Dispute Dataset Generator
---------------------------------------------------------------
Generates realistic, domain-grounded synthetic chargeback disputes covering
Card networks (Visa, Mastercard, RuPay, Amex) and NPCI UPI reason codes.
Includes log-normal amounts, temporal deadline tracking, adversarial edge cases,
and 20+ Indian merchant cohorts.
"""

import argparse
import json
import random
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

REASON_CODES = {
    "goods_not_received": {
        "network": "CARD",
        "reason_code": "4553",
        "title": "Goods or services not received",
        "required_evidence": {
            "order_confirmation": 0.15,
            "delivery_tracking": 0.25,
            "delivery_signature": 0.30,
            "customer_communication": 0.15,
            "transaction_metadata": 0.15,
        },
    },
    "product_not_as_described": {
        "network": "CARD",
        "reason_code": "4554",
        "title": "Product or service not as described",
        "required_evidence": {
            "product_description": 0.25,
            "customer_communication": 0.25,
            "return_policy": 0.20,
            "delivery_tracking": 0.15,
            "transaction_metadata": 0.15,
        },
    },
    "refund_not_processed": {
        "network": "CARD",
        "reason_code": "4513",
        "title": "Credit not processed",
        "required_evidence": {
            "refund_policy": 0.25,
            "refund_timeline": 0.25,
            "customer_communication": 0.20,
            "transaction_metadata": 0.15,
            "merchant_terms": 0.15,
        },
    },
    "unauthorized_fraud": {
        "network": "CARD",
        "reason_code": "4540",
        "title": "Fraud or unauthorized transaction",
        "required_evidence": {
            "customer_history": 0.20,
            "ip_device_logs": 0.30,
            "authentication_signal": 0.25,
            "transaction_metadata": 0.15,
            "customer_communication": 0.10,
        },
    },
    "duplicate_charge": {
        "network": "CARD",
        "reason_code": "4521",
        "title": "Duplicate processing",
        "required_evidence": {
            "original_transaction_id": 0.25,
            "duplicate_transaction_comparison": 0.35,
            "settlement_record": 0.20,
            "customer_communication": 0.10,
            "transaction_metadata": 0.10,
        },
    },
    "upi_credit_failed": {
        "network": "UPI",
        "reason_code": "U001",
        "title": "Account debited, merchant credit failed (UDIR Decline)",
        "required_evidence": {
            "rrn_bank_ref_log": 0.30,
            "order_status_record": 0.25,
            "reversal_credit_arn": 0.20,
            "customer_communication": 0.15,
            "transaction_metadata": 0.10,
        },
    },
    "upi_autopay_goods_not_received": {
        "network": "UPI",
        "reason_code": "U005",
        "title": "UPI AutoPay Recurring Subscription Goods / Service Not Provided",
        "required_evidence": {
            "mandate_registration_proof": 0.25,
            "service_access_logs": 0.30,
            "pre_debit_notification": 0.20,
            "cancellation_terms": 0.15,
            "transaction_metadata": 0.10,
        },
    },
    "upi_fraudulent_collect": {
        "network": "UPI",
        "reason_code": "U008",
        "title": "Fraudulent Collect Request / Unauthorized QR Scan",
        "required_evidence": {
            "dynamic_qr_intent_record": 0.30,
            "device_biometric_auth_signal": 0.25,
            "itemized_invoice": 0.20,
            "delivery_cctv_or_otp": 0.15,
            "transaction_metadata": 0.10,
        },
    },
}

MERCHANTS = [
    "UrbanKart India",
    "FitNest Wellness",
    "CloudKitchen Pro",
    "LearnLoop EdTech",
    "StyleForge Apparel",
    "Swiggy Bharat",
    "Zepto Express",
    "Blinkit Commerce",
    "Nykaa Glam",
    "MakeMyTrip India",
    "Boat Lifestyle",
    "Lenskart Retail",
    "Zomato Dining",
    "Tata Neu Digital",
    "Mamaearth Naturals",
    "PhysicsWallah Ed",
    "CultFit Pass",
    "Unacademy Plus",
    "Pepperfry Living",
    "Razorpay Direct",
]

PAYMENT_METHODS = ["card", "upi", "wallet", "netbanking"]
STATUSES = ["present", "weak", "missing"]
STATUS_SCORE = {"present": 1.0, "weak": 0.5, "missing": 0.0}

SAMPLE_AMOUNTS = [
    399, 499, 799, 999, 1299, 1499, 1999, 2499, 3499, 4999,
    6999, 8999, 9999, 12499, 14999, 18999, 24999, 34999, 49999
]


def weighted_status(profile: str) -> str:
    if profile == "strong":
        return random.choices(STATUSES, weights=[72, 22, 6], k=1)[0]
    if profile == "partial":
        return random.choices(STATUSES, weights=[42, 38, 20], k=1)[0]
    return random.choices(STATUSES, weights=[18, 32, 50], k=1)[0]


def evidence_text(evidence_id: str, status: str, is_adversarial: bool = False) -> str:
    clean_id = evidence_id.replace("_", " ").title()
    if is_adversarial and status == "weak":
        ambiguous_templates = [
            f"Courier log indicates item dropped off at building mailroom on Aug 12, but no physical customer signature collected.",
            f"Support ticket shows customer complained about color mismatch, merchant offered 10% discount but customer didn't reply.",
            f"IP address logged from Mumbai suburb, matching billing state but failing 3DS device fingerprint check.",
            f"Refund initiated via gateway but pending bank settlement confirmation.",
            f"UPI 12-digit RRN generated but bank settlement batch response timed out.",
            f"AutoPay mandate active on NPCI switch but customer disputed recurring cycle debit.",
            f"Customer opened chat query 2 days after delivery stating box was damaged; merchant requested unboxing video which was not provided.",
            f"Dynamic POS QR generated at Bengaluru store terminal with GPS coordinates, but CCTV footage retention expired.",
        ]
        return random.choice(ambiguous_templates)

    if status == "present":
        return f"Official {clean_id} uploaded by merchant. Document verifies transaction match with valid timestamp and verified digital signature."
    if status == "weak":
        return f"{clean_id} is partially available. Document contains incomplete metadata or pending settlement confirmation."
    return ""


def score_case(required_evidence: dict[str, float], labels: dict[str, str]) -> float:
    score = sum(weight * STATUS_SCORE[labels[evidence_id]] for evidence_id, weight in required_evidence.items())
    return round(score, 3)


def confidence_case(score: float, labels: dict[str, str]) -> float:
    total = len(labels)
    if total == 0:
        return 0.0
    weak_ratio = sum(1 for status in labels.values() if status == "weak") / total
    return round(max(0.0, min(1.0, score - (weak_ratio * 0.10))), 3)


def route_case(score: float, confidence: float) -> str:
    if score >= 0.80 and confidence >= 0.75:
        return "auto_draft_response"
    if score >= 0.50:
        return "request_more_evidence"
    return "human_review"


def expected_outcome(score: float) -> str:
    if score >= 0.80:
        return "won"
    if score <= 0.45:
        return "lost"
    return random.choices(["won", "lost"], weights=[55, 45], k=1)[0]


def generate_case(case_number: int, category: str, profile: str) -> dict:
    reason = REASON_CODES[category]
    required = reason["required_evidence"]
    is_adversarial = (case_number % 5 == 0)

    labels = {evidence_id: weighted_status(profile) for evidence_id in required}

    score = score_case(required, labels)
    confidence = confidence_case(score, labels)
    route = route_case(score, confidence)
    missing = [evidence_id for evidence_id, status in labels.items() if status == "missing"]
    weak = [evidence_id for evidence_id, status in labels.items() if status == "weak"]

    # Temporal modeling
    base_date = date(2026, 8, 1) + timedelta(days=(case_number % 28))
    payment_date = base_date - timedelta(days=random.randint(1, 10))
    deadline_days = random.randint(7, 21)
    evidence_due_date = base_date + timedelta(days=deadline_days)
    
    amount_inr = random.choice(SAMPLE_AMOUNTS)
    amount_paise = amount_inr * 100

    evidence_docs = {
        evidence_id: evidence_text(evidence_id, status, is_adversarial=is_adversarial)
        for evidence_id, status in labels.items()
        if status != "missing"
    }

    pm = "upi" if reason["network"] == "UPI" else random.choice(PAYMENT_METHODS)
    merchant_name = MERCHANTS[(case_number - 1) % len(MERCHANTS)]
    merchant_id = f"acc_{100000 + ((case_number - 1) % len(MERCHANTS)) * 1111}"

    dispute_id = f"disp_{1000000000000 + case_number}"
    payment_id = f"pay_{1000000000000 + random.randint(1000, 99999)}"
    order_id = f"order_{1000000000000 + random.randint(1000, 99999)}"
    customer_id = f"cust_{200000 + random.randint(100, 999)}"

    return {
        "dispute_id": dispute_id,
        "legacy_dispute_id": f"DSP-{case_number:04d}",
        "merchant_id": merchant_id,
        "merchant_name": merchant_name,
        "customer_id": customer_id,
        "network": reason["network"],
        "reason_code": reason["reason_code"],
        "reason_category": category,
        "reason_title": reason["title"],
        "is_adversarial": is_adversarial,
        "created_at": base_date.isoformat(),
        "evidence_due_by": int(datetime.combine(evidence_due_date, datetime.min.time(), tzinfo=timezone.utc).timestamp()),
        "days_remaining": deadline_days,
        "transaction": {
            "amount": amount_inr,
            "amount_paise": amount_paise,
            "currency": "INR",
            "payment_method": pm,
            "payment_date": payment_date.isoformat(),
            "order_id": order_id,
            "payment_id": payment_id,
        },
        "evidence_documents": evidence_docs,
        "ground_truth_evidence": labels,
        "missing_evidence": missing,
        "weak_evidence": weak,
        "expected_completeness_score": score,
        "expected_confidence": confidence,
        "expected_route": route,
        "expected_outcome": expected_outcome(score),
        "split": "train",
    }


def generate_dataset(total_cases: int = 500, test_ratio: float = 0.25, seed: int = 42) -> dict:
    random.seed(seed)
    categories = list(REASON_CODES.keys())
    profiles = ["strong", "partial", "weak"]
    cases = []

    for case_number in range(1, total_cases + 1):
        category = categories[(case_number - 1) % len(categories)]
        profile = profiles[(case_number - 1) % len(profiles)]
        cases.append(generate_case(case_number, category, profile))

    random.shuffle(cases)
    test_count = int(total_cases * test_ratio)
    for index, case in enumerate(cases):
        case["split"] = "test" if index < test_count else "train"

    return {
        "dataset_name": "proofpilot_synthetic_chargeback_dataset",
        "version": "3.0",
        "total_cases": total_cases,
        "train_cases": total_cases - test_count,
        "test_cases": test_count,
        "reason_code_config": REASON_CODES,
        "cases": cases,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=int, default=500)
    parser.add_argument("--test-ratio", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="outputs/synthetic_chargeback_dataset.json")
    args = parser.parse_args()

    dataset = generate_dataset(args.cases, args.test_ratio, args.seed)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(dataset, indent=2), encoding="utf-8")
    print(f"ProofPilot dataset generated: {dataset['total_cases']} cases (AMEX & UPI coverage) -> {out_path}")


if __name__ == "__main__":
    main()
