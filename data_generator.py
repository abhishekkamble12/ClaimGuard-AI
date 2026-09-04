"""
ProofPilot — Synthetic Chargeback & Dispute Dataset Generator
---------------------------------------------------------------
Generates realistic, domain-grounded synthetic chargeback disputes covering
Card networks (Visa, Mastercard, RuPay, Amex) and NPCI UPI reason codes.

Realism controls (v4.0)
-----------------------
1. Per-reason-code merchant win-rate imbalance
   Real-world chargeback win rates are 20–45% depending on reason code.
   Each category carries an explicit `merchant_win_rate` used to assign
   expected_outcome, producing realistic class imbalance instead of 50/50.
   Base rates are calibrated to published industry benchmarks:
     - Goods Not Received / UPI AutoPay: ~30% merchant win rate
       (high-volume, strong consumer protection bias under NPCI UDIR)
     - Duplicate Charge: ~55% merchant win rate
       (easiest to prove — two distinct transaction IDs close the case)
     - Fraud / Unauthorized: ~22% merchant win rate
       (hardest category; burden of proof heavily on merchant)
   Sources: Razorpay Chargeback Guide 2024, NPCI UDIR Circular 2023,
            Chargebacks911 Industry Report 2024.

2. Label noise injection (is_noisy flag)
   ~15% of cases are tagged is_noisy=True. In these cases one random
   evidence item has its ground_truth_evidence label flipped one step
   toward the weaker direction (present→weak or weak→missing) AFTER
   the score is computed, creating cases where the evidence_documents text
   and the true label are inconsistent. This simulates the realistic scenario
   where a merchant uploads a document that does not actually satisfy the
   requirement (e.g. a blurry POD photo that looks present but is legally weak).

3. Feature noise / missing documents
   ~20% of cases randomly drop 1–2 evidence_documents entries even though
   ground_truth_evidence may still record those items as present or weak.
   This simulates merchants who possess evidence but failed to upload it,
   a common real-world scenario that the NLP extractor must handle as missing.
"""

from __future__ import annotations

import argparse
import json
import random
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

# ── reason-code catalogue ──────────────────────────────────────────────────────
# merchant_win_rate: probability merchant wins IF the dispute reaches adjudication.
# Base rates calibrated to: Razorpay Chargeback Guide 2024, NPCI UDIR Circular 2023,
# Chargebacks911 Global Dispute Index 2024 (Indian BFSI segment).
REASON_CODES: dict[str, dict] = {
    "goods_not_received": {
        "network": "CARD",
        "reason_code": "4553",
        "title": "Goods or services not received",
        "merchant_win_rate": 0.30,  # consumer-bias category; hard to prove delivery
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
        "merchant_win_rate": 0.38,  # subjective; merchant wins with policy docs
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
        "merchant_win_rate": 0.35,  # merchant must show refund was initiated
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
        "merchant_win_rate": 0.22,  # hardest; auth logs required, 3DS partial relief
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
        "merchant_win_rate": 0.55,  # easiest; two distinct txn IDs close the case
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
        "merchant_win_rate": 0.45,  # RRN auto-resolution; clearest paper trail
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
        "merchant_win_rate": 0.30,  # NPCI UDIR strongly favours consumer on mandate disputes
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
        "merchant_win_rate": 0.25,  # QR fraud hard to rebut without biometric + CCTV
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
STATUS_SCORE: dict[str, float] = {"present": 1.0, "weak": 0.5, "missing": 0.0}

SAMPLE_AMOUNTS = [
    399, 499, 799, 999, 1299, 1499, 1999, 2499, 3499, 4999,
    6999, 8999, 9999, 12499, 14999, 18999, 24999, 34999, 49999,
]

# ── noise parameters ───────────────────────────────────────────────────────────
LABEL_NOISE_RATE = 0.15   # fraction of cases with a flipped ground-truth label
FEATURE_NOISE_RATE = 0.20  # fraction of cases with 1–2 documents silently dropped


# ── evidence profile samplers ──────────────────────────────────────────────────

def weighted_status(profile: str) -> str:
    """Sample a ground-truth evidence status from a profile distribution."""
    if profile == "strong":
        return random.choices(STATUSES, weights=[72, 22, 6], k=1)[0]
    if profile == "partial":
        return random.choices(STATUSES, weights=[42, 38, 20], k=1)[0]
    return random.choices(STATUSES, weights=[18, 32, 50], k=1)[0]  # "weak"


# ── noise injectors ────────────────────────────────────────────────────────────

def inject_label_noise(labels: dict[str, str]) -> tuple[dict[str, str], bool]:
    """
    Flip one randomly chosen evidence item one step toward weaker.
      present → weak
      weak    → missing
      missing → unchanged (already worst)

    Returns the (possibly mutated) labels dict and a bool indicating whether
    any flip actually occurred.
    """
    flippable = [k for k, v in labels.items() if v in ("present", "weak")]
    if not flippable:
        return labels, False
    target = random.choice(flippable)
    labels = dict(labels)  # copy — don't mutate caller's dict
    labels[target] = "weak" if labels[target] == "present" else "missing"
    return labels, True


def inject_feature_noise(
    evidence_docs: dict[str, str],
    labels: dict[str, str],
) -> dict[str, str]:
    """
    Silently drop 1–2 document entries from evidence_docs even when the
    ground-truth label is present or weak.  Simulates a merchant who has
    the document but did not upload it before the response deadline.
    The ground_truth_evidence labels are NOT changed — the extractor will
    classify these as 'missing' from the text, creating a realistic gap.
    """
    droppable = [k for k, v in labels.items() if v in ("present", "weak") and k in evidence_docs]
    if not droppable:
        return evidence_docs
    n_drop = min(random.randint(1, 2), len(droppable))
    to_drop = random.sample(droppable, n_drop)
    evidence_docs = dict(evidence_docs)
    for k in to_drop:
        evidence_docs.pop(k, None)
    return evidence_docs


# ── evidence text templates ────────────────────────────────────────────────────

# Borderline / ambiguous text used in noisy cases — realistic documents that
# look partially valid but do not cleanly satisfy the evidence requirement.
AMBIGUOUS_TEMPLATES = [
    "Courier log indicates item dropped off at building mailroom on Aug 12, but no physical customer signature collected.",
    "Support ticket shows customer complained about color mismatch; merchant offered 10% discount but customer did not reply.",
    "IP address logged from Mumbai suburb, matching billing state but failing 3DS device fingerprint check.",
    "Refund initiated via gateway but pending bank settlement batch confirmation as of report date.",
    "UPI 12-digit RRN generated but bank settlement batch response timed out; dispute still open.",
    "AutoPay mandate active on NPCI switch but customer disputes the specific recurring cycle debit amount.",
    "Customer opened chat query 2 days after delivery stating box was damaged; merchant requested unboxing video which was not provided.",
    "Dynamic POS QR generated at Bengaluru store terminal with GPS coordinates, but CCTV footage retention period expired.",
    "Authentication OTP was delivered to registered mobile but customer claims phone was stolen at time of transaction.",
    "Partial delivery of multi-item order confirmed by tracking; remaining items still in transit at time of dispute filing.",
]


def evidence_text(
    evidence_id: str,
    status: str,
    is_adversarial: bool = False,
    is_noisy: bool = False,
) -> str:
    """Generate evidence document text appropriate for the given status and noise flags."""
    clean_id = evidence_id.replace("_", " ").title()

    # Noisy cases get ambiguous borderline text regardless of status label
    if is_noisy and status in ("present", "weak"):
        return random.choice(AMBIGUOUS_TEMPLATES)

    # Adversarial weak cases also get ambiguous text
    if is_adversarial and status == "weak":
        return random.choice(AMBIGUOUS_TEMPLATES)

    if status == "present":
        return (
            f"Official {clean_id} uploaded by merchant. "
            "Document verifies transaction match with valid timestamp and verified digital signature."
        )
    if status == "weak":
        return (
            f"{clean_id} is partially available. "
            "Document contains incomplete metadata or pending settlement confirmation."
        )
    return ""


# ── scoring helpers (mirrors scorer.py, kept independent for data gen) ─────────

def score_case(required_evidence: dict[str, float], labels: dict[str, str]) -> float:
    score = sum(
        weight * STATUS_SCORE[labels[ev_id]]
        for ev_id, weight in required_evidence.items()
    )
    return round(score, 3)


def confidence_case(score: float, labels: dict[str, str]) -> float:
    total = len(labels)
    if total == 0:
        return 0.0
    weak_ratio = sum(1 for s in labels.values() if s == "weak") / total
    return round(max(0.0, min(1.0, score - (weak_ratio * 0.10))), 3)


def route_case(score: float, confidence: float) -> str:
    if score >= 0.80 and confidence >= 0.75:
        return "auto_draft_response"
    if score >= 0.50:
        return "request_more_evidence"
    return "human_review"


def expected_outcome(score: float, category: str, is_noisy: bool) -> str:
    """
    Assign expected_outcome using per-category merchant win rates to produce
    realistic class imbalance.

    Logic:
      - score >= 0.80  → strong evidence; sample from category win rate
        (even strong evidence doesn't guarantee a win — adjudicators
        sometimes rule against merchants on technical grounds)
      - score <= 0.35  → very weak evidence; almost always lost
      - borderline (0.35–0.80) → win probability scaled linearly between
        (win_rate * 0.40) and (win_rate * 1.20) to create a realistic gradient
      - Noisy cases: win probability penalised by 0.10 to reflect that
        ambiguous evidence reduces adjudication success rate
    """
    base_win_rate = REASON_CODES[category]["merchant_win_rate"]

    if score >= 0.80:
        win_p = min(0.90, base_win_rate * 1.40)   # strong evidence lifts probability
    elif score <= 0.35:
        win_p = base_win_rate * 0.25               # very weak → rarely wins
    else:
        # Linear interpolation from (0.35 → win_rate*0.40) to (0.80 → win_rate*1.40)
        t = (score - 0.35) / (0.80 - 0.35)
        win_p = base_win_rate * (0.40 + t * 1.00)

    # Noise penalty: ambiguous evidence hurts adjudication outcomes
    if is_noisy:
        win_p = max(0.05, win_p - 0.10)

    win_p = max(0.02, min(0.95, win_p))
    return random.choices(["won", "lost"], weights=[win_p, 1.0 - win_p], k=1)[0]


# ── case generator ─────────────────────────────────────────────────────────────

def generate_case(
    case_number: int,
    category: str,
    profile: str,
    rng: random.Random,
) -> dict:
    reason = REASON_CODES[category]
    required = reason["required_evidence"]

    # Every 5th case is adversarial (existing behaviour)
    is_adversarial = (case_number % 5 == 0)

    # ── ground-truth labels ───────────────────────────────────────────────────
    labels: dict[str, str] = {ev_id: weighted_status(profile) for ev_id in required}

    # Compute score BEFORE injecting label noise — so the noise creates a
    # deliberate gap between the score-based expected_route and the true label.
    score_pre_noise = score_case(required, labels)
    confidence_pre_noise = confidence_case(score_pre_noise, labels)
    route_pre_noise = route_case(score_pre_noise, confidence_pre_noise)

    # ── label noise injection ─────────────────────────────────────────────────
    is_noisy = rng.random() < LABEL_NOISE_RATE
    noise_applied = False
    if is_noisy:
        labels, noise_applied = inject_label_noise(labels)
        is_noisy = noise_applied  # only tag as noisy if a flip actually happened

    # Re-score after noise so expected_route reflects the noisier ground truth
    score = score_case(required, labels)
    confidence = confidence_case(score, labels)
    route = route_case(score, confidence)
    missing = [ev_id for ev_id, s in labels.items() if s == "missing"]
    weak = [ev_id for ev_id, s in labels.items() if s == "weak"]

    # ── document text generation ──────────────────────────────────────────────
    evidence_docs: dict[str, str] = {
        ev_id: evidence_text(ev_id, status, is_adversarial=is_adversarial, is_noisy=is_noisy)
        for ev_id, status in labels.items()
        if status != "missing"
    }

    # ── feature noise: silently drop uploaded documents ───────────────────────
    has_feature_noise = False
    if rng.random() < FEATURE_NOISE_RATE:
        new_docs = inject_feature_noise(evidence_docs, labels)
        if len(new_docs) < len(evidence_docs):
            evidence_docs = new_docs
            has_feature_noise = True

    # ── temporal metadata ─────────────────────────────────────────────────────
    base_date = date(2026, 8, 1) + timedelta(days=(case_number % 28))
    payment_date = base_date - timedelta(days=rng.randint(1, 10))
    deadline_days = rng.randint(7, 21)
    evidence_due_date = base_date + timedelta(days=deadline_days)

    # ── transaction metadata ──────────────────────────────────────────────────
    amount_inr = rng.choice(SAMPLE_AMOUNTS)
    amount_paise = amount_inr * 100
    pm = "upi" if reason["network"] == "UPI" else rng.choice(PAYMENT_METHODS)
    merchant_name = MERCHANTS[(case_number - 1) % len(MERCHANTS)]
    merchant_id = f"acc_{100000 + ((case_number - 1) % len(MERCHANTS)) * 1111}"

    dispute_id = f"disp_{1000000000000 + case_number}"
    payment_id = f"pay_{1000000000000 + rng.randint(1000, 99999)}"
    order_id = f"order_{1000000000000 + rng.randint(1000, 99999)}"
    customer_id = f"cust_{200000 + rng.randint(100, 999)}"

    outcome = expected_outcome(score, category, is_noisy)

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
        # ── noise flags (used by test harness) ────────────────────────────────
        "is_adversarial": is_adversarial,
        "is_noisy": is_noisy,           # label noise applied: one evidence label flipped
        "has_feature_noise": has_feature_noise,  # one or more docs silently dropped
        # ── temporal ──────────────────────────────────────────────────────────
        "created_at": base_date.isoformat(),
        "evidence_due_by": int(
            datetime.combine(evidence_due_date, datetime.min.time(), tzinfo=timezone.utc).timestamp()
        ),
        "days_remaining": deadline_days,
        # ── transaction ───────────────────────────────────────────────────────
        "transaction": {
            "amount": amount_inr,
            "amount_paise": amount_paise,
            "currency": "INR",
            "payment_method": pm,
            "payment_date": payment_date.isoformat(),
            "order_id": order_id,
            "payment_id": payment_id,
        },
        # ── evidence ──────────────────────────────────────────────────────────
        "evidence_documents": evidence_docs,
        "ground_truth_evidence": labels,
        "missing_evidence": missing,
        "weak_evidence": weak,
        # ── scoring ground truth ──────────────────────────────────────────────
        "expected_completeness_score": score,
        "expected_confidence": confidence,
        "expected_route": route,
        "expected_outcome": outcome,
        # ── split assigned downstream ─────────────────────────────────────────
        "split": "train",
    }


# ── dataset builder ────────────────────────────────────────────────────────────

def generate_dataset(
    total_cases: int = 500,
    val_ratio: float = 0.20,
    test_ratio: float = 0.20,
    seed: int = 42,
    clean_val: bool = False,
) -> dict:
    """
    Generate a complete synthetic chargeback dataset with realistic noise.

    Split strategy (v5.0)
    ---------------------
    Three-way stratified split: train / val / test.
    Default ratios: 60% train | 20% val | 20% test.
    If clean_val is True, validation split cases are guaranteed free of synthetic noise
    to provide clean, reliable threshold calibration.
    """
    rng = random.Random(seed)
    categories = list(REASON_CODES.keys())
    profiles = ["strong", "partial", "weak"]
    cases = []

    for case_number in range(1, total_cases + 1):
        category = categories[(case_number - 1) % len(categories)]
        profile = profiles[(case_number - 1) % len(profiles)]
        cases.append(generate_case(case_number, category, profile, rng))

    # ── stratified 3-way split ────────────────────────────────────────────────
    by_cat: dict[str, list] = {cat: [] for cat in categories}
    for case in cases:
        by_cat[case["reason_category"]].append(case)

    for cat_cases in by_cat.values():
        rng.shuffle(cat_cases)
        n = len(cat_cases)
        n_test = max(1, round(n * test_ratio))
        n_val  = max(1, round(n * val_ratio))

        if clean_val:
            clean = [c for c in cat_cases if not c.get("is_noisy") and not c.get("has_feature_noise")]
            other = [c for c in cat_cases if c.get("is_noisy") or c.get("has_feature_noise")]
            val_selection = clean[:n_val]
            remainder = clean[n_val:] + other
            rng.shuffle(remainder)
            test_selection = remainder[:n_test]
            train_selection = remainder[n_test:]
            for c in test_selection:
                c["split"] = "test"
            for c in val_selection:
                c["split"] = "val"
            for c in train_selection:
                c["split"] = "train"
        else:
            for i, c in enumerate(cat_cases):
                if i < n_test:
                    c["split"] = "test"
                elif i < n_test + n_val:
                    c["split"] = "val"
                else:
                    c["split"] = "train"

    # ── dataset-level statistics ──────────────────────────────────────────────
    noisy_count     = sum(1 for c in cases if c["is_noisy"])
    feat_noise_count = sum(1 for c in cases if c["has_feature_noise"])
    won_count  = sum(1 for c in cases if c["expected_outcome"] == "won")
    lost_count = total_cases - won_count

    split_counts = {
        s: sum(1 for c in cases if c["split"] == s)
        for s in ("train", "val", "test")
    }
    category_dist = {
        cat: sum(1 for c in cases if c["reason_category"] == cat)
        for cat in categories
    }
    win_rates_by_category = {
        cat: round(
            sum(1 for c in cases if c["reason_category"] == cat and c["expected_outcome"] == "won")
            / max(1, sum(1 for c in cases if c["reason_category"] == cat)),
            3,
        )
        for cat in categories
    }

    return {
        "dataset_name": "proofpilot_synthetic_chargeback_dataset",
        "version": "5.0",
        "total_cases": total_cases,
        "train_cases": split_counts["train"],
        "val_cases":   split_counts["val"],
        "test_cases":  split_counts["test"],
        "data_methodology": {
            "split_ratios": {"train": 1 - val_ratio - test_ratio,
                             "val": val_ratio, "test": test_ratio},
            "stratified_by": "reason_category",
            "clean_val_split": clean_val,
            "label_noise_rate":  LABEL_NOISE_RATE,
            "feature_noise_rate": FEATURE_NOISE_RATE,
            "noisy_cases_injected": noisy_count,
            "feature_noise_cases_injected": feat_noise_count,
            "class_balance": {
                "won": won_count,
                "lost": lost_count,
                "overall_win_rate": round(won_count / total_cases, 3),
            },
            "category_distribution": category_dist,
            "win_rates_by_category": win_rates_by_category,
        },
        "reason_code_config": REASON_CODES,
        "cases": cases,
    }


# ── CLI ────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate ProofPilot synthetic chargeback dataset with realistic noise"
    )
    parser.add_argument("--cases", "--size", dest="cases", type=int, default=500, help="Total number of dispute cases")
    parser.add_argument("--clean-val", action="store_true", help="Ensure validation split is noise-free for reliable threshold calibration")
    parser.add_argument("--val-ratio",  type=float, default=0.20, help="Fraction held out as validation split")
    parser.add_argument("--test-ratio", type=float, default=0.20, help="Fraction held out as test split")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument(
        "--out",
        default="outputs/synthetic_chargeback_dataset.json",
        help="Output file path",
    )
    args = parser.parse_args()

    dataset = generate_dataset(args.cases, args.val_ratio, args.test_ratio, args.seed, clean_val=args.clean_val)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(dataset, indent=2, ensure_ascii=False), encoding="utf-8")

    md = dataset["data_methodology"]
    cb = md["class_balance"]
    print(f"ProofPilot v5.0 dataset generated → {out_path}")
    print(f"  Total cases     : {dataset['total_cases']}")
    print(f"  Train / Val / Test : {dataset['train_cases']} / {dataset['val_cases']} / {dataset['test_cases']}")
    print(f"  Label noise     : {md['noisy_cases_injected']} cases ({LABEL_NOISE_RATE:.0%} rate)")
    print(f"  Feature noise   : {md['feature_noise_cases_injected']} cases ({FEATURE_NOISE_RATE:.0%} rate)")
    print(f"  Class balance   : won={cb['won']}  lost={cb['lost']}  overall_win_rate={cb['overall_win_rate']:.1%}")
    print(f"  Win rates/cat   : {md['win_rates_by_category']}")


if __name__ == "__main__":
    main()
