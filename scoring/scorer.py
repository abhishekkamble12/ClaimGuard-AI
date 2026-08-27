"""
ProofPilot — Completeness Scorer & Counterfactual Risk Engine
--------------------------------------------------------------
Loads a dispute and reason-code config, evaluates evidence statuses,
calculates readiness score, confidence, risk tier (LOW, MEDIUM, HIGH),
integrates ML Win Probability (scikit-learn), TF-IDF semantic relevance,
economic decision engine (CONTEST vs ACCEPT_LOSS), and produces
counterfactual risk improvement predictions.
"""

import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extraction.extractor import extract_and_classify_evidence
from ml.semantic_matcher import compute_semantic_relevance
from ml.win_predictor import get_win_predictor
from scoring.audit_log import log_decision

# Backward-compatibility alias for tests
infer_evidence_statuses = extract_and_classify_evidence

STATUS_SCORE: dict[str, float] = {
    "present": 1.0,
    "weak": 0.5,
    "missing": 0.0,
}

DEFAULT_AUTO_DRAFT_MIN_SCORE = 0.80
DEFAULT_AUTO_DRAFT_MIN_CONFIDENCE = 0.75
DEFAULT_REQUEST_MORE_MIN_SCORE = 0.50


def load_reason_code_config(config_path: str | Path) -> dict[str, Any]:
    """Load reason-code config from a JSON file or directory of JSON configs."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Reason code config not found: {path}")
    if path.is_dir():
        merged = {}
        for f in sorted(path.glob("*.json")):
            data = json.loads(f.read_text(encoding="utf-8"))
            merged.update(data.get("reason_codes", data))
        return merged
    data = json.loads(path.read_text(encoding="utf-8"))
    return data.get("reason_codes", data)


def score_evidence(
    required_evidence: dict[str, float],
    evidence_statuses: dict[str, str],
    evidence_documents: dict[str, Any] | None = None,
    rc_config_required: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Score individual evidence items, calculate TF-IDF relevance, and compute total weighted readiness."""
    element_results: dict[str, Any] = {}
    weighted_score = 0.0
    total_weight = sum(required_evidence.values())

    for evidence_id, weight in required_evidence.items():
        status = evidence_statuses.get(evidence_id, "missing")
        raw_score = STATUS_SCORE.get(status, 0.0)
        contribution = weight * raw_score

        # Calculate TF-IDF semantic relevance if text is provided
        raw_doc = evidence_documents.get(evidence_id, "") if evidence_documents else ""
        desc = ""
        if rc_config_required and evidence_id in rc_config_required:
            spec = rc_config_required[evidence_id]
            desc = spec.get("description", "") if isinstance(spec, dict) else ""

        semantic_rel = compute_semantic_relevance(raw_doc, desc)

        element_results[evidence_id] = {
            "status": status,
            "weight": weight,
            "score": raw_score,
            "weighted_contribution": round(contribution, 4),
            "semantic_relevance": semantic_rel,
        }
        weighted_score += contribution

    completeness_score = weighted_score / total_weight if total_weight > 0 else 0.0
    return {
        "elements": element_results,
        "completeness_score": round(completeness_score, 4),
    }


def compute_confidence(completeness_score: float, evidence_statuses: dict[str, str]) -> float:
    """Compute confidence level penalized by weak/ambiguous evidence."""
    total = len(evidence_statuses)
    if total == 0:
        return 0.0

    weak_count = sum(1 for s in evidence_statuses.values() if s == "weak")
    weak_ratio = weak_count / total
    raw_confidence = completeness_score - (weak_ratio * 0.10)
    return round(max(0.0, min(1.0, raw_confidence)), 4)


def compute_routing_and_risk(
    completeness_score: float,
    confidence: float,
    thresholds: dict[str, Any] | None = None,
) -> tuple[str, str]:
    """
    Returns (routing_decision, risk_level).
    - auto_draft_response -> LOW Risk
    - request_more_evidence -> MEDIUM Risk
    - human_review -> HIGH Risk
    """
    if completeness_score >= DEFAULT_AUTO_DRAFT_MIN_SCORE and confidence >= DEFAULT_AUTO_DRAFT_MIN_CONFIDENCE:
        return "auto_draft_response", "LOW"
    if completeness_score >= DEFAULT_REQUEST_MORE_MIN_SCORE:
        return "request_more_evidence", "MEDIUM"
    return "human_review", "HIGH"


def build_counterfactual_improvements(
    elements: dict[str, Any],
    required_evidence: dict[str, float],
    current_score: float,
) -> list[dict[str, Any]]:
    """
    Calculate projected score gain & new risk tier if missing/weak evidence is upgraded to present.
    """
    gaps = []
    total_weight = sum(required_evidence.values())

    for evidence_id, detail in elements.items():
        status = detail["status"]
        if status == "present":
            continue

        weight = detail["weight"]
        current_contribution = detail["weighted_contribution"]
        full_contribution = weight * 1.0
        gain = (full_contribution - current_contribution) / total_weight
        projected_score = round(min(1.0, current_score + gain), 4)

        _, projected_risk = compute_routing_and_risk(projected_score, projected_score)

        gaps.append(
            {
                "evidence_id": evidence_id,
                "current_status": status,
                "potential_score_gain": round(gain, 4),
                "score_if_added": projected_score,
                "projected_risk": projected_risk,
                "explanation": (
                    f"Adding '{evidence_id.replace('_', ' ').title()}' raises readiness from "
                    f"{current_score:.0%} to {projected_score:.0%} (Projected Risk: {projected_risk})."
                ),
            }
        )

    gaps.sort(key=lambda x: x["potential_score_gain"], reverse=True)
    return gaps


def score_dispute(
    dispute: dict[str, Any],
    reason_codes: dict[str, Any],
    *,
    use_ground_truth: bool = False,
    api_key: str | None = None,
    log_audit: bool = True,
) -> dict[str, Any]:
    """Score a single dispute case."""
    category = dispute.get("reason_category")
    if category not in reason_codes:
        raise ValueError(f"Unknown reason category '{category}'. Available: {list(reason_codes.keys())}")

    rc_config = reason_codes[category]
    raw_required: dict[str, Any] = rc_config["required_evidence"]
    required_evidence: dict[str, float] = {
        eid: (v["weight"] if isinstance(v, dict) else float(v)) for eid, v in raw_required.items()
    }
    thresholds: dict[str, Any] | None = rc_config.get("routing_thresholds")

    evidence_docs = dispute.get("evidence_documents", {})

    if use_ground_truth:
        evidence_statuses = dispute.get("ground_truth_evidence", {})
        evidence_source = "ground_truth_evidence"
    else:
        evidence_statuses = extract_and_classify_evidence(
            required_evidence,
            evidence_docs,
            api_key=api_key,
        )
        evidence_source = "evidence_documents"

    scoring = score_evidence(
        required_evidence,
        evidence_statuses,
        evidence_documents=evidence_docs,
        rc_config_required=raw_required,
    )
    completeness_score = scoring["completeness_score"]
    elements = scoring["elements"]
    confidence = compute_confidence(completeness_score, evidence_statuses)

    routing_decision, risk_level = compute_routing_and_risk(completeness_score, confidence, thresholds)
    gaps = build_counterfactual_improvements(elements, required_evidence, completeness_score)

    missing_evidence = [e["evidence_id"] for e in gaps if e["current_status"] == "missing"]
    weak_evidence = [e["evidence_id"] for e in gaps if e["current_status"] == "weak"]

    # Integrate Machine Learning Win Predictor & Financial EV Calculator
    win_predictor = get_win_predictor()
    partial_result = {
        "completeness_score": completeness_score,
        "confidence": confidence,
        "missing_evidence": missing_evidence,
        "weak_evidence": weak_evidence,
    }
    win_probability = win_predictor.predict_win_probability(dispute, partial_result)
    disputed_amount = float(dispute.get("transaction", {}).get("amount", 1000))
    expected_value = win_predictor.calculate_expected_financial_value(disputed_amount, win_probability)
    economic_rec = win_predictor.recommend_action(disputed_amount, win_probability)

    result = {
        "dispute_id": dispute.get("dispute_id"),
        "reason_code": rc_config.get("reason_code"),
        "reason_category": category,
        "reason_title": rc_config.get("title"),
        "completeness_score": completeness_score,
        "completeness_pct": f"{completeness_score:.0%}",
        "confidence": confidence,
        "confidence_pct": f"{confidence:.0%}",
        "risk_level": risk_level,
        "evidence_source": evidence_source,
        "routing_decision": routing_decision,
        "missing_evidence": missing_evidence,
        "weak_evidence": weak_evidence,
        "evidence_elements": elements,
        "gap_explanation": gaps,
        "win_probability": win_probability,
        "win_probability_pct": f"{win_probability:.0%}",
        "expected_financial_value": expected_value,
        "economic_recommendation": economic_rec,
        "ml_model_auc": win_predictor.auc_score,
        "ml_feature_importances": win_predictor.feature_importances_,
    }

    if log_audit and dispute.get("dispute_id"):
        try:
            log_decision(dispute["dispute_id"], result)
        except Exception:
            pass

    return result


def score_dataset(
    dataset_path: str | Path,
    config_path: str | Path,
    split: str | None = None,
    *,
    use_ground_truth: bool = False,
) -> list[dict[str, Any]]:
    """Batch score cases in synthetic dataset."""
    dataset = json.loads(Path(dataset_path).read_text(encoding="utf-8"))
    reason_codes = load_reason_code_config(config_path)

    cases = dataset["cases"]
    if split:
        cases = [c for c in cases if c.get("split") == split]

    results = []
    for dispute in cases:
        try:
            result = score_dispute(dispute, reason_codes, use_ground_truth=use_ground_truth, log_audit=False)
            results.append(result)
        except ValueError as exc:
            results.append({"dispute_id": dispute.get("dispute_id"), "error": str(exc)})
    return results


if __name__ == "__main__":
    BASE = Path(__file__).parent.parent
    dataset_path = BASE / "outputs" / "synthetic_chargeback_dataset.json"
    config_path = BASE / "config" / "reason_codes" / "amex.json"

    print("ProofPilot ML Scorer executing...")
    reason_codes = load_reason_code_config(config_path)
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))

    sample = dataset["cases"][0]
    result = score_dispute(sample, reason_codes)

    print("\n=== Sample ProofPilot ML Scoring Result ===")
    print(f"Dispute ID           : {result['dispute_id']}")
    print(f"Readiness Score      : {result['completeness_pct']}")
    print(f"Confidence           : {result['confidence_pct']}")
    print(f"ML Win Prob P(Win)   : {result['win_probability_pct']}")
    print(f"Expected ROI (EV)    : ₹{result['expected_financial_value']['expected_value_inr']}")
    print(f"Economic Decision    : {result['economic_recommendation']['action']} ({result['economic_recommendation']['reason']})")
    print(f"Risk Level           : {result['risk_level']}")
    print(f"Routing Decision     : {result['routing_decision']}")
