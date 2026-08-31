"""
ProofPilot — Completeness Scorer & Counterfactual Risk Engine
--------------------------------------------------------------
Loads a dispute and reason-code config, evaluates evidence statuses,
calculates readiness score, confidence, risk tier (LOW, MEDIUM, HIGH),
integrates ML Win Probability (scikit-learn), TF-IDF semantic relevance,
economic decision engine (CONTEST vs ACCEPT_LOSS), and produces
counterfactual risk improvement predictions.
Supports configurable scoring thresholds via config/scoring_thresholds.yaml
and real-time inference logging to ModelDriftMonitor.
"""

import json
import sys
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extraction.extractor import extract_and_classify_evidence
from ml.semantic_matcher import compute_semantic_relevance
from ml.win_predictor import get_win_predictor
from monitoring.model_monitor import get_drift_monitor
from scoring.audit_log import log_decision
from utils.logging_config import get_logger

logger = get_logger(__name__)

# Backward-compatibility alias for tests
infer_evidence_statuses = extract_and_classify_evidence

THRESHOLDS_CONFIG_PATH = ROOT / "config" / "scoring_thresholds.yaml"


def load_scoring_thresholds_config(config_path: Path = THRESHOLDS_CONFIG_PATH) -> dict[str, Any]:
    """Load configurable decision thresholds from YAML file with fallback to defaults."""
    defaults = {
        "routing_gates": {
            "auto_draft_min_score": 0.80,
            "auto_draft_min_confidence": 0.75,
            "request_more_min_score": 0.50,
        },
        "evidence_status_weights": {
            "present": 1.0,
            "weak": 0.5,
            "missing": 0.0,
        },
        "economic_defaults": {
            "dispute_fee_inr": 500.0,
            "min_contest_win_probability": 0.50,
        },
    }
    if config_path.exists():
        try:
            loaded = yaml.safe_load(config_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                return loaded
        except Exception as exc:
            logger.warning(f"Error loading {config_path}: {exc}. Using defaults.")
    return defaults


_SCORING_CONFIG = load_scoring_thresholds_config()

STATUS_SCORE: dict[str, float] = _SCORING_CONFIG.get("evidence_status_weights", {
    "present": 1.0,
    "weak": 0.5,
    "missing": 0.0,
})

DEFAULT_AUTO_DRAFT_MIN_SCORE = _SCORING_CONFIG.get("routing_gates", {}).get("auto_draft_min_score", 0.80)
DEFAULT_AUTO_DRAFT_MIN_CONFIDENCE = _SCORING_CONFIG.get("routing_gates", {}).get("auto_draft_min_confidence", 0.75)
DEFAULT_REQUEST_MORE_MIN_SCORE = _SCORING_CONFIG.get("routing_gates", {}).get("request_more_min_score", 0.50)


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
    evidence_documents: dict[str, str] | None = None,
    rc_config_required: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Calculate completeness score and per-element weighted contributions."""
    elements = {}
    total_score = 0.0

    for evidence_id, weight in required_evidence.items():
        status = evidence_statuses.get(evidence_id, "missing")
        raw_score = STATUS_SCORE.get(status, 0.0)
        weighted = weight * raw_score
        total_score += weighted

        # Calculate TF-IDF semantic relevance if documents and config description available
        semantic_sim = 0.0
        if evidence_documents and rc_config_required and evidence_id in evidence_documents:
            doc_text = evidence_documents.get(evidence_id, "")
            req_spec = rc_config_required.get(evidence_id, {})
            desc_text = req_spec.get("description", "") if isinstance(req_spec, dict) else str(evidence_id)
            if doc_text and desc_text:
                semantic_sim = compute_semantic_relevance(doc_text, desc_text)

        elements[evidence_id] = {
            "status": status,
            "weight": weight,
            "score": raw_score,
            "weighted_contribution": round(weighted, 4),
            "semantic_relevance": semantic_sim,
        }

    return {
        "completeness_score": round(min(1.0, total_score), 4),
        "elements": elements,
    }


def compute_confidence(
    completeness_score: float,
    evidence_statuses: dict[str, str],
) -> float:
    """Calculate confidence score penalized by proportion of weak evidence items."""
    total = len(evidence_statuses)
    if total == 0:
        return 0.0

    weak_count = sum(1 for s in evidence_statuses.values() if s == "weak")
    weak_ratio = weak_count / total

    # Penalize confidence by weak ratio * 0.10
    confidence = completeness_score - (weak_ratio * 0.10)
    return round(max(0.0, min(1.0, confidence)), 4)


def compute_routing_and_risk(
    completeness_score: float,
    confidence: float,
    thresholds: dict[str, Any] | None = None,
) -> tuple[str, str]:
    """Determine routing action and risk level from readiness and confidence."""
    t_auto_score = DEFAULT_AUTO_DRAFT_MIN_SCORE
    t_auto_conf = DEFAULT_AUTO_DRAFT_MIN_CONFIDENCE
    t_req_score = DEFAULT_REQUEST_MORE_MIN_SCORE

    if thresholds:
        t_auto_score = thresholds.get("auto_draft_response", {}).get("min_completeness_score", t_auto_score)
        t_auto_conf = thresholds.get("auto_draft_response", {}).get("min_confidence", t_auto_conf)
        t_req_score = thresholds.get("request_more_evidence", {}).get("min_completeness_score", t_req_score)

    if completeness_score >= t_auto_score and confidence >= t_auto_conf:
        return "auto_draft_response", "LOW"
    elif completeness_score >= t_req_score:
        return "request_more_evidence", "MEDIUM"
    else:
        return "human_review", "HIGH"


def build_counterfactual_improvements(
    elements: dict[str, Any],
    required_evidence: dict[str, float],
    current_score: float,
    thresholds: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Simulate projected score gain and risk tier upgrade if missing or weak evidence is supplied."""
    gaps = []

    for evidence_id, detail in elements.items():
        status = detail["status"]
        weight = detail["weight"]

        if status == "missing":
            gain = weight * 1.0  # From 0.0 to 1.0
            new_score = min(1.0, current_score + gain)
            new_route, new_risk = compute_routing_and_risk(new_score, new_score, thresholds)
            gaps.append({
                "evidence_id": evidence_id,
                "current_status": "missing",
                "potential_score_gain": round(gain, 4),
                "score_if_added": round(new_score, 4),
                "projected_risk": new_risk,
                "projected_route": new_route,
            })
        elif status == "weak":
            gain = weight * 0.5  # From 0.5 to 1.0
            new_score = min(1.0, current_score + gain)
            new_route, new_risk = compute_routing_and_risk(new_score, new_score, thresholds)
            gaps.append({
                "evidence_id": evidence_id,
                "current_status": "weak",
                "potential_score_gain": round(gain, 4),
                "score_if_added": round(new_score, 4),
                "projected_risk": new_risk,
                "projected_route": new_route,
            })

    # Sort by highest potential score gain first
    return sorted(gaps, key=lambda x: x["potential_score_gain"], reverse=True)


def build_decision_trace(
    completeness_score: float,
    confidence: float,
    risk_level: str,
    routing_decision: str,
    thresholds: dict[str, Any] | None = None,
    missing_count: int = 0,
    weak_count: int = 0,
) -> list[dict[str, Any]]:
    """Build step-by-step gate execution trace explaining the routing decision."""
    t_auto_score = DEFAULT_AUTO_DRAFT_MIN_SCORE
    t_auto_conf = DEFAULT_AUTO_DRAFT_MIN_CONFIDENCE
    t_req_score = DEFAULT_REQUEST_MORE_MIN_SCORE

    if thresholds:
        t_auto_score = thresholds.get("auto_draft_response", {}).get("min_completeness_score", t_auto_score)
        t_auto_conf = thresholds.get("auto_draft_response", {}).get("min_confidence", t_auto_conf)
        t_req_score = thresholds.get("request_more_evidence", {}).get("min_completeness_score", t_req_score)

    score_pass = completeness_score >= t_auto_score
    conf_pass = confidence >= t_auto_conf
    req_pass = completeness_score >= t_req_score

    steps = [
        {
            "step": 1,
            "gate": "Readiness Threshold Check",
            "threshold": f"Score >= {t_auto_score:.0%}",
            "value": f"{completeness_score:.0%}",
            "status": "pass" if score_pass else "fail",
            "message": (
                f"Evidence completeness score is {completeness_score:.0%}, meeting the {t_auto_score:.0%} auto-draft threshold."
                if score_pass
                else f"Evidence completeness score is {completeness_score:.0%}, below the {t_auto_score:.0%} auto-draft threshold."
            ),
        },
        {
            "step": 2,
            "gate": "Confidence Level Check",
            "threshold": f"Confidence >= {t_auto_conf:.0%}",
            "value": f"{confidence:.0%}",
            "status": "pass" if conf_pass else "fail",
            "message": (
                f"Confidence score is {confidence:.0%}, meeting the {t_auto_conf:.0%} confidence gate."
                if conf_pass
                else f"Confidence score is {confidence:.0%}, below the {t_auto_conf:.0%} gate (weak evidence penalty applied: {weak_count} weak item(s))."
            ),
        },
        {
            "step": 3,
            "gate": "Secondary Evidence Request Gate",
            "threshold": f"Score >= {t_req_score:.0%}",
            "value": f"{completeness_score:.0%}",
            "status": "pass" if req_pass else "fail",
            "message": (
                f"Dispute satisfies the {t_req_score:.0%} floor for requesting additional merchant evidence."
                if req_pass
                else f"Dispute score {completeness_score:.0%} is below {t_req_score:.0%}; manual risk team review required."
            ),
        },
        {
            "step": 4,
            "gate": "Final Routing Gate Resolution",
            "threshold": "Deterministic routing rule",
            "value": routing_decision,
            "status": "pass" if routing_decision == "auto_draft_response" else "info",
            "message": (
                "All primary gates passed. Routed to Automated Dispute Response Drafting."
                if routing_decision == "auto_draft_response"
                else (
                    "Primary gates blocked. Routed to Request Additional Merchant Evidence."
                    if routing_decision == "request_more_evidence"
                    else f"Critical evidence missing ({missing_count} item(s)). Routed to Human Review / Economic Loss Prevention."
                )
            ),
        },
    ]
    return steps


def score_dispute(
    dispute: dict[str, Any],
    reason_codes: dict[str, Any],
    use_ground_truth: bool = False,
    api_key: str | None = None,
    log_audit: bool = True,
) -> dict[str, Any]:
    """
    Score a single dispute case against reason code configurations.
    Returns complete readiness evaluation, ML prediction, SHAP explanation,
    and decision audit trace.
    """
    category = dispute.get("reason_category")
    if not category or category not in reason_codes:
        raise ValueError(f"Unknown or missing reason_category: {category}")

    rc_config = reason_codes[category]
    raw_required = rc_config.get("required_evidence", {})

    required_evidence: dict[str, float] = {}
    for k, v in raw_required.items():
        if isinstance(v, dict):
            required_evidence[k] = float(v.get("weight", 0.2))
        else:
            required_evidence[k] = float(v)

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
    gaps = build_counterfactual_improvements(elements, required_evidence, completeness_score, thresholds)

    missing_evidence = [e["evidence_id"] for e in gaps if e["current_status"] == "missing"]
    weak_evidence = [e["evidence_id"] for e in gaps if e["current_status"] == "weak"]

    # Integrate Machine Learning Win Predictor & Financial EV Calculator
    win_predictor = get_win_predictor()
    partial_result = {
        "completeness_score": completeness_score,
        "confidence": confidence,
        "missing_evidence": missing_evidence,
        "weak_evidence": weak_evidence,
        "evidence_elements": elements,
    }
    win_probability = win_predictor.predict_win_probability(dispute, partial_result)
    disputed_amount = float(dispute.get("transaction", {}).get("amount", 1000))
    expected_value = win_predictor.calculate_expected_financial_value(disputed_amount, win_probability)
    economic_rec = win_predictor.recommend_action(disputed_amount, win_probability)

    # Real-time inference logging to ModelDriftMonitor
    try:
        drift_mon = get_drift_monitor()
        if win_predictor.X_train_arr is not None and not drift_mon.baseline_features:
            drift_mon.set_baseline(win_predictor.X_train_arr, win_predictor.feature_names)
        feat_vector = win_predictor._extract_features(dispute, partial_result)
        drift_mon.log_inference(feat_vector, win_probability)
    except Exception as exc:
        logger.debug(f"Drift monitor logging skipped: {exc}")

    # Phase 3 XAI — Decision Pathway Trace
    decision_trace = build_decision_trace(
        completeness_score=completeness_score,
        confidence=confidence,
        risk_level=risk_level,
        routing_decision=routing_decision,
        thresholds=thresholds,
        missing_count=len(missing_evidence),
        weak_count=len(weak_evidence),
    )

    # Phase 1 XAI — Local SHAP explanation for current dispute
    local_shap = win_predictor.get_local_shap_explanation(dispute, partial_result)

    result = {
        "dispute_id": dispute.get("dispute_id", "unknown"),
        "merchant_name": dispute.get("merchant_name", "Merchant"),
        "network": dispute.get("network", rc_config.get("network", "CARD")),
        "reason_code": dispute.get("reason_code", rc_config.get("reason_code", "")),
        "reason_category": category,
        "reason_title": rc_config.get("title", ""),
        "evidence_source": evidence_source,
        "completeness_score": completeness_score,
        "completeness_pct": f"{completeness_score:.0%}",
        "confidence": confidence,
        "confidence_pct": f"{confidence:.0%}",
        "risk_level": risk_level,
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
        "decision_trace": decision_trace,
        "local_shap_explanation": local_shap,
    }

    if log_audit:
        try:
            log_decision(dispute.get("dispute_id", "unknown"), result)
        except Exception as exc:
            logger.warning(f"Failed to log decision audit trace: {exc}")

    return result


def score_dataset(
    dataset_path: str | Path,
    config_path: str | Path,
    split: str | None = None,
    use_ground_truth: bool = False,
    api_key: str | None = None,
) -> list[dict[str, Any]]:
    """Score a collection of disputes from a dataset JSON file."""
    d_path = Path(dataset_path)
    data = json.loads(d_path.read_text(encoding="utf-8"))
    reason_codes = load_reason_code_config(config_path)

    cases = data.get("cases", [])
    if split:
        cases = [c for c in cases if c.get("split") == split]

    results = []
    for case in cases:
        try:
            scored = score_dispute(
                case,
                reason_codes,
                use_ground_truth=use_ground_truth,
                api_key=api_key,
                log_audit=False,
            )
            results.append(scored)
        except Exception as exc:
            results.append({"dispute_id": case.get("dispute_id"), "error": str(exc)})

    return results
